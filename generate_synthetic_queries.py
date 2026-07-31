"""
Generate synthetic multilingual query examples from the ground-truth knowledge base.

- Downloads ground_truth_kb.csv from Hugging Face (no local copy needed in repo)
- For each (disease, language) pair, generates QUERIES_PER_LANGUAGE phrasings
- Tries a chain of free-tier models in order; falls back to the next one on
  quota/rate errors instead of crashing
- SAVES PROGRESS INCREMENTALLY after every (disease, language) pair, and
  SKIPS pairs already completed on a prior run -- safe to re-run daily until
  the full dataset is generated across multiple quota windows
- Logs which model produced each row (generated_by column) so you can spot
  check whether a fallback model produced weaker phrasing

This script does NOT invent facts. It only generates realistic PHRASINGS of how a
farmer/patient might describe symptoms already present in the ground-truth row.
The disease/cause/treatment facts are never touched by the model here.

Usage:
    export GEMINI_API_KEY="your-key-here"
    python generate_synthetic_queries.py

Requires:
    pip install google-genai requests
"""

import csv
import io
import json
import os
import time
import sys

import requests

try:
    from google import genai
    from google.genai import errors as genai_errors
except ImportError:
    print("Missing dependency. Run: pip install google-genai requests")
    sys.exit(1)

# ---- Config ----
HF_CSV_URL = "https://huggingface.co/datasets/MR-CODESPIKE/ground-truth-ob/resolve/main/ground_truth_kb.csv"
OUTPUT_CSV = "synthetic_queries.csv"
PROGRESS_FILE = "synthetic_queries_progress.json"  # tracks which (disease, language) pairs are done

LANGUAGES = ["Yoruba", "Hausa", "Igbo", "Nigerian Pidgin", "English"]

# Fallback chain: tried in order per call. Free-tier models most likely to be
# available/generous first, weaker/last-resort fallback at the end.
# Swap or reorder freely -- e.g. add "gemma-3-27b-it" earlier if you want to
# lean harder into the "Build with Gemma" story once you've checked its quota.
MODEL_CHAIN = [
    "gemini-2.5-flash-lite",  # most generous free tier: 15 RPM / 1,000 RPD
    "gemini-2.5-flash",       # 10 RPM / 250 RPD
    "gemini-2.0-flash-lite",  # separate quota bucket from 2.5 models
    "gemma-4-12b-it",         # Gemma 4 (open model, free tier) -- separate quota bucket
    "gemma-4-31b-it",
    "gemma-4-26b-a4b-it",
]

QUERIES_PER_LANGUAGE = 8  # full run, no small-batch test -- review manually after
SLEEP_BETWEEN_CALLS = 2.0  # stay under ~15-30 RPM depending on model
MAX_RETRIES_PER_MODEL = 1  # how many times to retry the SAME model before falling back

PROMPT_TEMPLATE = """You are helping build a dataset of realistic farmer/patient descriptions of symptoms, for a diagnostic assistant app used in Nigeria.

Given this VERIFIED disease record:
- Domain: {domain}
- Name: {name}
- Category: {category}
- Symptoms (ground truth, do not alter the facts): {symptoms}

Generate {n} different realistic ways a Nigerian farmer/animal owner/patient (depending on domain)
might describe THESE SAME symptoms out loud, in {language}, as if speaking to a diagnostic app.

Rules:
- Only describe symptoms already listed above. Do NOT mention causes, treatments, or the disease name.
- Use natural, informal, spoken-language phrasing a real person would use (not textbook language).
- Vary sentence structure and vocabulary between the {n} examples.
- For non-English languages, use the language as commonly spoken in Nigeria (colloquial, not formal/literary).

Return ONLY valid JSON, no markdown fences, no preamble, in this exact format:
[
  {{"query_text": "...", "translated_text": "..."}},
  ...
]
"translated_text" should be the natural English translation of query_text.
If language is English, translated_text should equal query_text.
"""


def download_kb(url):
    print(f"Downloading ground-truth KB from {url} ...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))
    rows = list(reader)
    print(f"Loaded {len(rows)} ground-truth rows.")
    return rows


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return set(tuple(pair) for pair in json.load(f))
    return set()


def save_progress(done_pairs):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump([list(pair) for pair in done_pairs], f)


def load_existing_output():
    """Load any rows already written from a previous run so we can append, not overwrite."""
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    return []


def save_output(rows):
    fieldnames = ["id", "domain", "language", "query_text", "translated_text",
                  "expected_match_id", "generated_by"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def call_model(client, model, row, language, n):
    prompt = PROMPT_TEMPLATE.format(
        domain=row["domain"],
        name=row["name"],
        category=row["category"],
        symptoms=row["symptoms"],
        language=language,
        n=n,
    )
    response = client.models.generate_content(model=model, contents=prompt)
    text = response.text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)  # let JSONDecodeError bubble up to caller


def generate_with_fallback(client, row, language, n):
    """Try each model in MODEL_CHAIN in order. Returns (examples, model_used) or (None, None)
    if every model in the chain failed (e.g. all quotas exhausted for this run)."""
    for model in MODEL_CHAIN:
        for attempt in range(MAX_RETRIES_PER_MODEL):
            try:
                examples = call_model(client, model, row, language, n)
                return examples, model
            except genai_errors.ClientError as e:
                if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                    print(f"    [{model}] quota/rate exhausted, trying next model...")
                    break  # move to next model, don't retry same one
                else:
                    print(f"    [{model}] client error: {e}")
                    break
            except json.JSONDecodeError:
                print(f"    [{model}] bad JSON output, retrying..." if attempt == 0 else f"    [{model}] bad JSON again, trying next model...")
                continue
            except Exception as e:
                print(f"    [{model}] unexpected error: {e}")
                break
    return None, None


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY environment variable first.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    kb_rows = download_kb(HF_CSV_URL)

    done_pairs = load_progress()
    out_rows = load_existing_output()
    query_counter = len(out_rows) + 1

    print(f"Resuming: {len(done_pairs)} (disease, language) pairs already completed, "
          f"{len(out_rows)} rows already saved.")

    all_exhausted_streak = 0  # if every model fails for several pairs in a row, stop the run

    for row in kb_rows:
        for language in LANGUAGES:
            pair_key = (row["id"], language)
            if pair_key in done_pairs:
                continue  # already done in a previous run

            print(f"Generating {QUERIES_PER_LANGUAGE}x {language} queries for {row['id']} ({row['name']})...")
            examples, model_used = generate_with_fallback(client, row, language, QUERIES_PER_LANGUAGE)

            if examples is None:
                all_exhausted_streak += 1
                print(f"  All models failed for this pair. Skipping for now (streak: {all_exhausted_streak}).")
                if all_exhausted_streak >= 5:
                    print("\nAll models appear exhausted for multiple pairs in a row. "
                          "Stopping cleanly -- progress is saved. Re-run this script "
                          "later (e.g. tomorrow, once quotas reset) to continue.")
                    save_output(out_rows)
                    save_progress(done_pairs)
                    sys.exit(0)
                time.sleep(SLEEP_BETWEEN_CALLS)
                continue

            all_exhausted_streak = 0
            for ex in examples:
                out_rows.append({
                    "id": f"query_{query_counter:04d}",
                    "domain": row["domain"],
                    "language": language,
                    "query_text": ex.get("query_text", ""),
                    "translated_text": ex.get("translated_text", ""),
                    "expected_match_id": row["id"],
                    "generated_by": model_used,
                })
                query_counter += 1

            done_pairs.add(pair_key)

            # Save progress after EVERY pair, not just at the end -- a crash or
            # a manual stop never loses more than one pair's worth of work.
            save_output(out_rows)
            save_progress(done_pairs)

            time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"\nDone. Wrote {len(out_rows)} synthetic queries to {OUTPUT_CSV}")
    print(f"Completed {len(done_pairs)} / {len(kb_rows) * len(LANGUAGES)} (disease, language) pairs.")
    if len(done_pairs) < len(kb_rows) * len(LANGUAGES):
        print("Some pairs are still missing -- re-run this script again "
              "(e.g. after quotas reset) to fill in the rest.")


if __name__ == "__main__":
    main()