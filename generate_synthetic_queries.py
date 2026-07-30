"""
Generate synthetic multilingual query examples from the ground-truth knowledge base.

Downloads ground_truth_kb.csv directly from Hugging Face (no need to keep a local
copy in the repo), generates realistic multilingual symptom phrasings via Gemini,
and writes synthetic_queries.csv.

This script does NOT invent facts. It only generates realistic PHRASINGS of how a
farmer/patient might describe symptoms already present in the ground-truth row.
The disease/cause/treatment facts are never touched by Gemini here.

Usage:
    export GEMINI_API_KEY="your-key-here"
    python generate_synthetic_queries.py

Requires:
    pip install google-genai requests --break-system-packages
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
except ImportError:
    print("Missing dependency. Run: pip install google-genai requests --break-system-packages")
    sys.exit(1)

# ---- Config ----
HF_CSV_URL = "https://huggingface.co/datasets/MR-CODESPIKE/ground-truth-ob/resolve/main/ground_truth_kb.csv"
OUTPUT_CSV = "synthetic_queries.csv"
LANGUAGES = ["Yoruba", "Hausa", "Igbo", "Nigerian Pidgin", "English"]
MODEL = "gemini-2.5-flash"  # fast + cheap; swap to a pro model if quality needs it
QUERIES_PER_LANGUAGE = 2  # start small; raise once pipeline is verified end-to-end
SLEEP_BETWEEN_CALLS = 1.0  # be nice to rate limits

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


def call_gemini(client, row, language, n):
    prompt = PROMPT_TEMPLATE.format(
        domain=row["domain"],
        name=row["name"],
        category=row["category"],
        symptoms=row["symptoms"],
        language=language,
        n=n,
    )
    response = client.models.generate_content(model=MODEL, contents=prompt)
    text = response.text.strip()
    # defensive cleanup in case the model wraps in fences despite instructions
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print(f"  [WARN] Could not parse response for {row['id']} / {language}: {text[:200]}")
        return []


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY environment variable first.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    kb_rows = download_kb(HF_CSV_URL)

    out_rows = []
    query_counter = 1

    for row in kb_rows:
        for language in LANGUAGES:
            print(f"Generating {QUERIES_PER_LANGUAGE}x {language} queries for {row['id']} ({row['name']})...")
            examples = call_gemini(client, row, language, QUERIES_PER_LANGUAGE)
            for ex in examples:
                out_rows.append({
                    "id": f"query_{query_counter:04d}",
                    "domain": row["domain"],
                    "language": language,
                    "query_text": ex.get("query_text", ""),
                    "translated_text": ex.get("translated_text", ""),
                    "expected_match_id": row["id"],
                })
                query_counter += 1
            time.sleep(SLEEP_BETWEEN_CALLS)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "domain", "language", "query_text", "translated_text", "expected_match_id"
        ])
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nDone. Wrote {len(out_rows)} synthetic queries to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()