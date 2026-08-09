"""
Strategy 4: Text Normalization Pipeline.

Used in two places:
  1. On training transcripts, so the model isn't learning inconsistent
     casing/punctuation conventions across mixed data sources
     (WaxalNLP vs FLEURS vs AfriVoice vs Common Voice).
  2. On model predictions at inference time, before writing the submission
     CSV, to match Zindi's expected format.

CONFIRMED from real Zindi Train.csv (38,176 clean transcripts, inspected
directly): WaxalNLP transcripts use MIXED CASE and FULL PUNCTUATION —
e.g. "Ekyuma ekyakolebwa Bamagulumeeru nga kiri mu makkati g'ennyanja
era nga kizingiddwako akaguwa ak'ekikobe." config.py's defaults
(NORMALIZE_LOWERCASE=False, NORMALIZE_STRIP_PUNCTUATION=False) now match
this. Re-verify against Zindi's actual scoring if you ever get signal on
it (e.g. from a leaderboard score comparison) — this is the best available
evidence, not a guarantee about the hidden ground truth's exact format.
"""

import re
import unicodedata

import config

# Basic punctuation to strip. Extend if the real transcripts show more
# (e.g. curly quotes, em-dashes) once you've inspected them.
_PUNCT_PATTERN = re.compile(r"[.,!?;:\"'\(\)\[\]\-–—…]")
_MULTI_SPACE_PATTERN = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    if text is None:
        return ""

    text = unicodedata.normalize("NFC", text)

    if config.NORMALIZE_LOWERCASE:
        text = text.lower()

    if config.NORMALIZE_STRIP_PUNCTUATION:
        text = _PUNCT_PATTERN.sub("", text)

    text = _MULTI_SPACE_PATTERN.sub(" ", text).strip()

    return text


def normalize_batch(texts: list[str]) -> list[str]:
    return [normalize_text(t) for t in texts]


if __name__ == "__main__":
    # Inspect a handful of real WaxalNLP transcripts to sanity-check the
    # normalization rules above BEFORE trusting them for the full pipeline.
    from datasets import load_dataset

    for lang_cfg in config.LANGUAGES:
        print(f"\n--- {lang_cfg} sample transcripts (raw) ---")
        ds = load_dataset("google/WaxalNLP", lang_cfg, split="train", streaming=True)
        for i, example in enumerate(ds):
            text_col = "text" if "text" in example else "transcription"
            raw = example.get(text_col, example.get("transcription", ""))
            print(f"  raw:  {raw!r}")
            print(f"  norm: {normalize_text(raw)!r}")
            if i >= 4:
                break
