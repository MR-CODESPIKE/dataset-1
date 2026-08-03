"""
Shared cleaning + label-normalization utilities used by every per-dataset
downloader script. Kept dependency-light (Pillow + imagehash only) so it
installs fast in a fresh GitHub Actions runner every job.
"""
import csv
import hashlib
import os
from pathlib import Path

from PIL import Image, UnidentifiedImageError
import imagehash

UNIFIED_COLUMNS = [
    "image_path",       # path relative to repo root inside the HF dataset repo
    "domain",           # crop | animal | human
    "source_dataset",   # short slug identifying which of the 8 sources this came from
    "disease_name",     # normalized disease label (snake_case, English)
    "species",          # e.g. cassava, maize, cattle, poultry, human
]

MIN_IMAGE_DIM = 32  # anything smaller than this is almost certainly junk/corrupt


def is_valid_image(path: Path, min_dim: int = MIN_IMAGE_DIM) -> bool:
    """Basic quality filter: rejects corrupt files, truncated files, and
    tiny/degenerate images. Does NOT do blur detection by default (see
    is_blurry) since that needs numpy/opencv which we only pull in if asked."""
    try:
        with Image.open(path) as img:
            img.verify()  # cheap structural check, doesn't decode full image
        with Image.open(path) as img:
            w, h = img.size
            if w < min_dim or h < min_dim:
                return False
            img.convert("RGB")  # force full decode to catch truncated files
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def is_blurry(path: Path, threshold: float = 60.0) -> bool:
    """Optional stronger quality filter using Laplacian variance.
    Requires opencv-python-headless + numpy. Called only if --strict-quality
    is passed to a downloader script, since it's slower and adds deps."""
    import cv2
    import numpy as np

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return True
    variance = cv2.Laplacian(img, cv2.CV_64F).var()
    return variance < threshold


def phash_of(path: Path):
    """Perceptual hash for near-duplicate detection (robust to resaves,
    minor crops/recompression -- catches near-dupes that md5 would miss)."""
    with Image.open(path) as img:
        return imagehash.phash(img.convert("RGB"))


def dedupe_images(image_paths: list[Path], hamming_threshold: int = 4) -> list[Path]:
    """Returns the subset of image_paths with near-duplicates removed.
    Keeps the first-seen image in each duplicate cluster. O(n^2) hash
    comparisons -- fine up to a few thousand images per dataset per job;
    for larger sets consider bucketing by hash prefix first."""
    kept: list[Path] = []
    kept_hashes: list = []
    for p in image_paths:
        try:
            h = phash_of(p)
        except (UnidentifiedImageError, OSError):
            continue  # already-corrupt, let is_valid_image handle rejection upstream
        is_dupe = False
        for kh in kept_hashes:
            if h - kh <= hamming_threshold:  # Hamming distance
                is_dupe = True
                break
        if not is_dupe:
            kept.append(p)
            kept_hashes.append(h)
    return kept


def exact_dedupe(image_paths: list[Path]) -> list[Path]:
    """Fast first pass: drop byte-identical files via md5 before the more
    expensive perceptual hash pass. Always run this first."""
    seen = set()
    kept = []
    for p in image_paths:
        h = hashlib.md5(p.read_bytes()).hexdigest()
        if h not in seen:
            seen.add(h)
            kept.append(p)
    return kept


def write_labels_csv(rows: list[dict], out_path: Path, append: bool = False):
    """Writes/appends rows to the unified labels.csv. Each row must contain
    exactly UNIFIED_COLUMNS keys. append=True is used when multiple dataset
    jobs write to the same csv artifact (handled via separate per-job CSVs
    merged in a final job, see merge_labels.py) so in practice each job
    calls this with append=False against its own job-local file."""
    mode = "a" if append and out_path.exists() else "w"
    write_header = not (append and out_path.exists())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=UNIFIED_COLUMNS)
        if write_header:
            writer.writeheader()
        for row in rows:
            missing = set(UNIFIED_COLUMNS) - set(row.keys())
            if missing:
                raise ValueError(f"Row missing columns {missing}: {row}")
            writer.writerow(row)


def normalize_label(raw_label: str) -> str:
    """Turns messy folder/class names into a consistent snake_case label.
    e.g. 'Corn___Common_rust' -> 'common_rust', 'Cassava Mosaic Disease' ->
    'cassava_mosaic_disease'. Intentionally simple/deterministic -- disease
    name mapping tables (per-source) do the semantic normalization; this
    just cleans formatting after that mapping is applied."""
    s = raw_label.strip().lower()
    s = s.replace("___", "_").replace("__", "_")
    s = s.replace("-", "_").replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")