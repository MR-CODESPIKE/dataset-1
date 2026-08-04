#!/usr/bin/env python3
"""
Downloads a Mendeley Data dataset via its public content API (keyed by DOI),
since Mendeley pages are JS-rendered and have no stable static zip URL.

Mendeley's API exposes file listings + direct download URLs at:
    https://api.mendeley.com/datasets/public/{doi_suffix}

This endpoint is public / unauthenticated for published datasets, so no
Mendeley token is required -- only HF_TOKEN for the push step.

Usage:
    python download_mendeley.py \
        --doi 10.17632/3d4yg89rtr.1 \
        --source-name tom2024 \
        --label-map TOM2024_MAP \
        --hf-repo yourname/nigeria-disease-datasets \
        --hf-subfolder crop/tom2024
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from clean_utils import (
    exact_dedupe, dedupe_images, is_valid_image, is_blurry,
    write_labels_csv, normalize_label,
)
import label_maps
from huggingface_hub import HfApi, upload_folder

WORK_DIR = Path("/tmp/mendeley_work")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MENDELEY_API = "https://api.mendeley.com/datasets/public"


def run(cmd: list[str]):
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def fetch_file_list(doi: str) -> list[dict]:
    doi_suffix = doi.split("/")[-1]  # e.g. "3d4yg89rtr.1"
    url = f"{MENDELEY_API}/{doi_suffix}"
    print(f"Fetching Mendeley file list from {url}", flush=True)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # Mendeley's public API returns dataset metadata including a "files"
    # array with content_details.download_url per file.
    files = data.get("files", [])
    if not files:
        print(f"ERROR: no files returned by Mendeley API for {doi}. "
              f"Response keys: {list(data.keys())}", file=sys.stderr)
        sys.exit(1)
    return files


def download_files(files: list[dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        filename = f.get("filename", "unknown_file")
        download_url = f.get("content_details", {}).get("download_url")
        if not download_url:
            print(f"WARNING: no download_url for {filename}, skipping", flush=True)
            continue
        dest = out_dir / filename
        print(f"Downloading {filename}...", flush=True)
        with requests.get(download_url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(chunk_size=8192):
                    fh.write(chunk)
        if dest.suffix.lower() == ".zip":
            print(f"Unzipping {dest}", flush=True)
            run(["unzip", "-q", "-o", str(dest), "-d", str(out_dir)])
            dest.unlink()  # free disk immediately


def find_class_folders(root: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            class_name = p.parent.name
            groups.setdefault(class_name, []).append(p)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doi", required=True)
    ap.add_argument("--source-name", required=True)
    ap.add_argument("--label-map", required=True)
    ap.add_argument("--hf-repo", required=True)
    ap.add_argument("--hf-subfolder", required=True)
    ap.add_argument("--strict-quality", default="false")
    args = ap.parse_args()
    strict_quality = args.strict_quality.lower() == "true"
    label_map: dict = getattr(label_maps, args.label_map)

    raw_dir = WORK_DIR / "raw" / args.source_name
    clean_dir = WORK_DIR / "clean" / args.source_name
    for d in (raw_dir, clean_dir):
        if d.exists():
            shutil.rmtree(d)

    files = fetch_file_list(args.doi)
    download_files(files, raw_dir)

    groups = find_class_folders(raw_dir)
    unmapped = [name for name in groups if name not in label_map]
    if unmapped:
        print(f"WARNING: unmapped TOM2024 class folders, skipping: {unmapped}", flush=True)
        print("Inspect these names and extend TOM2024_MAP in label_maps.py.", flush=True)

    rows = []
    clean_dir.mkdir(parents=True, exist_ok=True)
    for class_name, paths in groups.items():
        if class_name not in label_map:
            continue
        domain, species, disease_name = label_map[class_name]
        print(f"--- {class_name}: {len(paths)} raw images ---", flush=True)
        valid = [p for p in paths if is_valid_image(p)]
        if strict_quality:
            valid = [p for p in valid if not is_blurry(p)]
        valid = exact_dedupe(valid)
        valid = dedupe_images(valid)
        print(f"    kept {len(valid)}/{len(paths)} after cleaning", flush=True)

        norm_disease = normalize_label(disease_name)
        dest_subdir = clean_dir / species / norm_disease
        dest_subdir.mkdir(parents=True, exist_ok=True)
        for i, src in enumerate(valid):
            dest = dest_subdir / f"{args.source_name}_{i:05d}{src.suffix.lower()}"
            shutil.copy2(src, dest)
            rows.append({
                "image_path": f"{args.hf_subfolder}/{dest.relative_to(clean_dir)}",
                "domain": domain,
                "source_dataset": args.source_name,
                "disease_name": norm_disease,
                "species": species,
            })

    if not rows:
        print("ERROR: no images survived mapping + cleaning -- check TOM2024_MAP "
              "against the folder names printed above.", file=sys.stderr)
        sys.exit(1)

    labels_csv = clean_dir / "labels.csv"
    write_labels_csv(rows, labels_csv, append=False)
    print(f"=== Wrote {len(rows)} rows to {labels_csv} ===", flush=True)

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=args.hf_repo, repo_type="dataset", exist_ok=True)
    upload_folder(
        repo_id=args.hf_repo,
        repo_type="dataset",
        folder_path=str(clean_dir),
        path_in_repo=args.hf_subfolder,
        token=os.environ["HF_TOKEN"],
    )
    print(f"=== Done: {args.source_name} ({len(rows)} images) ===", flush=True)
    shutil.rmtree(raw_dir, ignore_errors=True)


if __name__ == "__main__":
    main()