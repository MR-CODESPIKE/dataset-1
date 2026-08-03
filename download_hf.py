#!/usr/bin/env python3
"""
Downloads a Hugging Face `datasets`-hosted image dataset, cleans it, maps
labels to the unified schema, and pushes it to the target HF dataset repo.

Two sources use this script:
  --dataset plantvillage   (mohanty/PlantVillage, "color" config)
  --dataset skincap        (joshuachou/SkinCAP)

SkinCAP has free-text disease captions rather than fixed class folders, so
its label handling is dynamic (normalize_label() applied directly to
whatever disease field is present) rather than table-based.

Requires env var: HF_TOKEN
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from clean_utils import (
    exact_dedupe, dedupe_images, is_valid_image, is_blurry,
    write_labels_csv, normalize_label,
)
import label_maps
from datasets import load_dataset
from huggingface_hub import HfApi, upload_folder

WORK_DIR = Path("/tmp/hf_work")


def process_plantvillage(clean_dir: Path, source_name: str, hf_subfolder: str, strict_quality: bool):
    ds = load_dataset("mohanty/PlantVillage", "color", split="train")
    label_names = ds.features["label"].names
    label_map = label_maps.PLANTVILLAGE_MAP

    unmapped = set(name for name in label_names if name not in label_map)
    if unmapped:
        print(f"WARNING: unmapped PlantVillage classes, skipping: {unmapped}", flush=True)
        print("Extend PLANTVILLAGE_MAP in label_maps.py to include these.", flush=True)

    rows = []
    counters: dict[str, int] = {}
    tmp_dir = WORK_DIR / "pv_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for i, example in enumerate(ds):
        label_idx = example["label"]
        class_name = label_names[label_idx]
        if class_name not in label_map:
            continue
        domain, species, disease_name = label_map[class_name]
        img = example["image"]
        tmp_path = tmp_dir / f"{i:06d}.jpg"
        img.convert("RGB").save(tmp_path, "JPEG", quality=90)
        counters[class_name] = counters.get(class_name, 0) + 1

    # quality filter + dedup pass, grouped by class to keep dedup scoped
    # (cross-class near-dupes are extremely unlikely and not worth the
    # O(n^2) cost across the full 40k+ image set)
    all_tmp = list(tmp_dir.glob("*.jpg"))
    valid = [p for p in all_tmp if is_valid_image(p)]
    if strict_quality:
        valid = [p for p in valid if not is_blurry(p)]
    valid = exact_dedupe(valid)
    valid = dedupe_images(valid)
    print(f"PlantVillage: kept {len(valid)}/{len(all_tmp)} after cleaning", flush=True)

    # re-derive label for each surviving temp file by re-reading dataset index
    kept_indices = {int(p.stem) for p in valid}
    per_class_counter: dict[str, int] = {}
    for i, example in enumerate(ds):
        if i not in kept_indices:
            continue
        label_idx = example["label"]
        class_name = label_names[label_idx]
        if class_name not in label_map:
            continue
        domain, species, disease_name = label_map[class_name]
        norm_disease = normalize_label(disease_name)
        dest_subdir = clean_dir / domain / species / norm_disease
        dest_subdir.mkdir(parents=True, exist_ok=True)
        n = per_class_counter.get(class_name, 0)
        dest = dest_subdir / f"{source_name}_{n:05d}.jpg"
        shutil.copy2(tmp_dir / f"{i:06d}.jpg", dest)
        per_class_counter[class_name] = n + 1
        rows.append({
            "image_path": f"{hf_subfolder}/{dest.relative_to(clean_dir)}",
            "domain": domain,
            "source_dataset": source_name,
            "disease_name": norm_disease,
            "species": species,
        })

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return rows


def process_skincap(clean_dir: Path, source_name: str, hf_subfolder: str, strict_quality: bool):
    ds = load_dataset("joshuachou/SkinCAP", split="train")
    tmp_dir = WORK_DIR / "skincap_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # SkinCAP's exact column names should be confirmed against the dataset
    # card at download time -- falling back gracefully across likely names.
    disease_col = None
    for candidate in ("disease", "diagnosis", "label", "condition"):
        if candidate in ds.column_names:
            disease_col = candidate
            break
    if disease_col is None:
        print(f"ERROR: could not find a disease/label column in SkinCAP. "
              f"Available columns: {ds.column_names}", file=sys.stderr)
        sys.exit(1)

    for i, example in enumerate(ds):
        img = example["image"]
        tmp_path = tmp_dir / f"{i:06d}.jpg"
        img.convert("RGB").save(tmp_path, "JPEG", quality=90)

    all_tmp = list(tmp_dir.glob("*.jpg"))
    valid = [p for p in all_tmp if is_valid_image(p)]
    if strict_quality:
        valid = [p for p in valid if not is_blurry(p)]
    valid = exact_dedupe(valid)
    valid = dedupe_images(valid)
    print(f"SkinCAP: kept {len(valid)}/{len(all_tmp)} after cleaning", flush=True)

    kept_indices = {int(p.stem) for p in valid}
    per_class_counter: dict[str, int] = {}
    rows = []
    for i, example in enumerate(ds):
        if i not in kept_indices:
            continue
        raw_label = str(example[disease_col])
        norm_disease = normalize_label(raw_label)
        dest_subdir = clean_dir / "human" / "skin" / norm_disease
        dest_subdir.mkdir(parents=True, exist_ok=True)
        n = per_class_counter.get(norm_disease, 0)
        dest = dest_subdir / f"{source_name}_{n:05d}.jpg"
        shutil.copy2(tmp_dir / f"{i:06d}.jpg", dest)
        per_class_counter[norm_disease] = n + 1
        rows.append({
            "image_path": f"{hf_subfolder}/{dest.relative_to(clean_dir)}",
            "domain": "human",
            "source_dataset": source_name,
            "disease_name": norm_disease,
            "species": "human",
        })

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["plantvillage", "skincap"])
    ap.add_argument("--source-name", required=True)
    ap.add_argument("--hf-repo", required=True)
    ap.add_argument("--hf-subfolder", required=True)
    ap.add_argument("--strict-quality", default="false")
    args = ap.parse_args()
    strict_quality = args.strict_quality.lower() == "true"

    clean_dir = WORK_DIR / "clean" / args.source_name
    if clean_dir.exists():
        shutil.rmtree(clean_dir)
    clean_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "plantvillage":
        rows = process_plantvillage(clean_dir, args.source_name, args.hf_subfolder, strict_quality)
    else:
        rows = process_skincap(clean_dir, args.source_name, args.hf_subfolder, strict_quality)

    if not rows:
        print("ERROR: no rows produced -- check label mapping / column names above.", file=sys.stderr)
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


if __name__ == "__main__":
    main()