#!/usr/bin/env python3
"""
Generic Kaggle dataset downloader/cleaner/pusher.

Handles: download via Kaggle API -> unzip -> quality filter -> dedup ->
label normalization via a mapping table -> batched push to a HF dataset
repo under a per-source subfolder.

Usage (see workflow YAML for the real per-dataset invocations):
    python download_kaggle.py \
        --slug shivamagarwal29/cow-lumpy-disease-dataset \
        --source-name cattle_lsd \
        --label-map CATTLE_LSD_MAP \
        --hf-repo yourname/nigeria-disease-datasets \
        --hf-subfolder animal/cattle_lsd \
        --batch-size-mb 2000 \
        --is-competition false

Requires env vars: KAGGLE_USERNAME, KAGGLE_KEY, HF_TOKEN
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from clean_utils import (
    exact_dedupe, dedupe_images, is_valid_image, is_blurry,
    write_labels_csv, normalize_label, UNIFIED_COLUMNS,
)
import label_maps
from huggingface_hub import HfApi, upload_folder

WORK_DIR = Path("/tmp/kaggle_work")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def run(cmd: list[str]):
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def download_and_unzip(slug: str, is_competition: bool, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    if is_competition:
        run(["kaggle", "competitions", "download", "-c", slug, "-p", str(out_dir)])
    else:
        run(["kaggle", "datasets", "download", "-d", slug, "-p", str(out_dir), "--force"])
    for zip_path in out_dir.glob("*.zip"):
        print(f"Unzipping {zip_path}", flush=True)
        run(["unzip", "-q", "-o", str(zip_path), "-d", str(out_dir)])
        zip_path.unlink()  # free disk immediately -- 14GB runner budget is tight


GENERIC_WRAPPER_NAMES = {"train", "test", "val", "validation", "data", "dataset", "images"}


def find_class_folders(root: Path) -> dict[str, list[Path]]:
    """Walks root and groups image files by their immediate parent folder
    name, which is the standard Kaggle image-classification convention
    (root/split?/ClassName/img.jpg). Returns {folder_name: [image_paths]}.

    Some datasets wrap class folders in a generic split folder (e.g.
    Train/ClassA/img.jpg with no separate Test/ at all, or images sitting
    directly inside Train/ with no class subfolder). To handle this
    without needing per-dataset special-casing:
    - if an image's immediate parent is a generic wrapper name, use the
      grandparent instead IF the grandparent isn't just `root` itself.
    - if that still resolves to nothing usable, fall back to the
      immediate parent so the original unmapped-folder warning still
      surfaces (rather than silently dropping images).
    """
    groups: dict[str, list[Path]] = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            parent = p.parent
            class_name = parent.name
            if class_name.lower() in GENERIC_WRAPPER_NAMES and parent != root:
                # try the grandparent as the real class name instead
                grandparent_name = parent.parent.name
                if grandparent_name and parent.parent != root:
                    class_name = grandparent_name
            groups.setdefault(class_name, []).append(p)
    return groups


def clean_group(paths: list[Path], strict_quality: bool) -> list[Path]:
    valid = [p for p in paths if is_valid_image(p)]
    if strict_quality:
        valid = [p for p in valid if not is_blurry(p)]
    valid = exact_dedupe(valid)
    valid = dedupe_images(valid)
    return valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True, help="Kaggle dataset or competition slug")
    ap.add_argument("--source-name", required=True, help="short id, e.g. cattle_lsd")
    ap.add_argument("--label-map", required=True, help="attribute name in label_maps.py")
    ap.add_argument("--hf-repo", required=True)
    ap.add_argument("--hf-subfolder", required=True)
    ap.add_argument("--is-competition", default="false")
    ap.add_argument("--strict-quality", default="false")
    ap.add_argument("--batch-size-mb", type=int, default=2000,
                     help="push to HF in batches under this size to avoid runner disk pressure")
    args = ap.parse_args()

    is_competition = args.is_competition.lower() == "true"
    strict_quality = args.strict_quality.lower() == "true"
    label_map: dict = getattr(label_maps, args.label_map)

    raw_dir = WORK_DIR / "raw" / args.source_name
    clean_dir = WORK_DIR / "clean" / args.source_name
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    if clean_dir.exists():
        shutil.rmtree(clean_dir)

    print(f"=== Downloading {args.slug} ===", flush=True)
    download_and_unzip(args.slug, is_competition, raw_dir)

    print("=== Grouping by class folder ===", flush=True)
    groups = find_class_folders(raw_dir)
    unmapped = [name for name in groups if name not in label_map]
    if unmapped:
        print(f"WARNING: unmapped class folders found, skipping them: {unmapped}", flush=True)
        for name in unmapped:
            sample = groups[name][:3]
            print(f"  Sample paths under '{name}':", flush=True)
            for s in sample:
                print(f"    {s.relative_to(raw_dir)}", flush=True)
        print("Add these to the relevant *_MAP in label_maps.py and re-run if needed.", flush=True)

    rows = []
    clean_dir.mkdir(parents=True, exist_ok=True)
    total_kept = 0
    for class_name, paths in groups.items():
        if class_name not in label_map:
            continue
        domain, species, disease_name = label_map[class_name]
        print(f"--- {class_name}: {len(paths)} raw images ---", flush=True)
        kept = clean_group(paths, strict_quality)
        print(f"    kept {len(kept)}/{len(paths)} after quality filter + dedup", flush=True)
        dest_subdir = clean_dir / domain / species / normalize_label(disease_name)
        dest_subdir.mkdir(parents=True, exist_ok=True)
        for i, src in enumerate(kept):
            dest = dest_subdir / f"{args.source_name}_{i:05d}{src.suffix.lower()}"
            shutil.copy2(src, dest)
            rel_path = f"{args.hf_subfolder}/{dest.relative_to(clean_dir)}"
            rows.append({
                "image_path": rel_path,
                "domain": domain,
                "source_dataset": args.source_name,
                "disease_name": normalize_label(disease_name),
                "species": species,
            })
        total_kept += len(kept)

    if not rows:
        print("ERROR: no images survived mapping + cleaning -- check label_maps.py against "
              "the actual folder names printed above.", file=sys.stderr)
        sys.exit(1)

    labels_csv = clean_dir / "labels.csv"
    write_labels_csv(rows, labels_csv, append=False)
    print(f"=== Wrote {len(rows)} rows to {labels_csv} ===", flush=True)

    print(f"=== Pushing {args.source_name} to {args.hf_repo}/{args.hf_subfolder} ===", flush=True)
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=args.hf_repo, repo_type="dataset", exist_ok=True)
    # upload_folder batches internally over HTTP; batch-size-mb is mainly a
    # documented intent here since huggingface_hub handles chunking, but we
    # still free local disk immediately after by not keeping raw_dir around.
    upload_folder(
        repo_id=args.hf_repo,
        repo_type="dataset",
        folder_path=str(clean_dir),
        path_in_repo=args.hf_subfolder,
        token=os.environ["HF_TOKEN"],
    )
    print(f"=== Done: {args.source_name} ({total_kept} images) ===", flush=True)

    # free disk for any subsequent local steps in the same job
    shutil.rmtree(raw_dir, ignore_errors=True)


if __name__ == "__main__":
    main()