#!/usr/bin/env python3
"""
Runs after all 8 download jobs complete. Downloads each per-subfolder
labels.csv already pushed to the HF dataset repo, concatenates them into
one master labels.csv at the repo root, and pushes that back.

This runs as its own GitHub Actions job with `needs:` on all 8 download
jobs, so it only executes if they all succeeded.
"""
import csv
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, upload_file

SUBFOLDERS = [
    "crop/plantvillage",
    "crop/cassava",
    "crop/maize",
    "crop/tom2024",
    "animal/cattle_lsd",
    "animal/poultry",
    "human/dermnet",
    "human/skincap",
]

UNIFIED_COLUMNS = ["image_path", "domain", "source_dataset", "disease_name", "species"]


def main():
    hf_repo = os.environ["HF_REPO"]
    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)

    all_rows = []
    for subfolder in SUBFOLDERS:
        remote_path = f"{subfolder}/labels.csv"
        try:
            local_path = hf_hub_download(
                repo_id=hf_repo, repo_type="dataset",
                filename=remote_path, token=token,
            )
        except Exception as e:
            print(f"WARNING: could not fetch {remote_path}: {e}", file=sys.stderr)
            continue
        with open(local_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            n = 0
            for row in reader:
                all_rows.append(row)
                n += 1
        print(f"{remote_path}: {n} rows", flush=True)

    if not all_rows:
        print("ERROR: no per-dataset labels.csv files found -- did all download "
              "jobs actually push successfully?", file=sys.stderr)
        sys.exit(1)

    out_path = Path("/tmp/labels_master.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=UNIFIED_COLUMNS)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in UNIFIED_COLUMNS})

    print(f"=== Master labels.csv: {len(all_rows)} total rows ===", flush=True)

    # per-domain counts for a quick sanity check in the Actions log
    domain_counts: dict[str, int] = {}
    for row in all_rows:
        domain_counts[row["domain"]] = domain_counts.get(row["domain"], 0) + 1
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain}: {count} images", flush=True)

    api.create_repo(repo_id=hf_repo, repo_type="dataset", exist_ok=True)
    upload_file(
        path_or_fileobj=str(out_path),
        path_in_repo="labels.csv",
        repo_id=hf_repo,
        repo_type="dataset",
        token=token,
    )
    print("=== Pushed master labels.csv to repo root ===", flush=True)


if __name__ == "__main__":
    main()