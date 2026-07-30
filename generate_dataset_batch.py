import os
import json
import time
import shutil
import glob
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from google import genai
from google.genai.errors import APIError
from datasets import Dataset, Features, Image, Value, load_dataset
from huggingface_hub import HfApi, hf_hub_download, create_repo

# ==================== CONFIGURATION ====================
HF_REPO_ID = os.environ.get("HF_REPO_ID")
HF_TOKEN = os.environ.get("HF_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

CHECKPOINT_FILE = "checkpoint.json"
TEMP_DATA_DIR = "./temp_kaggle_batch"

# Fallback chain for Gemini models
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash"]

# List of Kaggle Image Datasets to process in isolated storage batches
KAGGLE_BATCHES = [
    {
        "batch_id": "batch_01",
        "kaggle_slug": "smaranjitghose/corn-or-maize-leaf-disease-dataset",
        "crop_animal": "Maize",
        "condition": "Common Rust",
        "language": "Hausa"
    },
    {
        "batch_id": "batch_02",
        "kaggle_slug": "abdallahwagih/cassava-leaf-disease-classification",
        "crop_animal": "Cassava",
        "condition": "Mosaic Disease",
        "language": "Swahili"
    },
    {
        "batch_id": "batch_03",
        "kaggle_slug": "vbookshelf/rice-leaf-diseases",
        "crop_animal": "Rice",
        "condition": "Bacterial Blight",
        "language": "Yoruba"
    },
    {
        "batch_id": "batch_04",
        "kaggle_slug": "kaustubhb999/tomatoleaf",
        "crop_animal": "Tomato",
        "condition": "Late Blight",
        "language": "Igbo"
    }
]

PROMPT_TEMPLATE = """
You are an expert Agricultural and Veterinary Diagnostic AI.
Generate 3 high-quality synthetic diagnostic Q&A pairs for a farmer dealing with:

Crop/Livestock: {crop_animal}
Disease/Condition: {condition}
Language: {language}

Output MUST be a strict JSON list of objects matching this exact schema:
[
  {{
    "instruction": "Farmer symptom query in {language}",
    "input": "",
    "output": "Detailed diagnostic explanation and practical treatment protocol in {language}"
  }}
]
Do NOT include markdown formatting, code blocks, or preamble. Return ONLY the raw JSON array.
"""

# ==================== UTILITY FUNCTIONS ====================

def cleanup_runner_storage():
    """Wipes downloaded Kaggle images from the runner disk to free space."""
    if os.path.exists(TEMP_DATA_DIR):
        shutil.rmtree(TEMP_DATA_DIR)
        print(f"🧹 Runner Storage Cleaned: Erased local directory '{TEMP_DATA_DIR}'")

def load_checkpoint(api):
    """Downloads checkpoint file from Hugging Face if present."""
    try:
        local_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=CHECKPOINT_FILE,
            repo_type="dataset",
            token=HF_TOKEN
        )
        with open(local_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        print("ℹ️ No existing checkpoint found on Hugging Face. Initializing clean execution state.")
        return {"completed_batches": []}

def save_and_push_checkpoint(api, checkpoint_data):
    """Saves updated execution state to Hugging Face."""
    local_cp = "local_checkpoint.json"
    with open(local_cp, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2)

    api.upload_file(
        path_or_fileobj=local_cp,
        path_in_repo=CHECKPOINT_FILE,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN
    )
    if os.path.exists(local_cp):
        os.remove(local_cp)
    print("📌 Progress checkpoint saved to Hugging Face.")

def generate_text_with_fallback(genai_client, prompt):
    """Executes prompt with automatic fallback across specified Gemini models."""
    for model_name in GEMINI_MODELS:
        try:
            response = genai_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
            return json.loads(response.text)
        except APIError as e:
            print(f"⚠️ API Error on {model_name}: {e}. Retrying with next fallback...")
            time.sleep(2)
        except Exception as e:
            print(f"⚠️ Unexpected error on {model_name}: {e}. Trying fallback...")
            time.sleep(2)

    raise RuntimeError("🚨 All Gemini fallback models failed to generate content!")

def get_batch_images():
    """Finds image files extracted in the temporary batch directory."""
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]
    found_images = []
    for ext in image_extensions:
        found_images.extend(glob.glob(f"{TEMP_DATA_DIR}/**/{ext}", recursive=True))
    return found_images

# ==================== CONSOLDIATION FUNCTION ====================

def consolidate_dataset(api):
    """Downloads all batch parquet files from HF, merges them into train.parquet, and pushes to root."""
    print("\n" + "="*50)
    print("🎉 ALL BATCHES COMPLETE: MERGING MULTIMODAL DATASET")
    print("="*50)

    repo_files = api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset", token=HF_TOKEN)
    batch_files = [f for f in repo_files if f.startswith("data/batch_") and f.endswith(".parquet")]

    if not batch_files:
        print("❌ No batch chunk parquet files found on Hugging Face.")
        return

    merged_dfs = []
    for file_in_repo in batch_files:
        print(f"📥 Fetching batch chunk: {file_in_repo}...")
        local_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=file_in_repo,
            repo_type="dataset",
            token=HF_TOKEN
        )
        merged_dfs.append(pd.read_parquet(local_path))

    full_df = pd.concat(merged_dfs, ignore_index=False)
    print(f"📊 Merged {len(full_df)} total multimodal records.")

    consolidated_path = "train.parquet"
    full_df.to_parquet(consolidated_path)

    print("🚀 Uploading final consolidated 'train.parquet' to Hugging Face root...")
    api.upload_file(
        path_or_fileobj=consolidated_path,
        path_in_repo="train.parquet",
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN
    )

    if os.path.exists(consolidated_path):
        os.remove(consolidated_path)
    print("✅ Multimodal Dataset Consolidation Complete!")

# ==================== MAIN BATCH PIPELINE ====================

def main():
    if not HF_REPO_ID or not HF_TOKEN or not GEMINI_API_KEY:
        raise ValueError("Missing essential env secrets: HF_REPO_ID, HF_TOKEN, or GEMINI_API_KEY.")

    api = HfApi(token=HF_TOKEN)
    genai_client = genai.Client(api_key=GEMINI_API_KEY)

    # 1. Ensure HF repo exists
    create_repo(repo_id=HF_REPO_ID, repo_type="dataset", token=HF_TOKEN, exist_ok=True)

    # 2. Read state
    checkpoint = load_checkpoint(api)
    completed_batches = set(checkpoint.get("completed_batches", []))

    remaining_batches = [b for b in KAGGLE_BATCHES if b["batch_id"] not in completed_batches]

    if not remaining_batches:
        print("⚡ All Kaggle image dataset batches are already completed.")
        consolidate_dataset(api)
        return

    print(f"📋 Found {len(remaining_batches)} pending dataset batches.")

    # Kaggle API Authentication
    kaggle_api = KaggleApi()
    kaggle_api.authenticate()

    for batch in remaining_batches:
        batch_id = batch["batch_id"]
        kaggle_slug = batch["kaggle_slug"]
        crop = batch["crop_animal"]
        condition = batch["condition"]
        lang = batch["language"]

        print(f"\n🚀 --- STARTING {batch_id}: {crop} - {condition} ({lang}) ---")

        try:
            # Step A: Download ONLY THIS Kaggle batch dataset into local runner disk
            cleanup_runner_storage()
            os.makedirs(TEMP_DATA_DIR, exist_ok=True)
            print(f"📥 Downloading Kaggle dataset '{kaggle_slug}'...")
            kaggle_api.dataset_download_files(kaggle_slug, path=TEMP_DATA_DIR, unzip=True)

            local_images = get_batch_images()
            print(f"📸 Found {len(local_images)} extracted images for this batch.")

            if not local_images:
                print(f"⚠️ No image files found in '{kaggle_slug}'. Skipping batch.")
                cleanup_runner_storage()
                continue

            # Step B: Generate Gemini Synthetic Text QA
            prompt = PROMPT_TEMPLATE.format(crop_animal=crop, condition=condition, language=lang)
            print("🤖 Generating synthetic Q&A using Gemini fallback chain...")
            qa_records = generate_text_with_fallback(genai_client, prompt)

            # Step C: Pair images with text into a Hugging Face Multimodal Dataset
            multimodal_rows = []
            # Pair available images with the generated text records
            sample_count = min(len(local_images), 20) # Process up to 20 images per batch chunk
            for i in range(sample_count):
                qa_pair = qa_records[i % len(qa_records)]
                multimodal_rows.append({
                    "instruction": qa_pair["instruction"],
                    "input": qa_pair["input"],
                    "output": qa_pair["output"],
                    "crop_animal": crop,
                    "disease_condition": condition,
                    "language": lang,
                    "image": local_images[i] # File path auto-encoded by HF Datasets
                })

            batch_df = pd.DataFrame(multimodal_rows)
            features = Features({
                "instruction": Value("string"),
                "input": Value("string"),
                "output": Value("string"),
                "crop_animal": Value("string"),
                "disease_condition": Value("string"),
                "language": Value("string"),
                "image": Image()
            })

            hf_dataset_chunk = Dataset.from_pandas(batch_df, features=features)

            # Step D: Save Parquet chunk locally & Push to Hugging Face
            chunk_filename = f"{batch_id}.parquet"
            repo_chunk_path = f"data/{chunk_filename}"
            hf_dataset_chunk.to_parquet(chunk_filename)

            print(f"📤 Pushing {chunk_filename} to Hugging Face ('{repo_chunk_path}')...")
            api.upload_file(
                path_or_fileobj=chunk_filename,
                path_in_repo=repo_chunk_path,
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                token=HF_TOKEN
            )

            if os.path.exists(chunk_filename):
                os.remove(chunk_filename)

            # Step E: CRITICAL CLEANUP — Wipe the runner disk storage immediately
            cleanup_runner_storage()

            # Step F: Update checkpoint state on Hugging Face
            completed_batches.add(batch_id)
            checkpoint["completed_batches"] = list(completed_batches)
            save_and_push_checkpoint(api, checkpoint)

            print(f"✅ Successfully finished and cleaned up {batch_id}!")

        except Exception as e:
            print(f"❌ Batch {batch_id} failed: {e}")
            cleanup_runner_storage()
            raise e

    # Consolidation after loop finishes
    if len(completed_batches) == len(KAGGLE_BATCHES):
        consolidate_dataset(api)

if __name__ == "__main__":
    main()