import os
import json
import time
import glob
import pandas as pd
from pathlib import Path
from google import genai
from google.genai import types
from huggingface_hub import HfApi, hf_hub_download

# ==========================================
# CONFIGURATION
# ==========================================
HF_REPO_ID = os.environ.get("HF_REPO_ID") # e.g. "username/agri-vet-multilingual"
HF_TOKEN = os.environ.get("HF_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

CHECKPOINT_FILE = "checkpoint.json"
DATA_DIR = "data"

# Batch definitions matching Kaggle datasets
KAGGLE_BATCHES = [
    {"batch_id": "batch_01", "crop": "cassava", "kaggle_dataset": "abdallahwagih/cassava-leaf-disease-classification"},
    {"batch_id": "batch_02", "crop": "maize", "kaggle_dataset": "sriramr/corn-or-maize-leaf-disease-dataset"},
    {"batch_id": "batch_03", "crop": "rice", "kaggle_dataset": "vbookshelf/rice-leaf-diseases"},
    {"batch_id": "batch_04", "crop": "tomato", "kaggle_dataset": "kaustubhb999/tomatoleaf"},
    {"batch_id": "batch_05", "crop": "livestock_cattle", "kaggle_dataset": "alvarobas3/cow-disease-dataset"},
]

PROMPT_TEMPLATE = """
Generate 10 synthetic AgVet advisory Q&A pairs for local farmers dealing with {crop} diseases or health issues.

Output strict JSON with this exact key structure:
{{
  "samples": [
    {{
      "crop": "{crop}",
      "question_en": "What causes black spots on leaves?",
      "answer_en": "Black spots are typically caused by fungal infections...",
      "question_sw": "Ni nini kinachosababisha madoa meusi kwenye majani?",
      "answer_sw": "Madoa meusi kwa kawaida husababishwa na maambukizi ya fangasi...",
      "question_ha": "Mene ne yake sa bakake a ganye?",
      "answer_ha": "A yawancin lokuta, bakaken cututtuka na fitowa ne saboda fungus..."
    }}
  ]
}}
"""

# ==========================================
# HUGGING FACE & CHECKPOINT UTILS
# ==========================================
api = HfApi()

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    
    # Try downloading from Hugging Face if not locally available
    try:
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=CHECKPOINT_FILE, repo_type="dataset", token=HF_TOKEN)
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {"completed_batches": []}

def save_and_push_checkpoint(checkpoint_data):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint_data, f, indent=2)
    
    api.upload_file(
        path_or_fileobj=CHECKPOINT_FILE,
        path_in_repo=CHECKPOINT_FILE,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN
    )

# ==========================================
# GEMINI GENERATION LOGIC
# ==========================================
def generate_batch_data(crop_name):
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = PROMPT_TEMPLATE.format(crop=crop_name)
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.3
        )
    )
    
    parsed = json.loads(response.text)
    return parsed.get("samples", [])

# ==========================================
# DATASET CONSOLIDATION STEP (NEW)
# ==========================================
def consolidate_dataset():
    """Downloads all batch chunks from Hugging Face, merges them into Parquet & JSONL formats, and uploads them."""
    print("\n" + "="*50)
    print("🧹 STARTING DATASET MERGE & CONSOLIDATION PHASE")
    print("="*50)
    
    # 1. Fetch repo file list
    repo_files = api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset", token=HF_TOKEN)
    chunk_files = [f for f in repo_files if f.startswith("data/batch_") and f.endswith(".jsonl")]
    
    if not chunk_files:
        print("⚠️ No batch chunk files found to consolidate.")
        return

    all_records = []
    
    # 2. Fetch and read each chunk file
    for chunk_file in chunk_files:
        print(f"📥 Downloading chunk: {chunk_file}...")
        local_path = hf_hub_download(
            repo_id=HF_REPO_ID, 
            filename=chunk_file, 
            repo_type="dataset", 
            token=HF_TOKEN
        )
        with open(local_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_records.append(json.loads(line))

    print(f"\n✓ Collected {len(all_records)} total records across {len(chunk_files)} batch chunks.")

    # 3. Build unified dataframe
    df = pd.DataFrame(all_records)
    
    merged_parquet_path = "train.parquet"
    merged_jsonl_path = "train.jsonl"
    
    df.to_parquet(merged_parquet_path, index=False)
    df.to_json(merged_jsonl_path, orient="records", lines=True, force_ascii=False)

    # 4. Push merged dataset files to root/data on Hugging Face
    print("📤 Uploading merged dataset (`data/train.parquet` & `data/train.jsonl`) to Hugging Face...")
    api.upload_file(
        path_or_fileobj=merged_parquet_path,
        path_in_repo="data/train.parquet",
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN
    )
    api.upload_file(
        path_or_fileobj=merged_jsonl_path,
        path_in_repo="data/train.jsonl",
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN
    )

    # Cleanup local merged artifacts
    if os.path.exists(merged_parquet_path): os.remove(merged_parquet_path)
    if os.path.exists(merged_jsonl_path): os.remove(merged_jsonl_path)

    print("\n🎉 Consolidation complete! All batch records are merged and ready on Hugging Face.")

# ==========================================
# MAIN EXECUTION FLOW
# ==========================================
def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    checkpoint = load_checkpoint()
    completed = set(checkpoint.get("completed_batches", []))
    
    print(f"📋 Found {len(completed)} completed batches in checkpoint: {list(completed)}")
    
    batch_processed_in_this_run = False
    
    for batch in KAGGLE_BATCHES:
        batch_id = batch["batch_id"]
        crop = batch["crop"]
        
        if batch_id in completed:
            continue  # Skip already completed batches
            
        print(f"\n🚀 Processing {batch_id} ({crop})...")
        
        # Simulate local dataset fetch / processing
        samples = generate_batch_data(crop)
        
        # Save batch chunk locally
        batch_file_path = os.path.join(DATA_DIR, f"{batch_id}_{crop}.jsonl")
        with open(batch_file_path, "w", encoding="utf-8") as f:
            for item in samples:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        # Upload chunk file to Hugging Face repo
        hf_batch_path = f"data/{batch_id}_{crop}.jsonl"
        print(f"📤 Uploading batch chunk to Hugging Face: {hf_batch_path}")
        api.upload_file(
            path_or_fileobj=batch_file_path,
            path_in_repo=hf_batch_path,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN
        )
        
        # Update and upload checkpoint
        completed.add(batch_id)
        checkpoint["completed_batches"] = sorted(list(completed))
        save_and_push_checkpoint(checkpoint)
        
        print(f"✅ Successfully finished {batch_id}.")
        batch_processed_in_this_run = True
        break  # Process only 1 batch per GitHub Action run to avoid timeouts

    # Check if ALL batches are completed
    if len(completed) >= len(KAGGLE_BATCHES):
        print("\n🏆 All Kaggle batches have finished processing!")
        consolidate_dataset()
    elif not batch_processed_in_this_run:
        print("ℹ️ No pending batches to run in this trigger.")

if __name__ == "__main__":
    main()