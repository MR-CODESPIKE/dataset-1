import os
import json
import time
import glob
import pandas as pd
from pathlib import Path
from google import genai
from google.genai import types
from huggingface_hub import HfApi, hf_hub_download, create_repo

# Environment configurations
HF_REPO_ID = os.environ.get("HF_REPO_ID")
HF_TOKEN = os.environ.get("HF_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

CHECKPOINT_FILE = "checkpoint.json"
DATA_DIR = "data"

# Agri-Vet Multilingual dataset batch specifications
KAGGLE_BATCHES = [
    {"batch_id": "batch_01", "crop_animal": "Cassava", "disease_condition": "Mosaic Disease", "language": "English"},
    {"batch_id": "batch_02", "crop_animal": "Cassava", "disease_condition": "Brown Streak", "language": "Swahili"},
    {"batch_id": "batch_03", "crop_animal": "Maize", "disease_condition": "Fall Armyworm", "language": "Hausa"},
    {"batch_id": "batch_04", "crop_animal": "Maize", "disease_condition": "Maize Lethal Necrosis", "language": "Yoruba"},
    {"batch_id": "batch_05", "crop_animal": "Cattle", "disease_condition": "Foot and Mouth Disease", "language": "Fulfulde"},
    {"batch_id": "batch_06", "crop_animal": "Poultry", "disease_condition": "Newcastle Disease", "language": "Igbo"},
    {"batch_id": "batch_07", "crop_animal": "Tomato", "disease_condition": "Late Blight", "language": "French"},
    {"batch_id": "batch_08", "crop_animal": "Goat/Sheep", "disease_condition": "PPR (Peste des Petits Ruminants)", "language": "Amharic"}
]

PROMPT_TEMPLATE = """
You are an expert Agricultural and Veterinary Specialist.
Generate synthetic dataset entries for Agri-Vet diagnostic assistance.

Crop/Animal: {crop_animal}
Target Disease/Condition: {disease_condition}
Language: {language}

Generate a JSON list of 5 high-quality QA / instruction tuning pairs formatted strictly as:
[
  {{
    "instruction": "Symptom description or query from a farmer in {language} regarding {crop_animal} {disease_condition}",
    "input": "",
    "output": "Detailed diagnostic and treatment recommendation in {language}",
    "metadata": {{
      "crop_animal": "{crop_animal}",
      "condition": "{disease_condition}",
      "language": "{language}"
    }}
  }}
]
Return ONLY valid JSON with no markdown tags or conversational text.
"""

def load_checkpoint():
    """Load execution state from the Hugging Face repository if present."""
    try:
        local_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=CHECKPOINT_FILE,
            repo_type="dataset",
            token=HF_TOKEN
        )
        with open(local_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"ℹ️ No existing checkpoint found on Hugging Face ({e}). Initializing clean state.")
        return {"completed_batches": []}

def save_and_push_checkpoint(checkpoint, api):
    """Save progress locally and push update to Hugging Face."""
    os.makedirs(DATA_DIR, exist_ok=True)
    local_cp_path = os.path.join(DATA_DIR, CHECKPOINT_FILE)
    with open(local_cp_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)
    
    api.upload_file(
        path_or_fileobj=local_cp_path,
        path_in_repo=CHECKPOINT_FILE,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message="Update progress checkpoint"
    )
    print("📌 Progress checkpoint synchronized to Hugging Face.")

def generate_batch_data(client, batch_info):
    """Generate synthetic QA samples using Gemini 2.5 Flash."""
    prompt = PROMPT_TEMPLATE.format(
        crop_animal=batch_info["crop_animal"],
        disease_condition=batch_info["disease_condition"],
        language=batch_info["language"]
    )
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            response_mime_type="application/json"
        )
    )
    
    return json.loads(response.text)

def consolidate_dataset(api):
    """Fetch all generated batch files, merge into unified Parquet/JSONL files, and push to HF."""
    print("\n🎉 All batches complete! Starting dataset consolidation...")
    
    merged_data = []
    
    for batch in KAGGLE_BATCHES:
        filename = f"{DATA_DIR}/{batch['batch_id']}_{batch['crop_animal'].lower().replace('/', '_')}.jsonl"
        try:
            local_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                repo_type="dataset",
                token=HF_TOKEN
            )
            with open(local_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        merged_data.append(json.loads(line))
        except Exception as e:
            print(f"⚠️ Could not download {filename} for consolidation: {e}")

    if not merged_data:
        print("❌ No batch data found for consolidation.")
        return

    consolidated_jsonl = "train.jsonl"
    consolidated_parquet = "train.parquet"

    # Save local unified files
    with open(consolidated_jsonl, "w", encoding="utf-8") as f:
        for entry in merged_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    df = pd.DataFrame(merged_data)
    df.to_parquet(consolidated_parquet, index=False)

    print(f"📊 Consolidated {len(merged_data)} total samples.")

    # Upload consolidated outputs to repository root
    print("🚀 Pushing consolidated train.parquet to Hugging Face...")
    api.upload_file(
        path_or_fileobj=consolidated_parquet,
        path_in_repo="train.parquet",
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message="Add consolidated train.parquet dataset"
    )

    print("🚀 Pushing consolidated train.jsonl to Hugging Face...")
    api.upload_file(
        path_or_fileobj=consolidated_jsonl,
        path_in_repo="train.jsonl",
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message="Add consolidated train.jsonl dataset"
    )

    print("✅ Final dataset successfully consolidated and published!")

def main():
    if not HF_REPO_ID or not HF_TOKEN or not GEMINI_API_KEY:
        raise ValueError("Missing required environment variables: HF_REPO_ID, HF_TOKEN, or GEMINI_API_KEY.")

    os.makedirs(DATA_DIR, exist_ok=True)
    api = HfApi(token=HF_TOKEN)
    client = genai.Client(api_key=GEMINI_API_KEY)

    # 1. Ensure the dataset repository exists on Hugging Face
    print(f"🔍 Ensuring Hugging Face repository '{HF_REPO_ID}' exists...")
    create_repo(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN,
        exist_ok=True
    )
    print("✅ Repository verified on Hugging Face.")

    # 2. Check current progress
    checkpoint = load_checkpoint()
    completed = set(checkpoint.get("completed_batches", []))

    remaining_batches = [b for b in KAGGLE_BATCHES if b["batch_id"] not in completed]

    if not remaining_batches:
        print("⚡ All batches have already been processed.")
        consolidate_dataset(api)
        return

    # 3. Process ALL remaining batches in this single execution run
    print(f"🔄 Found {len(remaining_batches)} remaining batches. Processing all in this run...")

    for current_batch in remaining_batches:
        batch_id = current_batch["batch_id"]
        crop = current_batch["crop_animal"].lower().replace("/", "_")
        output_filename = f"{batch_id}_{crop}.jsonl"
        local_filepath = os.path.join(DATA_DIR, output_filename)
        repo_filepath = f"{DATA_DIR}/{output_filename}"

        print(f"\n⚙️ Processing {batch_id}: {current_batch['crop_animal']} - {current_batch['disease_condition']} ({current_batch['language']})...")

        try:
            records = generate_batch_data(client, current_batch)
            
            with open(local_filepath, "w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            print(f"💾 Saved {len(records)} generated records to {local_filepath}.")

            # Upload batch output file
            print(f"📤 Uploading {output_filename} to Hugging Face...")
            api.upload_file(
                path_or_fileobj=local_filepath,
                path_in_repo=repo_filepath,
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                commit_message=f"Add generated dataset batch {batch_id}"
            )
            print(f"✅ Successfully uploaded {output_filename}.")

            # Update checkpoint
            completed.add(batch_id)
            checkpoint["completed_batches"] = list(completed)
            save_and_push_checkpoint(checkpoint, api)

            # Small delay to respect rate limits
            time.sleep(2)

        except Exception as e:
            print(f"❌ Execution failed on {batch_id}: {e}")
            raise e

    # 4. Trigger final dataset compilation once loop finishes all batches
    if len(completed) == len(KAGGLE_BATCHES):
        consolidate_dataset(api)

if __name__ == "__main__":
    main()