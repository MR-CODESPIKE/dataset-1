"""
Embed the ground-truth knowledge base and upsert it into Chroma Cloud.

This is a ONE-TIME (or re-run-on-change) script, not something that runs per
request. Run it whenever ground_truth_kb.csv changes on Hugging Face.

- Downloads ground_truth_kb.csv from Hugging Face
- Embeds each row's searchable text using Gemini's gemini-embedding-001 model
- Upserts into a Chroma Cloud collection, storing all KB fields as metadata
  so the Render backend can retrieve the full record directly from Chroma
  (no second lookup needed)

Usage:
    export GEMINI_API_KEY="your-key-here"
    export CHROMA_API_KEY="your-chroma-cloud-key-here"
    export CHROMA_TENANT="your-tenant-id"        # optional if key is single-DB scoped
    export CHROMA_DATABASE="your-database-name"  # optional if key is single-DB scoped
    python embed_and_upsert.py

Requires:
    pip install chromadb google-genai requests
"""

import csv
import io
import os
import sys
import time

import requests

try:
    import chromadb
except ImportError:
    print("Missing dependency. Run: pip install chromadb google-genai requests")
    sys.exit(1)

try:
    from google import genai
except ImportError:
    print("Missing dependency. Run: pip install chromadb google-genai requests")
    sys.exit(1)

# ---- Config ----
HF_CSV_URL = "https://huggingface.co/datasets/MR-CODESPIKE/ground-truth-ob/resolve/main/ground_truth_kb.csv"
COLLECTION_NAME = "disease_kb"
EMBEDDING_MODEL = "gemini-embedding-001"
BATCH_SLEEP = 0.5  # be nice to embedding API rate limits


def download_kb(url):
    print(f"Downloading ground-truth KB from {url} ...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))
    rows = list(reader)
    print(f"Loaded {len(rows)} ground-truth rows.")
    return rows


def build_searchable_text(row):
    """
    The text we actually embed. This should read like how a user's extracted
    symptom description would be phrased, since that's what gets compared
    against at query time. Symptoms are the primary signal; name/category
    give the embedding model extra grounding.
    """
    return (
        f"{row['category']} — {row['name']}. "
        f"Symptoms: {row['symptoms']}"
    )


def embed_text(client, text):
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )
    return result.embeddings[0].values


def get_chroma_client():
    api_key = os.environ.get("CHROMA_API_KEY")
    if not api_key:
        print("Set CHROMA_API_KEY environment variable first.")
        sys.exit(1)

    tenant = os.environ.get("CHROMA_TENANT")
    database = os.environ.get("CHROMA_DATABASE")

    if tenant and database:
        return chromadb.CloudClient(tenant=tenant, database=database, api_key=api_key)
    # auto-resolve tenant/database from API key scope if not explicitly provided
    return chromadb.CloudClient(api_key=api_key)


def main():
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        print("Set GEMINI_API_KEY environment variable first.")
        sys.exit(1)

    genai_client = genai.Client(api_key=gemini_key)
    chroma_client = get_chroma_client()

    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    kb_rows = download_kb(HF_CSV_URL)

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for i, row in enumerate(kb_rows, start=1):
        text = build_searchable_text(row)
        print(f"[{i}/{len(kb_rows)}] Embedding {row['id']} ({row['name']})...")

        try:
            vector = embed_text(genai_client, text)
        except Exception as e:
            print(f"  [WARN] Failed to embed {row['id']}: {e}")
            continue

        ids.append(row["id"])
        embeddings.append(vector)
        documents.append(text)
        # Store every KB field as metadata so Render can build the full
        # response directly from what Chroma returns, no second lookup.
        metadatas.append({
            "domain": row["domain"],
            "name": row["name"],
            "category": row["category"],
            "symptoms": row["symptoms"],
            "cause": row["cause"],
            "prevention": row["prevention"],
            "guidance": row["guidance"],
            "severity": row["severity"],
            "source": row["source"],
            "requires_professional": row["requires_professional"],
            "disclaimer_required": row["disclaimer_required"],
        })

        time.sleep(BATCH_SLEEP)

    if not ids:
        print("Nothing embedded successfully -- aborting upsert.")
        sys.exit(1)

    print(f"\nUpserting {len(ids)} records into Chroma Cloud collection '{COLLECTION_NAME}'...")
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"Done. Collection '{COLLECTION_NAME}' now has {collection.count()} records.")


if __name__ == "__main__":
    main()