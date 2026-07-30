"""
Quick sanity test for the deployed Render /diagnose endpoint.

Sends a few sample text-based requests (one per domain) and prints the
full response, so you can eyeball whether retrieval + guidance generation
are working end-to-end.

Usage:
    python test_diagnose.py

No API keys needed here -- this just calls YOUR already-deployed backend,
which holds its own keys as environment variables on Render.
"""

import json
import requests

BASE_URL = "https://gemma-nigeria-diagnosis-api-s7bg.onrender.com"

TEST_CASES = [
    {
        "label": "Crop test (should match Cassava Mosaic Disease)",
        "payload": {
            "domain": "crop",
            "input_type": "text",
            "language_hint": "english",
            "text": "My cassava leaves are turning yellow with a mosaic pattern and the plant looks stunted.",
            "source": "app",
        },
    },
    {
        "label": "Animal test (should match Newcastle Disease)",
        "payload": {
            "domain": "animal",
            "input_type": "text",
            "language_hint": "english",
            "text": "My chickens' necks are twisting backward and they have green diarrhea, some just died suddenly.",
            "source": "app",
        },
    },
    {
        "label": "Human test (should match Malaria)",
        "payload": {
            "domain": "human",
            "input_type": "text",
            "language_hint": "english",
            "text": "I have a fever, chills, and I'm sweating a lot, with a bad headache.",
            "source": "app",
        },
    },
]


def run_test(label, payload):
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"{'='*60}")
    print(f"Request: {json.dumps(payload, indent=2)}")

    try:
        resp = requests.post(f"{BASE_URL}/diagnose", json=payload, timeout=60)
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request failed: {e}")
        return

    print(f"Status code: {resp.status_code}")
    try:
        print(f"Response:\n{json.dumps(resp.json(), indent=2)}")
    except ValueError:
        print(f"Raw response (not JSON): {resp.text}")


def main():
    print(f"Testing backend at {BASE_URL}")
    print("Note: first request may take 30-60s if Render's free tier had to cold-start.")

    for case in TEST_CASES:
        run_test(case["label"], case["payload"])


if __name__ == "__main__":
    main()