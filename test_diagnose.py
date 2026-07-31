"""
Quick sanity test for the deployed Render /diagnose endpoint.

Sends a few sample text-based requests (one per domain) plus one image-based
request, and prints the full response, so you can eyeball whether retrieval +
guidance generation are working end-to-end.

Usage:
    python test_diagnose.py

No API keys needed here -- this just calls YOUR already-deployed backend,
which holds its own keys as environment variables on Render.

Requires:
    synthetic_test_leaf.jpg in the same directory (a generated, clearly
    labeled synthetic test image -- NOT a real disease photo -- used only to
    verify the image-upload plumbing works, not to test real diagnostic
    accuracy on visual symptoms).
"""

import base64
import json
import os
import requests

BASE_URL = "https://gemma-nigeria-diagnosis-api-s7bg.onrender.com"
TEST_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "synthetic_test_leaf.jpg")

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


def run_image_test():
    label = "Image test (synthetic leaf image -- tests plumbing only, not real accuracy)"
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"{'='*60}")

    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"[SKIPPED] Test image not found at {TEST_IMAGE_PATH}")
        return

    with open(TEST_IMAGE_PATH, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "domain": "crop",
        "input_type": "image",
        "language_hint": "english",
        "image_base64": image_b64,
        "source": "app",
    }
    print("Request: (image payload omitted from log for brevity, "
          f"{len(image_b64)} base64 chars)")

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

    run_image_test()


if __name__ == "__main__":
    main()