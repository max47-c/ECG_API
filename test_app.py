"""
test_app.py — End-to-end test for the unified cloud ECG API.
Usage: python test_app.py
Requires: scipy (for loading .mat files), requests
"""
import scipy.io as sio
import requests
import numpy as np
import os

# ── Config ──
BASE_URL = os.environ.get("ECG_API_URL", "http://localhost:8000")
ANALYZE_URL = f"{BASE_URL}/analyze"
HEALTH_URL = f"{BASE_URL}/health"

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")

# ── Health check ──
print("=" * 50)
print("ECG Cloud API — End-to-End Test")
print("=" * 50)

try:
    r = requests.get(HEALTH_URL, timeout=10)
    r.raise_for_status()
    print(f"\n✅ Health check: {r.json()}")
except Exception as e:
    print(f"\n❌ Health check failed: {e}")
    print("Make sure the server is running: uvicorn app:app --port 8000")
    exit(1)

# ── Test each sample ──
test_files = {
    "A00004.mat": "A (Atrial Fibrillation)",
    "N00001.mat": "N (Normal)",
    "O00008.mat": "O (Other)",
    "Z00022.mat": "Z (Noisy)",
}

for filename, expected_desc in test_files.items():
    filepath = os.path.join(TEST_DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"\n⚠️  Skipping {filename} — file not found")
        continue

    mat = sio.loadmat(filepath)
    # Find the data array
    data_key = None
    for k in mat.keys():
        if not k.startswith("__") and isinstance(mat[k], np.ndarray):
            data_key = k
            break

    if data_key is None:
        print(f"\n⚠️  Skipping {filename} — no data array found")
        continue

    signal = mat[data_key].flatten().tolist()

    payload = {
        "values": signal,
        "sampling_rate": 300,
        "threshold": 0.5,
    }

    print(f"\n{'─' * 50}")
    print(f"Testing: {filename} (expected: {expected_desc})")
    print(f"Signal length: {len(signal)} samples")

    try:
        resp = requests.post(ANALYZE_URL, json=payload, timeout=120)
        resp.raise_for_status()
        result = resp.json()
        
        label = result["label"]
        conf = result["confidence"]
        confs = result["confidences"]
        
        print(f"  Label:      {label}")
        print(f"  Confidence: {conf:.4f}")
        print(f"  All confidences: {confs}")
        print(f"  Quality:    std={result['quality']['std']:.4f}, "
              f"zero_ratio={result['quality']['zero_ratio']:.4f}")
        print(f"  ✅ Success")
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Error: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"  Response: {e.response.text}")

print(f"\n{'=' * 50}")
print("Tests complete!")
