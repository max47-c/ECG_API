# app.py — Unified Cloud ECG Analysis API (100% cloud-based)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import math
from scipy.signal import resample_poly
import base64
from io import BytesIO

from preprocess import butter_filter, DFA_main, lyapunovExponent
from image_utils import make_wsst_image
from model import load_model, predict

import sys

app = FastAPI(
    title="ECG Analysis API — Cloud",
    description="100% cloud-based ECG classification using DualInputShuffleNet",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ONNX model at startup
try:
    session = load_model()
except Exception as e:
    print(f"CRITICAL: Model loading failed — {e}")
    sys.exit(1)


# ───── Pydantic models ─────

class ECGRequest(BaseModel):
    values: list[float]
    sampling_rate: int = Field(300, ge=50, le=2000)
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class QualityMetrics(BaseModel):
    length: int
    normalized_length: int
    sampling_rate: int
    std: float
    zero_ratio: float
    clipped_ratio: float


class ECGResponse(BaseModel):
    label: str
    confidence: float
    confidences: dict[str, float]
    quality: QualityMetrics


# ───── Helper functions ─────

def normalize_length(x: np.ndarray, target_len: int) -> np.ndarray:
    """Normalize signal length by tiling (if short) or center-cropping (if long)."""
    n = len(x)
    if n == target_len:
        return x
    if n < target_len:
        reps = int(np.ceil(target_len / n)) if n > 0 else 1
        y = np.tile(x if n > 0 else np.zeros(1, dtype=np.float32), reps)[:target_len]
        return y.astype(np.float32)
    start = (n - target_len) // 2
    return x[start : start + target_len].astype(np.float32)


def validate_signal(values: list[float]) -> dict:
    """Quick quality check on the raw signal."""
    arr = np.array(values, dtype=np.float32)
    return {
        "zero_ratio": float(np.mean(arr == 0.0)),
        "clipped_ratio": float(np.mean(np.abs(arr) > 5.0)),
        "std": float(np.std(arr)),
        "length": len(arr),
    }


# ───── Endpoints ─────

@app.post("/analyze", response_model=ECGResponse)
def analyze_ecg(req: ECGRequest):
    """
    Full ECG analysis pipeline:
    1. Validate signal quality
    2. Resample to 300 Hz
    3. Normalize to 10 seconds (3000 samples)
    4. Butterworth high-pass filter
    5. Extract DFA α₁ + Lyapunov exponent
    6. Generate WSST spectrogram
    7. ONNX inference → classification
    """
    # 1. Quality validation
    quality = validate_signal(req.values)

    ecg_in = np.array(req.values, dtype=np.float32)
    fs_in = int(req.sampling_rate)
    fs_target = 300

    # 2. Resample to 300 Hz
    if fs_in != fs_target and fs_in > 0:
        g = math.gcd(fs_target, fs_in)
        up = fs_target // g
        down = fs_in // g
        try:
            ecg = resample_poly(ecg_in, up, down).astype(np.float32)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Resampling error: {e}")
    else:
        ecg = ecg_in.astype(np.float32)

    # 3. Normalize length to 10 seconds (3000 points at 300 Hz)
    target_len = fs_target * 10
    ecg = normalize_length(ecg, target_len)

    # 4. Butterworth filter
    filtered = butter_filter(ecg, fs=fs_target)

    # 5. Feature extraction
    _, alpha1 = DFA_main(filtered)
    lyap = lyapunovExponent(filtered, fs=fs_target)
    features = [float(lyap), float(alpha1)]

    # 6. WSST spectrogram
    image = make_wsst_image(filtered, fs=fs_target, fast=False, bypass=False)

    # 7. ONNX inference
    result = predict(session, image, features)
    if result is None:
        raise HTTPException(status_code=500, detail="Inference failed")

    confidences = result["confidences"]
    top_conf = float(max(confidences.values()))
    label = result["label"] if top_conf >= req.threshold else "uncertain"

    return ECGResponse(
        label=label,
        confidence=top_conf,
        confidences=confidences,
        quality=QualityMetrics(
            length=quality["length"],
            normalized_length=target_len,
            sampling_rate=fs_target,
            std=quality["std"],
            zero_ratio=quality["zero_ratio"],
            clipped_ratio=quality["clipped_ratio"],
        ),
    )


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0", "mode": "cloud-only"}
