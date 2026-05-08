# cloud_app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import math
from scipy.signal import resample_poly
import base64
from io import BytesIO

# Your original preprocessing modules (must be available on cloud)
from preprocess import butter_filter, DFA_main, lyapunovExponent
from image_utils import make_wsst_image

app = FastAPI(title="Cloud Preprocessing Service")

class PreprocessRequest(BaseModel):
    values: list[float]
    sampling_rate: int  # original sampling rate (100-1000 Hz)

class PreprocessResponse(BaseModel):
    features: list[float]   # [lyapunov, alpha1]
    image_base64: str       # WSST image encoded as base64
    filtered_signal: list[float]  # optional, for debugging

def normalize_length(x: np.ndarray, target_len: int) -> np.ndarray:
    n = len(x)
    if n == target_len:
        return x
    if n < target_len:
        reps = int(np.ceil(target_len / n)) if n > 0 else 1
        y = np.tile(x if n > 0 else np.zeros(1, dtype=np.float32), reps)[:target_len]
        return y.astype(np.float32)
    start = (n - target_len) // 2
    return x[start:start + target_len].astype(np.float32)

@app.post("/preprocess", response_model=PreprocessResponse)
def preprocess(req: PreprocessRequest):
    # Validate sampling rate
    if req.sampling_rate < 100 or req.sampling_rate > 1000:
        raise HTTPException(status_code=422, detail="Unsupported sampling_rate")
    
    ecg_in = np.array(req.values, dtype=np.float32)
    fs_in = int(req.sampling_rate)
    fs_target = 300
    
    # Resample to 300 Hz
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
    
    # Normalize length to 10 seconds (3000 points)
    target_len = fs_target * 10
    ecg = normalize_length(ecg, target_len)
    
    # Filter
    filtered = butter_filter(ecg, fs=fs_target)
    
    # Features
    _, alpha1 = DFA_main(filtered)
    lyap = lyapunovExponent(filtered, fs=fs_target)
    features = [float(lyap), float(alpha1)]
    
    # WSST image (fast=False, bypass=False as per original)
    # Note: make_wsst_image may need fast/bypass params; here we use defaults
    # Adjust if your function expects extra args
    image = make_wsst_image(filtered, fs=fs_target, fast=False, bypass=False)
    
    # Encode image to base64 string for JSON transport
    buffered = BytesIO()
    image.save(buffered, format="PNG")  # assuming PIL Image
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return PreprocessResponse(
        features=features,
        image_base64=img_base64,
        filtered_signal=filtered.tolist()
    )

@app.get("/health")
def health():
    return {"status": "ok"}