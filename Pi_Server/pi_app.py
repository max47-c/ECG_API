# pi_app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import requests
import base64
from PIL import Image
import io
from model import load_model, predict   # <-- using your existing model.py
from fastapi.middleware.cors import CORSMiddleware
import sys

app = FastAPI(title="ECG Analysis API - Pi Edge Node")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8081", "http://localhost:3000", "*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

# CLOUD_URL = "http://localhost:8000/preprocess"
CLOUD_URL = "https://ecg-cloud-api-o653.onrender.com/preprocess"

# Load ONNX session using your model.py
try:
    session = load_model()   # returns ort.InferenceSession
except Exception as e:
    print(f"CRITICAL: Model loading failed - {e}")
    sys.exit(1)

# ----- Pydantic models (same as before) -----
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

def validate_signal(values: list[float]) -> dict:
    arr = np.array(values, dtype=np.float32)
    zero_ratio = float(np.mean(arr == 0.0))
    clipped = float(np.mean(np.abs(arr) > 5.0))
    std = float(np.std(arr))
    return {"zero_ratio": zero_ratio, "clipped_ratio": clipped, "std": std, "length": len(arr)}

@app.post("/analyze", response_model=ECGResponse)
def analyze_ecg(req: ECGRequest):
    # 1. Local validation
    quality = validate_signal(req.values)
    
    # 2. Call cloud for preprocessing
    try:
        cloud_resp = requests.post(
            CLOUD_URL,
            json={"values": req.values, "sampling_rate": req.sampling_rate},
            timeout=65.0
        )
        cloud_resp.raise_for_status()
        data = cloud_resp.json()
        features = data["features"]          # [lyap, alpha1]
        img_base64 = data["image_base64"]
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Cloud preprocessing unavailable: {e}")
    
    # 3. Decode base64 to PIL Image
    try:
        img_bytes = base64.b64decode(img_base64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image decoding failed: {e}")
    
    # 4. Inference using your model.py predict()
    result = predict(session, image, features)
    if result is None:
        raise HTTPException(status_code=500, detail="Inference failed")
    
    confidences = result["confidences"]
    top_conf = float(max(confidences.values()))
    label = result["label"] if top_conf >= req.threshold else "uncertain"
    
    quality_response = QualityMetrics(
        length=quality["length"],
        normalized_length=3000,
        sampling_rate=300,
        std=quality["std"],
        zero_ratio=quality["zero_ratio"],
        clipped_ratio=quality["clipped_ratio"],
    )
    
    return ECGResponse(
        label=label,
        confidence=top_conf,
        confidences=confidences,
        quality=quality_response
    )

@app.get("/health")
def health():
    return {"status": "ok"}