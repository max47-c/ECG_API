# ECG Analysis API

A cloud-based ECG classification API using a **DualInputShuffleNet** deep learning model. The API accepts raw ECG signals, performs signal processing and feature extraction, generates WSST (Wavelet Synchrosqueezed Transform) spectrograms, and classifies the signal into one of four categories via ONNX Runtime inference.

## Architecture

```
┌─────────────┐     ┌─────────────────────────────────────────────────────┐
│   Client    │────▶│                   FastAPI Server                     │
│  (.mat/.csv)│     │                                                     │
└─────────────┘     │  ┌──────────┐  ┌───────────┐  ┌──────────────────┐  │
                    │  │ Signal   │─▶│ Feature   │─▶│  DualInput       │  │
                    │  │ Pipeline │  │ Extraction │  │  ShuffleNet      │  │
                    │  │          │  │           │  │  (ONNX Runtime)   │  │
                    │  │ • Resample│  │ • DFA α₁  │  │                  │  │
                    │  │ • Normalize│  │ • Lyapunov│  │  ┌─────┐        │  │
                    │  │ • HPF     │  │           │  │  │Image│        │  │
                    │  └──────────┘  └───────────┘  │  │Path │        │  │
                    │                                │  └─────┘        │  │
                    │  ┌──────────┐                   │  ┌────────┐    │  │
                    │  │  WSST    │──────────────────▶│  │Features│    │  │
                    │  │Spectrogram│                   │  └────────┘    │  │
                    │  └──────────┘                   └──────────────────┘  │
                    └─────────────────────────────────────────────────────┘
```

## Classification Labels

| Label | Description |
|-------|-------------|
| **N** | Normal sinus rhythm |
| **A** | Atrial Fibrillation |
| **O** | Other abnormality |
| **Z** | Noisy / unreadable signal |

## Processing Pipeline

### 1. Signal Reception
- Accepts raw ECG values as a JSON list via `POST /analyze`
- Configurable `sampling_rate` (50–2000 Hz, default 300) and confidence `threshold` (0.0–1.0, default 0.5)

### 2. Quality Validation
- Checks zero-value ratio (flatline detection)
- Checks clipped ratio (amplitudes > 5.0)
- Computes signal standard deviation

### 3. Resampling
- Resamples signal to **300 Hz** using `scipy.signal.resample_poly`
- Handles arbitrary input sampling rates via rational resampling (GCD-based up/down factors)

### 4. Length Normalization
- Normalizes to **3000 samples** (10 seconds at 300 Hz)
- Short signals are tiled (repeated); long signals are center-cropped

### 5. Butterworth High-Pass Filter
- 3rd-order Butterworth filter
- Cutoff: **0.67 Hz** (removes baseline wander)
- Zero-phase filtering via `scipy.signal.filtfilt`
- Signal divided by 1000 (µV → mV conversion)

### 6. Feature Extraction

#### Detrended Fluctuation Analysis (DFA)
- Computes scaling exponent **α₁**
- Window sizes: 100 to 1000 samples (step 100)
- First-order polynomial detrending per window
- Slope of `log(win_length)` vs `log(fluctuation)` gives α₁

#### Largest Lyapunov Exponent
- Phase space reconstruction via time-delay embedding (dimension `m=10`, delay `τ=1`)
- Nearest neighbor search using KD-Tree
- Tracks average divergence of neighboring trajectories
- Linear regression on `log(divergence)` over time yields the exponent

### 7. WSST Spectrogram Generation
- **Wavelet Synchrosqueezed Transform** using a bump wavelet (`ssqueezepy`)
- 48 voice scales (the code also supports a 24-scale "fast" path, but `/analyze`
  always uses the full 48-scale render)
- Rendered as a 224×224 RGB image through a pre-computed Matplotlib `jet` LUT
  applied with NumPy — no figure/axes/PNG round-trip per request
  (the old Matplotlib path took ~20 s per request, this one ~0.1 s)

### 8. ONNX Inference
- **DualInputShuffleNet**: a dual-branch architecture combining:
  - **Image branch**: ShuffleNetV2 (pretrained) processing the WSST spectrogram
  - **Feature branch**: Fully-connected layers (2→256→128) processing the DFA + Lyapunov features
  - **Classifier**: FC(1152→512→256→4) fusing both branches
- Image preprocessing: resize to 224×224, normalize with ImageNet mean/std, HWC→CHW transpose
- Softmax applied to raw logits → per-class probabilities

### 9. Response
- If top confidence ≥ threshold → returns predicted label
- If top confidence < threshold → returns `"uncertain"`
- Returns full confidence vector and signal quality metrics

### 10. Concurrency
- Heavy analysis runs in a thread pool, serialized by an async semaphore
  (`MAX_CONCURRENCY`, default **1**)
- Requests beyond that wait in a bounded queue (`MAX_QUEUE`, default **8**)
- When the queue is full the API answers **503** with `Retry-After: 10` and a
  `detail` message — clients should back off and retry

## API Endpoints

### `POST /analyze`

Analyze an ECG signal.

**Request body:**
```json
{
  "values": [0.12, -0.34, 0.56, ...],
  "sampling_rate": 300,
  "threshold": 0.5
}
```

**Response:**
```json
{
  "label": "N",
  "confidence": 0.978,
  "confidences": {
    "N": 0.978,
    "A": 0.012,
    "O": 0.007,
    "Z": 0.003
  },
  "quality": {
    "length": 6500,
    "normalized_length": 3000,
    "sampling_rate": 300,
    "std": 0.423,
    "zero_ratio": 0.0,
    "clipped_ratio": 0.0
  }
}
```

### `GET /health`

Health check.

```json
{
  "status": "ok",
  "version": "2.1.0",
  "mode": "cloud-only",
  "max_concurrency": 1,
  "max_queue": 8,
  "waiting": 0
}
```

## Project Structure

```
ECG_API/
├── app.py                   # FastAPI application (endpoints, pipeline orchestration)
├── model.py                 # ONNX model loading, preprocessing, inference
├── preprocess.py             # Butterworth filter, DFA, Lyapunov exponent
├── image_utils.py            # WSST spectrogram generation
├── config.json               # Model configuration (labels, normalization, architecture)
├── ecg_model_single.onnx     # Trained DualInputShuffleNet model
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker image definition
├── test_app.py               # End-to-end test script
├── test_data/                # Test ECG signals (.mat files)
│   ├── A00004.mat            # Atrial Fibrillation
│   ├── N00001.mat            # Normal
│   ├── O00008.mat            # Other
│   └── Z00022.mat            # Noisy
├── Pi_Server/                # (placeholder — Raspberry Pi variant)
├── Cloud_Server/             # Contains test virtual environment
└── README.md                 # This file
```

## Deployment

The service is deployed on Render at `https://ecg-api-gakf.onrender.com`
(`POST /analyze`, `GET /health`). The cloud-only branch does not need a
Raspberry Pi.

### Docker

```bash
docker build -t ecg-api .
docker run -p 8000:8000 ecg-api
```

### Local

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Testing

Run the end-to-end test suite:

```bash
# Start the server first, then:
python test_app.py
```

The test script sends each `.mat` file in `test_data/` to the `/analyze` endpoint and verifies successful classification.

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computation |
| `scipy` | Signal processing (resampling, filtering, statistics) |
| `fastapi` | Web framework |
| `uvicorn` | ASGI server |
| `Pillow` | Image manipulation |
| `matplotlib` | Spectrogram rendering |
| `ssqueezepy` | Synchrosqueezed Wavelet Transform |
| `onnxruntime` | ONNX model inference |

## Model Architecture: DualInputShuffleNet

```
Input Image (224×224×3)          Input Features (2)
        │                              │
  ShuffleNetV2_x1.0               FC(2→256) → BN → ReLU → Dropout
  (pretrained)                          │
        │                           FC(256→128) → BN → ReLU → Dropout
        │                              │
    Flatten (1024)                  Flatten (128)
        │                              │
        └───────── Concat (1152) ──────┘
                        │
              FC(1152→512) → BN → ReLU → Dropout
                        │
              FC(512→256) → BN → ReLU → Dropout
                        │
                    FC(256→4)
                        │
                     Softmax
                        │
                  [N, A, O, Z]
```
