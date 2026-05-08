import onnxruntime as ort
import numpy as np
from PIL import Image
import json
import os

# Chemins
MODEL_PATH = "ecg_model_single.onnx"
CONFIG_PATH = "config.json"

# Chargement de la configuration
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

MEAN = np.array(config.get('NORMALIZE_MEAN', [0.485, 0.456, 0.406]), dtype=np.float32)
STD = np.array(config.get('NORMALIZE_STD', [0.229, 0.224, 0.225]), dtype=np.float32)

# Classes
if 'LABELS' in config:
    CLASSES = list(config['LABELS'].keys())
elif 'INV_LABELS' in config:
    inv = config['INV_LABELS']
    indices = sorted(int(k) for k in inv.keys())
    CLASSES = [inv[str(i)] for i in indices]
else:
    CLASSES = ["N", "A", "O", "Z"]

def preprocess_image(image_pil):
    """Prétraitement de l'image sans torchvision"""
    # Redimensionner
    img = image_pil.resize((224, 224))
    # Convertir en numpy array [0,255] -> [0,1]
    img = np.array(img).astype(np.float32) / 255.0
    # Normaliser
    img = (img - MEAN) / STD
    # Transposer HWC -> CHW
    img = np.transpose(img, (2, 0, 1))
    # Ajouter dimension batch
    img = np.expand_dims(img, axis=0)
    return img

def load_model():
    """Charge la session ONNX Runtime"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modèle ONNX introuvable : {MODEL_PATH}")
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    print("Session ONNX chargée avec succès.")
    return session

def predict(session, image_pil, ecg_features):
    """
    session : session ONNX
    image_pil : PIL.Image
    ecg_features : list ou np.array de forme (2,)
    Retourne un dict avec label et confiances.
    """
    # --- Validation image ---
    if not isinstance(image_pil, Image.Image):
        print("Erreur : image_pil doit être un objet PIL.Image")
        return None
    try:
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
    except Exception as e:
        print(f"Erreur conversion image : {e}")
        return None

    # --- Prétraitement image ---
    try:
        img_tensor = preprocess_image(image_pil)
    except Exception as e:
        print(f"Erreur prétraitement image : {e}")
        return None

    # --- Validation features ---
    try:
        if not isinstance(ecg_features, np.ndarray):
            ecg_features = np.array(ecg_features, dtype=np.float32)
        if ecg_features.shape == (2,):
            ecg_features = ecg_features.reshape(1, 2)
        elif ecg_features.shape == (1, 2):
            pass
        else:
            print(f"Erreur forme features : {ecg_features.shape}, attendu (2,) ou (1,2)")
            return None
    except Exception as e:
        print(f"Erreur features : {e}")
        return None

    # --- Inférence ONNX ---
    try:
        inputs = {'image': img_tensor.astype(np.float32), 'features': ecg_features.astype(np.float32)}
        outputs = session.run(None, inputs)[0]  # shape (1,4)
        # Softmax manuel
        exp = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        probs = probs[0]
    except Exception as e:
        print(f"Erreur inférence ONNX : {e}")
        return None

    # Sécurité
    probs = np.nan_to_num(probs, nan=0.0)
    if probs.sum() == 0:
        print("Attention : probabilités nulles.")
        return None

    idx = int(np.argmax(probs))
    return {
        "label": CLASSES[idx] if idx < len(CLASSES) else "Unknown",
        "confidences": {cls: float(probs[i]) for i, cls in enumerate(CLASSES) if i < len(probs)}
    }
