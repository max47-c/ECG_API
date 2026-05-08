import scipy.io as sio
import requests
import base64
import numpy as np
from PIL import Image
import io
import os

# Path to your .mat file
mat_path = r"E:\ECG_API\cloud_server\test_data\A00004.mat"

# Load the .mat file
mat = sio.loadmat(mat_path)

# Display keys to help identify the ECG signal array
print("Keys in .mat file:", [k for k in mat.keys() if not k.startswith('__')])

# Assume the ECG signal is stored under the first array key (often 'val', 'ecg', or something else)
data_key = None
for k in mat.keys():
    if not k.startswith('__') and isinstance(mat[k], np.ndarray):
        data_key = k
        break

if data_key is None:
    raise ValueError("No data array found in .mat file")

signal = mat[data_key].flatten().tolist()
print(f"Using key '{data_key}', signal length: {len(signal)}")

# Sampling rate – adjust based on your knowledge of the data.
# Common: 1000 Hz, 500 Hz, 360 Hz. For A0004 from PTB-XL, often 1000 Hz.
# Try 1000 first; if results seem off, change.
sampling_rate = 1000

payload = {
    "values": signal,
    "sampling_rate": sampling_rate
}

url = "http://localhost:8000/preprocess"

try:
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    result = response.json()
    print("\n✅ Cloud API responded successfully")
    print("Features (Lyapunov exponent, DFA alpha1):", result['features'])
    print("Filtered signal length:", len(result['filtered_signal']))
    
    # Decode and save the spectrogram image
    img_data = base64.b64decode(result['image_base64'])
    img = Image.open(io.BytesIO(img_data))
    output_img_path = "spectrogram_output.png"
    img.save(output_img_path)
    print(f"Spectrogram saved as {output_img_path}")
    
except requests.exceptions.RequestException as e:
    print("❌ Error calling cloud API:", e)
    if e.response is not None:
        print("Response text:", e.response.text)