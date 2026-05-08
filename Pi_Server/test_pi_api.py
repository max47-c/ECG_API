import scipy.io as sio
import requests
import numpy as np

# Load a short segment of the .mat file
mat = sio.loadmat(r"E:\ECG_API\Pi_Server\test_data\A00004.mat")
signal = mat['val'].flatten().tolist()[:9000]  # 9k points (10 sec at 900 Hz? but whatever)

payload = {
    "values": signal,
    "sampling_rate": 1000,
    "threshold": 0.5
}
url = "http://localhost:8081/analyze"
resp = requests.post(url, json=payload)
print(resp.status_code)
if resp.status_code == 200:
    print(resp.json())
else:
    print(resp.text)