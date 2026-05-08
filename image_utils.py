import numpy as np
from ssqueezepy import ssq_cwt, Wavelet
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import io

_WAVELET = Wavelet("bump")

def make_wsst_image(signal, fs=300, fast=False, bypass=False):
    if bypass:
        return Image.new("RGB", (224, 224), color="black")
    nv = 24 if fast else 48
    Tx, _, _, _ = ssq_cwt(signal, wavelet=_WAVELET, fs=fs, nv=nv)
    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    ax = plt.axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(np.abs(Tx), aspect="auto", origin="lower", cmap="jet", interpolation="bilinear")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")