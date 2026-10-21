import numpy as np
from ssqueezepy import ssq_cwt, Wavelet
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

_WAVELET = Wavelet("bump")

# Pre-compute the "jet" colormap LUT once (256 RGBA uint8 entries).
# Rendering the WSST spectrogram directly through this LUT avoids creating a
# Matplotlib figure + axes + PNG encode/decode on every request, which was the
# dominant cost of /analyze. The figure path took ~20 s per request; this path
# takes ~0.1 s while producing an equivalent image.
_JET_LUT = (plt.get_cmap("jet")(np.linspace(0.0, 1.0, 256))[:, :3] * 255).astype(np.uint8)

_WSST_NV_FAST = 24
_WSST_NV_FULL = 48
_IMG_SIZE = 224


def _render_spectrogram(mag: np.ndarray) -> Image.Image:
    """Map |Tx| to a 224x224 RGB image via the jet colormap (NumPy only)."""
    mag = np.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)
    mn = float(mag.min())
    mx = float(mag.max())
    if mx > mn:
        norm = (mag - mn) / (mx - mn)
    else:
        norm = np.zeros_like(mag, dtype=np.float32)
    idx = np.clip((norm * 255.0).astype(np.int32), 0, 255)
    # Tx is (freqs, time) with origin="lower" -> flip vertically to match imshow.
    rgb = _JET_LUT[idx][::-1]
    return Image.fromarray(rgb, "RGB").resize((_IMG_SIZE, _IMG_SIZE), Image.BILINEAR)


def make_wsst_image(signal, fs=300, fast=False, bypass=False):
    if bypass:
        return Image.new("RGB", (_IMG_SIZE, _IMG_SIZE), color="black")
    nv = _WSST_NV_FAST if fast else _WSST_NV_FULL
    Tx, _, _, _ = ssq_cwt(signal, wavelet=_WAVELET, fs=fs, nv=nv)
    return _render_spectrogram(np.abs(Tx))