"""Streamlit front end for InspectAI."""
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError
import streamlit as st

from src.inference import PatchCoreInspector

ROOT = Path(__file__).resolve().parent
BANK_PATH = ROOT / "models" / "patchcore_final_bank.pt"

st.set_page_config(page_title="InspectAI", page_icon="🔎", layout="wide")
st.title("InspectAI · Metal Nut Inspection")
st.caption("PatchCore-based industrial anomaly detection and localization")
st.write("Upload a photograph of a metal nut to inspect visual deviations from the normal training images.")

@st.cache_resource(show_spinner="Loading PatchCore model…")
def load_inspector(bank_location: str):
    return PatchCoreInspector(bank_location)

if not BANK_PATH.is_file():
    st.error("Final model bank not found at `models/patchcore_final_bank.pt`. Copy it from your project models folder.")
    st.stop()

upload = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
if upload is not None:
    try:
        source = Image.open(upload).convert("RGB")
        source.load()
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("Unable to read that image. Please upload a valid PNG or JPEG.")
        st.stop()

    try:
        inspector = load_inspector(str(BANK_PATH))
        begin = time.perf_counter()
        score, anomaly_map = inspector.predict(source)
        elapsed_ms = 1000 * (time.perf_counter() - begin)
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
        st.stop()

    a, b = st.columns(2)
    a.metric("Raw anomaly score", f"{score:.3f}")
    b.metric("Inference time", f"{elapsed_ms:.0f} ms")
    st.caption("Higher scores mean a greater departure from normal. This score is not a probability and no validated normal/defective threshold has been set.")

    # Display normalization is per-image and affects COLORS ONLY, never the raw score.
    low, high = np.percentile(anomaly_map, [5, 99])
    # Fixed visualization scale shared across images
    HEATMAP_MAX = 4.0  # Temporary display limit
    display_map = np.clip(anomaly_map / HEATMAP_MAX, 0, 1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7))
    axes[0].imshow(source)
    axes[0].set_title("Original")
    axes[1].imshow(display_map, cmap="inferno", vmin=0, vmax=1)
    axes[1].set_title("Anomaly heatmap")
    axes[2].imshow(source)
    axes[2].imshow(display_map, cmap="inferno", vmin=0, vmax=1, alpha=0.42)
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    st.info("Heatmap intensity is normalized separately for visualization. Red/bright areas are not necessarily verified defects. The model was evaluated on MVTec AD metal_nut images, not arbitrary industrial photos.")
