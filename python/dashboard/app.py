"""VisionGuard Enterprise Streamlit Dashboard (Python 3.11 Companion UI)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="VisionGuard — Industrial Anomaly Detection Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

RESULTS_DIR = Path("experiments/results")
MODELS_DIR = Path("models")

st.sidebar.title("VisionGuard")
page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "Live Inspection",
        "Model Laboratory",
        "Dataset Explorer",
        "Research Experiments",
        "Inspection History",
        "Methodology & About",
    ],
)


def load_verified_results() -> pd.DataFrame:
    if not RESULTS_DIR.exists():
        return pd.DataFrame()
    rows = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
            m = rec.get("metrics", {})
            rows.append(
                {
                    "Dataset": rec.get("dataset"),
                    "Category": rec.get("category"),
                    "Model": rec.get("model"),
                    "Image AUROC": m.get("image_auroc"),
                    "Pixel AUROC": m.get("pixel_auroc"),
                    "Image F1": m.get("image_f1"),
                    "Precision": m.get("precision"),
                    "Recall": m.get("recall"),
                    "Latency (ms)": rec.get("mean_latency_ms"),
                    "Checkpoint": rec.get("checkpoint_path"),
                }
            )
        except Exception:
            continue
    return pd.DataFrame(rows)


df = load_verified_results()

if page == "Executive Overview":
    st.title("VisionGuard — Executive Overview")
    st.caption(
        "Intelligent Industrial Defect Detection, Anomaly Localization, and Visual Inspection Platform"
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Verified Experiment Records", len(df))
    c2.metric(
        "Best Image AUROC",
        f"{df['Image AUROC'].max() * 100:.1f}%" if not df.empty and df["Image AUROC"].notna().any() else "Not evaluated",
    )
    ckpts = list(MODELS_DIR.rglob("*.pt")) + list(MODELS_DIR.rglob("*.npz")) if MODELS_DIR.exists() else []
    c3.metric("Saved Checkpoints", len(ckpts))
    if df.empty:
        st.info("Not evaluated yet. Run `python -m src.training.train --model patchcore --dataset mvtec_ad --category bottle`.")
    else:
        st.dataframe(df, use_container_width=True)

elif page == "Live Inspection":
    st.title("Live Visual Inspection")
    uploaded = st.file_uploader("Upload industrial product image (PNG / JPG / JPEG)", type=["png", "jpg", "jpeg"])
    ckpts = (
        sorted(list(MODELS_DIR.rglob("*.pt")) + list(MODELS_DIR.rglob("*.npz")))
        if MODELS_DIR.exists()
        else []
    )
    if not ckpts:
        st.warning("No trained checkpoints found under `models/`. Train a model first — VisionGuard never simulates inference.")
    elif uploaded is not None:
        st.image(uploaded, caption=f"Uploaded: {uploaded.name}", width=320)
        st.info("Select a saved checkpoint and invoke `ConvAutoencoderModel.load / PaDiMModel.load / PatchCoreModel.load` for real inference.")

elif page == "Model Laboratory":
    st.title("Model Laboratory")
    if df.empty:
        st.info("Not evaluated yet.")
    else:
        st.dataframe(df, use_container_width=True)

else:
    st.title(page)
    st.markdown(
        "Refer to `docs/ARCHITECTURE.md`, `docs/METHODOLOGY.md`, `docs/DATASETS.md`, and `docs/COMPETITION_DEMO.md` "
        "or use the interactive React/TypeScript VisionGuard Web Platform running on port 3000."
    )
