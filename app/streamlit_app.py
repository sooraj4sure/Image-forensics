"""
Streamlit UI (brief §11): upload -> preview -> analyze -> forensic report
(class, confidence, per-branch scores, metadata observations, Grad-CAM
heatmap, uncertainty/limitations notice). Calls the same ForensicsPipeline
that api/main.py uses, so predictions are identical between the two
serving surfaces by construction.

Run as: ./.venv/bin/streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# streamlit run app/streamlit_app.py (same as `python app/streamlit_app.py`)
# only puts THIS script's directory on sys.path, not the repo root — so
# `from src... import ...` below fails with ModuleNotFoundError unless we
# add the repo root explicitly. Confirmed by actually running this both
# via `python app/streamlit_app.py` and `streamlit run ...` during Stage 8
# deployment prep — the bare script execution failed with exactly that
# error before this fix was added.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.config import load_config
from src.inference.pipeline import ForensicsPipeline

st.set_page_config(page_title="ImageForensics AI", page_icon="🔍", layout="centered")


@st.cache_resource
def get_pipeline():
    """Load the model once per Streamlit server process, not per request."""
    cfg = load_config()
    return ForensicsPipeline(config_name="full_fusion", cfg=cfg)


def validate_upload(uploaded_file, cfg) -> Image.Image | None:
    contents = uploaded_file.getvalue()
    max_bytes = cfg.serving.max_upload_mb * 1024 * 1024
    if len(contents) > max_bytes:
        st.error(f"File too large: {len(contents)} bytes exceeds the {cfg.serving.max_upload_mb}MB limit.")
        return None

    ext = "." + uploaded_file.name.rsplit(".", 1)[-1].lower() if "." in uploaded_file.name else ""
    if ext not in cfg.serving.allowed_extensions:
        st.error(f"Unsupported file type {ext!r}. Allowed: {cfg.serving.allowed_extensions}")
        return None

    try:
        return Image.open(uploaded_file).convert("RGB")
    except UnidentifiedImageError:
        st.error("Uploaded file is not a valid/readable image.")
        return None


def main():
    st.title("🔍 ImageForensics AI")
    st.caption(
        "Classifies an image as REAL or AI_GENERATED with interpretable forensic "
        "evidence. Built with open-source CV/ML only — no paid APIs."
    )

    cfg = load_config()

    try:
        pipeline = get_pipeline()
    except FileNotFoundError as e:
        st.error(
            "No trained model checkpoint found. Train the full_fusion config first "
            "with `src.training.train_ablation`, then restart this app."
        )
        st.code(str(e))
        return

    uploaded_file = st.file_uploader(
        "Upload an image", type=[ext.lstrip(".") for ext in cfg.serving.allowed_extensions]
    )

    if uploaded_file is None:
        st.info("Upload an image above to get started.")
        return

    pil_image = validate_upload(uploaded_file, cfg)
    if pil_image is None:
        return

    st.image(pil_image, caption="Uploaded image", use_container_width=True)

    if st.button("Analyze", type="primary"):
        with st.spinner("Running forensic analysis..."):
            report, gradcam_overlay = pipeline.analyze(pil_image, include_gradcam=True)

        st.divider()
        st.subheader("Forensic Report")

        col1, col2 = st.columns(2)
        with col1:
            emoji = "🤖" if report.predicted_class == "AI_GENERATED" else "📷"
            st.metric("Prediction", f"{emoji} {report.predicted_class}")
        with col2:
            st.metric("Calibrated Confidence", f"{report.calibrated_confidence:.1%}")

        if not pipeline.temperature_fitted:
            st.warning(
                "⚠️ No fitted calibration found for this model — confidence shown is "
                "RAW (uncalibrated) softmax, which is commonly overconfident. Run "
                "`src.evaluation.fit_calibration` to fix this."
            )

        st.info(report.disclaimer)

        if gradcam_overlay is not None:
            st.subheader("Grad-CAM Explanation")
            st.image(gradcam_overlay, caption=report.gradcam_caption, use_container_width=True)

        st.subheader("Per-Branch Signal (diagnostic)")
        st.caption(
            "Relative feature-magnitude per active branch — NOT a calibrated "
            "sub-probability, just a rough diagnostic of which branches' "
            "representations were most active for this image."
        )
        st.bar_chart(report.branch_signal_norms)

        st.subheader("Metadata Observations")
        meta = report.metadata
        if meta["has_exif"]:
            st.write(f"**Camera:** {meta['camera_make'] or '—'} {meta['camera_model'] or ''}")
            st.write(f"**Software:** {meta['software'] or '—'}")
            st.write(f"**Date taken:** {meta['datetime_original'] or '—'}")
        else:
            st.write("No EXIF metadata found in this image.")
        st.caption(meta["note"])

        with st.expander("Raw report JSON"):
            st.json(report.to_dict())


if __name__ == "__main__":
    main()
