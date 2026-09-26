
"""
Streamlit UI (brief §11): upload -> preview -> analyze -> forensic report
(class, confidence, per-branch scores, metadata observations, Grad-CAM
heatmap, uncertainty/limitations notice). Calls the same ForensicsPipeline
that api/main.py uses, so predictions are identical between the two
serving surfaces by construction.

Visual theme: dark "forensic scanner" aesthetic — glass panels, neon
cyan/magenta accents, monospace HUD-style labels. Pure CSS/HTML on top
of native Streamlit widgets; no functional/pipeline logic changed.

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

st.set_page_config(page_title="Image Forensics AI", page_icon="🛰️", layout="centered")

# --------------------------------------------------------------------------
# THEME — dark glass / neon scanner look. Pure CSS injection; does not touch
# any pipeline or data logic below.
# --------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Space+Mono:wght@400;700&display=swap');

:root {
    --bg-void: #05070d;
    --bg-panel: rgba(18, 24, 38, 0.65);
    --border-glow: rgba(0, 229, 255, 0.35);
    --cyan: #00e5ff;
    --magenta: #ff2fd0;
    --green: #39ff8a;
    --amber: #ffb020;
    --text-main: #e8f1ff;
    --text-dim: #7d8aa3;
}

html, body, [class*="css"] {
    font-family: 'Space Mono', monospace;
}

.stApp {
    background:
        radial-gradient(circle at 15% 10%, rgba(0, 229, 255, 0.07), transparent 40%),
        radial-gradient(circle at 85% 90%, rgba(255, 47, 208, 0.06), transparent 45%),
        var(--bg-void);
    color: var(--text-main);
}

/* Title block */
.hud-title {
    font-family: 'Orbitron', sans-serif;
    font-weight: 900;
    font-size: 2.1rem;
    letter-spacing: 0.06em;
    background: linear-gradient(90deg, var(--cyan), var(--magenta));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
    text-shadow: 0 0 30px rgba(0, 229, 255, 0.25);
}
.hud-subtitle {
    color: var(--text-dim);
    font-size: 0.85rem;
    letter-spacing: 0.03em;
    margin-top: 4px;
    border-left: 2px solid var(--cyan);
    padding-left: 10px;
}

/* Section headers styled like scanner readouts */
.hud-section {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.95rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--cyan);
    margin: 1.6rem 0 0.6rem 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.hud-section::before {
    content: "";
    width: 8px;
    height: 8px;
    background: var(--cyan);
    box-shadow: 0 0 8px var(--cyan);
    border-radius: 50%;
    display: inline-block;
}

/* Glass panel wrapper */
.glass-panel {
    background: var(--bg-panel);
    border: 1px solid var(--border-glow);
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    backdrop-filter: blur(10px);
    box-shadow: 0 0 25px rgba(0, 229, 255, 0.05), inset 0 0 40px rgba(0, 229, 255, 0.02);
    margin-bottom: 1rem;
}

/* Verdict card */
.verdict-card {
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    border: 1px solid;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.verdict-ai {
    border-color: rgba(255, 47, 208, 0.5);
    background: linear-gradient(135deg, rgba(255, 47, 208, 0.12), rgba(18, 24, 38, 0.6));
    box-shadow: 0 0 35px rgba(255, 47, 208, 0.15);
}
.verdict-real {
    border-color: rgba(57, 255, 138, 0.5);
    background: linear-gradient(135deg, rgba(57, 255, 138, 0.10), rgba(18, 24, 38, 0.6));
    box-shadow: 0 0 35px rgba(57, 255, 138, 0.15);
}
.verdict-label {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: 0.08em;
}
.verdict-sub {
    color: var(--text-dim);
    font-size: 0.78rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.confidence-readout {
    font-family: 'Orbitron', sans-serif;
    font-size: 2.2rem;
    font-weight: 900;
    color: var(--cyan);
    text-shadow: 0 0 20px rgba(0, 229, 255, 0.4);
}

/* Scanline divider */
.scan-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--cyan), transparent);
    margin: 1.4rem 0;
    opacity: 0.5;
}

/* Buttons */
.stButton > button {
    font-family: 'Orbitron', sans-serif;
    letter-spacing: 0.1em;
    background: linear-gradient(90deg, rgba(0,229,255,0.15), rgba(255,47,208,0.15));
    border: 1px solid var(--cyan);
    color: var(--text-main);
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    box-shadow: 0 0 20px rgba(0, 229, 255, 0.5);
    border-color: var(--magenta);
    transform: translateY(-1px);
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 1px dashed var(--border-glow);
    border-radius: 12px;
    padding: 0.5rem;
    background: rgba(0, 229, 255, 0.02);
}

/* Metric-like tags */
.tag-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 6px; }
.hud-tag {
    font-size: 0.72rem;
    letter-spacing: 0.05em;
    color: var(--text-dim);
    border: 1px solid rgba(125, 138, 163, 0.3);
    border-radius: 20px;
    padding: 2px 10px;
}

.mono-note { color: var(--text-dim); font-size: 0.82rem; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


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
    st.markdown('<div class="hud-title">🛰️ IMAGE FORENSICS AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hud-subtitle">NEURAL FORENSIC SCANNER  REAL vs AI_GENERATED '
        '· open source CV/ML only · no paid APIs</div>',
        unsafe_allow_html=True,
    )
    st.write("")

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

    st.markdown('<div class="hud-section">UPLOAD TARGET</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=[ext.lstrip(".") for ext in cfg.serving.allowed_extensions],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.markdown(
            '<div class="glass-panel mono-note">⌁ Awaiting image input — drop a file above '
            'to begin forensic analysis.</div>',
            unsafe_allow_html=True,
        )
        return

    pil_image = validate_upload(uploaded_file, cfg)
    if pil_image is None:
        return

    st.markdown('<div class="hud-section">TARGET PREVIEW</div>', unsafe_allow_html=True)
    st.image(pil_image, use_container_width=True)

    st.write("")
    analyze_clicked = st.button("▶ RUN FORENSIC ANALYSIS", type="primary", use_container_width=True)

    if analyze_clicked:
        with st.spinner("⌁ Scanning pixel structure, frequency domain, and noise residuals..."):
            report, gradcam_overlay = pipeline.analyze(pil_image, include_gradcam=True)

        st.markdown('<div class="scan-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="hud-section">FORENSIC REPORT</div>', unsafe_allow_html=True)

        is_ai = report.predicted_class == "AI_GENERATED"
        verdict_class = "verdict-ai" if is_ai else "verdict-real"
        emoji = "🤖" if is_ai else "📷"

        st.markdown(
            f"""
            <div class="verdict-card {verdict_class}">
                <div class="verdict-sub">Prediction</div>
                <div class="verdict-label">{emoji} {report.predicted_class}</div>
                <div class="scan-divider" style="margin:0.9rem 0;"></div>
                <div class="verdict-sub">Calibrated Confidence</div>
                <div class="confidence-readout">{report.calibrated_confidence:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not pipeline.temperature_fitted:
            st.warning(
                "⚠️ No fitted calibration found for this model — confidence shown is "
                "RAW (uncalibrated) softmax, which is commonly overconfident. Run "
                "`src.evaluation.fit_calibration` to fix this."
            )

        st.markdown(
            f'<div class="glass-panel mono-note">ℹ {report.disclaimer}</div>',
            unsafe_allow_html=True,
        )

        if gradcam_overlay is not None:
            st.markdown('<div class="hud-section">GRAD-CAM EXPLANATION</div>', unsafe_allow_html=True)
            st.image(gradcam_overlay, caption=report.gradcam_caption, use_container_width=True)

        st.markdown('<div class="hud-section">PER-BRANCH SIGNAL (DIAGNOSTIC)</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="mono-note">Relative feature-magnitude per active branch — NOT a calibrated '
            'sub-probability, just a rough diagnostic of which branches\' representations were most '
            'active for this image.</div>',
            unsafe_allow_html=True,
        )
        st.bar_chart(report.branch_signal_norms)

        st.markdown('<div class="hud-section">METADATA OBSERVATIONS</div>', unsafe_allow_html=True)
        meta = report.metadata
        if meta["has_exif"]:
            st.markdown(
                f"""
                <div class="glass-panel">
                    <div class="tag-row">
                        <span class="hud-tag">CAMERA: {meta['camera_make'] or '—'} {meta['camera_model'] or ''}</span>
                        <span class="hud-tag">SOFTWARE: {meta['software'] or '—'}</span>
                        <span class="hud-tag">DATE: {meta['datetime_original'] or '—'}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="glass-panel mono-note">No EXIF metadata found in this image.</div>',
                unsafe_allow_html=True,
            )
        st.caption(meta["note"])

        with st.expander("⌁ Raw report JSON"):
            st.json(report.to_dict())


if __name__ == "__main__":
    main()