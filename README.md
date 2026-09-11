# ImageForensics AI

> Detects whether an uploaded image is **REAL** or **AI_GENERATED**, with interpretable
> forensic evidence (Grad-CAM, per-branch signal breakdown, metadata observations) —
> not just a label. Built with open-source CV/ML only, no paid APIs.

**Status:** 🚧 In progress — Stage 1 (environment setup) complete. See roadmap below.

---

## Table of Contents
*(filled in as sections below are written)*

- [ ] Problem & Motivation
- [ ] Architecture
- [ ] Dataset (source, license, splits)
- [ ] Preprocessing
- [ ] Model Architecture
- [ ] Training Procedure
- [ ] Ablation Study (4-config results table)
- [ ] Unseen-Generator Generalization Results
- [ ] Robustness Testing Results
- [ ] Calibration (ECE, reliability diagram)
- [ ] Grad-CAM Examples (incl. honest failure cases)
- [ ] Installation & Local Run Instructions
- [ ] API Usage
- [ ] Live Demo Link
- [ ] Limitations & Ethical Considerations
- [ ] Future Work

---

## Quickstart (current state — Stage 1)

```bash
git clone <repo-url>
cd image-forensics-ai
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt

# Verify config loads correctly
./.venv/bin/python -m src.config

# Run tests
./.venv/bin/python -m pytest tests/ -v
```

## Project Structure

```
image-forensics-ai/
├── configs/           # config.yaml — single source of truth for paths/hyperparams
├── data/               # raw/processed data (gitignored — see data/README.md, added in Stage 2)
├── src/
│   ├── config.py       # typed config loader (pydantic)
│   ├── utils.py         # shared utilities (seeding, etc.)
│   ├── data/            # dataset loading, splitting, metadata
│   ├── models/           # backbone + fusion architecture
│   ├── features/          # FFT/DCT + noise-residual feature extraction
│   ├── training/           # training loops, ablation runner
│   ├── evaluation/          # metrics, calibration, robustness tests
│   ├── explainability/       # Grad-CAM
│   └── inference/             # inference pipeline (shared by API + app)
├── api/                # FastAPI serving layer
├── app/                # Streamlit UI
├── models/             # trained model weights (gitignored)
├── tests/              # pytest suite
└── screenshots/        # UI/result screenshots for this README
```

## Roadmap

| Weeks | Focus |
|---|---|
| 1–2 | Env setup → dataset selection/licensing → cleaning/splits → ResNet-18 baseline → full evaluation |
| 3–4 | FFT/DCT branch → noise/residual branch → 4-way fusion ablation → unseen-generator eval → trimmed robustness |
| 5–6 | Grad-CAM → inference pipeline → FastAPI → Streamlit |
| 7–8 | Tests → HF Spaces deployment → final README |

## Non-Negotiable Constraints

- No paid LLM/AI API anywhere in the core detection system.
- No fabricated datasets, benchmarks, or metrics — every number here comes from an actual run.
- $0 recurring cost; final demo is a free, publicly deployed link (Hugging Face Spaces, CPU tier).

## License

*(to be finalized — likely MIT for code; dataset license documented separately once selected in Stage 2)*
