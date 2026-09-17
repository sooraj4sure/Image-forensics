---
title: ImageForensics AI
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.63.0"
app_file: app/streamlit_app.py
pinned: false
license: mit
---

# ImageForensics AI

> Detects whether an uploaded image is **REAL** or **AI_GENERATED**, with interpretable
> forensic evidence (Grad-CAM, per-branch signal breakdown, metadata observations) —
> not just a label. Built with open-source CV/ML only, no paid APIs.

**Status:** 🚧 Engineering complete (Stages 1-8), **blocked on real training data**. Every
line of the pipeline — dataset handling, baseline model, 4-config ablation, unseen-generator
eval, robustness testing, calibration, Grad-CAM, FastAPI/Streamlit serving, HF Spaces
deployment scaffolding — is built, tested (104 tests, 96%+ coverage), and verified
end-to-end on synthetic placeholder data. Sections below marked **PENDING** need a real
dataset trained through the pipeline before they can be filled with real numbers — per
this project's own ground rules, no metric is reported here without an actual run behind it.

---

## Table of Contents

- [Problem & Motivation](#problem--motivation)
- [Architecture](#architecture)
- [Dataset](#dataset) — PENDING real data
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training Procedure](#training-procedure)
- [Ablation Study](#ablation-study) — PENDING real data
- [Unseen-Generator Generalization](#unseen-generator-generalization) — PENDING real data
- [Robustness Testing](#robustness-testing) — PENDING real data
- [Calibration](#calibration) — PENDING real data
- [Grad-CAM Examples](#grad-cam-examples) — PENDING real data
- [Installation & Local Run](#installation--local-run)
- [API Usage](#api-usage)
- [Live Demo](#live-demo) — PENDING deployment
- [Limitations & Ethical Considerations](#limitations--ethical-considerations)
- [Future Work](#future-work)

---

## Problem & Motivation

Generative image models have become good enough that unaided visual inspection is no
longer a reliable way to tell a real photograph from a synthetic one. That matters
beyond curiosity — misattributed images (real ones dismissed as "AI slop", or synthetic
ones passed off as documentary evidence) have real consequences in journalism, content
moderation, and everyday trust in what a photo shows.

This project builds a binary REAL vs. AI_GENERATED classifier that tries to earn trust
the way a forensic tool should: not with a bare label, but with the *evidence* behind
it — which visual/frequency/noise signals drove the prediction (Grad-CAM), how
confident the model actually is once that confidence is calibrated against real
outcomes (not raw, likely-overconfident softmax), and an honest account of where it
fails (a held-out generator the model never trained on, and degraded/compressed
images it might see in the wild).

It is explicitly **not** claiming to be definitive proof of an image's origin — see
[Limitations & Ethical Considerations](#limitations--ethical-considerations).

## Architecture

```
Upload
  │
  ▼
Validation (file type, size)
  │
  ▼
Preprocessing (resize, normalize)
  │
  ├──────────────┬──────────────────┬───────────────────┐
  ▼              ▼                  ▼                    │
RGB/Spatial   Frequency-Domain   Noise/Residual      Metadata/EXIF
branch        branch (FFT)       branch (engineered   (shown separately —
(ResNet-18)   (small CNN)        features + MLP)       never fed to the model,
  │              │                  │                   never treated as
  └──────────────┴──────────────────┘                   AI-generation evidence)
         │
         ▼
   Feature Fusion (concat)
         │
         ▼
      Classifier
         │
         ▼
 Calibrated Probability (temperature scaling)
         │
         ├─────────────────────┐
         ▼                     ▼
  Grad-CAM Explanation    Forensic Report
  (RGB branch only)       (class, confidence, per-branch
                            signal, metadata, disclaimer)
```

Four ablation configs are built from the same three branches by toggling which are
active (`rgb_only`, `rgb_fft`, `rgb_residual`, `full_fusion`) — see
[`src/models/fusion.py`](src/models/fusion.py) — so the eventual comparison is
apples-to-apples rather than four independently-tuned models.

## Dataset

**PENDING.** See [`data/README.md`](data/README.md) for the two candidates under
consideration (GenImage, leaning towards, vs. CIFAKE as fallback) and why — this section
gets filled in with the actual chosen dataset, its verified license terms, class
balance, and generator breakdown once real data is downloaded and run through
[`src/data/build_metadata.py`](src/data/build_metadata.py).

## Preprocessing

- Images resized to 224×224 (ResNet-18's standard input size), ImageNet-mean/std
  normalized.
- Light augmentation on the RGB branch only (horizontal flip, mild color jitter) —
  deliberately avoids blur or heavy JPEG re-compression during training, since those
  would erase the very high-frequency artifacts the frequency and residual branches
  are trying to detect. That degradation is reserved for the dedicated
  [robustness evaluation](#robustness-testing), where testing *against* degraded
  input is the point.
- The frequency (FFT log-magnitude) and residual (engineered noise/edge/texture
  features) branches are computed from an unaugmented resize of the same image, so
  RGB-branch augmentation never leaks into what those branches see.
- See [`src/data/dataset.py`](src/data/dataset.py) and
  [`src/data/multibranch_dataset.py`](src/data/multibranch_dataset.py).

## Model Architecture

- **Backbone:** ResNet-18, ImageNet-pretrained, fine-tuned — deliberately not a larger
  model, since the point of this project is the forensic pipeline (fusion, calibration,
  explainability), not backbone size, and it needs to run acceptably on free CPU-tier
  hosting.
- **Frequency branch:** FFT log-magnitude spectrum of the grayscale image, fed through
  a small CNN (2 conv layers). DCT is implemented as an alternative representation to
  compare against if FFT underperforms (see
  [`src/features/frequency.py`](src/features/frequency.py)).
- **Residual branch:** 14 hand-justified engineered features from the high-frequency
  residual (original − Gaussian low-pass) — per-channel noise mean/std, high-frequency
  energy, local variance, edge density, cross-channel residual correlation, Laplacian
  variance — each with a one-line rationale in
  [`src/features/residual.py`](src/features/residual.py), fed through a small MLP.
- **Fusion:** feature vectors from active branches are concatenated and passed through
  a dropout + linear classifier head. See
  [`src/models/fusion.py`](src/models/fusion.py).

## Training Procedure

AdamW optimizer, cosine annealing LR schedule, early stopping on validation F1,
mixed precision when a GPU is available. Class-stratified train/val/test splits, with
one entire generator family held out of training and validation for the
[unseen-generator evaluation](#unseen-generator-generalization). Full config in
[`configs/config.yaml`](configs/config.yaml); training loop in
[`src/training/train.py`](src/training/train.py) (baseline) and
[`src/training/train_ablation.py`](src/training/train_ablation.py) (all 4 configs,
identical protocol).

## Ablation Study

**PENDING real data.** Will report accuracy/precision/recall/F1/ROC-AUC for all 4
configs (`rgb_only`, `rgb_fft`, `rgb_residual`, `full_fusion`) under an identical eval
protocol, answering: do the frequency and residual branches actually add signal over
RGB alone, or not? (Brief's own framing: measure it, don't assume it.)

## Unseen-Generator Generalization

**PENDING real data.** This is the project's strongest planned claim: train excluding
one generator family entirely, then report the AI-detection rate on that held-out
generator vs. the in-distribution AI-detection rate on generators the model *did*
train on. Note: because the held-out-generator test set is, by construction, AI-only
(single-class), the comparison metric is detection rate/recall specifically, not
accuracy/F1/ROC-AUC, which need both classes present — see
[`src/evaluation/unseen_generator.py`](src/evaluation/unseen_generator.py) for why.

## Robustness Testing

**PENDING real data.** Will report accuracy/F1 delta on the final fusion model under
JPEG compression (3 quality levels), downscale/upscale, center crop, Gaussian blur, and
added noise — see [`src/evaluation/robustness.py`](src/evaluation/robustness.py).
Screenshot-transform robustness and adversarial perturbation are explicitly out of
scope for v1 (see [Future Work](#future-work)).

## Calibration

**PENDING real data.** Will report Expected Calibration Error (ECE) before/after
temperature scaling, fit on the validation set, plus a reliability diagram — see
[`src/evaluation/calibration.py`](src/evaluation/calibration.py) and
[`src/evaluation/fit_calibration.py`](src/evaluation/fit_calibration.py).

## Grad-CAM Examples

**PENDING real data.** Will include at least one honest failure case (a real false
positive and a real false negative) alongside successful predictions, per this
project's own rule against showing only wins. Grad-CAM is computed over the RGB
branch only — see [`src/explainability/gradcam.py`](src/explainability/gradcam.py).
UI/API copy is explicit that the heatmap shows *regions that influenced the
prediction*, not regions proven to be AI-generated.

## Installation & Local Run

```bash
git clone <repo-url>
cd image-forensics-ai
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt              # runtime deps
pip install -r requirements-dev.txt           # + test deps (optional, local dev only)

# Verify config loads correctly
./.venv/bin/python -m src.config

# Run tests
./.venv/bin/python -m pytest tests/ -v

# Run tests with coverage
./.venv/bin/python -m pytest --cov=src --cov=api --cov-report=term-missing
```

Once real data is in `data/raw/` (see [`data/README.md`](data/README.md) for dataset
options and the expected layout):

```bash
./.venv/bin/python -m src.data.build_metadata                              # builds data/metadata/metadata.csv
./.venv/bin/python -m src.training.train                                   # trains the RGB-only baseline
./.venv/bin/python -m src.training.train_ablation                          # trains all 4 ablation configs
./.venv/bin/python -m src.evaluation.unseen_generator --config full_fusion # held-out-generator eval
./.venv/bin/python -m src.evaluation.run_robustness_suite                  # JPEG/resize/crop/blur/noise stress test
./.venv/bin/python -m src.evaluation.fit_calibration --config full_fusion  # fits temperature scaling on val set

# Serving (needs a trained full_fusion checkpoint + fitted temperature.json):
./.venv/bin/uvicorn api.main:app --reload           # FastAPI at http://127.0.0.1:8000 (docs at /docs)
./.venv/bin/streamlit run app/streamlit_app.py      # Streamlit UI
```

> **Note:** pretrained ImageNet weights are fetched from `download.pytorch.org` at
> first run — this needs to be reachable in your environment (it is on Colab/Kaggle;
> it was NOT reachable in the sandboxed dev environment this repo was scaffolded in,
> so all code paths were verified there with `pretrained: false` instead — see commit
> history for the specifics of what was and wasn't possible to verify directly).

For actual training on real dataset sizes, CPU training (as in the sandbox above) will
be very slow — use Colab/Kaggle's free GPU tier per this project's own compute
constraint.

See [`DEPLOYMENT.md`](DEPLOYMENT.md) for the full Hugging Face Spaces deployment flow
once you have a trained model you're happy with.

## API Usage

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@your_image.jpg"

curl -X POST http://127.0.0.1:8000/analyze \
  -F "file=@your_image.jpg"
# includes a base64-encoded Grad-CAM overlay PNG in gradcam_overlay_base64
```

`/predict` is a fast path (prediction + calibrated confidence, no Grad-CAM).
`/analyze` returns the full forensic report. Both responses include a `disclaimer`
field stating the prediction is a statistical estimate, not definitive proof of
origin — see [`api/main.py`](api/main.py) and
[`src/inference/pipeline.py`](src/inference/pipeline.py) (the single pipeline both
the API and the Streamlit app call, so they can't drift apart).

## Live Demo

**PENDING deployment.** Will be a Hugging Face Spaces link (Streamlit, CPU tier) —
see [`DEPLOYMENT.md`](DEPLOYMENT.md) for the deployment runbook and the free-tier
limitations (cold starts, CPU-only inference, memory/storage ceilings) that apply to
it.

## Limitations & Ethical Considerations

- **Not proof of origin.** This model produces a statistical estimate, not a
  definitive determination. Both false positives (real images flagged as AI) and
  false negatives (AI images flagged as real) are possible and, once real numbers
  exist, will be reported honestly here — including at least one real example of each.
- **Consequences of errors are asymmetric depending on context.** A false positive
  (wrongly flagging a real photo as AI-generated) could unfairly discredit genuine
  documentary evidence, harm a photographer's credibility, or fuel "it's fake"
  dismissal of real events. A false negative (missing an AI-generated image) could let
  synthetic content pass as authentic in contexts where that matters — misinformation,
  fraud, impersonation. Neither error type should be treated as the "safe" one to
  bias toward without knowing the deployment context.
- **Generator coverage is inherently incomplete.** The model is trained on a handful
  of generator families that existed at training time; new generators appear
  constantly, and the [unseen-generator evaluation](#unseen-generator-generalization)
  exists specifically to measure (not hide) how much accuracy drops against a
  generator the model never saw — but that drop will not be zero, and there is no
  guarantee it generalizes to *future* generators not yet in existence at all.
- **Metadata is supporting context only.** Missing or stripped EXIF data is extremely
  common for real photos (messaging apps, social media, screenshots strip it
  routinely) and is never treated as evidence of AI generation anywhere in this
  codebase — see [`src/inference/exif.py`](src/inference/exif.py).
- **Robustness has bounds.** The [robustness suite](#robustness-testing) tests a
  fixed, capped set of common degradations (compression, resize, crop, blur, noise).
  Adversarially-crafted inputs designed specifically to fool this model are explicitly
  out of scope for v1 (see [Future Work](#future-work)) and this model should not be
  assumed robust against a deliberate adversary.
- **No claim of state-of-the-art performance** is made anywhere in this project unless
  actually benchmarked against a real comparison.

## Future Work

Explicitly cut from v1 scope (see the original project brief §3) to protect time for
the core differentiator experiments:

- **AI_EDITED / MANIPULATED / UNKNOWN classes** — v1 is binary REAL vs. AI_GENERATED
  only. The architecture (separate branches feeding a fusion classifier) is meant to
  extend to additional classes later without a redesign, but those classes are not
  built now.
- **Screenshot-transform robustness and adversarial-example testing** — high effort,
  lower resume/learning signal at this project's scope; the [robustness
  suite](#robustness-testing) covers common non-adversarial degradations only.
- **Multi-source dataset assembly** — v1 deliberately uses one well-documented
  existing benchmark dataset rather than merging many disparate raw sources.
- **React frontend** — Streamlit only for v1; a more polished frontend is a natural
  next step once the core pipeline is validated on real data.
- **Full robustness cross-product grid** — the robustness suite tests each
  degradation independently on the final fusion model only, not every combination
  across every ablation config.

## Project Structure

```
image-forensics-ai/
├── configs/            # config.yaml — single source of truth for paths/hyperparams
├── data/                # raw/processed data (gitignored — see data/README.md)
├── src/
│   ├── config.py        # typed config loader (pydantic)
│   ├── utils.py          # shared utilities (seeding, etc.)
│   ├── data/              # dataset loading, splitting, metadata
│   ├── models/             # backbone + fusion architecture
│   ├── features/            # FFT/DCT + noise-residual feature extraction
│   ├── training/              # training loops, ablation runner
│   ├── evaluation/              # metrics, calibration, robustness tests
│   ├── explainability/            # Grad-CAM
│   ├── inference/                   # inference pipeline (shared by API + app)
│   └── deployment/                    # model packaging for HF Spaces
├── api/                 # FastAPI serving layer
├── app/                  # Streamlit UI
├── models/production/      # deployed model weights (git-lfs tracked)
├── tests/                # pytest suite (104 tests, 96%+ coverage)
└── screenshots/           # UI/result screenshots for this README
```

## Non-Negotiable Constraints

- No paid LLM/AI API anywhere in the core detection system.
- No fabricated datasets, benchmarks, or metrics — every number in this README comes
  from an actual run, which is exactly why several sections above are marked PENDING
  rather than filled with placeholder figures.
- $0 recurring cost; final demo is a free, publicly deployed link (Hugging Face
  Spaces, CPU tier).

## License

Code: MIT (see [`LICENSE`](LICENSE)). Dataset license will be documented here,
verbatim, once a real dataset is selected and verified — see
[`data/README.md`](data/README.md).
