# Deployment: Hugging Face Spaces

This is the runbook for shipping a trained model to a live, free, public
demo (brief §12). It assumes you've already trained `full_fusion` on real
data (Stages 1-7) and are happy with its results.

## 1. Package the model for deployment

```bash
./.venv/bin/python -m src.evaluation.fit_calibration --config full_fusion   # if not already done
./.venv/bin/python -m src.deployment.package_model --config full_fusion
```

This copies `runs/ablation_full_fusion/best_model.pt` (+ `temperature.json`
if present) into `models/production/` — the fixed path the deployed app
reads via the `IMAGEFORENSICS_MODEL_DIR` environment variable, decoupled
from the `runs/ablation_<config>` experiment-naming convention used during
development. The script prints the model's size and warns if it exceeds
100MB (brief §12: "well under 100MB so CPU inference latency stays
acceptable and the free tier's memory limits aren't hit").

## 2. Commit the packaged model

```bash
git lfs install                          # once per machine
git add .gitattributes models/production/
git commit -m "Package full_fusion model for deployment"
```

`.gitattributes` already tracks `models/production/*.pt` via git-lfs (see
that file) — regular `git add`/`commit` handles the LFS pointer
automatically once `git lfs install` has run.

## 3. Create the Space and push

1. Go to https://huggingface.co/new-space, choose **Streamlit** as the SDK
   and **CPU basic** (free tier) as the hardware.
2. Add it as a second git remote and push:

```bash
git remote add space https://huggingface.co/spaces/<your-username>/image-forensics-ai
git push space main
```

The root `README.md`'s YAML frontmatter (`sdk: streamlit`,
`app_file: app/streamlit_app.py`) tells HF Spaces how to run the app — no
separate Space-specific README needed. HF Spaces installs from
`requirements.txt` (the lean, runtime-only file — `requirements-dev.txt`
is not installed, keeping the build faster).

## 4. Set the model path environment variable

In the Space's **Settings → Variables and secrets**, add:

```
IMAGEFORENSICS_MODEL_DIR=models/production
```

This tells `ForensicsPipeline` to load the committed model rather than
looking for a `runs/ablation_full_fusion/` directory that won't exist in
the deployed environment.

## 5. Verify the live demo

Once the Space finishes building (first build is slower — installing
torch/torchvision on CPU takes a few minutes), open the Space URL, upload
a test image, and confirm the forensic report renders with a Grad-CAM
heatmap. Put the resulting URL in the main README's "Live Demo Link"
section (brief §13).

## Free-tier limitations (brief §12 — document explicitly)

- **Sleeping instances:** free CPU Spaces go to sleep after a period of
  inactivity and cold-start on the next visit (can take 30-60+ seconds).
- **CPU-only inference:** no GPU on the free tier — this is exactly why
  the model is kept ResNet-18-scale and well under 100MB (brief §6/§12).
- **Memory ceiling:** free Spaces have a limited RAM allocation (historically
  ~16GB, but confirm current limits on HF's pricing page — these change).
- **Storage limits:** the free tier caps total repo storage; git-lfs model
  files count against this.
- **Request limits / no SLA:** free Spaces have no uptime guarantee and can
  be rate-limited or reclaimed under HF's fair-use policies — check HF's
  current Spaces documentation for specifics, as these are the kind of
  terms that change over time and shouldn't be taken as fixed here.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'src'`** — this was a real bug
  found during deployment prep: Streamlit's script runner only puts the
  script's own directory on `sys.path`, not the repo root. Already fixed
  in `app/streamlit_app.py` (a `sys.path.insert` at the top of the file,
  before the `src.*` imports) — if you see this again after modifying that
  file, check that fix wasn't accidentally removed.
- **"No trained model checkpoint found"** in the deployed app — check that
  `IMAGEFORENSICS_MODEL_DIR` is set correctly in the Space's variables and
  that `models/production/best_model.pt` was actually committed (git-lfs
  pointer, not an empty/broken file — check the file size on the Space's
  Files tab).
- **Build fails on `torch` install** — free CPU Spaces occasionally hit
  build timeouts on the first (cold) build installing torch; retrying the
  build (Settings → Factory reboot) usually resolves this.
