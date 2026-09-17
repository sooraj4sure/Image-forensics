# Dataset

## Status: GenImage subset confirmed and organized. License NOT yet independently verified.

Per the project brief's non-negotiable rule ("verify current availability/licensing
rather than assuming last-known status is still accurate"), the license terms below
still need a manual check on the dataset's Kaggle page before this project's README
license section is finalized — nothing here should be taken as a confirmed license.

## Dataset: GenImage (subset)

- Source: https://www.kaggle.com/datasets/renhuang8/genimage-subset-detection
- Downloaded and inspected directly (not from memory) — confirmed structure:

| Folder | Class | Generator | Count |
|---|---|---|---|
| `real_pool/` | REAL | — | 6000 |
| `sd_pool/` | AI_GENERATED | Stable Diffusion | 5000 |
| `gan_pool/` | AI_GENERATED | GAN | 500 |
| `mj_pool/` | AI_GENERATED | Midjourney | 500 |

Total: 6000 real, 6000 AI (well balanced). Three distinct generator families,
satisfying the brief's requirement of 2-3 generator families for training with
one held out entirely for the unseen-generator test (§5/§8) — unlike CIFAKE
(considered earlier, single generator only, see git history for that discussion).

**Held-out generator: `gan` (gan_pool).** Chosen over holding out `sd_pool` (would
leave only 1000 AI training images total — too thin) and considered against holding
out `mj_pool` instead (also 500 images, tied on size) — `gan` was chosen as the more
meaningful generalization test: training only on diffusion-family generators
(Stable Diffusion + Midjourney) and testing whether the model generalizes to a
different generation architecture entirely (GAN), not just a different diffusion
variant.

## Raw data layout convention (matches src/data/build_metadata.py)

GenImage's original folder names are remapped to this project's convention:

```
genimage_pool_name  ->  this project's data/raw/ layout
─────────────────────────────────────────────────────────
real_pool/           -> data/raw/real/
sd_pool/              -> data/raw/ai/stable_diffusion/
gan_pool/              -> data/raw/ai/gan/            (held out — never used in train/val)
mj_pool/                -> data/raw/ai/midjourney/
```

Reorganization happens in Colab (see project chat history / Colab notebook) since
the ~4GB dataset is downloaded and prepared there directly, not on the local dev
machine or in this repo's own environment.

## Building metadata + splits

Once raw images are in place at `data/raw/` per the layout above:

```bash
./.venv/bin/python -m src.data.build_metadata
```

This writes `data/metadata/metadata.csv`. Given `held_out_generator: "gan"` in
`configs/config.yaml`, all 500 `gan_pool` images go to a dedicated `unseen_test`
split, excluded from train/val entirely (see `src/data/splits.py` for the
isolation logic and `src/evaluation/unseen_generator.py` for why that split's
evaluation uses AI-detection recall rather than full accuracy/F1/ROC-AUC — it's
single-class by construction).
