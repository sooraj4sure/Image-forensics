# Dataset — Stage 2 notes

## Status: dataset choice tentative, license/availability NOT yet independently verified

Per the project brief's non-negotiable rule ("verify current availability/licensing
rather than assuming last-known status is still accurate"), the two candidates below
are documented honestly as **candidates**, not confirmed choices. Web search was not
available when this was written, so **before downloading real data, confirm current
license terms and access on the dataset's actual current hosting page** — don't take
this file's word for it.

### Candidate A — GenImage (leaning towards this one)
- Real images (ImageNet-based) + AI images from 8 generator families (Midjourney,
  Stable Diffusion v1.4/v1.5, ADM, GLIDE, Wukong, VQDM, BigGAN).
- Preferred because the project's strongest claim (§8 of the brief, unseen-generator
  generalization) needs 2-3 generator families for training with one held out
  entirely — GenImage's structure supports that directly. CIFAKE (below) does not.
- **To verify before use:** current hosting location, whether download requires
  a request/gated access, and whether license terms differ per generator (some
  generator outputs may inherit that generator's own terms of service).

### Candidate B — CIFAKE (fallback)
- ~120k images, CIFAR-10 reals vs. Stable-Diffusion-generated fakes, 32×32 resolution.
- Simpler / more reliably accessible historically (Kaggle-hosted), but single
  generator family only — would need a supplementary source to get the
  generator diversity §5/§8 require, and 32×32 is well below the 224×224
  ResNet-18 input size the config assumes (would need upscaling, which is itself
  worth noting as a limitation if this path is taken).

### Decision to make (by you, once verified)
Fill in `configs/config.yaml` → `data.dataset_name`, `data.dataset_source_url`,
`data.dataset_license`, `data.generator_families`, `data.held_out_generator`
once confirmed. The pipeline code in `src/data/` is dataset-agnostic and only
needs the raw files laid out per the convention below — it doesn't care which
dataset you picked.

---

## Expected raw data layout

The metadata builder (`src/data/build_metadata.py`) expects:

```
data/raw/
├── real/
│   └── *.jpg / *.png / *.webp        # flat — source not distinguished, real is real
└── ai/
    ├── <generator_name_1>/
    │   └── *.jpg / *.png / *.webp
    ├── <generator_name_2>/
    │   └── ...
    └── <generator_name_3>/            # this one becomes held_out_generator in config
        └── ...
```

Generator subfolder names under `data/raw/ai/` become the `generator` column in
the metadata CSV and must match `data.generator_families` /
`data.held_out_generator` in `configs/config.yaml` exactly.

## Building metadata + splits

Once raw images are in place:

```bash
./.venv/bin/python -m src.data.build_metadata
```

This writes `data/metadata/metadata.csv` with columns:
`path, class, source, generator, split`

Split logic (see `src/data/splits.py`):
- The `held_out_generator` (from config) is **excluded from train and val entirely**
  and placed only in a dedicated unseen-generator test set — this is what makes
  the §8 experiment valid (the model must never see this generator during training).
- All other data is split train/val/test using the `val_split`/`test_split`
  fractions in config, **stratified by class** (real/AI) so class balance is
  preserved in every split.
- Splits are reproducible: identical input + identical `project.seed` in config
  always produces identical splits (verified in `tests/test_data_splits.py`
  using synthetic dummy data, since real data isn't downloaded in this
  environment).
