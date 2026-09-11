"""
Scan data/raw/ per the convention in data/README.md and write
data/metadata/metadata.csv with class/source/generator/split columns.

Run as: ./.venv/bin/python -m src.data.build_metadata
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import load_config
from src.data.splits import assign_splits

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def scan_raw_data(raw_dir: Path) -> pd.DataFrame:
    """
    Walk data/raw/real/ and data/raw/ai/<generator>/ and build a raw
    (pre-split) metadata DataFrame with columns: path, class, source, generator.
    """
    rows = []

    real_dir = raw_dir / "real"
    if real_dir.exists():
        for p in sorted(real_dir.rglob("*")):
            if p.suffix.lower() in IMAGE_EXTS:
                rows.append(
                    {
                        "path": str(p),
                        "class": "REAL",
                        "source": "real",
                        "generator": None,
                    }
                )

    ai_dir = raw_dir / "ai"
    if ai_dir.exists():
        for generator_dir in sorted(p for p in ai_dir.iterdir() if p.is_dir()):
            for p in sorted(generator_dir.rglob("*")):
                if p.suffix.lower() in IMAGE_EXTS:
                    rows.append(
                        {
                            "path": str(p),
                            "class": "AI_GENERATED",
                            "source": "ai",
                            "generator": generator_dir.name,
                        }
                    )

    return pd.DataFrame(rows, columns=["path", "class", "source", "generator"])


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows by path (brief §5: 'Deduplicate')."""
    before = len(df)
    df = df.drop_duplicates(subset=["path"]).reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"Deduplicated: {before} -> {after} rows ({before - after} removed)")
    return df


def main() -> None:
    cfg = load_config()
    raw_dir = cfg.paths.resolve("data_raw")
    metadata_path = cfg.paths.resolve("data_metadata")

    print(f"Scanning raw data at {raw_dir} ...")
    df = scan_raw_data(raw_dir)

    if df.empty:
        print(
            "No images found under data/raw/real/ or data/raw/ai/<generator>/. "
            "Download and place a real dataset first — see data/README.md."
        )
        return

    df = deduplicate(df)

    print(
        f"Found {len(df)} images: "
        f"{(df['class'] == 'REAL').sum()} REAL, "
        f"{(df['class'] == 'AI_GENERATED').sum()} AI_GENERATED, "
        f"generators: {sorted(df['generator'].dropna().unique().tolist())}"
    )

    df = assign_splits(
        df,
        held_out_generator=cfg.data.held_out_generator,
        val_split=cfg.data.val_split,
        test_split=cfg.data.test_split,
        seed=cfg.project.seed,
    )

    print(df["split"].value_counts())

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(metadata_path, index=False)
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
