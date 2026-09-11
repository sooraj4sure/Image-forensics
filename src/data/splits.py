"""
Train/val/test/unseen-generator splitting logic.

Kept separate from build_metadata.py so it can be unit-tested with a plain
pandas DataFrame — no filesystem or image I/O needed to verify the split
logic itself.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def assign_splits(
    df: pd.DataFrame,
    held_out_generator: str | None,
    val_split: float,
    test_split: float,
    seed: int,
) -> pd.DataFrame:
    """
    Assign a `split` column to a metadata DataFrame with columns
    [path, class, source, generator].

    Rules:
    - Rows where generator == held_out_generator are ALWAYS assigned to
      split="unseen_test" and are excluded from the train/val/test split
      below entirely (brief §8: the model must never see this generator
      during training).
    - Everything else is split into train/val/test, stratified by `class`
      so real/AI balance is preserved in each split.

    Raises if held_out_generator is set but not present in the data, or if
    the requested split fractions don't leave a viable train fraction —
    fail loudly rather than silently producing an empty split.
    """
    df = df.copy()

    if held_out_generator is not None:
        if held_out_generator not in df["generator"].unique():
            raise ValueError(
                f"held_out_generator={held_out_generator!r} not found in data. "
                f"Available generators: {sorted(df['generator'].dropna().unique())}"
            )
        is_held_out = df["generator"] == held_out_generator
        held_out_df = df[is_held_out].copy()
        held_out_df["split"] = "unseen_test"
        remaining_df = df[~is_held_out].copy()
    else:
        held_out_df = df.iloc[0:0].copy()
        held_out_df["split"] = pd.Series(dtype="object")
        remaining_df = df.copy()

    if val_split + test_split >= 1.0:
        raise ValueError(
            f"val_split ({val_split}) + test_split ({test_split}) must be < 1.0 "
            f"to leave a non-empty train split."
        )

    # First peel off test, then val, both stratified by class.
    train_val_df, test_df = train_test_split(
        remaining_df,
        test_size=test_split,
        stratify=remaining_df["class"],
        random_state=seed,
    )
    # val_split is expressed as a fraction of the ORIGINAL data; convert to
    # a fraction of what remains after removing test.
    relative_val_size = val_split / (1.0 - test_split)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        stratify=train_val_df["class"],
        random_state=seed,
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    result = pd.concat([train_df, val_df, test_df, held_out_df], ignore_index=True)
    return result
