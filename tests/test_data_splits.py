"""
Tests for src/data/splits.py using synthetic (fabricated-on-the-fly, clearly
not real-dataset) metadata. These test the SPLITTING LOGIC only — no real
images or real dataset numbers are involved, so this doesn't violate the
brief's "no fabricated metrics" rule (that rule is about reported results,
not about unit-test fixtures).
"""

import pandas as pd
import pytest

from src.data.splits import assign_splits


def make_synthetic_df(n_real=100, n_ai_per_gen=50, generators=("genA", "genB", "genC")):
    rows = []
    for i in range(n_real):
        rows.append({"path": f"real_{i}.jpg", "class": "REAL", "source": "real", "generator": None})
    for gen in generators:
        for i in range(n_ai_per_gen):
            rows.append(
                {"path": f"{gen}_{i}.jpg", "class": "AI_GENERATED", "source": "ai", "generator": gen}
            )
    return pd.DataFrame(rows)


def test_held_out_generator_fully_isolated():
    df = make_synthetic_df()
    result = assign_splits(df, held_out_generator="genC", val_split=0.15, test_split=0.15, seed=42)

    unseen = result[result["split"] == "unseen_test"]
    assert len(unseen) == 50
    assert (unseen["generator"] == "genC").all()

    # genC must not leak into train/val/test
    for split in ["train", "val", "test"]:
        split_df = result[result["split"] == split]
        assert "genC" not in split_df["generator"].values


def test_no_held_out_generator():
    df = make_synthetic_df()
    result = assign_splits(df, held_out_generator=None, val_split=0.15, test_split=0.15, seed=42)
    assert "unseen_test" not in result["split"].values
    assert set(result["split"].unique()) == {"train", "val", "test"}
    assert len(result) == len(df)


def test_split_proportions_roughly_correct():
    df = make_synthetic_df(n_real=200, n_ai_per_gen=100, generators=("genA", "genB", "genC"))
    result = assign_splits(df, held_out_generator="genC", val_split=0.15, test_split=0.15, seed=42)

    non_held_out = result[result["split"] != "unseen_test"]
    n = len(non_held_out)
    counts = non_held_out["split"].value_counts(normalize=True)

    assert abs(counts["test"] - 0.15) < 0.03
    assert abs(counts["val"] - 0.15) < 0.03
    assert abs(counts["train"] - 0.70) < 0.05


def test_class_balance_preserved_in_each_split():
    df = make_synthetic_df(n_real=200, n_ai_per_gen=100, generators=("genA", "genB", "genC"))
    result = assign_splits(df, held_out_generator="genC", val_split=0.2, test_split=0.2, seed=42)

    # Baseline must exclude the held-out generator — it's excluded from
    # train/val/test entirely, so it shouldn't be part of the comparison ratio.
    non_held_out_source = df[df["generator"] != "genC"]
    overall_ai_frac = (non_held_out_source["class"] == "AI_GENERATED").mean()
    for split in ["train", "val", "test"]:
        split_df = result[result["split"] == split]
        split_ai_frac = (split_df["class"] == "AI_GENERATED").mean()
        # Stratified split should keep class balance close to the source balance
        assert abs(split_ai_frac - overall_ai_frac) < 0.05


def test_reproducible_with_same_seed():
    df = make_synthetic_df()
    result1 = assign_splits(df, held_out_generator="genC", val_split=0.15, test_split=0.15, seed=42)
    result2 = assign_splits(df, held_out_generator="genC", val_split=0.15, test_split=0.15, seed=42)

    r1_sorted = result1.sort_values("path").reset_index(drop=True)
    r2_sorted = result2.sort_values("path").reset_index(drop=True)
    pd.testing.assert_series_equal(r1_sorted["split"], r2_sorted["split"])


def test_different_seed_gives_different_split():
    df = make_synthetic_df()
    result1 = assign_splits(df, held_out_generator="genC", val_split=0.15, test_split=0.15, seed=42)
    result2 = assign_splits(df, held_out_generator="genC", val_split=0.15, test_split=0.15, seed=99)

    r1_sorted = result1.sort_values("path").reset_index(drop=True)
    r2_sorted = result2.sort_values("path").reset_index(drop=True)
    # Not asserting total inequality (small chance of coincidence) — just that
    # different seeds produce a materially different assignment.
    disagreement = (r1_sorted["split"] != r2_sorted["split"]).mean()
    assert disagreement > 0.01


def test_raises_on_missing_held_out_generator():
    df = make_synthetic_df()
    with pytest.raises(ValueError, match="not found in data"):
        assign_splits(df, held_out_generator="genZ", val_split=0.15, test_split=0.15, seed=42)


def test_raises_on_invalid_split_fractions():
    df = make_synthetic_df()
    with pytest.raises(ValueError, match="must be < 1.0"):
        assign_splits(df, held_out_generator="genC", val_split=0.6, test_split=0.6, seed=42)
