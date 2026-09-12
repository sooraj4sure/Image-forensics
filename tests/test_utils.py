"""Test for src/utils.py's set_seed."""

import random

import numpy as np
import torch

from src.utils import set_seed


def test_set_seed_makes_python_random_reproducible():
    set_seed(123)
    a = [random.random() for _ in range(5)]
    set_seed(123)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_set_seed_makes_numpy_reproducible():
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_set_seed_makes_torch_reproducible():
    set_seed(123)
    a = torch.rand(5)
    set_seed(123)
    b = torch.rand(5)
    assert torch.equal(a, b)


def test_different_seeds_give_different_sequences():
    set_seed(1)
    a = np.random.rand(5)
    set_seed(2)
    b = np.random.rand(5)
    assert not np.array_equal(a, b)
