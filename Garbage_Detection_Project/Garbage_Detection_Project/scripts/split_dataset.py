"""Reusable deterministic splitting logic for YOLO datasets.

Used by prepare_final_dataset.py so every split in this project comes from
one auditable implementation.
"""
from __future__ import annotations

import random
from collections.abc import Sequence

SPLIT_ORDER = ("train", "val", "test")
DEFAULT_FRACTIONS = {"train": 0.80, "val": 0.10, "test": 0.10}


def deterministic_shuffle(items: Sequence, seed: int) -> list:
    items = list(items)
    rng = random.Random(seed)
    rng.shuffle(items)
    return items


def assign_splits(n: int, seed: int, fractions: dict | None = None) -> list[str]:
    """Return a list of n split names, stratified by the given fractions.

    Deterministic for a given (n, seed, fractions) so the same source listing
    always yields the same split assignment.
    """
    fractions = fractions or DEFAULT_FRACTIONS
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError(f"split fractions must sum to 1.0, got {fractions}")
    missing = set(SPLIT_ORDER) - set(fractions)
    if missing:
        raise ValueError(f"missing split fraction for {sorted(missing)}")

    order = deterministic_shuffle(range(n), seed)
    assignment = [None] * n
    n_train = int(n * fractions["train"])
    n_val = int(n * fractions["val"])
    boundaries = {
        "train": n_train,
        "val": n_train + n_val,
        "test": n,
    }
    for pos, original_index in enumerate(order):
        if pos < boundaries["train"]:
            assignment[original_index] = "train"
        elif pos < boundaries["val"]:
            assignment[original_index] = "val"
        else:
            assignment[original_index] = "test"
    return assignment
