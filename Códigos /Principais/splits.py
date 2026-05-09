#!/usr/bin/env python3
"""
Inter-subject representative split for animal-behavior datasets.

For a given features_full.csv with columns including 'subject_id' and 'label':
  - Split subjects into train / val / test (default 70/15/15 by count of subjects).
  - Search 200 random seeds.
  - Pick split where the test/val class distribution most closely matches the
    global class distribution (minimize sum of KL divergences).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

EXP_ROOT = Path("/Users/joanafontes/Documents/Datasets/experiment_dual")


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def _class_dist(labels: np.ndarray, classes: list[str]) -> np.ndarray:
    counts = pd.Series(labels).value_counts()
    return np.array([counts.get(c, 0) for c in classes], dtype=float)


def find_representative_split(
    features_full_csv: Path,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    n_iter: int = 200,
    seed: int = 42,
) -> dict:
    df = pd.read_csv(features_full_csv, usecols=["subject_id", "label"])
    df["subject_id"] = df["subject_id"].astype(str)
    subjects = df["subject_id"].unique()
    n_total = len(subjects)
    n_train = max(1, int(round(n_total * train_frac)))
    n_val = max(1, int(round(n_total * val_frac)))
    n_test = n_total - n_train - n_val
    # If split is degenerate (e.g. n_total=5 and rounding gives 0 test), rebalance.
    while n_test < 1 and n_train > 1:
        n_train -= 1
        n_test = n_total - n_train - n_val
    if n_total < 3:
        raise ValueError(f"Need at least 3 subjects, got {n_total}.")
    if n_val < 1 or n_test < 1:
        raise ValueError(f"Cannot split {n_total} subjects into train/val/test.")

    classes = sorted(df["label"].unique().tolist())
    global_dist = _class_dist(df["label"].values, classes)
    rng = np.random.default_rng(seed)

    best = None
    for _ in range(n_iter):
        perm = rng.permutation(subjects)
        train_subj = set(perm[:n_train])
        val_subj = set(perm[n_train:n_train + n_val])
        test_subj = set(perm[n_train + n_val:])

        # need at least 1 sample of any class in train, otherwise model can't learn
        train_lbls = df.loc[df["subject_id"].isin(train_subj), "label"].values
        if len(set(train_lbls)) < min(3, len(classes)):
            continue
        val_lbls = df.loc[df["subject_id"].isin(val_subj), "label"].values
        test_lbls = df.loc[df["subject_id"].isin(test_subj), "label"].values
        if len(val_lbls) == 0 or len(test_lbls) == 0:
            continue

        kl_test = _kl(_class_dist(test_lbls, classes), global_dist)
        kl_val = _kl(_class_dist(val_lbls, classes), global_dist)
        score = kl_test + kl_val

        if best is None or score < best["score"]:
            best = {
                "score": score,
                "kl_test": kl_test,
                "kl_val": kl_val,
                "train_subjects": sorted(train_subj),
                "val_subjects": sorted(val_subj),
                "test_subjects": sorted(test_subj),
                "n_train": len(train_subj),
                "n_val": len(val_subj),
                "n_test": len(test_subj),
                "global_class_dist": dict(zip(classes, global_dist.tolist())),
            }
    if best is None:
        raise RuntimeError("Failed to find any valid split.")
    best.pop("score")
    return best


def build_split(dataset: str) -> dict:
    """Build (or load cached) split for a dataset and save to splits.json."""
    out_path = EXP_ROOT / dataset / "splits.json"
    if out_path.exists():
        with open(out_path) as fh:
            return json.load(fh)
    feats_full = EXP_ROOT / dataset / "features" / "features_full.csv"
    if not feats_full.exists():
        raise FileNotFoundError(f"Missing {feats_full} – run extract_features first.")
    split = find_representative_split(feats_full)
    with open(out_path, "w") as fh:
        json.dump(split, fh, indent=2)
    print(f"  split saved → {out_path}")
    print(f"  train={split['n_train']}  val={split['n_val']}  test={split['n_test']}  "
          f"kl_test={split['kl_test']:.4f}  kl_val={split['kl_val']:.4f}")
    return split


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["vehkaoja", "marinara", "both"], default="both")
    args = ap.parse_args()

    targets = ["vehkaoja", "marinara"] if args.dataset == "both" else [args.dataset]
    for ds in targets:
        print(f"\n=== Building split for {ds} ===")
        build_split(ds)
