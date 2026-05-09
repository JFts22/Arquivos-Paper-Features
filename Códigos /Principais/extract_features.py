#!/usr/bin/env python3
"""
TSFEL feature extraction with subject_id preserved per window.

Supports two datasets:
- vehkaoja: /Vehkaoja/DogMoveData_marinara.csv  (12 sensors, DogID, Behavior)
- marinara: /Datasets/Marinara/df_raw.csv       (27 sensors, Subject, Position, Type)

Output for each dataset: features_{full,static,dynamic}.csv with columns
[<feature_*>, subject_id, label].
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tsfel

warnings.filterwarnings("ignore")

WINDOW_SIZE = 100
OVERLAP = 0.5
FS = 100
STRIDE = int(WINDOW_SIZE * (1 - OVERLAP))

VEHKAOJA_RAW = Path("/Users/joanafontes/Documents/Datasets/Vehkaoja/DogMoveData_marinara.csv")
MARINARA_RAW = Path("/Users/joanafontes/Documents/Datasets/Marinara/df_raw.csv")
EXP_ROOT = Path("/Users/joanafontes/Documents/Datasets/experiment_dual")

VEHKAOJA_SENSORS = [
    "ABack_x", "ABack_y", "ABack_z",
    "ANeck_x", "ANeck_y", "ANeck_z",
    "GBack_x", "GBack_y", "GBack_z",
    "GNeck_x", "GNeck_y", "GNeck_z",
]

VEHKAOJA_STATIC = frozenset(["Sitting", "Lying chest", "Standing", "Panting"])
VEHKAOJA_DYNAMIC = frozenset([
    "Walking", "Trotting", "Pacing", "Galloping", "Playing",
    "Jumping", "Shaking", "Bowing", "Tugging", "Sniffing",
    "Eating", "Drinking", "Carrying object",
])
VEHKAOJA_VALID = VEHKAOJA_STATIC | VEHKAOJA_DYNAMIC

MARINARA_META = {"Timestamp", "Type", "Position", "Breed", "Subject"}


def _window_majority(labels: np.ndarray, n_windows: int, valid: set | None = None) -> list[str]:
    """Return majority-vote label per sliding window."""
    out: list[str] = []
    for i in range(n_windows):
        start = i * STRIDE
        end = min(start + WINDOW_SIZE, len(labels))
        seg = labels[start:end]
        if valid is not None:
            seg = [l for l in seg if l in valid]
            if not seg:
                out.append("<undefined>")
                continue
        vals, cnts = np.unique(seg, return_counts=True)
        out.append(vals[cnts.argmax()])
    return out


def _extract_per_subject(
    df: pd.DataFrame,
    subject_col: str,
    label_col: str,
    sensor_cols: list[str],
    cfg_tsfel,
    valid_labels: set | None = None,
    sort_col: str | None = None,
    type_col: str | None = None,
    log_prefix: str = "",
) -> pd.DataFrame:
    """Extract TSFEL features per subject, preserving subject_id and label.

    If type_col is provided, also preserves a 'type' column for static/dynamic split.
    """
    feats_acc: list[pd.DataFrame] = []
    label_acc: list[str] = []
    subject_acc: list[str] = []
    type_acc: list[str] = []

    subjects = df[subject_col].unique()
    n_subjects = len(subjects)

    for idx, subj in enumerate(subjects):
        sub = df[df[subject_col] == subj]
        if sort_col and sort_col in sub.columns:
            sub = sub.sort_values(sort_col)
        sub = sub.reset_index(drop=True)

        n = len(sub)
        if n < WINDOW_SIZE:
            continue

        signal = sub[sensor_cols].astype(np.float32)
        labels = sub[label_col].values

        try:
            feats = tsfel.time_series_features_extractor(
                cfg_tsfel, signal, fs=FS,
                window_size=WINDOW_SIZE, overlap=OVERLAP, verbose=0,
            )
        except Exception as exc:
            print(f"  {log_prefix}skip subj={subj}: {exc}", flush=True)
            continue

        n_win = len(feats)
        win_labels = _window_majority(labels, n_win, valid_labels)

        if type_col is not None:
            types = sub[type_col].values
            win_types = _window_majority(types, n_win, valid=None)
            type_acc.extend(win_types)

        feats_acc.append(feats)
        label_acc.extend(win_labels)
        subject_acc.extend([str(subj)] * n_win)

        if (idx + 1) % 5 == 0 or (idx + 1) == n_subjects:
            total = sum(len(f) for f in feats_acc)
            print(f"  {log_prefix}[{idx+1:3d}/{n_subjects}] subj={subj} | total_windows={total:,}",
                  flush=True)

    feats_df = pd.concat(feats_acc, ignore_index=True)
    feats_df["subject_id"] = subject_acc
    feats_df["label"] = label_acc
    if type_col is not None:
        feats_df["type"] = type_acc
    return feats_df


def _save_three(feats_full: pd.DataFrame, out_dir: Path, *, valid_static: set, valid_dynamic: set) -> None:
    """Filter the full feature DF into static/dynamic and save all three CSVs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # full: keep only valid (static OR dynamic) rows
    full = feats_full[feats_full["label"].isin(valid_static | valid_dynamic)].reset_index(drop=True)
    static = feats_full[feats_full["label"].isin(valid_static)].reset_index(drop=True)
    dynamic = feats_full[feats_full["label"].isin(valid_dynamic)].reset_index(drop=True)

    for name, ds in [("full", full), ("static", static), ("dynamic", dynamic)]:
        path = out_dir / f"features_{name}.csv"
        ds.to_csv(path, index=False)
        print(f"  saved {name:8s}: {len(ds):,} rows × {ds.shape[1]} cols  →  {path}")


def extract_vehkaoja(limit_subjects: int | None = None) -> Path:
    """Extract TSFEL features from Vehkaoja DogMoveData_marinara.csv."""
    out_dir = EXP_ROOT / "vehkaoja" / "features"
    full_path = out_dir / "features_full.csv"
    static_path = out_dir / "features_static.csv"
    dynamic_path = out_dir / "features_dynamic.csv"
    if all(p.exists() for p in (full_path, static_path, dynamic_path)) and limit_subjects is None:
        print(f"  [CACHE] features already exist in {out_dir}")
        return out_dir

    print(f"Loading Vehkaoja raw CSV from {VEHKAOJA_RAW}...")
    df = pd.read_csv(VEHKAOJA_RAW)
    print(f"  {len(df):,} rows loaded.")

    if limit_subjects is not None:
        keep = df["DogID"].unique()[:limit_subjects]
        df = df[df["DogID"].isin(keep)].copy()
        print(f"  --limit: kept {limit_subjects} dogs ({len(df):,} rows).")

    cfg = tsfel.get_features_by_domain()

    # Mark every label, then filter at save time (so 'full' = static∪dynamic only)
    feats = _extract_per_subject(
        df, subject_col="DogID", label_col="Behavior",
        sensor_cols=VEHKAOJA_SENSORS, cfg_tsfel=cfg,
        valid_labels=VEHKAOJA_VALID, sort_col="t_sec",
        log_prefix="vehkaoja ",
    )
    _save_three(feats, out_dir,
                valid_static=VEHKAOJA_STATIC, valid_dynamic=VEHKAOJA_DYNAMIC)
    return out_dir


def extract_marinara(limit_subjects: int | None = None) -> Path:
    """Extract TSFEL features from Marinara df_raw.csv."""
    out_dir = EXP_ROOT / "marinara" / "features"
    full_path = out_dir / "features_full.csv"
    static_path = out_dir / "features_static.csv"
    dynamic_path = out_dir / "features_dynamic.csv"
    if all(p.exists() for p in (full_path, static_path, dynamic_path)) and limit_subjects is None:
        print(f"  [CACHE] features already exist in {out_dir}")
        return out_dir

    print(f"Loading Marinara raw CSV from {MARINARA_RAW}...")
    df = pd.read_csv(MARINARA_RAW)
    print(f"  {len(df):,} rows loaded.")

    if limit_subjects is not None:
        keep = df["Subject"].unique()[:limit_subjects]
        df = df[df["Subject"].isin(keep)].copy()
        print(f"  --limit: kept {limit_subjects} subjects ({len(df):,} rows).")

    sensor_cols = [c for c in df.columns if c not in MARINARA_META]
    print(f"  using {len(sensor_cols)} sensor channels.")

    cfg = tsfel.get_features_by_domain()

    feats = _extract_per_subject(
        df, subject_col="Subject", label_col="Position",
        sensor_cols=sensor_cols, cfg_tsfel=cfg,
        valid_labels=None, sort_col="Timestamp",
        type_col="Type",
        log_prefix="marinara ",
    )

    # Marinara: split by 'type' column, not by label set
    out_dir.mkdir(parents=True, exist_ok=True)
    full = feats.reset_index(drop=True)
    static = feats[feats["type"] == "static"].reset_index(drop=True)
    dynamic = feats[feats["type"] == "dynamic"].reset_index(drop=True)
    for name, ds in [("full", full), ("static", static), ("dynamic", dynamic)]:
        path = out_dir / f"features_{name}.csv"
        # drop the auxiliary 'type' column for static/dynamic (redundant); keep in full
        if name in ("static", "dynamic"):
            ds = ds.drop(columns=["type"])
        ds.to_csv(path, index=False)
        print(f"  saved {name:8s}: {len(ds):,} rows × {ds.shape[1]} cols  →  {path}")
    return out_dir


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["vehkaoja", "marinara", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None,
                    help="limit number of subjects (for smoke test)")
    args = ap.parse_args()

    if args.dataset in ("vehkaoja", "both"):
        print("\n" + "=" * 70)
        print(" Extracting Vehkaoja features")
        print("=" * 70)
        extract_vehkaoja(limit_subjects=args.limit)
    if args.dataset in ("marinara", "both"):
        print("\n" + "=" * 70)
        print(" Extracting Marinara features")
        print("=" * 70)
        extract_marinara(limit_subjects=args.limit)
