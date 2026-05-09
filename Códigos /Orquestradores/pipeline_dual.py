#!/usr/bin/env python3
"""
Orchestrator for the Vehkaoja vs Marinara feature-selection experiment.

Per dataset:
  1. Extract TSFEL features (with subject_id preserved)            → extract_features
  2. Build inter-subject representative split                       → splits
  3. Run 10 FS methods × 2 strategies (full vs static+dynamic)      → feature_selection
  4. Sweep DL hyperparameters (3 dims × 3 lrs) on (full, anova_f)   → models
  5. Train+evaluate RF / SVM / best-CNN / best-LSTM on every cell   → models
  6. Aggregate to summary.csv

Usage:
  python3 pipeline_dual.py                       # full run on both datasets
  python3 pipeline_dual.py --dataset vehkaoja    # one dataset only
  python3 pipeline_dual.py --smoke 5             # smoke test on 5 subjects
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import extract_features as ef
import feature_selection as fs
import models as mdl
import splits as sp

EXP_ROOT = Path("/Users/joanafontes/Documents/Datasets/experiment_dual")
K_FEATURES = 20
MIN_SAMPLES_PER_CLASS = 10  # in train

DL_DIMS = [32, 64, 128]
DL_LRS = [1e-2, 1e-3, 1e-4]


# ───────────────────────── helpers ─────────────────────────

def _load_features(dataset: str) -> dict[str, pd.DataFrame]:
    feat_dir = EXP_ROOT / dataset / "features"
    out = {}
    for name in ("full", "static", "dynamic"):
        out[name] = pd.read_csv(feat_dir / f"features_{name}.csv")
    return out


def _split_xy(
    df: pd.DataFrame, train: set, val: set, test: set,
    feature_cols: list[str] | None = None,
) -> dict:
    """Split a feature DataFrame by subject lists; return X/y per part + feature_cols."""
    df = df.copy()
    df["subject_id"] = df["subject_id"].astype(str)
    if feature_cols is None:
        meta = {"subject_id", "label", "type"}
        feature_cols = [c for c in df.columns if c not in meta]

    train_mask = df["subject_id"].isin(train)
    val_mask = df["subject_id"].isin(val)
    test_mask = df["subject_id"].isin(test)

    return {
        "X_train": df.loc[train_mask, feature_cols].values,
        "y_train": df.loc[train_mask, "label"].values,
        "X_val": df.loc[val_mask, feature_cols].values,
        "y_val": df.loc[val_mask, "label"].values,
        "X_test": df.loc[test_mask, feature_cols].values,
        "y_test": df.loc[test_mask, "label"].values,
        "feature_cols": feature_cols,
    }


def _filter_rare_classes(parts: dict, min_count: int) -> dict:
    """Keep only classes with ≥ min_count samples in train. Drop those rows from val/test too."""
    counts = pd.Series(parts["y_train"]).value_counts()
    valid = set(counts[counts >= min_count].index)
    if len(valid) < 2:
        # fallback: keep top classes
        valid = set(counts.head(2).index)

    def filt(X, y):
        m = np.isin(y, list(valid))
        return X[m], y[m]

    out = dict(parts)
    out["X_train"], out["y_train"] = filt(parts["X_train"], parts["y_train"])
    out["X_val"], out["y_val"] = filt(parts["X_val"], parts["y_val"])
    out["X_test"], out["y_test"] = filt(parts["X_test"], parts["y_test"])
    out["valid_classes"] = sorted(valid)
    return out


def _normalize_clean(parts: dict) -> dict:
    """Replace inf/nan with 0, fit MinMax on train only, apply to all."""
    out = dict(parts)
    for k in ("X_train", "X_val", "X_test"):
        out[k] = np.where(np.isfinite(parts[k]), parts[k], 0.0)
    scaler = MinMaxScaler()
    out["X_train"] = scaler.fit_transform(out["X_train"])
    out["X_val"] = scaler.transform(out["X_val"])
    out["X_test"] = scaler.transform(out["X_test"])
    out["scaler"] = scaler
    return out


def _encode_labels(parts: dict) -> dict:
    le = LabelEncoder().fit(parts["y_train"])
    out = dict(parts)
    out["y_train"] = le.transform(parts["y_train"])
    # val/test may contain classes not seen in train; drop those rows
    val_mask = np.isin(parts["y_val"], le.classes_)
    test_mask = np.isin(parts["y_test"], le.classes_)
    out["X_val"] = parts["X_val"][val_mask]
    out["y_val"] = le.transform(parts["y_val"][val_mask])
    out["X_test"] = parts["X_test"][test_mask]
    out["y_test"] = le.transform(parts["y_test"][test_mask])
    out["label_encoder"] = le
    out["n_classes"] = int(len(le.classes_))
    return out


def _slice_to_features(parts: dict, names: list[str]) -> dict:
    """Slice X arrays down to the named features only."""
    fcols = parts["feature_cols"]
    idx = [fcols.index(n) for n in names if n in fcols]
    if len(idx) != len(names):
        missing = [n for n in names if n not in fcols]
        raise KeyError(f"Selected features missing from frame: {missing[:5]}...")
    out = dict(parts)
    out["X_train"] = parts["X_train"][:, idx]
    out["X_val"] = parts["X_val"][:, idx]
    out["X_test"] = parts["X_test"][:, idx]
    return out


def _ensure_chi2_safe(parts: dict) -> dict:
    """Chi2 needs non-negative inputs — already enforced by MinMaxScaler [0,1]."""
    return parts  # placeholder, kept for clarity


# ───────────────────────── stage runners ─────────────────────────

def stage_extract(dataset: str, limit: int | None) -> None:
    print(f"\n[{dataset}] === Stage 1: extract features ===")
    if dataset == "vehkaoja":
        ef.extract_vehkaoja(limit_subjects=limit)
    else:
        ef.extract_marinara(limit_subjects=limit)


def stage_split(dataset: str) -> dict:
    print(f"\n[{dataset}] === Stage 2: build inter-subject split ===")
    return sp.build_split(dataset)


def stage_select_features(dataset: str, feats: dict, split: dict) -> dict:
    """Run all 10 methods × 2 strategies. Cache results JSON per (strategy, method)."""
    out_dir = EXP_ROOT / dataset / "selected"
    out_dir.mkdir(parents=True, exist_ok=True)

    train = set(split["train_subjects"])
    val = set(split["val_subjects"])
    test = set(split["test_subjects"])

    # Build train-only X for each subset
    parts_full = _normalize_clean(_filter_rare_classes(
        _split_xy(feats["full"], train, val, test), MIN_SAMPLES_PER_CLASS,
    ))
    parts_static = _normalize_clean(_filter_rare_classes(
        _split_xy(feats["static"], train, val, test), 5,
    ))
    parts_dynamic = _normalize_clean(_filter_rare_classes(
        _split_xy(feats["dynamic"], train, val, test), 5,
    ))

    # Encode labels just for FS (numeric required)
    le_full = LabelEncoder().fit(parts_full["y_train"])
    y_full = le_full.transform(parts_full["y_train"])
    le_s = LabelEncoder().fit(parts_static["y_train"])
    y_s = le_s.transform(parts_static["y_train"])
    le_d = LabelEncoder().fit(parts_dynamic["y_train"])
    y_d = le_d.transform(parts_dynamic["y_train"])

    print(f"[{dataset}] FS train sizes — full: {len(y_full):,} | static: {len(y_s):,} | dynamic: {len(y_d):,}")

    # Some features may be in static/dynamic but not full (unlikely, but safety)
    # Use the COMMON feature set across all three frames so combined indexing works.
    common_cols = list(
        set(parts_full["feature_cols"]) & set(parts_static["feature_cols"]) & set(parts_dynamic["feature_cols"])
    )
    common_cols.sort()

    def _slice(parts, cols):
        idx = [parts["feature_cols"].index(c) for c in cols]
        return parts["X_train"][:, idx]

    Xf = _slice(parts_full, common_cols)
    Xs = _slice(parts_static, common_cols)
    Xd = _slice(parts_dynamic, common_cols)

    selected_cache: dict = {}
    for method in fs.METHODS:
        # strategy A — full
        path_a = out_dir / f"full_{method}.json"
        if path_a.exists():
            sel_a = json.loads(path_a.read_text())["selected"]
            print(f"[{dataset}] [CACHE] full_{method}.json")
        else:
            t0 = time.time()
            sel_a = fs.select(Xf, y_full, common_cols, method, K_FEATURES)
            path_a.write_text(json.dumps({"strategy": "full", "method": method,
                                          "selected": sel_a}, indent=2))
            print(f"[{dataset}] full_{method:35s} {len(sel_a)} feats  ({time.time()-t0:.1f}s)")

        # strategy B — split
        path_b = out_dir / f"split_{method}.json"
        if path_b.exists():
            sel_b_payload = json.loads(path_b.read_text())
            sel_b = sel_b_payload["combined"]
            print(f"[{dataset}] [CACHE] split_{method}.json")
        else:
            t0 = time.time()
            sel_b_payload = fs.select_split_strategy(
                Xs, y_s, Xd, y_d, common_cols, method, k_each=10, k_total=K_FEATURES,
            )
            sel_b_payload.update({"strategy": "split", "method": method})
            path_b.write_text(json.dumps(sel_b_payload, indent=2))
            sel_b = sel_b_payload["combined"]
            print(f"[{dataset}] split_{method:34s} {len(sel_b)} feats  ({time.time()-t0:.1f}s)")

        selected_cache[("full", method)] = sel_a
        selected_cache[("split", method)] = sel_b

    return {
        "selected": selected_cache,
        "common_cols": common_cols,
        "parts_full": parts_full,  # already normalized & filtered
    }


def _final_dataset(parts_full: dict) -> dict:
    """Encode labels on parts_full (already filtered + normalized). Returns ready-to-train dict."""
    return _encode_labels(parts_full)


def stage_dl_sweep(dataset: str, parts_ready: dict, sel_cache: dict, common_cols: list[str]) -> dict:
    """Sweep DL configs on (full, anova_f) cell. Return best (dim, lr) per model."""
    sweep_dir = EXP_ROOT / dataset / "dl_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    selected = sel_cache[("full", "anova_f")]
    parts = _slice_to_features(parts_ready, selected)
    n_classes = parts["n_classes"]

    best = {}
    for model_name in ("cnn", "lstm"):
        path = sweep_dir / f"{model_name}_sweep.json"
        if path.exists():
            payload = json.loads(path.read_text())
            best[model_name] = payload["best"]
            print(f"[{dataset}] [CACHE] {model_name}_sweep.json — best: {payload['best']}")
            continue

        results = []
        print(f"\n[{dataset}] DL SWEEP — {model_name.upper()} (9 configs)")
        for d in DL_DIMS:
            for lr in DL_LRS:
                t0 = time.time()
                fn = mdl.train_eval_cnn if model_name == "cnn" else mdl.train_eval_lstm
                try:
                    metrics = fn(
                        parts["X_train"], parts["y_train"],
                        parts["X_val"], parts["y_val"],
                        parts["X_test"], parts["y_test"],
                        n_classes=n_classes, dim=d, lr=lr,
                    )
                except Exception as exc:
                    print(f"  dim={d:3d} lr={lr:.0e}  ERROR: {exc}")
                    metrics = {"val_accuracy": 0.0, "val_f1_weighted": 0.0,
                               "test_accuracy": 0.0, "test_f1_weighted": 0.0,
                               "error": str(exc)}
                results.append({"dim": d, "lr": lr, **metrics, "secs": round(time.time() - t0, 1)})
                print(f"  dim={d:3d} lr={lr:.0e}  val_f1={metrics['val_f1_weighted']:.4f}  "
                      f"test_f1={metrics['test_f1_weighted']:.4f}  ({time.time()-t0:.1f}s)")

        # pick best by val_f1_weighted
        best_cfg = max(results, key=lambda r: r["val_f1_weighted"])
        payload = {
            "model": model_name, "dataset": dataset,
            "cell": "full+anova_f",
            "best": {"dim": best_cfg["dim"], "lr": best_cfg["lr"]},
            "all_runs": results,
        }
        path.write_text(json.dumps(payload, indent=2))
        best[model_name] = payload["best"]
        print(f"[{dataset}] {model_name.upper()} sweep saved → best dim={best_cfg['dim']} lr={best_cfg['lr']}")

    return best


def stage_main_grid(
    dataset: str, parts_ready: dict, sel_cache: dict, dl_best: dict,
) -> None:
    """Run RF, SVM, best-CNN, best-LSTM across all (strategy, method) cells."""
    res_dir = EXP_ROOT / dataset / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    n_classes = parts_ready["n_classes"]

    for strategy in ("full", "split"):
        for method in fs.METHODS:
            selected = sel_cache[(strategy, method)]
            if not selected:
                continue
            parts = _slice_to_features(parts_ready, selected)

            for model_name in ("rf", "svm", "cnn", "lstm"):
                out_path = res_dir / f"{model_name}_{strategy}_{method}.json"
                if out_path.exists():
                    print(f"[{dataset}] [CACHE] {out_path.name}")
                    continue

                t0 = time.time()
                hp = {}
                try:
                    if model_name == "rf":
                        m = mdl.train_eval_rf(
                            parts["X_train"], parts["y_train"],
                            parts["X_val"], parts["y_val"],
                            parts["X_test"], parts["y_test"],
                        )
                    elif model_name == "svm":
                        m = mdl.train_eval_svm(
                            parts["X_train"], parts["y_train"],
                            parts["X_val"], parts["y_val"],
                            parts["X_test"], parts["y_test"],
                        )
                    elif model_name == "cnn":
                        hp = dl_best["cnn"]
                        m = mdl.train_eval_cnn(
                            parts["X_train"], parts["y_train"],
                            parts["X_val"], parts["y_val"],
                            parts["X_test"], parts["y_test"],
                            n_classes=n_classes, dim=hp["dim"], lr=hp["lr"],
                        )
                    else:  # lstm
                        hp = dl_best["lstm"]
                        m = mdl.train_eval_lstm(
                            parts["X_train"], parts["y_train"],
                            parts["X_val"], parts["y_val"],
                            parts["X_test"], parts["y_test"],
                            n_classes=n_classes, dim=hp["dim"], lr=hp["lr"],
                        )
                    payload = {
                        "dataset": dataset, "model": model_name,
                        "hyperparams": hp,
                        "strategy": strategy, "feature_selection_method": method,
                        "n_classes": n_classes,
                        "n_features_selected": len(selected),
                        "selected_features": selected,
                        **m,
                    }
                    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
                    print(f"[{dataset}] {model_name:4s} {strategy:5s} {method:35s} "
                          f"val_f1={m['val_f1_weighted']:.4f}  test_f1={m['test_f1_weighted']:.4f}  "
                          f"({time.time()-t0:.1f}s)")
                except Exception as exc:
                    print(f"[{dataset}] {model_name} {strategy} {method}  ERROR: {exc}")
                    traceback.print_exc()


def stage_aggregate(dataset: str) -> None:
    res_dir = EXP_ROOT / dataset / "results"
    rows = []
    for jf in sorted(res_dir.glob("*.json")):
        d = json.loads(jf.read_text())
        rows.append({
            "dataset": d.get("dataset", dataset),
            "model": d["model"],
            "strategy": d["strategy"],
            "method": d["feature_selection_method"],
            "val_acc": d["val_accuracy"],
            "val_f1": d["val_f1_weighted"],
            "test_acc": d["test_accuracy"],
            "test_f1": d["test_f1_weighted"],
        })
    if not rows:
        return
    df = pd.DataFrame(rows)
    out = EXP_ROOT / dataset / "analysis" / "summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n[{dataset}] summary saved → {out}  ({len(df)} rows)")


def stage_cross_dataset() -> None:
    rows = []
    for ds in ("vehkaoja", "marinara"):
        f = EXP_ROOT / ds / "analysis" / "summary.csv"
        if f.exists():
            rows.append(pd.read_csv(f))
    if not rows:
        return
    df = pd.concat(rows, ignore_index=True)
    pivot = df.pivot_table(
        index=["dataset", "model", "method"],
        columns="strategy",
        values="test_f1",
    ).reset_index()
    if "split" in pivot.columns and "full" in pivot.columns:
        pivot["delta_split_minus_full"] = pivot["split"] - pivot["full"]
    out = EXP_ROOT / "cross_dataset_summary.csv"
    pivot.to_csv(out, index=False)
    print(f"\ncross-dataset summary → {out}")


# ───────────────────────── main ─────────────────────────

def run_dataset(dataset: str, limit: int | None) -> None:
    print(f"\n{'#' * 70}\n# DATASET: {dataset.upper()}\n{'#' * 70}")
    stage_extract(dataset, limit)
    split = stage_split(dataset)
    feats = _load_features(dataset)

    # FS stage requires train subjects; produces selected feature lists
    train, val, test = (set(split["train_subjects"]),
                        set(split["val_subjects"]),
                        set(split["test_subjects"]))
    sel_payload = stage_select_features(dataset, feats, split)
    sel_cache = sel_payload["selected"]

    # Build the final training dataset (full features, encoded labels)
    parts_ready = _final_dataset(sel_payload["parts_full"])
    print(f"[{dataset}] training set ready: train={len(parts_ready['y_train']):,}  "
          f"val={len(parts_ready['y_val']):,}  test={len(parts_ready['y_test']):,}  "
          f"classes={parts_ready['n_classes']}")

    dl_best = stage_dl_sweep(dataset, parts_ready, sel_cache, sel_payload["common_cols"])
    stage_main_grid(dataset, parts_ready, sel_cache, dl_best)
    stage_aggregate(dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["vehkaoja", "marinara", "both"], default="both")
    ap.add_argument("--smoke", type=int, default=None,
                    help="limit number of subjects (smoke test)")
    args = ap.parse_args()

    targets = ["vehkaoja", "marinara"] if args.dataset == "both" else [args.dataset]
    for ds in targets:
        run_dataset(ds, args.smoke)
    if args.dataset == "both":
        stage_cross_dataset()
    print("\n✓ done.")


if __name__ == "__main__":
    main()
