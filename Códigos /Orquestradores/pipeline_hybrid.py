#!/usr/bin/env python3
"""
Hybrid feature-selection experiment on Vehkaoja.

Strategy under test (HYBRID): 5 features from static + 5 from dynamic + 10 from
full = 20 total (with dedup + top-up from full).
Baseline (FULL): 20 features from full dataset.

Methods evaluated: random_forest, chi_squared, lasso.
Models: RF, SVM, CNN, LSTM (DL hyperparams reused from experiment_dual sweep).

This script reuses features, split, and DL hyperparameters from the existing
experiment_dual/vehkaoja/ artifacts, so it does NOT re-extract or re-sweep.
Outputs go to experiment_hybrid_vehkaoja/.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import feature_selection as fs
import models as mdl
from pipeline_dual import (
    _split_xy, _filter_rare_classes, _normalize_clean,
    _encode_labels, _slice_to_features,
    MIN_SAMPLES_PER_CLASS,
)

DATASET = "vehkaoja"
SOURCE = Path("/Users/joanafontes/Documents/Datasets/experiment_dual") / DATASET
DEST = Path("/Users/joanafontes/Documents/Datasets/experiment_hybrid_vehkaoja")

METHODS = ["random_forest", "chi_squared", "lasso"]
K_TOTAL = 20

DL_BEST = {
    "cnn": json.loads((SOURCE / "dl_sweep" / "cnn_sweep.json").read_text())["best"],
    "lstm": json.loads((SOURCE / "dl_sweep" / "lstm_sweep.json").read_text())["best"],
}


def _load_features() -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_csv(SOURCE / "features" / f"features_{name}.csv")
        for name in ("full", "static", "dynamic")
    }


def _build_parts(feats: dict, split: dict) -> tuple[dict, dict, dict, list[str]]:
    """Build normalized/filtered (X,y) dicts for full, static, dynamic — TRAIN-only FS."""
    train = set(split["train_subjects"])
    val = set(split["val_subjects"])
    test = set(split["test_subjects"])

    parts_full = _normalize_clean(_filter_rare_classes(
        _split_xy(feats["full"], train, val, test), MIN_SAMPLES_PER_CLASS,
    ))
    parts_static = _normalize_clean(_filter_rare_classes(
        _split_xy(feats["static"], train, val, test), 5,
    ))
    parts_dynamic = _normalize_clean(_filter_rare_classes(
        _split_xy(feats["dynamic"], train, val, test), 5,
    ))

    common = sorted(set(parts_full["feature_cols"])
                    & set(parts_static["feature_cols"])
                    & set(parts_dynamic["feature_cols"]))
    return parts_full, parts_static, parts_dynamic, common


def _train_subset(parts: dict, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    idx = [parts["feature_cols"].index(c) for c in cols]
    return parts["X_train"][:, idx], parts["y_train"]


def stage_select(parts_full, parts_static, parts_dynamic, common: list[str]) -> dict:
    out_dir = DEST / "selected"
    out_dir.mkdir(parents=True, exist_ok=True)

    Xf, y_full_raw = _train_subset(parts_full, common)
    Xs, y_static_raw = _train_subset(parts_static, common)
    Xd, y_dynamic_raw = _train_subset(parts_dynamic, common)

    y_full = LabelEncoder().fit_transform(y_full_raw)
    y_s = LabelEncoder().fit_transform(y_static_raw)
    y_d = LabelEncoder().fit_transform(y_dynamic_raw)

    print(f"FS train sizes — full: {len(y_full):,} | static: {len(y_s):,} | dynamic: {len(y_d):,}")

    selected: dict[tuple[str, str], list[str]] = {}
    for method in METHODS:
        # Strategy A — full (20)
        path_a = out_dir / f"full_{method}.json"
        if path_a.exists():
            sel_a = json.loads(path_a.read_text())["selected"]
            print(f"[CACHE] full_{method}.json")
        else:
            t0 = time.time()
            sel_a = fs.select(Xf, y_full, common, method, K_TOTAL)
            path_a.write_text(json.dumps({"strategy": "full", "method": method,
                                          "selected": sel_a}, indent=2))
            print(f"full_{method:25s}  {len(sel_a)} feats  ({time.time()-t0:.1f}s)")

        # Strategy C — hybrid (10 full + 5 static + 5 dynamic)
        path_c = out_dir / f"hybrid_{method}.json"
        if path_c.exists():
            sel_c_payload = json.loads(path_c.read_text())
            sel_c = sel_c_payload["combined"]
            print(f"[CACHE] hybrid_{method}.json")
        else:
            t0 = time.time()
            sel_c_payload = fs.select_hybrid_strategy(
                Xs, y_s, Xd, y_d, Xf, y_full, common, method,
                k_static=5, k_dynamic=5, k_full=10, k_total=K_TOTAL,
            )
            sel_c_payload.update({"strategy": "hybrid", "method": method})
            path_c.write_text(json.dumps(sel_c_payload, indent=2))
            sel_c = sel_c_payload["combined"]
            print(f"hybrid_{method:23s}  {len(sel_c)} feats  ({time.time()-t0:.1f}s)")

        selected[("full", method)] = sel_a
        selected[("hybrid", method)] = sel_c

    return selected


def stage_evaluate(parts_ready: dict, selected: dict) -> None:
    res_dir = DEST / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    n_classes = parts_ready["n_classes"]

    for strategy in ("full", "hybrid"):
        for method in METHODS:
            sel = selected[(strategy, method)]
            parts = _slice_to_features(parts_ready, sel)
            for model_name in ("rf", "svm", "cnn", "lstm"):
                out = res_dir / f"{model_name}_{strategy}_{method}.json"
                if out.exists():
                    print(f"[CACHE] {out.name}")
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
                        hp = DL_BEST["cnn"]
                        m = mdl.train_eval_cnn(
                            parts["X_train"], parts["y_train"],
                            parts["X_val"], parts["y_val"],
                            parts["X_test"], parts["y_test"],
                            n_classes=n_classes, dim=hp["dim"], lr=hp["lr"],
                        )
                    else:
                        hp = DL_BEST["lstm"]
                        m = mdl.train_eval_lstm(
                            parts["X_train"], parts["y_train"],
                            parts["X_val"], parts["y_val"],
                            parts["X_test"], parts["y_test"],
                            n_classes=n_classes, dim=hp["dim"], lr=hp["lr"],
                        )

                    payload = {
                        "dataset": DATASET, "model": model_name, "hyperparams": hp,
                        "strategy": strategy, "feature_selection_method": method,
                        "n_classes": n_classes,
                        "n_features_selected": len(sel),
                        "selected_features": sel,
                        **m,
                    }
                    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
                    print(f"{model_name:4s} {strategy:6s} {method:18s}  "
                          f"val_f1={m['val_f1_weighted']:.4f}  "
                          f"test_f1={m['test_f1_weighted']:.4f}  "
                          f"({time.time()-t0:.1f}s)")
                except Exception as exc:
                    print(f"{model_name} {strategy} {method}  ERROR: {exc}")
                    traceback.print_exc()


def stage_aggregate() -> pd.DataFrame:
    rows = []
    for jf in sorted((DEST / "results").glob("*.json")):
        d = json.loads(jf.read_text())
        rows.append({
            "model": d["model"], "strategy": d["strategy"],
            "method": d["feature_selection_method"],
            "val_acc": d["val_accuracy"], "val_f1": d["val_f1_weighted"],
            "test_acc": d["test_accuracy"], "test_f1": d["test_f1_weighted"],
        })
    df = pd.DataFrame(rows)
    out = DEST / "analysis" / "summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nsummary saved → {out}  ({len(df)} rows)")
    return df


def stage_compare_with_split() -> pd.DataFrame:
    """Pull `split` results for the same 3 methods × 4 models from experiment_dual,
    and build a 3-strategy comparison: full / hybrid / split."""
    rows = []
    # full + hybrid (this experiment)
    for jf in sorted((DEST / "results").glob("*.json")):
        d = json.loads(jf.read_text())
        rows.append({"strategy": d["strategy"], "model": d["model"],
                     "method": d["feature_selection_method"],
                     "test_f1": d["test_f1_weighted"], "test_acc": d["test_accuracy"]})
    # split (existing experiment_dual)
    for jf in (SOURCE / "results").glob("*_split_*.json"):
        d = json.loads(jf.read_text())
        if d["feature_selection_method"] in METHODS:
            rows.append({"strategy": "split", "model": d["model"],
                         "method": d["feature_selection_method"],
                         "test_f1": d["test_f1_weighted"], "test_acc": d["test_accuracy"]})

    df = pd.DataFrame(rows)
    piv = df.pivot_table(index=["model", "method"], columns="strategy", values="test_f1").reset_index()
    if "full" in piv.columns and "hybrid" in piv.columns:
        piv["Δ_hybrid_vs_full"] = piv["hybrid"] - piv["full"]
    if "full" in piv.columns and "split" in piv.columns:
        piv["Δ_split_vs_full"] = piv["split"] - piv["full"]
    out = DEST / "analysis" / "comparison_3way.csv"
    piv.to_csv(out, index=False)
    print(f"3-way comparison saved → {out}")
    print()
    print(piv.round(4).to_string(index=False))
    return piv


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"Hybrid experiment on {DATASET} → {DEST}")
    print(f"Methods: {METHODS}")
    print(f"DL best configs: {DL_BEST}")

    feats = _load_features()
    split = json.loads((SOURCE / "splits.json").read_text())
    parts_full, parts_static, parts_dynamic, common = _build_parts(feats, split)

    selected = stage_select(parts_full, parts_static, parts_dynamic, common)

    parts_ready = _encode_labels(parts_full)
    print(f"Training set ready: train={len(parts_ready['y_train']):,}  "
          f"val={len(parts_ready['y_val']):,}  test={len(parts_ready['y_test']):,}  "
          f"classes={parts_ready['n_classes']}")

    stage_evaluate(parts_ready, selected)
    stage_aggregate()
    stage_compare_with_split()
    print("\n✓ done.")


if __name__ == "__main__":
    main()
