#!/usr/bin/env python3
"""
Extended FS experiment on Vehkaoja: adds 4 new methods to the existing
hybrid + interclass experiments.

New methods (this script):
    - mutual_information
    - select_k_best
    - xgboost (ExtraTrees substitute)
    - sequential_feature_selection

Each new method is computed for the 2 strategies that need new data:
    - hybrid     → results land in experiment_hybrid_vehkaoja/
    - interclass → results land in experiment_interclass_vehkaoja/

Existing methods (random_forest, lasso, chi_squared) are NOT recomputed.
Strategies `full` and `split` are NOT recomputed — they already exist in
experiment_dual/vehkaoja/results/ for all 10 methods.
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
from extract_features import VEHKAOJA_STATIC, VEHKAOJA_DYNAMIC
from pipeline_dual import (
    _split_xy, _filter_rare_classes, _normalize_clean,
    _encode_labels, _slice_to_features,
    MIN_SAMPLES_PER_CLASS,
)

DATASET = "vehkaoja"
SOURCE = Path("/Users/joanafontes/Documents/Datasets/experiment_dual") / DATASET
DEST_HYBRID = Path("/Users/joanafontes/Documents/Datasets/experiment_hybrid_vehkaoja")
DEST_INTER = Path("/Users/joanafontes/Documents/Datasets/experiment_interclass_vehkaoja")

NEW_METHODS = ["select_k_best", "xgboost", "mutual_information", "sequential_feature_selection"]
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


def _binarize(y_multiclass: np.ndarray) -> np.ndarray:
    out = np.empty(len(y_multiclass), dtype=np.int64)
    for i, lbl in enumerate(y_multiclass):
        if lbl in VEHKAOJA_STATIC:
            out[i] = 0
        elif lbl in VEHKAOJA_DYNAMIC:
            out[i] = 1
        else:
            out[i] = -1
    return out


def stage_select(parts_full, parts_static, parts_dynamic, common: list[str]) -> dict:
    """Run FS for the 4 new methods × 2 strategies (hybrid, interclass).
    Existing 10 cells (random_forest, chi_squared, lasso × hybrid/interclass)
    are kept as-is via cache check."""
    Xf, y_full_raw = _train_subset(parts_full, common)
    Xs, y_static_raw = _train_subset(parts_static, common)
    Xd, y_dynamic_raw = _train_subset(parts_dynamic, common)

    y_full_multi = LabelEncoder().fit_transform(y_full_raw)
    y_s = LabelEncoder().fit_transform(y_static_raw)
    y_d = LabelEncoder().fit_transform(y_dynamic_raw)

    y_full_bin = _binarize(y_full_raw)
    keep = y_full_bin >= 0
    Xf_bin, y_full_bin_clean = Xf[keep], y_full_bin[keep]
    n_static = int(np.sum(y_full_bin_clean == 0))
    n_dynamic = int(np.sum(y_full_bin_clean == 1))

    print(f"FS train sizes — full multi: {len(y_full_multi):,} | "
          f"static intra: {len(y_s):,} | dynamic intra: {len(y_d):,} | "
          f"binary: {len(y_full_bin_clean):,} (static={n_static:,}, dyn={n_dynamic:,})")

    selected: dict[tuple[str, str], list[str]] = {}
    for method in NEW_METHODS:
        # ── HYBRID strategy ──────────────────────────────────────────────
        path_h = DEST_HYBRID / "selected" / f"hybrid_{method}.json"
        if path_h.exists():
            sel_h_payload = json.loads(path_h.read_text())
            sel_h = sel_h_payload["combined"]
            print(f"[CACHE] hybrid_{method}.json")
        else:
            t0 = time.time()
            sel_h_payload = fs.select_hybrid_strategy(
                Xs, y_s, Xd, y_d, Xf, y_full_multi, common, method,
                k_static=5, k_dynamic=5, k_full=10, k_total=K_TOTAL,
            )
            sel_h_payload.update({"strategy": "hybrid", "method": method})
            path_h.parent.mkdir(parents=True, exist_ok=True)
            path_h.write_text(json.dumps(sel_h_payload, indent=2))
            sel_h = sel_h_payload["combined"]
            print(f"hybrid_{method:35s}  {len(sel_h)} feats  ({time.time()-t0:.1f}s)",
                  flush=True)

        # Strategy A — full (already exists)
        path_full_h = DEST_HYBRID / "selected" / f"full_{method}.json"
        if not path_full_h.exists():
            t0 = time.time()
            sel_full_h = fs.select(Xf, y_full_multi, common, method, K_TOTAL)
            path_full_h.write_text(json.dumps({"strategy": "full", "method": method,
                                               "selected": sel_full_h}, indent=2))
            print(f"full_{method:37s}  {len(sel_full_h)} feats  ({time.time()-t0:.1f}s)",
                  flush=True)
            selected[("full", method)] = sel_full_h
        else:
            selected[("full", method)] = json.loads(path_full_h.read_text())["selected"]

        # ── INTERCLASS strategy ──────────────────────────────────────────
        path_i = DEST_INTER / "selected" / f"interclass_{method}.json"
        if path_i.exists():
            sel_i_payload = json.loads(path_i.read_text())
            sel_i = sel_i_payload["combined"]
            print(f"[CACHE] interclass_{method}.json")
        else:
            t0 = time.time()
            sel_i_payload = fs.select_interclass_strategy(
                Xs, y_s, Xd, y_d, Xf_bin, y_full_bin_clean, common, method,
                k_static=5, k_dynamic=5, k_full=10, k_total=K_TOTAL,
            )
            sel_i_payload.update({"strategy": "interclass", "method": method,
                                  "binary_split": {"static": n_static,
                                                   "dynamic": n_dynamic}})
            path_i.parent.mkdir(parents=True, exist_ok=True)
            path_i.write_text(json.dumps(sel_i_payload, indent=2))
            sel_i = sel_i_payload["combined"]
            print(f"interclass_{method:31s}  {len(sel_i)} feats  ({time.time()-t0:.1f}s)",
                  flush=True)

        # Strategy A — full for interclass dir
        path_full_i = DEST_INTER / "selected" / f"full_{method}.json"
        if not path_full_i.exists():
            # Same as the hybrid full FS (multiclass on full).
            sel_full_h_path = DEST_HYBRID / "selected" / f"full_{method}.json"
            if sel_full_h_path.exists():
                path_full_i.write_text(sel_full_h_path.read_text())

        selected[("hybrid", method)] = sel_h
        selected[("interclass", method)] = sel_i

    return selected


def stage_evaluate(parts_ready: dict, selected: dict) -> None:
    n_classes = parts_ready["n_classes"]

    for method in NEW_METHODS:
        for strategy in ("full", "hybrid", "interclass"):
            sel = selected[(strategy, method)]
            parts = _slice_to_features(parts_ready, sel)
            res_dir = (DEST_HYBRID if strategy in ("full", "hybrid") else DEST_INTER) / "results"
            res_dir.mkdir(parents=True, exist_ok=True)

            for model_name in ("rf", "svm", "cnn", "lstm"):
                out = res_dir / f"{model_name}_{strategy}_{method}.json"
                if out.exists():
                    print(f"[CACHE] {strategy}/{out.name}")
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
                    print(f"{model_name:4s} {strategy:10s} {method:32s}  "
                          f"val_f1={m['val_f1_weighted']:.4f}  "
                          f"test_f1={m['test_f1_weighted']:.4f}  "
                          f"({time.time()-t0:.1f}s)", flush=True)
                except Exception as exc:
                    print(f"{model_name} {strategy} {method}  ERROR: {exc}",
                          flush=True)
                    traceback.print_exc()


def main() -> None:
    print(f"Extended experiment on {DATASET}")
    print(f"New methods: {NEW_METHODS}")
    print(f"Targets: hybrid → {DEST_HYBRID.name}/  ·  interclass → {DEST_INTER.name}/")

    feats = _load_features()
    split = json.loads((SOURCE / "splits.json").read_text())
    parts_full, parts_static, parts_dynamic, common = _build_parts(feats, split)

    print("\n=== Stage A: feature selection ===")
    selected = stage_select(parts_full, parts_static, parts_dynamic, common)

    print("\n=== Stage B: model evaluation ===")
    parts_ready = _encode_labels(parts_full)
    print(f"Training set: train={len(parts_ready['y_train']):,}  "
          f"val={len(parts_ready['y_val']):,}  test={len(parts_ready['y_test']):,}  "
          f"classes={parts_ready['n_classes']}")
    stage_evaluate(parts_ready, selected)

    print("\n✓ done.")


if __name__ == "__main__":
    main()
