#!/usr/bin/env python3
"""
Interclass feature-selection experiment on Vehkaoja.

Strategy under test (INTERCLASS): 10 features from the full dataset using a
BINARY static-vs-dynamic target + 5 features intra-static (multiclass)
+ 5 features intra-dynamic (multiclass). Total = 20 features (with dedup).

Compared against the existing FULL baseline (20 multiclass) — the new strategy's
purpose is to capture features that separate regimes (inter-class) on top of the
features that separate behaviours within each regime (intra-class).

Methods: random_forest, chi_squared, lasso. Models: RF, SVM, CNN, LSTM.
Reuses features, split, and DL hyperparams from experiment_dual/.
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
DEST = Path("/Users/joanafontes/Documents/Datasets/experiment_interclass_vehkaoja")

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
    """Build normalized/filtered (X, y) for full, static, dynamic. Train-only FS."""
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
    """Map each Vehkaoja behaviour label → 0 (static) or 1 (dynamic)."""
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
    out_dir = DEST / "selected"
    out_dir.mkdir(parents=True, exist_ok=True)

    Xf, y_full_raw = _train_subset(parts_full, common)
    Xs, y_static_raw = _train_subset(parts_static, common)
    Xd, y_dynamic_raw = _train_subset(parts_dynamic, common)

    # Multiclass labels (used by full baseline & by static/dynamic intra-class FS)
    y_full_multi = LabelEncoder().fit_transform(y_full_raw)
    y_s = LabelEncoder().fit_transform(y_static_raw)
    y_d = LabelEncoder().fit_transform(y_dynamic_raw)

    # Binary label for the inter-class FS on full
    y_full_bin = _binarize(y_full_raw)
    keep = y_full_bin >= 0
    Xf_bin, y_full_bin_clean = Xf[keep], y_full_bin[keep]

    n_static = int(np.sum(y_full_bin_clean == 0))
    n_dynamic = int(np.sum(y_full_bin_clean == 1))
    print(f"FS train sizes — full multi: {len(y_full_multi):,} | "
          f"full binary: {len(y_full_bin_clean):,}  "
          f"(static={n_static:,}, dynamic={n_dynamic:,}) | "
          f"static intra: {len(y_s):,} | dynamic intra: {len(y_d):,}")

    selected: dict[tuple[str, str], list[str]] = {}
    for method in METHODS:
        # Strategy A — full (20 multiclass)
        path_a = out_dir / f"full_{method}.json"
        if path_a.exists():
            sel_a = json.loads(path_a.read_text())["selected"]
            print(f"[CACHE] full_{method}.json")
        else:
            t0 = time.time()
            sel_a = fs.select(Xf, y_full_multi, common, method, K_TOTAL)
            path_a.write_text(json.dumps({"strategy": "full", "method": method,
                                          "selected": sel_a}, indent=2))
            print(f"full_{method:25s}  {len(sel_a)} feats  ({time.time()-t0:.1f}s)")

        # Strategy D — interclass (10 binary full + 5 static + 5 dynamic)
        path_d = out_dir / f"interclass_{method}.json"
        if path_d.exists():
            sel_d_payload = json.loads(path_d.read_text())
            sel_d = sel_d_payload["combined"]
            print(f"[CACHE] interclass_{method}.json")
        else:
            t0 = time.time()
            sel_d_payload = fs.select_interclass_strategy(
                Xs, y_s, Xd, y_d, Xf_bin, y_full_bin_clean,
                common, method,
                k_static=5, k_dynamic=5, k_full=10, k_total=K_TOTAL,
            )
            sel_d_payload.update({"strategy": "interclass", "method": method,
                                  "binary_split": {"static": n_static,
                                                   "dynamic": n_dynamic}})
            path_d.write_text(json.dumps(sel_d_payload, indent=2))
            sel_d = sel_d_payload["combined"]
            print(f"interclass_{method:21s}  {len(sel_d)} feats  ({time.time()-t0:.1f}s)")

        selected[("full", method)] = sel_a
        selected[("interclass", method)] = sel_d

    return selected


def stage_evaluate(parts_ready: dict, selected: dict) -> None:
    res_dir = DEST / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    n_classes = parts_ready["n_classes"]

    for strategy in ("full", "interclass"):
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
                    print(f"{model_name:4s} {strategy:10s} {method:18s}  "
                          f"val_f1={m['val_f1_weighted']:.4f}  "
                          f"test_f1={m['test_f1_weighted']:.4f}  "
                          f"({time.time()-t0:.1f}s)")
                except Exception as exc:
                    print(f"{model_name} {strategy} {method}  ERROR: {exc}")
                    traceback.print_exc()


def stage_aggregate() -> None:
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


def stage_compare_4way() -> pd.DataFrame:
    """Compare 4 strategies: full / split / hybrid / interclass."""
    rows = []
    # full + interclass (this experiment)
    for jf in sorted((DEST / "results").glob("*.json")):
        d = json.loads(jf.read_text())
        rows.append({"strategy": d["strategy"], "model": d["model"],
                     "method": d["feature_selection_method"],
                     "test_f1": d["test_f1_weighted"]})
    # split (from experiment_dual)
    for jf in (SOURCE / "results").glob("*_split_*.json"):
        d = json.loads(jf.read_text())
        if d["feature_selection_method"] in METHODS:
            rows.append({"strategy": "split", "model": d["model"],
                         "method": d["feature_selection_method"],
                         "test_f1": d["test_f1_weighted"]})
    # hybrid (from experiment_hybrid_vehkaoja)
    hyb_dir = Path("/Users/joanafontes/Documents/Datasets/experiment_hybrid_vehkaoja/results")
    for jf in hyb_dir.glob("*_hybrid_*.json"):
        d = json.loads(jf.read_text())
        if d["feature_selection_method"] in METHODS:
            rows.append({"strategy": "hybrid", "model": d["model"],
                         "method": d["feature_selection_method"],
                         "test_f1": d["test_f1_weighted"]})

    df = pd.DataFrame(rows)
    piv = df.pivot_table(index=["model", "method"], columns="strategy",
                         values="test_f1").reset_index()
    for col in ("split", "hybrid", "interclass"):
        if col in piv.columns and "full" in piv.columns:
            piv[f"Δ_{col}_vs_full"] = piv[col] - piv["full"]
    out = DEST / "analysis" / "comparison_4way.csv"
    piv.to_csv(out, index=False)
    print(f"4-way comparison saved → {out}\n")
    print(piv.round(4).to_string(index=False))
    return piv


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"Interclass experiment on {DATASET} → {DEST}")
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
    stage_compare_4way()
    print("\n✓ done.")


if __name__ == "__main__":
    main()
