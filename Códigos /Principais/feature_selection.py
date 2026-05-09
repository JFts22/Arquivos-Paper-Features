#!/usr/bin/env python3
"""
Feature selection wrappers — 10 methods, full or static+dynamic strategies.

Each method takes (X, y, feature_names, k) and returns a list of selected
feature names. Logic adapted from pipeline_vehkaoja.py:199 with one
correction: FS sees ONLY train rows (caller's responsibility to filter).
"""
from __future__ import annotations

import warnings
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, chi2,
    VarianceThreshold, RFE, SequentialFeatureSelector,
)
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

METHODS = [
    "select_k_best",
    "anova_f",
    "mutual_information",
    "variance_threshold",
    "chi_squared",
    "xgboost",
    "random_forest",
    "lasso",
    "rfe",
    "sequential_feature_selection",
]


def _subsample(X: np.ndarray, y: np.ndarray, max_samples: int = 20_000, seed: int = 42):
    if X.shape[0] <= max_samples:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=max_samples, replace=False)
    return X[idx], y[idx]


def select(X: np.ndarray, y: np.ndarray, feature_names: list[str], method: str, k: int) -> list[str]:
    """Run a feature-selection method and return the names of the selected features."""
    k = min(k, X.shape[1])

    if method in ("select_k_best", "anova_f"):
        sel = SelectKBest(f_classif, k=k).fit(X, y)
        return [feature_names[i] for i in sel.get_support(indices=True)]

    if method == "mutual_information":
        sel = SelectKBest(mutual_info_classif, k=k).fit(X, y)
        return [feature_names[i] for i in sel.get_support(indices=True)]

    if method == "chi_squared":
        # X assumed already in [0, 1] from caller.
        sel = SelectKBest(chi2, k=k).fit(X, y)
        return [feature_names[i] for i in sel.get_support(indices=True)]

    if method == "variance_threshold":
        sel = VarianceThreshold(threshold=0.005).fit(X)
        mask = sel.get_support()
        cands = [feature_names[i] for i, m in enumerate(mask) if m]
        if len(cands) <= k:
            return cands
        variances = np.var(X[:, mask], axis=0)
        top_idx = np.argsort(variances)[::-1][:k]
        return [cands[i] for i in top_idx]

    if method == "xgboost":
        # No XGBoost installed (libomp issues on macOS) — substitute with ExtraTrees.
        Xs, ys = _subsample(X, y)
        clf = ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        clf.fit(Xs, ys)
        top_idx = np.argsort(clf.feature_importances_)[::-1][:k]
        return [feature_names[i] for i in top_idx]

    if method == "random_forest":
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        top_idx = np.argsort(rf.feature_importances_)[::-1][:k]
        return [feature_names[i] for i in top_idx]

    if method == "lasso":
        Xs, ys = _subsample(X, y)
        lr = LogisticRegression(
            penalty="l1", solver="saga", C=0.1,
            max_iter=500, random_state=42, n_jobs=-1,
        )
        lr.fit(Xs, ys)
        importance = (
            np.abs(lr.coef_).mean(axis=0)
            if lr.coef_.ndim > 1
            else np.abs(lr.coef_[0])
        )
        top_idx = np.argsort(importance)[::-1][:k]
        return [feature_names[i] for i in top_idx]

    if method == "rfe":
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        sel = RFE(estimator=rf, n_features_to_select=k, step=0.1).fit(X, y)
        return [feature_names[i] for i in sel.get_support(indices=True)]

    if method == "sequential_feature_selection":
        if X.shape[1] > 100:
            var_rank = np.argsort(np.var(X, axis=0))[::-1][:100]
            X_pre = X[:, var_rank]
            fn_pre = [feature_names[i] for i in var_rank]
        else:
            X_pre, fn_pre = X, feature_names
        k_sfs = min(10, k)
        rf = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1)
        sel = SequentialFeatureSelector(
            rf, n_features_to_select=k_sfs,
            direction="forward", cv=3, n_jobs=-1,
        ).fit(X_pre, y)
        return [fn_pre[i] for i in sel.get_support(indices=True)]

    raise ValueError(f"Unknown method: {method}")


def select_hybrid_strategy(
    X_static: np.ndarray, y_static: np.ndarray,
    X_dynamic: np.ndarray, y_dynamic: np.ndarray,
    X_full: np.ndarray, y_full: np.ndarray,
    feature_names: list[str],
    method: str,
    k_static: int = 5, k_dynamic: int = 5, k_full: int = 10,
    k_total: int = 20,
) -> dict:
    """Strategy C (hybrid): k_full from full + k_static from static + k_dynamic from dynamic.

    Combined order: full first (preserves global ranking), then static, then dynamic.
    Dedup; top-up from extended full FS if union < k_total.
    """
    full_sel = select(X_full, y_full, feature_names, method, k_full)
    static_sel = select(X_static, y_static, feature_names, method, k_static)
    dynamic_sel = select(X_dynamic, y_dynamic, feature_names, method, k_dynamic)

    seen: set[str] = set()
    combined: list[str] = []
    for source in (full_sel, static_sel, dynamic_sel):
        for f in source:
            if f not in seen:
                seen.add(f)
                combined.append(f)

    # Top-up by extending full FS if union shorter than k_total
    if len(combined) < k_total:
        extras = select(X_full, y_full, feature_names, method,
                        min(k_total + 10, len(feature_names)))
        for f in extras:
            if f not in seen:
                seen.add(f)
                combined.append(f)
            if len(combined) >= k_total:
                break

    return {
        "full_features": full_sel,
        "static_features": static_sel,
        "dynamic_features": dynamic_sel,
        "combined": combined[:k_total],
    }


def select_interclass_strategy(
    X_static: np.ndarray, y_static: np.ndarray,
    X_dynamic: np.ndarray, y_dynamic: np.ndarray,
    X_full: np.ndarray, y_full_binary: np.ndarray,
    feature_names: list[str],
    method: str,
    k_static: int = 5, k_dynamic: int = 5, k_full: int = 10,
    k_total: int = 20,
) -> dict:
    """Strategy D (interclass): k_full from full with BINARY target (static vs dynamic)
    + k_static intra-static + k_dynamic intra-dynamic.

    Rationale: the 10 features from the full dataset now identify what discriminates
    *between* regimes (inter-class), complementing the 5+5 features that discriminate
    *within* each regime (intra-class).

    y_full_binary must be a 0/1 (or 2-valued) array aligned with X_full rows,
    where 0 = static window, 1 = dynamic window.
    """
    full_sel = select(X_full, y_full_binary, feature_names, method, k_full)
    static_sel = select(X_static, y_static, feature_names, method, k_static)
    dynamic_sel = select(X_dynamic, y_dynamic, feature_names, method, k_dynamic)

    seen: set[str] = set()
    combined: list[str] = []
    for source in (full_sel, static_sel, dynamic_sel):
        for f in source:
            if f not in seen:
                seen.add(f)
                combined.append(f)

    if len(combined) < k_total:
        extras = select(X_full, y_full_binary, feature_names, method,
                        min(k_total + 10, len(feature_names)))
        for f in extras:
            if f not in seen:
                seen.add(f)
                combined.append(f)
            if len(combined) >= k_total:
                break

    return {
        "full_features": full_sel,
        "static_features": static_sel,
        "dynamic_features": dynamic_sel,
        "combined": combined[:k_total],
    }


def select_split_strategy(
    X_static: np.ndarray, y_static: np.ndarray,
    X_dynamic: np.ndarray, y_dynamic: np.ndarray,
    feature_names: list[str],
    method: str,
    k_each: int = 10,
    k_total: int = 20,
) -> dict:
    """Strategy B: select k_each from static + k_each from dynamic, dedup, top up.

    Returns dict with static_features, dynamic_features, combined (unique up to k_total).
    """
    static_sel = select(X_static, y_static, feature_names, method, k_each)
    dynamic_sel = select(X_dynamic, y_dynamic, feature_names, method, k_each)

    # Build ordered combined list: interleave preserving order, dedup
    seen = set()
    combined: list[str] = []
    for s, d in zip(static_sel, dynamic_sel):
        if s not in seen:
            seen.add(s)
            combined.append(s)
        if d not in seen:
            seen.add(d)
            combined.append(d)
    # Append any leftovers (in case lengths differ)
    for f in static_sel + dynamic_sel:
        if f not in seen:
            seen.add(f)
            combined.append(f)

    # If combined < k_total (overlap reduced count), top up by re-running on static
    # with a larger k until we hit k_total or run out of features.
    if len(combined) < k_total:
        extras = select(X_static, y_static, feature_names, method, min(k_total + k_each, len(feature_names)))
        for f in extras:
            if f not in seen:
                seen.add(f)
                combined.append(f)
            if len(combined) >= k_total:
                break

    combined = combined[:k_total]
    return {
        "static_features": static_sel,
        "dynamic_features": dynamic_sel,
        "combined": combined,
    }
