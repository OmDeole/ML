"""Model fitting utilities (Random Forest = Group 10's assigned model)."""

from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

RF_DEFAULTS = dict(
    n_estimators=100,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
    n_jobs=-1,
    random_state=0,
)


def build_model(kind: str, **params):
    if kind == "rf":
        p = dict(RF_DEFAULTS)
        p.update(params)
        return RandomForestClassifier(**p)
    if kind == "xgb":
        from xgboost import XGBClassifier
        p = dict(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                 colsample_bytree=0.8, tree_method="hist", eval_metric="auc",
                 n_jobs=-1, random_state=0)
        p.update(params)
        return XGBClassifier(**p)
    if kind == "logit":
        from sklearn.linear_model import LogisticRegression
        p = dict(max_iter=1000, n_jobs=-1)
        p.update(params)
        return LogisticRegression(**p)
    raise ValueError(kind)


def fit_score(kind: str, Xtr, ytr, Xva, yva, Xte, yte, p_true_te, **params):
    """Fit and return AUCs + timing for one model/dataset/imputation combo."""
    model = build_model(kind, **params)
    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    fit_s = time.perf_counter() - t0

    out = {"fit_seconds": fit_s}
    for tag, X, y in (("train", Xtr, ytr), ("val", Xva, yva), ("test", Xte, yte)):
        out[f"{tag}_auc"] = float(roc_auc_score(y, model.predict_proba(X)[:, 1]))
    out["oracle_auc"] = float(roc_auc_score(yte, p_true_te))
    return out, model
