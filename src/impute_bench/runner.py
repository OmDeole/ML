"""Experiment runner: data -> imputation -> model -> metrics row."""

from __future__ import annotations

import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from . import imputers as I
from .modeling import fit_score


def prepare(cfg: D.DataConfig, seed_offset: int):
    ds = D.generate_gam_global(cfg, seed_offset=seed_offset)
    ohe = D.design_matrix(ds, cfg)
    names = ds.feature_names + [f"seg_{k}" for k in range(1, cfg.n_segments)]
    Xm = np.hstack([ds.X, ohe])
    Xc = np.hstack([ds.X_clean, ohe])
    idx = ds.split(cfg.train_frac, cfg.val_frac)
    return ds, Xm, Xc, names, idx


def run_cell(cfg: D.DataConfig, method_key: str, model_kind: str,
             seed_offset: int, model_params: dict | None = None,
             imputer_kw: dict | None = None) -> dict:
    """One (dataset, method, seed) cell."""
    model_params = model_params or {}
    imputer_kw = imputer_kw or {}
    ds, Xm, Xc, names, idx = prepare(cfg, seed_offset)
    tr, va, te = idx["train"], idx["val"], idx["test"]

    row = {
        "dataset": "gam_global", "method_key": method_key, "model": model_kind,
        "seed": seed_offset, "n_rows": cfg.n_rows, "missing_rate": cfg.missing_rate,
        "mechanism": cfg.missing_mechanism,
        "missing_features": ",".join(cfg.missing_features),
    }

    if method_key == "none":                      # control: no missingness at all
        Xtr, Xva, Xte = Xc[tr], Xc[va], Xc[te]
        row.update(method="No Missing", fit_seconds_imp=0.0,
                   transform_seconds_imp=0.0, peak_mem_mb_imp=0.0, n_output_cols=Xc.shape[1])
    elif method_key == "oracle":                  # true probability, no model
        row.update(method="Oracle", test_auc=float(np.nan), val_auc=np.nan,
                   train_auc=np.nan, oracle_auc=float(
                       __import__("sklearn.metrics", fromlist=["roc_auc_score"])
                       .roc_auc_score(ds.y[te], ds.p_true[te])),
                   fit_seconds=np.nan, fit_seconds_imp=0.0)
        return row
    else:
        imp = I.get_imputer(method_key, [names.index(f) for f in cfg.missing_features],
                            **imputer_kw)
        t0 = time.perf_counter()
        imp.fit(Xm[tr], ds.y[tr])                  # fit on TRAIN ONLY (leak-free)
        Xtr = imp.transform(Xm[tr]); Xva = imp.transform(Xm[va]); Xte = imp.transform(Xm[te])
        row.update(method=imp.name,
                   fit_seconds_imp=imp.cost.fit_seconds,
                   transform_seconds_imp=time.perf_counter() - t0,
                   peak_mem_mb_imp=imp.cost.peak_mem_mb,
                   n_output_cols=Xtr.shape[1])

    res, _ = fit_score(model_kind, Xtr, ds.y[tr], Xva, ds.y[va], Xte, ds.y[te],
                       ds.p_true[te], **model_params)
    row.update(res)
    return row


def run_grid(cfg: D.DataConfig, methods, model_kind: str, seeds,
             model_params=None, imputer_kw=None, out_csv: str | Path | None = None,
             log=print) -> pd.DataFrame:
    rows = []
    for s in seeds:
        for m in methods:
            t0 = time.perf_counter()
            r = run_cell(cfg, m, model_kind, s, model_params, imputer_kw)
            r["cell_seconds"] = time.perf_counter() - t0
            rows.append(r)
            log(f"  seed={s:<3d} {r['method']:<30s} test_auc="
                f"{r.get('test_auc', float('nan')):.4f} ({r['cell_seconds']:.1f}s)")
            if out_csv:
                pd.DataFrame(rows).to_csv(out_csv, index=False)
    return pd.DataFrame(rows)
