"""Calibrate the reconstructed data generator against the paper's Fig. 14.

Two unknowns are searched:
  1. which coefficient draw reproduces "Oracle AUC - 0.9703" (printed on Fig 14);
  2. which reading of the segment structure (per-segment coefficients vs. one
     shared vector) reproduces the paper's very large oracle->model gap
     (0.9703 oracle vs ~0.779 for "No Missing").

Writes results/calibration.csv.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from impute_bench import data as D            # noqa: E402
from impute_bench.modeling import build_model  # noqa: E402


def one(cfg: D.DataConfig, k: int, model: str = "xgb", n_est: int = 200):
    cfg = D.DataConfig(**{**cfg.to_dict(), "seed": cfg.seed + k})
    t0 = time.perf_counter()
    ds = D.generate_gam_global(cfg)
    gen_s = time.perf_counter() - t0
    idx = ds.split()
    tr, te = idx["train"], idx["test"]
    # include the one-hot segment columns, as the paper's standard preprocessing does
    ohe = D.design_matrix(ds, cfg)
    X = np.hstack([ds.X_clean, ohe])
    oracle = roc_auc_score(ds.y[te], ds.p_true[te])
    m = build_model(model, n_estimators=n_est)
    t0 = time.perf_counter()
    m.fit(X[tr], ds.y[tr])
    fit_s = time.perf_counter() - t0
    auc = roc_auc_score(ds.y[te], m.predict_proba(X[te])[:, 1])
    return dict(mode=cfg.coef_mode, draw=k, seed=cfg.seed, oracle_auc=oracle,
                model_auc=auc, gap=oracle - auc, gen_s=gen_s, fit_s=fit_s,
                pos_rate=float(ds.y.mean()),
                coef_absmean=float(np.abs(ds.coefs).mean()))


def main(n_draws: int = 6):
    rows = []
    base = D.DataConfig()
    for mode in ["per_segment", "shared"]:
        for k in range(n_draws):
            r = one(D.DataConfig(**{**base.to_dict(), "coef_mode": mode}), k)
            rows.append(r)
            print(f"{mode:12s} draw {k}: oracle={r['oracle_auc']:.4f} "
                  f"model={r['model_auc']:.4f} gap={r['gap']:.4f} pos={r['pos_rate']:.3f} "
                  f"gen={r['gen_s']:.1f}s fit={r['fit_s']:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "calibration.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("\nsummary by mode:")
    print(df.groupby("mode")[["oracle_auc", "model_auc", "gap"]].agg(["mean", "std", "min", "max"]))
    print("\nclosest to oracle=0.9703:")
    df["dist"] = (df.oracle_auc - 0.9703).abs()
    print(df.sort_values("dist").head(5)[["mode", "draw", "seed", "oracle_auc", "model_auc", "gap"]])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
