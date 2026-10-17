"""Which model/data regime reproduces the paper's *sensitivity* to missingness?

The paper's Fig. 14 shows:
    No Missing 0.7789  ->  Missing Indicator 0.7253  (-0.054)
                           Mean             0.7162  (-0.063)
                           Tree             0.6974  (-0.081, +/-0.042)

A model calibrated purely by capacity (e.g. depth-2 RF) matches 0.7789 but is
almost insensitive to the missingness (-0.005), which is wrong behaviour. This
script compares four regimes on one draw and reports the full profile:

    model in {xgb, rf}  x  segment one-hot in {yes, no}
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from impute_bench import data as D                       # noqa: E402
from impute_bench import imputers as I                   # noqa: E402
from impute_bench.modeling import fit_score              # noqa: E402

SEED = 20230272
METHODS = ["none", "cluster", "decile", "mean", "median", "indicator", "tree"]


def main(seed: int = SEED, models=("xgb", "rf")):
    cfg = D.DataConfig(seed=seed, coef_mode="per_segment")
    ds = D.generate_gam_global(cfg)
    ohe = D.design_matrix(ds, cfg)
    names = ds.feature_names + [f"seg_{k}" for k in range(1, cfg.n_segments)]
    idx = ds.split()
    tr, va, te = idx["train"], idx["val"], idx["test"]
    miss_cols = [names.index(f) for f in cfg.missing_features]
    ohe_names = [f"seg_{k}" for k in range(1, cfg.n_segments)]

    rows = []
    for model in models:
        for use_ohe in (True, False):
            keep = np.ones(len(names), bool) if use_ohe else np.array(
                [n not in ohe_names for n in names])
            Xm = np.hstack([ds.X, ohe])[:, keep]
            Xc = np.hstack([ds.X_clean, ohe])[:, keep]
            cols = [int(np.where(keep)[0][miss_cols.index(c)]) if use_ohe else c
                    for c in miss_cols]
            for m in METHODS:
                t0 = time.perf_counter()
                if m == "none":
                    Xtr, Xva, Xte = Xc[tr], Xc[va], Xc[te]
                else:
                    imp = I.get_imputer(m, cols).fit(Xm[tr], ds.y[tr])
                    Xtr, Xva, Xte = (imp.transform(Xm[tr]), imp.transform(Xm[va]),
                                     imp.transform(Xm[te]))
                res, _ = fit_score(model, Xtr, ds.y[tr], Xva, ds.y[va], Xte, ds.y[te],
                                   ds.p_true[te])
                rows.append(dict(model=model, ohe=use_ohe, method=imp_name(m),
                                 test_auc=res["test_auc"], train_auc=res["train_auc"],
                                 oracle=res["oracle_auc"], secs=time.perf_counter() - t0))
                print(f"{model:3s} ohe={int(use_ohe)} {imp_name(m):30s} "
                      f"auc={res['test_auc']:.4f} ({rows[-1]['secs']:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "regime_diagnosis.csv"
    df.to_csv(out, index=False)
    print("\n== profile by regime ==")
    for (model, ohe), g in df.groupby(["model", "ohe"]):
        base = g.loc[g.method == "No Missing", "test_auc"].iloc[0]
        d = (g.set_index("method").test_auc - base).round(4)
        print(f"\n{model} ohe={ohe}: no-missing {base:.4f}")
        print(d.to_string())
    print(f"\nwrote {out}")


def imp_name(m):
    return {"none": "No Missing", "cluster": "Clustering Imputation",
            "decile": "Decile Imputation", "mean": "Mean Imputation",
            "median": "Median Imputation", "indicator": "Missing Indicator Imputation",
            "tree": "Tree Imputation"}[m]


if __name__ == "__main__":
    main()
