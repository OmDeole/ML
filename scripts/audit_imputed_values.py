"""Two audits that support the mechanistic argument in the replication report.

1. `imputed_values.csv` — what value does each method actually insert, per feature?
   Under MCAR the six methods should converge on nearly the same constant.
2. `tree_variant.csv` — tree imputation with 'most_different' (the paper's literal
   instruction / the blog) vs 'closest' (the paper's prose), 3 iterations each.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from impute_bench import data as D, imputers as I        # noqa: E402
from impute_bench.runner import prepare                  # noqa: E402
from impute_bench.modeling import fit_score              # noqa: E402

CFG = D.DataConfig(seed=20230272, coef_mode="per_segment")
RF = dict(n_estimators=40, min_samples_leaf=100, max_features="sqrt")


def value_audit():
    ds, Xm, Xc, names, idx = prepare(CFG, seed_offset=0)
    tr, te = idx["train"], idx["test"]
    cols = [names.index(f) for f in CFG.missing_features]
    rows = []
    for key in ["mean", "median", "indicator", "decile", "cluster", "tree"]:
        imp = I.get_imputer(key, cols).fit(Xm[tr], ds.y[tr])
        Xt = imp.transform(Xm[te])
        for j, c in enumerate(CFG.missing_features):
            ci = cols[j]
            mask = np.isnan(Xm[te][:, ci])
            vals = Xt[mask, ci]
            uniq = np.unique(np.round(vals, 4))
            rows.append(dict(method=imp.name, feature=c,
                             imputed_value=float(np.median(vals)),
                             n_distinct_values=len(uniq),
                             n_missing=int(mask.sum())))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "imputed_values.csv", index=False)
    piv = df.pivot(index="method", columns="feature", values="imputed_value")
    print("Imputed constant per feature (test rows that were missing):")
    print(piv.round(4).to_string())
    print("\nRange across methods, per feature:")
    print((piv.max() - piv.min()).round(4).to_string())


def tree_variant():
    ds, Xm, Xc, names, idx = prepare(CFG, seed_offset=0)
    tr, va, te = idx["train"], idx["val"], idx["test"]
    cols = [names.index(f) for f in CFG.missing_features]
    rows = []
    for sel in ["most_different", "closest"]:
        for s in range(3):
            ds, Xm, Xc, names, idx = prepare(CFG, seed_offset=s)
            tr, va, te = idx["train"], idx["val"], idx["test"]
            imp = I.get_imputer("tree", cols, selection=sel).fit(Xm[tr], ds.y[tr])
            Xtr, Xva, Xte = (imp.transform(Xm[tr]), imp.transform(Xm[va]),
                             imp.transform(Xm[te]))
            r, _ = fit_score("rf", Xtr, ds.y[tr], Xva, ds.y[va], Xte, ds.y[te],
                             ds.p_true[te], **RF)
            rows.append(dict(selection=sel, seed=s, test_auc=r["test_auc"]))
            print(f"  tree/{sel:14s} seed={s} auc={r['test_auc']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "tree_variant.csv", index=False)
    print(df.groupby("selection").test_auc.agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    value_audit()
    print()
    tree_variant()
