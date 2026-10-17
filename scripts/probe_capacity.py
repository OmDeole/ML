"""How strong can an RF be on the calibrated GAM-global data set?

Motivation: the paper reports oracle 0.9703 but only 0.7789 for its 'No Missing'
model. We need to know (a) what a normal RF achieves when the segment feature is
available, and (b) what it achieves when it has to pool heterogeneous segments.
"""
from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from impute_bench import data as D  # noqa: E402

SEED = 20230272  # oracle = 0.9703


def main():
    cfg = D.DataConfig(seed=SEED, coef_mode="per_segment")
    ds = D.generate_gam_global(cfg)
    idx = ds.split()
    tr, te = idx["train"], idx["test"]
    ohe = D.design_matrix(ds, cfg)
    variants = {"with_ohe": np.hstack([ds.X_clean, ohe]),
                "no_ohe": ds.X_clean}
    rows = []
    for (name, X), msl, depth, nf in itertools.product(
            variants.items(), [1, 100, 1000], [None, 6], ["sqrt"]):
        m = RandomForestClassifier(n_estimators=60, min_samples_leaf=msl,
                                   max_depth=depth, max_features=nf,
                                   n_jobs=-1, random_state=0)
        t0 = time.perf_counter()
        m.fit(X[tr], ds.y[tr])
        fit = time.perf_counter() - t0
        a_te = roc_auc_score(ds.y[te], m.predict_proba(X[te])[:, 1])
        a_tr = roc_auc_score(ds.y[tr], m.predict_proba(X[tr])[:, 1])
        rows.append(dict(variant=name, min_samples_leaf=msl, max_depth=depth,
                         n_estimators=60, train_auc=a_tr, test_auc=a_te,
                         gap=a_tr - a_te, fit_s=fit))
        print(f"{name:9s} msl={msl:<5d} depth={str(depth):<4s} "
              f"train={a_tr:.4f} test={a_te:.4f} gap={a_tr-a_te:+.4f} ({fit:.0f}s)",
              flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "capacity_probe.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
