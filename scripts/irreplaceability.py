"""When does the imputation choice start to matter?

The paper's design injects nulls into 3 of its 5 signal features. That choice is
unstated and load-bearing: the audit in the replication report shows the model
routes around any imputed constant as long as enough signal survives. This
experiment sweeps the number of null-affected signal features (1, 2, 3, 5) and
measures the accuracy spread between methods — i.e. it locates the boundary where
imputation choice becomes consequential.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from impute_bench import data as D            # noqa: E402
from impute_bench.runner import run_grid      # noqa: E402

RF = dict(n_estimators=40, min_samples_leaf=100, max_features="sqrt")
METHODS = ["none", "mean", "median", "indicator", "decile", "tree"]


def main():
    rows = []
    t_all = time.perf_counter()
    for k in (1, 2, 3, 5):
        feats = tuple(f"x{i}" for i in range(1, k + 1))
        cfg = D.DataConfig(seed=20230272, coef_mode="per_segment",
                           missing_features=feats, missing_rate=0.5)
        print(f"\n== {k} of 5 signal features 50% null: {feats} ==", flush=True)
        df = run_grid(cfg, METHODS, "rf", [0, 1], model_params=RF, log=print)
        df["n_missing_signal"] = k
        rows.append(df)
        pd.concat(rows).to_csv(ROOT / "results" / "irreplaceability.csv", index=False)
    res = pd.concat(rows)
    res.to_csv(ROOT / "results" / "irreplaceability.csv", index=False)

    print("\n== summary ==")
    for k, g in res.groupby("n_missing_signal"):
        m = g.groupby("method").test_auc.mean()
        base = m["No Missing"]
        imp = m.drop("No Missing")
        print(f"k={k}: no-missing={base:.4f}  penalty(mean method)={base-imp.mean():.4f}  "
              f"spread={imp.max()-imp.min():.4f}  best={imp.idxmax()}")
    print(f"\ntotal {time.perf_counter()-t_all:.0f}s")


if __name__ == "__main__":
    main()
