"""Pick the coefficient draw whose oracle AUC best matches the paper's 0.9703.

Only the oracle is needed (no model fitting), so this is cheap and can scan many
seeds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from impute_bench import data as D  # noqa: E402

TARGET = 0.9703  # printed on Fig. 14 of the paper


def main(n_seeds: int = 200):
    base = D.DataConfig(coef_mode="per_segment")
    rows = []
    for k in range(n_seeds):
        cfg = D.DataConfig(**{**base.to_dict(), "seed": base.seed + k})
        ds = D.generate_gam_global(cfg)
        idx = ds.split()
        te = idx["test"]
        rows.append(dict(draw=k, seed=cfg.seed,
                         oracle_auc=roc_auc_score(ds.y[te], ds.p_true[te]),
                         pos_rate=float(ds.y.mean()),
                         coef_absmean=float(np.abs(ds.coefs).mean())))
    df = pd.DataFrame(rows)
    df["abs_err"] = (df.oracle_auc - TARGET).abs()
    out = ROOT / "results" / "coef_scan.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("abs_err").to_csv(out, index=False)
    print(df.sort_values("abs_err").head(10).to_string(index=False))
    best = df.sort_values("abs_err").iloc[0]
    print(f"\nBEST draw={int(best.draw)} seed={int(best.seed)} oracle={best.oracle_auc:.4f} "
          f"(err {best.abs_err:.4f})")
    print(f"oracle across draws: mean={df.oracle_auc.mean():.4f} "
          f"sd={df.oracle_auc.std():.4f} range=[{df.oracle_auc.min():.3f},{df.oracle_auc.max():.3f}]")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
