"""Master experiment driver.

    python scripts/run_all.py core        # replication, standard RF
    python scripts/run_all.py sweep       # null-rate x imputation method
    python scripts/run_all.py mechanism   # MCAR / MAR / MNAR
    python scripts/run_all.py cost        # imputer-only cost at 1M rows
    python scripts/run_all.py all
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from impute_bench import data as D                      # noqa: E402
from impute_bench import imputers as I                  # noqa: E402
from impute_bench.runner import run_grid                # noqa: E402

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
SEED_COEF = 20230272            # draw 48: oracle AUC = 0.9703
RF = dict(n_estimators=40, min_samples_leaf=100, max_features="sqrt")
METHODS = ["none", "cluster", "decile", "mean", "median", "indicator", "tree"]


def _cfg(**kw):
    base = dict(n_rows=250_000, n_segments=5, n_signal=5, n_noise=5, corr=0.5,
                coef_mode="per_segment", missing_features=("x1", "x2", "x3"),
                missing_rate=0.5, missing_mechanism="mcar", seed=SEED_COEF)
    base.update(kw)
    return D.DataConfig(**base)


def core(seeds=range(8)):                      # noqa: C901
    conf = json.loads((ROOT / "configs" / "core_gam_global_rf.json").read_text())
    cfg = D.DataConfig(**conf["data"])
    out = RESULTS / "core_rf.csv"
    print("== core (standard RF) ==", flush=True)
    df = run_grid(cfg, conf["methods"], "rf", list(seeds),
                  model_params=conf["model"]["params"], out_csv=out, log=_log)
    print(f"wrote {out} ({len(df)} rows)", flush=True)


def sweep(rates=(0.1, 0.3, 0.5, 0.7), seeds=range(3)):
    print("== null-rate sweep ==", flush=True)
    out = RESULTS / "null_rate_sweep.csv"
    rows = []
    for r in rates:
        cfg = _cfg(missing_rate=r)
        df = run_grid(cfg, METHODS, "rf", list(seeds), model_params=RF,
                      out_csv=None, log=_log)
        df["null_rate"] = r
        rows.append(df)
        pd.concat(rows).to_csv(out, index=False)
    print(f"wrote {out}", flush=True)


def mechanism(mechs=("mcar", "mar", "mnar"), seeds=range(3)):
    print("== missingness mechanism ==", flush=True)
    out = RESULTS / "mechanism.csv"
    rows = []
    for m in mechs:
        cfg = _cfg(missing_mechanism=m)
        df = run_grid(cfg, ["none", "mean", "median", "indicator", "tree"], "rf",
                      list(seeds), model_params=RF, out_csv=None, log=_log)
        df["mechanism"] = m
        rows.append(df)
        pd.concat(rows).to_csv(out, index=False)
    print(f"wrote {out}", flush=True)


def cost():
    """Imputation-only cost benchmark: fit on 150k, transform 1M rows."""
    print("== imputation cost benchmark ==", flush=True)
    cfg = _cfg()
    ds = D.generate_gam_global(cfg)
    ohe = D.design_matrix(ds, cfg)
    names = ds.feature_names + [f"seg_{k}" for k in range(1, cfg.n_segments)]
    Xm = np.hstack([ds.X, ohe])
    cols = [names.index(f) for f in cfg.missing_features]
    idx = ds.split()
    Xtr = Xm[idx["train"]]
    # a 1M-row scoring batch, built by tiling the test split
    Xte = Xm[idx["test"]]
    reps = int(np.ceil(1_000_000 / len(Xte)))
    Xbig = np.tile(Xte, (reps, 1))[:1_000_000]

    rows = []
    for m in ["mean", "median", "indicator", "decile", "cluster", "tree"]:
        imp = I.get_imputer(m, cols).fit(Xtr, ds.y[idx["train"]])
        t0 = time.perf_counter()
        _ = imp.transform(Xbig)
        t_score = time.perf_counter() - t0
        rows.append(dict(method=imp.name, fit_seconds=imp.cost.fit_seconds,
                         score_seconds_per_1m=t_score,
                         peak_mem_mb=imp.cost.peak_mem_mb,
                         n_output_cols=Xbig.shape[1] + imp.extra_cols))
        print(f"  {imp.name:30s} fit={imp.cost.fit_seconds:8.2f}s "
              f"score/1M={t_score:7.2f}s mem={imp.cost.peak_mem_mb:7.1f}MB", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "imputer_cost.csv", index=False)
    print("wrote results/imputer_cost.csv", flush=True)


def _log(msg):
    print(msg, flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.perf_counter()
    if what in ("core", "all"):
        core()
    if what in ("sweep", "all"):
        sweep()
    if what in ("mechanism", "all"):
        mechanism()
    if what in ("cost", "all"):
        cost()
    print(f"done in {time.perf_counter()-t0:.0f}s", flush=True)
