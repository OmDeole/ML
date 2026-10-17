"""How does imputation cost scale with the number of affected features?

The 10-feature benchmark shows imputation is cheap in absolute terms; what matters
for a deployment decision is how the cost grows when a real table has dozens of
columns with nulls (the paper's Lending Club arm has 80 features, 80% of them
with nulls).

Peak memory is sampled from RSS in a background thread (numpy allocations are not
visible to tracemalloc).
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from impute_bench import imputers as I  # noqa: E402

PROC = psutil.Process()


class RSSSampler:
    def __init__(self, period=0.005):
        self.period, self.peak, self._stop = period, 0.0, threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        base = PROC.memory_info().rss
        while not self._stop.is_set():
            self.peak = max(self.peak, PROC.memory_info().rss - base)
            time.sleep(self.period)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self.t.join()

    @property
    def mb(self):
        return self.peak / 2 ** 20


def main():
    rng = np.random.default_rng(0)
    rows = []
    n_rows = 250_000
    for p in (10, 30, 60):
        n_miss_cols = max(3, p // 3)
        X = rng.standard_normal((n_rows, p)).astype(np.float32)
        miss_cols = list(range(n_miss_cols))
        mask = rng.random((n_rows, n_miss_cols)) < 0.5
        X[:, miss_cols] = np.where(mask, np.nan, X[:, miss_cols])
        y = (rng.random(n_rows) < 0.5).astype(np.int8)
        Xbig = np.tile(X[: n_rows // 2], (8, 1))[:1_000_000]
        for key in ["mean", "median", "indicator", "decile", "cluster", "tree"]:
            t0 = time.perf_counter()
            imp = I.get_imputer(key, miss_cols).fit(X, y)
            fit_s = time.perf_counter() - t0
            with RSSSampler() as mem:
                t0 = time.perf_counter()
                _ = imp.transform(Xbig)
                score_s = time.perf_counter() - t0
            rows.append(dict(n_features=p, n_missing_features=n_miss_cols,
                             method=imp.name, fit_s=fit_s,
                             score_s_per_1m=score_s, peak_mem_mb=mem.mb))
            print(f"p={p:<3d} {imp.name:30s} fit={fit_s:7.2f}s "
                  f"score/1M={score_s:6.2f}s mem={mem.mb:7.1f}MB", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "cost_scaling.csv", index=False)
    print("\nfit seconds by #features:")
    print(df.pivot(index="method", columns="n_features", values="fit_s").round(3).to_string())
    print("\nscore seconds per 1M rows:")
    print(df.pivot(index="method", columns="n_features", values="score_s_per_1m").round(3).to_string())


if __name__ == "__main__":
    main()
