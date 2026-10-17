"""Run one experiment configuration (JSON) end to end.

Usage:
    PYTHONPATH=src python scripts/run_from_config.py configs/core_gam_global.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from impute_bench import data as D                 # noqa: E402
from impute_bench.runner import run_grid           # noqa: E402


def main(cfg_path: str):
    conf = json.loads(Path(cfg_path).read_text())
    cfg = D.DataConfig(**conf["data"])
    out = ROOT / conf["out_csv"]
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"== {conf['name']} ==")
    print(f"   data: {cfg.n_rows} rows, {cfg.n_signal} signal + {cfg.n_noise} noise, "
          f"{cfg.n_segments} segments, null rate {cfg.missing_rate} ({cfg.missing_mechanism}), "
          f"coef_mode={cfg.coef_mode}, seed={cfg.seed}")
    print(f"   model: {conf['model']['kind']} {conf['model']['params']}")
    print(f"   seeds: {conf['seeds']}")

    t0 = time.perf_counter()
    df = run_grid(cfg, conf["methods"], conf["model"]["kind"], conf["seeds"],
                  model_params=conf["model"]["params"],
                  imputer_kw=conf.get("imputer_kwargs", {}), out_csv=out)
    df["config"] = conf["name"]
    df.to_csv(out, index=False)
    print(f"\nwrote {out} ({len(df)} rows, {time.perf_counter()-t0:.0f}s)")
    piv = df.pivot_table(index="method", values=["test_auc", "train_auc"], aggfunc=["mean", "std"])
    print(piv.round(4).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
