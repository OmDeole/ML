"""Build every figure and summary table used in the reports."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402
import pandas as pd                  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from impute_bench.plots import BLUE, RED, PAPER_ORDER, fig_imputation, fig_lines  # noqa: E402

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
RES = ROOT / "results"

# --- Fig. 14 of the paper, decoded from the rasterised figure -----------------
PAPER = pd.DataFrame(
    [["Clustering Imputation", 0.7185, 0.0050],
     ["Decile Imputation", 0.7042, 0.0276],
     ["Mean Imputation", 0.7162, 0.0044],
     ["Median Imputation", 0.7146, 0.0047],
     ["Missing Indicator Imputation", 0.7253, 0.0013],
     ["No Missing", 0.7789, 0.0013],
     ["Tree Imputation", 0.6974, 0.0415]],
    columns=["method", "mean", "sd"]).set_index("method")
PAPER.to_csv(RES / "paper_fig14_decoded.csv")
ORACLE = 0.9703


def summarise(df, value="test_auc"):
    g = df.groupby("method")[value].agg(["mean", "std", "count"]).fillna(0)
    return g


def replication_table(ours, name):
    t = ours[["mean", "std"]].copy()
    t["paper_mean"] = PAPER["mean"]
    t["paper_sd"] = PAPER["sd"]
    t["delta_vs_paper"] = t["mean"] - t["paper_mean"]
    t["rank_ours"] = t["mean"].rank(ascending=False).astype(int)
    t["rank_paper"] = t["paper_mean"].rank(ascending=False).astype(int)
    t = t.loc[[m for m in PAPER_ORDER if m in t.index]]
    t.round(4).to_csv(RES / f"replication_table_{name}.csv")
    return t


def fig_compare(ours, title, out, paper=PAPER):
    """Left: our absolute results. Right: both series normalised to the no-missing
    control, which is the scale-free way to compare shapes across model families."""
    order = [m for m in PAPER_ORDER if m in ours.index]
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), dpi=170)

    ax = axes[0]
    ax.errorbar(x, ours.loc[order, "mean"], yerr=2 * ours.loc[order, "std"],
                fmt="o", color=BLUE, ecolor=BLUE, capsize=5, elinewidth=1.4, ms=5)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" Imputation", "\nImputation") for m in order],
                       fontsize=6.8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Test AUC (mean $\\pm$ 2 SD)", fontsize=8.5)
    ax.grid(axis="y", color="#e8e8e8", lw=0.6)
    ax.set_axisbelow(True)

    ax2 = axes[1]
    o_norm = ours.loc[order, "mean"] - ours.loc["No Missing", "mean"]
    p_norm = paper.loc[order, "mean"] - paper.loc["No Missing", "mean"]
    ax2.bar(x - 0.2, p_norm.values, width=0.4, color="#999999", label="Paper Fig. 14")
    ax2.bar(x + 0.2, o_norm.values, width=0.4, color=BLUE, label="Replication (RF)")
    ax2.axhline(0, color="black", lw=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace(" Imputation", "\nImputation") for m in order],
                        fontsize=6.8)
    ax2.set_title("Change vs. the no-missing control", fontsize=10)
    ax2.set_ylabel("$\\Delta$ AUC", fontsize=8.5)
    ax2.grid(axis="y", color="#e8e8e8", lw=0.6)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=8, frameon=False, loc="lower left")
    for a in axes:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_penalty(ours, out, title):
    """Change relative to the no-missing control: ours vs the paper."""
    order = [m for m in PAPER_ORDER if m in ours.index and m != "No Missing"]
    base_ours = ours.loc["No Missing", "mean"]
    base_paper = PAPER.loc["No Missing", "mean"]
    d_ours = (ours.loc[order, "mean"] - base_ours)
    d_paper = (PAPER.loc[order, "mean"] - base_paper)
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.2, 3.9), dpi=170)
    ax.bar(x - 0.19, d_paper.values, width=0.36, color="#999999", label="Paper (Fig. 14)")
    ax.bar(x + 0.19, d_ours.values, width=0.36, color=BLUE, label="Replication")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" Imputation", "\nImputation") for m in order], fontsize=7.5)
    ax.set_ylabel("$\\Delta$ AUC vs. no-missing", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", color="#e8e8e8", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main():
    summary = {}

    # ---- 1. replication, standard RF
    core = pd.read_csv(RES / "core_rf.csv")
    ours = summarise(core)
    t = replication_table(ours, "standard_rf")
    fig_compare(ours, "Replication — Random Forest (Group 10)", FIG / "fig1_replication_rf.png")
    fig_penalty(ours, FIG / "fig3_penalty_rf.png",
                "Missingness penalty — Random Forest vs paper")
    summary["standard_rf"] = ours.round(4).to_dict()

    # ---- 2. replication, paper-compatible capacity (depth 2)
    core2 = pd.read_csv(RES / "core_gam_global.csv")
    ours2 = summarise(core2)
    t2 = replication_table(ours2, "calibrated_depth2")
    fig_compare(ours2, "Replication — capacity-calibrated RF (depth 2)",
                FIG / "fig2_replication_depth2.png")
    summary["depth2"] = ours2.round(4).to_dict()

    # ---- 3. null-rate sweep
    sw = pd.read_csv(RES / "null_rate_sweep.csv")
    agg = sw.groupby(["method", "null_rate"])["test_auc"].agg(["mean", "std"]).reset_index()
    agg.to_csv(RES / "null_rate_summary.csv", index=False)
    fig_lines(agg[agg.method != "No Missing"], "null_rate", "mean", "method",
              out=FIG / "fig4_null_rate.png",
              xlabel="Null rate injected into x1–x3", ylabel="Test AUC",
              title="Accuracy as the missing-data budget grows")
    piv = agg.pivot(index="null_rate", columns="method", values="mean")
    piv.columns = [c.replace(" Imputation", "") for c in piv.columns]
    piv.round(4).to_csv(RES / "null_rate_pivot.csv")
    summary["null_rate"] = piv.round(4).to_dict()

    # ---- 4. mechanism
    mech = pd.read_csv(RES / "mechanism.csv")
    magg = mech.groupby(["method", "mechanism"])["test_auc"].agg(["mean", "std"]).reset_index()
    magg.to_csv(RES / "mechanism_summary.csv", index=False)
    piv_m = magg.pivot(index="method", columns="mechanism", values="mean").round(4)
    piv_m.to_csv(RES / "mechanism_pivot.csv")
    fig, ax = plt.subplots(figsize=(7.6, 4.0), dpi=170)
    mechs = [m for m in ["mcar", "mar", "mnar"] if m in piv_m.columns]
    xx = np.arange(len(piv_m))
    for i, m in enumerate(mechs):
        ax.bar(xx + (i - 1) * 0.26, piv_m[m].values, width=0.25, label=m.upper())
    ax.set_xticks(xx)
    ax.set_xticklabels([i.replace(" Imputation", "\nImputation") for i in piv_m.index], fontsize=8)
    ax.set_ylabel("Test AUC", fontsize=9)
    ax.set_title("Sensitivity to the missingness mechanism", fontsize=10)
    ax.grid(axis="y", color="#e8e8e8", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_mechanism.png", bbox_inches="tight")
    plt.close(fig)
    summary["mechanism"] = piv_m.to_dict()

    # ---- 4b. irreplaceability: how many affected signal features before
    #           the imputation choice starts to matter?
    irr_p = RES / "irreplaceability.csv"
    if irr_p.exists():
        irr = pd.read_csv(irr_p)
        ig = irr.groupby(["n_missing_signal", "method"])["test_auc"].agg(["mean", "std"]).reset_index()
        base = ig[ig.method == "No Missing"].set_index("n_missing_signal")["mean"]
        ig["penalty"] = ig.apply(lambda r: base.loc[r.n_missing_signal] - r["mean"], axis=1)
        ig["is_missing"] = ig.method != "No Missing"
        sp = (ig[ig.is_missing].groupby("n_missing_signal")
              .apply(lambda g: g["mean"].max() - g["mean"].min(), include_groups=False)
              .rename("spread"))
        pen = (ig[ig.is_missing].groupby("n_missing_signal")["penalty"].mean()
               .rename("mean_penalty"))
        summ = pd.concat([pen, sp], axis=1).reset_index()
        summ.to_csv(RES / "irreplaceability_summary.csv", index=False)

        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), dpi=170)
        for m, g in ig[ig.is_missing].groupby("method"):
            g = g.sort_values("n_missing_signal")
            axes[0].plot(g["n_missing_signal"], g["mean"], marker="o", ms=4, lw=1.4,
                         label=m.replace(" Imputation", ""))
        axes[0].axhline(base.iloc[0], ls="--", lw=1, color=RED)
        axes[0].annotate("no missing", xy=(3.4, base.iloc[0]), fontsize=7.5, color=RED,
                         va="bottom")
        axes[0].set_xlabel("Signal features carrying 50% nulls", fontsize=8.5)
        axes[0].set_ylabel("Test AUC", fontsize=8.5)
        axes[0].set_title("Accuracy vs. how much of the signal is damaged", fontsize=10)
        axes[0].grid(color="#e8e8e8", lw=0.6)
        axes[0].legend(fontsize=7, frameon=False)

        axes[1].plot(summ["n_missing_signal"], summ["mean_penalty"], marker="s",
                     color=BLUE, ms=5, lw=1.6, label="mean penalty vs no-missing")
        axes[1].plot(summ["n_missing_signal"], summ["spread"], marker="o",
                     color=RED, ms=5, lw=1.6, label="spread between methods")
        axes[1].axhline(0.0279, ls=":", color=RED, lw=1.2)
        axes[1].annotate("paper's between-method spread (0.028)",
                         xy=(1.05, 0.0279), fontsize=7.2, color=RED, va="top")
        axes[1].set_xlabel("Signal features carrying 50% nulls", fontsize=8.5)
        axes[1].set_ylabel("AUC", fontsize=8.5)
        axes[1].set_title("Penalty grows, method differences barely do", fontsize=10)
        axes[1].grid(color="#e8e8e8", lw=0.6)
        axes[1].legend(fontsize=7.5, frameon=False)
        for a in axes:
            for sp_ in ("top", "right"):
                a.spines[sp_].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIG / "fig9_irreplaceability.png", bbox_inches="tight")
        plt.close(fig)
        summary["irreplaceability"] = summ.round(4).to_dict("records")

    # ---- 5. cost
    cost = pd.read_csv(RES / "imputer_cost.csv")
    cost["total_seconds_per_1m"] = cost["score_seconds_per_1m"] + cost["fit_seconds"] / 10
    order_c = [m for m in PAPER_ORDER if m in set(cost.method)]
    cost = cost.set_index("method").loc[order_c].reset_index()
    cost.to_csv(RES / "deployment_cost.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.0, 4.0), dpi=170)
    xx = np.arange(len(cost))
    ax.bar(xx - 0.2, cost["score_seconds_per_1m"], width=0.38,
           color=BLUE, label="Score 1M rows")
    ax.bar(xx + 0.2, cost["fit_seconds"], width=0.38, color="#9fb3d1", label="Fit (150K rows)")
    ax.set_yscale("log")
    ax.set_xticks(xx)
    ax.set_xticklabels([m.replace(" Imputation", "\nImputation") for m in cost.method], fontsize=7.5)
    ax.set_ylabel("Seconds (log scale)", fontsize=9)
    ax.set_title("Preprocessing cost", fontsize=10)
    ax.grid(axis="y", color="#e8e8e8", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig7_cost_breakdown.png", bbox_inches="tight")
    plt.close(fig)

    # ---- 5b. cost scaling with the number of features
    cs = pd.read_csv(RES / "cost_scaling.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), dpi=170)
    for ax, col, lab in [(axes[0], "fit_s", "Fit time (s)"),
                         (axes[1], "score_s_per_1m", "Score 1M rows (s)")]:
        for m, g in cs.groupby("method"):
            g = g.sort_values("n_features")
            ax.plot(g["n_features"], g[col], marker="o", ms=4, lw=1.4,
                    label=m.replace(" Imputation", ""))
        ax.set_xlabel("Number of features (1/3 of them 50% null)", fontsize=8.5)
        ax.set_ylabel(lab, fontsize=8.5)
        ax.set_yscale("log")
        ax.grid(color="#e8e8e8", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_title("Fitting the imputer", fontsize=10)
    axes[1].set_title("Applying it to a scoring batch", fontsize=10)
    axes[1].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig8_cost_scaling.png", bbox_inches="tight")
    plt.close(fig)
    summary["cost_scaling"] = cs.round(4).to_dict("records")

    # Pareto: AUC (core_rf) vs preprocessing cost at 60 features (fit + score/1M)
    cs60 = cs[cs.n_features == 60][["method", "fit_s", "score_s_per_1m"]]
    cs60["total_seconds_per_1m"] = cs60["fit_s"] + cs60["score_s_per_1m"]
    pareto = ours.reset_index().merge(cs60, on="method", how="inner")
    pareto = pareto[pareto.method != "No Missing"]
    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=170)
    ax.scatter(pareto["total_seconds_per_1m"], pareto["mean"], s=70, color=BLUE, zorder=3)
    for _, r in pareto.iterrows():
        ax.annotate(r["method"].replace(" Imputation", ""),
                    (r["total_seconds_per_1m"], r["mean"]),
                    textcoords="offset points", xytext=(7, 5), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Cost at 60 features: fit + score 1M rows (s, log scale)", fontsize=9)
    ax.set_ylabel("Test AUC", fontsize=9)
    ax.set_title("Accuracy vs. deployment cost", fontsize=10)
    ax.grid(color="#e8e8e8", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_pareto.png", bbox_inches="tight")
    plt.close(fig)
    pareto.round(4).to_csv(RES / "pareto.csv", index=False)
    summary["pareto"] = pareto.round(4).to_dict("records")

    (RES / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))
    print("\nreplication (standard RF):\n", t.round(4).to_string())
    print("\nreplication (depth-2):\n", t2.round(4).to_string())


if __name__ == "__main__":
    main()
