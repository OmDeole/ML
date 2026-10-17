"""Plotting helpers, styled after the paper's Figures 12-15."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE = "#000087"
RED = "#d62728"
PAPER_ORDER = ["Clustering Imputation", "Decile Imputation", "Mean Imputation",
               "Median Imputation", "Missing Indicator Imputation",
               "No Missing", "Tree Imputation"]


def fig_imputation(summary, order=None, oracle=None, title="Null Imputation - GAM Global - AUC",
                   ylabel="AUC", out=None, paper_ref=None, ylim=None, figsize=(7.2, 4.2)):
    """`summary` : DataFrame indexed by method with columns mean, sd (2*sd shown)."""
    order = order or [m for m in PAPER_ORDER if m in summary.index]
    order += [m for m in summary.index if m not in order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=figsize, dpi=160)

    means = summary.loc[order, "mean"].values
    sds = summary.loc[order, "sd"].values

    ax.errorbar(x, means, yerr=2 * sds, fmt="o", color=BLUE, ecolor=BLUE,
                elinewidth=1.4, capsize=5, markersize=4.5, zorder=3,
                label="Replication (mean $\\pm$ 2 SD)")
    if paper_ref is not None:
        pr = paper_ref.loc[order]
        ax.errorbar(x + 0.18, pr["mean"], yerr=2 * pr["sd"], fmt="s",
                    color="#8c8c8c", ecolor="#8c8c8c", elinewidth=1.0,
                    capsize=3, markersize=3.5, alpha=0.85, zorder=2,
                    label="Paper Fig. 14")
    if oracle is not None:
        ax.axhline(oracle, ls="--", lw=1.1, color=RED, zorder=1)
        ax.annotate(f"Oracle AUC - {oracle:.4f}", xy=(0.99, oracle),
                    xycoords=("axes fraction", "data"), ha="right", va="bottom",
                    color=RED, fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=RED, lw=0.6))

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" Imputation", "\nImputation") for m in order],
                       fontsize=7.5)
    ax.set_xlabel("Imputation Method", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, pad=10)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=7.5, loc="lower right", frameon=False)
    fig.tight_layout()
    if out:
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    return fig


def fig_cost_pareto(df, out=None):
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=160)
    for _, r in df.iterrows():
        ax.scatter(r["total_seconds_per_1m"], r["test_auc"], s=45, color=BLUE, zorder=3)
        ax.annotate(r["method"].replace(" Imputation", ""),
                    (r["total_seconds_per_1m"], r["test_auc"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("Total pipeline cost per 1M rows (s, log scale)")
    ax.set_ylabel("AUC")
    ax.set_title("Accuracy vs. deployment cost", fontsize=10)
    ax.grid(color="#e6e6e6", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    if out:
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    return fig


def fig_lines(df, xcol, ycol, huecol, out=None, xlabel=None, ylabel="AUC",
              title="", logy=False):
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=160)
    for k, g in df.groupby(huecol):
        g = g.sort_values(xcol)
        ax.plot(g[xcol], g[ycol], marker="o", ms=4, lw=1.4,
                label=k.replace(" Imputation", ""))
    ax.set_xlabel(xlabel or xcol)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    if logy:
        ax.set_yscale("log")
    ax.grid(color="#e6e6e6", lw=0.6)
    ax.legend(fontsize=7.5, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    if out:
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    return fig
