"""
Null-imputation methods, reimplemented from Appendix C.3 of the paper.

Each imputer follows the paper's description literally where it is explicit and
records the computational cost of fit/transform (used by the deployment study).

    mean        - column mean of observed values
    median      - column median of observed values
    indicator   - NA -> -9999 sentinel + binary missing-indicator column per
                  affected feature (paper Table 16 / blog post)
    decile      - 10 quantile bins + a "missing" group; the bin whose target
                  probability is closest to the missing group's target
                  probability donates its median as the imputed value
    cluster     - k-means on the feature matrix; cluster mean per feature
    tree        - single-feature CART against the target; the leaf *most different*
                  from the missing group (smallest p-value of a Welch t-test, i.e.
                  the paper's "smallest t-score") donates its median. NOTE: the
                  paper's prose says "similarity" while its instruction and the
                  2OS blog say "smallest p-value" (most different); we follow the
                  explicit instruction, and `selection="closest"` gives the other
                  reading.

All imputers are fit on TRAIN ONLY and applied to train/val/test, which is the
only leak-free protocol: the paper does not state this explicitly, but a decile /
tree imputer fit on the full data would leak test targets into the features.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier

SENTINEL = -9999.0

try:
    import psutil
    _PROC = psutil.Process(os.getpid())

    def _rss_mb() -> float:
        return _PROC.memory_info().rss / 2 ** 20
except Exception:                                    # pragma: no cover
    def _rss_mb() -> float:
        return float("nan")


@dataclass
class CostRecord:
    method: str
    fit_seconds: float = 0.0
    transform_seconds: float = 0.0
    peak_mem_mb: float = 0.0
    n_output_cols: int = 0
    n_params: int = 0

    def as_dict(self):
        return dict(self.__dict__)


class BaseImputer:
    name = "base"

    def __init__(self, missing_cols, **kw):
        self.missing_cols = list(missing_cols)
        self.cost = CostRecord(method=self.name)

    # -- to be overridden ---------------------------------------------------
    def _fit(self, X, y):
        pass

    def _transform(self, X):
        return X

    # -- public API ---------------------------------------------------------
    def fit(self, X, y):
        rss0 = _rss_mb()
        t0 = time.perf_counter()
        self._fit(X, y)
        self.cost.fit_seconds = time.perf_counter() - t0
        self.cost.peak_mem_mb = max(0.0, _rss_mb() - rss0)
        return self

    def transform(self, X):
        t0 = time.perf_counter()
        out = self._transform(X).astype(np.float32)
        self.cost.transform_seconds += time.perf_counter() - t0
        assert not np.isnan(out).any(), f"{self.name}: NaNs left after transform"
        self.cost.n_output_cols = out.shape[1]
        return out

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    @property
    def extra_cols(self):
        return 0


# --------------------------------------------------------------------------- #
class MeanImputer(BaseImputer):
    name = "Mean Imputation"

    def _fit(self, X, y):
        obs = ~np.isnan(X)
        self.means_ = np.nanmean(X, axis=0)
        self.cost.n_params = int(obs.sum())

    def _transform(self, X):
        out = X.copy()
        for j in self.missing_cols:
            m = np.isnan(out[:, j])
            out[m, j] = self.means_[j]
        return out


class MedianImputer(BaseImputer):
    name = "Median Imputation"

    def _fit(self, X, y):
        self.medians_ = np.nanmedian(X, axis=0)
        self.cost.n_params = X.shape[1]

    def _transform(self, X):
        out = X.copy()
        for j in self.missing_cols:
            m = np.isnan(out[:, j])
            out[m, j] = self.medians_[j]
        return out


class IndicatorImputer(BaseImputer):
    """NA -> sentinel + binary indicator column (paper Table 16)."""
    name = "Missing Indicator Imputation"

    def _fit(self, X, y):
        self.cost.n_params = X.shape[1] + len(self.missing_cols)

    def _transform(self, X):
        ind = np.isnan(X[:, self.missing_cols]).astype(np.float32)
        out = X.copy()
        for j in self.missing_cols:
            out[np.isnan(out[:, j]), j] = SENTINEL
        return np.hstack([out, ind])

    @property
    def extra_cols(self):
        return len(self.missing_cols)


class DecileImputer(BaseImputer):
    """Target-matched decile imputation (paper Appendix C.3)."""

    name = "Decile Imputation"

    def __init__(self, missing_cols, n_bins: int = 10, **kw):
        super().__init__(missing_cols, **kw)
        self.n_bins = n_bins

    def _fit(self, X, y):
        self.rules_ = {}
        for j in self.missing_cols:
            col = X[:, j]
            obs = ~np.isnan(col)
            v, t = col[obs], y[obs]
            edges = np.unique(np.nanquantile(v, np.linspace(0, 1, self.n_bins + 1)))
            if len(edges) < 3:                      # degenerate column
                self.rules_[j] = ("const", float(np.median(v)))
                continue
            b = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, len(edges) - 2)
            # target probability and median per bin
            rates = np.array([t[b == k].mean() if (b == k).any() else np.nan
                              for k in range(len(edges) - 1)])
            meds = np.array([np.median(v[b == k]) if (b == k).any() else np.nan
                             for k in range(len(edges) - 1)])
            miss = np.isnan(col)
            p_miss = y[miss].mean() if miss.any() else np.nan
            if np.isnan(p_miss):                    # nothing missing in train
                self.rules_[j] = ("const", float(np.median(v)))
                continue
            k_best = int(np.nanargmin(np.abs(rates - p_miss)))
            self.rules_[j] = ("bin", float(meds[k_best]), float(edges[0]),
                              float(edges[-1]), k_best, p_miss)
        self.cost.n_params = self.n_bins * len(self.missing_cols)

    def _transform(self, X):
        out = X.copy()
        for j, rule in self.rules_.items():
            m = np.isnan(out[:, j])
            out[m, j] = rule[1]
        return out


class ClusterImputer(BaseImputer):
    """k-means clustering imputation (paper Appendix C.3)."""

    name = "Clustering Imputation"

    def __init__(self, missing_cols, n_clusters: int = 5, random_state: int = 0, **kw):
        super().__init__(missing_cols, **kw)
        self.n_clusters = n_clusters
        self.random_state = random_state

    def _fit(self, X, y):
        self.medians_ = np.nanmedian(X, axis=0)
        Xf = np.where(np.isnan(X), self.medians_, X)
        self.km_ = KMeans(n_clusters=self.n_clusters, n_init=3,
                          random_state=self.random_state).fit(Xf)
        labels = self.km_.labels_
        self.cluster_means_ = {}
        for j in self.missing_cols:
            col = X[:, j]
            self.cluster_means_[j] = np.array(
                [np.nanmean(col[labels == k]) if (labels == k).any() else self.medians_[j]
                 for k in range(self.n_clusters)]
            )
        self.cost.n_params = (self.n_clusters * X.shape[1]) + X.shape[1]

    def _transform(self, X):
        Xf = np.where(np.isnan(X), self.medians_, X)
        lab = self.km_.predict(Xf)
        out = X.copy()
        for j in self.missing_cols:
            m = np.isnan(out[:, j])
            out[m, j] = self.cluster_means_[j][lab[m]]
        return out


class TreeImputer(BaseImputer):
    """Single-feature CART + t-test matched leaf median (paper Appendix C.3)."""

    name = "Tree Imputation"

    def __init__(self, missing_cols, max_depth: int = 3, min_samples_leaf: int = 500,
                 random_state: int = 0, max_fit_rows: int | None = 100_000,
                 selection: str = "most_different", **kw):
        super().__init__(missing_cols, **kw)
        self.selection = selection
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.max_fit_rows = max_fit_rows

    def _fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n = len(y)
        sub = (rng.choice(n, self.max_fit_rows, replace=False)
               if self.max_fit_rows and n > self.max_fit_rows else np.arange(n))
        self.leaves_ = {}
        for j in self.missing_cols:
            col = X[sub, j]
            miss = np.isnan(col)
            if miss.all() or (~miss).all():
                self.leaves_[j] = [(-np.inf, np.inf, float(np.nanmedian(col)), 0.0)]
                continue
            fill = np.where(miss, np.nanmedian(col), col)
            dt = DecisionTreeClassifier(max_depth=self.max_depth,
                                        min_samples_leaf=self.min_samples_leaf,
                                        random_state=self.random_state)
            dt.fit(fill.reshape(-1, 1), y[sub])
            leaf_id = dt.apply(fill.reshape(-1, 1))
            t_miss = y[sub][miss]
            rules = []
            for leaf in np.unique(leaf_id):
                in_leaf = leaf_id == leaf
                if in_leaf.sum() < 5:
                    continue
                t_leaf = y[sub][in_leaf]
                # two-sample t-test leaf vs. missing group; small p = closer match
                try:
                    _, p_val = stats.ttest_ind(t_leaf, t_miss, equal_var=False)
                except Exception:
                    p_val = 1.0
                if not np.isfinite(p_val):
                    p_val = 1.0
                v_leaf = col[in_leaf]
                v_leaf = v_leaf[~np.isnan(v_leaf)]
                if len(v_leaf) == 0:
                    continue
                thr = dt.tree_.threshold[leaf]
                med = float(np.median(v_leaf))
                rules.append((float(thr), p_val, med))
            if not rules:
                rules = [(np.inf, 1.0, float(np.nanmedian(col)))]
            self.leaves_[j] = rules
        self.cost.n_params = self.max_depth * len(self.missing_cols)

    def _transform(self, X):
        out = X.copy()
        for j, rules in self.leaves_.items():
            m = np.isnan(out[:, j])
            if not m.any():
                continue
            # single-feature tree => leaf is identified by its split threshold
            # on the (median-filled) feature value.
            filled = np.where(m, np.nanmedian(out[:, j]) if not m.all() else 0.0, out[:, j])
            best_rule = (min(rules, key=lambda r: r[1]) if self.selection == "most_different"
                         else max(rules, key=lambda r: r[1]))
            val = best_rule[2]
            out[m, j] = val if np.isfinite(val) else 0.0
        return out


# --------------------------------------------------------------------------- #
REGISTRY = {
    "mean": MeanImputer,
    "median": MedianImputer,
    "indicator": IndicatorImputer,
    "decile": DecileImputer,
    "cluster": ClusterImputer,
    "tree": TreeImputer,
}

PAPER_ORDER = ["cluster", "decile", "mean", "median", "indicator", "tree"]


def get_imputer(key: str, missing_cols, **kw) -> BaseImputer:
    return REGISTRY[key](missing_cols, **kw)
