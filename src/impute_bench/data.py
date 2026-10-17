"""
Synthetic data generation for the replication of

    Johnson, Liu, Raza & McGuire (2023), "A Comparison of Modeling Preprocessing
    Techniques", 2nd Order Solutions, arXiv:2302.12042.

This module reconstructs the *null-imputation* arm of Appendix B.1, restricted to
the GAM-global functional form (the data set assigned to Group 10).

Everything that the paper states explicitly is implemented as stated; every value
that the paper leaves unspecified is exposed as a config field and flagged with
`ASSUMPTION` in its docstring so the reconstruction can be audited and varied.
See docs/RESEARCH_NOTES.md for the full traceability table.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Sequence

import numpy as np

# Numerical guards: the GAM-global form contains exp() and a reciprocal, which can
# overflow / divide by ~zero for extreme coefficient draws. The paper does not
# discuss this; we clip exponents and floor the denominator.
_EXP_CLIP = 30.0
_DENOM_FLOOR = 1e-6
_F_CLIP = 40.0


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    """Structural description of one GAM-global null-imputation data set.

    Paper anchors (Appendix B.1, Table 6):
        sample size 250K; 10 coefficients; 5 numeric features; 5 noise features;
        5 categorical segments; 3 features injected with 50% nulls.
    """

    n_rows: int = 250_000                     # paper: "sample size of 250K"
    n_segments: int = 5                       # Table 6: "Segments 5"
    n_signal: int = 5                         # Table 6: "Numeric Features 5" (x1..x5 of Eq. 2)
    n_noise: int = 5                          # Table 6: "Noise Features 5"
    corr: float = 0.5                         # Eq. 4: pairwise r = 0.5, block diagonal
    coef_low: float = -3.0                    # "ci in [-3, 3]"
    coef_high: float = 3.0
    # ASSUMPTION A3: the paper builds each data set "from the segment level" and
    # reports one coefficient count (10) for 5 segments. We read this as one
    # coefficient *vector* (10 terms of Eq. 2) drawn per segment. Set to
    # "shared" to instead draw a single vector used by all segments (sensitivity
    # analysis in scripts/calibrate_generator.py).
    coef_mode: str = "per_segment"            # "per_segment" | "shared"
    # ASSUMPTION A6: optional scaling of the log-odds. 1.0 = literal Eq. (2).
    # Values < 1 soften the response (lower oracle AUC); used to test whether the
    # paper's Fig. 14 is better explained by a well-fit model on lower-signal data.
    response_scale: float = 1.0
    # ASSUMPTION A4: which features receive nulls. Paper: "three features were
    # chosen". Names are the generated columns (x1..x10). Default = first three
    # signal features, i.e. 3 of the only 5 features that drive the response.
    missing_features: Sequence[str] = field(
        default_factory=lambda: ("x1", "x2", "x3")
    )
    missing_rate: float = 0.5                 # "null-rate of 50%"
    # ASSUMPTION A5: mechanism. Paper is silent; MCAR is the implicit default of
    # any experiment that injects nulls "in-place" at a fixed rate. The expansion
    # study adds MAR / MNAR / target-informative mechanisms.
    missing_mechanism: str = "mcar"
    train_frac: float = 0.60                  # paper uses 60% train splits (App. C.1)
    val_frac: float = 0.20
    seed: int = 20230224

    def to_dict(self) -> dict:
        d = asdict(self)
        d["missing_features"] = list(self.missing_features)
        return d

    @property
    def feature_names(self) -> list[str]:
        return [f"x{i}" for i in range(1, self.n_signal + self.n_noise + 1)]

    @property
    def block_size(self) -> int:
        return self.n_signal + self.n_noise


# --------------------------------------------------------------------------- #
# Functional form -- Eq. (2) of the paper
# --------------------------------------------------------------------------- #
def gam_global_terms(X5: np.ndarray, b: np.ndarray) -> np.ndarray:
    """GAM with global interactions, Eq. (2).

        f(x) = b1|x1| + b2 x2^2 + b3 log(|x3|+1) + exp(b4 x4) + (b5|x5|+1)^-1
               + b6 x1 x2 + b7 |x1 x2 x3|
               + b8 log(|x3+x4+x5|+1) + b9 max(x4, x5) + exp[b10 (x5-x3)]

    X5 : (n, 5) array of the five signal features
    b  : (10,) coefficient vector
    """
    x1, x2, x3, x4, x5 = (X5[:, i] for i in range(5))

    denom = b[4] * np.abs(x5) + 1.0
    # floor the denominator away from zero (guard for b5<0, |x5| ~ 1/|b5|)
    denom = np.where(np.abs(denom) < _DENOM_FLOOR,
                     np.sign(denom + 1e-12) * _DENOM_FLOOR, denom)

    terms = (
        b[0] * np.abs(x1)
        + b[1] * x2 ** 2
        + b[2] * np.log(np.abs(x3) + 1.0)
        + np.exp(np.clip(b[3] * x4, -_EXP_CLIP, _EXP_CLIP))
        + 1.0 / denom
        + b[5] * x1 * x2
        + b[6] * np.abs(x1 * x2 * x3)
        + b[7] * np.log(np.abs(x3 + x4 + x5) + 1.0)
        + b[8] * np.maximum(x4, x5)
        + np.exp(np.clip(b[9] * (x5 - x3), -_EXP_CLIP, _EXP_CLIP))
    )
    return np.clip(terms, -_F_CLIP, _F_CLIP)


def block_diag_cov(p: int, r: float = 0.5) -> np.ndarray:
    """Eq. (4): block-diagonal covariance, consecutive pairs with correlation r.

    For an odd number of features the final feature is left independent, as the
    paper states ("a variable independent of any other features").
    """
    Sigma = np.eye(p)
    for i in range(0, p - 1, 2):
        Sigma[i, i + 1] = Sigma[i + 1, i] = r
    return Sigma


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))  # overflow-safe


# --------------------------------------------------------------------------- #
# Missingness
# --------------------------------------------------------------------------- #
def make_missing_mask(n_rows: int, n_feats: int, cols: Sequence[int],
                      rate: float, rng: np.random.Generator,
                      mechanism: str = "mcar", y: np.ndarray | None = None,
                      X_signal: np.ndarray | None = None) -> np.ndarray:
    """Return a boolean mask (True = missing) for the whole feature matrix.

    MCAR  : i.i.d. Bernoulli(rate) per cell, per selected column.
    MAR   : probability depends on an *observed* column (x5), i.e. missingness is
            explainable from the data (logistic in x5, calibrated to `rate`).
    MNAR  : probability depends on the *unobserved* value itself (logistic in the
            feature's own z-score, calibrated to `rate`) -- the classic
            "high values are not reported" pattern.
    TARGET: probability depends on the true log-odds of y (informative missingness).
    """
    mask = np.zeros((n_rows, n_feats), dtype=bool)
    if rate <= 0:
        return mask

    if mechanism == "mcar":
        for c in cols:
            mask[:, c] = rng.random(n_rows) < rate
        return mask

    def _calibrate(score: np.ndarray) -> np.ndarray:
        """Shift/threshold so the marginal missing rate equals `rate` exactly."""
        thr = np.quantile(score, 1.0 - rate)
        return score > thr

    for c in cols:
        if mechanism == "mar":
            driver = X_signal[:, 4] if X_signal is not None else rng.standard_normal(n_rows)
            score = rng.random(n_rows) + 2.0 * sigmoid(driver)
            mask[:, c] = _calibrate(score)
        elif mechanism == "mnar":
            own = X_signal[:, min(c, X_signal.shape[1] - 1)]
            score = rng.random(n_rows) + 2.0 * sigmoid(own)
            mask[:, c] = _calibrate(score)
        elif mechanism == "target":
            assert y is not None
            score = rng.random(n_rows) + 2.0 * y
            mask[:, c] = _calibrate(score)
        else:
            raise ValueError(f"unknown mechanism {mechanism!r}")
    return mask


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    X: np.ndarray            # (n, p) float32, NaN where missing
    X_clean: np.ndarray      # (n, p) float32, never missing (for the control + oracle)
    y: np.ndarray            # (n,) int8
    p_true: np.ndarray       # (n,) float64, true P(Y=1|x) -> oracle AUC
    seg: np.ndarray          # (n,) int8
    miss_mask: np.ndarray    # (n, p) bool
    feature_names: list[str]
    coefs: np.ndarray        # (segments, 10)

    def split(self, train_frac=0.6, val_frac=0.2):
        n = len(self.y)
        n_tr = int(n * train_frac)
        n_va = int(n * val_frac)
        idx = {"train": np.arange(0, n_tr),
               "val": np.arange(n_tr, n_tr + n_va),
               "test": np.arange(n_tr + n_va, n)}
        return idx


def generate_gam_global(cfg: DataConfig, seed_offset: int = 0) -> Dataset:
    """Generate one GAM-global null-imputation data set.

    `seed_offset` re-draws the *sampling* of X, y and the missingness mask while
    keeping the calibrated coefficient draw fixed. That is the replication's
    iteration axis: all iterations share one oracle AUC (the paper annotates a
    single "Oracle AUC - 0.9703" on Fig. 14) but resample the data.
    """
    rng = np.random.default_rng(cfg.seed + 1000 * seed_offset)
    p = cfg.block_size
    names = cfg.feature_names

    # 1. features ~ N(0, Sigma), Eq. (4)
    Sigma = block_diag_cov(p, cfg.corr)
    X = rng.multivariate_normal(np.zeros(p), Sigma, size=cfg.n_rows).astype(np.float64)

    # 2. coefficients (ci ~ U[-3,3]); drawn from a stream independent of X so that
    #    changing `seed_offset` does not change the coefficients.
    coef_rng = np.random.default_rng(cfg.seed)
    if cfg.coef_mode == "shared":
        coefs = np.tile(coef_rng.uniform(cfg.coef_low, cfg.coef_high, 10), (cfg.n_segments, 1))
    else:
        coefs = coef_rng.uniform(cfg.coef_low, cfg.coef_high, (cfg.n_segments, 10))

    # 3. segments (the single categorical feature)
    seg = np.repeat(np.arange(cfg.n_segments),
                    int(np.ceil(cfg.n_rows / cfg.n_segments)))[: cfg.n_rows].astype(np.int8)
    rng.shuffle(seg)  # segments are not sorted in the file

    # 4. response, Eq. (2) evaluated within each segment
    f = np.empty(cfg.n_rows)
    for s in range(cfg.n_segments):
        m = seg == s
        f[m] = gam_global_terms(X[m, : cfg.n_signal], coefs[s])

    p_true = sigmoid(cfg.response_scale * f)
    y = rng.binomial(1, p_true).astype(np.int8)

    # 5. inject nulls
    cols = [names.index(nm) for nm in cfg.missing_features]
    mask = make_missing_mask(cfg.n_rows, p, cols, cfg.missing_rate, rng,
                             mechanism=cfg.missing_mechanism, y=y,
                             X_signal=X[:, : cfg.n_signal])
    X_miss = X.copy()
    X_miss[mask] = np.nan

    return Dataset(X=X_miss.astype(np.float32),
                   X_clean=X.astype(np.float32),
                   y=y, p_true=p_true, seg=seg, miss_mask=mask,
                   feature_names=names, coefs=coefs)


def design_matrix(ds: Dataset, cfg: DataConfig) -> np.ndarray:
    """Expand the single categorical segment feature into one-hot columns (N-1).

    Paper (App. A): standard preprocessing included OHE of categorical columns.
    """
    seg = ds.seg.astype(int)
    ohe = np.zeros((len(seg), cfg.n_segments - 1), dtype=np.float32)
    for k in range(1, cfg.n_segments):
        ohe[:, k - 1] = (seg == k).astype(np.float32)
    return ohe
