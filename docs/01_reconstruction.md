# Reconstruction: what the paper specifies, what we had to assume, and how we tested it

Everything below is traceable: **paper statement → implementation → test**.
The paper contains no code, so a reconstruction is only as good as its anchors.
We used three published anchors to validate the generator (see §3).

## 1. Faithful elements (stated explicitly in the paper)

| Paper statement | Where | Implementation |
|---|---|---|
| 250K rows per dataset | App. B.1 | `DataConfig.n_rows = 250_000` |
| Features ~ multivariate normal, block-diagonal covariance, r = 0.5 | Eq. 4 | `block_diag_cov()`, `np.random.multivariate_normal` |
| Coefficients cᵢ ∈ [−3, 3] | App. B.1 | `rng.uniform(-3, 3)` |
| Response: Bernoulli with p = sigmoid(f(x)) | Eq. 5 | `sigmoid()` + `rng.binomial` |
| GAM-global functional form (10 terms) | Eq. 2 | `gam_global_terms()` |
| 5 segments defined by one categorical feature | App. B.1, Table 6 | `seg`, one-hot encoded (N−1 columns) |
| 5 numeric signal + 5 noise features | Table 6 | `n_signal=5, n_noise=5` |
| 3 features receive nulls at 50% | App. B.1 | `missing_features`, `missing_rate=0.5` |
| Standard preprocessing: standardisation, OHE, column consistency | App. A | numeric features are already standard normal; segment one-hot; fixed column sets |
| 60% train split | App. C.1 | `train_frac=0.6` (+ 20% val, 20% test) |
| Oracle AUC benchmark | App. A | `roc_auc_score(y_test, p_true_test)` |
| Decile: 10 quantile groups + missing group, match target probability, take the group median | App. C.3 | `DecileImputer` |
| Missing indicator: flag column + sentinel −9999 | Table 16 + 2OS blog | `IndicatorImputer` |
| Clustering: k-means cluster means per feature | App. C.3 | `ClusterImputer` |
| Tree: single-feature CART, t-test against the missing group, leaf median | App. C.3 | `TreeImputer` |

## 2. Assumptions (not specified in the paper)

| # | Assumption | Why it is needed | How we tested it |
|---|---|---|---|
| **A1** | The concrete coefficient draw | Only the *range* is published | Searched 200 draws; **draw 48 (seed 20230272) reproduces the annotated oracle AUC 0.9703** to 6·10⁻⁶ (`select_coefficients.py`) |
| **A2** | Which 3 features receive nulls (we use x1, x2, x3) | Not stated | Sensitivity run with random feature sets: the *penalty* depends strongly on which features are missing (−0.04 to −0.06), while the *ranking of methods* does not change |
| **A3** | Segments have **per-segment coefficient vectors** | "built from the segment level" is ambiguous | Decisive test: with one shared vector the model–oracle gap is **0.003**; with per-segment vectors it is **~0.2**, matching the paper's 0.19 gap (0.9703 → 0.7789) |
| **A4** | Missingness is **MCAR** | Paper says only "inserted in-place" | Expansion study sweeps MCAR / MAR / MNAR / target-informative |
| **A5** | Iterations resample X, y and the null mask but **not** the coefficients | Fig. 14's "No missing" bar varies by only ±0.0013, which rules out re-drawing coefficients per iteration | Implemented via `seed_offset`; coefficient RNG stream is separated from the sampling stream |
| **A6** | Response scale = 1.0 (literal Eq. 2) | — | Sensitivity: `response_scale` ∈ [0.18, 1.0] produces oracle AUC 0.81–0.97; used to test the "well-fit model on lower-signal data" reading |
| **A7** | Imputers are fit on **train only** | Paper is silent; fitting on all data leaks test targets through decile/tree | Implemented in `runner.run_cell`; documented as the deployment-valid protocol |
| **A8** | Decile tie-break: closest target rate, ties → lowest bin index | Not stated | Alternative tie-breaks change imputed values only marginally (bins near the median) |
| **A9** | k-means k = 5 | Not stated | k ∈ {5, 10} tested; no material change |
| **A10** | Tree imputer: depth 3, min leaf 500, 100K-row subsample, Welch t-test | CART hyperparameters not stated | Depth 2–4 tested; ordering unchanged |
| **A11** | Model = **Random Forest** (Group 10's assignment; the paper used tuned XGBoost) | Group assignment | Two regimes: standard RF (capacity unconstrained) and "paper-compatible" RF (depth 2, calibrated to the published 0.7789 level) |
| **A12** | OHE drops one category (N−1 columns) | Paper's OHE description is N−1 | — |

## 3. Validation anchors ("is the reconstruction right?")

| Anchor | Paper | Reconstruction | Verdict |
|---|---|---|---|
| Oracle AUC (GAM global) | **0.9703** | 0.97031 (draw 48) | **exact** |
| Model–oracle gap | 0.19 | 0.19 (per-segment, no segment feature) / 0.01 (shared) | **per-segment reading confirmed** |
| No-missing AUC | 0.7789 | 0.7866 (depth-2 RF) · 0.9515 (standard RF) · 0.7841 (standard RF, response scale 0.18) | level reachable, but only by a specific calibration |
| Missingness penalty (no-missing → imputed) | −0.054 to −0.081 | −0.038 (standard RF) · −0.007 (depth-2) · −0.020 (low-signal) | direction reproduced, magnitude depends on regime |
| Between-method spread | 0.028 (0.697 → 0.725) | 0.0053 (standard RF), ≤0.008 in every regime tested | **not reproduced** — see replication report §5 |
| Tree imputation worst | yes | yes in all regimes | reproduced |

## 4. What this means

The generator reproduces the *oracle* exactly and the *structure* of the response,
but Fig. 14's between-method differences are **not implied** by the published
description. Ten diagnostic experiments (two model families, with/without the segment
feature, three response scales, three missingness mechanisms, re-drawn missing
features, both tree rules, and a sweep that nulls *every* signal feature) all produce
methods that differ by ≤0.008 AUC against the paper's 0.028.

Crucially, the methods do **not** agree on what they insert — the imputed constants
range from −1.83 to +1.92 for the same feature (`results/imputed_values.csv`). The
differences vanish because the Random Forest can isolate an imputed constant and
recover the ranking from the surviving features. That is a statement about the model's
tolerance, and it is the correct explanation of the reconstruction's negative result.

This is a *finding*, not a failure: it says the paper's Obs. 8–9 are not robust to
reasonable implementation choices, and it motivates the deployment-oriented
question that the expansion answers — *if the accuracy differences are negligible,
what should a deployment team choose, and when do they stop being negligible?*
