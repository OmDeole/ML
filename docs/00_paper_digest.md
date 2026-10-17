# Paper digest — "A Comparison of Modeling Preprocessing Techniques" (2OS, 2023)

**Reference.** T. Johnson, A. J. Liu, S. Raza, A. McGuire, *A Comparison of Modeling
Preprocessing Techniques*, 2nd Order Solutions, arXiv:2302.12042v2, 24 Feb 2023.
(White paper: 2os.com/insights; plain-language summary: 2os.medium.com; technical
follow-up on imputation: "Null Imputation Methods", 2OS Medium, Nov 2023.)

## 1. What problem does it attack?

Data scientists choose preprocessing steps (feature selection, categorical
encoding, missing-value handling) largely by habit. There are no agreed benchmarks.
2OS ran a large controlled horse race to produce empirical, quantitative guidance
for **tree-based binary classification on structured/tabular data**, with a focus on
XGBoost.

## 2. Experimental design

| Element | Design |
|---|---|
| Structure | test vs. **control** arm; control = standard preprocessing (column consistency, standardisation, one-hot encoding) |
| Model | XGBoost classifier, hyperparameters (gamma, learning rate, max depth, n estimators) tuned by **Bayesian optimisation** |
| Metric | **AUC**; for synthetic data the benchmark is the **"oracle" AUC** = AUC of the *true* generating probabilities vs. the sampled labels — the ceiling a model could reach |
| Iterations | each cell is run over many iterations (30 training sets); results are plotted as **mean ± 2 SD** to expose variability, not just the mean |
| Datasets | 3 synthetic generators + real-world Lending Club (2.26M rows, 70 numeric + 10 categorical, 80% of features have nulls) |

### Synthetic data generation (Appendix B.1)

* 250K rows per dataset; features drawn from a **multivariate normal** with a
  **block-diagonal covariance** (consecutive pairs correlate at r = 0.5, Eq. 4).
* Coefficients drawn uniformly in **[−3, 3]**.
* Response: `y ~ Bernoulli(p)`, `p = sigmoid(f(x))` — i.e. `f` is the log-odds.
* **Noise features** (pure random, no contribution) are injected to tempt the model.
* Three functional forms:
  * **Linear** — `Σ βᵢxᵢ` (Eq. 1)
  * **GAM global** — additive nonlinear terms **plus global interactions**
    (`x1x2`, `|x1x2x3|`, `max(x4,x5)`, `exp[β10(x5−x3)]`, …) (Eq. 2) ← **Group 10**
  * **Jumpy GAM local** — same idea but gated by indicator functions, so effects
    exist only in parts of the feature space (Eq. 3)
* Each experiment ("feature selection", "categorical encoding", "null imputation")
  had its own variant of the generator. 108 datasets in total.

## 3. The null-imputation experiment (Group 10's assignment)

* Uses the **basic** functional forms (Eqs. 1–3), 5 segments defined by one
  categorical feature, 10 coefficients, 5 numeric signal features, 5 noise
  features (Table 6).
* **Three features get 50% of their values replaced with nulls** (Tables 4–5).
* Six methods compared (Appendix C.3):
  mean, median, **missing indicator** (NA → −9999 plus a binary flag),
  **decile** (target-probability-matched bin median), **clustering** (k-means
  cluster means), **decision tree** (single-feature CART; the leaf whose target
  mean is closest to the missing group's donates its median).

### The three published observations

| # | Observation | Recommendation |
|---|---|---|
| 8 | Missing indicator imputation shows the greatest performance across **all** datasets, for both score and variability | excellent candidate for default |
| 9 | Single-point imputations (mean, median, decile) are all very similar | fine for simple models |
| 10 | Tree imputation has the most variable performance | **not recommended** |

### Numbers to reproduce (Fig. 14, GAM global)

Decoded from the published figure (the PDF embeds the plot as a raster, so the
values below were recovered by pixel-calibrating the y-axis):

| Method | AUC | ± 2 SD |
|---|---|---|
| Clustering | 0.7185 | 0.005 |
| Decile | 0.7042 | 0.028 |
| Mean | 0.7162 | 0.004 |
| Median | 0.7146 | 0.005 |
| **Missing indicator** | **0.7253** | 0.001 |
| No missing (ceiling) | 0.7789 | 0.001 |
| Tree | 0.6974 | **0.042** |
| *Oracle AUC (annotated)* | *0.9703* | — |

## 4. Other findings (context only — not our group's scope)

* Feature selection: XGB importance by **gain** is the most consistent; permutation
  importance, LASSO-style regularisation, and XGB weight are not recommended;
  correlation-reduction underperforms when the covariance is enforced.
* Categorical: frequency encoding fails on rigid/structured synthetic categories but
  wins on Lending Club; Helmert coding and OHE give identical tree models.
* Limitations the authors acknowledge: normal-only features, few features, XGB only,
  binary classification only, no interpretability analysis.

## 5. Why Group 10's slice is interesting

The imputation experiment is the one where **the missing-data mechanism is
unstated** and where the reported spread between methods (≈0.03 AUC) is largest
relative to the missingness penalty (≈0.05–0.08). It is therefore the slice where
"does the recommendation survive a faithful reimplementation?" is most meaningful —
and where a deployment-oriented extension (cost, null-rate budgets, mechanism
sensitivity) adds the most value.
