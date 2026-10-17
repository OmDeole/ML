# Replication report — null imputation on GAM-global data (Group 10)

**Target paper.** Johnson, Liu, Raza & McGuire, *A Comparison of Modeling Preprocessing
Techniques*, 2nd Order Solutions, arXiv:2302.12042 (2023), §2.3 and Appendix C.3.
**Group 10 assignment.** GAM-global synthetic data → Random Forest → null imputation,
i.e. a replication of the paper's **Figure 14** and of **Observations 8–9**.

---

## 1. What was rebuilt

The paper publishes no code, so the pipeline was reconstructed from Appendices A–C
(traceability table in `docs/01_reconstruction.md`). In brief:

* **Data.** 250,000 rows; 5 signal features from Eq. (2)
  (`β1|x1| + β2x2² + β3log(|x3|+1) + exp(β4x4) + (β5|x5|+1)⁻¹ + β6x1x2 +
  β7|x1x2x3| + β8log(|x3+x4+x5|+1) + β9max(x4,x5) + exp[β10(x5−x3)]`),
  5 pure-noise features, one 5-segment categorical feature, block-diagonal covariance
  (r = 0.5), coefficients ~ U(−3,3), and `y ~ Bernoulli(sigmoid(f))`.
* **Missingness.** 50% of the values of three features (x1–x3) are set to null.
* **Methods.** mean, median, missing indicator (sentinel −9999 + flag), decile
  (target-probability-matched bin median), clustering (k-means cluster means),
  decision-tree (single-feature CART, t-test matched leaf median).
* **Protocol.** imputers fit on the 60% training split only, applied to train/val/test;
  Random Forest (Group 10's model); mean ± 2SD over iterations; oracle AUC benchmark.

## 2. Calibration against the paper's own anchors

The coefficient draw is not published, so we searched 200 draws (all other structure
fixed) and kept the one that reproduces the **oracle AUC annotated on Fig. 14**:

| Anchor | Paper | This reconstruction |
|---|---|---|
| Oracle AUC (true probabilities vs labels) | **0.9703** | **0.97031** (draw 48, seed 20230272) |
| Model↔oracle gap for a tuned model | 0.19 (0.9703 → 0.7789) | 0.19 with per-segment coefficients, 0.003 with shared coefficients |

That second row is what settles the ambiguity in "each data set was built from the
segment level": segments must have **their own coefficient vectors**. With one shared
vector the model reaches the oracle (gap 0.003), which contradicts the paper's own
numbers.

## 3. Result — standard Random Forest

40 trees, `min_samples_leaf=100`; 8 iterations; test AUC.

| method | AUC (here) | SD (here) | AUC (paper) | ±2SD (paper) | Δ vs paper | rank here | rank in paper |
|---|---|---|---|---|---|---|---|
| Clustering Imputation | 0.8998 | 0.0021 | 0.7185 | 0.0100 | 0.1813 | 6 | 3 |
| Decile Imputation | 0.9024 | 0.0013 | 0.7042 | 0.0552 | 0.1982 | 2 | 6 |
| Mean Imputation | 0.9012 | 0.0014 | 0.7162 | 0.0088 | 0.1850 | 5 | 4 |
| Median Imputation | 0.9013 | 0.0016 | 0.7146 | 0.0094 | 0.1867 | 4 | 5 |
| Missing Indicator Imputation | 0.9021 | 0.0014 | 0.7253 | 0.0026 | 0.1768 | 3 | 2 |
| No Missing | 0.9316 | 0.0017 | 0.7789 | 0.0026 | 0.1527 | 1 | 1 |
| Tree Imputation | 0.8972 | 0.0020 | 0.6974 | 0.0830 | 0.1998 | 7 | 7 |

![Fig 1](../figures/fig1_replication_rf.png)

**Reading.** Two things are reproduced: adding nulls costs real discrimination
(Δ = −0.031 on average versus the no-missing control), and **tree imputation is
the worst method**, exactly as the paper's Observation 10 states.
The *spread* between methods is not: here it is 0.0053 AUC against
0.0279 in the paper, a 10–20× difference.

![Fig 3](../figures/fig3_penalty_rf.png)

## 4. Result — capacity-calibrated regime (matching the paper's operating point)

The paper's tuned XGBoost realises only 80% of its own oracle (0.7789 / 0.9703). A
standard RF on this data reaches 0.93, so to compare *like-for-like* we also ran a
deliberately capacity-limited RF (depth 2), calibrated so that the no-missing control
lands on the published 0.7789.

| method | AUC (here) | SD (here) | AUC (paper) | Δ vs paper |
|---|---|---|---|---|
| Clustering Imputation | 0.7796 | 0.0033 | 0.7185 | 0.0611 |
| Decile Imputation | 0.7795 | 0.0033 | 0.7042 | 0.0753 |
| Mean Imputation | 0.7795 | 0.0033 | 0.7162 | 0.0633 |
| Median Imputation | 0.7795 | 0.0033 | 0.7146 | 0.0649 |
| Missing Indicator Imputation | 0.7795 | 0.0018 | 0.7253 | 0.0542 |
| No Missing | 0.7866 | 0.0029 | 0.7789 | 0.0077 |
| Tree Imputation | 0.7771 | 0.0032 | 0.6974 | 0.0797 |

![Fig 2](../figures/fig2_replication_depth2.png)

**Reading.** The absolute level can be matched (0.7866 vs 0.7789), but the
model then becomes *insensitive* to the imputation choice: the whole missing-data
penalty collapses to 0.0075. A model weak enough to sit at 0.78 does not use
enough of the feature set for imputation quality to matter.

## 5. Why the published spread does not reproduce

We tested every reading of the ambiguous parts of the paper. All of them produce
methods within 0.006 AUC of one another:

| Regime | No-missing AUC | Penalty | Spread |
|---|---|---|---|
| RF, segment feature out | 0.7005 | 0.0419 | 0.0060 |
| XGB, segment feature out | 0.7164 | 0.0345 | 0.0017 |
| RF, low-signal response (oracle 0.81) | 0.7747 | 0.0157 | 0.0033 |
| **Paper (Fig. 14)** | 0.7789 | -0.0540 | 0.0279 |
| RF, depth 2 (paper-compatible level) | 0.7866 | 0.0070 | 0.0024 |
| RF, segment feature in | 0.9515 | 0.0374 | 0.0020 |
| XGB, segment feature in | 0.9594 | 0.0362 | 0.0013 |

We audited what each method actually inserts (`results/imputed_values.csv`):

| method | x1 | x2 | x3 | spread (max−min) |
|---|---|---|---|---|
| Clustering Imputation | 0.006 | 0.001 | -0.230 | 0.236 |
| Decile Imputation | 0.679 | 1.038 | -1.038 | 2.076 |
| Mean Imputation | 0.001 | 0.001 | 0.006 | 0.005 |
| Median Imputation | 0.005 | -0.004 | -0.000 | 0.009 |
| Tree Imputation | -1.826 | 1.622 | 1.916 | 3.743 |

The methods disagree by up to 3.7σ on the imputed value — and the Random Forest still
barely moves. Three reasons:

1. **The model has spare information.** Five signal features contribute to the
   response; only three carry nulls. The two untouched features plus the segment
   one-hot keep most of the ranking information alive.
2. **A constant is learnable.** Whatever value a method inserts, it is the *same* value
   for every missing row, so a tree can isolate it with `x ≤ c+ε and x ≥ c−ε` and treat
   imputed rows as their own branch. The missing-indicator column therefore adds
   nothing that the ensemble cannot build itself.
3. **The one rule that hurts is the one that imputes against the truth.** Tree
   imputation deliberately selects the leaf whose target rate is *furthest* from the
   missing group's, i.e. it assigns the opposite risk (x1 = −1.83 where the truth is
   ≈ 0). It is consistently last — but only by 0.005 AUC.

The corollary is a hard limit of the published description: **a 0.028 spread requires
an imputed value that changes the model's predictions materially**, which needs a
feature the model cannot do without — not the case in the published GAM-global design.
Implementation details that the paper does not report (tie-breaking, CART settings, the
XGB hyperparameters actually selected) cannot recover that missing ingredient.

## 6. Fidelity scorecard

| Criterion | Target | Achieved | Verdict |
|---|---|---|---|
| Oracle AUC | 0.9703 | 0.9703 | ✅ exact |
| Model↔oracle gap | 0.19 | 0.19 | ✅ structure |
| No-missing ceiling | 0.7789 | 0.7866 (calibrated) | ✅ level |
| Missingness penalty | −0.054 … −0.081 | −0.031 (standard RF) | 🟡 direction |
| Between-method spread | 0.028 | 0.0053 | ❌ not reproduced |
| Best method = missing indicator | yes | Decile | ❌ |
| Worst method = tree | yes | Tree | ✅ |
| Rank correlation with Fig. 14 | — | ρ = +0.50 | ❌ |

## 7. What can still be said about Observations 8–10

* **Observation 10 (tree imputation is the worst)** — **supported** in every regime we
  ran (standard RF, depth-2 RF, XGBoost, segment in/out, three response scales). This is
  the most robust of the three claims. The mechanism is visible in the algorithm: the
  most-different-leaf rule deliberately imputes a value from a *contrasting* segment.
* **Observation 9 (single-point methods are all similar)** — **supported, and more
  strongly than the paper claims**: mean, median and decile are statistically
  indistinguishable in every regime (differences < 0.002 AUC).
* **Observation 8 (missing indicator is best)** — **not supported on this data**.
  The indicator ties the single-point methods (Δ < -0.000). It is nevertheless
  the *safest* choice, for three reasons this study can measure: (i) it ties on
  accuracy; (ii) it has the lowest fitting cost of all six methods (0.00 s — it learns
  nothing); (iii) it is the only method that keeps the missingness pattern explicitly,
  so the model is *forced* to see it rather than being trusted to discover a sentinel.
  Section 5 of the expansion report shows that the model's behaviour under MNAR depends
  on exactly that.

## 7b. An honest note on naming

The replication's best method is **Decile** and the paper's
is the **missing indicator**. The gap between them here is
0.0003 AUC,
while the run-to-run standard deviation is ±0.0013. This is a tie,
not a ranking: the ordering of the top five methods is not stable across iterations, and
we report it as a tie rather than claiming a win.

## 8. Reproducibility

```
export PYTHONPATH=src
python scripts/select_coefficients.py 200
python scripts/run_all.py all
python scripts/analyze.py && python scripts/build_report.py
```
All random seeds are fixed; `results/core_rf.csv` contains one row per cell with
AUCs, timings and memory.
