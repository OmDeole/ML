# Executive summary — read this first

**Assignment.** Group 10: GAM-global synthetic data → Random Forest → null
imputation, i.e. recreate the paper's **Figure 14** and **Observations 8–9**.

**What we built.** A complete, seeded, reproducible pipeline: the paper's Eq. (2)
GAM-global generator (250K rows, 5 signal + 5 noise features, 5 segments, block
covariance r = 0.5, coefficients U(−3,3)), 50% null injection into x1–x3, the six
documented imputation methods, Random Forest, and an oracle-AUC benchmark. Nothing is
hard-coded: `scripts/` regenerates every figure and table.

**Headline.** The reconstruction reproduces the paper's *generative* anchors exactly
but **not** the ranking that Fig. 14 is built on:

| Anchor | Paper | Ours | Verdict |
|---|---|---|---|
| Oracle AUC | 0.9703 | 0.9703 | exact |
| Model ↔ oracle gap | 0.19 | 0.19 (per-segment coefficients) | structure confirmed |
| No-missing ceiling | 0.7789 | 0.7866 (capacity-calibrated) · 0.9316 (standard RF) | level reachable |
| Missingness penalty | −0.054 … −0.081 | −0.031 (standard RF) | direction only |
| Between-method spread | 0.028 | 0.0053 | **not reproduced** |
| Best method | missing indicator | Decile | differs |
| Worst method | tree | Tree | matches |

**Why.** Not because the methods agree — they do not. For the same three features the
inserted constants range from **−1.83 to +1.92** (audited in
`results/imputed_values.csv`): decile imputation lands near ±1 because the target rate
is U-shaped in |x1| and x2², and the tree rule lands at the *most different* leaf.
The reason the AUC does not move is the **model**: with five signal features, three of
them half-missing, a Random Forest routes around any constant and the surviving
information dominates. Imputation choice only matters when the affected feature is
irreplaceable — and here it is not. Nine alternative reconstructions (model family,
segment in/out, response scale, missingness mechanism, re-drawn missing features) all
stay within a 0.006 spread.

**What survives from the paper.** Observation 10 (tree imputation is worst) holds in
every regime. Observation 9 (single-point methods are equivalent) holds, more strongly
than the paper claims. Observation 8 (missing indicator is best) does **not** hold on
accuracy alone — but it *does* hold once cost is counted, which is the hook for the
extension.

**The extension (deployment study).** If accuracy is a tie, decide on deployment
criteria. We measured fit and scoring cost at production scale, accuracy across null
rates from 10% to 70%, and sensitivity to the missingness mechanism:

* **Fitting is where the methods separate**: at 60 features the clustering imputer needs
  7.3 s per retrain and tree imputation 1.24 s, against
  0.12 s for mean imputation and **0.00 s** for the missing indicator (it
  learns nothing). At the paper's Lending Club scale (80 features, 80% with nulls) that
  gap is the largest recurring cost in the pipeline.
* Scoring is dominated by the data copy, not the method (0.7–1.3 s
  per 1M rows at 60 features); the indicator's only penalty is 20 extra columns
  (+33% peak memory).
* The accuracy penalty is nearly **linear in the null rate** and the method ordering never
  changes — so with no accuracy difference the choice reduces to: **mean/median** if you do
  not need the missingness signal, **missing indicator** if you do. Both are at the cheap
  end of the fit-cost frontier; decile/tree/cluster are not.
* MCAR ≈ MAR; MNAR slightly *inflates* AUC because the null pattern becomes a label
  proxy — a deployment risk, not a benefit.
* Tree imputation's poor showing is explained by its rule, not by "trees are variable":
  the paper's literal instruction ("smallest t-score") selects the leaf *most different*
  from the missing group and is 0.005 AUC worse and 3× more variable than the
  "closest leaf" reading of the same paragraph.

**Bottom line.** The paper's *recommendation* survives, but not for the reason it
gives: the missing indicator does not beat single-point imputation on accuracy in a
faithful reimplementation — it ties it, at comparable cost, while keeping an auditable
"was it missing?" signal. Cluster and tree imputation buy nothing and cost 10–60× more
to fit; decile's 0.001 edge is inside the noise band.

