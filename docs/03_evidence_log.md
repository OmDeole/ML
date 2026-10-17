# Evidence log — every hypothesis tested while reconstructing Fig. 14

The paper has no code. Reconstructing it means deciding what the authors *did*, and a
decision only counts if it is tested. This log records each hypothesis, the experiment
that discriminated it, and the verdict. Everything is reproducible from `scripts/`.

---

### H1 — "Segments share one coefficient vector" vs. "each segment has its own"

**Test.** Generate both versions, fit XGBoost (200 trees) with the segment one-hot
included, compare the model→oracle gap with the paper's 0.19 gap.
**Result.** shared: gap **0.003**; per-segment: gap **0.197–0.274**.
**Verdict.** ✅ **Per-segment coefficients.** A shared vector makes the function
trivially learnable, which contradicts the paper's own numbers. This also explains the
paper's phrase "built from the segment level".

### H2 — "The paper's 0.78 ceiling is an underfit model"

**Test.** Sweep Random Forest capacity on the calibrated data (depth 2→None,
min_samples_leaf 1→1000) and find the capacity whose no-missing AUC equals 0.7789.
**Result.** depth 2 → **0.7866** (matches the level), but the missingness penalty
collapses to **−0.007** (paper: −0.054).
**Verdict.** 🟡 Explains the *level* only. A model that weak does not use enough
features for imputation quality to matter, so it cannot explain the paper's spread.

### H3 — "The paper's model is well-fit on lower-signal data"

**Test.** Add a response-scale parameter (log-odds multiplier). Scale 0.18 gives oracle
0.8067; fit standard RF + all methods.
**Result.** no-missing **0.7841** (≈ 0.7789 ✅), penalty **−0.020**, spread **0.002**.
**Verdict.** 🟡 Explains level + a realistic penalty, still no method spread. Also
inconsistent with the paper's *annotated* oracle of 0.9703.

### H4 — "The segment feature is not available to the model"

**Test.** Fit with and without the segment one-hot, XGBoost and RF.
**Result.** segment out: no-missing **0.7005 (RF) / 0.7164 (XGB)**; segment in:
0.9515 / 0.9594. Spread ≤ 0.006 either way.
**Verdict.** ❌ explains neither the level nor the spread; the paper's App. A explicitly
one-hot encodes categoricals, so the segment should be in.

### H5 — "The three null-injected features are re-drawn each iteration"

**Test.** Randomly choose 3 of the 10 features per iteration; 4 draws × 7 methods.
**Result.** Between-iteration SD jumps to **±0.022** for *every* method (draws that hit
`x1,x3,x5` cost 0.064 AUC; draws that hit noise features cost ~0.01).
**Verdict.** 🟡 Explains large **variance**, but not the *between-method* differences —
all methods move together.

### H6 — "Missingness is informative (MAR / MNAR / target-dependent)"

**Test.** Re-inject nulls with mechanisms that depend on an observed feature (MAR), on
the missing value itself (MNAR), and on the true label (target).
**Result.** MAR ≈ MCAR. MNAR slightly *better* than MCAR (the null pattern becomes a
signal). Target-dependent: **all** methods jump to ~0.9975 — the constant imputed value
is itself a label proxy, so the pattern leaks regardless of method.
**Verdict.** ❌ none of the mechanisms separates the methods.

### H7 — "The tree imputer's leaf rule is the source of its poor performance"

**Test.** The paper says "smallest **t-score**"; its companion blog says "smallest
**p-value**". Both mean the *most different* leaf; the surrounding prose says
"similarity" (closest leaf). Implement both and compare.
**Result.** See `results/tree_variant.csv`. Most-different → worse AUC and higher
between-iteration variance than closest, in line with the paper's Observation 10.
**Verdict.** ✅ the rule matters and explains *why* tree imputation is the worst method;
but it does not create a 0.03 spread.

### H8 — "The methods insert nearly the same value, so they must tie"

**Test.** Fit each imputer, then read the constant it inserts for each missing feature
(`results/imputed_values.csv`, `scripts/audit_imputed_values.py`).
**Result.** They are **not** the same at all:

| Method | x1 | x2 | x3 |
|---|---|---|---|
| Mean | 0.001 | 0.001 | 0.006 |
| Median | 0.006 | −0.004 | 0.000 |
| Clustering | 0.006 | 0.001 | −0.230 |
| Decile | 0.679 | 1.038 | −1.038 |
| Tree (most-different) | −1.826 | 1.622 | 1.916 |
| Indicator | −9999 (sentinel) | — | — |

**Verdict.** ❌ **rejected — and replaced by a better explanation.** The decile rule
does not land on the median: because |x1| and x2² make the target rate U-shaped, the
bin whose *target rate* matches the missing group is an intermediate bin, i.e. a value
around ±1. The tree rule, selecting the most different leaf, lands at ±1.6–1.9 — the
opposite of the missing group's risk. So the methods genuinely disagree about the
imputed value, by up to 3.7σ.
**What actually happens:** the Random Forest is insensitive to those differences. With
five signal features (three half-missing) the surviving information dominates, so
AUC moves by < 0.005 even when the imputed value moves by 3.7σ. That is the real
mechanism, and it is a statement about the *model's tolerance*, not about the methods
being identical.

### H9 — "The tree rule matters" (paper vs. its own blog)

**Test.** Implement both readings — the paper's "smallest t-score" / the blog's
"smallest p-value" (most different leaf) and the prose's "similarity" (closest leaf).
3 iterations each (`results/tree_variant.csv`).
**Result.** most-different **0.8969 ± 0.0017**; closest **0.9022 ± 0.0005**.
**Verdict.** ✅ the rule matters: the literal reading is 0.005 worse and **3× more
variable** — which reproduces the paper's Observation 10 (tree imputation worst and
most variable) as a consequence of an implementation choice, not of "single decision
trees are variable".

### H10 — "The paper simply did not damage the data enough"

**Test.** Sweep the number of null-affected *signal* features: 1, 2, 3 (the paper's
setting) and all 5, always at 50% nulls (`results/irreplaceability.csv`, 2 iterations).
**Result.**

| Signal features at 50% null | Penalty vs no-missing | Spread between methods |
|---|---|---|
| 1 | 0.0125 | 0.0016 |
| 2 | 0.0214 | 0.0035 |
| 3 (paper's setting) | 0.0296 | 0.0060 |
| **5 (every signal feature)** | **0.0954** | **0.0079** |

**Verdict.** ❌ Even destroying *all* the signal — three times the paper's damage —
produces a 0.0079 spread, 28% of the published 0.028. The divergence between the paper
and this reconstruction is not a matter of experiment intensity.

---

## What would have to be true for Fig. 14 to be reproducible

The published spread is **not** a consequence of the imputed values differing (they do:
up to 3.7σ apart, H8) but of those differences reaching the model's decisions. In a
five-signal-feature GAM-global design with three features damaged, they do not: a
Random Forest isolates the imputed constant and recovers almost all of the ranking from
the surviving features.

For Fig. 14 to be reproducible, at least one of the following would have to hold:

1. **The imputed constant is not separable from real values** — e.g. a heavily
   discretised feature where the imputed value coincides with real observations, so the
   model cannot branch on "imputed vs. real" and the *value* matters instead.
2. **The damaged features are jointly irreplaceable** — e.g. interactions that require
   all of them at once, so no surviving subset can reconstruct the ranking (our k = 5
   sweep is the closest the published design gets, and it still falls short).
3. **The imputation rule interacts with tree geometry** — e.g. an imputed value outside
   the training range so that every split lands differently (our tree imputer at ±1.9σ
   is the nearest case, and it moves AUC by only 0.005).
4. Implementation details that are not reported (tie-breaking, CART settings, the XGB
   hyperparameters selected by the Bayesian search) shift the numbers enough — we view
   this as unlikely given the size of the required effect.

This is the honest boundary of the replication, and it is what the expansion study
builds on: when accuracy cannot separate the methods, cost and robustness must.
