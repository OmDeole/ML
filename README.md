# Null-imputation benchmark — replication & extension of 2OS (2023)

Replication of the **null-imputation** experiment in

> **T. Johnson, A. J. Liu, S. Raza, A. McGuire.** *A Comparison of Modeling Preprocessing
> Techniques.* 2nd Order Solutions, [arXiv:2302.12042](https://arxiv.org/abs/2302.12042) (2023).

covering **Group 10's slice**: the GAM-global synthetic data set modelled with a **Random
Forest**, i.e. a reconstruction of the paper's **Figure 14** and **Observations 8–9**.

On top of the replication sits a **deployment-oriented extension**: preprocessing cost,
missing-data budgets, missingness mechanisms, and how irreplaceable a feature must be
before the imputation choice starts to matter.

---

## Headline

| Anchor | Paper | This repo | Verdict |
|---|---|---|---|
| Oracle AUC (annotated on Fig. 14) | **0.9703** | **0.97031** | ✅ exact |
| Model ↔ oracle gap | 0.19 | 0.19 | ✅ structure confirmed |
| No-missing ceiling | 0.7789 | 0.7866 *(capacity-calibrated)* · 0.9316 *(standard RF)* | ✅ level reachable |
| Missingness penalty | −0.054 … −0.081 | −0.031 *(standard RF)* | 🟡 direction only |
| Between-method spread | **0.028** | **0.0053** | ❌ **not reproduced** |
| Worst method = tree imputation | yes | yes | ✅ |

**The reconstruction reproduces the paper's *generator* exactly but not the *ranking*
that Figure 14 is built on.** This is the main finding, not a footnote: the published
differences between the six imputation methods are not implied by the published
description of the data or the methods.

### Why (the interesting part)

The methods do **not** agree on what they insert. Measured on the same three features,
the imputed constants range from **−1.83 to +1.92** — up to 3.7σ apart:

| Method | x1 | x2 | x3 |
|---|---|---|---|
| Mean | 0.001 | 0.001 | 0.006 |
| Median | 0.006 | −0.004 | 0.000 |
| Clustering | 0.006 | 0.001 | −0.230 |
| Decile | 0.679 | 1.038 | −1.038 |
| Tree (paper's literal rule) | −1.826 | 1.622 | 1.916 |

The Random Forest simply doesn't care: it isolates the imputed constant as its own
branch and rebuilds the ranking from the surviving features. Even nulling **all five**
signal features — three times the paper's damage — yields only a 0.008 spread.

Ten discriminating experiments are logged in
[`docs/03_evidence_log.md`](docs/03_evidence_log.md).

### What survives from the paper

* **Observation 10** (tree imputation is worst) — **holds in every regime tested**.
  It is explained by the paper's own rule ("smallest t-score" → the leaf *most different*
  from the missing group), not by "trees are variable".
* **Observation 9** (single-point methods are equivalent) — **holds**, more strongly
  than the paper claims.
* **Observation 8** (missing indicator is best) — **does not hold on accuracy**. It ties
  the single-point methods. It *does* hold once cost is counted, which is the hook for
  the extension.

---

## The extension: if accuracy is a tie, decide on deployment criteria

| Finding | Number |
|---|---|
| Fit cost separates the methods | At 60 features: **7.3 s** (clustering) vs **1.2 s** (tree) vs **0.12 s** (mean) vs **0.00 s** (missing indicator) — a 63× ratio |
| Scoring is dominated by the data copy, not the method | 0.7–1.3 s per 1M rows across all six |
| Accuracy penalty is ~linear in null rate | 10% → 70% nulls: −0.008 → −0.040 AUC, ordering never changes |
| MCAR ≈ MAR; MNAR slightly *inflates* AUC | The null pattern becomes a label proxy — a deployment **risk**, not a benefit |
| Irreplaceability threshold | Spread grows 0.0016 → 0.0079 as 1 → 5 signal features are damaged; still 28% of the paper's |

**Decision rule.** Impute with the cheapest single-point method unless the fact that a
value is missing is itself decision-relevant — in which case add the indicator column.
Do not pay for a learned imputer on data like this.

---

## Read the reports

| Document | Contents |
|---|---|
| **[reports/00_executive_summary.md](reports/00_executive_summary.md)** | one page, start here |
| [reports/02_replication_report.md](reports/02_replication_report.md) | the replication: calibration, results, why the spread doesn't reproduce, fidelity scorecard |
| [reports/03_expansion_report.md](reports/03_expansion_report.md) | the deployment study and decision rule |
| [reports/report.html](reports/report.html) | the two above combined, figures embedded, self-contained |
| [docs/00_paper_digest.md](docs/00_paper_digest.md) | structured digest of the paper |
| [docs/01_reconstruction.md](docs/01_reconstruction.md) | every design choice: paper statement → implementation → assumption → test |
| [docs/03_evidence_log.md](docs/03_evidence_log.md) | all 10 hypotheses tested, with verdicts |

*(GitHub renders the Markdown reports with their figures inline. `report.html` is for
reading offline as a single file.)*

---

## Quick start

```bash
python -m pip install -r requirements.txt
export PYTHONPATH=src

# 1. find the coefficient draw that reproduces the paper's oracle AUC (0.9703)
python scripts/select_coefficients.py 200

# 2. the replication itself
python scripts/run_from_config.py configs/core_gam_global_rf.json

# 3. everything: replication + null-rate sweep + mechanisms + cost benchmark
python scripts/run_all.py all

# 4. rebuild figures and reports from the results
python scripts/analyze.py && python scripts/build_report.py
```

Fast end-to-end check (~10 s):

```bash
python scripts/run_from_config.py configs/smoke.json
```

---

## Repository layout

```
configs/                 experiment configurations (JSON)
src/impute_bench/
  data.py                Eq. (2) GAM-global generator: block covariance, per-segment
                         coefficients, null injection (MCAR / MAR / MNAR / target)
  imputers.py            the six paper methods (mean, median, indicator, decile,
                         cluster, tree), each with fit/transform cost accounting
  modeling.py            Random Forest (default) / XGBoost / logistic regression
  runner.py              one cell = data → impute (fit on train only) → model → AUCs
  plots.py               paper-style figures
scripts/                 calibration, experiments, audits, analysis, report build
results/                 every CSV produced (raw runs + summaries + decoded Fig. 14)
figures/                 final figures
reports/                 the write-ups
docs/                    digest, reconstruction traceability, evidence log
```

---

## Design notes

**The generator is validated, not assumed.** The coefficient draw is not published, so
200 draws were searched and the one reproducing the annotated oracle AUC was kept
(draw 48, seed `20230272`). A second anchor settles an ambiguity the paper leaves open:
with one shared coefficient vector the model reaches the oracle (gap 0.003), contradicting
the paper's own numbers — so **segments must have their own coefficient vectors**.

**Two model regimes are reported.** The paper's tuned XGBoost realises only 80% of its
own oracle. A standard Random Forest on this data reaches 0.93, so the repo reports both
a standard RF and a capacity-calibrated (depth-2) RF that lands on the published 0.7789
operating point. The standard RF is the scientifically informative one; the calibrated
one is for like-for-like level comparison.

**Imputers are fit on the training split only.** The paper is silent on this, but a
decile or tree imputer fit on all data would leak test targets into the features.

**Why the paper's figures aren't in this repo.** The paper is not redistributed. Its
figures are rasterised inside the PDF, so `scripts/extract_paper_figure.py` re-creates
the comparison crop from a copy you provide yourself, and
[`results/paper_fig14_decoded.csv`](results/paper_fig14_decoded.csv) carries the values
recovered by pixel-calibrating the y-axis.

---

## Limitations

* One synthetic generator (GAM-global). The paper's linear and jumpy-GAM arms were not run.
* Random Forest primary (per the group assignment); the paper used tuned XGBoost.
* The six methods were implemented from prose; the alternative readings were tested
  (see the evidence log) without changing the conclusions.
* Costs measured single-threaded on a 2-core sandbox — ratios matter more than absolutes.
* The Lending Club arm (the paper's Fig. 13) is not included; it needs the ~1.5 GB Kaggle
  dataset.

---

## Citation

If you use this code, cite the original paper:

```bibtex
@article{johnson2023comparison,
  title   = {A Comparison of Modeling Preprocessing Techniques},
  author  = {Johnson, Tosan and Liu, Alice J. and Raza, Syed and McGuire, Aaron},
  journal = {arXiv preprint arXiv:2302.12042},
  year    = {2023},
  doi     = {10.48550/arXiv.2302.12042}
}
```

See [`CITATION.cff`](CITATION.cff). Code released under the [MIT licence](LICENSE).
