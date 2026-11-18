# Analysis scripts (review-round experiments)

These scripts reproduce the uncertainty-quantification and validation results in the
manuscript. They import the shared library `scd_common.py` and expect the six raw
datasets in `../data/` (see the top-level README).

- `scd_common.py` — shared library (loading, teacher, binning, Laplace posterior, NSGA-II, metrics).
- `review_experiments.py` — bootstrap CIs + DeLong; lambda sweep; Bayes-decision quantization; corrected equal-opportunity.
- `review_experiments2.py` — Polya-Gamma Gibbs vs Laplace; simulation-based calibration.
- `review_experiments3.py` — learning-rate / identification diagnostic; teacher-informed prior (kappa) ablation; TOST equivalence; rho / budget-constrained quantization; isotonic-calibrated teacher; penalty sensitivity; corporate split-sensitivity; NSGA-II convergence.
- `horseshoe_experiment.py` — horseshoe global-local variable selection (Polya-Gamma + Makalic-Schmidt Gibbs); per-feature inclusion scores.

Run from the repository root, e.g. `python analysis/horseshoe_experiment.py`.
