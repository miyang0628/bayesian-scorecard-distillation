# M2: validate the Laplace posterior against exact Polya-Gamma Gibbs, and run SBC.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import scd_common as S
import matplotlib.pyplot as plt
from review_experiments import pg_gibbs   # reuse the sampler

FIG = "results/figures"

# ---------- PG-Gibbs vs Laplace on a small domain (corporate, hard label) ----------
d = S.prep_all("corporate")
Dtr = d["Dtr"]; ytr = d["y"][d["tr"]].astype(float)
Pi = S.penalty_matrix(Dtr.shape[1], d["groups"], lam_s=2.0, lam_r=1e-3) + 1e-3*np.eye(Dtr.shape[1])
# Laplace
beta_map, cov = S.fit_scorecard_laplace(Dtr, ytr, Pi)
sd_lap = np.sqrt(np.diag(cov))
# PG-Gibbs (exact augmentation)
draws = pg_gibbs(Dtr, d["y"][d["tr"]].astype(int), Pi, iters=2000, burn=800, seed=11)
mean_pg = draws.mean(0); sd_pg = draws.std(0)
corr_mean = np.corrcoef(beta_map, mean_pg)[0,1]
corr_sd = np.corrcoef(sd_lap, sd_pg)[0,1]
print(f"[M2] PG vs Laplace  corr(mean)={corr_mean:.3f}  corr(sd)={corr_sd:.3f}  "
      f"median|sd_pg-sd_lap|/sd_pg={np.median(np.abs(sd_pg-sd_lap)/sd_pg):.3f}")

fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.6))
lim = max(np.abs(beta_map).max(), np.abs(mean_pg).max())*1.05
ax[0].plot([-lim,lim],[-lim,lim], color="0.7", lw=1, ls="--")
ax[0].scatter(mean_pg, beta_map, s=10, color="0.2")
ax[0].set_xlabel("Polya-Gamma posterior mean"); ax[0].set_ylabel("Laplace MAP")
lim2 = max(sd_pg.max(), sd_lap.max())*1.05
ax[1].plot([0,lim2],[0,lim2], color="0.7", lw=1, ls="--")
ax[1].scatter(sd_pg, sd_lap, s=10, color="0.2")
ax[1].set_xlabel("Polya-Gamma posterior SD"); ax[1].set_ylabel("Laplace posterior SD")
fig.tight_layout()
for e in ("png","pdf"): fig.savefig(f"{FIG}/A1_pg_vs_laplace.{e}", dpi=600, bbox_inches="tight")
plt.close(fig)

# ---------- Simulation-based calibration (SBC) for the Laplace posterior ----------
# Reduced design to keep the simulation light; prior = N(0, Pi^{-1}).
dd = S.prep_all("taiwan")
cols = list(range(min(16, dd["Dtr"].shape[1])))     # intercept + first bins
X = dd["Dtr"][:, cols]
Pi2 = S.penalty_matrix(X.shape[1], [(nm,[i for i in idx if i in cols],od) for (nm,idx,od) in dd["groups"]],
                       lam_s=2.0, lam_r=1e-2) + 1e-2*np.eye(X.shape[1])
Pinv = np.linalg.inv(Pi2); Lprior = np.linalg.cholesky(Pinv)
rng = np.random.default_rng(21)
R, Lpost = 300, 200; ranks = []
nsub = min(3000, X.shape[0]); Xs = X[:nsub]
for r in range(R):
    beta_true = Lprior @ rng.standard_normal(X.shape[1])
    p = 1/(1+np.exp(-(Xs @ beta_true))); y = (rng.random(nsub) < p).astype(float)
    bm, cv = S.fit_scorecard_laplace(Xs, y, Pi2)
    Lc = np.linalg.cholesky(0.5*(cv+cv.T) + 1e-9*np.eye(len(bm)))
    post = bm[None,:] + rng.standard_normal((Lpost, len(bm))) @ Lc.T
    ranks.append((post < beta_true[None,:]).sum(0))     # rank per component
ranks = np.array(ranks)                                   # R x d
# Aggregate ranks of a few coefficients into a histogram (should be uniform)
sel = ranks[:, 1:6].ravel()
fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.hist(sel, bins=20, color="0.5", edgecolor="0.2")
exp = len(sel)/20
ax.axhline(exp, color="0.15", lw=1, ls="--")
ax.set_xlabel("Rank statistic"); ax.set_ylabel("Count")
for e in ("png","pdf"): fig.savefig(f"{FIG}/A2_sbc_ranks.{e}", dpi=600, bbox_inches="tight")
plt.close(fig)
# chi-square uniformity check
from scipy.stats import chisquare
cnt,_ = np.histogram(sel, bins=20)
chi_p = chisquare(cnt).pvalue
print(f"[SBC] rank-histogram uniformity chi-square p={chi_p:.3f} (large = consistent with uniform)")
print("saved figures: A1_pg_vs_laplace, A2_sbc_ranks")
