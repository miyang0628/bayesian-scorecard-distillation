# Additional experiments to address the first-round review (M2, M3, M4, minor).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, json, time
import scd_common as S
rng = np.random.default_rng(7)

# ---------- DeLong test for two correlated ROC AUCs ----------
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5*(i+j-1)+1
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2

def delong_auc_var(y, p1, p2):
    # Returns (auc1, auc2, p_value) for H0: AUC1==AUC2 on paired predictions.
    pos = y == 1; neg = ~pos
    m = int(pos.sum()); n = int(neg.sum())
    preds = [p1, p2]; aucs = []; V10 = np.zeros((2, m)); V01 = np.zeros((2, n))
    for k, p in enumerate(preds):
        X = p[pos]; Y = p[neg]
        tx = _midrank(X); ty = _midrank(Y); txy = _midrank(np.concatenate([X, Y]))
        aucs.append((txy[:m].sum() - tx.sum())/(m*n))
        V10[k] = (txy[:m] - tx)/n
        V01[k] = 1.0 - (txy[m:] - ty)/m
    S10 = np.cov(V10); S01 = np.cov(V01)
    Sall = S10/m + S01/n
    var = Sall[0, 0] + Sall[1, 1] - 2*Sall[0, 1]
    from scipy.stats import norm
    z = (aucs[0]-aucs[1])/np.sqrt(var) if var > 0 else 0.0
    pval = 2*(1-norm.cdf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(pval)

def boot_ci(stat_fn, n, reps=500, seed=1):
    r = np.random.default_rng(seed); vals = []
    for _ in range(reps):
        idx = r.integers(0, n, n); vals.append(stat_fn(idx))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(np.mean(vals)), float(lo), float(hi)

# ---------- Polya-Gamma sampler (truncated series) ----------
def rpg1(z, rng, K=200):
    # Sample PG(1, z) via truncated infinite convolution (vectorised over z).
    z = np.asarray(z, float); n = len(z)
    k = np.arange(1, K+1)
    denom = (k-0.5)**2 + (z[:, None]**2)/(4*np.pi**2)     # n x K
    g = rng.standard_gamma(1.0, size=(n, K))
    x = (1.0/(2*np.pi**2))*np.sum(g/denom, axis=1)
    return x

def pg_gibbs(X, y, Pi, iters=1500, burn=500, seed=3):
    rng = np.random.default_rng(seed); d = X.shape[1]
    beta = np.zeros(d); kappa = y - 0.5; draws = []
    XtK = X.T @ kappa
    for t in range(iters):
        f = X @ beta
        omega = rpg1(f, rng)
        V = np.linalg.inv((X*omega[:, None]).T @ X + Pi)
        m = V @ (XtK + Pi @ np.zeros(d))
        L = np.linalg.cholesky(0.5*(V+V.T) + 1e-10*np.eye(d))
        beta = m + L @ rng.standard_normal(d)
        if t >= burn: draws.append(beta.copy())
    return np.array(draws)

# ============================================================
print("### M4: bootstrap CIs + DeLong ###")
rowsA = []; rowsG = []
for name in S.DATASET_ORDER:
    d = S.prep_all(name); yv = d["y"][d["ev"]]
    pT = d["pT"][d["ev"]]
    b_h, _ = S.fit_student(d, lam=1.0); p_h = S.predict_proba_m(d["Dev"], b_h)
    b_d, cov = S.fit_student(d, lam=0.2); p_d = S.predict_proba_m(d["Dev"], b_d)
    n = len(yv)
    aT, lT, hT = boot_ci(lambda ix: S.auc(yv[ix], pT[ix]), n, seed=1)
    aH, lH, hH = boot_ci(lambda ix: S.auc(yv[ix], p_h[ix]), n, seed=2)
    aD, lD, hD = boot_ci(lambda ix: S.auc(yv[ix], p_d[ix]), n, seed=3)
    _, _, pTD = delong_auc_var(yv, pT, p_d)     # teacher vs distilled
    _, _, pHD = delong_auc_var(yv, p_h, p_d)     # baseline vs distilled
    rowsA.append(dict(domain=d["info"]["domain"],
        teacher=f"{aT:.3f} [{lT:.3f},{hT:.3f}]",
        baseline=f"{aH:.3f} [{lH:.3f},{hH:.3f}]",
        distilled=f"{aD:.3f} [{lD:.3f},{hD:.3f}]",
        delong_teach_vs_dist=round(pTD,3), delong_base_vs_dist=round(pHD,3)))
    # calibration gate beta: teacher-distilled vs miscalibrated-distilled, reliability diff CI
    lg = np.log(np.clip(d["pT"],1e-6,1-1e-6)/(1-np.clip(d["pT"],1e-6,1-1e-6)))
    pmis = 1/(1+np.exp(-lg/0.5))
    def fit_t(t):
        P = S.penalty_matrix(d["Dtr"].shape[1], d["groups"]); bb,_=S.fit_scorecard_laplace(d["Dtr"],t,P)
        return S.predict_proba_m(d["Dev"], bb)
    p_te = fit_t(0.8*d["pT"][d["tr"]]+0.2*d["y"][d["tr"]])
    p_mi = fit_t(0.8*pmis[d["tr"]]+0.2*d["y"][d["tr"]])
    _,_,pbeta = delong_auc_var(yv, p_te, p_mi)
    def reldiff(ix):
        return S.brier_decomp(yv[ix],p_te[ix])["reliability"]-S.brier_decomp(yv[ix],p_mi[ix])["reliability"]
    md, lo, hi = boot_ci(reldiff, n, seed=5)
    rowsG.append(dict(domain=d["info"]["domain"], delong_precondition_p=round(pbeta,3),
        rel_diff_teacher_minus_miscal=f"{md:.2e} [{lo:.2e},{hi:.2e}]",
        excludes_zero=bool(hi<0)))
print(pd.DataFrame(rowsA).to_string(index=False))
print()
print(pd.DataFrame(rowsG).to_string(index=False))
pd.DataFrame(rowsA).to_csv("results/tables/out_M4_ablation_ci.csv", index=False)
pd.DataFrame(rowsG).to_csv("results/tables/out_M4_gate_ci.csv", index=False)

# ============================================================
print("\n### minor: lambda sweep ###")
rl = []
for name in S.DATASET_ORDER:
    d = S.prep_all(name); yv = d["y"][d["ev"]]
    for lam in [0.0, 0.2, 0.5, 1.0]:
        b,_ = S.fit_student(d, lam=lam); p = S.predict_proba_m(d["Dev"], b)
        rl.append(dict(domain=d["info"]["domain"], lam=lam,
            AUC=round(S.auc(yv,p),3), rel=S.brier_decomp(yv,p)["reliability"]))
sweep = pd.DataFrame(rl); print(sweep.to_string(index=False))
sweep.to_csv("results/tables/out_lambda_sweep.csv", index=False)

# ============================================================
print("\n### M3: Bayes-decision quantization + grade-stability credible probability ###")
rowsQ = []
for name in S.DATASET_ORDER:
    d = S.prep_all(name); b, cov = S.fit_student(d, lam=0.2)
    draws = S.posterior_sample(b, cov, 300)
    s_map = S.predict_margin(d["Dev"], b); g_map = S.grade_bands(s_map, 10)
    # Bayes decision: choose grid minimising posterior expected loss with stability penalty
    best = None
    for pdo, grid in [(40,1),(20,1),(20,5),(10,10)]:
        # fidelity: E|a c - q|^2 approximated at MAP; stability via migration under posterior
        pts = S.pdo_points(b[1:], pdo, grid); bq = b.copy(); bq[1:] = -(np.log(2)/pdo)*pts
        s_q = S.predict_margin(d["Dev"], bq); g_q = S.grade_bands(s_q, 10)
        fidel = np.mean((s_map - s_q)**2)
        mig = S.migration_rate(g_map, g_q)
        loss = fidel + 2.0*mig            # rho = 2.0
        if best is None or loss < best[0]: best = (loss, pdo, grid, bq, g_q)
    _, pdo_s, grid_s, bq_s, g_q_s = best
    # grade-stability credible probability under the posterior (vs the Bayes-decision grades)
    stable = np.zeros(len(d["ev"]))
    for j in range(draws.shape[0]):
        gg = S.grade_bands(S.predict_margin(d["Dev"], draws[j]), 10)
        stable += (gg == g_q_s)
    stable /= draws.shape[0]
    rowsQ.append(dict(domain=d["info"]["domain"], bayes_PDO=pdo_s, bayes_grid=grid_s,
        mean_grade_stability_prob=round(float(stable.mean()),3),
        frac_cases_stable_ge_0p9=round(float((stable>=0.9).mean()),3)))
Q = pd.DataFrame(rowsQ); print(Q.to_string(index=False)); Q.to_csv("results/tables/out_M3_bayes_quant.csv", index=False)

# ============================================================
print("\n### minor: corrected equal-opportunity (condition on favourable label y=0) ###")
rf = []
for name in ["telco","adult","taiwan"]:
    d = S.prep_all(name); b,_ = S.fit_student(d, lam=0.2); p = S.predict_proba_m(d["Dev"], b)
    prot = d["prot"].to_numpy()[d["ev"]]; yv = d["y"][d["ev"]]
    appr = p <= np.quantile(p, 0.7); gs = pd.unique(prot)
    air = min(appr[prot==g].mean() for g in gs)/max(appr[prot==g].mean() for g in gs)
    # equal opportunity: approval rate among the favourable class (non-event, y=0)
    eo = {g: appr[(prot==g)&(yv==0)].mean() for g in gs}
    eod = abs(eo[gs[0]]-eo[gs[1]])
    rf.append(dict(domain=d["info"]["domain"], groups=",".join(map(str,gs)),
        AIR=round(float(air),3), EOD_y0=round(float(eod),3)))
F = pd.DataFrame(rf); print(F.to_string(index=False)); F.to_csv("results/tables/out_fairness_eod.csv", index=False)

print("\nDONE part 1")
