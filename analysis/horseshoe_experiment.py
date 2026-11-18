# R3-1: horseshoe global-local prior for feature selection via Polya-Gamma + Makalic-Schmidt Gibbs.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import scd_common as S
rng = np.random.default_rng(101)

def rpg1(z, rng, K=100):
    z = np.asarray(z, float); n = len(z); k = np.arange(1, K+1)
    denom = (k-0.5)**2 + (z[:, None]**2)/(4*np.pi**2)
    g = rng.standard_gamma(1.0, size=(n, K))
    return (1.0/(2*np.pi**2))*np.sum(g/denom, axis=1)

def ig(shape, scale, rng):   # inverse-gamma(shape, scale): 1/Gamma(shape, rate=scale)
    return 1.0/rng.gamma(shape, 1.0/np.maximum(scale, 1e-12))

def horseshoe_select(name, iters=700, burn=250, seed=7, nmax=6000):
    d = S.prep_all(name)
    # subsample the training fold to keep the Polya-Gamma Gibbs tractable
    tr = d["tr"]
    if len(tr) > nmax:
        tr = np.random.default_rng(seed).choice(tr, nmax, replace=False)
    Dsub, _, _ = S.design(d["X"].iloc[tr], d["spec"], d["num"], d["cat"])
    X0 = Dsub.copy()
    groups = [(nm, idx) for (nm, idx, od) in d["groups"]]      # feature -> column indices (intercept at 0)
    mu = X0[:, 1:].mean(0); sd = X0[:, 1:].std(0); sd[sd < 1e-8] = 1.0
    X = X0.copy(); X[:, 1:] = (X0[:, 1:]-mu)/sd
    y = d["y"][tr].astype(float); n, dim = X.shape
    kappa = y - 0.5
    # feature blocks (exclude intercept col 0)
    feats = [(nm, idx) for nm, idx in groups]
    beta = np.zeros(dim)
    lam2 = np.ones(len(feats)); nu = np.ones(len(feats)); tau2 = 1.0; xi = 1.0
    r = np.random.default_rng(seed)
    draws_kappa = []   # shrinkage weight 1-kappa per feature
    XtK = X.T @ kappa
    for t in range(iters):
        omega = rpg1(X @ beta, r)
        # prior precision: intercept ridge; each feature block ~ N(0, tau2*lam2_j * I)
        prec = np.zeros(dim); prec[0] = 1e-3
        for j, (nm, idx) in enumerate(feats):
            prec[idx] = 1.0/(tau2*lam2[j] + 1e-12)
        A = (X*omega[:, None]).T @ X + np.diag(prec)
        L = np.linalg.cholesky(0.5*(A+A.T) + 1e-9*np.eye(dim))
        m = np.linalg.solve(A, XtK)
        z = np.linalg.solve(L.T, r.standard_normal(dim))
        beta = m + z
        # update local scales lam2_j and tau2 (Makalic-Schmidt)
        T = 0.0
        for j, (nm, idx) in enumerate(feats):
            bj = float(np.sum(beta[idx]**2)); Bj = len(idx)
            lam2[j] = ig((Bj+1)/2.0, 1.0/nu[j] + bj/(2.0*tau2), r)
            nu[j] = ig(1.0, 1.0 + 1.0/lam2[j], r)
            T += bj/lam2[j]
        tau2 = ig((dim-1+1)/2.0, 1.0/xi + T/2.0, r)
        xi = ig(1.0, 1.0 + 1.0/tau2, r)
        if t >= burn:
            phi2 = tau2*lam2
            draws_kappa.append(1.0 - 1.0/(1.0+phi2))     # signal weight per feature
    W = np.array(draws_kappa)                             # (draws x n_feat)
    rows = []
    for j, (nm, idx) in enumerate(feats):
        incl = float(W[:, j].mean())
        pip = float((W[:, j] > 0.5).mean())
        rows.append(dict(feature=nm, inclusion_score=round(incl, 3), PIP=round(pip, 3)))
    return pd.DataFrame(rows).sort_values("inclusion_score", ascending=False)

for name in ["gmsc", "taiwan"]:
    print("="*50, name)
    tab = horseshoe_select(name)
    print(tab.to_string(index=False))
    tab.to_csv(f"results/tables/out_horseshoe_{name}.csv", index=False)
