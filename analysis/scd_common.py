# Shared library for the Bayesian Scorecard Distillation study (development copy).
# This exact content is embedded (verbatim) as the common prelude of each notebook.
import warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 20251118
rng = np.random.default_rng(SEED)

# ---- Grayscale, caption-free figure style (seaborn) ----
sns.set_theme(style="whitegrid", context="paper")
_GRAY = [str(v) for v in (0.15, 0.45, 0.65, 0.80)]
plt.rcParams.update({
    "image.cmap": "Greys", "axes.prop_cycle": plt.cycler(color=_GRAY),
    "axes.edgecolor": "0.2", "grid.color": "0.85", "savefig.bbox": "tight",
    "font.size": 10, "axes.grid": True, "legend.frameon": False,
})

# ---- Project root resolution (never printed, keeps paths anonymous) ----
def _root():
    p = Path.cwd().resolve()
    for q in (p, *p.parents):
        if (q / "results").is_dir() and (q / "notebooks").is_dir():
            return q
    return p
ROOT = _root()
FIGDIR, TABDIR, DATADIR = ROOT/"results"/"figures", ROOT/"results"/"tables", ROOT/"data"

def save_fig(fig, stem):
    # Save PNG and PDF at 600 dpi; no path is printed (anonymity).
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR/f"{stem}.{ext}", dpi=600)
    plt.close(fig)

def save_table(df, stem, index=False):
    # Persist a results table as CSV and LaTeX; no path is printed.
    df.to_csv(TABDIR/f"{stem}.csv", index=index)
    try:
        df.to_latex(TABDIR/f"{stem}.tex", index=index, escape=True,
                    float_format=lambda v: f"{v:.4f}")
    except Exception:
        pass

# ---- Dataset registry: target, columns to drop (leakage / duplicates), protected ----
DATASETS = {
    "telco":     dict(file="telco_churn.csv",      domain="Telecom churn"),
    "adult":     dict(file="adult.parquet",        domain="Income"),
    "corporate": dict(file="corporate_rating.csv", domain="Corporate rating"),
    "creditrisk":dict(file="credit_risk.csv",      domain="Consumer loan"),
    "gmsc":      dict(file="gmsc.parquet",          domain="Retail credit"),
    "taiwan":    dict(file="taiwan.parquet",        domain="Card default"),
}
DATASET_ORDER = ["telco", "adult", "corporate", "creditrisk", "gmsc", "taiwan"]

def _read(name):
    f = DATADIR/DATASETS[name]["file"]
    return pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)

def load_clean(name):
    """Return (X_df, y, protected, info). X_df holds raw features (numeric+categorical),
    y is a 0/1 numpy array, protected is a pandas Series or None."""
    df = _read(name).copy()
    prot = None
    if name == "telco":
        y = (df["Churn"] == "Yes").astype(int).to_numpy()
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        prot = df["gender"].astype(str)
        X = df.drop(columns=["customerID", "Churn"])
    elif name == "adult":
        y = df["HIGH_INCOME"].astype(int).to_numpy()
        prot = df["sex"].astype(str)
        X = df.drop(columns=["income", "HIGH_INCOME", "AGE_YEARS", "SEX_STD",
                             "EDU_STD", "VIP", "education"])
    elif name == "corporate":
        ig = {"AAA", "AA", "A", "BBB"}
        y = df["Rating"].isin(ig).astype(int).to_numpy()
        X = df.drop(columns=["Rating", "Name", "Symbol", "Rating Agency Name", "Date"])
    elif name == "creditrisk":
        y = df["loan_status"].astype(int).to_numpy()
        X = df.drop(columns=["loan_status"])
    elif name == "gmsc":
        y = df["DEFAULT"].astype(int).to_numpy()
        X = df.drop(columns=["DEFAULT", "VIP"])
    elif name == "taiwan":
        y = df["DEFAULT"].astype(int).to_numpy()
        prot = df["SEX"].map({1: "male", 2: "female"}).astype(str)
        X = df.drop(columns=["DEFAULT", "VIP", "AGE_YEARS", "SEX_STD"])
    else:
        raise ValueError(name)
    info = dict(domain=DATASETS[name]["domain"], n=len(y), p=X.shape[1],
               base_rate=float(y.mean()))
    return X, y, prot, info

def feature_types(X):
    # Robust across pandas dtypes (incl. pandas 3.0 'str'/'string'): numeric unless not.
    num = [c for c in X.columns
           if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    return num, cat

def to_lgb_frame(X):
    # Cast categoricals to pandas 'category' so LightGBM handles them natively.
    num, cat = feature_types(X)
    Z = X.copy()
    for c in cat:
        Z[c] = Z[c].astype("category")
    for c in num:
        Z[c] = pd.to_numeric(Z[c], errors="coerce")
    return Z, num, cat

# ---- Triple split: fit / stop(+tune) / eval (held out) ----
def triple_split(n, seed=SEED):
    idx = np.arange(n); r = np.random.default_rng(seed); r.shuffle(idx)
    a, b = int(0.6*n), int(0.8*n)
    return idx[:a], idx[a:b], idx[b:]

# ---- Teacher: high-capacity gradient boosting (interaction-capable) ----
def fit_teacher(Xtr, ytr, cat, seed=SEED):
    model = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=31, max_depth=-1,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        min_child_samples=50, reg_lambda=1.0, random_state=seed, n_jobs=-1, verbose=-1)
    model.fit(Xtr, ytr, categorical_feature=cat if cat else "auto")
    return model

# ---- Depth-one stump ensemble and its exact additive (pred_contrib) reduction ----
def fit_stump_additive(Xtr, ytr_target, cat, seed=SEED):
    # Depth-1 (num_leaves=2) logistic boosting on a soft or hard target in [0,1].
    dtr = lgb.Dataset(Xtr, label=ytr_target, categorical_feature=cat if cat else "auto",
                      free_raw_data=False)
    params = dict(objective="binary", num_leaves=2, max_depth=1, learning_rate=0.05,
                  min_child_samples=50, verbose=-1, seed=seed)
    booster = lgb.train(params, dtr, num_boost_round=300)
    return booster

def additive_contrib(booster, X):
    # SHAP-style exact additive decomposition: for depth-1 trees the per-feature
    # contributions sum exactly to the raw margin. Returns (contrib[:, :P], base).
    c = booster.predict(X, pred_contrib=True)
    return c[:, :-1], c[:, -1]

# ---- WoE-style binned design matrix (numeric: quantile bins; categorical: groups) ----
def make_bins(Xtr, num, cat, n_bins=5, rare=0.02):
    spec = {}
    for f in num:
        x = pd.to_numeric(Xtr[f], errors="coerce").to_numpy(dtype=float)
        xv = x[~np.isnan(x)]
        qs = np.unique(np.nanquantile(xv, np.linspace(0, 1, n_bins+1))) if xv.size else np.array([0.0])
        if len(qs) < 2:                      # constant / near-degenerate feature
            edges = np.array([-np.inf, np.inf])
        else:
            edges = qs.astype(float); edges[0] = -np.inf; edges[-1] = np.inf
        spec[f] = ("num", edges)
    for f in cat:
        vc = Xtr[f].astype(str).value_counts(normalize=True)
        keep = vc[vc >= rare].index.tolist()
        spec[f] = ("cat", keep)
    return spec

def design(X, spec, num, cat):
    cols, groups = [], []   # groups: list of (name, col_index_list, ordered?)
    mats = []
    for f in num:
        kind, edges = spec[f]
        x = pd.to_numeric(X[f], errors="coerce").to_numpy(dtype=float)
        b = np.digitize(np.where(np.isnan(x), edges[0], x), edges[1:-1], right=False)
        nb = len(edges) - 1
        M = np.zeros((len(x), nb)); M[np.arange(len(x)), b] = 1.0
        idx = list(range(len(cols), len(cols)+nb)); cols += [f"{f}#{k}" for k in range(nb)]
        groups.append((f, idx, True)); mats.append(M)
    for f in cat:
        kind, keep = spec[f]
        s = X[f].astype(str); cats = keep + ["__other__"]
        code = s.where(s.isin(keep), "__other__")
        M = np.zeros((len(s), len(cats)))
        for j, cval in enumerate(cats):
            M[(code == cval).to_numpy(), j] = 1.0
        idx = list(range(len(cols), len(cols)+len(cats))); cols += [f"{f}={c}" for c in cats]
        groups.append((f, idx, False)); mats.append(M)
    D = np.hstack(mats) if mats else np.zeros((len(X), 0))
    D = np.hstack([np.ones((len(X), 1)), D])            # intercept in column 0
    groups = [(nm, [i+1 for i in idx], od) for (nm, idx, od) in groups]
    return D, cols, groups

# ---- Penalty: RW1 smoothness on ordered (numeric) groups + tiny ridge ----
def penalty_matrix(d, groups, lam_s=2.0, lam_r=1e-3):
    P = np.zeros((d, d))
    for nm, idx, ordered in groups:
        if ordered and len(idx) >= 2:
            for a, b in zip(idx[:-1], idx[1:]):
                P[a, a] += lam_s; P[b, b] += lam_s; P[a, b] -= lam_s; P[b, a] -= lam_s
    for j in range(1, d):   # ridge on all but intercept
        P[j, j] += lam_r
    return P

# ---- Generalized distillation posterior via Laplace (MAP + inverse-Hessian) ----
def fit_scorecard_laplace(D, target, P, n_newton=60, tol=1e-9):
    d = D.shape[1]; beta = np.zeros(d)
    for _ in range(n_newton):
        eta = D @ beta; p = 1/(1+np.exp(-eta)); W = np.clip(p*(1-p), 1e-6, None)
        g = D.T @ (p - target) + P @ beta
        H = (D * W[:, None]).T @ D + P
        step = np.linalg.solve(H, g); beta -= step
        if np.max(np.abs(step)) < tol: break
    cov = np.linalg.inv(H)
    return beta, cov

def predict_margin(D, beta): return D @ beta
def predict_proba_m(D, beta): return 1/(1+np.exp(-(D @ beta)))

def posterior_sample(beta, cov, n, seed=SEED):
    L = np.linalg.cholesky(cov + 1e-10*np.eye(len(beta)))
    z = np.random.default_rng(seed).standard_normal((n, len(beta)))
    return beta[None, :] + z @ L.T

# ---- Metrics ----
def auc(y, s): return roc_auc_score(y, s)

def ece(y, p, bins=15):
    e = np.linspace(0, 1, bins+1); idx = np.digitize(p, e[1:-1]); tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any(): tot += m.mean()*abs(y[m].mean() - p[m].mean())
    return float(tot)

def brier_decomp(y, p, bins=15):
    # Murphy decomposition: Brier = Reliability - Resolution + Uncertainty.
    e = np.linspace(0, 1, bins+1); idx = np.digitize(p, e[1:-1])
    ybar = y.mean(); rel = 0.0; res = 0.0; n = len(y)
    for b in range(bins):
        m = idx == b; nk = m.sum()
        if nk:
            pk = p[m].mean(); ok = y[m].mean()
            rel += nk*(pk-ok)**2; res += nk*(ok-ybar)**2
    rel /= n; res /= n; unc = ybar*(1-ybar)
    return dict(reliability=float(rel), resolution=float(res), uncertainty=float(unc),
                brier=float(np.mean((p-y)**2)))

# ---- PDO quantization and grade banding ----
def pdo_points(c, pdo=40.0, grid=1.0):
    raw = -(pdo/np.log(2))*c
    return np.round(raw/grid)*grid

def grade_bands(scores, K=10):
    qs = np.quantile(scores, np.linspace(0, 1, K+1)); qs[0]=-np.inf; qs[-1]=np.inf
    return np.digitize(scores, qs[1:-1])

def migration_rate(g1, g2): return float(np.mean(g1 != g2))

# ---- Compact NSGA-II (minimisation of a vector of objectives) ----
def _nd_sort(F):
    n = len(F); S=[[] for _ in range(n)]; nd=np.zeros(n,int); rank=np.zeros(n,int); fronts=[[]]
    for p in range(n):
        for q in range(n):
            if all(F[p]<=F[q]) and any(F[p]<F[q]): S[p].append(q)
            elif all(F[q]<=F[p]) and any(F[q]<F[p]): nd[p]+=1
        if nd[p]==0: rank[p]=0; fronts[0].append(p)
    i=0
    while fronts[i]:
        nxt=[]
        for p in fronts[i]:
            for q in S[p]:
                nd[q]-=1
                if nd[q]==0: rank[q]=i+1; nxt.append(q)
        i+=1; fronts.append(nxt)
    return fronts[:-1]

def _crowding(F):
    n,m=F.shape; dist=np.zeros(n)
    for k in range(m):
        o=np.argsort(F[:,k]); dist[o[0]]=dist[o[-1]]=np.inf
        rng_=F[o[-1],k]-F[o[0],k] or 1.0
        for i in range(1,n-1): dist[o[i]]+=(F[o[i+1],k]-F[o[i-1],k])/rng_
    return dist

def nsga2(eval_fn, init_pop, n_gen=8, seed=SEED):
    r=np.random.default_rng(seed); pop=list(init_pop)
    F=np.array([eval_fn(ind) for ind in pop])
    for _ in range(n_gen):
        # variation: uniform crossover + bit/char mutation on dict-encoded genomes
        kids=[]
        for _ in range(len(pop)):
            a,b=pop[r.integers(len(pop))],pop[r.integers(len(pop))]
            child={k:(a[k] if r.random()<0.5 else b[k]) for k in a}
            kids.append(_mutate(child,r))
        Fk=np.array([eval_fn(ind) for ind in kids])
        allp=pop+kids; allF=np.vstack([F,Fk])
        fronts=_nd_sort(allF); newidx=[]
        for fr in fronts:
            if len(newidx)+len(fr)<=len(pop): newidx+=fr
            else:
                cd=_crowding(allF[fr]); order=np.argsort(-cd)
                newidx+=[fr[i] for i in order[:len(pop)-len(newidx)]]; break
        pop=[allp[i] for i in newidx]; F=allF[newidx]
    fronts=_nd_sort(F)
    return [pop[i] for i in fronts[0]], F[fronts[0]]

def _mutate(ind, r):
    out=dict(ind)
    for k in out:
        if k.startswith("inc_") and r.random()<0.15: out[k]=1-out[k]
        if k.startswith("bins_") and r.random()<0.15: out[k]=int(r.choice([3,5,8]))
    if not any(out[k] for k in out if k.startswith("inc_")):
        kk=[k for k in out if k.startswith("inc_")][0]; out[kk]=1
    return out

# ---- Teacher-probability cache + convenience builders (keep notebooks standalone) ----
CACHE = ROOT/"data"/"processed"
try: CACHE.mkdir(parents=True, exist_ok=True)
except Exception: pass

def get_pT(name, seed=SEED):
    # Cached teacher probabilities aligned to load_clean() row order (no path printed).
    f = CACHE/f"{name}_pT.parquet"
    if f.exists(): return pd.read_parquet(f)["pT"].to_numpy()
    X, y, prot, info = load_clean(name); Z, num, cat = to_lgb_frame(X)
    tr, st, ev = triple_split(len(y), seed); m = fit_teacher(Z.iloc[tr], y[tr], cat, seed)
    pT = m.predict_proba(Z)[:, 1]
    try: pd.DataFrame({"pT": pT}).to_parquet(f)
    except Exception: pass
    return pT

def prep_all(name, n_bins=5, seed=SEED):
    X, y, prot, info = load_clean(name); num, cat = feature_types(X)
    tr, st, ev = triple_split(len(y), seed); pT = get_pT(name, seed)
    spec = make_bins(X.iloc[tr], num, cat, n_bins=n_bins)
    Dtr, cols, groups = design(X.iloc[tr], spec, num, cat)
    Dev, _, _ = design(X.iloc[ev], spec, num, cat)
    return dict(name=name, X=X, y=y, prot=prot, info=info, num=num, cat=cat,
                tr=tr, st=st, ev=ev, pT=pT, spec=spec, Dtr=Dtr, Dev=Dev,
                cols=cols, groups=groups)

def fit_student(d, lam=0.3, lam_s=2.0, lam_r=1e-3):
    P = penalty_matrix(d["Dtr"].shape[1], d["groups"], lam_s, lam_r)
    target = (1-lam)*d["pT"][d["tr"]] + lam*d["y"][d["tr"]]
    return fit_scorecard_laplace(d["Dtr"], target, P)
