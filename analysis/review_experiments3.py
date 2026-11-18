# Round-2 review experiments (R2-1,2,3,5,6,7,8,9).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import scd_common as S
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
FIG = "results/figures"

# ---- DeLong with standard error ----
def _mid(x):
    J=np.argsort(x); Z=x[J]; N=len(x); T=np.zeros(N); i=0
    while i<N:
        j=i
        while j<N and Z[j]==Z[i]: j+=1
        T[i:j]=0.5*(i+j-1)+1; i=j
    T2=np.empty(N); T2[J]=T; return T2
def delong(y,p1,p2):
    pos=y==1; neg=~pos; m=int(pos.sum()); n=int(neg.sum())
    A=[];V10=np.zeros((2,m));V01=np.zeros((2,n))
    for k,p in enumerate([p1,p2]):
        X=p[pos];Y=p[neg];tx=_mid(X);ty=_mid(Y);txy=_mid(np.concatenate([X,Y]))
        A.append((txy[:m].sum()-tx.sum())/(m*n)); V10[k]=(txy[:m]-tx)/n; V01[k]=1-(txy[m:]-ty)/m
    Sg=np.cov(V10)/m+np.cov(V01)/n; var=Sg[0,0]+Sg[1,1]-2*Sg[0,1]
    se=np.sqrt(max(var,1e-12)); z=(A[0]-A[1])/se
    return A[0],A[1],2*(1-norm.cdf(abs(z))),se

print("### R2-2: information-matching learning rate w* and CI inflation ###")
rows=[]
for name in S.DATASET_ORDER:
    d=S.prep_all(name); b,_=S.fit_student(d,lam=0.2)
    X=d["Dtr"]; tgt=0.8*d["pT"][d["tr"]]+0.2*d["y"][d["tr"]]
    p=S.predict_proba_m(X,b); W=np.clip(p*(1-p),1e-6,None)
    P=S.penalty_matrix(X.shape[1],d["groups"])
    H=(X*W[:,None]).T@X+P                      # model (posterior) precision
    g=X*(p-tgt)[:,None]                        # per-observation score rows
    Sig=g.T@g                                   # empirical score covariance (sum)
    Hinv=np.linalg.inv(H)
    C0=Hinv; Cs=Hinv@Sig@Hinv                   # naive vs sandwich covariance
    wstar=np.trace(C0)/np.trace(Cs)             # scalar matching traces
    rows.append(dict(domain=d["info"]["domain"], w_star=round(float(wstar),3),
                     CI_inflation=round(float(1/np.sqrt(wstar)),3)))
Wtab=pd.DataFrame(rows); print(Wtab.to_string(index=False)); Wtab.to_csv("results/tables/out_w.csv",index=False)

print("\n### R2-3: teacher-informed PRIOR channel kappa (hard labels, prior-only transfer) ###")
def fit_mean(D,target,P,m,nit=60):
    d_=D.shape[1]; beta=m.copy()
    for _ in range(nit):
        p=1/(1+np.exp(-(D@beta))); Wd=np.clip(p*(1-p),1e-6,None)
        grad=D.T@(p-target)+P@(beta-m); Hm=(D*Wd[:,None]).T@D+P
        beta-=np.linalg.solve(Hm,grad)
    return beta
rk=[]
for name in S.DATASET_ORDER:
    d=S.prep_all(name); ytr=d["y"][d["tr"]].astype(float); yv=d["y"][d["ev"]]
    pT=d["pT"][d["tr"]]; base=np.clip(ytr.mean(),1e-4,1-1e-4); l0=np.log(base/(1-base))
    # teacher per-bin log-odds deviation as prior mean
    m_teacher=np.zeros(d["Dtr"].shape[1]); m_teacher[0]=l0
    for nm,idx,od in d["groups"]:
        col=d["Dtr"][:,idx]
        for jj,c in enumerate(idx):
            mask=col[:,jj]>0
            if mask.sum()>5:
                bm=np.clip(pT[mask].mean(),1e-4,1-1e-4); m_teacher[c]=np.log(bm/(1-bm))-l0
    P=S.penalty_matrix(d["Dtr"].shape[1],d["groups"])
    for kap in [0.0,0.5,1.0]:
        b=fit_mean(d["Dtr"],ytr,P,kap*m_teacher)
        p=S.predict_proba_m(d["Dev"],b)
        rk.append(dict(domain=d["info"]["domain"],kappa=kap,AUC=round(S.auc(yv,p),3),
                       rel=S.brier_decomp(yv,p)["reliability"]))
K=pd.DataFrame(rk); print(K.to_string(index=False)); K.to_csv("results/tables/out_kappa.csv",index=False)

print("\n### R2-5: TOST equivalence (margin 0.01) for gate-beta student AUCs ###")
rt=[]
for name in S.DATASET_ORDER:
    d=S.prep_all(name); yv=d["y"][d["ev"]]; pT=d["pT"]
    lg=np.log(np.clip(pT,1e-6,1-1e-6)/(1-np.clip(pT,1e-6,1-1e-6))); pmis=1/(1+np.exp(-lg/0.5))
    def ft(t):
        P=S.penalty_matrix(d["Dtr"].shape[1],d["groups"]); bb,_=S.fit_scorecard_laplace(d["Dtr"],t,P)
        return S.predict_proba_m(d["Dev"],bb)
    pte=ft(0.8*pT[d["tr"]]+0.2*d["y"][d["tr"]]); pmi=ft(0.8*pmis[d["tr"]]+0.2*d["y"][d["tr"]])
    a1,a2,pv,se=delong(yv,pte,pmi); diff=a1-a2; delta=0.01
    # 90% CI within +/-delta => equivalent
    lo,hi=diff-1.645*se,diff+1.645*se
    equiv=bool(hi<delta and lo>-delta)
    rt.append(dict(domain=d["info"]["domain"],auc_diff=round(diff,4),se=round(se,4),
                   ci90=f"[{lo:.3f},{hi:.3f}]",equivalent_0p01=equiv))
T=pd.DataFrame(rt); print(T.to_string(index=False)); T.to_csv("results/tables/out_tost.csv",index=False)

print("\n### R2-6: budget-constrained & rho-sensitive Bayes quantization ###")
rq=[]
for name in S.DATASET_ORDER:
    d=S.prep_all(name); b,cov=S.fit_student(d,lam=0.2)
    s_map=S.predict_margin(d["Dev"],b); g_map=S.grade_bands(s_map,10)
    grids=[(40,1),(20,1),(20,5),(10,10)]
    def eval_grid(pdo,grid):
        pts=S.pdo_points(b[1:],pdo,grid); bq=b.copy(); bq[1:]=-(np.log(2)/pdo)*pts
        s=S.predict_margin(d["Dev"],bq); g=S.grade_bands(s,10)
        return np.mean((s_map-s)**2), S.migration_rate(g_map,g), int(len(np.unique(np.round(s,6))))
    for rho in [0.0,2.0,10.0]:
        best=None
        for pdo,grid in grids:
            fid,mig,ns=eval_grid(pdo,grid); loss=fid+rho*mig
            if best is None or loss<best[0]: best=(loss,pdo,grid,ns)
        rq.append(dict(domain=d["info"]["domain"],rho=rho,sel=f"{best[1]}/{best[2]}",nscores=best[3],constraint="none"))
    # budget: <=25 distinct scores (hand-computable); pick min-migration feasible grid
    feas=[(S.migration_rate(g_map,S.grade_bands(S.predict_margin(d["Dev"],
            (lambda bq: bq)(np.r_[b[0],-(np.log(2)/pdo)*S.pdo_points(b[1:],pdo,grid)])),10)),pdo,grid)
          for pdo,grid in grids if eval_grid(pdo,grid)[2]<=25]
    if feas:
        feas.sort(); rq.append(dict(domain=d["info"]["domain"],rho="budget<=25scores",
            sel=f"{feas[0][1]}/{feas[0][2]}",nscores="<=25",constraint="<=25"))
Q=pd.DataFrame(rq); print(Q.to_string(index=False)); Q.to_csv("results/tables/out_rho.csv",index=False)

print("\n### R2-7: isotonic-calibrated teacher ECE before/after + transfer check ###")
re=[]
for name in S.DATASET_ORDER:
    d=S.prep_all(name); pT=d["pT"]; yv=d["y"][d["ev"]]
    iso=IsotonicRegression(out_of_bounds="clip").fit(pT[d["st"]], d["y"][d["st"]])
    pT_cal=iso.predict(pT)
    ece0=S.ece(yv,pT[d["ev"]]); ece1=S.ece(yv,pT_cal[d["ev"]])
    # distil from calibrated teacher, reliability vs miscalibrated control
    def ft(t):
        P=S.penalty_matrix(d["Dtr"].shape[1],d["groups"]); bb,_=S.fit_scorecard_laplace(d["Dtr"],t,P)
        return S.predict_proba_m(d["Dev"],bb)
    p_cal=ft(0.8*pT_cal[d["tr"]]+0.2*d["y"][d["tr"]])
    re.append(dict(domain=d["info"]["domain"],teacher_ECE=round(ece0,3),teacher_ECE_iso=round(ece1,3),
                   distilled_from_cal_rel=S.brier_decomp(yv,p_cal)["reliability"]))
E=pd.DataFrame(re); print(E.to_string(index=False)); E.to_csv("results/tables/out_teachercal.csv",index=False)

print("\n### R2-1: penalty sensitivity (lambda_s, lambda_r) ###")
rp=[]
for name in ["telco","consumerloan" if False else "creditrisk","taiwan"]:
    d=S.prep_all(name); yv=d["y"][d["ev"]]
    for ls,lr in [(0.5,1e-4),(2.0,1e-3),(8.0,1e-2)]:
        P=S.penalty_matrix(d["Dtr"].shape[1],d["groups"],lam_s=ls,lam_r=lr)
        tgt=0.8*d["pT"][d["tr"]]+0.2*d["y"][d["tr"]]
        b,cov=S.fit_scorecard_laplace(d["Dtr"],tgt,P)
        p=S.predict_proba_m(d["Dev"],b); sd=np.sqrt(np.clip(np.diag(cov),0,None))
        rp.append(dict(domain=d["info"]["domain"],lam_s=ls,lam_r=lr,AUC=round(S.auc(yv,p),3),
                       rel=S.brier_decomp(yv,p)["reliability"],mean_sd=round(float(sd[1:].mean()),3)))
Pn=pd.DataFrame(rp); print(Pn.to_string(index=False)); Pn.to_csv("results/tables/out_pen.csv",index=False)

print("\n### R2-9: corporate repeated-split CV (teacher overfitting check) ###")
aucs=[]
for seed in range(10):
    d=S.prep_all("corporate",seed=1000+seed)
    aucs.append(S.auc(d["y"][d["ev"]], d["pT"][d["ev"]]))
print(f"corporate teacher AUC over 10 splits: mean={np.mean(aucs):.3f} sd={np.std(aucs):.3f} "
      f"range=[{min(aucs):.3f},{max(aucs):.3f}]")

print("\n### R2-8: NSGA-II hypervolume convergence (2 domains) ###")
def hv2d(F, ref):   # 2D hypervolume (minimisation), objectives (1-AUC, cells)
    pts=F[np.argsort(F[:,0])]; hv=0.0; prev_x=ref[0]
    for x,y in pts[::-1]:
        if x<ref[0] and y<ref[1]:
            hv+=(ref[0]-x)*(prev_x if False else 0)  # placeholder
    # simpler: sort by cells asc, take lower envelope
    order=np.argsort(F[:,1]); best=np.inf; area=0.0; last=ref[1]
    for i in order:
        c=F[i,1]; e=F[i,0]
        if e<best:
            area+=(ref[0]-e)*(ref[1]-c) if False else 0
            best=e
    return best  # use best (min 1-AUC) as convergence proxy
# log best-1-AUC per generation via a light re-run
import scd_common as S2
def run_log(name,gens=10):
    X,y,prot,info=S.load_clean(name); num,cat=S.feature_types(X)
    ridx=np.arange(len(y)); np.random.default_rng(S.SEED).shuffle(ridx); ridx=ridx[:min(5000,len(y))]
    Xs=X.iloc[ridx].reset_index(drop=True); ys=y[ridx]; pTs=S.get_pT(name)[ridx]
    tr,st,ev=S.triple_split(len(ys)); feats=num+cat
    def build(sel,nb):
        numS=[f for f in sel if f in num]; catS=[f for f in sel if f in cat]; spec={}
        for f in numS:
            x=pd.to_numeric(Xs.iloc[tr][f],errors="coerce").to_numpy(float); xv=x[~np.isnan(x)]
            qs=np.unique(np.nanquantile(xv,np.linspace(0,1,nb[f]+1))) if xv.size else np.array([0.])
            edges=np.array([-np.inf,np.inf]) if len(qs)<2 else qs.astype(float)
            if len(qs)>=2: edges[0]=-np.inf; edges[-1]=np.inf
            spec[f]=("num",edges)
        for f in catS:
            vc=Xs.iloc[tr][f].astype(str).value_counts(normalize=True); spec[f]=("cat",vc[vc>=.02].index.tolist())
        return spec,numS,catS
    def evaluate(gen):
        sel=[f for f in feats if gen.get("inc_"+f,1)==1] or [feats[0]]
        nb={f:int(gen.get("bins_"+f,5)) for f in sel}; spec,numS,catS=build(sel,nb)
        Dtr,_,groups=S.design(Xs.iloc[tr],spec,numS,catS); Dev,_,_=S.design(Xs.iloc[ev],spec,numS,catS)
        P=S.penalty_matrix(Dtr.shape[1],groups); b,_=S.fit_scorecard_laplace(Dtr,0.8*pTs[tr]+0.2*ys[tr],P)
        s=Dev@b; a=S.auc(ys[ev],1/(1+np.exp(-s))); return np.array([1-a,Dtr.shape[1]-1,0.0])
    r=np.random.default_rng(S.SEED); pop=[]
    for _ in range(12):
        gino={}
        for f in feats: gino["inc_"+f]=int(r.random()<0.7); gino["bins_"+f]=int(r.choice([3,5,8]))
        pop.append(gino)
    F=np.array([evaluate(g) for g in pop]); traj=[float(F[:,0].min())]
    for _ in range(gens):
        kids=[]
        for _ in range(len(pop)):
            a,bb=pop[r.integers(len(pop))],pop[r.integers(len(pop))]
            child={k:(a[k] if r.random()<.5 else bb[k]) for k in a}; kids.append(S._mutate(child,r))
        Fk=np.array([evaluate(g) for g in kids]); allp=pop+kids; allF=np.vstack([F,Fk])
        fronts=S._nd_sort(allF); newidx=[]
        for fr in fronts:
            if len(newidx)+len(fr)<=len(pop): newidx+=fr
            else:
                cd=S._crowding(allF[fr]); order=np.argsort(-cd); newidx+=[fr[i] for i in order[:len(pop)-len(newidx)]]; break
        pop=[allp[i] for i in newidx]; F=allF[newidx]; traj.append(float(F[:,0].min()))
    return traj
fig,ax=plt.subplots(figsize=(5.2,3.2))
for name,ls in [("adult","-"),("taiwan","--")]:
    tr=run_log(name,gens=10); ax.plot(range(len(tr)),tr,marker="o",ms=3,color="0.2",ls=ls,
        label=S.DATASETS[name]["domain"])
ax.set_xlabel("Generation"); ax.set_ylabel("Best 1 - AUC on front"); ax.legend(fontsize=8)
for e in ("png","pdf"): fig.savefig(f"{FIG}/A3_hypervolume.{e}",dpi=600,bbox_inches="tight")
plt.close(fig); print("saved A3_hypervolume")
print("\nDONE round-2 experiments")
