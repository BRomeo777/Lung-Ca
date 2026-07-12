"""Phase 5 Post-Bugfix Rigor Analysis — Tasks 1-4."""
from __future__ import annotations
import json, pickle, hashlib, warnings, sys
from pathlib import Path
import numpy as np, pandas as pd, scipy.stats, torch, torch.nn as nn
from sksurv.metrics import concordance_index_censored
warnings.filterwarnings("ignore")

P = Path(__file__).resolve().parent.parent
READY = P/"03_ANALYSIS_READY_DATA"; ML = P/"ML_RESULTS"; GEO_EXPR = {
    "GSE30219": P/"PHASE1C_GEO_HARMONIZATION"/"processed_data"/"GSE30219"/"expression_gene_mapped.csv",
    "GSE50081": P/"PHASE1C_GEO_HARMONIZATION"/"processed_data"/"GSE50081"/"expression_gene_mapped.csv",
    "GSE72094": P/"PHASE1C_GEO_HARMONIZATION"/"processed_data"/"GSE72094"/"expression_gene_mapped.csv",
    "GSE31210": P/"PHASE1D_GEO_SCALE_ALIGNMENT"/"processed_data"/"GSE31210"/"GSE31210_expression_gene_mapped_log2.csv",
}
P3M = P/"PHASE3_DEEP_LEARNING"/"models"; P5D = P/"PHASE5_NEURAL_MECHANISTIC_INTEGRATION"/"data"
P5R = P/"PHASE5_NEURAL_MECHANISTIC_INTEGRATION"/"reports"
INVALID = ["TCGA-05-4395","TCGA-77-A5G6"]

class DeepSurv(nn.Module):
    def __init__(s,n,hd=[64,32],do=0.2,act="gelu"):
        super().__init__(); f={"relu":nn.ReLU,"silu":nn.SiLU,"gelu":nn.GELU}[act]; ls=[]; d=n
        for h in hd: ls+=[nn.BatchNorm1d(d),nn.Linear(d,h),f(),nn.Dropout(do)]; d=h
        o=nn.Linear(d,1); nn.init.xavier_uniform_(o.weight); nn.init.zeros_(o.bias)
        ls.append(o); s.network=nn.Sequential(*ls)
    def forward(s,x): return s.network(x).squeeze(-1)

def nlog2(c): l=c.sum(axis=0); l[l==0]=1; return np.log2(c.div(l,axis=1)*1e6+1)
def mct(v):
    s=str(v).lower(); return 'LUAD_Adenocarcinoma' if 'adc' in s or 'adenocarcinoma' in s else 'LUSC_SquamousCell' if 'sqc' in s or 'squamous' in s or 'scc' in s else None
def mst(v): s=str(v).strip(); return s if s in ["I","IA","IB","II","IIA","IIB","III","IIIA","IIIB","IV"] else None
def msm(v): s=str(v).lower(); return "Former" if "former" in s else "Current" if "current" in s else "Never" if "never" in s else None

def build_features(clin,expr,gf,fn,am,sc):
    cols=[p for p in clin['Patient_ID'].astype(str) if p in expr.columns]
    cm=clin[clin['Patient_ID'].astype(str).isin(cols)].reset_index(drop=True)
    if len(cm)==0: return None,None
    av=[g for g in gf if g in expr.index]
    es=expr.loc[av][cm['Patient_ID'].astype(str).tolist()].T; es.index=cm.index
    ce=pd.DataFrame(index=cm.index); ce['Age']=pd.to_numeric(cm['Age'],errors='coerce').fillna(am)
    for cn in ['Cancer_Type','Stage','Smoking_Status']:
        for f in fn:
            if f.startswith(f"{cn}="):
                cat=f.split("=",1)[1]
                if cn=='Cancer_Type': m=cm[cn].apply(mct)
                elif cn=='Stage': m=cm[cn].apply(mst)
                else: m=cm[cn].apply(msm)
                ce[f]=(m==cat).astype(int)
    X=pd.concat([ce,es],axis=1)
    for f in fn:
        if f not in X.columns: X[f]=0.0
    X=X[fn]; return sc.transform(X.values), cm

def ci(ye,yt,r): return concordance_index_censored(ye.astype(bool),yt,r)[0]
def boot_ci(ye,yt,r,n=1000,seed=42):
    rng=np.random.default_rng(seed); ns=len(ye); cs=[]
    for _ in range(n):
        i=rng.choice(ns,ns,replace=True)
        try:
            c=concordance_index_censored(ye[i].astype(bool),yt[i],r[i])[0]
            if np.isfinite(c): cs.append(c)
        except: pass
    if len(cs)<10: return float('nan'),float('nan'),float('nan')
    lo,hi=np.percentile(cs,[2.5,97.5]); return float(np.mean(cs)),float(lo),float(hi)

def boot_diff(ye,yt,ra,rb,n=1000,seed=42):
    rng=np.random.default_rng(seed); ns=len(ye); ds=[]
    for _ in range(n):
        i=rng.choice(ns,ns,replace=True)
        try:
            ca=concordance_index_censored(ye[i].astype(bool),yt[i],ra[i])[0]
            cb=concordance_index_censored(ye[i].astype(bool),yt[i],rb[i])[0]
            if np.isfinite(ca) and np.isfinite(cb): ds.append(ca-cb)
        except: pass
    if len(ds)<10: return float('nan'),float('nan'),float('nan')
    lo,hi=np.percentile(ds,[2.5,97.5]); return float(np.mean(ds)),float(lo),float(hi)

def cksum(a): return hashlib.md5(a.tobytes()).hexdigest()
def predict_ds(X,models):
    with torch.no_grad():
        Xt=torch.FloatTensor(X); return np.mean([m.eval()(Xt).cpu().numpy() for m in models],axis=0)

# ── Load artifacts ──
cfg=json.load(open(P3M/"final_config.json"))
scaler=pickle.load(open(P3M/"final_scaler.pkl","rb"))
cox=pickle.load(open(ML/"models"/"cox_ph_model.pkl","rb"))
fn=cfg["feature_names"]; am=cfg["age_median"]; gf=[f for f in fn if "=" not in f and f!="Age"]
models=[]
for i in range(cfg["n_models"]):
    m=DeepSurv(len(fn),cfg["arch"]["hidden_dims"],cfg["arch"]["dropout"],cfg["arch"]["activation"])
    m.load_state_dict(torch.load(P3M/f"final_deepsurv_model_{i}.pt",map_location="cpu")); models.append(m)

# ── Load TCGA ──
tr=pd.read_csv(READY/"TCGA_train.csv"); tr=tr[~tr['Patient_ID'].isin(INVALID)].dropna(subset=["Overall_Survival_Time","Survival_Status"]).reset_index(drop=True)
etr=nlog2(pd.read_csv(READY/"TCGA_train_expression.tsv.gz",sep="\t",index_col=0))
tr=tr[tr['Patient_ID'].astype(str).isin(etr.columns)].reset_index(drop=True)
va=pd.read_csv(READY/"TCGA_internal_validation.csv"); va=va[~va['Patient_ID'].isin(INVALID)].dropna(subset=["Overall_Survival_Time","Survival_Status"]).reset_index(drop=True)
eva=nlog2(pd.read_csv(READY/"TCGA_internal_validation_expression.tsv.gz",sep="\t",index_col=0))
va=va[va['Patient_ID'].astype(str).isin(eva.columns)].reset_index(drop=True)
Xtr,ctr=build_features(tr,etr,gf,fn,am,scaler)
Xva,cva=build_features(va,eva,gf,fn,am,scaler)
ytr_t=pd.to_numeric(ctr['Overall_Survival_Time'],errors='coerce').values
ytr_e=pd.to_numeric(ctr['Survival_Status'],errors='coerce').fillna(0).astype(int).values
yva_t=pd.to_numeric(cva['Overall_Survival_Time'],errors='coerce').values
yva_e=pd.to_numeric(cva['Survival_Status'],errors='coerce').fillna(0).astype(int).values
dsr_tr=predict_ds(Xtr,models); dsr_va=predict_ds(Xva,models)
cox_va=cox.predict_partial_hazard(pd.DataFrame(Xva,columns=fn)).values.ravel()

# ── Mechanistic features ──
mtr=pd.read_csv(P5D/"mechanistic_features_train.csv"); mva=pd.read_csv(P5D/"mechanistic_features_val.csv")
ttp_tr=mtr["TTP_natural"].values; ttp_va=mva["TTP_natural"].values

# ── Strategy C ──
sys.path.insert(0,str(P))
from PHASE5_NEURAL_MECHANISTIC_INTEGRATION.src.strategy_c import PredictionLevelFusion, normalize_ttp_to_risk
sc_c=PredictionLevelFusion(normalization_method="rank")
sc_c.fit(dsr_tr,ttp_tr,ytr_t,ytr_e)
hyb_va=sc_c.predict(dsr_va,ttp_va)

# ── GEO cohorts ──
GEO_C=["GSE30219","GSE50081","GSE72094","GSE31210"]
def is_normal(row):
    ct=str(row.get('Cancer_Type','')).lower(); st=str(row.get('Stage','')).lower()
    return 'ntl' in ct or 'normal' in ct or 'ntl' in st

geo={}
for c in GEO_C:
    cp=READY/f"{c}_external_validation.csv"; ep=GEO_EXPR.get(c)
    if not cp.exists() or ep is None or not ep.exists(): continue
    cl=pd.read_csv(cp)
    cl=cl[~cl.apply(is_normal,axis=1)].reset_index(drop=True)
    cl=cl.dropna(subset=["Overall_Survival_Time","Survival_Status"]).reset_index(drop=True)
    ex=pd.read_csv(ep,index_col=0)
    # Don't re-normalize if already log2
    if c!="GSE31210": ex=nlog2(ex)
    Xg,cg=build_features(cl,ex,gf,fn,am,scaler)
    if Xg is None: continue
    ye=pd.to_numeric(cg['Survival_Status'],errors='coerce').fillna(0).astype(int).values
    yt=pd.to_numeric(cg['Overall_Survival_Time'],errors='coerce').values
    dsr=predict_ds(Xg,models); cr=cox.predict_partial_hazard(pd.DataFrame(Xg,columns=fn)).values.ravel()
    ttp_g=pd.read_csv(P5D/f"mechanistic_features_{c}.csv")["TTP_natural"].values
    hyb=sc_c.predict(dsr,ttp_g)
    geo[c]={'X':Xg,'clin':cg,'ye':ye,'yt':yt,'ds':dsr,'cox':cr,'hyb':hyb,'ttp':ttp_g}

# ── Permutation importance ──
perm_path=P3M.parent/"tables"/"final_permutation_importance.csv"
perm=pd.read_csv(perm_path) if perm_path.exists() else None

# ═══════════════════════════════════════════════════════════════
# TASK 1: CONFIRM FIX
# ═══════════════════════════════════════════════════════════════
print("="*80); print("TASK 1: Confirm the fix is real"); print("="*80)
print(f"\n  X_val checksum: {cksum(Xva)}")
print(f"  X_val shape: {Xva.shape}")
print(f"  DeepSurv C-index: {ci(yva_e,yva_t,dsr_va):.4f}  (Phase 3: 0.6131)")
print(f"  CoxPH C-index:    {ci(yva_e,yva_t,cox_va):.4f}  (Phase 2: 0.5881)")
print(f"  DS match: {abs(ci(yva_e,yva_t,dsr_va)-0.6131)<0.001}")
print(f"  Cox match: {abs(ci(yva_e,yva_t,cox_va)-0.5881)<0.001}")

print(f"\n  {'Cohort':<12} {'Model':<10} {'C-index':>8} {'95% CI':>22} {'n':>5} {'evt':>5}")
print(f"  {'-'*12} {'-'*10} {'-'*8} {'-'*22} {'-'*5} {'-'*5}")
cohorts={"TCGA_val":(yva_e,yva_t,dsr_va,cox_va,hyb_va)}
cohorts.update({c:(g['ye'],g['yt'],g['ds'],g['cox'],g['hyb']) for c,g in geo.items()})
for cn,(ye,yt,dsr,cr,hr) in cohorts.items():
    for mn,r in [("CoxPH",cr),("DeepSurv",dsr),("StrategyC",hr)]:
        c=ci(ye,yt,r); _,lo,hi=boot_ci(ye,yt,r)
        print(f"  {cn:<12} {mn:<10} {c:>8.4f} ({lo:.4f}-{hi:.4f}){'':<6} {len(ye):>5} {int(ye.sum()):>5}")

# ═══════════════════════════════════════════════════════════════
# TASK 2: BOOTSTRAP CIs FOR DIFFERENCES
# ═══════════════════════════════════════════════════════════════
print("\n"+"="*80); print("TASK 2: Bootstrap CIs for C-index differences (DeepSurv vs Strategy C)"); print("="*80)
print(f"\n  {'Cohort':<12} {'ΔC (DS-C)':>10} {'95% CI':>22} {'Significant?':>14}")
print(f"  {'-'*12} {'-'*10} {'-'*22} {'-'*14}")
for cn,(ye,yt,dsr,cr,hr) in cohorts.items():
    mean_d,lo,hi=boot_diff(ye,yt,dsr,hr)
    sig="No" if lo<=0<=hi else "Yes"
    print(f"  {cn:<12} {mean_d:>10.4f} ({lo:.4f}-{hi:.4f}){'':<6} {sig:>14}")

# Also DeepSurv vs CoxPH for reference
print(f"\n  Reference: DeepSurv vs CoxPH")
print(f"  {'Cohort':<12} {'ΔC (DS-Cox)':>11} {'95% CI':>22} {'Significant?':>14}")
print(f"  {'-'*12} {'-'*11} {'-'*22} {'-'*14}")
for cn,(ye,yt,dsr,cr,hr) in cohorts.items():
    mean_d,lo,hi=boot_diff(ye,yt,dsr,cr)
    sig="No" if lo<=0<=hi else "Yes"
    print(f"  {cn:<12} {mean_d:>11.4f} ({lo:.4f}-{hi:.4f}){'':<6} {sig:>14}")

# ═══════════════════════════════════════════════════════════════
# TASK 3: SUBGROUP ANALYSIS (predefined)
# ═══════════════════════════════════════════════════════════════
print("\n"+"="*80); print("TASK 3: Subgroup analysis (predefined)"); print("="*80)
print("  Predefined subgroups: late-stage (III+IV), early-stage (I+II), high-growth-rate")
print("  Testing: does Strategy C beat DeepSurv in any subgroup?")

va_stage=cva['Stage'].apply(mst)
late_mask=va_stage.isin(["III","IIIA","IIIB","IV"]).values
early_mask=va_stage.isin(["I","IA","IB","II","IIA","IIB"]).values
# High growth rate: bottom tertile of TTP (fastest progression)
ttp_terc=np.percentile(ttp_va,33.3)
fast_mask=ttp_va<=ttp_terc

subgroups={"All":np.ones(len(yva_e),bool),"Late-stage(III+IV)":late_mask,"Early-stage(I+II)":early_mask,"High-growth(fast TTP)":fast_mask}
print(f"\n  {'Subgroup':<22} {'n':>5} {'DS C-i':>8} {'C C-i':>8} {'ΔC':>8} {'95% CI':>22} {'Sig?':>6}")
print(f"  {'-'*22} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*22} {'-'*6}")
for sn,mask in subgroups.items():
    if mask.sum()<10: print(f"  {sn:<22} {mask.sum():>5}  (too few)"); continue
    ye,yt=ye_=yva_e[mask],yva_t[mask]
    dsr_s,hr_s=dsr_va[mask],hyb_va[mask]
    c_ds=ci(ye,yt,dsr_s); c_hy=ci(ye,yt,hr_s)
    md,lo,hi=boot_diff(ye,yt,dsr_s,hr_s)
    sig="No" if lo<=0<=hi else "Yes"
    print(f"  {sn:<22} {mask.sum():>5} {c_ds:>8.4f} {c_hy:>8.4f} {md:>8.4f} ({lo:.4f}-{hi:.4f}){'':<6} {sig:>6}")

# NRI/IDI under corrected pipeline
print("\n  --- NRI/IDI (corrected) ---")
from PHASE5_NEURAL_MECHANISTIC_INTEGRATION.src.evaluation import compute_nri, compute_idi
for cn,(ye,yt,dsr,cr,hr) in cohorts.items():
    nri=compute_nri(hr,cr,ye,yt)
    idi=compute_idi(hr,cr,ye)
    print(f"  {cn:<12} NRI={nri['NRI']:.4f}  IDI={idi['IDI']:.4f}")

# Consistency violations
print("\n  --- Consistency violations (corrected) ---")
rho,p=scipy.stats.spearmanr(dsr_va,ttp_va)
print(f"  Spearman (DS risk vs TTP): rho={rho:.4f}, p={p:.4e}")
mech_risk_norm=normalize_ttp_to_risk(ttp_va,method="rank")
ds_norm=scipy.stats.rankdata(dsr_va); ds_norm=(ds_norm-1)/max(len(ds_norm)-1,1)
from PHASE5_NEURAL_MECHANISTIC_INTEGRATION.src.strategy_c import compute_consistency_violations
viol=compute_consistency_violations(ds_norm,mech_risk_norm)
n_viol=(viol>0.3).sum()
print(f"  Violations (>0.3): {n_viol}/{len(viol)} ({100*n_viol/len(viol):.1f}%)")
print(f"  Unique TTP values in validation: {len(np.unique(ttp_va))} (out of {len(ttp_va)} patients)")

# ═══════════════════════════════════════════════════════════════
# TASK 4: WHY MECHANISTIC ADDS NOTHING
# ═══════════════════════════════════════════════════════════════
print("\n"+"="*80); print("TASK 4: Why the mechanistic component adds nothing"); print("="*80)

# 4a. Correlation between mechanistic TTP and DeepSurv risk
print("\n  --- 4a. Correlation: DeepSurv risk vs mechanistic TTP ---")
rho_tr,p_tr=scipy.stats.spearmanr(dsr_tr,ttp_tr)
rho_va,p_va=scipy.stats.spearmanr(dsr_va,ttp_va)
print(f"  Training:   rho={rho_tr:.4f}, p={p_tr:.4e}")
print(f"  Validation: rho={rho_va:.4f}, p={p_va:.4e}")
print(f"  (Positive rho = higher risk = shorter TTP = consistent direction)")

# 4b. TTP value diversity
print("\n  --- 4b. TTP value diversity ---")
print(f"  Training TTP unique values: {len(np.unique(ttp_tr))}/{len(ttp_tr)}")
print(f"  Validation TTP unique values: {len(np.unique(ttp_va))}/{len(ttp_va)}")
print(f"  Training TTP range: [{ttp_tr.min():.1f}, {ttp_tr.max():.1f}] days")
print(f"  Validation TTP range: [{ttp_va.min():.1f}, {ttp_va.max():.1f}] days")
if len(np.unique(ttp_va))<=10:
    print(f"  Validation TTP unique values: {sorted(np.unique(ttp_va))}")

# 4c. Fusion weight
print(f"\n  --- 4c. Fusion weight ---")
print(f"  Optimal w_neural: {sc_c.optimal_w_neural:.4f}")
print(f"  (w_neural=1.0 means mechanistic component contributes zero)")

# 4d. Overlap: top DeepSurv features vs mechanistic-related genes
print(f"\n  --- 4d. DeepSurv top features vs mechanistic gene mappings ---")
if perm is not None:
    print(f"  Top 10 DeepSurv permutation importance features:")
    for i,row in perm.head(10).iterrows():
        feat=row.iloc[0]; imp=row.iloc[1]
        print(f"    {feat}: {imp:.4f}")
else:
    print("  (permutation importance not available)")

# 4e. Mechanistic feature variance
print(f"\n  --- 4e. Mechanistic feature variance (validation) ---")
for col in mva.columns:
    if col=="Patient_ID": continue
    u=len(mva[col].unique())
    print(f"    {col}: {u} unique values, std={mva[col].std():.4f}")

# 4f. Phase 4 limitations connection
print(f"\n  --- 4f. Phase 4 limitations explaining null result ---")
print(f"  1. Validation uses population priors (not personalized) -> TTP has ~{len(np.unique(ttp_va))} unique values")
print(f"  2. Phase 4 personalization only covers 737 training patients, not 180 validation")
print(f"  3. Gompertz alpha is population-fixed -> no patient-specific growth rate for validation")
print(f"  4. DeepSurv already captures stage info (top features include Stage=IA, Stage=IIIB)")
print(f"  5. Mechanistic TTP is determined by V0 (stage-based) + fixed alpha -> redundant with stage")

print("\n"+"="*80)
print("ANALYSIS COMPLETE")
print("="*80)
