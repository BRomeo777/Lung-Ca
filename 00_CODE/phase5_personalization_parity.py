"""
Phase 5 Personalization Parity — Closes the personalization gap and definitively settles
whether mechanistic integration adds value when validation/GEO patients receive the same
gene-to-parameter personalization as training patients.

Steps 1-6 of the intensive final prompt.
"""
from __future__ import annotations
import json, pickle, hashlib, warnings, sys, time
from pathlib import Path
import numpy as np, pandas as pd, scipy.stats, torch, torch.nn as nn
from sksurv.metrics import concordance_index_censored
warnings.filterwarnings("ignore")

P = Path(__file__).resolve().parent.parent
READY = P/"03_ANALYSIS_READY_DATA"; ML = P/"ML_RESULTS"
P3M = P/"PHASE3_DEEP_LEARNING"/"models"; P3T = P/"PHASE3_DEEP_LEARNING"/"tables"
P4D = P/"PHASE4_MECHANISTIC_MODELING"/"data"; P4M = P/"PHASE4_MECHANISTIC_MODELING"/"models"
P5D = P/"PHASE5_NEURAL_MECHANISTIC_INTEGRATION"/"data"; P5R = P/"PHASE5_NEURAL_MECHANISTIC_INTEGRATION"/"reports"
P5S = P/"PHASE5_NEURAL_MECHANISTIC_INTEGRATION"/"src"
GEO_EXPR = {
    "GSE30219": P/"PHASE1C_GEO_HARMONIZATION"/"processed_data"/"GSE30219"/"expression_gene_mapped.csv",
    "GSE50081": P/"PHASE1C_GEO_HARMONIZATION"/"processed_data"/"GSE50081"/"expression_gene_mapped.csv",
    "GSE72094": P/"PHASE1C_GEO_HARMONIZATION"/"processed_data"/"GSE72094"/"expression_gene_mapped.csv",
    "GSE31210": P/"PHASE1D_GEO_SCALE_ALIGNMENT"/"processed_data"/"GSE31210"/"GSE31210_expression_gene_mapped_log2.csv",
}
INVALID = ["TCGA-05-4395","TCGA-77-A5G6"]
GEO_COHORTS = ["GSE30219","GSE50081","GSE72094","GSE31210"]

sys.path.insert(0, str(P4M))
sys.path.insert(0, str(P5S))
sys.path.insert(0, str(P))

from personalization_function import personalize_parameters, GENE_PARAM_MAPPINGS, STAGE_V0
from mechanistic_features import extract_mechanistic_features, MECH_FEATURE_NAMES
from strategy_a import FeatureLevelFusion
from strategy_b import PhysicsInformedTrainer
from strategy_c import PredictionLevelFusion, normalize_ttp_to_risk, compute_consistency_violations
from evaluation import compute_cindex, compute_nri, compute_idi, bootstrap_cindex

# ── Model ──
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
def is_normal(row):
    ct=str(row.get('Cancer_Type','')).lower(); st=str(row.get('Stage','')).lower()
    return 'ntl' in ct or 'normal' in ct or 'ntl' in st

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

def predict_ds(X,models):
    with torch.no_grad():
        Xt=torch.FloatTensor(X); return np.mean([m.eval()(Xt).cpu().numpy() for m in models],axis=0)

def personalize_patient(row, pid, expr, gene_features, priors, ensg2sym, scaler_mean, scaler_scale, feature_names, ds_risk=None):
    """Personalize ODE parameters for a single patient using gene expression."""
    patient_features = {
        'Age': pd.to_numeric(row.get('Age'), errors='coerce'),
        'Cancer_Type': row.get('Cancer_Type', 'LUAD_Adenocarcinoma'),
        'Stage': row.get('Stage', 'IIB'),
        'Smoking_Status': row.get('Smoking_Status', 'Former'),
    }
    # Add gene expression values (Ensembl IDs)
    if pid in expr.columns:
        for g in gene_features:
            if g in expr.index:
                patient_features[g] = float(expr.loc[g, pid])
    params, bounds = personalize_parameters(
        patient_features, priors, ensg2sym,
        scaler_mean=scaler_mean, scaler_scale=scaler_scale,
        feature_names=feature_names,
        deepsurv_risk=ds_risk,
    )
    return params

# ═══════════════════════════════════════════════════════════════
# LOAD ARTIFACTS
# ═══════════════════════════════════════════════════════════════
print("="*80); print("LOADING ARTIFACTS"); print("="*80)
cfg=json.load(open(P3M/"final_config.json"))
scaler=pickle.load(open(P3M/"final_scaler.pkl","rb"))
cox=pickle.load(open(ML/"models"/"cox_ph_model.pkl","rb"))
fn=cfg["feature_names"]; am=cfg["age_median"]; gf=[f for f in fn if "=" not in f and f!="Age"]
models=[]
for i in range(cfg["n_models"]):
    m=DeepSurv(len(fn),cfg["arch"]["hidden_dims"],cfg["arch"]["dropout"],cfg["arch"]["activation"])
    m.load_state_dict(torch.load(P3M/f"final_deepsurv_model_{i}.pt",map_location="cpu")); models.append(m)

priors_full=json.load(open(P4D/"population_priors.json"))
priors={k:v for k,v in priors_full.items() if not k.startswith("_")}
ensg_map=pd.read_csv(P/"PHASE1C_GEO_HARMONIZATION"/"reports"/"phase1B_ensg_to_symbol_mapping.csv")
ensg2sym=dict(zip(ensg_map['Ensembl_ID'],ensg_map['Gene_Symbol']))
sym2ensg={v:k for k,v in ensg2sym.items() if k in gf}

# The 13 personalization genes
persion_genes=list(GENE_PARAM_MAPPINGS.keys())
print(f"  Personalization genes ({len(persion_genes)}): {persion_genes}")
print(f"  Ensembl IDs mapped for these genes: {len([g for g in persion_genes if g in sym2ensg])}/{len(persion_genes)}")

# Load TCGA
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
print(f"  TCGA train: {len(tr)}, val: {len(va)}")
print(f"  DeepSurv C-index (val): {ci(yva_e,yva_t,dsr_va):.4f}")

# Load old (population-prior) mechanistic features for comparison
mtr_old=pd.read_csv(P5D/"mechanistic_features_train.csv")
mva_old=pd.read_csv(P5D/"mechanistic_features_val.csv")

# ═══════════════════════════════════════════════════════════════
# STEP 1: GENE COVERAGE AUDIT
# ═══════════════════════════════════════════════════════════════
print("\n"+"="*80); print("STEP 1: Gene Coverage Audit"); print("="*80)

# Check which Ensembl IDs for the 13 genes are in the DeepSurv feature names
gene_ensg_coverage={}
for gsym in persion_genes:
    ensg_id=sym2ensg.get(gsym)
    in_features = ensg_id in fn if ensg_id else False
    gene_ensg_coverage[gsym]={"ensg":ensg_id, "in_features":in_features}

print(f"\n  Gene-to-Ensembl mapping (for DeepSurv feature names):")
print(f"  {'Gene':<10} {'Ensembl ID':<25} {'In features?':>12}")
for gsym,info in gene_ensg_coverage.items():
    print(f"  {gsym:<10} {str(info['ensg']):<25} {'Yes' if info['in_features'] else 'No':>12}")

# Check coverage in each cohort's expression data
cohorts_expr={"TCGA_val":eva}
for c in GEO_COHORTS:
    ep=GEO_EXPR.get(c)
    if ep and ep.exists():
        cohorts_expr[c]=pd.read_csv(ep,index_col=0)

print(f"\n  Gene coverage by cohort (Ensembl IDs present in expression matrix):")
print(f"  {'Gene':<10} {'TCGA_val':>10}", end="")
for c in GEO_COHORTS: print(f" {c:>12}", end="")
print()
for gsym in persion_genes:
    ensg_id=sym2ensg.get(gsym)
    print(f"  {gsym:<10}", end="")
    for cn in ["TCGA_val"]+GEO_COHORTS:
        if cn in cohorts_expr:
            present = ensg_id in cohorts_expr[cn].index if ensg_id else False
            print(f" {'Yes':>10}" if present else f" {'No':>10}", end="")
        else:
            print(f" {'N/A':>10}", end="")
    print()

# Summary coverage
print(f"\n  Coverage summary:")
for cn in ["TCGA_val"]+GEO_COHORTS:
    if cn not in cohorts_expr: continue
    n_present=sum(1 for gs in persion_genes if sym2ensg.get(gs) and sym2ensg[gs] in cohorts_expr[cn].index)
    print(f"  {cn:<12}: {n_present}/{len(persion_genes)} genes ({100*n_present/len(persion_genes):.0f}%)")

# Root cause
print(f"\n  ROOT CAUSE: In run_phase5_integration.py lines 364-378, validation patients")
print(f"  receive dict(population_priors) with only stage-based V0. The function")
print(f"  personalize_parameters() is never called for validation/GEO patients.")
print(f"  This is an ENGINEERING OVERSIGHT, not a data limitation — the expression")
print(f"  data and scaler are available for all cohorts.")

# ═══════════════════════════════════════════════════════════════
# STEP 2: APPLY PERSONALIZATION
# ═══════════════════════════════════════════════════════════════
print("\n"+"="*80); print("STEP 2: Apply Personalization to Validation + GEO"); print("="*80)

# Personalize TCGA validation
print("  Personalizing TCGA validation patients...")
val_params_list=[]
n_personalized_val=0
for idx,row in cva.iterrows():
    pid=str(row['Patient_ID'])
    params=personalize_patient(row,pid,eva,gf,priors,ensg2sym,scaler.mean_,scaler.scale_,fn,ds_risk=float(dsr_va[idx]))
    # Check if any gene adjustments were applied (non-default alpha)
    if params["alpha"] != priors["alpha"]:
        n_personalized_val+=1
    val_params_list.append({"Patient_ID":pid, **params})
val_params_df=pd.DataFrame(val_params_list)
val_mech=extract_mechanistic_features(val_params_df,priors,n_mc=5,rng=np.random.default_rng(42))
val_mech.to_csv(P5D/"mechanistic_features_val_personalized.csv",index=False)
print(f"  Personalized {n_personalized_val}/{len(cva)} validation patients (gene adjustments applied)")
print(f"  Saved: mechanistic_features_val_personalized.csv")

# Personalize GEO
geo_data={}
geo_mech_personalized={}
geo_coverage_report={}
for c in GEO_COHORTS:
    ep=GEO_EXPR.get(c)
    if not ep or not ep.exists(): continue
    cl=pd.read_csv(READY/f"{c}_external_validation.csv")
    cl=cl[~cl.apply(is_normal,axis=1)].reset_index(drop=True)
    cl=cl.dropna(subset=["Overall_Survival_Time","Survival_Status"]).reset_index(drop=True)
    ex=pd.read_csv(ep,index_col=0)
    if c!="GSE31210": ex=nlog2(ex)
    Xg,cg=build_features(cl,ex,gf,fn,am,scaler)
    if Xg is None: continue
    ye=pd.to_numeric(cg['Survival_Status'],errors='coerce').fillna(0).astype(int).values
    yt=pd.to_numeric(cg['Overall_Survival_Time'],errors='coerce').values
    dsr=predict_ds(Xg,models); cr=cox.predict_partial_hazard(pd.DataFrame(Xg,columns=fn)).values.ravel()
    geo_data[c]={'X':Xg,'clin':cg,'ye':ye,'yt':yt,'ds':dsr,'cox':cr,'expr':ex}

    # Personalize
    n_pers=0
    params_list=[]
    for idx,row in cg.iterrows():
        pid=str(row['Patient_ID'])
        params=personalize_patient(row,pid,ex,gf,priors,ensg2sym,scaler.mean_,scaler.scale_,fn,ds_risk=float(dsr[idx]))
        if params["alpha"] != priors["alpha"]: n_pers+=1
        params_list.append({"Patient_ID":pid, **params})
    gparams_df=pd.DataFrame(params_list)
    gmf=extract_mechanistic_features(gparams_df,priors,n_mc=0,rng=np.random.default_rng(42),skip_mc=True)
    gmf.to_csv(P5D/f"mechanistic_features_{c}_personalized.csv",index=False)
    geo_mech_personalized[c]=gmf
    geo_coverage_report[c]={"n_total":len(cg),"n_personalized":n_pers,"pct":100*n_pers/len(cg) if len(cg)>0 else 0}
    print(f"  {c}: personalized {n_pers}/{len(cg)} ({100*n_pers/len(cg):.0f}%)")

# ── Before/After variance comparison ──
print(f"\n  --- Before/After Variance Comparison ---")
print(f"  {'Cohort':<12} {'Feature':<25} {'Before':>10} {'After':>10} {'Before':>10} {'After':>10}")
print(f"  {'':12} {'':25} {'unique':>10} {'unique':>10} {'std':>10} {'std':>10}")
print(f"  {'-'*12} {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

def variance_report(cn, old_df, new_df, features_to_check):
    for feat in features_to_check:
        old_u=len(old_df[feat].unique()) if feat in old_df.columns else 0
        new_u=len(new_df[feat].unique()) if feat in new_df.columns else 0
        old_s=old_df[feat].std() if feat in old_df.columns else 0
        new_s=new_df[feat].std() if feat in new_df.columns else 0
        print(f"  {cn:<12} {feat:<25} {old_u:>10} {new_u:>10} {old_s:>10.4f} {new_s:>10.4f}")

key_features=["TTP_natural","growth_rate_alpha","k_immune_log","initial_volume_log","mu_E","rho"]
variance_report("TCGA_val", mva_old, val_mech, key_features)
for c in GEO_COHORTS:
    if c not in geo_mech_personalized: continue
    old_g=pd.read_csv(P5D/f"mechanistic_features_{c}.csv")
    variance_report(c, old_g, geo_mech_personalized[c], key_features)

# ═══════════════════════════════════════════════════════════════
# STEP 3: RE-RUN ALL 3 FUSION STRATEGIES WITH PERSONALIZED INPUT
# ═══════════════════════════════════════════════════════════════
print("\n"+"="*80); print("STEP 3: Re-run All 3 Fusion Strategies with Personalized Input"); print("="*80)

# Get personalized TTP for training (already personalized from Phase 4)
ttp_tr=mtr_old["TTP_natural"].values  # Training was already personalized
ttp_va_pers=val_mech["TTP_natural"].values
mech_tr_arr=mtr_old[MECH_FEATURE_NAMES].values
mech_va_pers_arr=val_mech[MECH_FEATURE_NAMES].values

# Strategy C: Prediction-Level Fusion (re-fit weight)
print("  Fitting Strategy C (prediction-level fusion)...")
sc_c=PredictionLevelFusion(normalization_method="rank")
sc_c.fit(dsr_tr,ttp_tr,ytr_t,ytr_e)
hyb_va_c=sc_c.predict(dsr_va,ttp_va_pers)
print(f"  Strategy C w_neural: {sc_c.optimal_w_neural:.4f}")

# Strategy A: Feature-Level Fusion (retrain)
print("  Training Strategy A (feature-level fusion)...")
t0=time.time()
strat_a=FeatureLevelFusion(n_models=2,seeds=(123,456))
strat_a.fit(Xtr,mech_tr_arr,ytr_t,ytr_e,n_epochs=200)
hyb_va_a=strat_a.predict(Xva,mech_va_pers_arr)
print(f"  Strategy A trained in {time.time()-t0:.1f}s")

# Strategy B: Physics-Informed (retrain)
print("  Training Strategy B (physics-informed)...")
t0=time.time()
strat_b=PhysicsInformedTrainer(input_dim=len(fn),n_models=2,seeds=(123,456))
strat_b.fit(Xtr,ytr_t,ytr_e,ttp_tr,n_epochs=200)
hyb_va_b=strat_b.predict(Xva)
print(f"  Strategy B trained in {time.time()-t0:.1f}s")

# Evaluate on GEO
for c in GEO_COHORTS:
    if c not in geo_data: continue
    gd=geo_data[c]; gmf=geo_mech_personalized[c]
    ttp_g=gmf["TTP_natural"].values
    mech_g_arr=gmf[MECH_FEATURE_NAMES].values
    gd['hyb_c']=sc_c.predict(gd['ds'],ttp_g)
    gd['hyb_a']=strat_a.predict(gd['X'],mech_g_arr)
    gd['hyb_b']=strat_b.predict(gd['X'])
    gd['ttp_pers']=ttp_g

# ── Performance comparison table ──
print(f"\n  --- Performance Comparison: Population-Prior vs Personalized vs DeepSurv ---")
print(f"  {'Cohort':<12} {'Model':<12} {'C-index':>8} {'95% CI':>22}")
print(f"  {'-'*12} {'-'*12} {'-'*8} {'-'*22}")

# Load old (population-prior) hybrid results for comparison
# Strategy C with old TTP
sc_c_old=PredictionLevelFusion(normalization_method="rank")
sc_c_old.fit(dsr_tr,ttp_tr,ytr_t,ytr_e)
ttp_va_old=mva_old["TTP_natural"].values
hyb_va_c_old=sc_c_old.predict(dsr_va,ttp_va_old)

results_summary=[]
cohorts_all={"TCGA_val":(yva_e,yva_t,dsr_va,cox_va,hyb_va_a,hyb_va_b,hyb_va_c,hyb_va_c_old,ttp_va_pers,ttp_va_old)}
for c in GEO_COHORTS:
    if c not in geo_data: continue
    gd=geo_data[c]
    ttp_g_old=pd.read_csv(P5D/f"mechanistic_features_{c}.csv")["TTP_natural"].values
    hyb_c_old=sc_c_old.predict(gd['ds'],ttp_g_old)
    cohorts_all[c]=(gd['ye'],gd['yt'],gd['ds'],gd['cox'],gd['hyb_a'],gd['hyb_b'],gd['hyb_c'],hyb_c_old,gd['ttp_pers'],ttp_g_old)

for cn,(ye,yt,dsr,cr,ha,hb,hc,hc_old,ttp_p,ttp_old) in cohorts_all.items():
    for mn,r in [("DeepSurv",dsr),("CoxPH",cr),("StratA_pers",ha),("StratB_pers",hb),("StratC_pers",hc),("StratC_old",hc_old)]:
        c=ci(ye,yt,r); _,lo,hi=boot_ci(ye,yt,r)
        print(f"  {cn:<12} {mn:<12} {c:>8.4f} ({lo:.4f}-{hi:.4f})")
        results_summary.append({"cohort":cn,"model":mn,"cindex":c,"ci_lo":lo,"ci_hi":hi})

# ── ΔC comparison ──
print(f"\n  --- ΔC (Hybrid - DeepSurv): Population-Prior vs Personalized ---")
print(f"  {'Cohort':<12} {'Strategy':<12} {'ΔC old':>10} {'ΔC pers':>10} {'95% CI (pers)':>22} {'Sig?':>6}")
print(f"  {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*22} {'-'*6}")
for cn,(ye,yt,dsr,cr,ha,hb,hc,hc_old,ttp_p,ttp_old) in cohorts_all.items():
    for sn,hr_pers,hr_old in [("C",hc,hc_old),("A",ha,None),("B",hb,None)]:
        dc_pers=ci(ye,yt,hr_pers)-ci(ye,yt,dsr)
        if hr_old is not None:
            dc_old=ci(ye,yt,hr_old)-ci(ye,yt,dsr)
        else:
            dc_old=float('nan')
        md,lo,hi=boot_diff(ye,yt,hr_pers,dsr)
        sig="No" if lo<=0<=hi else "Yes"
        print(f"  {cn:<12} {sn:<12} {dc_old:>10.4f} {dc_pers:>10.4f} ({lo:.4f}-{hi:.4f}){'':<6} {sig:>6}")

# ═══════════════════════════════════════════════════════════════
# STEP 4: CONFOUND CHECK
# ═══════════════════════════════════════════════════════════════
print("\n"+"="*80); print("STEP 4: Confound Check"); print("="*80)

# Check correlation between personalized mechanistic features and stage
va_stage=cva['Stage'].apply(mst)
stage_codes={s:i for i,s in enumerate(["I","IA","IB","II","IIA","IIB","III","IIIA","IIIB","IV"])}
stage_num=va_stage.map(stage_codes).fillna(-1).values

print(f"\n  Correlation between personalized mechanistic features and stage (TCGA val):")
print(f"  {'Feature':<25} {'Spearman rho':>12} {'p-value':>12}")
for feat in ["TTP_natural","growth_rate_alpha","initial_volume_log","k_immune_log","mu_E","rho"]:
    vals=val_mech[feat].values
    if np.std(vals)>0 and np.std(stage_num)>0:
        rho,p=scipy.stats.spearmanr(vals,stage_num)
        print(f"  {feat:<25} {rho:>12.4f} {p:>12.4e}")
    else:
        print(f"  {feat:<25} {'N/A':>12} {'N/A':>12}")

# Check if TTP is redundant with stage
print(f"\n  Stage vs TTP: Is mechanistic TTP just a proxy for stage?")
rho_ttp_stage,p_ttp_stage=scipy.stats.spearmanr(ttp_va_pers,stage_num)
print(f"  Spearman(TTP, stage): rho={rho_ttp_stage:.4f}, p={p_ttp_stage:.4e}")

# Also check DeepSurv risk vs personalized TTP
rho_ds_ttp,p_ds_ttp=scipy.stats.spearmanr(dsr_va,ttp_va_pers)
print(f"  Spearman(DeepSurv risk, personalized TTP): rho={rho_ds_ttp:.4f}, p={p_ds_ttp:.4e}")

# If any strategy beats DeepSurv, check if it holds within stage strata
best_hybrid=max([("A",hyb_va_a),("B",hyb_va_b),("C",hyb_va_c)], key=lambda x: ci(yva_e,yva_t,x[1]))
best_ci=ci(yva_e,yva_t,best_hybrid[1])
ds_ci=ci(yva_e,yva_t,dsr_va)
print(f"\n  Best hybrid: Strategy {best_hybrid[0]} (C-index={best_ci:.4f}) vs DeepSurv ({ds_ci:.4f})")
if best_ci > ds_ci:
    print(f"  -> Positive signal detected! Checking if it holds within stage strata...")
    for stage_group,mask in [("Early(I-II)",va_stage.isin(["I","IA","IB","II","IIA","IIB"]).values),
                              ("Late(III-IV)",va_stage.isin(["III","IIIA","IIIB","IV"]).values)]:
        if mask.sum()<10: continue
        ci_ds=ci(yva_e[mask],yva_t[mask],dsr_va[mask])
        ci_hy=ci(yva_e[mask],yva_t[mask],best_hybrid[1][mask])
        print(f"    {stage_group}: DS={ci_ds:.4f}, Hybrid={ci_hy:.4f}, ΔC={ci_hy-ci_ds:.4f} (n={mask.sum()})")
else:
    print(f"  -> No positive signal. Confound check not triggered (null result confirmed).")

# ═══════════════════════════════════════════════════════════════
# STEP 5: RECOMPUTE NRI/IDI AND CONSISTENCY VIOLATIONS
# ═══════════════════════════════════════════════════════════════
print("\n"+"="*80); print("STEP 5: NRI/IDI and Consistency Violations (Personalized)"); print("="*80)

print(f"\n  --- NRI/IDI: Hybrid (Strategy C, personalized) vs CoxPH ---")
print(f"  {'Cohort':<12} {'NRI':>10} {'IDI':>10} {'NRI old':>10} {'IDI old':>10}")
print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
for cn,(ye,yt,dsr,cr,ha,hb,hc,hc_old,ttp_p,ttp_old) in cohorts_all.items():
    nri_new=compute_nri(hc,cr,ye,yt)
    idi_new=compute_idi(hc,cr,ye)
    nri_old=compute_nri(hc_old,cr,ye,yt)
    idi_old=compute_idi(hc_old,cr,ye)
    print(f"  {cn:<12} {nri_new['NRI']:>10.4f} {idi_new['IDI']:>10.4f} {nri_old['NRI']:>10.4f} {idi_old['IDI']:>10.4f}")

# Consistency violations
print(f"\n  --- Consistency Violations (Personalized vs Population-Prior) ---")
print(f"  {'Cohort':<12} {'rho new':>10} {'p new':>12} {'Viol% new':>10} {'rho old':>10} {'Viol% old':>10}")
print(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
for cn,(ye,yt,dsr,cr,ha,hb,hc,hc_old,ttp_p,ttp_old) in cohorts_all.items():
    rho_new,p_new=scipy.stats.spearmanr(dsr,ttp_p)
    mech_risk_new=normalize_ttp_to_risk(ttp_p,method="rank")
    ds_norm=scipy.stats.rankdata(dsr); ds_norm=(ds_norm-1)/max(len(ds_norm)-1,1)
    viol_new=compute_consistency_violations(ds_norm,mech_risk_new)
    pct_viol_new=100*(viol_new>0.3).sum()/len(viol_new)

    rho_old,p_old=scipy.stats.spearmanr(dsr,ttp_old)
    mech_risk_old=normalize_ttp_to_risk(ttp_old,method="rank")
    viol_old=compute_consistency_violations(ds_norm,mech_risk_old)
    pct_viol_old=100*(viol_old>0.3).sum()/len(viol_old)

    print(f"  {cn:<12} {rho_new:>10.4f} {p_new:>12.4e} {pct_viol_new:>10.1f} {rho_old:>10.4f} {pct_viol_old:>10.1f}")

# ═══════════════════════════════════════════════════════════════
# STEP 6: INFEASIBLE CASES
# ═══════════════════════════════════════════════════════════════
print("\n"+"="*80); print("STEP 6: Infeasible Cases Report"); print("="*80)
total_val=len(cva); personalized_val=n_personalized_val
print(f"  TCGA validation: {personalized_val}/{total_val} personalized ({100*personalized_val/total_val:.1f}%)")
for c in GEO_COHORTS:
    if c not in geo_coverage_report: continue
    r=geo_coverage_report[c]
    print(f"  {c}: {r['n_personalized']}/{r['n_total']} personalized ({r['pct']:.1f}%)")

# Save results
results_df=pd.DataFrame(results_summary)
results_df.to_csv(P5R/"personalization_parity_results.csv",index=False)
print(f"\n  Results saved: {P5R}/personalization_parity_results.csv")

print("\n"+"="*80); print("PERSONALIZATION PARITY ANALYSIS COMPLETE"); print("="*80)
