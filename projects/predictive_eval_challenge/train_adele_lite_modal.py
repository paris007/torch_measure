#!/usr/bin/env python3
from __future__ import annotations
import io, json, math, re, zipfile
from pathlib import Path
import modal

DATASET_ID = "aims-foundations/measurement-db"
OUT_REL = Path("projects/predictive_eval_challenge/codabench_submissions/adele_lite_modal/artifacts")
REGISTRY_FILES = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "datasets>=2.19.0", "pandas>=2.2.0", "numpy>=1.26.0",
    "scikit-learn>=1.4.0", "pyarrow>=15.0.0"
)
app = modal.App("cs321m-adele-lite", image=image)

CAT = {
 "swebench":"coding","livecodebench":"coding","bigcodebench":"coding","humaneval":"coding","mbpp":"coding",
 "bfcl":"tool_use","agentdojo":"tool_use","androidworld":"tool_use","tau2":"tool_use",
 "matharena":"math","mathvista_mini":"math_vision","gsm8k":"math","aime":"math",
 "ai2d_test":"vision","mmbench_v11":"vision","mmmu":"vision",
 "mmlupro":"knowledge","hle":"knowledge","mmlu":"knowledge","gpqa":"knowledge",
 "rewardbench":"preference","ultrafeedback":"preference","mtbench":"chat",
 "afrimedqa":"medical","medqa":"medical","cybench":"cyber"
}
CAT_NAMES = ["coding","tool_use","math","math_vision","vision","knowledge","preference","chat","medical","cyber","other"]

def ss(x): return "" if x is None else str(x)
def norm(x): return ss(x).strip().lower()
def subj(s):
    m = re.search(r"^Name:\s*(.+)$", s or "", flags=re.MULTILINE)
    return (m.group(1) if m else (s or "")).strip().lower()
def key(*xs): return "||".join(map(str, xs))
def clip(p, eps=1e-4):
    try: p=float(p)
    except Exception: return 0.5
    return min(1-eps, max(eps, p))
def cat_of(b): return CAT.get(norm(b), "other")
def choose(cols, opts):
    for o in opts:
        if o in cols: return o
    low={str(c).lower():c for c in cols}
    for o in opts:
        if o.lower() in low: return low[o.lower()]
    raise KeyError(f"Missing columns {opts}; got {list(cols)[:30]}")

def demand(text):
    t = ss(text); lo=t.lower(); chars=len(t); words=re.findall(r"\w+",lo); n=len(words); opts=len(re.findall(r"(?:^|\n)\s*\(?[A-Ja-j]\)?[\).:]",t))
    def has(lst): return 1.0 if any(x in lo for x in lst) else 0.0
    return [
      math.log1p(chars)/8, math.log1p(n)/7, float(chars<180), float(chars>1800),
      float("```" in t or "def " in lo or "class " in lo or "import " in lo or "function" in lo),
      float(any(s in t for s in ["∑","√","≤","≥","≈","∫","$","\\frac","^2"," x "," y "])),
      float(("|" in t and "\n" in t) or "table" in lo), float(opts>=3), min(opts,10)/10,
      has(["image","diagram","figure","chart","graph","shown above","picture"]),
      has(["patient","diagnosis","symptom","treatment","clinical","physician","disease","dose"]),
      has(["vulnerability","exploit","payload","xss","sql injection","cve","malware","cyber"]),
      has(["legal","court","contract","plaintiff","defendant","statute","liability"]),
      has(["stock","finance","revenue","profit","portfolio","interest rate","bond"]),
      has(["if and only if","therefore","implies","logical","deduce","inference","valid"]),
      has(["calculate","compute","solve","equation","numeric","probability","expected value"]),
      has(["prove","derive","show that","theorem","lemma"]),
      has(["step by step","multi-step","first,","second,","finally","plan"]),
      has(["unknown","uncertain","ambiguous","insufficient information","cannot determine"]),
      has(["harmful","unsafe","refuse","policy","jailbreak","safety"]),
      has(["tool","api","function call","browser","terminal","execute"]),
      min(t.count("?"),4)/4,
      min(t.count("\n")/max(1,n/25),2)/2
    ]

def normalize(df):
    import pandas as pd
    cols=set(df.columns)
    y=choose(cols,["label","response","correct","score","passed"])
    it=choose(cols,["item_content","item_description","item","prompt","question"])
    su=choose(cols,["subject_content","subject_description","model_content","model_id","subject"])
    be=choose(cols,["benchmark","benchmark_name","dataset","eval_name"])
    co=next((c for c in ["condition","test_condition","setting","prompting"] if c in cols), None)
    out=pd.DataFrame()
    out["label"]=df[y].astype(float)
    out["item_content"]=df[it].map(ss)
    out["subject_content"]=df[su].map(ss)
    out["benchmark"]=df[be].map(ss)
    out["condition"]=df[co].map(ss) if co else "none"
    out.loc[out["condition"].isin(["","nan","None","NaN"]),"condition"]="none"
    out["subject_name"]=out["subject_content"].map(subj)
    out["category"]=out["benchmark"].map(cat_of)
    return out[out["label"].isin([0.0,1.0])].reset_index(drop=True)

def render_subject_content(subject: dict, fallback: str) -> str:
    display_name = subject.get("display_name") or fallback
    lines = [f"Name: {display_name}"]
    for k, lab in (("provider","Organization"),("params","Parameters"),("release_date","Released"),("family","Family")):
        v = subject.get(k)
        if v not in (None, ""): lines.append(f"{lab}: {v}")
    return "\n".join(lines)

def list_response_files():
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(repo_id=DATASET_ID, repo_type="dataset")
    return sorted(f for f in files if f.endswith(".parquet") and f not in REGISTRY_FILES and not f.endswith("_traces.parquet"))

def load_runtime_df():
    import pandas as pd
    from datasets import Features, Value, load_dataset
    feats = Features({"subject_id":Value("string"),"item_id":Value("string"),"benchmark_id":Value("string"),"trial":Value("int64"),"test_condition":Value("string"),"response":Value("float64"),"correct_answer":Value("string"),"trace":Value("string")})
    responses = load_dataset(DATASET_ID, data_files=list_response_files(), features=feats, split="train").to_pandas()
    items = load_dataset(DATASET_ID, data_files="items.parquet", split="train").to_pandas()
    subjects = load_dataset(DATASET_ID, data_files="subjects.parquet", split="train").to_pandas()
    item_cols=[c for c in ["item_id","content"] if c in items.columns]
    subject_cols=[c for c in ["subject_id","display_name","provider","params","release_date","family"] if c in subjects.columns]
    df=responses.merge(items[item_cols],on="item_id",how="left").merge(subjects[subject_cols],on="subject_id",how="left")
    df["subject_content"]=[render_subject_content(r,fb) for r,fb in zip(df[subject_cols].to_dict("records"),df["subject_id"].astype(str),strict=False)]
    df["benchmark"]=df["benchmark_id"].astype(str)
    df["condition"]=df["test_condition"].fillna("none").replace("","none").astype(str)
    df["item_content"]=df["content"].fillna("").astype(str)
    df["label"]=df["response"].astype(float)
    df["subject_name"]=df["subject_content"].map(subj)
    df["category"]=df["benchmark"].map(cat_of)
    return df[df["label"].isin([0.0,1.0])].reset_index(drop=True)

def smooth(s,n,parent,k=25): return float((s+k*parent)/(n+k))
def fit_prior(df):
    g=float(df.label.mean()); P={"global_mean":g,"strength":25.0}
    P["subject"]={s:smooth(float(p.label.sum()),len(p),g) for s,p in df.groupby("subject_name")}
    P["benchmark"]={b:smooth(float(p.label.sum()),len(p),g) for b,p in df.groupby("benchmark")}
    P["category"]={c:smooth(float(p.label.sum()),len(p),g) for c,p in df.groupby("category")}
    P["benchmark_condition"]={}
    for (b,c),p in df.groupby(["benchmark","condition"]): P["benchmark_condition"][key(b,c)]=smooth(float(p.label.sum()),len(p),P["benchmark"].get(b,g))
    P["subject_benchmark"]={}
    for (s,b),p in df.groupby(["subject_name","benchmark"]):
        parent=.65*P["subject"].get(s,g)+.35*P["benchmark"].get(b,g)
        P["subject_benchmark"][key(s,b)]=smooth(float(p.label.sum()),len(p),parent)
    P["subject_benchmark_condition"]={}
    for (s,b,c),p in df.groupby(["subject_name","benchmark","condition"]):
        parent=P["subject_benchmark"].get(key(s,b),P["benchmark_condition"].get(key(b,c),g))
        P["subject_benchmark_condition"][key(s,b,c)]=smooth(float(p.label.sum()),len(p),parent)
    P["subject_category"]={}
    for (s,c),p in df.groupby(["subject_name","category"]): P["subject_category"][key(s,c)]=smooth(float(p.label.sum()),len(p),P["subject"].get(s,g))
    return P
def base(row,P):
    g=float(P["global_mean"]); s=row.get("subject_name") or subj(row.get("subject_content","")); b=ss(row.get("benchmark","")); c=ss(row.get("condition","none") or "none")
    if key(s,b,c) in P["subject_benchmark_condition"]: return clip(.95*P["subject_benchmark_condition"][key(s,b,c)]+.05*g)
    if key(s,b) in P["subject_benchmark"]: return clip(.92*P["subject_benchmark"][key(s,b)]+.08*g)
    vals=[(P["benchmark_condition"].get(key(b,c)),.45),(P["subject"].get(s),.40),(P["benchmark"].get(b),.15)]
    av=[(v,w) for v,w in vals if v is not None]
    if not av: return clip(g)
    p=sum(v*w for v,w in av)/sum(w for _,w in av)
    return clip(.85*p+.15*g)
def features(row,P):
    item=demand(row.get("item_content","")); g=float(P["global_mean"])
    s=row.get("subject_name") or subj(row.get("subject_content","")); b=ss(row.get("benchmark","")); c=ss(row.get("condition","none") or "none"); ca=row.get("category") or cat_of(b)
    sv=float(P["subject"].get(s,g))-g
    sc=float(P["subject_category"].get(key(s,ca),P["subject"].get(s,g)))-g
    bv=float(P["benchmark"].get(b,g))-g
    bcv=float(P["benchmark_condition"].get(key(b,c),P["benchmark"].get(b,g)))-g
    ctx=[sv,sc,bv,bcv,1.0 if c.lower() in ["none","","nan"] else 0.0]
    return item+ctx+[x*sc for x in item]+[x*bv for x in item]+[1.0 if ca==n else 0.0 for n in CAT_NAMES]

@app.function(timeout=4*60*60, memory=32768)
def train_remote(ridge_alpha: float=100.0, max_rows: int|None=None)->bytes:
    import numpy as np, pandas as pd
    from sklearn.linear_model import Ridge
    df=load_runtime_df()
    if max_rows: df=df.sample(min(max_rows,len(df)),random_state=321).reset_index(drop=True)
    print(f"Loaded {len(df):,} rows")
    P=fit_prior(df); rows=df.to_dict("records")
    p0=np.array([base(r,P) for r in rows],dtype=np.float32); y=df.label.to_numpy(dtype=np.float32)
    info=np.clip(p0*(1-p0),0.03,None); target=np.clip((y-p0)/info,-5,5).astype(np.float32)
    X=np.array([features(r,P) for r in rows],dtype=np.float32)
    xm=X.mean(0).astype(np.float32); xs=(X.std(0)+1e-6).astype(np.float32); Xz=(X-xm)/xs
    model=Ridge(alpha=float(ridge_alpha),fit_intercept=True,random_state=321)
    model.fit(Xz,target,sample_weight=info)
    pred=model.predict(Xz); corr=float(np.corrcoef(pred,target)[0,1]) if len(target)>2 else 0.0
    print(f"feature_dim={X.shape[1]} train_corr={corr:.4f}")
    npz=io.BytesIO()
    np.savez_compressed(npz,coef=model.coef_.astype(np.float32),intercept=np.array([model.intercept_],dtype=np.float32),x_mean=xm,x_std=xs,train_corr=np.array([corr],dtype=np.float32))
    out=io.BytesIO()
    with zipfile.ZipFile(out,"w",zipfile.ZIP_DEFLATED) as z:
        z.writestr("smoothed_prior.json",json.dumps(P))
        z.writestr("adele_lite_model.npz",npz.getvalue())
        z.writestr("README_adele_lite.txt",f"ridge_alpha={ridge_alpha}\nrows={len(df)}\nfeature_dim={X.shape[1]}\ntrain_corr={corr}\n")
    return out.getvalue()
@app.local_entrypoint()
def main(ridge_alpha: float=100.0, max_rows: int=0):
    data=train_remote.remote(ridge_alpha=ridge_alpha,max_rows=None if max_rows<=0 else max_rows)
    out=Path.cwd()/OUT_REL; out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data),"r") as z: z.extractall(out)
    print(f"Wrote artifacts to {out}")
