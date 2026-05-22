from __future__ import annotations
import json, math, re
from collections import defaultdict
from pathlib import Path
import numpy as np
EPS=1e-4
ETA=0.05
CLIPD=1.2
PRIOR_PATH=Path(__file__).parent/"artifacts"/"smoothed_prior.json"
MODEL_PATH=Path(__file__).parent/"artifacts"/"adele_lite_model.npz"
OFFSET_CLIP=.05; SHRINK_N=5.0; W_GLOBAL=.25; W_CATEGORY=.50; W_BC=.25
CAT={"swebench":"coding","livecodebench":"coding","bigcodebench":"coding","humaneval":"coding","mbpp":"coding","bfcl":"tool_use","agentdojo":"tool_use","androidworld":"tool_use","tau2":"tool_use","matharena":"math","mathvista_mini":"math_vision","gsm8k":"math","aime":"math","ai2d_test":"vision","mmbench_v11":"vision","mmmu":"vision","mmlupro":"knowledge","hle":"knowledge","mmlu":"knowledge","gpqa":"knowledge","rewardbench":"preference","ultrafeedback":"preference","mtbench":"chat","afrimedqa":"medical","medqa":"medical","cybench":"cyber"}
CAT_NAMES=["coding","tool_use","math","math_vision","vision","knowledge","preference","chat","medical","cyber","other"]
def ss(x): return "" if x is None else str(x)
def norm(x): return ss(x).strip().lower()
def clip(p,eps=EPS):
    try: p=float(p)
    except Exception: return .5
    return float(min(1-eps,max(eps,p)))
def logit(p): p=clip(p); return math.log(p/(1-p))
def sig(x):
    if x>=0:
        z=math.exp(-x); return 1/(1+z)
    z=math.exp(x); return z/(1+z)
def subj(s):
    m=re.search(r"^Name:\s*(.+)$",s or "",flags=re.MULTILINE)
    return (m.group(1) if m else (s or "")).strip().lower()
def key(*xs): return "||".join(map(str,xs))
def cat(r): return CAT.get(norm(r.get("benchmark","")),"other")
def gkey(r): return key(cat(r),ss(r.get("condition","none") or "none"))
def bckey(r): return key(ss(r.get("benchmark","")),ss(r.get("condition","none") or "none"))
def opts(t): return sum(1 for line in ss(t).splitlines() if re.match(r"\s*\(?[A-Ja-j]\)?[\).:]", line))
def demand(text):
    t=ss(text); lo=t.lower(); chars=len(t); words=re.findall(r"\w+",lo); n=len(words); op=opts(t)
    def has(L): return 1.0 if any(x in lo for x in L) else 0.0
    return [math.log1p(chars)/8,math.log1p(n)/7,float(chars<180),float(chars>1800),float("```" in t or "def " in lo or "class " in lo or "import " in lo or "function" in lo),float(any(s in t for s in ["∑","√","≤","≥","≈","∫","$","\\\\frac","^2"," x "," y "])),float(("|" in t and "\n" in t) or "table" in lo),float(op>=3),min(op,10)/10,has(["image","diagram","figure","chart","graph","shown above","picture"]),has(["patient","diagnosis","symptom","treatment","clinical","physician","disease","dose"]),has(["vulnerability","exploit","payload","xss","sql injection","cve","malware","cyber"]),has(["legal","court","contract","plaintiff","defendant","statute","liability"]),has(["stock","finance","revenue","profit","portfolio","interest rate","bond"]),has(["if and only if","therefore","implies","logical","deduce","inference","valid"]),has(["calculate","compute","solve","equation","numeric","probability","expected value"]),has(["prove","derive","show that","theorem","lemma"]),has(["step by step","multi-step","first,","second,","finally","plan"]),has(["unknown","uncertain","ambiguous","insufficient information","cannot determine"]),has(["harmful","unsafe","refuse","policy","jailbreak","safety"]),has(["tool","api","function call","browser","terminal","execute"]),min(t.count("?"),4)/4,min(t.count("\n")/max(1,n/25),2)/2]
P=json.loads(PRIOR_PATH.read_text()) if PRIOR_PATH.exists() else {"global_mean":.6528605818748474}
A=np.load(MODEL_PATH,allow_pickle=False) if MODEL_PATH.exists() else None
_RK=None; _OFF=(0.0,{},{})
def base(r):
    gm=float(P.get("global_mean",.5)); s=subj(ss(r.get("subject_content",""))); b=ss(r.get("benchmark","")); c=ss(r.get("condition","none") or "none")
    if key(s,b,c) in P.get("subject_benchmark_condition",{}): return clip(.95*P["subject_benchmark_condition"][key(s,b,c)]+.05*gm)
    if key(s,b) in P.get("subject_benchmark",{}): return clip(.92*P["subject_benchmark"][key(s,b)]+.08*gm)
    vals=[(P.get("benchmark_condition",{}).get(key(b,c)),.45),(P.get("subject",{}).get(s),.40),(P.get("benchmark",{}).get(b),.15)]
    av=[(v,w) for v,w in vals if v is not None]
    if not av: return clip(gm)
    p=sum(float(v)*w for v,w in av)/sum(w for _,w in av)
    return clip(.85*p+.15*gm)
def lkey(L):
    if not L: return ()
    try: return tuple(sorted((ss(r.get("benchmark","")),ss(r.get("condition","")),ss(r.get("subject_content","")),ss(r.get("item_content","")),float(r.get("label",0) or 0)) for r in L))
    except Exception: return ("bad",len(L))
def sh(xs):
    if not xs: return 0.0
    n=float(len(xs)); return (n/(n+SHRINK_N))*(sum(xs)/n)
def fitoff(L):
    if not L: return (0.0,{},{})
    allr=[]; bg=defaultdict(list); bb=defaultdict(list)
    for r in L:
        if "label" not in r: continue
        try: rr=float(r["label"])-base(r)
        except Exception: continue
        allr.append(rr); bg[gkey(r)].append(rr); bb[bckey(r)].append(rr)
    return sh(allr),{k:sh(v) for k,v in bg.items()},{k:sh(v) for k,v in bb.items()}
def off(r):
    g,bg,bb=_OFF
    return max(-OFFSET_CLIP,min(OFFSET_CLIP,W_GLOBAL*g+W_CATEGORY*float(bg.get(gkey(r),0))+W_BC*float(bb.get(bckey(r),0))))
def feat(r):
    item=demand(ss(r.get("item_content",""))); gm=float(P.get("global_mean",.5)); s=subj(ss(r.get("subject_content",""))); b=ss(r.get("benchmark","")); c=ss(r.get("condition","none") or "none"); ca=cat(r)
    sv=float(P.get("subject",{}).get(s,gm))-gm; sc=float(P.get("subject_category",{}).get(key(s,ca),P.get("subject",{}).get(s,gm)))-gm; bv=float(P.get("benchmark",{}).get(b,gm))-gm; bcv=float(P.get("benchmark_condition",{}).get(key(b,c),P.get("benchmark",{}).get(b,gm)))-gm
    return np.array(item+[sv,sc,bv,bcv,1.0 if c.lower() in ["none","","nan"] else 0.0]+[x*sc for x in item]+[x*bv for x in item]+[1.0 if ca==n else 0.0 for n in CAT_NAMES],dtype=np.float32)
def delta(r):
    if A is None: return 0.0
    try:
        x=feat(r); z=(x-A["x_mean"].astype(np.float32))/A["x_std"].astype(np.float32)
        d=float(np.dot(A["coef"].astype(np.float32),z)+float(A["intercept"].reshape(-1)[0]))
        return max(-CLIPD,min(CLIPD,d))
    except Exception as e:
        print(f"[adele] delta fallback {e}",flush=True); return 0.0
def predict(input:dict,labeled:list[dict]|None=None)->float:
    global _RK,_OFF
    try:
        k=lkey(labeled)
        if k!=_RK: _RK=k; _OFF=fitoff(labeled)
        p0=clip(base(input)+off(input))
        return clip(sig(logit(p0)+ETA*delta(input)))
    except Exception as e:
        print(f"[adele] predict fallback {e}",flush=True); return clip(float(P.get("global_mean",.5)))
