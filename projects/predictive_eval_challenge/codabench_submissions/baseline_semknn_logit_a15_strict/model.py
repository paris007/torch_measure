from __future__ import annotations
import hashlib, json, math, re
from collections import defaultdict
from pathlib import Path
from typing import Mapping
import numpy as np

EPS=1e-4
ARTIFACT_PATH=Path(__file__).parent/"artifacts"/"smoothed_prior.json"
ENCODER_ID="sentence-transformers/all-MiniLM-L6-v2"
MAX_ITEM_CHARS=900
MAX_SEQ_LENGTH=128

KNN_MODE="logit"
KNN_ALPHA=0.15
KNN_TOPK=3
KNN_TAU=0.06
KNN_MIN_SIM=0.35
KNN_CLIP=0.6

OFFSET_CLIP=0.05
SHRINK_N=5.0
W_GLOBAL=0.25
W_CATEGORY=0.50
W_BC=0.25

CATEGORY_BY_BENCHMARK={
"swebench":"coding","livecodebench":"coding","bigcodebench":"coding","humaneval":"coding","mbpp":"coding",
"bfcl":"tool_use","agentdojo":"tool_use","androidworld":"tool_use","tau2":"tool_use",
"matharena":"math","mathvista_mini":"math_vision","gsm8k":"math","aime":"math",
"ai2d_test":"vision","mmbench_v11":"vision","mmmu":"vision",
"mmlupro":"knowledge","hle":"knowledge","mmlu":"knowledge","gpqa":"knowledge",
"rewardbench":"preference","ultrafeedback":"preference","mtbench":"chat",
"afrimedqa":"medical","medqa":"medical","cybench":"cyber",
}

def _clip(x,eps=EPS):
    try: x=float(x)
    except Exception: return 0.5
    if not math.isfinite(x): return 0.5
    return float(min(1-eps,max(eps,x)))
def _logit(p):
    p=_clip(p); return math.log(p/(1-p))
def _sigmoid(x):
    if x>=0:
        z=math.exp(-x); return 1/(1+z)
    z=math.exp(x); return z/(1+z)
def _norm(s): return str(s or "").strip().lower()
def _subj(s):
    m=re.search(r"^Name:\s*(.+)$", s or "", flags=re.MULTILINE)
    return (m.group(1) if m else (s or "")).strip().lower()
def _key(*parts): return "||".join(str(p) for p in parts)
def _cat(row):
    braw=str(row.get("benchmark","")); return CATEGORY_BY_BENCHMARK.get(_norm(braw), braw)
def _gkey(row): return _key(_cat(row), str(row.get("condition","none") or "none"))
def _bckey(row): return _key(str(row.get("benchmark","")), str(row.get("condition","none") or "none"))
def _embed_text(row):
    return f"Benchmark: {row.get('benchmark','')}\nCondition: {row.get('condition','none') or 'none'}\nItem: {str(row.get('item_content','') or '')[:MAX_ITEM_CHARS]}"
ART=json.loads(ARTIFACT_PATH.read_text()) if ARTIFACT_PATH.exists() else {"global_mean":0.6528605818748474}

ENC=None
try:
    from sentence_transformers import SentenceTransformer
    ENC=SentenceTransformer(ENCODER_ID)
    ENC.max_seq_length=MAX_SEQ_LENGTH
except Exception as e:
    print(f"[semknn] encoder init fallback: {e}", flush=True)
    ENC=None

_ROUND_KEY=None
_ROUND_OFFSETS=(0.0,{},{})
_LAB=[]
_CACHE={}

def _base(row: Mapping[str,object])->float:
    gm=float(ART.get("global_mean",0.5))
    s=_subj(str(row.get("subject_content","")))
    b=str(row.get("benchmark","")); c=str(row.get("condition","none") or "none")
    sbc=ART.get("subject_benchmark_condition",{}).get(_key(s,b,c))
    if sbc is not None: return _clip(0.95*float(sbc)+0.05*gm)
    sb=ART.get("subject_benchmark",{}).get(_key(s,b))
    if sb is not None: return _clip(0.92*float(sb)+0.08*gm)
    bc=ART.get("benchmark_condition",{}).get(_key(b,c))
    ps=ART.get("subject",{}).get(s); pb=ART.get("benchmark",{}).get(b)
    vals=[(bc,0.45),(ps,0.40),(pb,0.15)]
    av=[(v,w) for v,w in vals if v is not None]
    if not av: return _clip(gm)
    tw=sum(w for _,w in av)
    p=sum(float(v)*w for v,w in av)/tw
    return _clip(0.85*p+0.15*gm)

def _lkey(labeled):
    if not labeled: return ()
    try:
        return tuple(sorted((str(r.get("benchmark","")),str(r.get("condition","")),str(r.get("subject_content","")),str(r.get("item_content","")),float(r.get("label",0) or 0)) for r in labeled))
    except Exception:
        return ("bad",len(labeled))

def _shr(xs):
    if not xs: return 0.0
    n=float(len(xs)); return (n/(n+SHRINK_N))*(sum(xs)/n)

def _fit_offsets(labeled):
    if not labeled: return (0.0,{},{})
    allr=[]; bg=defaultdict(list); bb=defaultdict(list)
    for r in labeled:
        if "label" not in r: continue
        try: res=float(r["label"])-_base(r)
        except Exception: continue
        if not math.isfinite(res): continue
        allr.append(res); bg[_gkey(r)].append(res); bb[_bckey(r)].append(res)
    return (_shr(allr), {k:_shr(v) for k,v in bg.items()}, {k:_shr(v) for k,v in bb.items()})

def _offset(row):
    g,bg,bb=_ROUND_OFFSETS
    off=W_GLOBAL*g + W_CATEGORY*float(bg.get(_gkey(row),0.0)) + W_BC*float(bb.get(_bckey(row),0.0))
    return min(OFFSET_CLIP,max(-OFFSET_CLIP,off))
def _prior(row): return _clip(_base(row)+_offset(row))

def _emb(row):
    if ENC is None: return None
    txt=_embed_text(row)
    h=hashlib.sha256(txt.encode("utf-8",errors="ignore")).hexdigest()
    if h in _CACHE: return _CACHE[h]
    try:
        e=ENC.encode([txt], normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        arr=np.asarray(e,dtype=np.float32).reshape(-1)
        _CACHE[h]=arr
        return arr
    except Exception as ex:
        print(f"[semknn] embed fallback: {ex}", flush=True)
        return None

def _fit_knn(labeled):
    if not labeled or ENC is None: return []
    rows=[]
    for r in labeled:
        if "label" not in r: continue
        e=_emb(r)
        if e is None: continue
        try:
            p=_prior(r); y=float(r["label"])
        except Exception: continue
        prob_res=y-p
        log_res=max(-1.0,min(1.0, prob_res/max(0.05,p*(1-p))))
        rows.append({"e":e,"g":_gkey(r),"bc":_bckey(r),"pr":float(prob_res),"lr":float(log_res)})
    return rows

def _knn(row):
    if not _LAB or ENC is None: return 0.0
    e=_emb(row)
    if e is None: return 0.0
    cand=[r for r in _LAB if r["bc"]==_bckey(row)]
    if len(cand)<2: cand=[r for r in _LAB if r["g"]==_gkey(row)]
    if not cand: return 0.0
    sims=[]
    for r in cand:
        s=float(np.dot(e,r["e"]))
        if s>=KNN_MIN_SIM: sims.append((s,r))
    if not sims: return 0.0
    sims.sort(key=lambda x:x[0], reverse=True); sims=sims[:KNN_TOPK]
    ws=[]; vs=[]
    for s,r in sims:
        ws.append(math.exp((s-max(KNN_MIN_SIM,0.0))/max(KNN_TAU,1e-3)))
        vs.append(r["lr"] if KNN_MODE=="logit" else r["pr"])
    sw=sum(ws)
    if sw<=0: return 0.0
    val=sum(w*v for w,v in zip(ws,vs))/sw
    return min(KNN_CLIP,max(-KNN_CLIP,val))

def _refresh(labeled):
    global _ROUND_KEY,_ROUND_OFFSETS,_LAB
    k=_lkey(labeled)
    if k==_ROUND_KEY: return
    _ROUND_KEY=k
    _ROUND_OFFSETS=_fit_offsets(labeled)
    _LAB=_fit_knn(labeled)

def predict(input: dict, labeled: list[dict] | None=None) -> float:
    try:
        _refresh(labeled)
        p=_prior(input)
        r=_knn(input)
        if r==0.0: return _clip(p)
        if KNN_MODE=="logit": return _clip(_sigmoid(_logit(p)+KNN_ALPHA*r))
        return _clip(p+KNN_ALPHA*r)
    except Exception as e:
        print(f"[semknn] predict fallback: {e}", flush=True)
        return _clip(float(ART.get("global_mean",0.5)))
