#!/usr/bin/env python3
from __future__ import annotations
import shutil, textwrap, zipfile
from pathlib import Path

MODEL = r"""
from __future__ import annotations
import json, math, re
from collections import defaultdict
from pathlib import Path
from typing import Mapping
import numpy as np

EPS=1e-4
FACTOR_WEIGHT=__W__
OFFSET_CLIP=0.05
SHRINK_N=5.0
W_GLOBAL=0.25
W_CATEGORY=0.50
W_BC=0.25
MAX_ITEM_CHARS=800
MAX_SEQ_LENGTH=128
PRIOR_PATH=Path(__file__).parent/"artifacts"/"smoothed_prior.json"
FACTOR_PATH=Path(__file__).parent/"artifacts"/"factor_pge.npz"

CATEGORY_BY_BENCHMARK={
"swebench":"coding","livecodebench":"coding","bigcodebench":"coding","humaneval":"coding","mbpp":"coding",
"bfcl":"tool_use","agentdojo":"tool_use","androidworld":"tool_use","tau2":"tool_use",
"matharena":"math","mathvista_mini":"math_vision","gsm8k":"math","aime":"math",
"ai2d_test":"vision","mmbench_v11":"vision","mmmu":"vision",
"mmlupro":"knowledge","hle":"knowledge","mmlu":"knowledge","gpqa":"knowledge",
"rewardbench":"preference","ultrafeedback":"preference","mtbench":"chat",
"afrimedqa":"medical","medqa":"medical","cybench":"cyber",
}

def _clip(x, eps=EPS):
    if not math.isfinite(float(x)): return 0.5
    return float(min(1-eps, max(eps, float(x))))
def _logit(p):
    p=_clip(p); return math.log(p/(1-p))
def _sigmoid(x):
    if x>=0:
        z=math.exp(-x); return 1/(1+z)
    z=math.exp(x); return z/(1+z)
def _norm(s): return str(s or "").strip().lower()
def _parse_subject_name(s):
    m=re.search(r"^Name:\s*(.+)$", s or "", flags=re.MULTILINE)
    return (m.group(1) if m else (s or "")).strip().lower()
def _key(*parts): return "||".join(str(p) for p in parts)
def _group_key(row):
    braw=str(row.get("benchmark","")); b=_norm(braw); c=str(row.get("condition","none") or "none")
    return _key(CATEGORY_BY_BENCHMARK.get(b,braw), c)
def _bc_key(row): return _key(str(row.get("benchmark","")), str(row.get("condition","none") or "none"))
def _format_item_text(row):
    return f"Benchmark: {row.get('benchmark','')}\nCondition: {row.get('condition','none') or 'none'}\nItem: {str(row.get('item_content',''))[:MAX_ITEM_CHARS]}"
def _gelu(x): return 0.5*x*(1+np.tanh(math.sqrt(2/math.pi)*(x+0.044715*x**3)))

PRIOR=json.loads(PRIOR_PATH.read_text()) if PRIOR_PATH.exists() else {"global_mean":0.6528605818748474}

FA=None; ENC=None; SUBJECT_THETA={}; GLOBAL_THETA=0.0; FACTOR_T=1.0; LAYERS=[]
if FACTOR_PATH.exists():
    try:
        FA=np.load(FACTOR_PATH, allow_pickle=False)
        GLOBAL_THETA=float(FA["global_theta"])
        if "temperature" in FA.files:
            t=float(FA["temperature"]); FACTOR_T=t if t>0 and math.isfinite(t) else 1.0
        SUBJECT_THETA={str(n):float(v) for n,v in zip(FA["subject_names"].tolist(), FA["subject_theta"].tolist())}
        idxs=sorted(int(k[len("mlp_w"):]) for k in FA.files if k.startswith("mlp_w"))
        LAYERS=[(np.asarray(FA[f"mlp_w{i}"],dtype=np.float32), np.asarray(FA[f"mlp_b{i}"],dtype=np.float32)) for i in idxs]
        from sentence_transformers import SentenceTransformer
        ENC=SentenceTransformer(str(FA["encoder_id"]))
        ENC.max_seq_length=MAX_SEQ_LENGTH
    except Exception as e:
        print(f"[factor_blend] init fallback: {e}", flush=True)
        FA=None; ENC=None; LAYERS=[]

_ROUND_KEY=None
_ROUND_OFFSETS=(0.0,{},{})

def _prior_base(row: Mapping[str,object]) -> float:
    gm=float(PRIOR.get("global_mean",0.5))
    s=_parse_subject_name(str(row.get("subject_content","")))
    b=str(row.get("benchmark","")); c=str(row.get("condition","none") or "none")
    sbc=PRIOR.get("subject_benchmark_condition",{}).get(_key(s,b,c))
    if sbc is not None: return _clip(0.95*float(sbc)+0.05*gm)
    sb=PRIOR.get("subject_benchmark",{}).get(_key(s,b))
    if sb is not None: return _clip(0.92*float(sb)+0.08*gm)
    bc=PRIOR.get("benchmark_condition",{}).get(_key(b,c))
    ps=PRIOR.get("subject",{}).get(s)
    pb=PRIOR.get("benchmark",{}).get(b)
    vals=[(bc,0.45),(ps,0.40),(pb,0.15)]
    avail=[(v,w) for v,w in vals if v is not None]
    if not avail: return _clip(gm)
    tw=sum(w for _,w in avail)
    p=sum(float(v)*w for v,w in avail)/tw
    return _clip(0.85*p+0.15*gm)

def _labeled_key(labeled):
    if not labeled: return ()
    try:
        return tuple(sorted((str(r.get("benchmark","")),str(r.get("condition","")),str(r.get("subject_content","")),str(r.get("item_content","")),float(r.get("label",0) or 0)) for r in labeled))
    except Exception:
        return ("bad",len(labeled))

def _shrunk_mean(xs):
    if not xs: return 0.0
    n=float(len(xs)); return (n/(n+SHRINK_N))*(sum(xs)/n)

def _fit_offsets(labeled):
    if not labeled: return (0.0,{}, {})
    allr=[]; by_group=defaultdict(list); by_bc=defaultdict(list)
    for r in labeled:
        if "label" not in r: continue
        try: resid=float(r["label"])-_prior_base(r)
        except Exception: continue
        if not math.isfinite(resid): continue
        allr.append(resid); by_group[_group_key(r)].append(resid); by_bc[_bc_key(r)].append(resid)
    return (_shrunk_mean(allr), {k:_shrunk_mean(v) for k,v in by_group.items()}, {k:_shrunk_mean(v) for k,v in by_bc.items()})

def _offset(row):
    g,groups,bcs=_ROUND_OFFSETS
    off=W_GLOBAL*g + W_CATEGORY*float(groups.get(_group_key(row),0.0)) + W_BC*float(bcs.get(_bc_key(row),0.0))
    return float(min(OFFSET_CLIP,max(-OFFSET_CLIP,off)))

def _cat_prior(row): return _clip(_prior_base(row)+_offset(row))

def _item_params(embed):
    h=embed.reshape(1,-1)
    for j,(w,b) in enumerate(LAYERS):
        h=h@w.T+b
        if j<len(LAYERS)-1: h=_gelu(h)
    log_a=float(h.ravel()[0]); diff=float(h.ravel()[1])
    return math.exp(log_a), diff

def _factor(row):
    if FA is None or ENC is None or not LAYERS: return None
    try:
        emb=ENC.encode([_format_item_text(row)], normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        a,b=_item_params(np.asarray(emb,dtype=np.float32).reshape(-1))
        theta=SUBJECT_THETA.get(_parse_subject_name(str(row.get("subject_content",""))), GLOBAL_THETA)
        return _clip(_sigmoid((a*theta-b)/max(FACTOR_T,1e-3)))
    except Exception as e:
        print(f"[factor_blend] factor fallback: {e}", flush=True)
        return None

def predict(input: dict, labeled: list[dict] | None=None) -> float:
    global _ROUND_KEY, _ROUND_OFFSETS
    try:
        k=_labeled_key(labeled)
        if k != _ROUND_KEY:
            _ROUND_KEY=k; _ROUND_OFFSETS=_fit_offsets(labeled)
        p0=_cat_prior(input)
        pf=_factor(input)
        if pf is None or FACTOR_WEIGHT <= 0: return _clip(p0)
        return _clip(_sigmoid((1-FACTOR_WEIGHT)*_logit(p0) + FACTOR_WEIGHT*_logit(pf)))
    except Exception as e:
        print(f"[factor_blend] predict fallback: {e}", flush=True)
        return _clip(float(PRIOR.get("global_mean",0.5)))
"""

LABEL = r"""
from __future__ import annotations
import hashlib

def acquisition_function(input: dict) -> float:
    text="\n".join([input.get("benchmark",""),input.get("condition",""),input.get("subject_content",""),input.get("item_content","")])
    return float(int(hashlib.sha256(text.encode("utf-8",errors="ignore")).hexdigest()[:12],16)/16**12)
"""

VARIANTS={
    "baseline_category_offset_factor_w03":0.03,
    "baseline_category_offset_factor_w05":0.05,
    "baseline_category_offset_factor_w08":0.08,
    "baseline_category_offset_factor_w10":0.10,
}

def find_project():
    here=Path.cwd().resolve()
    for p in [here, here/"projects"/"predictive_eval_challenge", Path(__file__).resolve().parent]:
        if (p/"codabench_submissions"/"baseline"/"artifacts"/"smoothed_prior.json").exists():
            return p
    raise FileNotFoundError("Run from repo root or place this script inside projects/predictive_eval_challenge.")

def zipdir(src, out):
    if out.exists(): out.unlink()
    with zipfile.ZipFile(out,"w",zipfile.ZIP_DEFLATED) as z:
        for f in sorted(src.rglob("*")):
            if f.is_file(): z.write(f, f.relative_to(src))

def main():
    project=find_project()
    subroot=project/"codabench_submissions"
    dist=project/"dist"; dist.mkdir(parents=True, exist_ok=True)
    prior=subroot/"baseline"/"artifacts"/"smoothed_prior.json"
    factor=subroot/"factor_pge"/"artifacts"/"factor_pge.npz"
    if not factor.exists():
        raise FileNotFoundError(f"Missing {factor}. Re-run/copy the factor_pge artifact first.")

    for name,w in VARIANTS.items():
        dst=subroot/name
        if dst.exists(): shutil.rmtree(dst)
        (dst/"artifacts").mkdir(parents=True, exist_ok=True)
        (dst/"model.py").write_text(MODEL.replace("__W__",repr(float(w))).lstrip(), encoding="utf-8")
        (dst/"labeling.py").write_text(LABEL.lstrip(), encoding="utf-8")
        (dst/"models.txt").write_text("sentence-transformers/all-MiniLM-L6-v2\n", encoding="utf-8")
        shutil.copyfile(prior, dst/"artifacts"/"smoothed_prior.json")
        shutil.copyfile(factor, dst/"artifacts"/"factor_pge.npz")
        out=dist/f"{name}_submission.zip"
        zipdir(dst,out)
        print(f"READY: {out}")

    print("\nUpload once each in this order:")
    for i,n in enumerate(VARIANTS,1):
        print(f"{i}. {n}_submission.zip")
    print("\nIf any gets -0.59 or -0.58, rerun it. If all are worse, keep baseline_category_offset.")

if __name__=="__main__":
    main()
