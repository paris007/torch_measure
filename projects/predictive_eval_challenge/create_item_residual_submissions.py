#!/usr/bin/env python3
'''Create Codabench ZIPs for the Modal-trained item residual model.'''
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


MODEL_TEMPLATE = r'''
from __future__ import annotations
import hashlib, json, math, re
from collections import defaultdict
from pathlib import Path
from typing import Mapping
import numpy as np

EPS = 1e-4
ETA = __ETA__
RESIDUAL_CLIP = __RESIDUAL_CLIP__
MAX_ITEM_CHARS = 1200
MAX_SEQ_LENGTH = 256
PRIOR_PATH = Path(__file__).parent / "artifacts" / "smoothed_prior.json"
MODEL_PATH = Path(__file__).parent / "artifacts" / "item_residual_model.npz"
OFFSET_CLIP = 0.05
SHRINK_N = 5.0
W_GLOBAL = 0.25
W_CATEGORY = 0.50
W_BC = 0.25

CATEGORY_BY_BENCHMARK = {
    "swebench": "coding", "livecodebench": "coding", "bigcodebench": "coding",
    "humaneval": "coding", "mbpp": "coding",
    "bfcl": "tool_use", "agentdojo": "tool_use", "androidworld": "tool_use", "tau2": "tool_use",
    "matharena": "math", "mathvista_mini": "math_vision", "gsm8k": "math", "aime": "math",
    "ai2d_test": "vision", "mmbench_v11": "vision", "mmmu": "vision",
    "mmlupro": "knowledge", "hle": "knowledge", "mmlu": "knowledge", "gpqa": "knowledge",
    "rewardbench": "preference", "ultrafeedback": "preference", "mtbench": "chat",
    "afrimedqa": "medical", "medqa": "medical", "cybench": "cyber",
}

def _clip_probability(value: float, eps: float = EPS) -> float:
    if not math.isfinite(float(value)): return 0.5
    return float(min(1.0 - eps, max(eps, float(value))))

def _logit(p: float) -> float:
    p = _clip_probability(p)
    return math.log(p / (1.0 - p))

def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x); return 1.0 / (1.0 + z)
    z = math.exp(x); return z / (1.0 + z)

def _norm(s: object) -> str:
    return str(s or "").strip().lower()

def _parse_subject_name(subject_content: str) -> str:
    match = re.search(r"^Name:\s*(.+)$", subject_content or "", flags=re.MULTILINE)
    if match: return match.group(1).strip().lower()
    return (subject_content or "").strip().lower()

def _key(*parts: object) -> str:
    return "||".join(str(part) for part in parts)

def _category(row: Mapping[str, object]) -> str:
    braw = str(row.get("benchmark", ""))
    return str(CATEGORY_BY_BENCHMARK.get(_norm(braw), braw))

def _group_key(row: Mapping[str, object]) -> str:
    return _key(_category(row), str(row.get("condition", "none") or "none"))

def _bc_key(row: Mapping[str, object]) -> str:
    return _key(str(row.get("benchmark", "")), str(row.get("condition", "none") or "none"))

def _format_item_text(row: Mapping[str, object]) -> str:
    item = str(row.get("item_content", "") or "")[:MAX_ITEM_CHARS]
    return f"Benchmark: {row.get('benchmark', '')}\\nCondition: {row.get('condition', 'none') or 'none'}\\nItem: {item}"

PRIOR = json.loads(PRIOR_PATH.read_text()) if PRIOR_PATH.exists() else {"global_mean": 0.6528605818748474}
RESID = None
ENCODER = None
EMBED_CACHE = {}

try:
    RESID = np.load(MODEL_PATH, allow_pickle=False)
    ENCODER_ID = str(RESID["encoder_id"])
    from sentence_transformers import SentenceTransformer
    ENCODER = SentenceTransformer(ENCODER_ID)
    ENCODER.max_seq_length = MAX_SEQ_LENGTH
except Exception as exc:
    print(f"[item_residual] init fallback: {exc}", flush=True)
    RESID = None
    ENCODER = None

_ROUND_CACHE_KEY = None
_ROUND_OFFSETS = (0.0, {}, {})

def _base_predict(row: Mapping[str, object]) -> float:
    gm = float(PRIOR.get("global_mean", 0.5))
    s = _parse_subject_name(str(row.get("subject_content", "")))
    b = str(row.get("benchmark", ""))
    c = str(row.get("condition", "none") or "none")
    sbc = PRIOR.get("subject_benchmark_condition", {}).get(_key(s, b, c))
    if sbc is not None: return _clip_probability(0.95 * float(sbc) + 0.05 * gm)
    sb = PRIOR.get("subject_benchmark", {}).get(_key(s, b))
    if sb is not None: return _clip_probability(0.92 * float(sb) + 0.08 * gm)
    bc = PRIOR.get("benchmark_condition", {}).get(_key(b, c))
    ps = PRIOR.get("subject", {}).get(s)
    pb = PRIOR.get("benchmark", {}).get(b)
    pairs = [(bc, 0.45), (ps, 0.40), (pb, 0.15)]
    avail = [(v, w) for v, w in pairs if v is not None]
    if not avail: return _clip_probability(gm)
    tw = sum(w for _, w in avail)
    p = sum(float(v) * w for v, w in avail) / tw
    return _clip_probability(0.85 * p + 0.15 * gm)

def _labeled_key(labeled):
    if not labeled: return ()
    try:
        return tuple(sorted((str(r.get("benchmark","")), str(r.get("condition","")), str(r.get("subject_content","")), str(r.get("item_content","")), float(r.get("label",0) or 0)) for r in labeled))
    except Exception:
        return ("bad", len(labeled))

def _shrunk_mean(xs):
    if not xs: return 0.0
    n = float(len(xs))
    return float((n / (n + SHRINK_N)) * (sum(xs) / n))

def _fit_offsets(labeled):
    if not labeled: return (0.0, {}, {})
    allr = []; by_group = defaultdict(list); by_bc = defaultdict(list)
    for r in labeled:
        if "label" not in r: continue
        try: resid = float(r["label"]) - _base_predict(r)
        except Exception: continue
        if not math.isfinite(resid): continue
        allr.append(resid); by_group[_group_key(r)].append(resid); by_bc[_bc_key(r)].append(resid)
    return (_shrunk_mean(allr), {k: _shrunk_mean(v) for k, v in by_group.items()}, {k: _shrunk_mean(v) for k, v in by_bc.items()})

def _adaptive_offset(row):
    g, groups, bcs = _ROUND_OFFSETS
    off = W_GLOBAL * g + W_CATEGORY * float(groups.get(_group_key(row), 0.0)) + W_BC * float(bcs.get(_bc_key(row), 0.0))
    return float(min(OFFSET_CLIP, max(-OFFSET_CLIP, off)))

def _category_prior(row):
    return _clip_probability(_base_predict(row) + _adaptive_offset(row))

def _item_delta(row):
    if RESID is None or ENCODER is None: return 0.0
    text = _format_item_text(row)
    h = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    if h in EMBED_CACHE:
        x = EMBED_CACHE[h]
    else:
        try:
            emb = ENCODER.encode([text], normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
            x = np.asarray(emb, dtype=np.float32).reshape(-1)
            EMBED_CACHE[h] = x
        except Exception as exc:
            print(f"[item_residual] encode fallback: {exc}", flush=True)
            return 0.0
    try:
        z = (x - np.asarray(RESID["x_mean"], dtype=np.float32)) / np.asarray(RESID["x_std"], dtype=np.float32)
        delta = float(np.dot(np.asarray(RESID["coef"], dtype=np.float32), z) + float(np.asarray(RESID["intercept"]).reshape(-1)[0]))
        return float(min(RESIDUAL_CLIP, max(-RESIDUAL_CLIP, delta)))
    except Exception as exc:
        print(f"[item_residual] delta fallback: {exc}", flush=True)
        return 0.0

def predict(input: dict, labeled: list[dict] | None = None) -> float:
    global _ROUND_CACHE_KEY, _ROUND_OFFSETS
    try:
        k = _labeled_key(labeled)
        if k != _ROUND_CACHE_KEY:
            _ROUND_CACHE_KEY = k
            _ROUND_OFFSETS = _fit_offsets(labeled)
        p0 = _category_prior(input)
        delta = _item_delta(input)
        return _clip_probability(_sigmoid(_logit(p0) + ETA * delta))
    except Exception as exc:
        print(f"[item_residual] predict fallback: {exc}", flush=True)
        return _clip_probability(float(PRIOR.get("global_mean", 0.5)))
'''

LABELING = r'''
from __future__ import annotations
import hashlib
SALT = "item_residual_modal_salt10_parism"

def acquisition_function(input: dict) -> float:
    text = "\\n".join([SALT, input.get("benchmark",""), input.get("condition",""), input.get("subject_content",""), input.get("item_content","")])
    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    return float(int(digest[:12], 16) / 16**12)
'''

VARIANTS = {
    "baseline_item_resid_eta005": 0.05,
    "baseline_item_resid_eta010": 0.10,
    "baseline_item_resid_eta015": 0.15,
    "baseline_item_resid_eta020": 0.20,
    "baseline_item_resid_eta030": 0.30,
}

def _find_project_dir() -> Path:
    here = Path.cwd().resolve()
    for p in [here, here / "projects" / "predictive_eval_challenge", Path(__file__).resolve().parent]:
        if (p / "codabench_submissions").exists():
            return p
    raise FileNotFoundError("Run from repo root or place this script in projects/predictive_eval_challenge.")

def _write_zip(src_dir: Path, out_path: Path) -> None:
    if out_path.exists(): out_path.unlink()
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(src_dir.rglob("*")):
            if f.is_file(): zf.write(f, f.relative_to(src_dir))

def main() -> None:
    project = _find_project_dir()
    sub_root = project / "codabench_submissions"
    artifact_dir = sub_root / "item_residual_modal" / "artifacts"
    prior = artifact_dir / "smoothed_prior.json"
    model = artifact_dir / "item_residual_model.npz"
    if not prior.exists() or not model.exists():
        raise FileNotFoundError(f"Missing residual artifacts in {artifact_dir}. Run train_item_residual_modal.py first.")

    encoder_id = "sentence-transformers/all-mpnet-base-v2"
    try:
        import numpy as np
        art = np.load(model, allow_pickle=False)
        encoder_id = str(art["encoder_id"])
    except Exception:
        pass

    dist = project / "dist"
    dist.mkdir(parents=True, exist_ok=True)

    for name, eta in VARIANTS.items():
        dst = sub_root / name
        if dst.exists(): shutil.rmtree(dst)
        (dst / "artifacts").mkdir(parents=True, exist_ok=True)
        code = MODEL_TEMPLATE.replace("__ETA__", repr(float(eta))).replace("__RESIDUAL_CLIP__", repr(1.5))
        (dst / "model.py").write_text(code.lstrip(), encoding="utf-8")
        (dst / "labeling.py").write_text(LABELING.lstrip(), encoding="utf-8")
        (dst / "models.txt").write_text(f"{encoder_id}\\n", encoding="utf-8")
        shutil.copyfile(prior, dst / "artifacts" / "smoothed_prior.json")
        shutil.copyfile(model, dst / "artifacts" / "item_residual_model.npz")
        zip_path = dist / f"{name}_submission.zip"
        _write_zip(dst, zip_path)
        print(f"READY: {zip_path}")

    print("\\nUpload once each in this order:")
    for i, name in enumerate(VARIANTS, start=1):
        print(f"{i}. {name}_submission.zip")

if __name__ == "__main__":
    main()
