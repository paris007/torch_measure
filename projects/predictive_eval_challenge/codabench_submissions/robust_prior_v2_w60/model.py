"""Codabench robust ensemble of factor / PGE and two smoothed-prior baselines.

This variant keeps the factor/PGE component from the main ensemble, but routes
between two prior families at runtime:

* v1: the older 3-way weighted prior, which generalized better on some hidden
  submissions.
* v2: the newer subject-benchmark-condition prior, which wins local cold-start
  validation but can overfit a hidden category's public cell history.

When Codabench provides labeled examples for a category, the wrapper compares
the two priors on those labels and softly shifts the blend toward the better
one. With no labels it uses a conservative mostly-v1 blend.

The two component artifacts (`factor_pge.npz`, `smoothed_prior.json`) and an
optional `ensemble.json` (mixture weight + factor temperature override) live
under `artifacts/`. If the factor artifact fails to load the wrapper
gracefully degrades to the smoothed-prior-only prediction so the submission
never crashes.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Mapping

import numpy as np

from acquisition_util import baseline_predict_v1


EPS = 1e-4
ARTIFACTS = Path(__file__).parent / "artifacts"
FACTOR_PATH = ARTIFACTS / "factor_pge.npz"
BASELINE_PATH = ARTIFACTS / "smoothed_prior.json"
ENSEMBLE_PATH = ARTIFACTS / "ensemble.json"
MAX_ITEM_CHARS = 800
MAX_SEQ_LENGTH = 128
DEFAULT_FACTOR_WEIGHT = 0.0
DEFAULT_V2_PRIOR_WEIGHT = 0.60
ROUTER_SHRINKAGE = 6.0
ROUTER_STRENGTH = 1.5
DISABLE_FACTOR = True


def _clip_probability(value: float, eps: float = EPS) -> float:
    if not math.isfinite(value):
        return 0.5
    return float(min(1.0 - eps, max(eps, value)))


def _logit(p: float) -> float:
    p = _clip_probability(p)
    return math.log(p / (1.0 - p))


def _parse_subject_name(subject_content) -> str:
    text = str(subject_content) if subject_content is not None else ""
    match = re.search(r"^Name:\s*(.+)$", text, flags=re.MULTILINE)
    if match:
        return match.group(1).strip().lower()
    return text.strip().lower()


def _format_item_text(row: Mapping[str, object]) -> str:
    item_content = str(row.get("item_content", ""))[:MAX_ITEM_CHARS]
    return (
        f"Benchmark: {row.get('benchmark', '')}\n"
        f"Condition: {row.get('condition', 'none') or 'none'}\n"
        f"Item: {item_content}"
    )


def _key(*parts: object) -> str:
    return "||".join(str(part) for part in parts)


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


# --- Load ensemble config -----------------------------------------------------
FACTOR_WEIGHT = DEFAULT_FACTOR_WEIGHT
FACTOR_TEMPERATURE_OVERRIDE: float | None = None
PER_BENCH_WEIGHTS: dict[str, float] = {}
PER_BENCH_TEMPERATURES: dict[str, float] = {}
if ENSEMBLE_PATH.exists():
    try:
        _cfg = json.loads(ENSEMBLE_PATH.read_text())
        if "factor_weight" in _cfg:
            w = float(_cfg["factor_weight"])
            if 0.0 <= w <= 1.0:
                FACTOR_WEIGHT = w
        if "factor_temperature" in _cfg:
            FACTOR_TEMPERATURE_OVERRIDE = float(_cfg["factor_temperature"])
        # v7: per-benchmark factor weights / temperatures from the local
        # diagnostic. Each entry can be either a bare float (weight only) or a
        # dict {"factor_weight": w, "factor_temperature": T}.
        for bench, cfg in (_cfg.get("per_benchmark") or {}).items():
            if isinstance(cfg, (int, float)):
                w = float(cfg)
                if 0.0 <= w <= 1.0:
                    PER_BENCH_WEIGHTS[str(bench)] = w
            elif isinstance(cfg, dict):
                if "factor_weight" in cfg:
                    w = float(cfg["factor_weight"])
                    if 0.0 <= w <= 1.0:
                        PER_BENCH_WEIGHTS[str(bench)] = w
                if "factor_temperature" in cfg:
                    t = float(cfg["factor_temperature"])
                    if t > 0 and math.isfinite(t):
                        PER_BENCH_TEMPERATURES[str(bench)] = t
        if PER_BENCH_WEIGHTS:
            print(
                f"[ensemble] per-benchmark weights loaded for "
                f"{len(PER_BENCH_WEIGHTS)} benchmarks; global default={FACTOR_WEIGHT:.3f}",
                flush=True,
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[ensemble] could not parse ensemble.json: {exc}", flush=True)


# --- Load baseline (smoothed prior) ------------------------------------------
BASELINE: dict = {"global_mean": 0.5}
try:
    if BASELINE_PATH.exists():
        BASELINE = json.loads(BASELINE_PATH.read_text())
except Exception as exc:  # noqa: BLE001
    print(f"[ensemble] could not load baseline: {exc}", flush=True)


def _baseline_predict_v1(row: Mapping[str, object]) -> float:
    """v1 / v3-era 3-way weighted baseline (generalizes on hidden slice)."""
    return baseline_predict_v1(row)


def _baseline_predict_v2(row: Mapping[str, object]) -> float:
    """v2 baseline: prefer subject-benchmark-condition, then fall back upward."""
    global_mean = float(BASELINE.get("global_mean", 0.5))
    subject_name = _parse_subject_name(row.get("subject_content"))
    benchmark = str(row.get("benchmark", ""))
    condition = str(row.get("condition", "none") or "none")

    sbc = BASELINE.get("subject_benchmark_condition", {}).get(_key(subject_name, benchmark, condition))
    if sbc is not None:
        return _clip_probability(0.95 * float(sbc) + 0.05 * global_mean)
    sb = BASELINE.get("subject_benchmark", {}).get(_key(subject_name, benchmark))
    if sb is not None:
        return _clip_probability(0.92 * float(sb) + 0.08 * global_mean)
    bc = BASELINE.get("benchmark_condition", {}).get(_key(benchmark, condition))
    s = BASELINE.get("subject", {}).get(subject_name)
    b = BASELINE.get("benchmark", {}).get(benchmark)
    pairs = [(bc, 0.45), (s, 0.40), (b, 0.15)]
    avail = [(v, w) for v, w in pairs if v is not None]
    if not avail:
        return _clip_probability(global_mean)
    total_weight = sum(w for _, w in avail)
    pred = sum(float(v) * w for v, w in avail) / total_weight
    return _clip_probability(0.85 * pred + 0.15 * global_mean)


def _logit_blend(p_v1: float, p_v2: float, v2_weight: float) -> float:
    w = min(1.0, max(0.0, float(v2_weight)))
    mixed_logit = (1.0 - w) * _logit(p_v1) + w * _logit(p_v2)
    return _clip_probability(1.0 / (1.0 + math.exp(-mixed_logit)))


def _prior_router_weight(
    labeled: list[dict] | None,
    cat: tuple[str, str],
) -> float:
    """Softly route toward whichever prior family fits this category's labels."""
    rows = [
        r
        for r in (labeled or [])
        if _row_category(r) == cat and "label" in r
    ]
    if not rows:
        return DEFAULT_V2_PRIOR_WEIGHT

    loss_v1 = 0.0
    loss_v2 = 0.0
    n = 0
    for row in rows:
        try:
            y = float(row.get("label", 0.0) or 0.0)
            p1 = _clip_probability(_baseline_predict_v1(row))
            p2 = _clip_probability(_baseline_predict_v2(row))
        except Exception:  # noqa: BLE001
            continue
        loss_v1 += -(y * math.log(p1) + (1.0 - y) * math.log(1.0 - p1))
        loss_v2 += -(y * math.log(p2) + (1.0 - y) * math.log(1.0 - p2))
        n += 1
    if n == 0:
        return DEFAULT_V2_PRIOR_WEIGHT

    diff = (loss_v2 - loss_v1) / n
    raw_v2_weight = 1.0 / (1.0 + math.exp(max(-20.0, min(20.0, ROUTER_STRENGTH * diff))))
    shrink = n / (n + ROUTER_SHRINKAGE)
    return (1.0 - shrink) * DEFAULT_V2_PRIOR_WEIGHT + shrink * raw_v2_weight


def _baseline_predict(
    row: Mapping[str, object],
    labeled: list[dict] | None = None,
) -> float:
    p1 = _baseline_predict_v1(row)
    p2 = _baseline_predict_v2(row)
    v2_weight = _prior_router_weight(labeled, _row_category(row))
    return _logit_blend(p1, p2, v2_weight)


# --- Load factor / PGE -------------------------------------------------------
FACTOR_ARTIFACT = None
ENCODER = None
SUBJECT_THETA: dict[str, float] = {}
GLOBAL_THETA = 0.0
GLOBAL_MEAN_FACTOR = 0.5
TEMPERATURE = 1.0
MLP_LAYERS: list[tuple[np.ndarray, np.ndarray]] = []

if FACTOR_PATH.exists():
    try:
        FACTOR_ARTIFACT = np.load(FACTOR_PATH, allow_pickle=False)
        GLOBAL_THETA = float(FACTOR_ARTIFACT["global_theta"])
        GLOBAL_MEAN_FACTOR = float(FACTOR_ARTIFACT["global_mean"])
        if "temperature" in FACTOR_ARTIFACT.files:
            t_val = float(FACTOR_ARTIFACT["temperature"])
            if t_val > 0 and math.isfinite(t_val):
                # Stored T is fit by recalibrate_temperature.py on a
                # cold-start holdout; trust it directly.
                TEMPERATURE = t_val
        if FACTOR_TEMPERATURE_OVERRIDE is not None and FACTOR_TEMPERATURE_OVERRIDE > 0:
            TEMPERATURE = FACTOR_TEMPERATURE_OVERRIDE
        SUBJECT_THETA = {
            str(name): float(value)
            for name, value in zip(
                FACTOR_ARTIFACT["subject_names"].tolist(),
                FACTOR_ARTIFACT["subject_theta"].tolist(),
            )
        }
        layer_indices = sorted(
            int(k[len("mlp_w") :])
            for k in FACTOR_ARTIFACT.files
            if k.startswith("mlp_w")
        )
        MLP_LAYERS = [
            (
                np.asarray(FACTOR_ARTIFACT[f"mlp_w{i}"], dtype=np.float32),
                np.asarray(FACTOR_ARTIFACT[f"mlp_b{i}"], dtype=np.float32),
            )
            for i in layer_indices
        ]
        from sentence_transformers import SentenceTransformer

        ENCODER = SentenceTransformer(
            str(FACTOR_ARTIFACT["encoder_id"]),
            local_files_only=True,
        )
        ENCODER.max_seq_length = MAX_SEQ_LENGTH
    except Exception as exc:  # noqa: BLE001 - graceful fallback to baseline
        print(f"[ensemble] could not initialize factor: {exc}", flush=True)
        FACTOR_ARTIFACT = None
        ENCODER = None


def _item_params_from_embedding(embed: np.ndarray) -> tuple[float, float]:
    h = embed.reshape(1, -1)
    for idx, (w, bias) in enumerate(MLP_LAYERS):
        h = h @ w.T + bias
        if idx < len(MLP_LAYERS) - 1:
            h = _gelu(h)
    log_a = float(h.ravel()[0])
    b = float(h.ravel()[1])
    return math.exp(log_a), b


def _factor_predict(row: Mapping[str, object], temperature: float | None = None) -> float:
    if FACTOR_ARTIFACT is None or ENCODER is None or not MLP_LAYERS:
        return _clip_probability(GLOBAL_MEAN_FACTOR)
    try:
        embedding = ENCODER.encode(
            [_format_item_text(row)],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        embed_arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
        a, b = _item_params_from_embedding(embed_arr)
        subject_name = _parse_subject_name(row.get("subject_content"))
        theta = SUBJECT_THETA.get(subject_name, GLOBAL_THETA)
        T = float(temperature) if temperature is not None else float(TEMPERATURE)
        logit = (a * theta - b) / max(T, 1e-3)
        return _clip_probability(1.0 / (1.0 + math.exp(-logit)))
    except Exception as exc:  # noqa: BLE001
        print(f"[ensemble] _factor_predict fallback: {exc}", flush=True)
        return _clip_probability(GLOBAL_MEAN_FACTOR)


def _factor_weight_for(row: Mapping[str, object]) -> float:
    """v7: per-benchmark factor weight, fallback to global FACTOR_WEIGHT."""
    bench = str(row.get("benchmark", ""))
    if bench in PER_BENCH_WEIGHTS:
        return PER_BENCH_WEIGHTS[bench]
    return float(FACTOR_WEIGHT)


def _factor_temperature_for(row: Mapping[str, object]) -> float | None:
    bench = str(row.get("benchmark", ""))
    return PER_BENCH_TEMPERATURES.get(bench)


def _base_predict(
    row: Mapping[str, object],
    labeled: list[dict] | None = None,
) -> float:
    """Per-benchmark weighted logit-mean of factor and smoothed-prior baseline."""
    baseline_p = _baseline_predict(row, labeled=labeled)
    if DISABLE_FACTOR:
        return baseline_p
    if FACTOR_ARTIFACT is None or ENCODER is None or not MLP_LAYERS:
        return baseline_p
    w = _factor_weight_for(row)
    if w <= 0.0:
        return baseline_p
    factor_p = _factor_predict(row, temperature=_factor_temperature_for(row))
    if w >= 1.0:
        return factor_p
    combined_logit = w * _logit(factor_p) + (1.0 - w) * _logit(baseline_p)
    return _clip_probability(1.0 / (1.0 + math.exp(-combined_logit)))


_ROUND_CACHE_KEY = None
_ROUND_GLOBAL_OFFSET = 0.0
_ROUND_CATEGORY_OFFSETS: dict[tuple[str, str], float] = {}
_CALIB_CLIP = 0.08
_CALIB_CLIP_BASELINE_ONLY = 0.12
_CALIB_SHRINKAGE = 5.0
_MAX_LABELS_PER_CAT = 5


def _labeled_key(labeled: list[dict] | None):
    if not labeled:
        return ()
    try:
        return tuple(
            sorted(
                (
                    str(row.get("benchmark", "")),
                    str(row.get("condition", "")),
                    str(row.get("subject_content", "")),
                    str(row.get("item_content", "")),
                    float(row.get("label", 0.0) or 0.0),
                )
                for row in labeled
            )
        )
    except Exception:  # noqa: BLE001
        return ("__unsortable__", len(labeled))


def _row_category(row: Mapping[str, object]) -> tuple[str, str]:
    return (str(row.get("benchmark", "")), str(row.get("condition", "")))


def _calibration_offsets(
    labeled: list[dict] | None,
) -> tuple[float, dict[tuple[str, str], float]]:
    """Compute (global, per-category) residual offsets with EB shrinkage.

    v6: uniform labeling -> unbiased residual estimates per category. We
    shrink each per-cat mean toward the global mean with weight
    n / (n + _CALIB_SHRINKAGE), then clip to +-_CALIB_CLIP.
    """
    if not labeled:
        return 0.0, {}
    residuals_global: list[float] = []
    bucket: dict[tuple[str, str], list[float]] = {}
    for row in labeled:
        if "label" not in row:
            continue
        try:
            r = float(row["label"]) - _base_predict(row)
        except Exception:  # noqa: BLE001
            continue
        residuals_global.append(r)
        bucket.setdefault(_row_category(row), []).append(r)
    if not residuals_global:
        return 0.0, {}
    global_mean = sum(residuals_global) / len(residuals_global)
    global_offset = max(-_CALIB_CLIP, min(_CALIB_CLIP, global_mean))
    per_cat: dict[tuple[str, str], float] = {}
    for cat, rs in bucket.items():
        n = len(rs)
        local_mean = sum(rs) / n
        w = n / (n + _CALIB_SHRINKAGE)
        shrunk = w * local_mean + (1.0 - w) * global_mean
        per_cat[cat] = max(-_CALIB_CLIP, min(_CALIB_CLIP, shrunk))
    return global_offset, per_cat


def _labeled_logit_blend(
    base: float,
    labeled: list[dict] | None,
    cat: tuple[str, str],
) -> float:
    """Shrink toward observed category mean in logit space (K <= 5)."""
    rows = [
        r
        for r in (labeled or [])
        if _row_category(r) == cat and "label" in r
    ]
    if not rows:
        return base
    n_eff = min(len(rows), _MAX_LABELS_PER_CAT)
    obs = sum(float(r.get("label", 0.0) or 0.0) for r in rows) / len(rows)
    w = n_eff / (n_eff + _CALIB_SHRINKAGE)
    blended_logit = (1.0 - w) * _logit(base) + w * _logit(obs)
    p = _clip_probability(1.0 / (1.0 + math.exp(-blended_logit)))
    clip = _CALIB_CLIP if FACTOR_ARTIFACT is not None else _CALIB_CLIP_BASELINE_ONLY
    return _clip_probability(max(base - clip, min(base + clip, p)))


def _resolve_round(labeled: list[dict] | None) -> None:
    global _ROUND_CACHE_KEY, _ROUND_GLOBAL_OFFSET, _ROUND_CATEGORY_OFFSETS
    key = _labeled_key(labeled)
    if key == _ROUND_CACHE_KEY:
        return
    _ROUND_GLOBAL_OFFSET, _ROUND_CATEGORY_OFFSETS = _calibration_offsets(labeled)
    _ROUND_CACHE_KEY = key


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    """Return P(correct) for one hidden subject-item pair.

    v9: v1 baseline ensemble + logit-space label blend when K labels exist
    for this category; otherwise EB-shrunk residual offsets.
    """
    try:
        _resolve_round(labeled)
        base = _base_predict(input, labeled=labeled)
        cat = _row_category(input)
        n_cat = sum(
            1
            for r in (labeled or [])
            if _row_category(r) == cat and "label" in r
        )
        if n_cat:
            return _labeled_logit_blend(base, labeled, cat)
        offset = _ROUND_CATEGORY_OFFSETS.get(cat, _ROUND_GLOBAL_OFFSET)
        return _clip_probability(base + offset)
    except Exception as exc:  # noqa: BLE001
        print(f"[ensemble] predict fallback: {exc}", flush=True)
        return _clip_probability(GLOBAL_MEAN_FACTOR)
