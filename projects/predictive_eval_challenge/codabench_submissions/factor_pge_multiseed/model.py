"""Codabench multi-seed factor / PGE submission.

Loads N independently-trained 2-PL + MLP factor models that share the same
SentenceTransformer encoder (all-MiniLM-L6-v2) and were calibrated on the
same cold-start holdout (matharena, mmlupro, rewardbench, swebench). At
inference each seed produces its own P(correct); we average in probability
space (stable under per-seed temperature differences), then add the v6
per-category residual offset.

Each seed contributes its own (subject_theta, MLP, T). Subject ability theta
is *not* averaged across seeds because the factor model is identifiable only
up to a sign-and-shift, so different seeds put different models on different
scales. We always evaluate each seed end-to-end and combine only the final
probabilities -- this is the only mathematically clean ensemble for this
model family.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Mapping

import numpy as np


EPS = 1e-4
ARTIFACT_DIR = Path(__file__).parent / "artifacts"
MAX_ITEM_CHARS = 800
MAX_SEQ_LENGTH = 128


def _clip_probability(value: float, eps: float = EPS) -> float:
    if not math.isfinite(value):
        return 0.5
    return float(min(1.0 - eps, max(eps, value)))


def _parse_subject_name(subject_content: str) -> str:
    match = re.search(r"^Name:\s*(.+)$", subject_content or "", flags=re.MULTILINE)
    if match:
        return match.group(1).strip().lower()
    return (subject_content or "").strip().lower()


def _format_item_text(row: Mapping[str, object]) -> str:
    item_content = str(row.get("item_content", ""))[:MAX_ITEM_CHARS]
    return (
        f"Benchmark: {row.get('benchmark', '')}\n"
        f"Condition: {row.get('condition', 'none') or 'none'}\n"
        f"Item: {item_content}"
    )


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


# --- per-seed parameter container -------------------------------------------


class _SeedModel:
    """Holds the per-seed factor model state (subject thetas, MLP, T)."""

    __slots__ = (
        "subject_theta",
        "global_theta",
        "global_mean",
        "temperature",
        "mlp_layers",
    )

    def __init__(self, npz: np.lib.npyio.NpzFile):
        self.global_theta = float(npz["global_theta"])
        self.global_mean = float(npz["global_mean"])
        t_val = float(npz["temperature"]) if "temperature" in npz.files else 1.0
        self.temperature = (
            t_val if t_val > 0 and math.isfinite(t_val) else 1.0
        )
        self.subject_theta = {
            str(name): float(value)
            for name, value in zip(
                npz["subject_names"].tolist(),
                npz["subject_theta"].tolist(),
            )
        }
        layer_indices = sorted(
            int(k[len("mlp_w") :]) for k in npz.files if k.startswith("mlp_w")
        )
        self.mlp_layers = [
            (
                np.asarray(npz[f"mlp_w{i}"], dtype=np.float32),
                np.asarray(npz[f"mlp_b{i}"], dtype=np.float32),
            )
            for i in layer_indices
        ]

    def predict_from_embedding(self, embed: np.ndarray, subject_name: str) -> float:
        h = embed.reshape(1, -1)
        for idx, (w, bias) in enumerate(self.mlp_layers):
            h = h @ w.T + bias
            if idx < len(self.mlp_layers) - 1:
                h = _gelu(h)
        log_a = float(h.ravel()[0])
        b = float(h.ravel()[1])
        a = math.exp(log_a)
        theta = self.subject_theta.get(subject_name, self.global_theta)
        logit = (a * theta - b) / max(self.temperature, 1e-3)
        return 1.0 / (1.0 + math.exp(-logit))


SEEDS: list[_SeedModel] = []
ENCODER = None
ENCODER_ID: str | None = None
GLOBAL_MEAN_DEFAULT = 0.5

if ARTIFACT_DIR.exists():
    artifact_paths = sorted(ARTIFACT_DIR.glob("factor_pge_seed*.npz"))
    if not artifact_paths:
        # Fall back to single-seed artifact for backwards compatibility.
        single = ARTIFACT_DIR / "factor_pge.npz"
        if single.exists():
            artifact_paths = [single]
    means: list[float] = []
    for path in artifact_paths:
        try:
            npz = np.load(path, allow_pickle=False)
            seed = _SeedModel(npz)
            SEEDS.append(seed)
            means.append(seed.global_mean)
            if ENCODER_ID is None and "encoder_id" in npz.files:
                ENCODER_ID = str(npz["encoder_id"])
        except Exception as exc:  # noqa: BLE001
            print(
                f"[factor_pge_multiseed] failed to load {path.name}: {exc}",
                flush=True,
            )
    if means:
        GLOBAL_MEAN_DEFAULT = float(np.mean(means))

if SEEDS and ENCODER_ID is not None:
    try:
        from sentence_transformers import SentenceTransformer

        ENCODER = SentenceTransformer(ENCODER_ID)
        ENCODER.max_seq_length = MAX_SEQ_LENGTH
    except Exception as exc:  # noqa: BLE001
        print(f"[factor_pge_multiseed] could not load encoder: {exc}", flush=True)
        ENCODER = None

print(
    f"[factor_pge_multiseed] loaded {len(SEEDS)} seeds, encoder={ENCODER_ID}",
    flush=True,
)


# --- calibration round cache -------------------------------------------------


_ROUND_CACHE_KEY = None
_ROUND_GLOBAL_OFFSET = 0.0
_ROUND_CATEGORY_OFFSETS: dict[tuple[str, str], float] = {}
_CALIB_CLIP = 0.08
_CALIB_SHRINKAGE = 5.0


def _base_predict(row: Mapping[str, object]) -> float:
    if not SEEDS or ENCODER is None:
        return _clip_probability(GLOBAL_MEAN_DEFAULT)
    try:
        embedding = ENCODER.encode(
            [_format_item_text(row)],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        embed_arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
        subject_name = _parse_subject_name(str(row.get("subject_content", "")))
        probs = [
            seed.predict_from_embedding(embed_arr, subject_name) for seed in SEEDS
        ]
        return _clip_probability(float(np.mean(probs)))
    except Exception as exc:  # noqa: BLE001
        print(f"[factor_pge_multiseed] _base_predict fallback: {exc}", flush=True)
        return _clip_probability(GLOBAL_MEAN_DEFAULT)


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
    """Compute (global, per-category) residual offsets with EB shrinkage."""
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


def _resolve_round(labeled: list[dict] | None) -> None:
    global _ROUND_CACHE_KEY, _ROUND_GLOBAL_OFFSET, _ROUND_CATEGORY_OFFSETS
    key = _labeled_key(labeled)
    if key == _ROUND_CACHE_KEY:
        return
    _ROUND_GLOBAL_OFFSET, _ROUND_CATEGORY_OFFSETS = _calibration_offsets(labeled)
    _ROUND_CACHE_KEY = key


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    """Return P(correct) for one hidden subject-item pair (multi-seed)."""
    try:
        _resolve_round(labeled)
        base = _base_predict(input)
        cat = _row_category(input)
        offset = _ROUND_CATEGORY_OFFSETS.get(cat, _ROUND_GLOBAL_OFFSET)
        return _clip_probability(base + offset)
    except Exception as exc:  # noqa: BLE001
        print(f"[factor_pge_multiseed] predict fallback: {exc}", flush=True)
        return _clip_probability(GLOBAL_MEAN_DEFAULT)
