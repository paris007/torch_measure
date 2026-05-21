# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Train a Prediction-Guided Evaluation (PGE) submission.

Pipeline:
  1. Fit a 2-PL logistic factor model on observed (subject, item) responses to
     recover ability `theta_i`, discrimination `a_j`, and difficulty `b_j`.
  2. Train an MLP that maps item-text embeddings to (a_j, b_j) so cold-start
     items can be scored at test time.
  3. Persist subject abilities by name, MLP weights, and the encoder ID to a
     single `.npz` artifact consumed by the Codabench wrapper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from torch_measure.models import (
    SmoothedPriorPredictiveEvaluator,
    format_item_text,
    parse_subject_name,
)

try:
    from data_loading import item_variant_key
except ImportError:  # pragma: no cover - allows package-style imports
    from projects.predictive_eval_challenge.data_loading import item_variant_key


ENCODER_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_ARTIFACT_PATH = (
    "projects/predictive_eval_challenge/"
    "codabench_submissions/factor_pge/artifacts/factor_pge.npz"
)
MAX_ITEM_CHARS = 800
MAX_SEQ_LENGTH = 128


def encode_unique_items(
    df: pd.DataFrame,
    model_id: str,
    batch_size: int = 128,
    device: str | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Encode each unique item exactly once."""
    from sentence_transformers import SentenceTransformer

    unique = df[["item_content", "benchmark", "condition"]].copy()
    unique["item_key"] = unique.apply(item_variant_key, axis=1)
    unique["item_content"] = unique["item_content"].astype(str).str.slice(0, MAX_ITEM_CHARS)
    unique = unique.drop_duplicates("item_key").reset_index(drop=True)
    texts = [format_item_text(row) for row in unique.to_dict("records")]
    encoder = SentenceTransformer(model_id, device=device) if device else SentenceTransformer(model_id)
    encoder.max_seq_length = MAX_SEQ_LENGTH
    embeddings = np.asarray(
        encoder.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )
    item_to_row = {str(item): idx for idx, item in enumerate(unique["item_key"].astype(str).values)}
    return embeddings, item_to_row


def fit_factor_model(
    subject_ids: np.ndarray,
    item_ids: np.ndarray,
    labels: np.ndarray,
    n_subjects: int,
    n_items: int,
    n_epochs: int = 5,
    lr: float = 0.05,
    batch_size: int = 32768,
    weight_decay: float = 1e-4,
    seed: int = 321,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a 2-PL logistic factor model via mini-batch SGD.

    Returns
    -------
    theta : (n_subjects,) ability per subject
    a     : (n_items,)    discrimination per item
    b     : (n_items,)    difficulty per item
    """
    torch.manual_seed(seed)
    dev = torch.device(device)
    theta = nn.Parameter(torch.zeros(n_subjects, device=dev))
    log_a = nn.Parameter(torch.zeros(n_items, device=dev))
    b = nn.Parameter(torch.zeros(n_items, device=dev))

    subj_t = torch.as_tensor(subject_ids, dtype=torch.long, device=dev)
    item_t = torch.as_tensor(item_ids, dtype=torch.long, device=dev)
    label_t = torch.as_tensor(labels, dtype=torch.float32, device=dev)

    optimizer = torch.optim.Adam(
        [theta, log_a, b], lr=lr, weight_decay=weight_decay
    )
    bce = nn.BCEWithLogitsLoss()

    n = subj_t.shape[0]
    rng = np.random.default_rng(seed)
    for epoch in range(n_epochs):
        order = rng.permutation(n)
        order_t = torch.as_tensor(order, dtype=torch.long, device=dev)
        total = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = order_t[start : start + batch_size]
            s = subj_t[idx]
            j = item_t[idx]
            y = label_t[idx]
            logit = torch.exp(log_a[j]) * theta[s] - b[j]
            loss = bce(logit, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            n_batches += 1
        print(f"  epoch {epoch + 1}/{n_epochs}  loss={total / max(n_batches, 1):.5f}", flush=True)

    return (
        theta.detach().cpu().numpy().astype(np.float32),
        torch.exp(log_a).detach().cpu().numpy().astype(np.float32),
        b.detach().cpu().numpy().astype(np.float32),
    )


class ItemParamMLP(nn.Module):
    """Map item embedding → (log_a, b). 'a' is positive via exp(log_a).

    Supports a configurable number of hidden layers; dropout is applied between
    hidden layers during training but bypassed at eval (the wrapper exports
    weights only, so eval is the regime that ships to Codabench).
    """

    def __init__(self, in_dim: int, hidden: int = 64, n_hidden_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(max(1, n_hidden_layers)):
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fit_item_param_mlp(
    item_embeddings: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    hidden: int = 64,
    n_hidden_layers: int = 1,
    dropout: float = 0.0,
    n_epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 321,
    device: str = "cpu",
) -> ItemParamMLP:
    """Supervise an MLP to predict (log_a, b) from item embeddings."""
    torch.manual_seed(seed)
    dev = torch.device(device)
    mlp = ItemParamMLP(
        in_dim=item_embeddings.shape[1],
        hidden=hidden,
        n_hidden_layers=n_hidden_layers,
        dropout=dropout,
    ).to(dev)
    x = torch.as_tensor(item_embeddings, dtype=torch.float32, device=dev)
    targets = torch.stack(
        [
            torch.as_tensor(np.log(np.clip(a, 1e-3, None)), dtype=torch.float32, device=dev),
            torch.as_tensor(b, dtype=torch.float32, device=dev),
        ],
        dim=1,
    )
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    for epoch in range(n_epochs):
        mlp.train()
        optimizer.zero_grad()
        preds = mlp(x)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 25 == 0:
            print(f"  mlp epoch {epoch + 1}/{n_epochs}  loss={float(loss):.5f}", flush=True)
    mlp.eval()
    return mlp


def mlp_to_numpy(mlp: ItemParamMLP) -> dict[str, np.ndarray]:
    """Extract MLP weights so the Codabench wrapper can reconstruct it with numpy.

    Dropout layers do not have parameters and are skipped automatically.
    Linear-layer indices preserve their position inside `mlp.net` so the
    wrapper can reconstruct the layer order from the keys.
    """
    weights: dict[str, np.ndarray] = {}
    for i, module in enumerate(mlp.net):
        if isinstance(module, nn.Linear):
            weights[f"mlp_w{i}"] = module.weight.detach().cpu().numpy().astype(np.float32)
            weights[f"mlp_b{i}"] = module.bias.detach().cpu().numpy().astype(np.float32)
    return weights


def calibrate_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    grid: np.ndarray | None = None,
) -> float:
    """Return the scalar T that minimizes NLL of sigmoid(logits / T) vs labels.

    Uses a 1D grid search across a wide range; T=1 (no rescaling) is always in
    the grid, so the optimum is no worse than the uncalibrated model.
    """
    if grid is None:
        grid = np.concatenate(
            [
                np.linspace(0.2, 1.0, 33),
                np.linspace(1.05, 3.0, 40),
            ]
        )
        if not np.any(np.isclose(grid, 1.0)):
            grid = np.append(grid, 1.0)
    eps = 1e-4
    best_T = 1.0
    best_nll = float("inf")
    y = labels.astype(np.float64)
    for T in grid:
        p = 1.0 / (1.0 + np.exp(-logits / T))
        p = np.clip(p, eps, 1.0 - eps)
        nll = -float(np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
        if nll < best_nll:
            best_nll = nll
            best_T = float(T)
    return best_T


def train_factor_pge(
    data_path: str | Path,
    output_path: str | Path,
    encoder_id: str = ENCODER_ID,
    max_rows: int = 1_000_000,
    seed: int = 321,
    batch_size: int = 128,
    device: str | None = None,
    factor_epochs: int = 5,
    mlp_hidden: int = 64,
    mlp_hidden_layers: int = 1,
    mlp_dropout: float = 0.0,
    mlp_epochs: int = 200,
    calibration_frac: float = 0.05,
    theta_weight_decay: float = 1e-3,
) -> Path:
    """Three-stage PGE training with temperature calibration.

    Stage 1: 2-PL factor model on observed (subject, item) responses.
    Stage 2: MLP from item embedding to (a, b).
    Stage 3: Held-out temperature scalar so sigmoid(logit / T) is calibrated.
    Stage 4: Persist subject abilities, MLP weights, encoder ID, and T.
    """
    df = pd.read_parquet(data_path)
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)

    # Carve out a calibration slice up-front so it never trains.
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if calibration_frac > 0:
        n_cal = max(1, int(len(df) * calibration_frac))
        cal_df = df.iloc[:n_cal].copy()
        train_df = df.iloc[n_cal:].copy()
    else:
        cal_df = df.iloc[:0].copy()
        train_df = df.copy()

    # Subject and item indexing.
    baseline = SmoothedPriorPredictiveEvaluator.fit(train_df.to_dict("records"))
    train_df["subject_name"] = train_df["subject_content"].astype(str).map(parse_subject_name)
    train_df["item_key"] = train_df.apply(item_variant_key, axis=1)

    subject_names = sorted(train_df["subject_name"].unique().tolist())
    subject_to_idx = {name: idx for idx, name in enumerate(subject_names)}
    item_keys = sorted(train_df["item_key"].unique().tolist())
    item_to_idx = {key: idx for idx, key in enumerate(item_keys)}

    subject_ids = train_df["subject_name"].map(subject_to_idx).to_numpy(dtype=np.int64)
    item_ids = train_df["item_key"].map(item_to_idx).to_numpy(dtype=np.int64)
    labels = train_df["label"].to_numpy(dtype=np.float32)

    print(
        f"[factor] fitting 2-PL: {len(train_df)} rows, {len(subject_names)} subjects, "
        f"{len(item_keys)} items (calibration holdout: {len(cal_df)} rows)",
        flush=True,
    )
    theta, a, b = fit_factor_model(
        subject_ids=subject_ids,
        item_ids=item_ids,
        labels=labels,
        n_subjects=len(subject_names),
        n_items=len(item_keys),
        n_epochs=factor_epochs,
        weight_decay=theta_weight_decay,
        seed=seed,
    )

    # Encode all unique items present in train OR calibration so we can score
    # the calibration holdout at the end without an extra encoding pass.
    all_items_for_encoding = pd.concat(
        [
            train_df[["item_content", "benchmark", "condition"]],
            cal_df[["item_content", "benchmark", "condition"]],
        ],
        ignore_index=True,
    )
    print("[factor] encoding unique items", flush=True)
    item_embeddings, item_text_to_row = encode_unique_items(
        all_items_for_encoding,
        encoder_id,
        batch_size=batch_size,
        device=device,
    )

    # Align item embeddings to the factor-model item index using item_key string.
    embed_by_factor_idx = np.zeros((len(item_keys), item_embeddings.shape[1]), dtype=np.float32)
    for key, factor_idx in item_to_idx.items():
        row = item_text_to_row.get(key)
        if row is not None:
            embed_by_factor_idx[factor_idx] = item_embeddings[row]

    print(
        f"[factor] fitting item-param MLP (hidden={mlp_hidden}, layers={mlp_hidden_layers}, "
        f"dropout={mlp_dropout})",
        flush=True,
    )
    mlp = fit_item_param_mlp(
        item_embeddings=embed_by_factor_idx,
        a=a,
        b=b,
        hidden=mlp_hidden,
        n_hidden_layers=mlp_hidden_layers,
        dropout=mlp_dropout,
        n_epochs=mlp_epochs,
        seed=seed,
    )

    # Held-out temperature calibration.
    temperature = 1.0
    if len(cal_df) > 0:
        print(f"[factor] calibrating temperature on {len(cal_df)} held-out rows", flush=True)
        cal_df = cal_df.copy()
        cal_df["item_key"] = cal_df.apply(item_variant_key, axis=1)
        cal_df["subject_name"] = cal_df["subject_content"].astype(str).map(parse_subject_name)
        cal_embeds = np.stack(
            [
                item_embeddings[item_text_to_row[k]]
                if k in item_text_to_row
                else item_embeddings.mean(axis=0)
                for k in cal_df["item_key"].astype(str).values
            ],
            axis=0,
        )
        mlp.eval()
        with torch.no_grad():
            out = mlp(torch.as_tensor(cal_embeds, dtype=torch.float32)).cpu().numpy()
        log_a_cal = out[:, 0]
        b_cal = out[:, 1]
        a_cal = np.exp(np.clip(log_a_cal, -8, 8))
        global_theta = float(theta.mean())
        theta_cal = np.array(
            [
                float(theta[subject_to_idx[name]]) if name in subject_to_idx else global_theta
                for name in cal_df["subject_name"].values
            ],
            dtype=np.float32,
        )
        logits_cal = a_cal * theta_cal - b_cal
        y_cal = cal_df["label"].to_numpy(dtype=np.float32)
        # NLL at T=1 vs the optimal T.
        eps = 1e-4
        p1 = np.clip(1.0 / (1.0 + np.exp(-logits_cal)), eps, 1.0 - eps)
        nll_1 = float(-np.mean(y_cal * np.log(p1) + (1.0 - y_cal) * np.log(1.0 - p1)))
        temperature = calibrate_temperature(logits_cal, y_cal)
        pT = np.clip(1.0 / (1.0 + np.exp(-logits_cal / temperature)), eps, 1.0 - eps)
        nll_T = float(-np.mean(y_cal * np.log(pT) + (1.0 - y_cal) * np.log(1.0 - pT)))
        print(
            f"[factor] T*={temperature:.3f}  NLL(T=1)={nll_1:.5f} -> NLL(T*)={nll_T:.5f}",
            flush=True,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    savez_kwargs: dict[str, np.ndarray] = dict(
        encoder_id=np.array(encoder_id),
        subject_names=np.array(subject_names),
        subject_theta=theta,
        global_theta=np.array(float(theta.mean()), dtype=np.float32),
        global_mean=np.array(baseline.global_mean, dtype=np.float32),
        mlp_hidden=np.array(mlp_hidden, dtype=np.int32),
        temperature=np.array(temperature, dtype=np.float32),
    )
    savez_kwargs.update(mlp_to_numpy(mlp))
    np.savez(output_path, **savez_kwargs)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Joined runtime examples parquet.")
    parser.add_argument("--out", default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--encoder-id", default=ENCODER_ID)
    parser.add_argument("--max-rows", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--device",
        default=None,
        help="Encoder device: 'cpu', 'mps', 'cuda', or omit for auto.",
    )
    parser.add_argument("--factor-epochs", type=int, default=5)
    parser.add_argument("--mlp-hidden", type=int, default=64)
    parser.add_argument("--mlp-hidden-layers", type=int, default=1)
    parser.add_argument("--mlp-dropout", type=float, default=0.0)
    parser.add_argument("--mlp-epochs", type=int, default=200)
    parser.add_argument("--calibration-frac", type=float, default=0.05)
    parser.add_argument("--theta-weight-decay", type=float, default=1e-3)
    args = parser.parse_args()

    output_path = train_factor_pge(
        args.data,
        args.out,
        encoder_id=args.encoder_id,
        max_rows=args.max_rows,
        seed=args.seed,
        batch_size=args.batch_size,
        device=args.device,
        factor_epochs=args.factor_epochs,
        mlp_hidden=args.mlp_hidden,
        mlp_hidden_layers=args.mlp_hidden_layers,
        mlp_dropout=args.mlp_dropout,
        mlp_epochs=args.mlp_epochs,
        calibration_frac=args.calibration_frac,
        theta_weight_decay=args.theta_weight_decay,
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
