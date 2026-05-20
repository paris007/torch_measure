# CS321M Predictive AI Evaluation Challenge

End-to-end training, evaluation, and Codabench-submission code for the CS321M
Predictive AI Evaluation Challenge. Given a hidden
`(subject, item, benchmark, condition)` tuple, the submission predicts
`P(correct)` — the probability that the AI subject answers that item correctly.

The evaluation is **item cold-start**: the held-out items are not in the public
training matrix, but the subjects mostly are. The primary metric is
negative log-loss (NLL); AUC-ROC is secondary, so calibration matters as much
as ranking.

---

## Repository layout

```text
.
├── README.md                       # this file
├── REPORT.md                       # full methods + results write-up
├── requirements.txt                # offline training / local eval deps
├── requirements-modal.txt          # extra deps for Modal GPU training
├── pyproject.toml
├── .gitignore
│
├── data_loading.py                 # load public HF parquet collection
├── download_data.py                # CLI: materialize runtime_examples.parquet
│
├── train_baseline.py               # smoothed-prior baseline (subject/benchmark/condition cells)
├── fit_richer_baseline.py          # v2 baseline: add 4-way (s, b, c) cells with EB shrinkage
├── train_embedding.py              # SentenceTransformer + logistic head
├── train_factor.py                 # 2-PL factor model + item-text-to-factor MLP
├── recalibrate_temperature.py      # fit T on a whole-benchmark cold-start holdout
│
├── eval_ensemble.py                # local NLL/AUC for single + multi-seed factor models
├── eval_baseline_variants.py       # compare baseline v1 / v2-mixed / v2-sbc-first
├── per_benchmark_diag.py           # grid-search per-benchmark factor weight + T
├── local_cv.py                     # whole-benchmark cross-validation harness
├── validate.py                     # simple holdout eval used by the slurm jobs
│
├── modal_train_factor.py           # Modal entrypoints: single-seed and multi-seed training
│
├── make_submission.py              # build flat Codabench ZIPs
├── smoke_submission.py             # import + predict() smoke test on one ZIP
│
├── codabench_submissions/          # 5 Codabench-ready submission directories
│   ├── baseline/                   # smoothed prior with 4-way (s,b,c) cells
│   ├── embedding/                  # SentenceTransformer embedding head
│   ├── factor_pge/                 # 2-PL factor + item-text MLP
│   ├── factor_baseline_ensemble/   # logit-mean ensemble + per-benchmark weights
│   └── factor_pge_multiseed/       # 5-seed factor ensemble (MiniLM)
│
├── dist/                           # built submission ZIPs + per-benchmark configs
├── slurm/                          # Schmidt-Slurm batch scripts
├── tests/                          # pytest robustness tests
└── report/                         # archived methods/results draft
```

Each `codabench_submissions/<name>/` is shaped so it can be zipped directly,
with `model.py` at the archive root.

---

## Submission contract (recap)

Every submission ZIP exposes:

```python
def predict(input: dict, labeled: list[dict] | None = None) -> float:
    ...
```

`input` keys: `benchmark`, `condition`, `subject_content`, `item_content`.

Optional `labeling.py` exposes:

```python
def acquisition_function(input: dict) -> float:
    ...
```

Codabench picks the top **K = 5** items per category for labeling. The
revealed labels are passed back to `predict()` as `labeled` so the wrapper can
apply a small residual calibration offset.

Any HuggingFace repos required at runtime go in `models.txt`; Codabench
pre-downloads them before importing `model.py`. Submissions have **no
outbound internet access** at runtime, so all training is offline.

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 1. Pull and join the public HF data (one-shot, ~3 GB on disk).
python download_data.py --out data/runtime_examples.parquet

# 2. Train the smoothed-prior baseline (v2, with 4-way (s,b,c) cells).
python fit_richer_baseline.py \
  --data data/runtime_examples.parquet \
  --out codabench_submissions/baseline/artifacts/smoothed_prior.json

# 3. Train the 2-PL factor / PGE model (MiniLM encoder, CPU-OK).
python train_factor.py \
  --data data/runtime_examples.parquet \
  --out codabench_submissions/factor_pge/artifacts/factor_pge.npz

# 4. Recalibrate the factor temperature on a whole-benchmark holdout.
python recalibrate_temperature.py \
  --artifact codabench_submissions/factor_pge/artifacts/factor_pge.npz \
  --data data/runtime_examples.parquet

# 5. Smoke-test a submission before building the ZIP.
python smoke_submission.py codabench_submissions/baseline
python smoke_submission.py codabench_submissions/factor_baseline_ensemble

# 6. Build the flat Codabench ZIPs into dist/.
python make_submission.py baseline
python make_submission.py factor_pge
python make_submission.py factor_baseline_ensemble
python make_submission.py factor_pge_multiseed
python make_submission.py embedding
```

The built ZIPs land in `dist/<name>_submission.zip` and are ready to upload to
Codabench.

---

## The five submissions

| # | Submission                                | Idea                                                                  | Codabench NLL |
| - | ----------------------------------------- | --------------------------------------------------------------------- | ------------- |
| 1 | `baseline/`                               | Smoothed prior over `(subject, benchmark, condition)` cells           | ~ −0.69       |
| 2 | `embedding/`                              | `all-mpnet-base-v2` embeddings + logistic head + subject prior        | ~ −0.69       |
| 3 | `factor_pge/`                             | 2-PL factor model with item-text-to-factor MLP (MiniLM encoder)       | −0.62 → −0.66 |
| 4 | `factor_baseline_ensemble/`               | Per-benchmark logit-mean ensemble of factor + smoothed-prior baseline | −0.61 → −0.66 |
| 5 | `factor_pge_multiseed/`                   | 5-seed MiniLM factor ensemble (trained in parallel on Modal)          | −0.62         |

Detailed methodology, design choices, ablations, and full Codabench history
live in `REPORT.md`.

---

## Running on Modal (optional)

`modal_train_factor.py` runs the factor / PGE trainer on Modal GPUs. The
`train_multi_seed` entrypoint spawns N parallel single-seed jobs and gathers
their artifacts into `ensemble_artifacts/<label>/`:

```bash
pip install modal
modal token new   # one-time

modal run modal_train_factor.py::train_multi_seed \
    --encoder-id sentence-transformers/all-MiniLM-L6-v2 \
    --seeds 0,1,2,3,4 \
    --holdout-mode random_rows \
    --out-label minilm_multiseed_v2
```

The artifacts then plug directly into `codabench_submissions/factor_pge_multiseed/artifacts/`.

---

## Running on Slurm

Schmidt-cluster batch scripts live under `slurm/`:

```bash
sbatch slurm/train_baseline.sbatch
sbatch slurm/train_embedding.sbatch
sbatch slurm/train_factor.sbatch
```

All three use the `cs321m` partition/QoS and reuse the cached parquet when
present.

---

## Tests

```bash
pytest tests/
```

The tests import each submission's `model.py`, load its artifacts, and call
`predict()` on synthetic inputs to catch regressions in artifact handling and
graceful-degrade paths.
