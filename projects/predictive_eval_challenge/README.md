# Predictive Evaluation Challenge

This project contains competition-specific code for the CS321M Predictive
Evaluation Challenge. Reusable model logic lives in `src/torch_measure/models`;
this folder holds data loading, training, validation, Codabench submission
wrappers, Slurm jobs, and report notes.

## Runtime Contract

Codabench imports a flat submission ZIP and calls:

```python
def predict(input: dict, labeled: list[dict] | None = None) -> float:
    ...
```

`input` contains exactly four string fields:

- `benchmark`
- `condition`
- `subject_content`
- `item_content`

`labeling.py` is optional and may define:

```python
def acquisition_function(input: dict) -> float:
    ...
```

The submission must not make outbound network calls at runtime. Small fitted
artifacts should be bundled in the ZIP under `artifacts/`; large Hugging Face
models should be declared in `models.txt`.

## Local Structure

```text
projects/predictive_eval_challenge/
  data_loading.py
  download_data.py
  train_baseline.py
  train_embedding.py
  validate.py
  smoke_submission.py
  make_submission.py
  codabench_submissions/
    baseline/
    embedding/
    factor_pge/
  slurm/
  report/
```

Each `codabench_submissions/<name>/` directory is shaped so it can be zipped
directly, with `model.py` at the top level.

## Data Loading

Use `data_loading.py` to load the public Hugging Face parquet collection. It
explicitly lists response parquet files and excludes registry and trace files;
do not use the shortcut `load_dataset("aims-foundations/measurement-db")`.

## Local Baseline Workflow

From the repository root:

```bash
PYTHON=/opt/anaconda3/envs/torch_measure/bin/python

$PYTHON -m pip install -e .
$PYTHON projects/predictive_eval_challenge/download_data.py \
  --out projects/predictive_eval_challenge/data/runtime_examples.parquet
$PYTHON projects/predictive_eval_challenge/train_baseline.py \
  --data projects/predictive_eval_challenge/data/runtime_examples.parquet
$PYTHON projects/predictive_eval_challenge/validate.py \
  --data projects/predictive_eval_challenge/data/runtime_examples.parquet \
  --max-rows 10000
$PYTHON projects/predictive_eval_challenge/smoke_submission.py \
  projects/predictive_eval_challenge/codabench_submissions/baseline
$PYTHON projects/predictive_eval_challenge/make_submission.py baseline
```

The built ZIP is written to:

```text
projects/predictive_eval_challenge/dist/baseline_submission.zip
```

## Local Embedding Workflow

The embedding pipeline encodes each unique item text with
`sentence-transformers/all-mpnet-base-v2`, appends a smoothed subject-mean
prior, and fits a logistic head. Weights are saved as plain numpy arrays so
the Codabench wrapper does not need scikit-learn at runtime.

```bash
$PYTHON projects/predictive_eval_challenge/train_embedding.py \
  --data projects/predictive_eval_challenge/data/runtime_examples.parquet \
  --max-rows 1000000
$PYTHON projects/predictive_eval_challenge/smoke_submission.py \
  projects/predictive_eval_challenge/codabench_submissions/embedding
$PYTHON projects/predictive_eval_challenge/make_submission.py embedding
```

The embedding submission declares the encoder in `models.txt`, so the
Codabench platform pre-fetches it before importing `model.py`.

## Schmidt Slurm Workflow

Submit the baseline job from the repository root:

```bash
sbatch projects/predictive_eval_challenge/slurm/train_baseline.sbatch
```

Submit the embedding job from the repository root:

```bash
sbatch projects/predictive_eval_challenge/slurm/train_embedding.sbatch
```

The embedding job requests a GPU (`--gres=gpu:1`) for sentence-transformer
encoding and reuses the cached parquet from the baseline job if it exists.
Both jobs use the `cs321m` partition and QoS.

## Tests

Run the reusable predictive-evaluation unit tests:

```bash
/opt/anaconda3/envs/torch_measure/bin/python -m pytest tests/test_models/test_predictive_eval.py
```
