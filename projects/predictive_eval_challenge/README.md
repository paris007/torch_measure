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

## Schmidt Slurm Workflow

Submit the baseline job from the repository root:

```bash
sbatch projects/predictive_eval_challenge/slurm/train_baseline.sbatch
```

The job uses the `cs321m` partition and QoS, downloads the public data, trains
the baseline artifact, smoke-tests the submission folder, and builds the ZIP.

## Tests

Run the reusable predictive-evaluation unit tests:

```bash
/opt/anaconda3/envs/torch_measure/bin/python -m pytest tests/test_models/test_predictive_eval.py
```
