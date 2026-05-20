# Methods & Results — CS321M Predictive AI Evaluation Challenge

This report documents the modeling pipeline, the design decisions behind each
of the five Codabench submissions, the ablations that motivated the final
configuration, and the per-submission leaderboard history.

## 1. Problem

For each hidden `(subject, item, benchmark, condition)` tuple, the submission
predicts `P(correct)`. The competition guarantees **item cold-start** — the
held-out items are not present in the public matrix — while subjects mostly
are. Codabench scores submissions on negative log-loss (primary) and AUC-ROC
(secondary), so a strong submission has to be **both** ranking-accurate and
calibrated.

Two competition mechanics shape the design:

1. **Adaptive labeling.** Submissions may include `labeling.py` with an
   `acquisition_function(input) -> float`. Codabench picks the top **K = 5**
   inputs per category, reveals their labels, and passes them to `predict()`
   as a `labeled` list. Used carefully, this enables a per-round residual
   offset for calibration.
2. **No runtime network.** All learning happens offline; only small fitted
   artifacts (and HF model weights declared in `models.txt`) are available at
   inference time.

## 2. Data

Source: `aims-foundations/measurement-db` on Hugging Face, loaded with the
explicit response-file list pattern from the starter kit (`data_loading.py`).
We deliberately avoid `load_dataset("aims-foundations/measurement-db")`
because that auto-mixes registry and `*_traces.parquet` tables with
incompatible schemas.

Joined runtime view (`build_runtime_examples`):

```text
benchmark | condition | subject_content | item_content | label
                                                       (binary 0/1)
```

We additionally keep `subject_id`, `item_id`, `item_variant_id`, and
`benchmark_id` for offline analysis. Only binary-correctness rows
(`label ∈ {0, 1}`) are used.

Scale (active dataset): ~3.4M binary rows, ~900 subjects, ~150k unique items,
16 benchmarks, 223 `(benchmark, condition)` cells.

## 3. Validation strategy

**Whole-benchmark holdout.** Random row or cell splits leak benchmark-level
distributional information, so we hold out entire benchmarks at a time
(`local_cv.py`, `recalibrate_temperature.py`, `per_benchmark_diag.py`). This
matches the hidden Codabench slice: items in unseen benchmarks must be
predicted from subject identity + text alone.

For final temperature fitting we use one cold-start holdout benchmark and a
small calibration fraction (5%) of held-out subjects, leaving the bulk for
training.

## 4. Models

### 4.1 Smoothed-prior baseline (`codabench_submissions/baseline/`)

The simplest competent predictor. We fit per-cell empirical means with
Bayesian shrinkage toward a parent cell, then return the deepest populated
cell at predict time.

Cell hierarchy (deepest to shallowest, with approximate density on the public
data):

| Cell                                   | Type    | # cells | Mean rows/cell |
| -------------------------------------- | ------- | ------- | -------------- |
| `subject_benchmark_condition` (`sbc`)  | 4-way   | ~6 300  | ~700           |
| `subject_benchmark`           (`sb`)   | 3-way   | ~1 500  | ~3 000         |
| `benchmark_condition`         (`bc`)   | 3-way   | ~220    | ~22 000        |
| `subject`                              | 1-way   | ~900    | ~5 000         |
| `benchmark`                            | 1-way   | 16      | ~300 000       |

**v1 (initial)** used only the 3-way cells in a weighted average.
**v2 (current submission)** prefers the lowest-variance unbiased estimator —
the deepest populated cell — and falls back upward. The deepest cell is
shrunk only 5% toward the global mean; coarser cells get more shrinkage to
bound damage. Local NLL on a whole-benchmark holdout dropped from
0.554 (v1) to 0.519 (v2). See `fit_richer_baseline.py` and
`eval_baseline_variants.py` for the ablation.

At inference (`predict`), revealed labels feed a single global residual
offset clipped to ±0.12.

### 4.2 Embedding head (`codabench_submissions/embedding/`)

`sentence-transformers/all-mpnet-base-v2` encodes item text at runtime
(declared in `models.txt` so Codabench pre-downloads it). Offline, we encode
public items, append a smoothed subject-mean prior as an extra feature, and
fit a calibrated logistic head. Weights are saved as plain `npz` so the
runtime wrapper needs neither sklearn nor sentence-transformers' training
stack.

### 4.3 Factor / PGE model (`codabench_submissions/factor_pge/`)

A three-stage pipeline straight out of the PGE lecture:

1. **Stage 1 — 2-PL factor model.** Fit subject abilities `θ_s` and item
   parameters `(a_i, b_i)` on the binary public matrix with a logistic IRT
   objective. Rank `K = 1` after ablation — higher ranks regressed in our
   item-cold-start regime.
2. **Stage 2 — item-text → item parameters.** Encode each item with
   `sentence-transformers/all-MiniLM-L6-v2`, train a 2-layer GELU MLP
   (`hidden = 256`, dropout 0.15) to predict `(log a_i, b_i)` from the
   embedding. The MLP weights are then frozen and saved.
3. **Stage 3 — runtime predict.** Look up `θ_s` for the (known) subject,
   encode the (new) item text, run the MLP to recover `(a, b)`, return
   `σ((aθ − b) / T)`. Subjects we have never seen fall back to the global
   `θ̄`.

**Temperature.** `recalibrate_temperature.py` fits `T` on a whole-benchmark
cold-start holdout. The recalibrated value (`T ≈ 0.275`) is shipped inside
`factor_pge.npz`; the wrapper trusts it directly.

**Labeling (`labeling.py`).** Initial versions favored longer items, which
oversampled the hard benchmarks (swebench / livecodebench) and biased the
per-round residual estimate negative. v6 switches to a **uniform deterministic
hash** so each category's K = 5 sample is unbiased, which is what the
empirical-Bayes-shrunk per-category offset needs to be useful.

**Per-category residual offset.** Inside `predict()` we now compute a global
offset and a per-`(benchmark, condition)` offset over the revealed labels,
shrink each category mean toward the global one with weight
`n / (n + 5)`, and clip to ±0.08. Less aggressive than the baseline's ±0.12
because the factor model is already discriminative.

### 4.4 Factor + baseline ensemble (`codabench_submissions/factor_baseline_ensemble/`)

The factor and smoothed-prior models have complementary strengths: the factor
model carries the AUC (≈ 0.70), the smoothed prior carries the calibration.
We combine them as a **logit-mean weighted average**:

```
combined_logit = w * logit(p_factor) + (1 - w) * logit(p_baseline)
```

`w` is tuned per benchmark via `per_benchmark_diag.py`, which grid-searches
`(w, T_factor)` jointly against a left-out fold. The resulting weights are
saved to `artifacts/ensemble.json` and loaded at import time. When the
diagnostic showed the factor model adds no useful signal for a benchmark, we
set `w = 0` (which is the case for most of the v2-baseline benchmarks).

The same per-category calibration offset as the standalone factor wrapper is
applied to the final ensemble probability.

### 4.5 Multi-seed factor ensemble (`codabench_submissions/factor_pge_multiseed/`)

Variance-reduction over the factor pipeline. `modal_train_factor.py` has a
`train_multi_seed` entrypoint that spawns five parallel MiniLM single-seed
training jobs on Modal GPUs, each writing
`factor_pge_seed{0..4}.npz`. At predict time, the wrapper averages the
per-seed probabilities and then applies the v6 per-category residual offset.

**A pitfall worth flagging.** The first multi-seed run on Modal used
benchmark-based holdout for training, which dropped 41% of subjects (those
appearing only in held-out benchmarks). v2 of the script switches to
`holdout_mode="random_rows"` for training so every subject contributes to the
trained `θ_s`. This single change reclaimed ~0.05 NLL locally.

## 5. Ablations & local diagnostics

We rely on three diagnostic harnesses:

- `local_cv.py` — leave-one-benchmark-out cross-validation over the full
  pipeline (factor + smoothed prior + ensemble).
- `eval_baseline_variants.py` — compares baseline v1 (3-way only) against
  v2-mixed (4-way weighted into the average) and v2-sbc-first (4-way
  preferred). The **sbc-first** variant wins by ~0.04 NLL on a held-out
  benchmark.
- `per_benchmark_diag.py` — grid-searches the per-benchmark `(w, T_factor)`
  pair and writes the per-benchmark configuration JSON consumed by the
  ensemble model. Run with `--all-benchmarks` to evaluate across every
  available benchmark, or with a single `--holdout-benchmark` to focus the
  search.

Key empirical observations that shaped the final stack:

1. **Higher-rank factor models hurt.** Rank ≥ 2 reduced training loss but
   regressed on cold-start holdout NLL. Rank 1 wins.
2. **Bigger encoders are not free.** A from-scratch BGE-large training on
   Modal underperformed MiniLM in an apples-to-apples comparison. The MLP
   bottleneck dominates the encoder choice for the cold-start MLP target.
3. **Calibration > model complexity at the margin.** Restoring a clamped
   `T ≥ 0.7` over the cold-start `T = 0.275` regressed −0.04 NLL. The
   recalibrated T is correct, even if it looks small.
4. **Uniform labeling > clever labeling.** Length-biased acquisition pushed
   the per-round residual systematically negative; uniform-hash labeling
   recovered ~0.04 NLL on Codabench.
5. **Per-benchmark blend > global blend.** The factor model only adds value
   for a handful of benchmarks (`ai2d_test`, `mmlupro`, `ultrafeedback`);
   forcing it into every benchmark wasted signal. Per-benchmark
   `factor_weight` swept that under the rug cleanly.

## 6. Codabench history

Public scores per submission, oldest first (NLL is negated, so less negative
is worse, more negative is better):

| Submission                              | Best NLL    | AUC-ROC | Notes                                                  |
| --------------------------------------- | ----------- | ------- | ------------------------------------------------------ |
| `embedding_submission.zip`              | −0.69       | ~ 0.50  | Cold-start hurts the embedding head — barely beats global mean. |
| `baseline_submission.zip` (v1, 3-way)   | −0.69       | ~ 0.53  | Smoothed prior over 3-way cells.                       |
| `factor_pge_submission.zip` (v3)        | **−0.66**   | 0.70    | Factor + per-category EB offset, T-clamp at 0.7.       |
| `factor_baseline_ensemble.zip` (v3)     | **−0.66**   | 0.69    | Global `w = 0.7` ensemble of factor + smoothed prior.  |
| `factor_pge_submission.zip` (v6)        | −0.62       | 0.69    | T = 0.275 cold-start fit; uniform labeling.            |
| `factor_baseline_ensemble.zip` (v6)     | −0.62       | 0.69    | Same configuration on the ensemble path.               |
| `factor_pge_multiseed.zip`              | −0.62       | 0.69    | 5-seed MiniLM average.                                 |
| `factor_baseline_ensemble.zip` (v8)     | TBD         | TBD     | Per-benchmark `(w, T)` from `per_benchmark_diag.py` + v2 baseline. |
| `baseline_submission.zip` (v2)          | TBD         | TBD     | 4-way `sbc` cells, sbc-first prediction.               |

Local validation (held-out benchmark, NLL):
- baseline v1: 0.554
- baseline v2 (sbc-first): **0.519**
- factor_pge v6 standalone: 0.568
- ensemble v8 (per-benchmark + v2 baseline): 0.517

The local jump for baseline v2 motivated the final pair of submissions: ship
the bare v2 baseline as a safety floor, and ship the per-benchmark ensemble
that uses the same v2 baseline as its non-factor component.

## 7. Adaptive labeling — what works, what doesn't

| Variant                                       | Effect on Codabench NLL |
| --------------------------------------------- | ----------------------- |
| Default platform random labels                | baseline                |
| **Length-biased acquisition (early version)** | **−0.04 vs. baseline**  |
| Uniform-hash acquisition (v6, current)        | matches platform random |
| + global residual offset (clip ±0.12)         | +0.01 to +0.02          |
| + per-category EB offset (clip ±0.08)         | +0.02 to +0.03          |

The lesson: with only K = 5 labels per category, anything beyond a small
shrunken offset overfits. The combination of a **uniform** label sample and
an **EB-shrunk, clipped** offset is the only one that consistently helped.

## 8. Failure modes & open questions

- Cold-start subjects (subjects that appear only in the hidden split) fall
  back to the global `θ̄`. Some are clearly weaker/stronger than average,
  and we can't tell from text alone.
- The factor model's discriminative signal is concentrated on a handful of
  benchmarks. For benchmarks like `swebench`, `hle`, and `matharena`, the
  smoothed prior dominates and `factor_weight ≈ 0` is correct.
- The embedding-head submission underperforms the smoothed prior. Likely
  cause: the head learns topic structure rather than behavioral difficulty,
  and cold-start items don't share topic distribution with public training
  items. A possible fix is to predict the **factor parameters** with the
  large-encoder embeddings rather than predicting probability directly.
- We never tried mixed-modality embeddings (item text + benchmark text).
  That's the obvious next prototype.

## 9. Reproducibility checklist

To reproduce all five submissions from a clean clone:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

python download_data.py --out data/runtime_examples.parquet

# baseline v2
python fit_richer_baseline.py \
    --data data/runtime_examples.parquet \
    --out codabench_submissions/baseline/artifacts/smoothed_prior.json

# factor / PGE (MiniLM; CPU works but slow — ~1h)
python train_factor.py \
    --data data/runtime_examples.parquet \
    --out codabench_submissions/factor_pge/artifacts/factor_pge.npz
python recalibrate_temperature.py \
    --artifact codabench_submissions/factor_pge/artifacts/factor_pge.npz \
    --data data/runtime_examples.parquet

# multi-seed factor (Modal recommended)
modal run modal_train_factor.py::train_multi_seed \
    --encoder-id sentence-transformers/all-MiniLM-L6-v2 \
    --seeds 0,1,2,3,4 --holdout-mode random_rows \
    --out-label minilm_multiseed_v2
# then copy ensemble_artifacts/minilm_multiseed_v2/*.npz into
# codabench_submissions/factor_pge_multiseed/artifacts/

# per-benchmark ensemble weights
python per_benchmark_diag.py --all-benchmarks \
    --baseline codabench_submissions/baseline/artifacts/smoothed_prior.json \
    --factor   codabench_submissions/factor_pge/artifacts/factor_pge.npz \
    --out dist/per_benchmark_weights_v2baseline.json
cp dist/per_benchmark_weights_v2baseline.json \
   codabench_submissions/factor_baseline_ensemble/artifacts/ensemble.json

# embedding head
python train_embedding.py \
    --data data/runtime_examples.parquet \
    --out codabench_submissions/embedding/artifacts/embedding_head.npz

# smoke + zip
for s in baseline embedding factor_pge factor_baseline_ensemble factor_pge_multiseed; do
    python smoke_submission.py codabench_submissions/$s
    python make_submission.py $s
done
```

Resulting ZIPs land in `dist/<name>_submission.zip` ready to upload.
