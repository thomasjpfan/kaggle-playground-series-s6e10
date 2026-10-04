# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Kaggle Playground Series S6E10: binary classification of `satisfaction`. The data is a synthetic version of the *Airline Passenger Satisfaction* dataset. Submissions give a probability per `id`, and CV is scored with ROC AUC. `analysis.md` holds the EDA findings and modeling ideas. Read it before doing any feature engineering. Use only the competition's own `train.csv` and `test.csv`, never the original dataset.

## Running code

All computation runs remotely on [Modal](https://modal.com). Nothing reads data locally. The local environment (managed by `uv`, Python >=3.14) only has `modal` installed. Each script declares its own remote image and dependencies with `modal.Image...uv_pip_install(...)`, so you add a new library to the script's image, not to `pyproject.toml`.

```bash
uv run modal run eda.py                    # EDA (local_entrypoint -> eda)
uv run modal run eda2.py --no-checks --kinds base,te_fd   # feature-set experiments + OOF error analysis
uv run modal run eda.py::list_files        # list files on the data volume
uv run modal run eval.py                   # LightGBM 5-fold CV, CPU (count + in-fold target encoding)
uv run modal run eval_catboost.py          # CatBoost 5-fold CV, L4 GPU (variants: native, allcat)
uv run modal run eval_catboost.py --variants allcat
uv run modal run eval_sdm.py               # tabular foundation models (SDM), A100 GPU
uv run modal run eval_sdm.py --models tabiclv2 --folds 0 --context-sizes 10000,50000 --num-estimators 4,8
uv run modal run blend.py                  # rank-blend every OOF file in /data/oof-s6e10/
uv run modal run tune.py --model lightgbm  # Optuna on fold 0 (parallel batches), then 5-fold confirm of top configs
uv run modal run tune.py --model catboost --n-trials 60 --batch-size 12 --confirm-top 2
uv run modal run blend.py --names catboost_allcat,lightgbm_te
uv run modal run blend.py --submit-weights "catboost_allcat:0.6,lightgbm_te:0.4"   # write a submission CSV
uv run modal volume get kaggle /submissions-s6e10/<file>.csv .                     # download it for Kaggle upload
```

The `eval_sdm.py` CLI options are comma-separated lists. They form a grid of `model × context_size × num_estimators`, and each config runs on the given folds.

There are no tests, linter or build step.

## Architecture and conventions

- **Data:** the Modal volume `kaggle` is mounted at `/data`. Competition files (`train.csv`, `test.csv`) are in `/data/playground-series-s6e10/`. The Modal volume `hf-cache` is mounted at `/hf-cache` as `HF_HOME` for model weights. Call `hf_cache.commit()` after downloading weights so they persist.
- **OOF predictions** for ensembling go in `/data/oof-s6e10/<model>_<variant>.npy`, in `train.csv` row order. `eval.py` (`lightgbm_te`) and `eval_catboost.py` (`catboost_<variant>`) write them; call `volume.commit()` after writing. `blend.py` scores single models, pairwise weights and greedy ensemble selection on rank-normalized OOFs.
- **Test predictions:** `eval.py` and `eval_catboost.py` also average their 5 fold models' predictions on `test.csv` into `/data/test-preds-s6e10/<name>.npy` (same name as the OOF file, `test.csv` row order). `blend.py --submit-weights` rank-blends those with the given weights, prints the same blend's OOF AUC, checks ids against `sample_submission.csv`, and writes `/data/submissions-s6e10/<weights>_<timestamp>.csv`.
- **Test features** must match the CV features: count encodings use train-derived counts (unseen → 0), and the `Flight Distance` target encoding for test is fitted on each fold's training rows, like the validation rows. Joins that must keep row order pass `maintain_order="left"`.
- **Modal resources:** set `cpu=`, `gpu=` and `timeout=` on `@app.function` as needed, but never `memory=`, since Modal bursts memory when required.
- **Eval script pattern** (`eval.py`, `eval_catboost.py`, `eval_sdm.py`):
  - `train_fold` is a Modal function that handles one fold.
  - `cross_validate` fans the folds out with `.map` / `.starmap`, then aggregates the per-fold AUC and the OOF AUC.
  - `@app.local_entrypoint() main` prints the summary.
  - New model experiments should follow this pattern, with each script defining its own `modal.App` name.
- **Fold consistency:** every eval uses `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the full train set. This keeps OOF predictions and AUCs comparable across scripts. Keep these settings in any new eval.
- `eval_sdm.py` uses NVIDIA `structured-data-models` (`sdm`), pinned to a git commit, on Python 3.12 with cudf. It samples `CONTEXT_SIZE` in-context rows per estimator from the training fold, then predicts the validation fold in batches. When a fold runs out of GPU memory it returns `auc=None` instead of failing the whole run.
- `tune.py` imports the feature code from `eval.py` and `eval_catboost.py` (shipped with `add_local_python_source`), so tuned models use the same features. Trial logs go to `/data/tune-s6e10/<model>.json`; confirmed configs save OOFs as `<model>_tuned<rank>`. Fold 0 is the tuning fold, so compare configs on folds 1–4.
- `eval_catboost.py` variant `allcat` adds string-typed categorical copies of every rating, `Age` and `Flight Distance`, alongside the numeric originals.
- Polars is the dataframe library. The exception is that CatBoost `Pool`s are built from `.to_pandas()`.
