"""Optuna tuning of the tree models on one fold, then 5-fold confirmation of the best configs.

Features come from eval.py (LightGBM: counts + in-fold target encoding) and
eval_catboost.py (CatBoost "allcat"), so tuned models stay comparable to them.
"""

import modal

import eval as lgb_eval
import eval_catboost as cb_eval

app = modal.App("kaggle-playground-tune")

volume = modal.Volume.from_name("kaggle")
VOLUME_PATH = "/data"
DATA_DIR = f"{VOLUME_PATH}/playground-series-s6e10"
OOF_DIR = f"{VOLUME_PATH}/oof-s6e10"
TUNE_DIR = f"{VOLUME_PATH}/tune-s6e10"

image = (
    modal.Image.debian_slim()
    .uv_pip_install("polars", "pandas", "pyarrow", "lightgbm", "catboost", "scikit-learn", "optuna")
    .add_local_python_source("eval", "eval_catboost")
)

TARGET = "satisfaction"
N_FOLDS = 5
SEED = 42  # Same folds as eval.py.
TUNE_FOLD = 0


def lgb_space(trial):
    return {
        "max_bin": trial.suggest_categorical("max_bin", [255, 1023, 4095]),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 2000, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 100, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "cat_smooth": trial.suggest_float("cat_smooth", 1, 100, log=True),
        "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 1e-3, 10, log=True),
    }


def cb_space(trial):
    return {
        "depth": trial.suggest_int("depth", 6, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 30, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.1, 10, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 2),
        "border_count": trial.suggest_categorical("border_count", [128, 254]),
        "max_ctr_complexity": trial.suggest_int("max_ctr_complexity", 1, 4),
        "one_hot_max_size": trial.suggest_categorical("one_hot_max_size", [2, 8]),
    }


@app.function(image=image, volumes={VOLUME_PATH: volume}, cpu=8, timeout=2 * 3600)
def lgb_fold(params: dict, fold: int):
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    X, y, features = lgb_eval.load_train()
    train_idx, valid_idx = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y))[fold]
    X, features = lgb_eval.add_target_encoding(X, y, features, train_idx, valid_idx)

    p = {**lgb_eval.PARAMS, **params}
    dataset_params = {"max_bin": p["max_bin"]}
    cat_idx = [features.index(c) for c in lgb_eval.CATEGORICAL]
    dtrain = lgb.Dataset(X[train_idx], y[train_idx], feature_name=features, categorical_feature=cat_idx, params=dataset_params)
    dvalid = lgb.Dataset(X[valid_idx], y[valid_idx], reference=dtrain, params=dataset_params)
    model = lgb.train(p, dtrain, 20000, valid_sets=[dvalid], callbacks=[lgb.early_stopping(200, verbose=False)])
    preds = model.predict(X[valid_idx], num_iteration=model.best_iteration)
    return {"fold": fold, "auc": float(roc_auc_score(y[valid_idx], preds)), "iters": model.best_iteration,
            "valid_idx": valid_idx, "preds": preds}


@app.function(image=image, gpu="L4", volumes={VOLUME_PATH: volume}, timeout=4 * 3600)
def cb_fold(params: dict, fold: int):
    from catboost import CatBoostClassifier, Pool
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    X, y, cat_features = cb_eval.load_train("allcat")
    train_idx, valid_idx = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y))[fold]
    dtrain = Pool(X.iloc[train_idx], y[train_idx], cat_features=cat_features)
    dvalid = Pool(X.iloc[valid_idx], y[valid_idx], cat_features=cat_features)
    model = CatBoostClassifier(**{**cb_eval.PARAMS, "iterations": 20000, **params})
    model.fit(dtrain, eval_set=dvalid, use_best_model=True)
    preds = model.predict_proba(dvalid)[:, 1]
    return {"fold": fold, "auc": float(roc_auc_score(y[valid_idx], preds)), "iters": model.get_best_iteration(),
            "valid_idx": valid_idx, "preds": preds}


MODELS = {"lightgbm": (lgb_space, lgb_fold), "catboost": (cb_space, cb_fold)}


@app.function(image=image, volumes={VOLUME_PATH: volume}, timeout=12 * 3600)
def tune(model: str, n_trials: int, batch_size: int, learning_rate: float):
    """Batched ask/tell: each batch of trials runs in parallel containers on TUNE_FOLD."""
    import json
    import os

    import optuna

    space, fold_fn = MODELS[model]
    sampler = optuna.samplers.TPESampler(seed=SEED, multivariate=True, constant_liar=True, n_startup_trials=batch_size)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    os.makedirs(TUNE_DIR, exist_ok=True)

    log = []
    while len(log) < n_trials:
        trials = [study.ask() for _ in range(min(batch_size, n_trials - len(log)))]
        params = [{**space(t), "learning_rate": learning_rate} for t in trials]
        results = list(fold_fn.starmap([(p, TUNE_FOLD) for p in params], return_exceptions=True))
        for t, p, r in zip(trials, params, results):
            if isinstance(r, dict):
                study.tell(t, r["auc"])
                log.append({"number": t.number, "auc": r["auc"], "iters": int(r["iters"]), "params": p})
                print(f"trial {t.number}: auc={r['auc']:.6f} iters={r['iters']} {p}")
            else:
                study.tell(t, state=optuna.trial.TrialState.FAIL)
                log.append({"number": t.number, "auc": None, "error": repr(r), "params": p})
                print(f"trial {t.number}: failed {r!r}")
        with open(f"{TUNE_DIR}/{model}.json", "w") as f:
            json.dump(log, f, indent=1)
        volume.commit()
        print(f"-- {len(log)}/{n_trials} done, best so far {study.best_value:.6f}")
    return log


@app.function(image=image, volumes={VOLUME_PATH: volume}, timeout=12 * 3600)
def confirm(model: str, params: dict, name: str):
    """Run all folds for one config and save its OOF for blend.py."""
    import os

    import numpy as np
    import polars as pl
    from sklearn.metrics import roc_auc_score

    _, fold_fn = MODELS[model]
    results = sorted(fold_fn.starmap([(params, f) for f in range(N_FOLDS)]), key=lambda r: r["fold"])
    y = pl.read_csv(f"{DATA_DIR}/train.csv", columns=[TARGET])[TARGET].cast(pl.Int8).to_numpy()
    oof = np.zeros(len(y))
    for r in results:
        oof[r["valid_idx"]] = r["preds"]
    os.makedirs(OOF_DIR, exist_ok=True)
    np.save(f"{OOF_DIR}/{name}.npy", oof)
    volume.commit()
    return {
        "name": name,
        "params": params,
        "folds": [{"fold": r["fold"], "auc": r["auc"], "iters": int(r["iters"])} for r in results],
        "oof_auc": float(roc_auc_score(y, oof)),
    }


@app.local_entrypoint()
def main(
    model: str = "lightgbm",
    n_trials: int = 60,
    batch_size: int = 12,
    learning_rate: float = 0.05,
    confirm_top: int = 2,
    confirm_learning_rate: float = 0.0,
):
    import statistics

    log = tune.remote(model, n_trials, batch_size, learning_rate)
    ok = sorted((t for t in log if t["auc"] is not None), key=lambda t: -t["auc"])
    print(f"\n== top {model} trials on fold {TUNE_FOLD} ==")
    for t in ok[:10]:
        print(f"auc={t['auc']:.6f} iters={t['iters']} {t['params']}")

    # Optionally refit the best configs with a lower learning rate for the 5-fold run.
    lr = confirm_learning_rate or learning_rate
    jobs = [(model, {**t["params"], "learning_rate": lr}, f"{model}_tuned{i}") for i, t in enumerate(ok[:confirm_top])]
    for s in confirm.starmap(jobs):
        aucs = [f["auc"] for f in s["folds"]]
        print(f"\n== {s['name']} (lr={lr}) ==")
        for f in s["folds"]:
            print(f"fold {f['fold']}: auc={f['auc']:.6f} iters={f['iters']}")
        # Fold 0 was used for tuning, so folds 1-4 are the unbiased comparison.
        print(f"mean fold auc: {statistics.mean(aucs):.6f}  mean folds 1-4: {statistics.mean(aucs[1:]):.6f}")
        print(f"OOF auc: {s['oof_auc']:.6f}")
