import modal

app = modal.App("kaggle-playground-eval")

volume = modal.Volume.from_name("kaggle")
VOLUME_PATH = "/data"
DATA_DIR = f"{VOLUME_PATH}/playground-series-s6e10"
OOF_DIR = f"{VOLUME_PATH}/oof-s6e10"
OOF_NAME = "lightgbm_te"

image = modal.Image.debian_slim().uv_pip_install("polars", "lightgbm", "scikit-learn")

TARGET = "satisfaction"
CATEGORICAL = ["Gender", "Customer Type", "Type of Travel", "Class"]
N_FOLDS = 5
SEED = 42
COUNT_COLS = ["Age", "Flight Distance"]
# Exact Flight Distance values carry signal beyond a smooth trend (see analysis.md).
TE_COLS = ["Flight Distance"]
TE_ALPHA = 20

# Tuned with tune.py (lightgbm_tuned0). max_bin=4095 keeps exact Flight Distance values apart.
PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "max_bin": 4095,
    "num_leaves": 100,
    "min_child_samples": 50,
    "lambda_l2": 15.19,
    "feature_fraction": 0.509,
    "bagging_fraction": 0.941,
    "bagging_freq": 1,
    "cat_smooth": 2.82,
    "min_sum_hessian_in_leaf": 0.00286,
    "seed": SEED,
    "verbosity": -1,
}


def load_train():
    import polars as pl

    train = pl.read_csv(f"{DATA_DIR}/train.csv")
    # Encode categoricals as integer codes so LightGBM can treat them natively.
    train = train.with_columns(
        pl.col(c).cast(pl.Categorical).to_physical().cast(pl.Int32) for c in CATEGORICAL
    )
    train = train.with_columns(pl.len().over(c).alias(f"{c}_count") for c in COUNT_COLS)
    features = [c for c in train.columns if c not in ("id", TARGET)]
    X = train.select(features).to_numpy().astype("float32")
    y = train[TARGET].cast(pl.Int8).to_numpy()
    return X, y, features


def target_encode(key, y, fit_idx, apply_idx, prior):
    import polars as pl

    stats = (
        pl.DataFrame({"k": key[fit_idx], "y": y[fit_idx]})
        .group_by("k")
        .agg(pl.col("y").sum().alias("s"), pl.len().alias("n"))
    )
    a = pl.DataFrame({"k": key[apply_idx]}).join(stats, on="k", how="left").fill_null(0)
    return ((a["s"] + TE_ALPHA * prior) / (a["n"] + TE_ALPHA)).to_numpy()


def add_target_encoding(X, y, features, train_idx, valid_idx):
    """Smoothed target encoding: inner K-fold OOF on the training fold, the full training fold for validation."""
    import numpy as np
    from sklearn.model_selection import KFold

    prior = y[train_idx].mean()
    encoded = []
    for c in TE_COLS:
        key = X[:, features.index(c)]
        enc = np.empty(len(y), dtype="float32")
        for inner_fit, inner_apply in KFold(N_FOLDS, shuffle=True, random_state=SEED).split(train_idx):
            enc[train_idx[inner_apply]] = target_encode(key, y, train_idx[inner_fit], train_idx[inner_apply], prior)
        enc[valid_idx] = target_encode(key, y, train_idx, valid_idx, prior)
        encoded.append(enc)
    return np.column_stack([X, *encoded]), features + [f"{c}_te" for c in TE_COLS]


@app.function(image=image, volumes={VOLUME_PATH: volume}, cpu=8, timeout=3600)
def train_fold(fold: int):
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    X, y, features = load_train()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    train_idx, valid_idx = list(skf.split(X, y))[fold]
    X, features = add_target_encoding(X, y, features, train_idx, valid_idx)

    cat_idx = [features.index(c) for c in CATEGORICAL]
    dataset_params = {"max_bin": PARAMS["max_bin"]}
    dtrain = lgb.Dataset(
        X[train_idx], y[train_idx], feature_name=features, categorical_feature=cat_idx, params=dataset_params
    )
    dvalid = lgb.Dataset(X[valid_idx], y[valid_idx], reference=dtrain, params=dataset_params)

    model = lgb.train(
        PARAMS,
        dtrain,
        num_boost_round=5000,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    preds = model.predict(X[valid_idx], num_iteration=model.best_iteration)
    auc = roc_auc_score(y[valid_idx], preds)
    print(f"fold {fold}: auc={auc:.6f} best_iteration={model.best_iteration}")

    importance = dict(zip(features, model.feature_importance("gain").tolist()))
    return {
        "fold": fold,
        "auc": auc,
        "best_iteration": model.best_iteration,
        "valid_idx": valid_idx,
        "preds": preds,
        "y": y[valid_idx],
        "importance": importance,
    }


@app.function(image=image, volumes={VOLUME_PATH: volume}, timeout=3600)
def cross_validate():
    import os

    import numpy as np
    from sklearn.metrics import roc_auc_score

    results = sorted(train_fold.map(range(N_FOLDS)), key=lambda r: r["fold"])

    oof_y = np.concatenate([r["y"] for r in results])
    oof_preds = np.concatenate([r["preds"] for r in results])

    # Saved for ensembling; row order matches train.csv.
    oof = np.zeros(len(oof_y))
    for r in results:
        oof[r["valid_idx"]] = r["preds"]
    os.makedirs(OOF_DIR, exist_ok=True)
    np.save(f"{OOF_DIR}/{OOF_NAME}.npy", oof)
    volume.commit()

    importance = {}
    for r in results:
        for k, v in r["importance"].items():
            importance[k] = importance.get(k, 0.0) + v / N_FOLDS

    return {
        "folds": [
            {"fold": r["fold"], "auc": float(r["auc"]), "best_iteration": int(r["best_iteration"])}
            for r in results
        ],
        "oof_auc": float(roc_auc_score(oof_y, oof_preds)),
        "importance": importance,
    }


@app.local_entrypoint()
def main():
    import statistics

    summary = cross_validate.remote()

    aucs = [f["auc"] for f in summary["folds"]]
    for f in summary["folds"]:
        print(f"fold {f['fold']}: auc={f['auc']:.6f} best_iteration={f['best_iteration']}")
    print(f"mean fold auc: {statistics.mean(aucs):.6f} ± {statistics.pstdev(aucs):.6f}")
    print(f"OOF auc: {summary['oof_auc']:.6f}")

    print("\nmean gain importance:")
    for k, v in sorted(summary["importance"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:35s} {v:14.1f}")
