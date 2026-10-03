import modal

app = modal.App("kaggle-playground-eval")

volume = modal.Volume.from_name("kaggle")
VOLUME_PATH = "/data"
DATA_DIR = f"{VOLUME_PATH}/playground-series-s6e10"

image = modal.Image.debian_slim().uv_pip_install("polars", "lightgbm", "scikit-learn")

TARGET = "satisfaction"
CATEGORICAL = ["Gender", "Customer Type", "Type of Travel", "Class"]
N_FOLDS = 5
SEED = 42

PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
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
    features = [c for c in train.columns if c not in ("id", TARGET)]
    X = train.select(features).to_numpy().astype("float32")
    y = train[TARGET].cast(pl.Int8).to_numpy()
    return X, y, features


@app.function(image=image, volumes={VOLUME_PATH: volume}, cpu=8, memory=8192, timeout=3600)
def train_fold(fold: int):
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    X, y, features = load_train()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    train_idx, valid_idx = list(skf.split(X, y))[fold]

    cat_idx = [features.index(c) for c in CATEGORICAL]
    dtrain = lgb.Dataset(X[train_idx], y[train_idx], feature_name=features, categorical_feature=cat_idx)
    dvalid = lgb.Dataset(X[valid_idx], y[valid_idx], reference=dtrain)

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


@app.function(image=image, timeout=3600)
def cross_validate():
    import numpy as np
    from sklearn.metrics import roc_auc_score

    results = sorted(train_fold.map(range(N_FOLDS)), key=lambda r: r["fold"])

    oof_y = np.concatenate([r["y"] for r in results])
    oof_preds = np.concatenate([r["preds"] for r in results])

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
