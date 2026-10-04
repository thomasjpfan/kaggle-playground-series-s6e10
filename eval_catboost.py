import modal

app = modal.App("kaggle-playground-eval-catboost")

volume = modal.Volume.from_name("kaggle")
VOLUME_PATH = "/data"
DATA_DIR = f"{VOLUME_PATH}/playground-series-s6e10"
OOF_DIR = f"{VOLUME_PATH}/oof-s6e10"
TEST_PRED_DIR = f"{VOLUME_PATH}/test-preds-s6e10"

image = modal.Image.debian_slim().uv_pip_install("polars", "pandas", "pyarrow", "catboost", "scikit-learn")

TARGET = "satisfaction"
CATEGORICAL = ["Gender", "Customer Type", "Type of Travel", "Class"]
N_FOLDS = 5
SEED = 42  # Same folds as eval.py.

# "native": only the 4 string columns are categorical.
# "allcat": also adds categorical copies of every rating, Age and Flight Distance,
# so CatBoost's ordered target statistics can use exact values and their combinations.
VARIANTS = ["native", "allcat"]
EXTRA_CAT = ["Age", "Flight Distance"]

# Tuned on "allcat" with tune.py (catboost_tuned0). max_ctr_complexity=4 lets CatBoost
# combine up to 4 categoricals; one_hot_max_size=2 target-encodes the ratings instead of one-hot.
PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "learning_rate": 0.05,
    "depth": 7,
    "l2_leaf_reg": 1.107,
    "random_strength": 3.133,
    "bagging_temperature": 0.108,
    "border_count": 128,
    "max_ctr_complexity": 4,
    "one_hot_max_size": 2,
    "iterations": 10000,
    "od_type": "Iter",
    "od_wait": 200,
    "random_seed": SEED,
    "task_type": "GPU",
    "metric_period": 50,
    "verbose": 0,
}


def load_data(variant: str):
    import polars as pl

    train = pl.read_csv(f"{DATA_DIR}/train.csv")
    test = pl.read_csv(f"{DATA_DIR}/test.csv")
    ratings = [c for c in train.columns if train[c].dtype == pl.Int64 and c not in ("id", *EXTRA_CAT) and train[c].max() <= 5]
    cat_features = list(CATEGORICAL)
    if variant == "allcat":
        copies = ratings + EXTRA_CAT
        train, test = (df.with_columns(pl.col(c).cast(pl.Utf8).alias(f"{c}_cat") for c in copies) for df in (train, test))
        cat_features += [f"{c}_cat" for c in copies]
    y = train[TARGET].cast(pl.Int8).to_numpy()
    X = train.drop("id", TARGET).to_pandas()
    X_test = test.drop("id").select(X.columns.tolist()).to_pandas()
    return X, y, cat_features, X_test


def load_train(variant: str):
    X, y, cat_features, _ = load_data(variant)
    return X, y, cat_features


@app.function(image=image, gpu="L4", volumes={VOLUME_PATH: volume}, timeout=3 * 3600)
def train_fold(variant: str, fold: int):
    from catboost import CatBoostClassifier, Pool
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    X, y, cat_features, X_test = load_data(variant)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    train_idx, valid_idx = list(skf.split(X, y))[fold]

    dtrain = Pool(X.iloc[train_idx], y[train_idx], cat_features=cat_features)
    dvalid = Pool(X.iloc[valid_idx], y[valid_idx], cat_features=cat_features)
    model = CatBoostClassifier(**PARAMS)
    model.fit(dtrain, eval_set=dvalid, use_best_model=True)

    preds = model.predict_proba(dvalid)[:, 1]
    test_preds = model.predict_proba(Pool(X_test, cat_features=cat_features))[:, 1]
    auc = roc_auc_score(y[valid_idx], preds)
    print(f"{variant} fold {fold}: auc={auc:.6f} best_iteration={model.get_best_iteration()}")
    return {
        "variant": variant,
        "fold": fold,
        "auc": auc,
        "best_iteration": model.get_best_iteration(),
        "valid_idx": valid_idx,
        "preds": preds,
        "test_preds": test_preds,
        "importance": dict(zip(X.columns, model.get_feature_importance().tolist())),
    }


@app.function(image=image, volumes={VOLUME_PATH: volume}, timeout=6 * 3600)
def cross_validate(variants: list[str]):
    import os

    import numpy as np
    import polars as pl
    from sklearn.metrics import roc_auc_score

    y = pl.read_csv(f"{DATA_DIR}/train.csv", columns=[TARGET])[TARGET].cast(pl.Int8).to_numpy()
    results = list(train_fold.starmap([(v, f) for v in variants for f in range(N_FOLDS)]))

    os.makedirs(OOF_DIR, exist_ok=True)
    os.makedirs(TEST_PRED_DIR, exist_ok=True)
    summary = []
    for v in variants:
        rs = sorted((r for r in results if r["variant"] == v), key=lambda r: r["fold"])
        oof = np.zeros(len(y))
        importance = {}
        for r in rs:
            oof[r["valid_idx"]] = r["preds"]
            for k, val in r["importance"].items():
                importance[k] = importance.get(k, 0.0) + val / N_FOLDS
        # Saved for ensembling; row order matches train.csv.
        np.save(f"{OOF_DIR}/catboost_{v}.npy", oof)
        # Mean of the fold models' test predictions; row order matches test.csv.
        np.save(f"{TEST_PRED_DIR}/catboost_{v}.npy", np.mean([r["test_preds"] for r in rs], axis=0))
        summary.append({
            "variant": v,
            "folds": [{"fold": r["fold"], "auc": float(r["auc"]), "best_iteration": int(r["best_iteration"])} for r in rs],
            "oof_auc": float(roc_auc_score(y, oof)),
            "importance": importance,
        })
    volume.commit()
    return summary


@app.local_entrypoint()
def main(variants: str = ",".join(VARIANTS)):
    import statistics

    for s in cross_validate.remote(variants.split(",")):
        print(f"\n== {s['variant']} ==")
        for f in s["folds"]:
            print(f"fold {f['fold']}: auc={f['auc']:.6f} best_iteration={f['best_iteration']}")
        aucs = [f["auc"] for f in s["folds"]]
        print(f"mean fold auc: {statistics.mean(aucs):.6f} ± {statistics.pstdev(aucs):.6f}")
        print(f"OOF auc: {s['oof_auc']:.6f}")
        print("top importance:")
        for k, v in sorted(s["importance"].items(), key=lambda kv: -kv[1])[:12]:
            print(f"  {k:40s} {v:8.2f}")
