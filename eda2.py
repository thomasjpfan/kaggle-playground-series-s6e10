import modal

app = modal.App("kaggle-playground-eda2")

volume = modal.Volume.from_name("kaggle")
VOLUME_PATH = "/data"
DATA_DIR = f"{VOLUME_PATH}/playground-series-s6e10"

image = modal.Image.debian_slim().uv_pip_install("polars", "lightgbm", "scikit-learn")

TARGET = "satisfaction"
CATEGORICAL = ["Gender", "Customer Type", "Type of Travel", "Class"]
NUMERIC = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
N_FOLDS = 5
SEED = 42  # Same folds as eval.py.


def load():
    import polars as pl

    train = pl.read_csv(f"{DATA_DIR}/train.csv").with_columns(pl.col(TARGET).cast(pl.Int8))
    test = pl.read_csv(f"{DATA_DIR}/test.csv")
    ratings = [c for c in train.columns if c not in ("id", TARGET, *CATEGORICAL, *NUMERIC)]
    return train, test, ratings


def add_features(df, ratings, kind):
    """Feature sets compared by `fold_experiment`."""
    import polars as pl

    df = df.with_columns(pl.col(c).cast(pl.Categorical).to_physical().cast(pl.Int32) for c in CATEGORICAL)
    if kind == "base":
        return df
    if kind.startswith("count"):
        cols = NUMERIC if kind == "count_all" else NUMERIC[:2]
        return df.with_columns(pl.len().over(c).alias(f"{c}_count") for c in cols)
    if kind.startswith("te"):
        # Counts here; the in-fold target encoding is added in `fold_experiment`.
        return df.with_columns(pl.len().over(c).alias(f"{c}_count") for c in NUMERIC[:2])
    df = df.with_columns(
        pl.sum_horizontal([(pl.col(c) == 0).cast(pl.Int8) for c in ratings]).alias("n_zeros"),
        pl.sum_horizontal([(pl.col(c) == 5).cast(pl.Int8) for c in ratings]).alias("n_fives"),
        pl.sum_horizontal([(pl.col(c) >= 4).cast(pl.Int8) for c in ratings]).alias("n_high"),
        pl.mean_horizontal(ratings).alias("rating_mean"),
        pl.min_horizontal(ratings).alias("rating_min"),
        (pl.col("Type of Travel") * 3 + pl.col("Class")).alias("travel_class"),
        (pl.col("Arrival Delay in Minutes") - pl.col("Departure Delay in Minutes")).alias("delay_diff"),
    )
    if kind == "fe_count":
        for c in NUMERIC[:2]:
            df = df.with_columns(pl.len().over(c).alias(f"{c}_count"))
    return df


TE_COLS = {
    "te_fd": ["Flight Distance"],
    "te_num": NUMERIC,
    "te_num_pairs": NUMERIC + [("Flight Distance", "Class"), ("Flight Distance", "Type of Travel"), ("Age", "Customer Type")],
}


def add_target_encoding(df, feats, tr, va, X, y, cols, alpha=20):
    """Smoothed target encoding: inner 5-fold OOF on the training part, full training part for validation."""
    import numpy as np
    import polars as pl
    from sklearn.model_selection import KFold

    prior = y[tr].mean()
    new = []
    for col in cols:
        keys = list(col) if isinstance(col, tuple) else [col]
        key = df.select(pl.concat_str([pl.col(k).cast(pl.Utf8) for k in keys], separator="|")).to_series().to_numpy()

        def encode(fit_idx, apply_idx):
            f = pl.DataFrame({"k": key[fit_idx], "y": y[fit_idx]}).group_by("k").agg(pl.col("y").sum().alias("s"), pl.len().alias("n"))
            a = pl.DataFrame({"k": key[apply_idx]}).join(f, on="k", how="left").fill_null(0)
            return ((a["s"] + alpha * prior) / (a["n"] + alpha)).to_numpy()

        enc = np.empty(len(y), dtype="float32")
        for itr, iva in KFold(5, shuffle=True, random_state=SEED).split(tr):
            enc[tr[iva]] = encode(tr[itr], tr[iva])
        enc[va] = encode(tr, va)
        new.append(enc)
        feats = feats + ["te_" + "_x_".join(keys)]
    return np.column_stack([X] + new), feats


@app.function(image=image, volumes={VOLUME_PATH: volume}, memory=8192, timeout=3600)
def data_checks():
    import polars as pl

    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_hide_dataframe_shape(True)

    train, test, ratings = load()
    features = [c for c in train.columns if c not in ("id", TARGET)]

    print("## exact feature duplicates in train (label noise ceiling)")
    g = train.group_by(features).agg(pl.len().alias("n"), pl.col(TARGET).mean().alias("rate"))
    dup = g.filter(pl.col("n") > 1)
    print("groups with >1 row:", dup.height, "rows in them:", dup["n"].sum())
    print("conflicting groups:", dup.filter((pl.col("rate") > 0) & (pl.col("rate") < 1)).height)

    print("\n## test rows whose exact features appear in train")
    m = test.join(g, on=features, how="inner")
    print("matched test rows:", m.height, "of", test.height)

    print("\n## duplicates on ratings + categoricals only (ignores numeric)")
    keys = ratings + CATEGORICAL
    g2 = train.group_by(keys).agg(pl.len().alias("n"), pl.col(TARGET).mean().alias("rate"))
    print("unique rating/categorical profiles:", g2.height)
    big = g2.filter(pl.col("n") >= 20)
    print("profiles with >=20 rows:", big.height, "covering", big["n"].sum(), "rows")
    # Within-profile label purity: how deterministic is the target given ratings+cats?
    purity = big.select(
        ((pl.col("rate") - 0.5).abs() * 2 * pl.col("n")).sum() / pl.col("n").sum()
    ).item()
    print("weighted purity |2*rate-1| in big profiles:", round(purity, 4))

    print("\n## cardinality and value-frequency signal")
    for c in NUMERIC + ratings:
        print(f"{c:40s} n_unique train={train[c].n_unique():6d} test={test[c].n_unique():6d}")
    for c in ["Age", "Flight Distance"]:
        cnt = train.group_by(c).agg(pl.len().alias("cnt"))
        t = train.join(cnt, on=c).with_columns(pl.col("cnt").qcut(5, allow_duplicates=True).alias("bin"))
        print(f"\n### {c}: target rate by value-frequency quintile")
        print(t.group_by("bin").agg(pl.len().alias("n"), pl.col(TARGET).mean().alias("rate")).sort("bin"))
        only_test = test.join(cnt, on=c, how="anti").height
        print("test rows with unseen value:", only_test)

    print("\n## Flight Distance: per-value target rate vs neighbourhood (is exact value informative?)")
    fd = (
        train.group_by("Flight Distance")
        .agg(pl.len().alias("n"), pl.col(TARGET).mean().alias("rate"))
        .filter(pl.col("n") >= 30)
        .sort("Flight Distance")
        .with_columns(pl.col("rate").rolling_mean(21, center=True).alias("smooth"))
        .drop_nulls()
    )
    resid = fd.select(((pl.col("rate") - pl.col("smooth")) ** 2 * pl.col("n")).sum() / pl.col("n").sum()).item()
    binom = fd.select((pl.col("smooth") * (1 - pl.col("smooth"))).mean()).item()
    print("values with n>=30:", fd.height)
    print("weighted MSE of exact-value rate vs smoothed:", round(resid, 5), "| expected binomial noise per row ~", round(binom, 4))
    print("ratio (>>1/mean_n means exact value carries extra signal):", round(resid / binom, 5), "1/mean_n:", round(1 / fd["n"].mean(), 5))

    print("\n## Age x Customer Type")
    print(
        train.with_columns(pl.col("Age").cut([18, 25, 40, 60]).alias("age_bin"))
        .group_by("Customer Type", "age_bin")
        .agg(pl.len().alias("n"), pl.col(TARGET).mean().alias("rate"))
        .sort("Customer Type", "age_bin")
    )

    print("\n## Personal travel segment (9.7% satisfied): what lifts it?")
    p = train.filter(pl.col("Type of Travel") == "Personal Travel")
    rows = []
    for c in ratings:
        rows.append({"feature": c, "pearson_in_personal": p.select(pl.corr(c, TARGET)).item(),
                     "pearson_all": train.select(pl.corr(c, TARGET)).item()})
    print(pl.DataFrame(rows).sort("pearson_in_personal", descending=True))
    print(p.group_by("Inflight wifi service").agg(pl.len().alias("n"), pl.col(TARGET).mean().alias("rate")).sort("Inflight wifi service"))


@app.function(image=image, volumes={VOLUME_PATH: volume}, cpu=8, memory=8192, timeout=3600)
def fold_experiment(kind: str, fold: int):
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    train, _, ratings = load()
    df = add_features(train, ratings, kind)
    feats = [c for c in df.columns if c not in ("id", TARGET)]
    X = df.select(feats).to_numpy().astype("float32")
    y = df[TARGET].to_numpy()
    tr, va = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y))[fold]

    if kind.startswith("te"):
        X, feats = add_target_encoding(df, feats, tr, va, X, y, cols=TE_COLS[kind])
    cats = CATEGORICAL + (["travel_class"] if "travel_class" in feats else [])
    if kind == "count_numcat":
        cats = cats + ["Age"]
    if kind == "fe_catratings":
        cats = cats + ratings
    params = {
        "objective": "binary", "metric": "auc", "learning_rate": 0.05, "num_leaves": 63,
        "min_child_samples": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
        "seed": SEED, "verbosity": -1, "max_cat_to_onehot": 8,
    }
    dtr = lgb.Dataset(X[tr], y[tr], feature_name=feats, categorical_feature=[feats.index(c) for c in cats])
    dva = lgb.Dataset(X[va], y[va], reference=dtr)
    model = lgb.train(params, dtr, 5000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    preds = model.predict(X[va], num_iteration=model.best_iteration)
    return {"kind": kind, "fold": fold, "auc": roc_auc_score(y[va], preds), "iters": model.best_iteration,
            "valid_idx": va, "preds": preds}


@app.function(image=image, volumes={VOLUME_PATH: volume}, memory=8192, timeout=3 * 3600)
def experiments(kinds: list[str]):
    import numpy as np
    import polars as pl
    from sklearn.metrics import roc_auc_score

    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_hide_dataframe_shape(True)

    results = list(fold_experiment.starmap([(k, f) for k in kinds for f in range(N_FOLDS)]))
    train, _, ratings = load()
    y = train[TARGET].to_numpy()

    oof = {}
    print("## feature-set comparison (same folds as eval.py)")
    for k in kinds:
        rs = sorted((r for r in results if r["kind"] == k), key=lambda r: r["fold"])
        o = np.zeros(len(y))
        for r in rs:
            o[r["valid_idx"]] = r["preds"]
        oof[k] = o
        aucs = [r["auc"] for r in rs]
        print(f"{k:15s} OOF auc={roc_auc_score(y, o):.6f} folds={np.round(aucs, 6).tolist()} iters={[r['iters'] for r in rs]}")
    if len(kinds) > 1:
        rank = sum(pl.Series(oof[k]).rank().to_numpy() for k in kinds)
        print(f"{'rank-avg all':15s} OOF auc={roc_auc_score(y, rank):.6f}")

    print("\n## error analysis on base OOF")
    o = oof[kinds[0]]
    df = train.with_columns(
        pl.Series("pred", o),
        pl.sum_horizontal([(pl.col(c) == 0).cast(pl.Int8) for c in ratings]).alias("n_zeros"),
    ).with_columns((-(pl.col(TARGET) * pl.col("pred").log() + (1 - pl.col(TARGET)) * (1 - pl.col("pred")).log())).alias("ll"))

    def seg(col):
        out = []
        for (v,), g in df.group_by(col):
            yy, pp = g[TARGET].to_numpy(), g["pred"].to_numpy()
            auc = roc_auc_score(yy, pp) if 0 < yy.mean() < 1 else None
            out.append({col: v, "n": g.height, "rate": yy.mean(), "auc": auc, "mean_ll": g["ll"].mean(),
                        "share_of_total_ll": g["ll"].sum() / df["ll"].sum()})
        print(pl.DataFrame(out).sort(col))

    for c in ["Type of Travel", "Class", "Customer Type", "n_zeros"]:
        print(f"\n### by {c}")
        seg(c)

    print("\n### calibration (pred decile)")
    print(df.with_columns(pl.col("pred").qcut(10).alias("dec")).group_by("dec")
          .agg(pl.len().alias("n"), pl.col("pred").mean().alias("pred"), pl.col(TARGET).mean().alias("rate")).sort("dec"))

    print("\n### confidently wrong rows (|pred - y| > 0.9)")
    bad = df.filter((pl.col("pred") - pl.col(TARGET)).abs() > 0.9)
    print("count:", bad.height, f"({bad.height / df.height:.4%})")
    print(bad.select(TARGET, "pred", "Type of Travel", "Class", "Online boarding", "Inflight wifi service", "n_zeros").head(15))


@app.local_entrypoint()
def main(checks: bool = True, kinds: str = "base,fe,fe_catratings,fe_count"):
    if checks:
        data_checks.remote()
    if kinds:
        experiments.remote(kinds.split(","))
