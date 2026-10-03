import os

import modal

app = modal.App("kaggle-playground")

volume = modal.Volume.from_name("kaggle")
VOLUME_PATH = "/data"
DATA_DIR = f"{VOLUME_PATH}/playground-series-s6e10"

hf_cache = modal.Volume.from_name("hf-cache")
HF_HOME = "/hf-cache"

image = (
    modal.Image.debian_slim()
    .uv_pip_install("polars")
    .env({"HF_HOME": HF_HOME})
)


@app.function(image=image, volumes={VOLUME_PATH: volume, HF_HOME: hf_cache})
def list_files():
    print(f"HF_HOME={os.environ['HF_HOME']}")
    for root, _, files in os.walk(VOLUME_PATH):
        for name in sorted(files):
            path = os.path.join(root, name)
            size = os.path.getsize(path)
            print(f"{os.path.relpath(path, VOLUME_PATH)}\t{size:,} bytes")


@app.function(image=image, volumes={VOLUME_PATH: volume})
def eda():
    import polars as pl

    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_hide_dataframe_shape(True)

    target = "satisfaction"
    train = pl.read_csv(f"{DATA_DIR}/train.csv").with_columns(pl.col(target).cast(pl.Int8))
    test = pl.read_csv(f"{DATA_DIR}/test.csv")

    categorical = ["Gender", "Customer Type", "Type of Travel", "Class"]
    ratings = [c for c, t in train.schema.items() if t == pl.Int64 and train[c].max() <= 5 and c != "id"]
    numeric = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]

    print("## shapes", train.shape, test.shape)
    print("## target rate", train[target].mean())

    for col in categorical:
        print(f"\n## {col}")
        print(
            train.group_by(col)
            .agg(pl.len().alias("n"), pl.col(target).mean().alias("rate"))
            .with_columns((pl.col("n") / pl.col("n").sum()).alias("share"))
            .sort(col)
        )

    print("\n## Travel x Class")
    print(
        train.group_by("Type of Travel", "Class")
        .agg(pl.len().alias("n"), pl.col(target).mean().alias("rate"))
        .sort("Type of Travel", "Class")
    )

    print("\n## rating value -> satisfaction rate (columns = rating value)")
    rows = []
    for col in ratings:
        g = train.group_by(col).agg(pl.col(target).mean()).sort(col)
        row = {"feature": col}
        for v, r in g.iter_rows():
            row[str(v)] = round(r, 3)
        cnt = train.group_by(col).len().sort(col)
        row["share_0"] = round(cnt.filter(pl.col(col) == 0)["len"].sum() / len(train), 4)
        rows.append(row)
    print(pl.DataFrame(rows))

    print("\n## correlation with target (pearson / spearman)")
    corr_rows = []
    for col in ratings + numeric:
        corr_rows.append({
            "feature": col,
            "pearson": train.select(pl.corr(col, target)).item(),
            "spearman": train.select(pl.corr(col, target, method="spearman")).item(),
        })
    print(pl.DataFrame(corr_rows).sort("pearson", descending=True))

    print("\n## numeric quantiles by target")
    for col in numeric:
        print(f"### {col}")
        print(
            train.group_by(target)
            .agg(
                pl.col(col).mean().alias("mean"),
                *[pl.col(col).quantile(q).alias(f"q{int(q*100)}") for q in (0.1, 0.5, 0.9, 0.99)],
                (pl.col(col) == 0).mean().alias("frac_zero"),
            )
            .sort(target)
        )

    print("\n## age bins")
    print(
        train.with_columns(pl.col("Age").cut([18, 25, 40, 60]).alias("age_bin"))
        .group_by("age_bin").agg(pl.len().alias("n"), pl.col(target).mean().alias("rate")).sort("age_bin")
    )

    print("\n## flight distance bins")
    print(
        train.with_columns(pl.col("Flight Distance").cut([500, 1000, 2000, 3000]).alias("dist_bin"))
        .group_by("dist_bin").agg(pl.len().alias("n"), pl.col(target).mean().alias("rate")).sort("dist_bin")
    )

    print("\n## delay relationship")
    d = train.drop_nulls("Arrival Delay in Minutes")
    print("corr dep/arr delay", d.select(pl.corr("Departure Delay in Minutes", "Arrival Delay in Minutes")).item())
    print("frac arr==dep", (d["Departure Delay in Minutes"] == d["Arrival Delay in Minutes"]).mean())
    print(
        train.with_columns(pl.col("Arrival Delay in Minutes").cut([0, 15, 60]).alias("arr_bin"))
        .group_by("arr_bin").agg(pl.len().alias("n"), pl.col(target).mean().alias("rate")).sort("arr_bin")
    )
    print("null arrival delay rate:", train.filter(pl.col("Arrival Delay in Minutes").is_null())[target].mean())

    print("\n## mean of ratings & count of 5s vs target")
    feat = train.with_columns(
        pl.mean_horizontal(ratings).alias("rating_mean"),
        pl.sum_horizontal([(pl.col(c) == 5).cast(pl.Int32) for c in ratings]).alias("n_fives"),
        pl.sum_horizontal([(pl.col(c) == 0).cast(pl.Int32) for c in ratings]).alias("n_zeros"),
    )
    for c in ["rating_mean", "n_fives", "n_zeros"]:
        print(c, "pearson", feat.select(pl.corr(c, target)).item())
    print(feat.group_by("n_zeros").agg(pl.len().alias("n"), pl.col(target).mean().alias("rate")).sort("n_zeros"))

    print("\n## inter-rating correlations > 0.5")
    pairs = []
    for i, a in enumerate(ratings):
        for b in ratings[i + 1:]:
            r = train.select(pl.corr(a, b)).item()
            if abs(r) > 0.5:
                pairs.append({"a": a, "b": b, "r": r})
    print(pl.DataFrame(pairs).sort("r", descending=True))

    print("\n## train vs test drift")
    drift = []
    for col in ratings + numeric:
        drift.append({
            "feature": col,
            "train_mean": train[col].mean(), "test_mean": test[col].mean(),
            "train_std": train[col].std(), "test_std": test[col].std(),
            "test_nulls": test[col].null_count(),
        })
    print(pl.DataFrame(drift))
    for col in categorical:
        tr = train[col].value_counts(normalize=True).sort(col)
        te = test[col].value_counts(normalize=True).sort(col)
        print(col, dict(zip(tr[col], tr["proportion"].round(4))), dict(zip(te[col], te["proportion"].round(4))))
    print("test ids", test["id"].min(), test["id"].max())


@app.local_entrypoint()
def main():
    eda.remote()
