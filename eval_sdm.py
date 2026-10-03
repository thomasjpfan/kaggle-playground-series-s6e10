import modal

app = modal.App("kaggle-playground-eval-sdm")

volume = modal.Volume.from_name("kaggle")
VOLUME_PATH = "/data"
DATA_DIR = f"{VOLUME_PATH}/playground-series-s6e10"

hf_cache = modal.Volume.from_name("hf-cache")
HF_HOME = "/hf-cache"

SDM_COMMIT = "b982fbf4188ea8db17249fe81338723106100b63"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        f"structured-data-models @ git+https://github.com/NVIDIA/structured-data-models.git@{SDM_COMMIT}",
        "cudf-cu13>=26.8",
        "polars",
        "scikit-learn",
        extra_index_url="https://pypi.nvidia.com",
    )
    .env({"HF_HOME": HF_HOME})
)

TARGET = "satisfaction"
N_FOLDS = 5
SEED = 42  # Same folds as eval.py.

MODELS = ["kumo-tabular", "tabiclv2"]
NUM_ESTIMATORS = 8
# Rows of the training fold given to each estimator as in-context examples.
# Every estimator draws its own random subset; a size larger than the fold
# uses the whole fold.
CONTEXT_SIZE = 50_000
QUERY_BATCH_SIZE = 10_000


def load_train():
    import polars as pl

    train = pl.read_csv(f"{DATA_DIR}/train.csv").drop("id")
    return train.with_columns(pl.col(TARGET).cast(pl.Int8))


@app.function(
    image=image,
    gpu="A100",
    volumes={VOLUME_PATH: volume, HF_HOME: hf_cache},
    memory=32768,
    timeout=3 * 3600,
)
def train_fold(model_name: str, fold: int, context_size: int, num_estimators: int):
    import time

    import sdm
    import torch
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    config = {"model": model_name, "fold": fold, "context_size": context_size, "num_estimators": num_estimators}

    train = load_train()
    y = train[TARGET].to_numpy()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    train_idx, valid_idx = list(skf.split(y, y))[fold]

    device = torch.device("cuda")
    arrow = train.to_arrow()
    stypes = sdm.infer_stypes(arrow, overrides={TARGET: "categorical"})
    table = sdm.TableTensor.from_arrow(table=arrow, stypes=stypes, device=device)

    if model_name == "kumo-tabular":
        model = sdm.models.KumoTabular(task="classification", size="large", device=device)
    else:
        model = sdm.models.TabICLv2(task="classification", device=device)
    hf_cache.commit()

    generator = torch.Generator(device).manual_seed(SEED + fold)
    train_idx = torch.from_numpy(train_idx).to(device)
    # [num_estimators, context_size]: a different random context per estimator.
    context_idx = torch.stack(
        [
            train_idx[torch.randperm(len(train_idx), generator=generator, device=device)[:context_size]]
            for _ in range(num_estimators)
        ]
    )
    context = table[context_idx.flatten()].unflatten(0, tuple(context_idx.shape))

    start = time.perf_counter()
    try:
        with torch.amp.autocast("cuda", torch.float16):
            model.fit(
                x=context.drop_columns(TARGET),
                y=context[TARGET],
                generator=generator,
            )
            preds = []
            for batch_start in range(0, len(valid_idx), QUERY_BATCH_SIZE):
                batch_idx = torch.from_numpy(valid_idx[batch_start : batch_start + QUERY_BATCH_SIZE]).to(device)
                query = table[batch_idx].drop_columns(TARGET)
                out = model.predict(query.expand(num_estimators, *query.size()))
                column = out.columns[sdm.Stype.numerical].index("1")
                preds.append(out.numerical[..., column].float().cpu())
    except torch.OutOfMemoryError:
        print(f"{config}: out of GPU memory")
        return {**config, "auc": None, "seconds": None, "preds": None, "y": None}
    finally:
        model.clear()
    elapsed = time.perf_counter() - start
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    preds = torch.cat(preds).numpy()
    auc = roc_auc_score(y[valid_idx], preds)
    print(f"{config}: auc={auc:.6f} time={elapsed:.1f}s peak_mem={peak_gb:.1f}GB")
    return {
        **config,
        "auc": float(auc),
        "seconds": elapsed,
        "peak_gb": peak_gb,
        "preds": preds,
        "y": y[valid_idx],
    }


@app.function(image=image, timeout=6 * 3600)
def cross_validate(model_names: list[str], folds: list[int], context_sizes: list[int], num_estimators: list[int]):
    import numpy as np
    from sklearn.metrics import roc_auc_score

    configs = [(m, c, e) for m in model_names for c in context_sizes for e in num_estimators]
    args = [(m, fold, c, e) for m, c, e in configs for fold in folds]
    results = list(train_fold.starmap(args, return_exceptions=True))

    summary = []
    for m, c, e in configs:
        rs = sorted(
            (
                r
                for r in results
                if isinstance(r, dict) and (r["model"], r["context_size"], r["num_estimators"]) == (m, c, e)
            ),
            key=lambda r: r["fold"],
        )
        ok = [r for r in rs if r["auc"] is not None]
        entry = {
            "model": m,
            "context_size": c,
            "num_estimators": e,
            "folds": [
                {"fold": r["fold"], "auc": r["auc"], "seconds": r["seconds"], "peak_gb": r.get("peak_gb")}
                for r in rs
            ],
            "failed": len(folds) - len(ok),
            "oof_auc": None,
        }
        if ok and len(ok) == len(folds):
            entry["oof_auc"] = float(
                roc_auc_score(np.concatenate([r["y"] for r in ok]), np.concatenate([r["preds"] for r in ok]))
            )
        summary.append(entry)
    for r in results:
        if not isinstance(r, dict):
            print("error:", repr(r))
    return summary


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",")]


@app.local_entrypoint()
def main(
    models: str = ",".join(MODELS),
    folds: str = ",".join(str(f) for f in range(N_FOLDS)),
    context_sizes: str = str(CONTEXT_SIZE),
    num_estimators: str = str(NUM_ESTIMATORS),
):
    import statistics

    summary = cross_validate.remote(models.split(","), _ints(folds), _ints(context_sizes), _ints(num_estimators))

    for s in summary:
        print(f"\n== {s['model']} (context_size={s['context_size']}, num_estimators={s['num_estimators']}) ==")
        for f in s["folds"]:
            if f["auc"] is None:
                print(f"fold {f['fold']}: out of GPU memory")
            else:
                print(f"fold {f['fold']}: auc={f['auc']:.6f} time={f['seconds']:.1f}s peak_mem={f['peak_gb']:.1f}GB")
        aucs = [f["auc"] for f in s["folds"] if f["auc"] is not None]
        if s["failed"]:
            print(f"failed folds: {s['failed']}")
        if aucs:
            print(f"mean fold auc: {statistics.mean(aucs):.6f} ± {statistics.pstdev(aucs):.6f}")
        if s["oof_auc"] is not None:
            print(f"OOF auc: {s['oof_auc']:.6f}")
