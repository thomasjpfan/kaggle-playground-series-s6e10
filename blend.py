import modal

app = modal.App("kaggle-playground-blend")

volume = modal.Volume.from_name("kaggle")
VOLUME_PATH = "/data"
DATA_DIR = f"{VOLUME_PATH}/playground-series-s6e10"
OOF_DIR = f"{VOLUME_PATH}/oof-s6e10"

image = modal.Image.debian_slim().uv_pip_install("polars", "numpy", "scipy", "scikit-learn")

TARGET = "satisfaction"


@app.function(image=image, volumes={VOLUME_PATH: volume}, timeout=3600)
def blend(names: list[str] | None = None):
    import glob
    import os

    import numpy as np
    import polars as pl
    from scipy.stats import rankdata
    from sklearn.metrics import roc_auc_score

    y = pl.read_csv(f"{DATA_DIR}/train.csv", columns=[TARGET])[TARGET].cast(pl.Int8).to_numpy()
    paths = sorted(glob.glob(f"{OOF_DIR}/*.npy"))
    oofs = {os.path.basename(p)[:-4]: np.load(p) for p in paths}
    if names:
        oofs = {k: v for k, v in oofs.items() if k in names}
    # AUC only depends on order, so blend normalized ranks.
    ranks = {k: rankdata(v) / len(v) for k, v in oofs.items()}

    print("## single models")
    single = {k: roc_auc_score(y, v) for k, v in oofs.items()}
    for k, auc in sorted(single.items(), key=lambda kv: -kv[1]):
        print(f"{k:25s} {auc:.6f}")

    keys = list(ranks)
    print("\n## rank correlation")
    corr = np.corrcoef([ranks[k] for k in keys])
    for i, a in enumerate(keys):
        print(f"{a:25s} " + " ".join(f"{corr[i, j]:.4f}" for j in range(len(keys))))

    print("\n## equal-weight rank average of all")
    print(f"{roc_auc_score(y, sum(ranks.values())):.6f}")

    print("\n## best weight per pair (grid over w in [0, 1])")
    grid = np.linspace(0, 1, 21)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            scores = [roc_auc_score(y, w * ranks[a] + (1 - w) * ranks[b]) for w in grid]
            j = int(np.argmax(scores))
            print(f"{a} + {b}: w({a})={grid[j]:.2f} auc={scores[j]:.6f}")

    # Greedy forward selection with replacement (Caruana ensemble selection).
    print("\n## greedy ensemble selection")
    chosen, blend_sum, best = [], np.zeros(len(y)), 0.0
    for _ in range(20):
        cand = {k: roc_auc_score(y, (blend_sum + r) / (len(chosen) + 1)) for k, r in ranks.items()}
        k, auc = max(cand.items(), key=lambda kv: kv[1])
        if auc <= best + 1e-6:
            break
        chosen.append(k)
        blend_sum += ranks[k]
        best = auc
        print(f"+ {k:25s} auc={auc:.6f}")
    weights = {k: chosen.count(k) / len(chosen) for k in dict.fromkeys(chosen)}
    print("weights:", {k: round(v, 3) for k, v in weights.items()})
    # Note: the weights are fitted on the same OOF used to score them, so this is slightly optimistic.


@app.local_entrypoint()
def main(names: str = ""):
    blend.remote(names.split(",") if names else None)
