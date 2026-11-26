"""Estimate calibrated future-path entropy from native-rate receiver prefixes."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
import argparse, json, pathlib, pickle
import numpy as np, pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

p = argparse.ArgumentParser()
p.add_argument("--work", default=".work/models")
p.add_argument("--output", default=".work/rei")
args = p.parse_args()
W = pathlib.Path(args.work)
O = pathlib.Path(args.output)
O.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(2027)
d = pd.read_parquet(W / "routes.parquet")
a = np.load(W / "sequences.npz")
s = a["motion"]
live = a["live"]
times = np.arange(31) * 0.1
N = len(d)
K = 8
prefix = live[:, 10:41] & np.logical_and.accumulate(
    np.isfinite(s[:, 10:41]).all(2), axis=1
)
y = np.full((N, 31, 4), np.nan, np.float32)
valid = np.zeros((N, 31), bool)
for t in range(31):
    j = 10 + t
    y[:, t] = np.concatenate(
        [s[:, j + 5, :2] - s[:, j, :2], s[:, j + 10, :2] - s[:, j, :2]], axis=1
    )
    valid[:, t] = (
        prefix[:, t]
        & live[:, j : j + 11].all(1)
        & np.isfinite(s[:, j : j + 11]).all(axis=(1, 2))
    )
train = (d.split == "train").to_numpy()
val = (d.split == "validation").to_numpy()
select = val & ((d.game_id // 5) % 2 == 0).to_numpy()
cal = val & ~select
test = (d.split == "test").to_numpy()
fitmask = valid & train[:, None]
fitmask[:, np.arange(31) % 4 != 0] = False
ii, tt = np.where(fitmask)
# Modes are learned only from training futures; future positions are targets, never features.
km = MiniBatchKMeans(n_clusters=K, batch_size=4096, n_init=10, random_state=2027).fit(
    y[ii, tt]
)
centers = km.cluster_centers_
labels = np.zeros((N, 31), int)
vi, vt = np.where(valid)
labels[vi, vt] = km.predict(y[vi, vt])
print(
    "Training frames",
    len(ii),
    "eligible test frames",
    int(valid[test].sum()),
    flush=True,
)
from rei_core import prefix_features, shannon, integrate_entropy

X, B = prefix_features(d, s)


def temper(q, t):
    z = np.log(np.maximum(q, 1e-9)) / t
    z -= z.max(-1, keepdims=True)
    q = np.exp(z)
    return q / q.sum(-1, keepdims=True)


def evaluate(q, mask):
    i, t = np.where(mask)
    pr = q[i, t]
    yy = labels[i, t]
    conf = pr.max(1)
    correct = pr.argmax(1) == yy
    ece = 0
    for b in range(10):
        ix = (conf >= b / 10) & (conf < (b + 1) / 10 if b < 9 else conf <= 1)
        if ix.any():
            ece += ix.mean() * abs(conf[ix].mean() - correct[ix].mean())
    dist = cdist(y[i, t], centers)
    pair = cdist(centers, centers)
    es = np.sum(pr * dist, axis=1) - 0.5 * np.einsum("ni,ij,nj->n", pr, pair, pr)
    order = np.argsort(-pr, axis=1)
    sortedp = np.take_along_axis(pr, order, axis=1)
    included = np.cumsum(sortedp, axis=1) - sortedp < 0.9
    cover = np.any((order == yy[:, None]) & included, axis=1)
    return {
        "frames": len(i),
        "log_loss": float(log_loss(yy, pr, labels=range(K))),
        "accuracy": float(correct.mean()),
        "brier": float(np.mean(np.sum((pr - np.eye(K)[yy]) ** 2, axis=1))),
        "ece": float(ece),
        "energy_score_four_dimensional_yards": float(es.mean()),
        "prediction_set_90_coverage": float(cover.mean()),
        "prediction_set_90_mean_modes": float(included.sum(1).mean()),
    }


results = {}
probs = {}
models = {}
for name, xx in [("Context and speed baseline", B), ("Movement prefix", X)]:
    model = HistGradientBoostingClassifier(
        max_iter=100,
        max_leaf_nodes=15,
        learning_rate=0.08,
        l2_regularization=20,
        early_stopping=False,
        random_state=2027,
    )
    model.fit(xx[ii, tt], labels[ii, tt])
    q = (
        model.predict_proba(xx.reshape(-1, xx.shape[-1]))
        .reshape(N, 31, K)
        .astype(np.float32)
    )
    ci, ct = np.where(valid & cal[:, None])
    temp = min(
        [0.7, 0.85, 1, 1.15, 1.3, 1.5, 1.8],
        key=lambda z: log_loss(labels[ci, ct], temper(q[ci, ct], z), labels=range(K)),
    )
    q = temper(q, temp).astype(np.float32)
    probs[name] = q
    models[name] = (model, temp)
    results[name] = {
        "temperature": temp,
        "selection": evaluate(q, valid & select[:, None]),
        "test": evaluate(q, valid & test[:, None]),
    }
    print(name, json.dumps(results[name]), flush=True)
q = probs["Movement prefix"]
entropy = shannon(q)
assert np.allclose(q.sum(2), 1, atol=1e-5)
assert np.min(entropy) >= -1e-6 and np.max(entropy) <= np.log(K) + 1e-6
# Eligibility requires a full observed future second; this is explicit duration selection.
rows = []
for i in np.flatnonzero(~train):
    ix = np.flatnonzero(valid[i])
    n = next((j for j in range(31) if not valid[i, j]), 31)
    if n < 6:
        continue
    T = times[n - 1]
    score = integrate_entropy(q[i, :n], times[:n])
    row = d.iloc[i][
        [
            "row_id",
            "game_id",
            "play_id",
            "nfl_id",
            "display_name",
            "route_ran",
            "nfl_position",
            "split",
            "separation_at_pass_forward",
            "cushion",
            "team_coverage_man_zone",
        ]
    ].to_dict()
    row.update(
        rei=float(score),
        duration_seconds=float(T),
        mean_speed=float(s[i, 10 : 10 + n, 2].mean()),
        mean_effective_modes=float(np.exp(entropy[i, :n]).mean()),
        fixed_one_second_rei=float(
            100 * np.trapezoid(entropy[i, :11], times[:11]) / np.log(K)
        )
        if n >= 11
        else np.nan,
    )
    rows.append(row)
r = pd.DataFrame(rows)
# Descriptive context-adjusted entropy: fitted on validation routes, independent of Combine.
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

cat = ["route_ran", "nfl_position", "team_coverage_man_zone"]
num = ["duration_seconds", "mean_speed", "cushion"]
for col in cat:
    r[col] = r[col].fillna("Missing")
adjust = make_pipeline(
    ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
            ("num", SimpleImputer(), num),
        ]
    ),
    Ridge(alpha=20),
)
vr = r.split == "validation"
adjust.fit(r.loc[vr, cat + num], r.loc[vr, "rei"])
r["excess_rei"] = r.rei - adjust.predict(r[cat + num])
r = r[r.split == "test"].copy()
r.to_csv(O / "route-entropy.csv.gz", index=False)
players = []
for nfl, g in r.groupby("nfl_id"):
    v = (
        g.groupby("game_id")
        .agg(total=("excess_rei", "sum"), n=("row_id", "size"))
        .to_numpy()
    )
    draw = rng.integers(0, len(v), (500, len(v)))
    sums = v[draw].sum(1)
    ci = np.quantile(sums[:, 0] / sums[:, 1], [0.025, 0.975])
    players.append(
        {
            "nfl_id": int(nfl),
            "player": g.display_name.iloc[0],
            "routes": len(g),
            "games": len(v),
            "rei": float(g.rei.mean()),
            "excess_rei": float(g.excess_rei.mean()),
            "ci_low": float(ci[0]),
            "ci_high": float(ci[1]),
            "provisional": len(g) < 50 or len(v) < 5,
        }
    )
pd.DataFrame(players).to_csv(O / "player-entropy.csv", index=False)
r["half"] = r.game_id % 2
h = (
    r.groupby(["nfl_id", "half"])
    .agg(value=("excess_rei", "mean"), n=("row_id", "size"))
    .unstack()
    .dropna()
)
h = h[h["n"].min(axis=1) >= 25]
repeat = float(spearmanr(h["value"][0], h["value"][1]).statistic)
# Paired game bootstrap for probabilistic improvement, accounting for clustered frames.
i, t = np.where(valid & test[:, None])
delta = -np.log(q[i, t, labels[i, t]]) + np.log(
    probs["Context and speed baseline"][i, t, labels[i, t]]
)
g = (
    pd.DataFrame({"game": d.game_id.to_numpy()[i], "delta": delta, "n": 1})
    .groupby("game")
    .sum()
    .to_numpy()
)
draw = rng.integers(0, len(g), (800, len(g)))
sums = g[draw].sum(1)
change = {
    "movement_minus_baseline_log_loss": float(delta.mean()),
    "game_bootstrap_ci95": np.quantile(
        sums[:, 0] / sums[:, 1], [0.025, 0.975]
    ).tolist(),
}
np.savez_compressed(
    O / "entropy-probabilities.npz",
    row_ids=d.row_id.to_numpy(),
    probabilities=q,
    entropy=entropy,
    valid=valid,
    centers=centers,
)
pickle.dump(
    {"models": models, "modes": km, "adjustment": adjust},
    open(O / "entropy-model.pkl", "wb"),
)
summary = {
    "name": "Route Entropy Index",
    "version": "prototype 1",
    "modes": K,
    "horizon_seconds": 1,
    "path_sample_offsets_seconds": [0.5, 1.0],
    "native_step_seconds": 0.1,
    "training_frames": len(ii),
    "metrics": results,
    "paired_improvement": change,
    "scored_routes": len(r),
    "players": len(players),
    "qualified_players": sum(not x["provisional"] for x in players),
    "mean_rei": float(r.rei.mean()),
    "repeatability": {"players": len(h), "spearman": repeat},
    "mode_centers_relative_yards": centers.tolist(),
    "checks": {
        "probability_sums": True,
        "entropy_bounds": True,
        "uniform_rei": float(
            -100 * np.sum(np.full(K, 1 / K) * np.log(np.full(K, 1 / K))) / np.log(K)
        ),
        "certain_rei": 0.0,
        "no_combine_inputs": True,
        "training_only_mode_dictionary": True,
    },
    "notes": [
        "Entropy is over eight learned future-path modes, not exact route labels or all physically possible paths. Modes summarize positions at 0.5 and 1 second ahead in normalized field coordinates.",
        "2025 is a development holdout; not a new confirmatory test.",
        "Forecast origins stop one full second before release/end or missing data, capped at 3 seconds. At least 0.5 seconds of contiguous origins is required. This favors longer-observed plays.",
        "Entropy measures model predictive ambiguity, including model uncertainty; it is not proof of deception or receiver quality.",
        "The same fixed mode dictionary must be used for player comparisons. Mode count and granularity affect the scale.",
        "Defender response is a separate validation axis; it is not added to Shannon entropy.",
    ],
}
(O / "entropy-summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
print("REI COMPLETE", summary["scored_routes"], summary["repeatability"], flush=True)
