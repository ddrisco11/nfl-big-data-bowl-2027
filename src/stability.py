"""Audit score sensitivity to model architecture and observation window."""

import os
from shared import *
from scipy.stats import spearmanr

D, S, C, H, V, Y = load()
r = pd.read_csv(O / "route-unpredictability.csv.gz")
a = pd.read_csv(O / "player-unpredictability.csv")
qualified = set(a.loc[~a.provisional, "nfl_id"])
lookup = {int(x): i for i, x in enumerate(D.row_id)}
adjust, _ = pickle.load(open(W / "score-adjustment.pkl", "rb"))
r["duration_squared"] = r.duration_seconds**2
r["sideline_distance"] = np.minimum(r.snap_y, 160 / 3 - r.snap_y)
cat = [
    "route_ran",
    "nfl_position",
    "offense_formation",
    "receiver_alignment",
    "lined_up_position",
]
num = [
    "down",
    "yards_to_go",
    "duration_seconds",
    "duration_squared",
    "sideline_distance",
]
for c in cat:
    r[c] = r[c].fillna("Missing")
expected = adjust.predict(r[cat + num])
base = r.groupby("nfl_id").excess_unpredictability.mean()
out = []
for model in [
    "Motion families",
    "Sequence GRU",
    "Joint route and path",
    "Temporal convolution",
]:
    p = np.load(W / (model.replace(" ", "_") + ".npy"))
    scores = []
    for row in r.itertuples():
        i = lookup[row.row_id]
        n = int(row.native_frames)
        scores.append(
            100 * np.trapezoid(1 - p[i, :n, Y[i]], TIMES[:n]) / row.duration_seconds
        )
    z = r[["nfl_id"]].copy()
    z["score"] = np.array(scores) - expected
    other = z.groupby("nfl_id").score.mean()
    ids = sorted(qualified)
    out.append(
        {
            "model": model,
            "player_rank_correlation": float(
                spearmanr(base.loc[ids], other.loc[ids]).statistic
            ),
            "players": len(ids),
        }
    )
# Same eligible routes for both full-window and fixed-window comparisons.
f = r.dropna(subset=["fixed_two_second_unpredictability"])
g = f.groupby("nfl_id").agg(
    fixed=("fixed_two_second_unpredictability", "mean"),
    variable=("unpredictability", "mean"),
    n=("row_id", "size"),
)
g = g[g.n >= 50]
meta = {
    "model_sensitivity": out,
    "duration_sensitivity": {
        "n_players": len(g),
        "raw_score_rank_correlation_same_routes": float(
            spearmanr(g.fixed, g.variable).statistic
        ),
    },
    "notes": [
        "Model sensitivity subtracts the same validation-fitted context expectation from each candidate; high rank agreement is not proof of defender deception.",
        "Window sensitivity compares unadjusted scores on the same routes with at least two seconds observed; this changes the eligible population.",
    ],
}
(O / "stability.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
