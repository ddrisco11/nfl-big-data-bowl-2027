"""Relate future-path entropy to provided separation at release with clustered uncertainty."""

import argparse, json, pathlib
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

p = argparse.ArgumentParser()
p.add_argument("--output", default=".work/rei")
p.add_argument("--work", default=".work/models")
a = p.parse_args()
O = pathlib.Path(a.output)
rng = np.random.default_rng(2027)
r = pd.read_csv(O / "route-entropy.csv.gz")
d = pd.read_parquet(pathlib.Path(a.work) / "routes.parquet")
cols = [
    "row_id",
    "draft_year",
    "offense_formation",
    "receiver_alignment",
    "down",
    "yards_to_go",
    "snap_y",
]
r = r.merge(d[cols], on="row_id", validate="1:1")
total = len(r)
r = r.dropna(subset=["separation_at_pass_forward"]).copy().reset_index(drop=True)
assert (r.separation_at_pass_forward >= 0).all()
r["sideline_distance"] = np.minimum(r.snap_y, 160 / 3 - r.snap_y)
cat = [
    "route_ran",
    "nfl_position",
    "team_coverage_man_zone",
    "offense_formation",
    "receiver_alignment",
]
num = [
    "down",
    "yards_to_go",
    "sideline_distance",
    "duration_seconds",
    "mean_speed",
    "cushion",
]
for c in cat:
    r[c] = r[c].fillna("Missing")
# Cross-fit separation expectations by game. This is descriptive adjustment, not causal identification.
r["separation_residual"] = np.nan
for train, test in GroupKFold(5).split(r, groups=r.game_id):
    prep = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
            ("num", make_pipeline(SimpleImputer(), StandardScaler()), num),
        ]
    )
    model = make_pipeline(prep, Ridge(alpha=20))
    model.fit(r.iloc[train][cat + num], r.iloc[train].separation_at_pass_forward)
    r.loc[test, "separation_residual"] = r.iloc[
        test
    ].separation_at_pass_forward - model.predict(r.iloc[test][cat + num])
players = (
    r.groupby("nfl_id")
    .agg(
        routes=("row_id", "size"),
        games=("game_id", "nunique"),
        rei=("rei", "mean"),
        excess_rei=("excess_rei", "mean"),
        separation=("separation_at_pass_forward", "mean"),
        adjusted_separation=("separation_residual", "mean"),
    )
    .reset_index()
)
players = players[(players.routes >= 50) & (players.games >= 5)].copy()


def player_corr(x, y):
    rho = float(spearmanr(x, y).statistic)
    n = len(x)
    boot = []
    for _ in range(2000):
        ix = rng.integers(0, n, n)
        boot.append(spearmanr(x[ix], y[ix]).statistic)
    return {
        "rho": rho,
        "ci95": np.nanquantile(boot, [0.025, 0.975]).tolist(),
        "players": n,
    }


raw = player_corr(players.rei.to_numpy(), players.separation.to_numpy())
adj = player_corr(players.excess_rei.to_numpy(), players.adjusted_separation.to_numpy())
mixed = player_corr(players.excess_rei.to_numpy(), players.separation.to_numpy())
# Route-level Spearman, with game-cluster bootstrap of fixed empirical ranks.
x = rankdata(r.rei)
y = rankdata(r.separation_at_pass_forward)
g = (
    pd.DataFrame(
        {
            "game": r.game_id,
            "n": 1,
            "x": x,
            "y": y,
            "xx": x * x,
            "yy": y * y,
            "xy": x * y,
        }
    )
    .groupby("game")
    .sum()
    .to_numpy()
)
draw = rng.integers(0, len(g), (2000, len(g)))
sums = g[draw].sum(1)
n, sx, sy, sxx, syy, sxy = sums.T
boot = (sxy - sx * sy / n) / np.sqrt((sxx - sx * sx / n) * (syy - sy * sy / n))
route = {
    "rho": float(spearmanr(r.rei, r.separation_at_pass_forward).statistic),
    "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
    "routes": len(r),
    "games": len(g),
}
# Same-player relation is a sensitivity analysis, after removing player means of ranks.
x = pd.Series(rankdata(r.rei))
y = pd.Series(rankdata(r.separation_at_pass_forward))
xr = x - x.groupby(r.nfl_id).transform("mean")
yr = y - y.groupby(r.nfl_id).transform("mean")
within = float(np.corrcoef(xr, yr)[0, 1])
result = {
    "outcome": "Provided separation_at_pass_forward in yards",
    "rei_routes": total,
    "routes_with_separation": len(r),
    "missing_separation": total - len(r),
    "player_raw": raw,
    "player_adjusted_rei_raw_separation": mixed,
    "player_context_adjusted": adj,
    "route_raw": route,
    "within_player_rank_residual_correlation": within,
    "notes": [
        "Player primary sample requires >=50 routes with separation and >=5 games. Both player means use exactly the same matched routes.",
        "Player intervals resample players 2000 times; they omit forecast-model fitting uncertainty and dependence between players in shared games/teams.",
        "Separation adjustment is ridge regression cross-fit in 5 folds grouped by game, controlling recorded route, position, coverage category, formation, alignment, down, yards to go, sideline distance, observed duration, mean speed and initial cushion.",
        "Excess REI uses the original validation-fitted entropy context adjustment. Separation residuals use the exploratory 2025 development sample with each game predicted out of fold.",
        "Route-level interval resamples games using fixed empirical ranks; it is an approximate cluster-bootstrap interval. Routes within a player are not independent.",
        "Within-player sensitivity demeans empirical ranks within receiver; it has no inferential interval and does not adjust play context.",
        "Release separation is a single supplied outcome, not cumulative separation; its exact defender-selection definition remains unverified.",
        "Associations are descriptive, not proof that entropy creates space; observed duration and speed are retrospective controls. No confirmatory holdout or causal effect is claimed.",
    ],
}
(O / "separation-association.json").write_text(json.dumps(result, indent=2))
players.to_csv(O / "separation-player-analysis.csv", index=False)
r.to_csv(O / "separation-route-analysis.csv.gz", index=False)
print(json.dumps(result, indent=2))
