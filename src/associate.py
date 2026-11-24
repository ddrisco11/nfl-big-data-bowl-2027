"""Test player-level Combine associations with multiplicity-adjusted inference."""

import os
from shared import *
from scipy.stats import rankdata

rng = np.random.default_rng(2027)
a = pd.read_csv(O / "player-unpredictability.csv")
a = a[~a.provisional].merge(
    pd.read_csv(O / "combine-features.csv"), on="nfl_id", suffixes=("", "_combine")
)
features = [
    "combine_height",
    "combine_weight",
    "ten_yd_split",
    "forty",
    "vertical",
    "broad_jump",
    "three_cone",
    "short_shuttle",
    "forty_sensor_speed95",
    "forty_sensor_accel95",
    "slant_speed95",
    "slant_turn_rate90",
    "slant_speed_at_turn",
    "slant_speed_retention",
    "curl_speed95",
    "curl_turn_rate90",
    "curl_speed_at_turn",
    "curl_speed_retention",
]


def residual(z, column):
    yy = rankdata(z[column])
    cats = pd.get_dummies(
        z[["position", "draft_year"]].astype(str), drop_first=True, dtype=float
    ).to_numpy()
    x = np.column_stack(
        [np.ones(len(z)), cats, np.log1p(z.routes), z.mean_observed_duration]
    )
    return yy - x @ np.linalg.lstsq(x, yy, rcond=None)[0]


def corr(z, x, y):
    return float(np.corrcoef(residual(z, x), residual(z, y))[0, 1])


out = []
for outcome in ["excess_unpredictability", "excess_brier"]:
    for feature in features:
        z = a.dropna(subset=[feature, outcome]).reset_index(drop=True)
        if len(z) < 25:
            out.append(
                {
                    "feature": feature,
                    "outcome": outcome,
                    "n": len(z),
                    "status": "insufficient sample",
                }
            )
            continue
        x = residual(z, feature)
        y = residual(z, outcome)
        rho = float(np.corrcoef(x, y)[0, 1])
        null = np.array([np.corrcoef(x, rng.permutation(y))[0, 1] for _ in range(3000)])
        p = float((1 + np.sum(np.abs(null) >= abs(rho))) / 3001)
        boot = [
            corr(z.iloc[rng.integers(0, len(z), len(z))], feature, outcome)
            for _ in range(800)
        ]
        out.append(
            {
                "feature": feature,
                "outcome": outcome,
                "n": len(z),
                "rho": rho,
                "ci95": np.nanquantile(boot, [0.025, 0.975]).tolist(),
                "p": p,
                "status": "exploratory",
            }
        )
ps = np.array([r.get("p", 1) for r in out])
order = np.argsort(ps)
q = np.minimum.accumulate((ps[order] * len(out) / np.arange(1, len(out) + 1))[::-1])[
    ::-1
]
for i, k in enumerate(order):
    out[k]["fdr_q"] = float(min(q[i], 1))
# Association with separation: average per route, no untracked-defender geometry.
r = pd.read_csv(O / "route-unpredictability.csv.gz")
sep = (
    r.groupby("nfl_id")
    .agg(
        separation=("separation_at_pass_forward", "mean"),
        sep_n=("separation_at_pass_forward", "count"),
    )
    .reset_index()
)
z = a.merge(sep, on="nfl_id")
z = z[z.sep_n >= 50].dropna(subset=["separation"])
rho = corr(z, "excess_unpredictability", "separation")
boot = [
    corr(
        z.iloc[rng.integers(0, len(z), len(z))], "excess_unpredictability", "separation"
    )
    for _ in range(1000)
]
result = {
    "tests": out,
    "players": len(a),
    "separation": {
        "n": len(z),
        "rho": rho,
        "ci95": np.nanquantile(boot, [0.025, 0.975]).tolist(),
    },
    "notes": [
        "Two outcomes × 18 Combine features; one BH correction across all 36 planned tests, including untestable features as p=1.",
        "Complete-case player-level partial rank correlations adjust for position, draft class, route count and observed duration. Score already adjusts route and play context using validation data.",
        "Intervals bootstrap players; residual permutation p-values are exploratory. Shared team/scheme and model-estimation uncertainty remain.",
        "Separation check uses available precomputed separation at release, not a claim about nearest defenders in the cohort tracking.",
    ],
}
(O / "combine-associations.json").write_text(
    json.dumps(result, indent=2, allow_nan=False)
)
a.to_csv(O / "association-player-data.csv", index=False)
print("Association minimum q", min(t["fdr_q"] for t in out))
print("Separation", result["separation"])
