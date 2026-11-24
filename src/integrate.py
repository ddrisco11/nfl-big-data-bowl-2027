"""Integrate native-frame prediction losses and estimate context-adjusted player scores."""

import os
from shared import *
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr

D, S, C, H, V, Y = load()
P = np.load(W / "Movement_recognition.npy")
rng = np.random.default_rng(2027)
rows = []
# np.trapezoid implements the integral on native 100 ms frames, not probability interpolation.
for i in np.flatnonzero((D.split != "train").to_numpy()):
    valid = np.flatnonzero(V[i])
    n = len(valid)
    if n < 6:
        continue
    # V is prefix-contiguous; do not bridge missing frames.
    assert np.array_equal(valid, np.arange(n))
    p = P[i, :n]
    truth = int(Y[i])
    t = TIMES[:n]
    duration = float(t[-1])
    error = 1 - p[:, truth]
    brier = np.sum((p - np.eye(12)[truth]) ** 2, axis=1)
    surprise = -np.log(np.maximum(p[:, truth], 1e-8))
    u = 100 * np.trapezoid(error, t) / duration
    bs = 50 * np.trapezoid(brier, t) / duration
    ll = np.trapezoid(surprise, t) / duration
    entropy = -np.sum(p * np.log(np.maximum(p, 1e-8)), axis=1)
    ambiguity = 100 * np.trapezoid(entropy, t) / (duration * np.log(12))
    wrong = (p.argmax(1) != truth) & (p.max(1) >= 0.8)
    wrong_fraction = np.trapezoid(wrong.astype(float), t) / duration
    fixed = 100 * np.trapezoid(error[:21], t[:21]) / 2 if n >= 21 else np.nan
    row = D.iloc[i][
        [
            "row_id",
            "game_id",
            "play_id",
            "nfl_id",
            "display_name",
            "nfl_position",
            "draft_year",
            "split",
            "route_ran",
            "offense_formation",
            "receiver_alignment",
            "lined_up_position",
            "down",
            "yards_to_go",
            "snap_y",
            "separation_at_pass_forward",
        ]
    ].to_dict()
    row.update(
        duration_seconds=duration,
        native_frames=n,
        unpredictability=float(u),
        integrated_brier=float(bs),
        integrated_log_loss=float(ll),
        integrated_ambiguity=float(ambiguity),
        confident_wrong_fraction=float(wrong_fraction),
        fixed_two_second_unpredictability=float(fixed),
        capped_at_three_seconds=bool(n == 31),
    )
    rows.append(row)
r = pd.DataFrame(rows)
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
prep = ColumnTransformer(
    [
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ("num", SimpleImputer(), num),
    ]
)
model = make_pipeline(prep, Ridge(alpha=20))
val = r.split == "validation"
model.fit(r.loc[val, cat + num], r.loc[val, "unpredictability"])
r["expected_unpredictability"] = model.predict(r[cat + num])
r["excess_unpredictability"] = r.unpredictability - r.expected_unpredictability
# Separate proper-scoring-rule residual provides a sensitivity check.
bm = make_pipeline(
    ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
            ("num", SimpleImputer(), num),
        ]
    ),
    Ridge(alpha=20),
)
bm.fit(r.loc[val, cat + num], r.loc[val, "integrated_brier"])
r["excess_brier"] = r.integrated_brier - bm.predict(r[cat + num])
pickle.dump((model, bm), open(W / "score-adjustment.pkl", "wb"))
r[r.split == "test"].to_csv(
    O / "route-unpredictability.csv.gz", index=False, compression="gzip"
)
test = r[r.split == "test"]
players = []
for nfl, g in test.groupby("nfl_id"):
    agg = g.groupby("game_id").agg(
        total=("excess_unpredictability", "sum"), n=("row_id", "size")
    )
    arr = agg.to_numpy()
    draw = rng.integers(0, len(arr), (1000, len(arr)))
    sums = arr[draw].sum(1)
    boot = sums[:, 0] / sums[:, 1]
    ci = np.quantile(boot, [0.025, 0.975])
    players.append(
        {
            "nfl_id": int(nfl),
            "player": g.display_name.iloc[0],
            "position": g.nfl_position.iloc[0],
            "draft_year": int(g.draft_year.iloc[0]),
            "routes": len(g),
            "games": len(agg),
            "unpredictability": float(g.unpredictability.mean()),
            "excess_unpredictability": float(g.excess_unpredictability.mean()),
            "excess_ci_low": float(ci[0]),
            "excess_ci_high": float(ci[1]),
            "bootstrap_se": float(np.std(boot)),
            "integrated_brier": float(g.integrated_brier.mean()),
            "integrated_ambiguity": float(g.integrated_ambiguity.mean()),
            "confident_wrong_fraction": float(g.confident_wrong_fraction.mean()),
            "excess_brier": float(g.excess_brier.mean()),
            "fixed_two_second_score": float(g.fixed_two_second_unpredictability.mean())
            if g.fixed_two_second_unpredictability.notna().any()
            else None,
            "fixed_two_second_routes": int(
                g.fixed_two_second_unpredictability.notna().sum()
            ),
            "mean_observed_duration": float(g.duration_seconds.mean()),
            "provisional": bool(len(g) < 50 or len(agg) < 5),
        }
    )
a = pd.DataFrame(players)
qualified = ~a.provisional
tau = max(
    float(
        a.loc[qualified, "excess_unpredictability"].var()
        - np.mean(a.loc[qualified, "bootstrap_se"] ** 2)
    ),
    0,
)
grand = float(a.loc[qualified, "excess_unpredictability"].mean())
a["reliability_weight"] = tau / (tau + a.bootstrap_se**2 + 1e-10)
a["shrunk_excess_score"] = grand + a.reliability_weight * (
    a.excess_unpredictability - grand
)
a.to_csv(O / "player-unpredictability.csv", index=False)
# Repeatability uses disjoint games, and does not claim independent player-quality validation.
test = test.copy()
test["half"] = test.game_id % 2
hh = (
    test.groupby(["nfl_id", "half"])
    .agg(score=("excess_unpredictability", "mean"), n=("row_id", "size"))
    .reset_index()
    .pivot(index="nfl_id", columns="half", values=["score", "n"])
    .dropna()
)
hh = hh[hh["n"].min(axis=1) >= 25]
rep = float(spearmanr(hh["score"][0], hh["score"][1]).statistic)
checks = {
    "unit_interval_bounds": bool(r.unpredictability.between(-0.0001, 100.0001).all()),
    "proper_brier_bounds": bool(r.integrated_brier.between(-0.0001, 100.0001).all()),
    "native_step_seconds": 0.1,
    "constant_probability_check": float(
        100 * np.trapezoid(np.full(31, 0.7), TIMES) / 3
    ),
    "uniform_probability_reference": float(100 * (1 - 1 / 12)),
}
assert checks["unit_interval_bounds"] and checks["proper_brier_bounds"]
assert abs(checks["constant_probability_check"] - 70) < 1e-8
meta = {
    "formula": "100 / T × trapezoid(1 - P(actual route | prefix), native frame times)",
    "window": "Snap to last eligible observed frame before pass release/end, capped at 3 seconds; minimum 0.5 seconds. Separate fixed 0–2-second score.",
    "n_scored_test_routes": len(test),
    "n_players": len(a),
    "n_qualified_players": int(qualified.sum()),
    "mean_route_score": float(test.unpredictability.mean()),
    "median_route_score": float(test.unpredictability.median()),
    "fraction_reaching_three_second_cap": float(test.capped_at_three_seconds.mean()),
    "split_game_repeatability": {"players": len(hh), "spearman": rep},
    "proper_score_agreement": float(
        spearmanr(
            a.loc[qualified, "excess_unpredictability"],
            a.loc[qualified, "excess_brier"],
        ).statistic
    ),
    "checks": checks,
    "notes": [
        "Score 0 means probability 1 on the actual route throughout; 100 means probability 0 throughout. A uniform 12-route prediction scores 91.67, not 50.",
        "Integrated entropy is also reported: many plausible routes and a confidently wrong prediction are different phenomena. The confident-wrong fraction flags frames with incorrect top route and confidence >=0.8.",
        "This is model unpredictability. It is not a causal measure of deception, receiver quality, or defender perception.",
        "Duration normalization prevents a longer route automatically accumulating more error, but cannot remove all duration-selection effects. Use the fixed-window sensitivity measure.",
        "Context adjustment is fit only on validation plays, using recorded route, position, formation, alignment, down, yards-to-go, sideline distance and duration. No test outcome or Combine feature fits that adjustment.",
        "Player means weight each eligible route equally. Bootstrap intervals resample games within a player; provisional flags indicate fewer than 50 routes or 5 games.",
        "Shrunk scores pull noisy estimates toward the qualified-player mean. Displayed bootstrap intervals describe the unshrunk excess score.",
        "The 1-p score is intuitive but is not a strictly proper scoring rule. Integrated multiclass Brier and log loss are included as robustness measures.",
        "Only eligible 2025 regular-season WR/TE routes are scored; these are not complete career totals or all 510 prospects. Scores do not cover the full route after pass release or beyond the three-second analysis window.",
    ],
}
(O / "unpredictability-method.json").write_text(
    json.dumps(meta, indent=2, allow_nan=False)
)
print(json.dumps(meta, indent=2))
print("INTEGRAL SCORE COMPLETE", flush=True)
