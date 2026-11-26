"""Position-group pre-snap separation expectation without player-identity or post-snap inputs."""

import os
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
import pathlib, json, pickle, warnings
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, GroupKFold, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore", category=UserWarning)
O = pathlib.Path(os.environ.get("NFL_PRESNAP_OUTPUT", ".work/presnap"))
p = pd.read_parquet(
    str(pathlib.Path(os.environ.get("NFL_BASE_WORK", ".work/base")) / "context.parquet")
)
db = p[(p.season_type == "REG") & p.nfl_position.isin(["CB", "FS", "SS"])]
audit = (
    db.groupby("nfl_position")
    .agg(
        rows=("nfl_id", "size"),
        separation_available=("separation_at_pass_forward", "count"),
    )
    .to_dict("index")
)
R = pd.read_parquet(
    str(
        pathlib.Path(os.environ.get("NFL_COMBINE_OUTPUT", ".work/combine-upgrades"))
        / "receiver-routes.parquet"
    )
)
G = (
    pd.read_csv(
        str(
            pathlib.Path(os.environ.get("NFL_COMBINE_OUTPUT", ".work/combine-upgrades"))
            / "players.csv"
        )
    )
    .sort_values("nfl_id")
    .reset_index(drop=True)
)
R["defense_team"] = np.where(
    R.possession_team == R.home_team_abbr, R.visitor_team_abbr, R.home_team_abbr
)
R["clock_seconds"] = R.game_clock.map(
    lambda v: float(str(v).split(":")[0]) * 60 + float(str(v).split(":")[1])
    if ":" in str(v)
    else np.nan
)
R["score_margin"] = np.where(
    R.possession_team == R.home_team_abbr,
    R.pre_snap_home_score - R.pre_snap_visitor_score,
    R.pre_snap_visitor_score - R.pre_snap_home_score,
)
R["own_yardline"] = np.where(
    R.possession_team == R.yardline_side, R.yardline_number, 100 - R.yardline_number
)
cat = [
    "offense_formation",
    "receiver_alignment",
    "lined_up_position",
    "in_motion_at_ball_snap",
    "possession_team",
    "defense_team",
    "season",
]
num = [
    "down",
    "yards_to_go",
    "quarter",
    "clock_seconds",
    "score_margin",
    "own_yardline",
    "cushion",
    "snap_y",
    "snap_x",
]
if "snap_x" not in R:
    R = R.merge(
        pd.read_parquet(
            str(
                pathlib.Path(os.environ.get("NFL_BASE_WORK", ".work/base"))
                / "snaps.parquet"
            )
        )[["row_id", "snap_x"]],
        on="row_id",
        how="left",
        validate="1:1",
    )
for c in cat:
    R[c] = R[c].fillna("Missing").astype(str)


def context_model(kind):
    pre = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat,
            ),
            (
                "numeric",
                make_pipeline(SimpleImputer(add_indicator=True), StandardScaler()),
                num,
            ),
        ]
    )
    if kind == "ridge":
        reg = Ridge(alpha=100)
    elif kind == "hist_gradient_boosting":
        reg = HistGradientBoostingRegressor(
            max_iter=150,
            max_leaf_nodes=15,
            l2_regularization=30,
            learning_rate=0.06,
            early_stopping=False,
            random_state=2027,
        )
    else:
        reg = CatBoostRegressor(
            iterations=250,
            depth=4,
            learning_rate=0.04,
            l2_leaf_reg=30,
            verbose=False,
            thread_count=4,
            random_seed=2027,
            allow_writing_files=False,
        )
    return make_pipeline(pre, reg)


features = cat + num
Y = "separation_at_pass_forward"
foldrecords = []
routepred = []
targets = {}
for repeat in range(3):
    for fold, (tr, te) in enumerate(
        KFold(5, shuffle=True, random_state=2027 + repeat).split(G)
    ):
        ai = G.iloc[tr].nfl_id.to_numpy()
        bi = G.iloc[te].nfl_id.to_numpy()
        a = R[R.nfl_id.isin(ai)].reset_index(drop=True)
        b = R[R.nfl_id.isin(bi)].reset_index(drop=True)
        inner = list(GroupKFold(3).split(a, groups=a.nfl_id))
        candidates = []
        for name in ["ridge", "hist_gradient_boosting", "catboost"]:
            pred = np.empty(len(a))
            for u, v in inner:
                m = context_model(name)
                m.fit(a.iloc[u][features], a.iloc[u][Y])
                pred[v] = m.predict(a.iloc[v][features])
            candidates.append((mean_squared_error(a[Y], pred), name, pred))
        mse, name, apred = min(candidates, key=lambda z: z[0])
        m = context_model(name)
        m.fit(a[features], a[Y])
        bp = m.predict(b[features])
        a["residual"] = a[Y] - apred
        b["residual"] = b[Y] - bp

        def agg(z):
            return z.groupby("nfl_id").agg(
                raw=(Y, "mean"), adjusted=("residual", "mean"), n=(Y, "size")
            )

        targets[repeat, fold] = (agg(a).loc[ai], agg(b).loc[bi])
        foldrecords.append(
            {
                "repeat": repeat,
                "fold": fold,
                "selected": name,
                "training_players": len(tr),
                "evaluation_players": len(te),
                "inner_mse": {n: float(loss) for loss, n, _ in candidates},
                "test_mse": float(mean_squared_error(b[Y], bp)),
            }
        )
        out = b[["row_id", "nfl_id", Y]].copy()
        out["expected"] = bp
        out["above_expected"] = b.residual
        out["repeat"] = repeat
        out["fold"] = fold
        routepred.append(out)
        print("baseline", repeat, fold, name, flush=True)
        pickle.dump(targets, open(O / "targets.pkl", "wb"))
pred = pd.concat(routepred, ignore_index=True)
pred.to_csv(O / "route-predictions.csv.gz", index=False)
(O / "baseline-folds.json").write_text(json.dumps(foldrecords, indent=2))
from combine_model_library import estimator

records = []
for (repeat, fold), (aa, bb) in targets.items():
    tr = G.set_index("nfl_id").loc[aa.index].reset_index()
    te = G.set_index("nfl_id").loc[bb.index].reset_index()
    for name in ["linear", "ridge", "elastic_net", "neural_net", "catboost", "xgboost"]:
        pipe, grid = estimator(name, "official", "values")
        fit = GridSearchCV(
            pipe,
            grid,
            cv=KFold(3, shuffle=True, random_state=2027),
            scoring="neg_mean_squared_error",
        )
        fit.fit(tr, aa.adjusted)
        out = fit.predict(te)
        for nfl, y, v in zip(te.nfl_id, bb.adjusted, out):
            records.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "model": name,
                    "nfl_id": int(nfl),
                    "above_expected": float(y),
                    "prediction": float(v),
                }
            )
    print("combine", repeat, fold, flush=True)
c = pd.DataFrame(records)
c.to_csv(O / "combine-predictions.csv", index=False)
stats = []
for name, z in c.groupby("model"):
    rep = []
    for i, g in z.groupby("repeat"):
        rep.append(
            {
                "repeat": int(i),
                "r2": float(r2_score(g.above_expected, g.prediction)),
                "pearson": float(pearsonr(g.above_expected, g.prediction).statistic),
                "rmse": float(
                    np.sqrt(mean_squared_error(g.above_expected, g.prediction))
                ),
            }
        )
    stats.append(
        {
            "model": name,
            "mean_r2": float(np.mean([v["r2"] for v in rep])),
            "mean_pearson": float(np.mean([v["pearson"] for v in rep])),
            "repeats": rep,
        }
    )
baselines = []
for rep, g in pred.groupby("repeat"):
    baselines.append(
        {
            "repeat": int(rep),
            "route_r2": float(r2_score(g[Y], g.expected)),
            "route_rmse": float(np.sqrt(mean_squared_error(g[Y], g.expected))),
            "route_pearson": float(pearsonr(g[Y], g.expected).statistic),
        }
    )
players = pred.groupby("nfl_id").agg(
    actual=(Y, "mean"),
    expected=("expected", "mean"),
    above_expected=("above_expected", "mean"),
)
players.to_csv(O / "player-above-expected.csv")
old = pickle.load(
    open(
        str(
            pathlib.Path(os.environ.get("NFL_COMBINE_OUTPUT", ".work/combine-upgrades"))
            / "target-cache-timing.pkl"
        ),
        "rb",
    )
)
comparisons = []
for rep in range(3):
    pairs = []
    for fold in range(5):
        _, b = targets[rep, fold]
        _, ob = old["repeated", rep, fold]
        pairs.append(b[["adjusted"]].join(ob[["adjusted"]], rsuffix="_old"))
    z = pd.concat(pairs)
    comparisons.append(float(pearsonr(z.adjusted, z.adjusted_old).statistic))
summary = {
    "db_audit": audit,
    "population": "82 wide receivers; 30,566 nonmissing release-separation observations in regular-season 2023–2025",
    "expected_separation_definition": "Pooled WR outcome regression based only on beginning-of-play information. Each evaluated player is excluded from training and from selecting the baseline model.",
    "categorical_inputs": cat,
    "numeric_inputs": num,
    "excluded": [
        "player identity",
        "Combine measurements",
        "recorded route",
        "actual post-snap coverage",
        "time to throw",
        "pass result",
        "target status",
        "receiving yards",
    ],
    "baseline_evaluation": baselines,
    "combine_to_above_expected": stats,
    "player_actual_expected_correlation": float(
        pearsonr(players.actual, players.expected).statistic
    ),
    "player_above_expected_sd_yards": float(players.above_expected.std()),
    "old_postsnap_vs_new_presnap_adjusted_player_correlations": comparisons,
    "selected_model_counts": pd.Series([x["selected"] for x in foldrecords])
    .value_counts()
    .to_dict(),
    "notes": [
        "All receivers contribute to fitting the population expectation when in a training fold; no baseline is fit separately to each player average.",
        "WR population only. Tight ends, safeties and cornerbacks require their own outcome definitions/population baselines.",
        "Cushion is treated as a supplied starting-condition measure; snap position and pre-snap situation are included. Actual separation is only the dependent variable.",
        "Route-level baseline candidates are selected by player-grouped inner validation. Training-player residuals use cross-fitted baseline predictions. Outer evaluation players do not fit or select the nuisance model or Combine weights.",
        "Nested player partitions do not exclude shared games or teams; team context is known pre-snap and can capture scheme and roster effects. Residuals are descriptive above expectation, not isolated causal player skill.",
        "The same previously examined 2023–2025 data is used. Repeated validation is exploratory, not pristine confirmation.",
        "Positive receiver residual means more separation than expected. For confirmed DB outcomes the useful sign would reverse: expected separation allowed minus actual separation allowed. Those DB outcomes are unavailable; no nearest-recorded-player substitute is used.",
    ],
}
(O / "results.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
