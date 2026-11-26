"""Repeated nested player evaluation of Combine features predicting raw and adjusted separation."""

import os
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
import json, pathlib, pickle, warnings, time
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, GroupKFold, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)
O = pathlib.Path(os.environ.get("NFL_COMBINE_OUTPUT", ".work/combine-upgrades"))
G = pd.read_csv(O / "players.csv").sort_values("nfl_id").reset_index(drop=True)
R = pd.read_parquet(O / "receiver-routes.parquet")
official = [
    "combine_height",
    "combine_weight",
    "hand_size",
    "arm_length",
    "wing_span",
    "ten_yd_split",
    "forty",
    "vertical",
    "broad_jump",
    "three_cone",
    "short_shuttle",
    "bench_reps",
]
sensor = ["forty_sensor_speed95", "forty_sensor_accel95"] + [
    d + "_" + v
    for d in ["slant", "curl"]
    for v in [
        "speed95",
        "turn_rate90",
        "speed_at_turn",
        "speed_retention",
        "braking90",
        "exit_speed",
        "fractional_braking90",
    ]
]
cat = [
    "route_ran",
    "team_coverage_type",
    "offense_formation",
    "receiver_alignment",
    "lined_up_position",
    "season",
]
num = [
    "cushion",
    "time_to_throw",
    "down",
    "yards_to_go",
    "yardline_number",
    "quarter",
    "sideline_distance",
]
for c in cat:
    R[c] = R[c].fillna("Missing").astype(str)


def nuisance():
    return make_pipeline(
        ColumnTransformer(
            [
                ("c", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
                (
                    "n",
                    make_pipeline(SimpleImputer(add_indicator=True), StandardScaler()),
                    num,
                ),
            ]
        ),
        Ridge(alpha=100),
    )


def targets(tr, te):
    ids = G.iloc[tr].nfl_id.to_numpy()
    tid = G.iloc[te].nfl_id.to_numpy()
    a = R[R.nfl_id.isin(ids)].copy()
    b = R[R.nfl_id.isin(tid)].copy()
    a["adjusted"] = np.nan
    for u, v in GroupKFold(3).split(a, groups=a.nfl_id):
        m = nuisance()
        m.fit(a.iloc[u], a.iloc[u].separation_at_pass_forward)
        a.iloc[v, a.columns.get_loc("adjusted")] = a.iloc[
            v
        ].separation_at_pass_forward - m.predict(a.iloc[v])
    m = nuisance()
    m.fit(a, a.separation_at_pass_forward)
    b["adjusted"] = b.separation_at_pass_forward - m.predict(b)

    def agg(z):
        return (
            z.groupby("nfl_id")
            .agg(
                raw=("separation_at_pass_forward", "mean"),
                adjusted=("adjusted", "mean"),
                n=("adjusted", "size"),
                rawvar=("separation_at_pass_forward", "var"),
                adjustedvar=("adjusted", "var"),
            )
            .fillna(0)
        )

    aa = agg(a).loc[ids]
    bb = agg(b).loc[tid]
    return (aa, bb)


class Build(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, mode="values", interactions=False):
        self.columns = columns
        self.mode = mode
        self.interactions = interactions

    def fit(self, X, y=None):
        z = X[self.columns]
        self.med_ = z.median().fillna(0)
        self.mean_ = z.fillna(self.med_).mean()
        self.sd_ = z.fillna(self.med_).std(ddof=0).replace(0, 1)
        return self

    def transform(self, X):
        z = X[self.columns]
        flags = z.isna().astype(float).to_numpy()
        q = (z.fillna(self.med_) - self.mean_) / self.sd_
        parts = []
        if self.mode != "flags":
            parts.append(q.to_numpy())
        if self.mode != "no_flags":
            parts.append(flags)
        if self.interactions:
            pairs = [
                ("forty_sensor_speed95", "curl_braking90"),
                ("forty_sensor_accel95", "slant_speed_retention"),
                ("combine_weight", "short_shuttle"),
            ]
            parts.append(np.column_stack([q[a] * q[b] for a, b in pairs]))
        return np.column_stack(parts)


def estimator(name, feature_set, mode):
    cols = official if feature_set == "official" else official + sensor
    if name == "linear":
        m = LinearRegression()
        grid = {}
    elif name == "ridge":
        m = Ridge()
        grid = {"model__alpha": [1, 10, 100]}
    elif name == "elastic_net":
        m = ElasticNet(max_iter=5000)
        grid = {"model__alpha": [0.01, 0.1], "model__l1_ratio": [0.2, 0.8]}
    elif name == "neural_net":
        m = MLPRegressor(
            solver="lbfgs", hidden_layer_sizes=(8,), max_iter=800, random_state=2027
        )
        grid = {
            "model__alpha": [1, 10, 100],
            "model__hidden_layer_sizes": [(8,), (16,)],
        }
    elif name == "catboost":
        m = CatBoostRegressor(
            iterations=250,
            learning_rate=0.03,
            depth=2,
            verbose=False,
            random_seed=2027,
            thread_count=4,
            allow_writing_files=False,
        )
        grid = {"model__l2_leaf_reg": [10, 50], "model__depth": [2, 3]}
    else:
        m = XGBRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=1,
            reg_lambda=20,
            n_jobs=4,
            random_state=2027,
        )
        grid = {"model__min_child_weight": [5, 15], "model__max_depth": [1, 2]}
    return (
        Pipeline(
            [
                ("build", Build(cols, mode, feature_set == "interactions")),
                ("scale", StandardScaler()),
                ("model", m),
            ]
        ),
        grid,
    )


folds = []
for repeat in range(3):
    for fold, (tr, te) in enumerate(
        KFold(5, shuffle=True, random_state=2027 + repeat).split(G)
    ):
        folds.append(("repeated", repeat, fold, tr, te))
for year in [2024, 2025]:
    tr = np.flatnonzero(G.draft_year.to_numpy() < year)
    te = np.flatnonzero(G.draft_year.to_numpy() == year)
    folds.append(("draft_class", 0, year, tr, te))
cache = O / "target-cache-timing.pkl"
if cache.exists():
    T = pickle.load(open(cache, "rb"))
else:
    T = {}
    for kind, rep, fold, tr, te in folds:
        T[kind, rep, fold] = targets(tr, te)
        print("prepared targets", kind, rep, fold, flush=True)
    pickle.dump(T, open(cache, "wb"))
records = []
params = []
shrink = []
for kind, rep, fold, tr, te in folds:
    aa, bb = T[kind, rep, fold]
    configs = [
        (name, "official", "values", False, 1)
        for name in [
            "linear",
            "ridge",
            "elastic_net",
            "neural_net",
            "catboost",
            "xgboost",
        ]
    ]
    configs += [
        (name, fs, "values", False, 1)
        for name in ["ridge", "elastic_net", "neural_net"]
        for fs in ["sensors", "interactions"]
    ]
    configs += [("ridge", "official", mode, False, 1) for mode in ["flags", "no_flags"]]
    configs += [("ridge", "official", "values", True, 1)]
    configs += [
        ("ridge", "official", "values", False, minimum) for minimum in [25, 50, 100]
    ]
    for outcome in ["raw", "adjusted"]:
        mu = float(aa[outcome].mean())
        sampling = aa[outcome + "var"] / aa.n
        between = max(float(aa[outcome].var(ddof=1) - sampling.mean()), 0)
        reliability = between / (between + sampling + 1e-10)
        shrunk = mu + reliability * (aa[outcome] - mu)
        shrink.append(
            {
                "kind": kind,
                "repeat": rep,
                "fold": fold,
                "outcome": outcome,
                "between_variance": between,
                "mean_reliability": float(reliability.mean()),
            }
        )
        for name, fs, mode, pooling, minimum in configs:
            keepa = aa.n.to_numpy() >= minimum
            keepb = bb.n.to_numpy() >= minimum
            ti = tr[keepa]
            vi = te[keepb]
            if len(ti) < 12 or len(vi) < 2:
                continue
            y = (shrunk if pooling else aa[outcome]).to_numpy()[keepa]
            truth = bb[outcome].to_numpy()[keepb]
            pipe, grid = estimator(name, fs, mode)
            inner = KFold(3, shuffle=True, random_state=2027)
            search = GridSearchCV(
                pipe,
                grid,
                cv=inner,
                scoring="neg_mean_squared_error",
                n_jobs=1,
                error_score="raise",
            )
            search.fit(G.iloc[ti], y)
            pred = search.predict(G.iloc[vi])
            label = f"{name}|{fs}|{mode}|pool={pooling}|min={minimum}"
            base = float(aa.loc[G.iloc[ti].nfl_id, outcome].mean())
            params.append(
                {
                    "kind": kind,
                    "repeat": rep,
                    "fold": fold,
                    "outcome": outcome,
                    "configuration": label,
                    "parameters": search.best_params_,
                }
            )
            for idx, yv, pv in zip(vi, truth, pred):
                records.append(
                    {
                        "kind": kind,
                        "repeat": rep,
                        "fold": fold,
                        "outcome": outcome,
                        "configuration": label,
                        "nfl_id": int(G.iloc[idx].nfl_id),
                        "truth": float(yv),
                        "prediction": float(pv),
                        "baseline": base,
                    }
                )
        print("COMPLETE", kind, rep, fold, outcome, flush=True)
    pd.DataFrame(records).to_csv(O / "predictions.csv", index=False)
(O / "parameters.json").write_text(json.dumps(params, indent=2))
(O / "shrinkage.json").write_text(json.dumps(shrink, indent=2))
print("ALL FITS COMPLETE", flush=True)
