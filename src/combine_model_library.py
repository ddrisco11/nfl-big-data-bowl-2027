import os
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
