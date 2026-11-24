"""Shared route labels, static context, and probability-temperature utilities."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
import pathlib, numpy as np, pandas as pd, polars as pl

OLD = pathlib.Path(os.environ.get("NFL_BASE_WORK", ".work/base"))
DATA = pathlib.Path(os.environ["NFL_DATA_DIR"])
CLASSES = [
    "ANGLE",
    "CORNER",
    "CROSS",
    "FLAT",
    "GO",
    "HITCH",
    "IN",
    "OUT",
    "POST",
    "SCREEN",
    "SLANT",
    "WHEEL",
]


def load_routes():
    p = pl.read_parquet(OLD / "context.parquet").to_pandas()
    s = pl.read_parquet(OLD / "snaps.parquet").to_pandas()
    c = pl.read_parquet(OLD / "combine.parquet").to_pandas()
    d = (
        p[p.route_eligible & p.nfl_position.isin(["WR", "TE"])]
        .merge(s, on="row_id")
        .merge(c.drop(columns=["draft_year"]), on="nfl_id", how="left")
        .sort_values("row_id")
        .reset_index(drop=True)
    )
    r = pl.read_parquet(OLD / "trajectories.parquet").to_pandas()
    r = r[r.row_id.isin(d.row_id)]
    lookup = pd.Series(np.arange(len(d)), index=d.row_id)
    A = np.full((len(d), 21, 6), np.nan)
    live = np.zeros((len(d), 21), bool)
    ii = lookup.loc[r.row_id].to_numpy()
    jj = (r.frame // 2).to_numpy()
    A[ii, jj] = r[["nx", "ny", "s", "a", "vx", "vy"]].to_numpy()
    live[ii, jj] = r.live.to_numpy()
    A[:, :, 0] -= d.snap_x.to_numpy()[:, None]
    A[:, :, 1] -= d.snap_y.to_numpy()[:, None]
    mirror = np.where(d.snap_y < 80 / 3, 1.0, -1.0)
    A[:, :, 1] *= mirror[:, None]
    A[:, :, 5] *= mirror[:, None]
    d["mirror"] = mirror
    return d, A, live


CAT = ["lined_up_position", "nfl_position", "offense_formation", "receiver_alignment"]


def make_features(d, A, obs):
    j = obs * 5
    X = d[
        [
            "down",
            "yards_to_go",
            "quarter",
            "pre_snap_home_score",
            "pre_snap_visitor_score",
            "in_motion_at_ball_snap",
        ]
        + CAT
    ].copy()
    for c in CAT:
        X[c] = X[c].fillna("Missing").astype(str)
    X["in_motion_at_ball_snap"] = (
        X.in_motion_at_ball_snap.astype(str)
        .str.lower()
        .map({"true": 1, "false": 0})
        .fillna(0)
    )
    own = np.where(
        d.yardline_side == d.possession_team, d.yardline_number, 100 - d.yardline_number
    )
    X["yards_from_own_goal"] = own
    X["snap_from_los"] = d.snap_x - (own + 10)
    X["sideline_distance"] = np.minimum(d.snap_y, 160 / 3 - d.snap_y)
    X["score_difference"] = np.where(
        d.possession_team == d.home_team_abbr,
        d.pre_snap_home_score - d.pre_snap_visitor_score,
        d.pre_snap_visitor_score - d.pre_snap_home_score,
    )
    clock = d.game_clock.str.split(":", expand=True)
    X["quarter_seconds"] = pd.to_numeric(
        clock[0], errors="coerce"
    ) * 60 + pd.to_numeric(clock[1], errors="coerce")
    new = {}
    for k in range(j + 1):
        for z, c in enumerate(["dx", "dy_inward", "speed", "accel", "vx", "vy_inward"]):
            new[f"{c}_{k}"] = A[:, k, z]
    for z, c in enumerate(["dx", "dy", "speed", "accel", "vx", "vy"]):
        new[c + "_mean"] = np.nanmean(A[:, : j + 1, z], axis=1)
        new[c + "_change"] = A[:, j, z] - A[:, max(0, j - 2), z]
    if j:
        diff = np.diff(A[:, : j + 1, :2], axis=1)
        dist = np.linalg.norm(diff, axis=2)
        new["path_length"] = dist.sum(1)
        new["straightness"] = np.linalg.norm(A[:, j, :2], axis=1) / np.maximum(
            dist.sum(1), 0.01
        )
        for z, c in enumerate(["x", "y"]):
            new["recent_" + c + "_velocity"] = (A[:, j, z] - A[:, max(0, j - 2), z]) / (
                min(2, j) * 0.2
            )
        ang = np.arctan2(A[:, : j + 1, 5], A[:, : j + 1, 4])
        new["absolute_turn"] = np.abs(
            np.arctan2(np.sin(np.diff(ang, axis=1)), np.cos(np.diff(ang, axis=1)))
        ).sum(1)
    return pd.concat([X, pd.DataFrame(new)], axis=1)


def masks(d, A, live, obs):
    valid = (
        live[:, obs * 5]
        & np.isfinite(A[:, : obs * 5 + 1]).all(axis=(1, 2))
        & (d.season_type == "REG").to_numpy()
    )
    return [valid & (d.split == s).to_numpy() for s in ["train", "validation", "test"]]


def temp_prob(P, t):
    z = np.log(np.maximum(P, 1e-9)) / t
    z -= z.max(1, keepdims=True)
    q = np.exp(z)
    return q / q.sum(1, keepdims=True)
