"""Extract interpretable Combine route-drill mechanics and audit player outcomes."""

import os
import pathlib, json
import numpy as np, pandas as pd, polars as pl

O = pathlib.Path(os.environ.get("NFL_COMBINE_OUTPUT", ".work/combine-upgrades"))
data = pathlib.Path(os.environ["NFL_DATA_DIR"])
c = pd.read_csv(
    str(
        pathlib.Path(os.environ.get("NFL_OUTPUT_DIR", ".work/results"))
        / "combine-features.csv"
    )
)
q = (
    pl.scan_csv(
        data / "combine_tracking.csv", null_values=["NA", ""], infer_schema_length=10000
    )
    .filter(pl.col("drill_name").is_in(["SLANT_ROUTE_LEFT", "CURL_ROUTE_RIGHT"]))
    .collect()
    .to_pandas()
)
rows = []
for (nfl, drill, event, attempt), z in q.groupby(
    ["nfl_id", "drill_name", "event_id", "attempt"]
):
    z = z.sort_values("time")
    t = (
        pd.to_datetime(z.time).astype("datetime64[ns]").astype("int64").to_numpy()
        / 1000000000.0
    )
    s = z.s.to_numpy()
    direction = np.deg2rad(z.dir.to_numpy())
    dt = t[2:] - t[:-2]
    decel = -(s[2:] - s[:-2]) / np.maximum(dt, 0.01)
    turn = np.abs(
        np.arctan2(
            np.sin(direction[2:] - direction[:-2]),
            np.cos(direction[2:] - direction[:-2]),
        )
    ) / np.maximum(dt, 0.01)
    dist = np.cumsum(np.r_[0, np.hypot(np.diff(z.x), np.diff(z.y))])
    fraction = dist[1:-1] / max(dist[-1], 0.01)
    valid = (
        (dt > 0.15) & (dt < 0.25) & (fraction > 0.15) & (fraction < 0.8) & (s[1:-1] > 2)
    )
    if valid.sum() < 3:
        continue
    k = np.flatnonzero(valid)[np.argmax(turn[valid])] + 1
    label = "slant" if drill.startswith("SLANT") else "curl"
    rows.append(
        {
            "nfl_id": int(nfl),
            "drill": label,
            "braking90": float(np.quantile(np.maximum(decel[valid], 0), 0.9)),
            "exit_speed": float(s[k + 1 : min(k + 4, len(s))].mean()),
            "fractional_braking90": float(
                np.quantile(
                    np.maximum(decel[valid], 0) / np.maximum(s[1:-1][valid], 2), 0.9
                )
            ),
        }
    )
a = pd.DataFrame(rows)
a.to_csv(O / "drill-attempt-features.csv", index=False)
f = (
    a.groupby(["nfl_id", "drill"])[["braking90", "exit_speed", "fractional_braking90"]]
    .median()
    .unstack()
)
f.columns = [d + "_" + v for v, d in f.columns]
c = c.merge(f.reset_index(), on="nfl_id", how="left")
c.to_csv(O / "combine-features.csv", index=False)
p = pd.read_parquet(
    str(pathlib.Path(os.environ.get("NFL_BASE_WORK", ".work/base")) / "context.parquet")
)
p = p[
    (p.season_type == "REG")
    & (p.nfl_position == "WR")
    & p.separation_at_pass_forward.notna()
].copy()
p["sideline_distance"] = (
    np.minimum(
        np.where(
            p.play_direction == "left",
            160 / 3 - p.get("snap_y", np.nan),
            p.get("snap_y", np.nan),
        ),
        80 / 3,
    )
    if "snap_y" in p
    else np.nan
)
snaps = pd.read_parquet(
    str(pathlib.Path(os.environ.get("NFL_BASE_WORK", ".work/base")) / "snaps.parquet")
)[["row_id", "snap_y"]]
p = p.drop(columns=["sideline_distance"]).merge(
    snaps, on="row_id", how="left", validate="1:1"
)
p["sideline_distance"] = np.minimum(p.snap_y, 160 / 3 - p.snap_y)
p.to_parquet(O / "receiver-routes.parquet", index=False)
g = p.groupby("nfl_id").agg(
    player=("display_name", "first"),
    draft_year=("draft_year", "first"),
    observations=("separation_at_pass_forward", "size"),
    games=("game_id", "nunique"),
    mean_separation=("separation_at_pass_forward", "mean"),
)
g = g.reset_index().merge(c.drop(columns="draft_year"), on="nfl_id", validate="1:1")
g.to_csv(O / "players.csv", index=False)
features = [
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
audit = {
    "players": len(g),
    "routes": len(p),
    "draft_classes": {str(k): int(v) for k, v in g.draft_year.value_counts().items()},
    "official_missing_fraction": {k: float(g[k].isna().mean()) for k in features},
    "sensor_available": {
        k: int(g[k].notna().sum())
        for k in c
        if k.startswith(("slant_", "curl_", "forty_sensor_"))
    },
    "missing_context_fraction": {
        k: float(p[k].isna().mean())
        for k in [
            "cushion",
            "dropback_duration",
            "snap_y",
            "route_ran",
            "team_coverage_type",
        ]
    },
}
(O / "data-audit.json").write_text(json.dumps(audit, indent=2))
print(json.dumps(audit, indent=2))
