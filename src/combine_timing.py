import pathlib, json
import os
import polars as pl, numpy as np, pandas as pd

O = pathlib.Path(os.environ.get("NFL_COMBINE_OUTPUT", ".work/combine-upgrades"))
p = pd.read_parquet(O / "receiver-routes.parquet")
data = pathlib.Path(os.environ["NFL_DATA_DIR"])
parts = []
for year in [2023, 2024, 2025]:
    keys = pl.from_pandas(
        p.loc[p.season == year, ["game_id", "play_id"]].drop_duplicates()
    )
    q = (
        pl.scan_csv(
            data / f"game_tracking_{year}.csv",
            null_values=["NA", ""],
            infer_schema_length=10000,
        )
        .filter(pl.col("event").is_in(["ball_snap", "pass_forward"]))
        .join(keys.lazy(), on=["game_id", "play_id"])
        .with_columns(pl.col("time").str.to_datetime().alias("ts"))
        .collect()
    )
    snap = (
        q.filter(pl.col("event") == "ball_snap")
        .group_by("game_id", "play_id")
        .agg(
            pl.col("ts").min().alias("snap"),
            pl.col("ts").n_unique().alias("snap_times"),
        )
    )
    release = (
        q.filter(pl.col("event") == "pass_forward")
        .group_by("game_id", "play_id")
        .agg(
            pl.col("ts").min().alias("release"),
            pl.col("ts").n_unique().alias("release_times"),
        )
    )
    z = (
        snap.join(release, on=["game_id", "play_id"])
        .filter((pl.col("snap_times") == 1) & (pl.col("release_times") == 1))
        .with_columns(
            ((pl.col("release") - pl.col("snap")).dt.total_milliseconds() / 1000).alias(
                "time_to_throw"
            )
        )
    )
    parts.append(z.select("game_id", "play_id", "time_to_throw").to_pandas())
t = pd.concat(parts)
p = p.merge(t, on=["game_id", "play_id"], how="left", validate="m:1")
p.loc[~p.time_to_throw.between(0.1, 10), "time_to_throw"] = np.nan
p.to_parquet(O / "receiver-routes.parquet", index=False)
a = json.loads((O / "data-audit.json").read_text())
a["missing_context_fraction"]["time_to_throw"] = float(p.time_to_throw.isna().mean())
(O / "data-audit.json").write_text(json.dumps(a, indent=2))
print("Time to throw availability", p.time_to_throw.notna().mean())
