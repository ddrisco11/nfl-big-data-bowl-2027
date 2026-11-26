import os
import pathlib, json
import polars as pl

D = pathlib.Path(os.environ["NFL_DATA_DIR"])
O = pathlib.Path(os.environ.get("NFL_SCREEN_OUTPUT", ".work/performance-screen"))
meta = (
    pl.read_parquet(
        str(
            pathlib.Path(os.environ.get("NFL_BASE_WORK", ".work/base"))
            / "context.parquet"
        )
    )
    .filter(pl.col("season_type") == "REG")
    .select("game_id", "play_id", "nfl_id")
)
keys = ["game_id", "play_id", "nfl_id"]
parts = []
for year in [2023, 2024, 2025]:
    q = pl.scan_csv(
        D / f"game_tracking_{year}.csv",
        null_values=["NA", ""],
        infer_schema_length=10000,
    ).with_columns(pl.col("time").str.to_datetime().alias("ts"))
    snap = (
        q.filter(pl.col("event") == "ball_snap")
        .group_by(keys)
        .agg(pl.col("ts").min().alias("snap"), pl.len().alias("snaps"))
        .filter(pl.col("snaps") == 1)
    )
    end = (
        q.filter(
            pl.col("event").is_in(
                [
                    "pass_forward",
                    "tackle",
                    "out_of_bounds",
                    "qb_sack",
                    "pass_outcome_incomplete",
                    "touchdown",
                    "fumble",
                ]
            )
        )
        .group_by(keys)
        .agg(pl.col("ts").min().alias("end"))
    )
    z = (
        q.join(snap, on=keys)
        .join(end, on=keys, how="left")
        .join(meta.lazy(), on=keys)
        .filter(
            (pl.col("ts") >= pl.col("snap"))
            & (pl.col("ts") <= pl.col("snap") + pl.duration(seconds=3))
            & (pl.col("end").is_null() | (pl.col("ts") < pl.col("end")))
        )
        .select("nfl_id", "s", "a")
        .collect()
    )
    parts.append(z)
    print(year, len(z), flush=True)
z = (
    pl.concat(parts)
    .group_by("nfl_id")
    .agg(
        pl.len().alias("movement_frames"),
        pl.col("s").mean().alias("game_mean_speed"),
        pl.col("s").quantile(0.95).alias("game_speed95"),
        pl.col("s").quantile(0.99).alias("game_speed99"),
        pl.col("a").quantile(0.95).alias("game_acceleration95"),
    )
)
z.write_csv(O / "movement-targets.csv")
print("Movement aggregation complete")
