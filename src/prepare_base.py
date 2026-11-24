"""Join competition tables and build the audited route cohort and Combine summaries."""

import os
import polars as pl, numpy as np, json, pathlib

D = pathlib.Path(os.environ["NFL_DATA_DIR"])
W = pathlib.Path(os.environ.get("NFL_BASE_WORK", ".work/base"))
K = ["game_id", "play_id", "nfl_id"]
PK = K[:2]
p = pl.read_csv(
    D / "player_play.csv", null_values=["NA", ""], infer_schema_length=10000
)
g = pl.read_csv(D / "games.csv", null_values=["NA", ""])
bio = pl.read_csv(D / "players.csv", null_values=["NA", ""])
p = p.join(g, on="game_id", validate="m:1").join(
    bio.select("nfl_id", "display_name", "draft_year", "nfl_position"),
    on="nfl_id",
    validate="m:1",
)
p = p.filter(
    pl.col("play_direction").is_in(["left", "right"])
    & (pl.col("play_nullified_by_penalty") == "N")
)
p = p.with_columns((pl.col("game_id") % 5 == 0).alias("validation"))
p = p.with_columns(
    pl.when(pl.col("season") == 2025)
    .then(pl.lit("test"))
    .when(pl.col("validation"))
    .then(pl.lit("validation"))
    .otherwise(pl.lit("train"))
    .alias("split")
)
p = p.with_columns(pl.col("route_ran").is_not_null().alias("route_eligible"))
# Define dropback vs designed run using labels, never as inputs. R is a scramble.
p = p.with_columns(
    pl.when(pl.col("pass_result").is_in(["C", "I", "IN", "S", "R"]))
    .then(1)
    .when(
        pl.col("pass_result").is_null()
        & pl.col("play_description").str.contains(
            r"(?i)(left|right) (end|tackle|guard)|up the middle"
        )
    )
    .then(0)
    .otherwise(None)
    .alias("is_pass")
)
p = p.with_row_index("row_id")
p.write_parquet(W / "context.parquet")
R = p.filter(pl.col("route_eligible")).select("row_id").to_series().to_numpy()
np.save(W / "route_ids.npy", R)
trajectories = []
snaps = []
rp = []
aud = []
for year in [2023, 2024, 2025]:
    q = pl.scan_csv(
        D / f"game_tracking_{year}.csv",
        null_values=["NA", ""],
        infer_schema_length=10000,
    ).with_columns(pl.col("time").str.to_datetime().alias("ts"))
    s = (
        q.filter(pl.col("event") == "ball_snap")
        .group_by(K)
        .agg(pl.col("ts").min().alias("snap_time"), pl.len().alias("snap_count"))
        .collect()
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
        .group_by(K)
        .agg(pl.col("ts").min().alias("end_time"))
        .collect()
    )
    st = s.join(end, on=K, how="left").filter(pl.col("snap_count") == 1)
    z = (
        q.join(st.lazy(), on=K)
        .join(
            p.select(
                K
                + [
                    "row_id",
                    "play_direction",
                    "route_eligible",
                    "team_abbr",
                    "possession_team",
                ]
            ).lazy(),
            on=K,
        )
        .with_columns(
            ((pl.col("ts") - pl.col("snap_time")).dt.total_milliseconds() / 100)
            .round()
            .cast(pl.Int32)
            .alias("frame")
        )
    )
    z = z.filter(
        (pl.col("frame") >= 0) & (pl.col("frame") <= 40) & (pl.col("frame") % 2 == 0)
    ).with_columns(
        pl.when(pl.col("play_direction") == "left")
        .then(120 - pl.col("x"))
        .otherwise(pl.col("x"))
        .alias("nx"),
        pl.when(pl.col("play_direction") == "left")
        .then(160 / 3 - pl.col("y"))
        .otherwise(pl.col("y"))
        .alias("ny"),
        pl.when(pl.col("play_direction") == "left").then(-1).otherwise(1).alias("sign"),
    )
    z = z.with_columns(
        (pl.col("dir") * np.pi / 180)
        .sin()
        .mul(pl.col("s") * pl.col("sign"))
        .alias("vx"),
        (pl.col("dir") * np.pi / 180)
        .cos()
        .mul(pl.col("s") * pl.col("sign"))
        .alias("vy"),
        ((pl.col("end_time").is_null()) | (pl.col("ts") < pl.col("end_time"))).alias(
            "live"
        ),
    )
    z = z.select(
        "row_id",
        "frame",
        "nx",
        "ny",
        "s",
        "a",
        "vx",
        "vy",
        "live",
        "team_abbr",
        "possession_team",
    ).collect()
    dup = z.select(pl.struct(["row_id", "frame"]).is_duplicated().sum()).item()
    assert dup == 0, dup
    snap = z.filter(pl.col("frame") == 0).select(
        "row_id", pl.col("nx").alias("snap_x"), pl.col("ny").alias("snap_y")
    )
    snaps.append(snap)
    z = z.join(snap, on="row_id").with_columns(
        (pl.col("nx") - pl.col("snap_x")).alias("dx"),
        (pl.col("ny") - pl.col("snap_y")).alias("dy"),
    )
    rz = z.filter(pl.col("row_id").is_in(R)).select(
        "row_id", "frame", "nx", "ny", "s", "a", "vx", "vy", "live"
    )
    trajectories.append(rz)
    # Prefix aggregates contain only frames at or before observation, all available cohort players.
    for obs in [0, 10, 20]:
        prefix = z.filter(pl.col("frame") <= obs)
        per = prefix.group_by("row_id").agg(
            pl.col("s").mean().alias("prefix_speed"),
            pl.col("a").mean().alias("prefix_accel"),
        )
        at = (
            z.filter(pl.col("frame") == obs)
            .join(per, on="row_id")
            .join(p.select("row_id", *PK), on="row_id")
            .with_columns(
                (pl.col("team_abbr") == pl.col("possession_team")).alias("offense")
            )
        )
        aa = []
        for side in [True, False]:
            ss = "off" if side else "def"
            part = (
                at.filter(pl.col("offense") == side)
                .group_by(PK)
                .agg(
                    pl.len().alias(ss + "_tracked"),
                    *[
                        pl.col(c).mean().alias(ss + "_" + c)
                        for c in [
                            "s",
                            "a",
                            "vx",
                            "vy",
                            "dx",
                            "dy",
                            "prefix_speed",
                            "prefix_accel",
                        ]
                    ],
                )
            )
            aa.append(part)
        agg = (
            aa[0]
            .join(aa[1], on=PK, how="full", coalesce=True)
            .with_columns(pl.lit(obs / 10).alias("observation"))
        )
        rp.append(agg)
    aud.append(
        {
            "season": year,
            "unique_snap_players": s.height,
            "multiple_snap_players": s.filter(pl.col("snap_count") != 1).height,
            "retained_frames": z.height,
            "duplicate_sampled_keys": dup,
            "route_frames": rz.height,
        }
    )
    print(year, aud[-1], flush=True)
pl.concat(trajectories).write_parquet(W / "trajectories.parquet")
pl.concat(snaps).write_parquet(W / "snaps.parquet")
pl.concat(rp, how="diagonal_relaxed").write_parquet(W / "runpass_tracking.parquet")
# Compare movement measured in the same drill across prospects, not arbitrary mixed drills.
c = pl.read_csv(
    D / "combine_tracking.csv", null_values=["NA", ""], infer_schema_length=10000
).filter(pl.col("drill_type") == "FORTY_YARD_DASH")
a = c.group_by("nfl_id", "event_id", "attempt").agg(
    pl.col("s").quantile(0.95).alias("forty_sensor_speed95"),
    pl.col("a").quantile(0.95).alias("forty_sensor_accel95"),
)
a = a.group_by("nfl_id").agg(
    pl.col("forty_sensor_speed95").median(),
    pl.col("forty_sensor_accel95").median(),
    pl.len().alias("forty_sensor_attempts"),
)
comb = pl.read_csv(
    D / "combine_results.csv", null_values=["NA", ""], infer_schema_length=10000
).join(a, on="nfl_id", how="left")
comb.write_parquet(W / "combine.parquet")
(
    pathlib.Path(os.environ.get("NFL_OUTPUT_DIR", ".work/results"))
    / "cohort-audit.json"
).write_text(json.dumps(aud, indent=2))
print("Prepared", p.height, "player plays and", len(R), "route labels", flush=True)
