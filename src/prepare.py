"""Extract native 10 Hz causal sequences and training-only historical tendencies."""

import os
import sys, json, pathlib

from common import load_routes, DATA, OLD, CLASSES
import polars as pl, numpy as np, pandas as pd

W = pathlib.Path(os.environ.get("NFL_MODEL_WORK", ".work/models"))
O = pathlib.Path(os.environ.get("NFL_OUTPUT_DIR", ".work/results"))
D, A, L = load_routes()
D = D[D.season_type == "REG"].reset_index(drop=True)
D.to_parquet(W / "routes.parquet", index=False)
K = ["game_id", "play_id", "nfl_id"]
PK = K[:2]
p = pl.read_parquet(OLD / "context.parquet")
target = pl.from_pandas(
    D[["row_id", "game_id", "play_id", "nfl_id", "snap_x", "snap_y", "mirror"]]
)
N = len(D)
lookup = pd.Series(np.arange(N), index=D.row_id)
S = np.full((N, 51, 6), np.nan, dtype=np.float32)
C = np.full((N, 51, 16), np.nan, dtype=np.float32)
live = np.zeros((N, 51), bool)
audit = []
for year in [2023, 2024, 2025]:
    q = pl.scan_csv(
        DATA / f"game_tracking_{year}.csv",
        null_values=["NA", ""],
        infer_schema_length=10000,
    ).with_columns(pl.col("time").str.to_datetime().alias("ts"))
    snaps = (
        q.filter(pl.col("event") == "ball_snap")
        .group_by(K)
        .agg(pl.col("ts").min().alias("snap_time"), pl.len().alias("snap_count"))
        .collect()
        .filter(pl.col("snap_count") == 1)
    )
    ambiguous = (
        snaps.group_by(PK)
        .agg(pl.col("snap_time").n_unique().alias("n"))
        .filter(pl.col("n") > 1)
    )
    ends = (
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
    z = (
        q.join(snaps.lazy(), on=K)
        .join(ends.lazy(), on=K, how="left")
        .join(
            p.filter(pl.col("season_type") == "REG")
            .select(K + ["row_id", "play_direction", "team_abbr", "possession_team"])
            .lazy(),
            on=K,
        )
        .with_columns(
            ((pl.col("ts") - pl.col("snap_time")).dt.total_milliseconds() / 100).alias(
                "tick"
            )
        )
    )
    z = z.filter(
        (pl.col("tick") >= -10)
        & (pl.col("tick") <= 40)
        & ((pl.col("tick") - pl.col("tick").round()).abs() < 1e-6)
    ).with_columns(
        pl.col("tick").cast(pl.Int32),
        pl.when(pl.col("play_direction") == "left")
        .then(120 - pl.col("x"))
        .otherwise(pl.col("x"))
        .alias("nx"),
        pl.when(pl.col("play_direction") == "left")
        .then(160 / 3 - pl.col("y"))
        .otherwise(pl.col("y"))
        .alias("ny"),
        pl.when(pl.col("play_direction") == "left").then(-1).otherwise(1).alias("sign"),
        (pl.col("team_abbr") == pl.col("possession_team")).alias("offense"),
    )
    z = (
        z.with_columns(
            (pl.col("dir") * np.pi / 180)
            .sin()
            .mul(pl.col("s") * pl.col("sign"))
            .alias("vx"),
            (pl.col("dir") * np.pi / 180)
            .cos()
            .mul(pl.col("s") * pl.col("sign"))
            .alias("vy"),
            (pl.col("end_time").is_null() | (pl.col("ts") < pl.col("end_time"))).alias(
                "live"
            ),
        )
        .select(
            *PK, "row_id", "tick", "nx", "ny", "s", "a", "vx", "vy", "offense", "live"
        )
        .collect()
    )
    assert z.select(pl.struct(["row_id", "tick"]).is_duplicated().sum()).item() == 0
    # Sum at the same frame. Subtract focal player from offense to avoid duplicating own motion.
    groups = z.group_by(PK + ["tick", "offense"]).agg(
        pl.len().alias("n"),
        *[pl.col(c).sum().alias(c + "_sum") for c in ["nx", "ny", "s", "vx", "vy"]],
        (pl.col("nx") ** 2).sum().alias("nx2_sum"),
        (pl.col("ny") ** 2).sum().alias("ny2_sum"),
    )
    r = z.join(target, on=K[:2] + ["row_id"], how="inner").to_pandas()
    ind = lookup.loc[r.row_id].to_numpy()
    j = (r.tick + 10).to_numpy(int)
    for c, col in enumerate(["nx", "ny", "s", "a", "vx", "vy"]):
        a = r[col].to_numpy()
        a = a - r.snap_x if col == "nx" else a - r.snap_y if col == "ny" else a
        a = a * r.mirror if col in ["ny", "vy"] else a
        S[ind, j, c] = a
    live[ind, j] = r.live
    for side in [True, False]:
        g = groups.filter(pl.col("offense") == side).drop("offense").to_pandas()
        b = r.merge(g, on=PK + ["tick"], how="left", validate="m:1")
        n = b.n.fillna(0).to_numpy() - int(side)
        n = np.maximum(n, 0)
        means = []
        for col in ["nx", "ny", "s", "vx", "vy"]:
            total = b[col + "_sum"].fillna(0).to_numpy() - (
                b[col].to_numpy() if side else 0
            )
            mean = np.divide(
                total, n, out=np.zeros_like(total, dtype=float), where=n > 0
            )
            if col == "nx":
                mean -= b.nx.to_numpy()
            if col == "ny":
                mean = (mean - b.ny.to_numpy()) * b.mirror.to_numpy()
            if col == "vy":
                mean *= b.mirror.to_numpy()
            mean[n == 0] = 0
            means.append(mean)
        spreads = []
        for col in ["nx", "ny"]:
            sm = b[col + "_sum"].fillna(0).to_numpy() - (
                b[col].to_numpy() if side else 0
            )
            ss = b[col + "2_sum"].fillna(0).to_numpy() - (
                b[col].to_numpy() ** 2 if side else 0
            )
            den = np.maximum(n, 1)
            spreads.append(np.sqrt(np.maximum(ss / den - (sm / den) ** 2, 0)))
        block = np.column_stack([n, *means, *spreads])
        start = 0 if side else 8
        C[ind, j, start : start + 8] = block
    audit.append(
        {
            "season": year,
            "sampled_rows": z.height,
            "route_frames": r.shape[0],
            "plays_with_inconsistent_snap_times": ambiguous.height,
        }
    )
    print(audit[-1], flush=True)
    # Context from ambiguous snap plays is excluded instead of mixing different absolute times.
    if ambiguous.height:
        bad = set(map(tuple, ambiguous.select(PK).rows()))
        mask = np.array([(g, p) in bad for g, p in zip(D.game_id, D.play_id)])
        C[mask] = np.nan
# Earlier-game historical tendencies: never current game or later games, and never held-out labels.
H = np.zeros((N, 29), np.float32)
records = D.sort_values(["game_id", "play_id"])
player = {}
team = {}
global_counts = np.ones(12)
for game, group in records.groupby("game_id", sort=True):
    for idx, row in group.iterrows():
        pc = player.get(row.nfl_id, np.zeros(12))
        tc = team.get(row.team_abbr, np.zeros(12))
        prior = global_counts / global_counts.sum()
        H[idx, :12] = (pc + 20 * prior) / (pc.sum() + 20)
        H[idx, 12:24] = (tc + 50 * prior) / (tc.sum() + 50)
        H[idx, 24:] = [
            np.log1p(pc.sum()),
            np.log1p(tc.sum()),
            pc.sum() == 0,
            tc.sum() == 0,
            np.log1p(global_counts.sum() - 12),
        ]
    # Only training-game labels update the store. All validation/test label histories are frozen out.
    for idx, row in group[group.split == "train"].iterrows():
        k = CLASSES.index(row.route_ran)
        player.setdefault(row.nfl_id, np.zeros(12))[k] += 1
        team.setdefault(row.team_abbr, np.zeros(12))[k] += 1
        global_counts[k] += 1
np.savez_compressed(W / "sequences.npz", motion=S, context=C, live=live, history=H)
families = {
    "Vertical": ["GO", "WHEEL"],
    "In-breaking": ["IN", "POST", "SLANT"],
    "Out-breaking": ["OUT", "CORNER"],
    "Crossing": ["CROSS"],
    "Stop": ["HITCH"],
    "Underneath": ["ANGLE", "FLAT", "SCREEN"],
}
(O / "family-map.json").write_text(json.dumps(families, indent=2))
(O / "data-audit.json").write_text(
    json.dumps(
        {
            "counts": audit,
            "route_rows": N,
            "pre_snap_full_prefix_fraction": float(
                np.isfinite(S[:, :11]).all(axis=(1, 2)).mean()
            ),
            "history_policy": "Earlier training games only; current-game, validation-game and test-game labels never update priors.",
            "family_note": "Analyst-defined coarse labels; not independent ground truth. WHEEL and POST memberships are explicit modeling choices.",
        },
        indent=2,
    )
)
print("PREPARATION COMPLETE", N, flush=True)
