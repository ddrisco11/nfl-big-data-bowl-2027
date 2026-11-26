"""Audit exploratory candidate-defender movement lags without claiming known matchups."""

import argparse, json, pathlib
import numpy as np, pandas as pd, polars as pl
from rei_core import response_onset

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--base", default=".work/base")
p.add_argument("--work", default=".work/models")
p.add_argument("--output", default=".work/rei")
args = p.parse_args()
D = pathlib.Path(args.data)
O = pathlib.Path(args.output)
O.mkdir(parents=True, exist_ok=True)
r = pd.read_parquet(pathlib.Path(args.work) / "routes.parquet")
a = np.load(pathlib.Path(args.work) / "sequences.npz")
S = a["motion"]
live = a["live"]
meta = pd.read_parquet(pathlib.Path(args.base) / "context.parquet")
r["array_index"] = np.arange(len(r))
r = r[r.split == "test"]
defenders = meta[
    (meta.season == 2025)
    & (meta.season_type == "REG")
    & (meta.team_abbr != meta.possession_team)
    & (meta.coverage_assignment == "MAN")
    & meta.nfl_position.isin(["CB", "SS", "FS", "ILB", "OLB"])
]
keys = pl.from_pandas(defenders[["game_id", "play_id"]].drop_duplicates())
print("Candidate man-defender records", len(defenders), "plays", len(keys), flush=True)
q = (
    pl.scan_csv(
        D / "game_tracking_2025.csv", null_values=["NA", ""], infer_schema_length=10000
    )
    .join(keys.lazy(), on=["game_id", "play_id"])
    .with_columns(pl.col("time").str.to_datetime().alias("ts"))
    .collect()
    .to_pandas()
)
groups = {(g, p): z for (g, p), z in q.groupby(["game_id", "play_id"])}
defgroups = {(g, p): z for (g, p), z in defenders.groupby(["game_id", "play_id"])}
routegroups = {(g, p): z for (g, p), z in r.groupby(["game_id", "play_id"])}
counts = {
    "eligible_test_routes": len(r),
    "routes_in_plays_with_recorded_man_defender": 0,
    "detected_receiver_breaks": 0,
    "no_candidate_with_complete_prebreak_tracking": 0,
    "candidate_too_far": 0,
    "ambiguous_among_recorded_candidates": 0,
    "another_recorded_receiver_closer": 0,
    "accepted_candidate_pairs": 0,
}
records = []
for key, rr in routegroups.items():
    if key not in groups:
        continue
    play = groups[key]
    dm = defgroups[key]
    for row in rr.itertuples():
        counts["routes_in_plays_with_recorded_man_defender"] += 1
        i = row.array_index
        f = S[i]
        breakj = None
        # Retrospective 0.2 s persistence confirmation; no claim of online onset detection.
        for j in range(15, 36):
            if (
                not live[i, j - 5 : j + 3].all()
                or not np.isfinite(f[j - 5 : j + 3]).all()
            ):
                continue
            before = f[j - 3 : j, 4:6].mean(0)
            after = f[j : j + 3, 4:6].mean(0)
            if np.linalg.norm(before) < 3 or np.linalg.norm(after) < 2:
                continue
            angle = np.arctan2(
                before[0] * after[1] - before[1] * after[0], np.dot(before, after)
            )
            each = np.arctan2(
                before[0] * f[j : j + 3, 5] - before[1] * f[j : j + 3, 4],
                f[j : j + 3, 4:6] @ before,
            )
            if abs(angle) >= np.deg2rad(35) and np.all(
                np.sign(angle) * each >= np.deg2rad(25)
            ):
                breakj = j
                break
        if breakj is None:
            continue
        counts["detected_receiver_breaks"] += 1
        j = breakj
        own = play[play.nfl_id == row.nfl_id]
        snap = own.loc[own.event == "ball_snap", "ts"]
        if len(snap) != 1:
            continue
        snap = snap.iloc[0]
        candidates = []
        for defender in dm.itertuples():
            z = play[play.nfl_id == defender.nfl_id].copy()
            ticks = (z.ts - snap).dt.total_seconds().to_numpy() * 10
            good = np.isclose(ticks, np.round(ticks)) & (ticks >= -10) & (ticks <= 40)
            z = z.loc[good]
            ticks = np.round(ticks[good]).astype(int) + 10
            track = np.full((51, 4), np.nan)
            sign = -1 if row.play_direction == "left" else 1
            xx = 120 - z.x.to_numpy() if sign == -1 else z.x.to_numpy()
            yy = 160 / 3 - z.y.to_numpy() if sign == -1 else z.y.to_numpy()
            rad = np.deg2rad(z.dir.to_numpy())
            track[ticks] = np.column_stack(
                [
                    xx,
                    yy,
                    sign * z.s.to_numpy() * np.sin(rad),
                    sign * z.s.to_numpy() * np.cos(rad) * row.mirror,
                ]
            )
            track[:, 0] -= row.snap_x
            track[:, 1] = (track[:, 1] - row.snap_y) * row.mirror
            if not np.isfinite(track[j - 5 : j + 1]).all():
                continue
            distance = float(np.linalg.norm(track[j, :2] - f[j, :2]))
            candidates.append((distance, int(defender.nfl_id), track))
        candidates.sort(key=lambda x: x[0])
        if not candidates:
            counts["no_candidate_with_complete_prebreak_tracking"] += 1
            continue
        dist, nfl, track = candidates[0]
        if dist > 4:
            counts["candidate_too_far"] += 1
            continue
        if len(candidates) > 1 and candidates[1][0] - dist < 1:
            counts["ambiguous_among_recorded_candidates"] += 1
            continue
        # Exclude pairings that are ambiguous even within the incomplete recorded receiver cohort.
        absdef = np.array(
            [row.snap_x + track[j, 0], row.snap_y + track[j, 1] * row.mirror]
        )
        other = []
        for mate in rr.itertuples():
            if mate.nfl_id == row.nfl_id:
                continue
            mf = S[mate.array_index, j, :2]
            if np.isfinite(mf).all():
                other.append(
                    np.linalg.norm(
                        absdef
                        - [mate.snap_x + mf[0], mate.snap_y + mf[1] * mate.mirror]
                    )
                )
        if other and min(other) < dist + 0.75:
            counts["another_recorded_receiver_closer"] += 1
            continue
        counts["accepted_candidate_pairs"] += 1
        # Lateral velocity change toward the receiver's break, relative to defender prebreak baseline.
        side = (
            np.array([-before[1], before[0]]) / np.linalg.norm(before) * np.sign(angle)
        )
        baseline = track[j - 5 : j - 2, 2:4].mean(0)
        signal = (track[:, 2:4] - baseline) @ side
        last = j - 1
        for k in range(j, min(j + 13, 51)):
            if (
                not live[i, k]
                or not np.isfinite(track[k]).all()
                or not np.isfinite(f[k]).all()
            ):
                break
            last = k
        for threshold in [0.75, 1.0, 1.25]:
            onset = response_onset(signal, j, last, threshold)
            status = (
                "movement response observed" if onset is not None else "right censored"
            )
            lag = (onset - j) * 0.1 if onset is not None else None
            records.append(
                {
                    "row_id": int(row.row_id),
                    "game_id": int(row.game_id),
                    "play_id": int(row.play_id),
                    "receiver_id": int(row.nfl_id),
                    "candidate_defender_id": nfl,
                    "break_seconds": (j - 10) * 0.1,
                    "candidate_distance_yards": dist,
                    "recorded_man_candidates": len(candidates),
                    "threshold_yards_per_second": threshold,
                    "response_lag_seconds": lag,
                    "confirmation_seconds_after_onset": 0.2,
                    "last_observed_lag_seconds": max(0, (last - j) * 0.1),
                    "status": status,
                    "pairing": "inferred candidate; actual assignment unknown",
                }
            )
res = pd.DataFrame(records)
res.to_csv(O / "candidate-defender-responses.csv", index=False)
summary = []
for th in [0.75, 1.0, 1.25]:
    z = res[res.threshold_yards_per_second == th] if len(res) else pd.DataFrame()
    known = z.dropna(subset=["response_lag_seconds"]) if len(z) else z
    summary.append(
        {
            "threshold_yards_per_second": th,
            "candidate_pairs": len(z),
            "observed_responses": len(known),
            "censored": len(z) - len(known),
            "anticipatory_or_simultaneous": int((known.response_lag_seconds <= 0).sum())
            if len(known)
            else 0,
            "median_lag_among_observed_seconds": float(
                known.response_lag_seconds.median()
            )
            if len(known)
            else None,
        }
    )
result = {
    "name": "Candidate-defender movement response lag",
    "status": "exploratory proxy; not identified defender reaction time",
    "counts": counts,
    "threshold_sensitivity": summary,
    "notes": [
        "Coverage assignment MAN names a role, not a receiver-defender matchup. Actual responsibility cannot be recovered from these fields.",
        "Nearest available MAN candidate within 4 yards at receiver break; require a 1-yard margin over other recorded MAN candidates and no other recorded receiver within candidate distance +0.75 yards. Unrecorded players can invalidate every pairing.",
        "Break: 35-degree mean heading change, three-frame persistence, prebreak speed >=3 and postbreak speed >=2 yards/second. Candidate onset is retrospective and confirmed 0.2 seconds later.",
        "Response: sustained change in candidate defender lateral velocity toward the receiver break relative to a prebreak baseline. This can reflect anticipation, shared play geometry, or a different assignment, not a causal response.",
        "Search lags -0.2 to +1.0 seconds; three-frame persistence. Unobserved responses are censored, never assigned a fabricated 1-second delay.",
        "No receiver quality ranking or causal Combine association is inferred from this proxy. It is kept separate from REI.",
    ],
}
(O / "reaction-summary.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
