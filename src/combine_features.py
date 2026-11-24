"""Extract matched Combine route-drill speed and turning features."""

import polars as pl
from shared import *
from common import DATA, OLD

# Same named drills compared across WRs; no pooling incomparable drill sequences.
c = pl.read_csv(
    DATA / "combine_tracking.csv", null_values=["NA", ""], infer_schema_length=10000
).to_pandas()
attempts = []
for drill, label in [("SLANT_ROUTE_LEFT", "slant"), ("CURL_ROUTE_RIGHT", "curl")]:
    z = c[c.drill_name == drill]
    for (nfl, event, attempt), a in z.groupby(["nfl_id", "event_id", "attempt"]):
        a = a.sort_values("time")
        t = (
            pd.to_datetime(a.time).astype("datetime64[ns]").astype("int64").to_numpy()
            / 1e9
        )
        s = a.s.to_numpy()
        ang = a.dir.to_numpy() * np.pi / 180
        if len(a) < 9:
            continue
        dt = t[2:] - t[:-2]
        turn = (
            np.abs(np.arctan2(np.sin(ang[2:] - ang[:-2]), np.cos(ang[2:] - ang[:-2])))
            / np.maximum(dt, 0.01)
            * 180
            / np.pi
        )
        dist = np.cumsum(np.r_[0, np.hypot(np.diff(a.x), np.diff(a.y))])
        mid = dist[1:-1] / max(dist[-1], 0.01)
        valid = (dt > 0.15) & (dt < 0.25) & (s[1:-1] > 2) & (mid > 0.15) & (mid < 0.8)
        if valid.sum() < 3:
            continue
        k = np.flatnonzero(valid)[np.argmax(turn[valid])] + 1
        pre = np.mean(s[max(0, k - 3) : k])
        post = np.mean(s[k + 1 : min(len(s), k + 4)])
        attempts.append(
            {
                "nfl_id": int(nfl),
                "drill": label,
                "speed95": float(np.quantile(s, 0.95)),
                "turn_rate90": float(np.quantile(turn[valid], 0.9)),
                "speed_at_turn": float(s[k]),
                "speed_retention": float(post / max(pre, 0.1)),
            }
        )
a = pd.DataFrame(attempts)
a.to_csv(O / "combine-drill-attempts.csv", index=False)
f = (
    a.groupby(["nfl_id", "drill"])[
        ["speed95", "turn_rate90", "speed_at_turn", "speed_retention"]
    ]
    .median()
    .unstack("drill")
)
f.columns = [f"{d}_{x}" for x, d in f.columns]
f = f.reset_index()
comb = (
    pl.read_parquet(OLD / "combine.parquet")
    .to_pandas()
    .merge(f, on="nfl_id", how="left")
)
comb.to_csv(O / "combine-features.csv", index=False)
