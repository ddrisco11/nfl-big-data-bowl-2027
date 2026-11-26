import os

"Fixed-ridge wide target screen with repeated player-held-out predictions and permutation FDR."
import pathlib, json, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore", category=RuntimeWarning)
O = pathlib.Path(os.environ.get("NFL_SCREEN_OUTPUT", ".work/performance-screen"))
D = pathlib.Path(os.environ["NFL_DATA_DIR"])
p = pd.read_parquet(
    str(pathlib.Path(os.environ.get("NFL_BASE_WORK", ".work/base")) / "context.parquet")
)
p = p[p.season_type == "REG"].copy()
c = pd.read_parquet(
    str(pathlib.Path(os.environ.get("NFL_BASE_WORK", ".work/base")) / "combine.parquet")
)
career = pd.read_csv(D / "player_career_successes.csv", na_values=["NA", ""])
movement = pd.read_csv(O / "movement-targets.csv")
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
groups = {
    "WR": ["WR"],
    "TE": ["TE"],
    "CB": ["CB"],
    "Safety": ["FS", "SS"],
    "OL": ["C", "G", "T"],
    "Edge": ["DE", "OLB"],
    "Interior_DL": ["DT", "NT"],
}
rng = np.random.default_rng(2027)
allout = []
targetdata = []
predout = []
metadata = []
for group, positions in groups.items():
    z = p[p.nfl_position.isin(positions)].copy()
    bio = (
        z.groupby("nfl_id")
        .agg(
            player=("display_name", "first"),
            position=("nfl_position", "first"),
            draft_year=("draft_year", "first"),
            recorded_plays=("row_id", "size"),
        )
        .reset_index()
    )
    data = (
        bio.merge(c.drop(columns="draft_year"), on="nfl_id", validate="1:1")
        .merge(career, on="nfl_id", validate="1:1")
        .merge(movement, on="nfl_id", how="left", validate="1:1")
    )
    tasks = []

    def addmean(col, label, category, minimum, subset=None, description=None):
        q = z if subset is None else z[subset]
        v = pd.to_numeric(q[col], errors="coerce")
        q = q[["nfl_id"]].copy()
        q["v"] = v
        g = q.groupby("nfl_id").v.agg(["mean", "count"])
        name = "target_" + label
        data[name] = data.nfl_id.map(g["mean"])
        data[name + "_n"] = data.nfl_id.map(g["count"])
        tasks.append(
            (
                name,
                label,
                category,
                minimum,
                description or f"Mean {col} over nonmissing relevant observations",
            )
        )

    for col, label, cat, minimum in [
        ("separation_at_pass_forward", "separation_at_release", "receiving", 25),
        ("rec_yards", "yards_per_recorded_reception", "receiving", 10),
        ("yards_after_catch", "yards_after_catch", "receiving", 10),
        ("expected_yards_after_catch", "expected_yac", "receiving", 10),
        ("cushion", "starting_cushion", "deployment", 25),
        ("player_get_off", "pass_rush_getoff_time", "pass_rush", 25),
        ("time_to_pressure", "time_to_pressure_when_recorded", "pass_rush", 10),
        ("time_to_qb_hurry", "time_to_hurry_when_recorded", "pass_rush", 10),
        (
            "peak_pressure_probability_allowed",
            "peak_pressure_probability_allowed",
            "blocking",
            25,
        ),
        ("pass_rushers_encountered", "pass_rushers_encountered", "blocking", 25),
        (
            "time_to_pressure_allowed",
            "time_to_pressure_allowed_when_recorded",
            "blocking",
            10,
        ),
    ]:
        addmean(col, label, cat, minimum)
    z["yac_over_expected"] = z.yards_after_catch - z.expected_yards_after_catch
    addmean("yac_over_expected", "yac_above_expected", "receiving", 10)
    for col, label, cat in [
        ("pressure_allowed", "pressure_allowed_rate", "blocking"),
        ("sack_allowed", "sack_allowed_rate", "blocking"),
        ("extended_sack_allowed", "extended_sack_allowed_rate", "blocking"),
        ("sack", "sack_credit_per_recorded_pass_rush_observation", "pass_rush"),
        ("quick_pressure", "quick_pressure_given_recorded_pressure", "pass_rush"),
    ]:
        addmean(col, label, cat, 25)
    addmean(
        "target",
        "target_rate_on_routes",
        "receiving",
        25,
        subset=z.route_ran.notna(),
        description="Mean target flag among recorded receiver routes",
    )
    addmean(
        "tackle",
        "solo_tackle_share_of_recorded_tackle_events",
        "defending",
        25,
        description="Mean tackle flag among nonmissing tackle-event rows; not tackle rate per defensive snap",
    )
    for col in [
        "career_offensive_snaps",
        "career_defensive_snaps",
        "career_special_teams_snaps",
        "career_games_active",
        "career_games_started",
    ]:
        label = col
        name = "target_" + label
        data[name] = pd.to_numeric(data[col], errors="coerce")
        data[name + "_n"] = data.recorded_plays
        tasks.append(
            (
                name,
                label,
                "playing_time",
                25,
                "Career total; confounded by time since draft",
            )
        )
    for col in [
        "career_offensive_snaps",
        "career_defensive_snaps",
        "career_special_teams_snaps",
        "career_games_started",
    ]:
        label = col + "_per_active_game"
        name = "target_" + label
        data[name] = pd.to_numeric(data[col], errors="coerce") / pd.to_numeric(
            data.career_games_active, errors="coerce"
        ).replace(0, np.nan)
        data[name + "_n"] = data.career_games_active
        tasks.append(
            (
                name,
                label,
                "playing_time",
                5,
                "Career ratio per active game; does not equal per-season efficiency",
            )
        )
    for col in [
        "game_mean_speed",
        "game_speed95",
        "game_speed99",
        "game_acceleration95",
    ]:
        name = "target_" + col
        data[name] = data[col]
        data[name + "_n"] = data.movement_frames
        tasks.append(
            (
                name,
                col,
                "movement",
                250,
                "Tracking summary from snap through release/end, capped at 3 seconds; all recorded frames pooled",
            )
        )
    for target, label, category, minimum, description in tasks:
        a = data[(data[target + "_n"] >= minimum) & data[target].notna()].reset_index(
            drop=True
        )
        n = len(a)
        if n < 30 or a[target].nunique() < 6 or a[target].std() < 1e-08:
            continue
        X = a[features]
        y = a[target].to_numpy(float)
        matrices = []
        base = []
        for rep in range(3):
            H = np.zeros((n, n))
            B = np.zeros((n, n))
            for tr, te in KFold(5, shuffle=True, random_state=2027 + rep).split(a):
                prep = make_pipeline(
                    SimpleImputer(
                        strategy="median", add_indicator=True, keep_empty_features=True
                    ),
                    StandardScaler(),
                )
                xx = prep.fit_transform(X.iloc[tr])
                xt = prep.transform(X.iloc[te])
                m = Ridge(alpha=10).fit(xx, np.eye(n)[tr])
                H[te] = m.predict(xt)
                meta = pd.get_dummies(
                    a[["position", "draft_year"]].astype(str), dtype=float
                )
                ps = StandardScaler()
                xx = ps.fit_transform(meta.iloc[tr])
                xt = ps.transform(meta.iloc[te])
                b = Ridge(alpha=10).fit(xx, np.eye(n)[tr])
                B[te] = b.predict(xt)
            matrices.append(H)
            base.append(B)
        preds = np.stack([H @ y for H in matrices])
        rs = [float(pearsonr(y, q).statistic) for q in preds]
        r2 = [float(r2_score(y, q)) for q in preds]
        rho = [float(spearmanr(y, q).statistic) for q in preds]
        rmse = [float(np.sqrt(mean_squared_error(y, q))) for q in preds]
        base_r2 = [float(r2_score(y, H @ y)) for H in base]
        ext = []
        for rep in range(3):
            q = np.empty(n)
            for tr, te in KFold(5, shuffle=True, random_state=2027 + rep).split(a):
                meta = pd.get_dummies(
                    a[["position", "draft_year"]].astype(str), dtype=float
                )
                vals = np.column_stack([X.to_numpy(), meta.to_numpy()])
                prep = make_pipeline(
                    SimpleImputer(
                        strategy="median", add_indicator=True, keep_empty_features=True
                    ),
                    StandardScaler(),
                )
                xx = prep.fit_transform(vals[tr])
                xt = prep.transform(vals[te])
                q[te] = Ridge(alpha=10).fit(xx, y[tr]).predict(xt)
            ext.append(r2_score(y, q))
        permutations = np.stack([rng.permutation(y) for _ in range(1000)], axis=1)
        null = []
        for H in matrices:
            v = H @ permutations
            yc = permutations - permutations.mean(0)
            vc = v - v.mean(0)
            null.append(
                np.sum(yc * vc, axis=0)
                / np.maximum(
                    np.sqrt(np.sum(yc**2, axis=0) * np.sum(vc**2, axis=0)), 1e-12
                )
            )
        null = np.mean(null, axis=0)
        pvalue = float((1 + np.sum(null >= np.mean(rs))) / 1001)
        row = {
            "group": group,
            "target": label,
            "category": category,
            "players": n,
            "minimum_observations": minimum,
            "description": description,
            "mean_pearson": float(np.mean(rs)),
            "repeat_pearson_min": min(rs),
            "repeat_pearson_max": max(rs),
            "mean_spearman": float(np.mean(rho)),
            "mean_r2": float(np.mean(r2)),
            "mean_rmse": float(np.mean(rmse)),
            "metadata_only_r2": float(np.mean(base_r2)),
            "combine_plus_metadata_r2": float(np.mean(ext)),
            "incremental_r2_vs_metadata": float(np.mean(ext) - np.mean(base_r2)),
            "permutation_p": pvalue,
        }
        allout.append(row)
        for i, nfl in enumerate(a.nfl_id):
            for rep in range(3):
                predout.append(
                    {
                        "group": group,
                        "target": label,
                        "nfl_id": int(nfl),
                        "repeat": rep,
                        "actual": float(y[i]),
                        "prediction": float(preds[rep, i]),
                    }
                )
        print(
            group,
            label,
            n,
            "r",
            round(row["mean_pearson"], 3),
            "R2",
            round(row["mean_r2"], 3),
            flush=True,
        )
    data.to_csv(O / (group + "-targets.csv"), index=False)
q = pd.DataFrame(allout)
order = np.argsort(q.permutation_p.to_numpy())
pv = q.permutation_p.to_numpy()[order]
fdr = np.minimum.accumulate((pv * len(q) / np.arange(1, len(q) + 1))[::-1])[::-1]
q["fdr_q"] = 1.0
q.loc[order, "fdr_q"] = np.minimum(fdr, 1)
q = q.sort_values("mean_pearson", ascending=False)
q.to_csv(O / "ranked-targets.csv", index=False)
pd.DataFrame(predout).to_csv(O / "predictions.csv", index=False)
summary = {
    "tests": len(q),
    "features": features,
    "model": "Ridge alpha=10 fixed before screening; no target-dependent tuning",
    "evaluation": "Three repeats of five-fold player-held-out evaluation; imputation, missingness indicators and scaling fitted only on training players",
    "permutations": 1000,
    "multiple_testing": "One-sided positive-correlation permutation p-values, BH correction across every tested position-group/target pair",
    "ranked_targets": q.to_dict("records"),
    "notes": [
        "Exploratory target search. Ranking and reporting the best targets introduces selection effects even with FDR; independent confirmation is required.",
        "Permutation inference assumes exchangeable player outcomes; team/scheme and class effects may violate this. Metadata baselines expose obvious position/draft-class effects but are not causal controls.",
        "Minimum sample is 30 players, with target-specific minimum observations. No missing game outcome is silently replaced with zero.",
        "Movement measures test physical translation and are not equivalent to football success. Starting cushion is deployment context. Career totals depend strongly on follow-up since draft.",
        "Event-conditional means/rates use only nonmissing relevant observations; they do not represent full-snap rates.",
        "All targets use regular-season 2023–2025 or supplied career summaries. The same previously inspected development cohort is reused.",
        "No DB separation outcome or verified matchup is available. Defensive outcomes are limited to available event fields and movement.",
    ],
}
(O / "results.json").write_text(json.dumps(summary, indent=2))
print("SCREEN COMPLETE", len(q), flush=True)
