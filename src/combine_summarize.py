import json, pathlib
import os
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

O = pathlib.Path(os.environ.get("NFL_COMBINE_OUTPUT", ".work/combine-upgrades"))
p = pd.read_csv(O / "predictions.csv")
rng = np.random.default_rng(2027)


def metrics(z):
    y = z.truth.to_numpy()
    v = z.prediction.to_numpy()
    return {
        "r2": float(r2_score(y, v)),
        "pearson": float(pearsonr(y, v).statistic) if np.std(v) > 0 else None,
        "spearman": float(spearmanr(y, v).statistic) if np.std(v) > 0 else None,
        "rmse": float(np.sqrt(mean_squared_error(y, v))),
        "mae": float(mean_absolute_error(y, v)),
        "baseline_rmse": float(np.sqrt(mean_squared_error(y, z.baseline))),
    }


summary = []
for (kind, outcome, config), g in p.groupby(["kind", "outcome", "configuration"]):
    parts = g.groupby("repeat" if kind == "repeated" else "fold")
    scores = [{"split": int(k), **metrics(z)} for k, z in parts]
    row = {
        "kind": kind,
        "outcome": outcome,
        "configuration": config,
        "players": int(g.nfl_id.nunique()),
        "splits": scores,
    }
    if kind == "repeated":
        row["mean_metrics"] = {
            k: float(np.nanmean([s[k] if s[k] is not None else np.nan for s in scores]))
            for k in ["r2", "pearson", "spearman", "rmse", "mae", "baseline_rmse"]
        }
        row["repeat_r2_range"] = [
            min((s["r2"] for s in scores)),
            max((s["r2"] for s in scores)),
        ]
        ids = sorted(g.nfl_id.unique())
        arr = []
        for rep, z in g.groupby("repeat"):
            z = z.set_index("nfl_id").loc[ids]
            arr.append(np.column_stack([z.truth, z.prediction]))
        arr = np.stack(arr)
        draw = rng.integers(0, len(ids), (1000, len(ids)))
        boots = []
        for idx in draw:
            a = arr[:, idx, 0]
            b = arr[:, idx, 1]
            sst = ((a - a.mean(1, keepdims=True)) ** 2).sum(1)
            boots.append(np.mean(1 - ((a - b) ** 2).sum(1) / sst))
        row["conditional_player_bootstrap_r2_ci95"] = np.quantile(
            boots, [0.025, 0.975]
        ).tolist()
    summary.append(row)
pairs = []
for outcome in ["raw", "adjusted"]:
    base = p[
        (p.kind == "repeated")
        & (p.outcome == outcome)
        & (p.configuration == "ridge|official|values|pool=False|min=1")
    ]
    for config in [
        "ridge|sensors|values|pool=False|min=1",
        "ridge|interactions|values|pool=False|min=1",
        "ridge|official|values|pool=True|min=1",
        "ridge|official|no_flags|pool=False|min=1",
        "ridge|official|flags|pool=False|min=1",
    ]:
        z = p[
            (p.kind == "repeated")
            & (p.outcome == outcome)
            & (p.configuration == config)
        ].merge(
            base,
            on=["nfl_id", "repeat", "fold"],
            suffixes=("", "_base"),
            validate="1:1",
        )
        delta = (z.truth - z.prediction) ** 2 - (z.truth_base - z.prediction_base) ** 2
        d = (
            pd.DataFrame({"id": z.nfl_id, "delta": delta})
            .groupby("id")
            .delta.mean()
            .to_numpy()
        )
        boot = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
        pairs.append(
            {
                "outcome": outcome,
                "alternative": config,
                "mean_mse_change_vs_official_ridge": float(d.mean()),
                "conditional_player_bootstrap_ci95": np.quantile(
                    boot, [0.025, 0.975]
                ).tolist(),
            }
        )
result = {
    "data": json.loads((O / "data-audit.json").read_text()),
    "evaluation": summary,
    "paired_comparisons": pairs,
    "notes": [
        "Three repetitions of five-fold player-held-out outer evaluation. Three-fold inner tuning; same outer partitions for every configuration. No player outcomes enter both outer fitting and evaluation.",
        "Context target models are trained on outer-training players, with player-cross-fitted residuals for fitting targets and outer-training-only context predictions for evaluation players. Context models use route, coverage, formation, alignment, season, cushion, event-derived time to throw, down, distance, yard line, quarter, sideline distance.",
        "All configuration comparisons use raw held-out player averages or held-out context residual averages as truth, never shrunken held-out truth. Partial pooling shrinks training targets using outer-training between-player and within-player variances; the nuisance and pooling construction is fixed within inner tuning.",
        "Intervals resample players from fixed out-of-fold predictions. They do not refit models or account for shared games/teams and are exploratory. Repeat ranges describe split sensitivity, not confidence intervals.",
        "Draft-class tests train on earlier draft classes and evaluate 2024 or 2025. NFL outcomes use all available 2023–2025 careers, so these are class-transfer tests, not historically executable forecasts with equal follow-up.",
        "Nonparticipants and missing measurements are not random. Compare official values plus flags, values without flags, and flags alone. Sensor missingness may encode drill participation.",
        "Interactions were specified before evaluation: standardized speed95 × curl braking90; acceleration95 × slant speed retention; weight × shuttle time. Signs are not constrained. No causal interpretation is claimed.",
        "2025 has been examined repeatedly. These are exploratory analyses with no pristine confirmatory holdout. Model/configuration selection after seeing these results needs an independent confirmation.",
    ],
}
(O / "results.json").write_text(json.dumps(result, indent=2))
pd.json_normalize(summary).to_csv(O / "model-comparison.csv", index=False)
print("SUMMARY COMPLETE", len(summary))
for r in summary:
    if r["kind"] == "repeated" and "min=1" in r["configuration"]:
        print(
            r["outcome"],
            r["configuration"],
            "R2",
            round(r["mean_metrics"]["r2"], 3),
            "r",
            round(r["mean_metrics"]["pearson"], 3),
            "CI",
            np.round(r["conditional_player_bootstrap_r2_ci95"], 3),
            flush=True,
        )
