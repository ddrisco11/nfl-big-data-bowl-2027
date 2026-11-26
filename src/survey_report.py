import os
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nfl-matplotlib")
import pathlib, json
import numpy as np, pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

O = pathlib.Path(os.environ.get("NFL_SCREEN_OUTPUT", ".work/performance-screen"))
r = json.loads((O / "results.json").read_text())
q = pd.read_csv(O / "ranked-targets.csv")
p = pd.read_csv(O / "predictions.csv")
rng = np.random.default_rng(2027)
follow = []
for row in q.head(12).itertuples():
    z = p[(p.group == row.group) & (p.target == row.target)]
    ids = sorted(z.nfl_id.unique())
    arr = np.stack(
        [
            g.set_index("nfl_id").loc[ids][["actual", "prediction"]].to_numpy()
            for _, g in z.groupby("repeat")
        ]
    )
    draw = rng.integers(0, len(ids), (1000, len(ids)))
    boot = []
    for ix in draw:
        vals = arr[:, ix]
        a = vals[:, :, 0]
        b = vals[:, :, 1]
        a -= a.mean(1, keepdims=True)
        b -= b.mean(1, keepdims=True)
        boot.append(
            np.mean(
                np.sum(a * b, axis=1)
                / np.maximum(
                    np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1)), 1e-10
                )
            )
        )
    data = pd.read_csv(O / (row.group + "-targets.csv"))
    target = "target_" + row.target
    data = data[
        (data[target + "_n"] >= row.minimum_observations) & data[target].notna()
    ]
    classes = []
    for year in [2024, 2025]:
        tr = data[data.draft_year < year]
        te = data[data.draft_year == year]
        if len(tr) < 15 or len(te) < 10:
            continue
        m = make_pipeline(
            SimpleImputer(
                strategy="median", add_indicator=True, keep_empty_features=True
            ),
            StandardScaler(),
            Ridge(alpha=10),
        )
        m.fit(tr[r["features"]], tr[target])
        pred = m.predict(te[r["features"]])
        classes.append(
            {
                "class": year,
                "training_players": len(tr),
                "evaluation_players": len(te),
                "pearson": float(pearsonr(te[target], pred).statistic),
                "r2": float(r2_score(te[target], pred)),
            }
        )
    follow.append(
        {
            "group": row.group,
            "target": row.target,
            "conditional_bootstrap_pearson_ci95": np.quantile(
                boot, [0.025, 0.975]
            ).tolist(),
            "draft_class_transfer": classes,
        }
    )
r["top_target_sensitivity"] = follow
r["notes"] += [
    "Top-target bootstrap intervals reuse fixed out-of-fold fits and are not selection-adjusted. Class-transfer checks reuse the selected targets and all available careers; they are sensitivity analyses, not independent confirmations."
]
(O / "results.json").write_text(json.dumps(r, indent=2))
(O / "top-target-sensitivity.json").write_text(json.dumps(follow, indent=2))
fig, ax = plt.subplots(figsize=(11, 5.6))
top = q.head(12).iloc[::-1]
colors = [
    {
        "movement": "#14634c",
        "receiving": "#3c78a8",
        "playing_time": "#a57a43",
        "pass_rush": "#885f9c",
    }.get(c, "#777777")
    for c in top.category
]
labels = [a.group + " · " + a.target.replace("_", " ") for a in top.itertuples()]
ax.barh(labels, top.mean_pearson, color=colors)
ax.set(
    xlabel="Mean held-out Pearson correlation · 3 × 5 player-level folds",
    title="Which game metrics can official Combine measurements predict?",
)
ax.spines[["top", "right"]].set_visible(False)
fig.set_facecolor("#f7faf8")
ax.set_facecolor("#f7faf8")
fig.tight_layout()
fig.text(
    0.03,
    -0.025,
    "Exploratory screen of 97 targets • fixed ridge α=10 • movement, event outcomes, and playing time are distinct constructs",
    fontsize=9,
)
fig.savefig(O / "target-screen.png", dpi=180, bbox_inches="tight")
lines = [
    "# Broad Combine-to-game predictive screen",
    "",
    "A deliberately small model surveys which available game measurements can be predicted from official Combine results and body dimensions. Targets are aggregated by player and evaluated separately within seven position groups.",
    "",
    "**Model:** fixed ridge linear regression, alpha=10; 12 official measurements plus training-fold missingness indicators. No sensor features, draft year, or identity enter the main prediction. Three repeated five-fold player-held-out partitions produce the reported correlations. No target-dependent model tuning is performed.",
    "",
    "**Scope:** 97 group/target pairs, minimum 30 eligible players per pair, and metric-specific minimum observed events. Missing outcome values are excluded; they are not silently converted to zero.",
    "",
    "## Highest correlations",
    "",
    "| Group | Target | Players | Pearson r | Held-out R² | Exploratory FDR q |",
    "|---|---|---:|---:|---:|---:|",
]
for a in q.head(15).itertuples():
    lines.append(
        f"| {a.group} | {a.target.replace('_', ' ')} | {a.players} | {a.mean_pearson:.3f} | {a.mean_r2:.3f} | {a.fdr_q:.3f} |"
    )
lines += [
    "",
    "![Target survey](target-screen.png)",
    "",
    "## Interpretation",
    "",
    "The clearest signal is physical transfer: Combine measurements predict high-end game speed, particularly among edge players. This is not equivalent to predicting sacks or coverage quality. Receiver separation remains a leading performance outcome in this screen. Interior defensive-line get-off time is a smaller-sample candidate. Expected YAC is itself a model-derived contextual expectation, not actual YAC above expectation.",
    "",
    "The receiver-separation sample requires at least 25 nonmissing observations and contains 74 players, so it is not directly comparable to the earlier all-82-player experiment. A high correlation with negative R² can indicate poor calibration and is not a good predictive model by itself.",
    "",
    "## Controls and target definitions",
    "",
    "A separate ridge baseline predicts each target using position subgroup and draft class only. Adding Combine measurements to this metadata is evaluated on identical folds. This checks whether a signal is largely an exposure/class or broad-role effect. It is a descriptive sensitivity check, not causal adjustment.",
    "",
    "Per-target descriptions, sample sizes, observation thresholds, repeat ranges, RMSE, metadata baselines, and incremental R² are in `results.json`. Movement speed/acceleration uses recorded cohort frames from snap through release/end, capped at three seconds. Outcomes pool 2023–2025; career summaries have unequal follow-up. Starting cushion is deployment context. Pressure times are conditional on recorded pressure events, and tackle share is conditional on recorded tackle events.",
    "",
    "## Multiple testing and uncertainty",
    "",
    "For each target, 1,000 shuffled-player-outcome regressions are evaluated with the same fixed features and partitions. Exact linear prediction operators make these refits cheap. Positive-correlation permutation p-values receive one Benjamini–Hochberg correction across all 97 tested group/target pairs. Team, class and heteroskedastic effects can challenge outcome exchangeability, so q-values are exploratory rather than definitive discovery claims.",
    "",
    "The top targets additionally receive player-bootstrap intervals and later-draft-class transfer checks, stored in `top-target-sensitivity.json`. These do not undo selection of the targets from this dataset. Independent data is needed before claiming that any target generalizes.",
    "",
    "## Complete ranking",
    "",
    "| Group | Target | Category | N | Pearson r | R² | ΔR² vs metadata | q |",
    "|---|---|---|---:|---:|---:|---:|---:|",
]
for a in q.itertuples():
    lines.append(
        f"| {a.group} | {a.target} | {a.category} | {a.players} | {a.mean_pearson:.3f} | {a.mean_r2:.3f} | {a.incremental_r2_vs_metadata:.3f} | {a.fdr_q:.3f} |"
    )
(O / "report.md").write_text("\n".join(lines))
print("Report and sensitivity checks complete")
