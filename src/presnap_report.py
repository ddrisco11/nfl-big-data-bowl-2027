import os
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nfl-matplotlib")
import pathlib, json
import numpy as np, pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

O = pathlib.Path(os.environ.get("NFL_PRESNAP_OUTPUT", ".work/presnap"))
r = json.loads((O / "results.json").read_text())
p = pd.read_csv(O / "player-above-expected.csv")
g = pd.read_csv(O / "route-predictions.csv.gz")
base = {
    k: float(np.mean([v[k] for v in r["baseline_evaluation"]]))
    for k in ["route_r2", "route_rmse", "route_pearson"]
}
qual = g.groupby("nfl_id").row_id.nunique()
p["observations"] = p.nfl_id.map(qual)
raw_expected = float(spearmanr(p.actual, p.expected).statistic)
raw_excess = float(spearmanr(p.actual, p.above_expected).statistic)
r["baseline_mean_metrics"] = base
r["raw_average_vs_above_expected_player_spearman"] = raw_excess
r["raw_vs_expected_player_spearman"] = raw_expected
(O / "results.json").write_text(json.dumps(r, indent=2))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
fig.set_facecolor("#f7faf8")
for a in ax:
    a.set_facecolor("#f7faf8")
    a.spines[["top", "right"]].set_visible(False)
ax[0].scatter(p.expected, p.actual, c="#14634c", s=35, alpha=0.8)
limits = [
    min(p.expected.min(), p.actual.min()) - 0.1,
    max(p.expected.max(), p.actual.max()) + 0.1,
]
ax[0].plot(limits, limits, "--", color="#8aa596")
ax[0].set(
    title="Player actual and pre-snap expected separation",
    xlabel="Expected separation (yards)",
    ylabel="Actual separation (yards)",
)
ax[1].scatter(p.actual, p.above_expected, c="#3c78a8", s=35, alpha=0.8)
ax[1].axhline(0, color="#8aa596")
ax[1].set(
    title="Opportunity adjustment changes the target",
    xlabel="Actual separation (yards)",
    ylabel="Separation above expectation (yards)",
)
fig.tight_layout()
fig.text(
    0.04,
    -0.04,
    "82 WRs · average of player-excluded forecasts · pre-snap inputs only · exploratory 2023–2025 sample",
    fontsize=9,
)
fig.savefig(O / "presnap-expectation.png", dpi=180, bbox_inches="tight")
lines = [
    "# Separation above pre-snap expectation",
    "",
    "For a receiver on play i, the population model estimates expected separation from beginning-of-play information. It pools all available WR observations in its training partition. It does not use the receiver’s identity or average separation as an input.",
    "",
    "$$\\widehat{s}_i=f_{\\mathrm{WR}}(X_{i,\\mathrm{snap}}),\\qquad \\mathrm{SAE}_a=\\frac{1}{N_a}\\sum_{i:a(i)=a}(s_i-\\widehat{s}_i).$$",
    "",
    "A positive receiver value means more separation than expected for the starting situation. If a verified defender-level separation-allowed outcome were available, the useful defensive sign would be reversed: expected minus observed separation allowed.",
    "",
    "## Defensive-back feasibility",
    "",
    "| Position | Regular-season rows | Release-separation observations |",
    "|---|---:|---:|---:|",
]
lines[-2] = "| Position | Regular-season rows | Release-separation observations |"
lines[-1] = "|---|---:|---:|"
for k, v in r["db_audit"].items():
    lines.append(f"| {k} | {v['rows']:,} | {v['separation_available']} |")
lines += [
    "",
    "There are no observed release-separation outcomes on CB, FS or SS rows. Coverage roles do not identify the covered receiver. With incomplete defender tracking, receiver separation cannot reliably be assigned to the nearest recorded DB. No defensive-back separation model is fitted.",
    "",
    "## Expected-separation model",
    "",
    "Candidates are ridge regression, histogram gradient boosting and CatBoost regression. Separation is continuous, so the model is a regressor. A classifier would require arbitrary separation bins. Candidate selection uses inner player-grouped validation; the outer evaluated player never fits or selects the expected-value model. The entire process repeats across three five-fold player partitions.",
    "",
    "Inputs: " + ", ".join(r["categorical_inputs"] + r["numeric_inputs"]) + ".",
    "",
    "Recorded route, realized coverage, time to throw, target status, pass outcomes, player identity and Combine variables are excluded. Cushion is treated as the supplied starting-distance measurement. Team identifiers and season are available before the play and help represent scheme and opponent context, but cannot fully isolate player effects.",
    "",
    f"Average route-level held-out R² is **{base['route_r2']:.3f}**, RMSE **{base['route_rmse']:.3f} yards**, and Pearson correlation **{base['route_pearson']:.3f}**. These are route-level forecasting metrics, distinct from player-average prediction metrics.",
    "",
    "Selected model counts: " + str(r["selected_model_counts"]) + ".",
    "",
    "![Actual and expected separation](presnap-expectation.png)",
    "",
    "## Combine prediction of the new above-expected target",
    "",
    "Only official Combine measurements and body dimensions enter the following models. Targets are mean receiver residuals from the pre-snap population model. Training-player residuals are cross-fitted. Outer test-player residuals are predicted by the baseline fitted only on training players.",
    "",
    "| Model | Mean held-out R² | Mean Pearson r |",
    "|---|---:|---:|",
]
for v in r["combine_to_above_expected"]:
    lines.append(f"| {v['model']} | {v['mean_r2']:.3f} | {v['mean_pearson']:.3f} |")
lines += [
    "",
    f"Across the 82 player summaries, raw average separation and above-expected separation have Spearman correlation **{raw_excess:.3f}**. The above-expected player estimates have SD **{r['player_above_expected_sd_yards']:.3f} yards**. The previous richer adjustment and the new pre-snap-only adjustment correlate at {min(r['old_postsnap_vs_new_presnap_adjusted_player_correlations']):.3f}–{max(r['old_postsnap_vs_new_presnap_adjusted_player_correlations']):.3f} across repeat-specific player estimates.",
    "",
    "## Limits",
    "",
] + ["- " + n for n in r["notes"]]
(O / "report.md").write_text("\n".join(lines))
print("Report rendered", base)
