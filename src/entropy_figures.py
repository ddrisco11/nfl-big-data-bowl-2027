"""Render aggregate Route Entropy Index and defender-coverage diagnostics."""

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nfl-matplotlib")
import argparse, json, pathlib
import numpy as np, pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--output", default=".work/rei")
p.add_argument("--figures", default="figures")
args = p.parse_args()
O = pathlib.Path(args.output)
F = pathlib.Path(args.figures)
F.mkdir(parents=True, exist_ok=True)
summary = json.loads((O / "entropy-summary.json").read_text())
reaction = json.loads((O / "reaction-summary.json").read_text())
r = pd.read_csv(O / "route-entropy.csv.gz")
centers = np.array(summary["mode_centers_relative_yards"])
green = "#14634c"
blue = "#3c78a8"
orange = "#bd713c"
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "#f7faf8",
        "axes.facecolor": "#f7faf8",
        "text.color": "#17362c",
        "axes.labelcolor": "#17362c",
    }
)


def save(fig, name):
    fig.savefig(F / (name + ".png"), dpi=170, bbox_inches="tight")
    fig.savefig(F / (name + ".svg"), bbox_inches="tight")
    plt.close(fig)


fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
for k, c in enumerate(centers):
    points = np.array([[0, 0], c[:2], c[2:]])
    ax[0].plot(points[:, 1], points[:, 0], "-o", lw=1.8, ms=3, label=f"{k+1}")
    ax[0].text(c[3] + 0.1, c[2], str(k + 1), fontsize=10)
ax[0].set(
    title="Eight learned future patterns",
    xlabel="Lateral displacement inward (yards)",
    ylabel="Forward displacement (yards)",
)
ax[0].axvline(0, color="#cccccc", lw=0.7)
ax[0].set_aspect("equal", adjustable="datalim")
ax[1].hist(r.rei, bins=np.arange(0, 101, 5), color=green, alpha=0.85)
ax[1].set(
    title="Route Entropy Index",
    xlabel="Average normalized entropy (0–100)",
    ylabel="Eligible receiver routes",
)
ax[1].text(
    0.97, 0.94, f"n = {len(r):,}", ha="right", va="top", transform=ax[1].transAxes
)
r["half"] = r.game_id % 2
h = (
    r.groupby(["nfl_id", "half"])
    .agg(value=("excess_rei", "mean"), n=("row_id", "size"))
    .unstack()
    .dropna()
)
h = h[h["n"].min(axis=1) >= 25]
ax[2].scatter(h["value"][0], h["value"][1], s=27, color=blue, alpha=0.75)
ax[2].axhline(0, color="#cccccc", lw=0.7)
ax[2].axvline(0, color="#cccccc", lw=0.7)
ax[2].set(
    title="Player-score repeatability",
    xlabel="Adjusted REI · game group A",
    ylabel="Adjusted REI · game group B",
)
ax[2].text(
    0.04,
    0.95,
    f"Spearman ρ = {summary['repeatability']['spearman']:.2f}\nn = {len(h)} players",
    va="top",
    transform=ax[2].transAxes,
)
fig.tight_layout()
fig.text(
    0.04,
    -0.06,
    "Patterns summarize displacement at 0.5 and 1.0 s. Entropy measures model ambiguity, not proven deception.",
    fontsize=10,
)
save(fig, "route-entropy-index")
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
models = summary["metrics"]
names = ["Context + speed", "Movement prefix"]
vals = [
    models[n]["test"]["log_loss"]
    for n in ["Context and speed baseline", "Movement prefix"]
]
ax[0].bar(names, vals, color=[blue, green], width=0.55)
for j, v in enumerate(vals):
    ax[0].text(j, v + 0.02, f"{v:.3f}", ha="center")
ax[0].set(
    title="Movement improves probability forecasts",
    ylabel="Eight-mode log loss (lower is better)",
    ylim=(0, 1.45),
)
counts = reaction["counts"]
labs = [
    "Test receiver routes",
    "Recorded MAN defender\non the play",
    "Detected sharp break",
    "Accepted candidate pair",
]
values = [
    counts[x]
    for x in [
        "eligible_test_routes",
        "routes_in_plays_with_recorded_man_defender",
        "detected_receiver_breaks",
        "accepted_candidate_pairs",
    ]
]
ax[1].barh(labs[::-1], values[::-1], color=[orange, orange, blue, blue], height=0.55)
ax[1].set_xscale("log")
ax[1].set_xlim(1, 60000)
for j, v in enumerate(values[::-1]):
    ax[1].text(v * 1.1, j, f"{v:,}", va="center")
ax[1].set(
    title="Defender response remains a feasibility test", xlabel="Count (log scale)"
)
fig.tight_layout()
fig.text(
    0.04,
    -0.09,
    "2025 development holdout. Candidate matchups are inferred; actual defender responsibility is unknown.\nThe strict break/pairing screen is not a census of all breaks or valid matchups.",
    fontsize=10,
)
save(fig, "entropy-validation")
print("Rendered REI figures")
