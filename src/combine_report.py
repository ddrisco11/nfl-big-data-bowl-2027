import os
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nfl-matplotlib")
import json, pathlib
import numpy as np, pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

O = pathlib.Path(os.environ.get("NFL_COMBINE_OUTPUT", ".work/combine-upgrades"))
r = json.loads((O / "results.json").read_text())
e = r["evaluation"]
lookup = {(z["kind"], z["outcome"], z["configuration"]): z for z in e}
plt.rcParams.update(
    {
        "figure.facecolor": "#f7faf8",
        "axes.facecolor": "#f7faf8",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
configs = [
    n + "|official|values|pool=False|min=1"
    for n in ["linear", "ridge", "elastic_net", "neural_net", "catboost", "xgboost"]
]
for ax, outcome in zip(axes, ["raw", "adjusted"]):
    z = [lookup["repeated", outcome, c] for c in configs]
    val = np.array([a["mean_metrics"]["r2"] for a in z])
    ci = np.array([a["conditional_player_bootstrap_r2_ci95"] for a in z])
    ax.barh(np.arange(6), val, color="#14634c")
    ax.errorbar(
        val,
        np.arange(6),
        xerr=np.maximum(0, np.array([val - ci[:, 0], ci[:, 1] - val])),
        fmt="none",
        color="#526a60",
        capsize=3,
    )
    ax.axvline(0, color="#8aa096")
    ax.set(
        yticks=np.arange(6),
        yticklabels=[
            "Linear",
            "Ridge",
            "Elastic net",
            "Neural net",
            "CatBoost",
            "XGBoost",
        ],
        xlabel="Mean held-out R² · conditional player-bootstrap 95% interval",
        title="Raw mean separation"
        if outcome == "raw"
        else "Separation above context expectation",
    )
    ax.invert_yaxis()
fig.tight_layout()
fig.text(
    0.05,
    -0.03,
    "82 WRs · 3 × 5 player-held-out folds · inner tuning · negative R² means worse than predicting the evaluation mean",
    fontsize=9,
)
fig.savefig(O / "model-comparison.png", dpi=180, bbox_inches="tight")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
configs = [
    "ridge|official|values|pool=False|min=1",
    "ridge|sensors|values|pool=False|min=1",
    "ridge|interactions|values|pool=False|min=1",
    "ridge|official|values|pool=True|min=1",
    "ridge|official|no_flags|pool=False|min=1",
    "ridge|official|flags|pool=False|min=1",
]
labels = [
    "Official + missing flags",
    "+ sensors",
    "+ three interactions",
    "Shrink training averages",
    "Official without flags",
    "Missing flags only",
]
for ax, outcome in zip(axes, ["raw", "adjusted"]):
    vals = [lookup["repeated", outcome, c]["mean_metrics"]["r2"] for c in configs]
    ax.barh(labels, vals, color="#3c78a8")
    ax.axvline(0, color="#8aa096")
    ax.set_title(outcome.capitalize() + " separation · ridge")
    ax.set_xlabel("Mean held-out R²")
    ax.invert_yaxis()
fig.tight_layout()
fig.savefig(O / "feature-comparison.png", dpi=180, bbox_inches="tight")
lines = [
    "# Combine movement and receiver separation",
    "",
    "One outcome per wide receiver: average provided separation at release across available regular-season plays. Official Combine measurements are compared with sensor-derived movement mechanics.",
    "",
    f"The sample contains {r['data']['players']} players and {r['data']['routes']:,} observed separation measurements. Training-only preprocessing and identical player partitions allow paired comparisons.",
    "",
    "## Repeated evaluation",
    "",
    "| Model · official measurements | Raw R² | Raw Pearson r | Adjusted R² | Adjusted Pearson r |",
    "|---|---:|---:|---:|---:|",
]
for c in configs[:0]:
    pass
for name in ["linear", "ridge", "elastic_net", "neural_net", "catboost", "xgboost"]:
    c = name + "|official|values|pool=False|min=1"
    a = lookup["repeated", "raw", c]["mean_metrics"]
    b = lookup["repeated", "adjusted", c]["mean_metrics"]
    lines.append(
        f"| {name} | {a['r2']:.3f} | {a['pearson']:.3f} | {b['r2']:.3f} | {b['pearson']:.3f} |"
    )
lines += [
    "",
    "![Repeated evaluation](model-comparison.png)",
    "",
    "## Movement tracking, interactions, and uncertainty",
    "",
    "| Ridge configuration | Raw R² | Adjusted R² |",
    "|---|---:|---:|",
]
for label, c in zip(labels, configs):
    lines.append(
        f"| {label} | {lookup['repeated', 'raw', c]['mean_metrics']['r2']:.3f} | {lookup['repeated', 'adjusted', c]['mean_metrics']['r2']:.3f} |"
    )
lines += [
    "",
    "![Feature comparisons](feature-comparison.png)",
    "",
    "## Minimum observation sensitivity",
    "",
    "| Minimum observations | Players | Raw ridge R² | Adjusted ridge R² |",
    "|---|---:|---:|---:|",
]
for minimum in [1, 25, 50, 100]:
    c = f"ridge|official|values|pool=False|min={minimum}"
    a = lookup["repeated", "raw", c]
    b = lookup["repeated", "adjusted", c]
    lines.append(
        f"| {minimum} | {a['players']} | {a['mean_metrics']['r2']:.3f} | {b['mean_metrics']['r2']:.3f} |"
    )
lines += [
    "",
    "## Draft-class transfer",
    "",
    "| Configuration | Outcome | 2024 R² | 2025 R² |",
    "|---|---|---:|---:|",
]
for c in [
    "ridge|official|values|pool=False|min=1",
    "neural_net|official|values|pool=False|min=1",
    "ridge|sensors|values|pool=False|min=1",
]:
    for out in ["raw", "adjusted"]:
        z = lookup["draft_class", out, c]
        v = {a["split"]: a["r2"] for a in z["splits"]}
        lines.append(
            f"| {c.split('|')[0]} · {c.split('|')[1]} | {out} | {v[2024]:.3f} | {v[2025]:.3f} |"
        )
lines += (
    ["", "## Missingness audit", ""]
    + [
        f"- {k}: {v:.1%} missing"
        for k, v in r["data"]["official_missing_fraction"].items()
    ]
    + [
        "",
        f"Event-derived time to throw is available for {1 - r['data']['missing_context_fraction']['time_to_throw']:.1%} of routes; ambiguous event times are excluded. The supplied dropback-duration column is almost entirely missing.",
        "",
        "## Interpretation and methods",
        "",
    ]
    + ["- " + n for n in r["notes"]]
    + [
        "",
        "Exact fold metrics, bootstrap intervals, paired MSE differences, and selected parameters are in results.json and parameters.json.",
    ]
)
(O / "report.md").write_text("\n".join(lines))
print("REPORT COMPLETE")
