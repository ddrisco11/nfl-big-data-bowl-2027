"""Render publication figures from the local analysis outputs."""

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nfl-matplotlib")
from shared import *
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

F = pathlib.Path(os.environ.get("NFL_FIGURE_DIR", "figures"))
F.mkdir(parents=True, exist_ok=True)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": "#253c36",
        "text.color": "#17362c",
        "axes.edgecolor": "#adc2b8",
        "xtick.color": "#526a60",
        "ytick.color": "#526a60",
        "figure.facecolor": "#f7faf8",
        "axes.facecolor": "#f7faf8",
        "savefig.facecolor": "#f7faf8",
    }
)
green = "#14634c"
blue = "#3c78a8"
orange = "#bd713c"
purple = "#885f9c"


def save(fig, name):
    fig.savefig(F / (name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(F / (name + ".svg"), bbox_inches="tight")
    plt.close(fig)


r = json.loads((O / "results.json").read_text())
m = pd.DataFrame(r["metrics"])
d, s, c, h, v, y = load()
p = np.load(W / "Movement_recognition.npy")
test = (d.split == "test").to_numpy()
meta = json.loads((O / "unpredictability-method.json").read_text())
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
names = [
    "Motion families",
    "Sequence GRU",
    "Joint route and path",
    "Temporal convolution",
    "Movement recognition",
]
colors = ["#a3ada8", blue, purple, orange, green]
for name, color in zip(names, colors):
    z = m[m.model == name]
    ax[0].plot(
        z.observation,
        z.accuracy,
        label=name,
        color=color,
        lw=3 if name == "Movement recognition" else 1.7,
        marker="o",
        ms=4,
    )
    ax[1].plot(
        z.observation,
        z.log_loss,
        color=color,
        lw=3 if name == "Movement recognition" else 1.7,
        marker="o",
        ms=4,
    )
ax[0].yaxis.set_major_formatter(PercentFormatter(1))
ax[0].set(
    title="Route recognition improves with movement",
    ylabel="Exact-route accuracy",
    xlabel="Seconds after snap",
    xticks=[0, 1, 2, 3],
    ylim=(0.2, 0.86),
)
ax[1].set(
    title="Probability quality also improves",
    ylabel="Multiclass log loss (lower is better)",
    xlabel="Seconds after snap",
    xticks=[0, 1, 2, 3],
)
fig.legend(
    *ax[0].get_legend_handles_labels(),
    loc="lower center",
    ncol=3,
    bbox_to_anchor=(0.5, -0.08),
    frameon=False,
    fontsize=10,
)
fig.text(
    0.08,
    -0.16,
    "2025 development holdout • eligible routes decline from 19,328 at snap to 8,203 at 3 s.",
    fontsize=10,
    color="#526a60",
)
fig.tight_layout()
save(fig, "model-performance")
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
calrows = []
for t, color in [(10, blue), (20, green), (30, orange)]:
    ok = test & v[:, t]
    pr = p[ok, t]
    conf = pr.max(1)
    correct = pr.argmax(1) == y[ok]
    xx = []
    yy = []
    nn = []
    for b in range(10):
        z = (conf >= b / 10) & (conf < ((b + 1) / 10) if b < 9 else conf <= 1)
        if z.sum() >= 20:
            xx.append(conf[z].mean())
            yy.append(correct[z].mean())
            nn.append(int(z.sum()))
            calrows.append(
                {
                    "seconds": t / 10,
                    "confidence": float(xx[-1]),
                    "accuracy": float(yy[-1]),
                    "n": nn[-1],
                }
            )
    ax[0].plot(xx, yy, "o-", label=f"{t/10:g} s", color=color, lw=2, ms=5)
ax[0].plot([0, 1], [0, 1], "--", color="#a3ada8", lw=1)
ax[0].set(
    title="Confidence tracks observed accuracy",
    xlabel="Predicted top-route probability",
    ylabel="Observed accuracy",
    xlim=(0, 1),
    ylim=(0, 1),
)
ax[0].xaxis.set_major_formatter(PercentFormatter(1))
ax[0].yaxis.set_major_formatter(PercentFormatter(1))
ax[0].legend(frameon=False)
z = m[(m.model == "Movement recognition") & (m.observation == 2)].iloc[0]
subs = [
    x
    for x in r["subgroups"]
    if x["model"] == "Movement recognition"
    and x["observation"] == 2
    and x["subgroup"] in ["seen players", "unseen players"]
]
ax[1].barh(
    [
        x["subgroup"]
        .replace("unseen players", "Absent from training")
        .replace("seen players", "Present in training")
        for x in subs
    ],
    [x["accuracy"] for x in subs],
    color=[green, blue],
    height=0.5,
)
for i, x in enumerate(subs):
    ax[1].text(
        x["accuracy"] + 0.012,
        i,
        f"{x['accuracy']:.1%} · n={x['n']:,}",
        va="center",
        fontsize=11,
    )
ax[1].set(
    title="Similar accuracy on new players",
    xlabel="Exact-route accuracy at 2 seconds",
    xlim=(0, 0.85),
)
ax[1].xaxis.set_major_formatter(PercentFormatter(1))
fig.tight_layout()
save(fig, "calibration")
(O / "calibration-bins.json").write_text(json.dumps(calrows, indent=2))
# Illustrative actual held-out probability curve, chosen near the median score among full-window routes.
routes = pd.read_csv(O / "route-unpredictability.csv.gz")
z = routes[routes.native_frames == 31]
row = z.iloc[np.argmin(abs(z.unpredictability - z.unpredictability.median()))]
i = int(np.flatnonzero(d.row_id == row.row_id)[0])
prob = p[i, :, y[i]]
u = float(row.unpredictability)
fig, ax = plt.subplots(figsize=(10.5, 4.8))
ax.fill_between(
    TIMES, prob, 1, color="#d9e9df", label="Area of uncertainty: 1 − P(recorded route)"
)
ax.plot(TIMES, prob, color=green, lw=3, label="Probability assigned to recorded route")
ax.set(
    xlim=(0, 3),
    ylim=(0, 1),
    xlabel="Seconds after snap",
    ylabel="Probability",
    title="A route score is the average shaded area",
)
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.text(
    0.05,
    0.82,
    rf"$U_i=\frac{{100}}{{T_i}}\int_0^{{T_i}}[1-p_{{i,t}}(r_i)]\,dt={u:.1f}$",
    transform=ax.transAxes,
    fontsize=21,
    color=green,
)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False, ncol=1)
fig.text(
    0.1,
    -0.23,
    "One observed 2025 route • native 100 ms probabilities • selected near median full-window score.",
    fontsize=10,
    color="#526a60",
)
save(fig, "integral")
# Repeatability uses disjoint game groups; points are anonymous player aggregates.
routes["half"] = routes.game_id % 2
hh = (
    routes.groupby(["nfl_id", "half"])
    .agg(score=("excess_unpredictability", "mean"), n=("row_id", "size"))
    .reset_index()
    .pivot(index="nfl_id", columns="half", values=["score", "n"])
    .dropna()
)
hh = hh[hh["n"].min(axis=1) >= 25]
players = pd.read_csv(O / "player-unpredictability.csv")
players = players[~players.provisional]
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
ax[0].scatter(
    hh["score"][0], hh["score"][1], s=36, color=green, alpha=0.72, edgecolors="white"
)
ax[0].axhline(0, color="#bdcdc4", lw=1)
ax[0].axvline(0, color="#bdcdc4", lw=1)
ax[0].set(
    title="Repeatability is modest",
    xlabel="Excess unpredictability · game group A",
    ylabel="Excess unpredictability · game group B",
)
ax[0].text(
    0.04,
    0.93,
    f"Spearman ρ = {meta['split_game_repeatability']['spearman']:.2f}\nn = {len(hh)} players",
    va="top",
    transform=ax[0].transAxes,
)
ax[1].scatter(
    players.excess_unpredictability,
    players.excess_brier,
    s=36,
    color=blue,
    alpha=0.72,
    edgecolors="white",
)
ax[1].set(
    title="Agreement with a proper score is strong",
    xlabel="Context-adjusted 1 − p score",
    ylabel="Context-adjusted Brier score",
)
ax[1].text(
    0.04,
    0.93,
    f"Spearman ρ = {meta['proper_score_agreement']:.2f}\nn = {len(players)} players",
    va="top",
    transform=ax[1].transAxes,
)
fig.tight_layout()
save(fig, "score-validation")
a = json.loads((O / "combine-associations.json").read_text())
tests = [
    x for x in a["tests"] if x["outcome"] == "excess_unpredictability" and "rho" in x
]
tests.sort(key=lambda x: x["rho"])
labels = {
    "combine_height": "Height",
    "combine_weight": "Weight",
    "ten_yd_split": "10-yard split",
    "forty": "40-yard time",
    "vertical": "Vertical jump",
    "broad_jump": "Broad jump",
    "three_cone": "Three-cone time",
    "short_shuttle": "Short-shuttle time",
    "forty_sensor_speed95": "40-yard sensor speed (95th pct.)",
    "forty_sensor_accel95": "40-yard sensor accel. (95th pct.)",
}
fig, ax = plt.subplots(figsize=(10.5, max(5, len(tests) * 0.36)))
xx = np.array([x["rho"] for x in tests])
ci = np.array([x["ci95"] for x in tests])
ax.errorbar(
    xx,
    np.arange(len(tests)),
    xerr=np.array([xx - ci[:, 0], ci[:, 1] - xx]),
    fmt="o",
    color=green,
    ecolor="#9cbaaa",
    capsize=3,
)
ax.axvline(0, color="#8c9e95", ls="--", lw=1)
ax.set(
    yticks=np.arange(len(tests)),
    yticklabels=[
        labels.get(
            x["feature"],
            x["feature"]
            .replace("_", " ")
            .replace("speed95", "speed (95th pct.)")
            .replace("turn rate90", "turn rate (90th pct.)")
            .capitalize(),
        )
        + f"  (n={x['n']})"
        for x in tests
    ],
    xlabel="Partial rank correlation with excess unpredictability · 95% player bootstrap interval",
    title="Combine associations remain exploratory",
)
ax.set_xlim(-0.7, 0.7)
fig.text(
    0.02,
    -0.02,
    f"36 planned tests across two scores; smallest BH-adjusted q = {min(x['fdr_q'] for x in a['tests']):.3f}.\nIntervals are pointwise, not multiplicity-adjusted. Complete-case samples vary by drill.",
    fontsize=10,
    color="#526a60",
)
fig.tight_layout()
save(fig, "combine-associations")
print("Rendered five figures")
