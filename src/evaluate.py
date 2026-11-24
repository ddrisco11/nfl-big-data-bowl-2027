"""Select calibrated ensembles and evaluate route recognition and location forecasts."""

import os
from shared import *
from sklearn.metrics import log_loss
from itertools import combinations

D, S, C, H, V, Y = load()
sp = splits(D, V)
files = {
    "Temporal convolution": "Temporal_convolution",
    "Motion families": "Motion_families",
    "Historical tendencies": "Historical_tendencies",
    "Sequence GRU": "Sequence_GRU",
    "Joint route and path": "Joint_route_and_path",
}
P = {n: np.load(W / (f + ".npy")) for n, f in files.items()}
si, st = np.where(sp["selection"])
rng = np.random.default_rng(2027)
loss = lambda p: float(log_loss(Y[si], p[si, st], labels=range(12)))
selection = {n: loss(p) for n, p in P.items()}
chosen = {}
metrics = []
for n, p in P.items():
    metrics.extend(score(Y, p, sp["test"], n))
for purpose in ["Movement recognition", "History-aware forecast"]:
    allowed = [
        n
        for n in files
        if purpose == "History-aware forecast" or n != "Historical tendencies"
    ]
    ranked = sorted(allowed, key=lambda n: selection[n])
    choices = [(selection[n], {n: 1.0}) for n in ranked]
    # Validation-selected blends, not test-driven selection. Include both useful model types.
    for a, b in combinations(ranked[:4], 2):
        for alpha in [0.25, 0.5, 0.75]:
            choices.append(
                (loss(alpha * P[a] + (1 - alpha) * P[b]), {a: alpha, b: 1 - alpha})
            )
    _, weights = min(choices, key=lambda x: x[0])
    pp = sum(w * P[n] for n, w in weights.items())
    pp, temp = calibrate(pp, Y, sp["calibration"])
    chosen[purpose] = {
        "weights": weights,
        "temperature": temp,
        "selection_log_loss": loss(pp),
    }
    P[purpose] = pp
    np.save(W / (purpose.replace(" ", "_") + ".npy"), pp)
    metrics.extend(score(Y, pp, sp["test"], purpose))
# Reference for paired differences: the native-rate hierarchical tree model.
B = P["Motion families"]
# Subgroups and calibrated confidence diagnostics.
seen = set(D.loc[D.split == "train", "nfl_id"])
unseen = ~D.nfl_id.isin(seen).to_numpy()
subgroups = []
for purpose in chosen:
    for name, mask in [
        ("unseen players", unseen),
        ("seen players", ~unseen),
        ("WR", (D.nfl_position == "WR").to_numpy()),
        ("TE", (D.nfl_position == "TE").to_numpy()),
    ]:
        subgroups.extend(
            [
                {**m, "subgroup": name}
                for m in score(Y, P[purpose], sp["test"] & mask[:, None], purpose)
            ]
        )


def bootstrap(game, delta):
    z = (
        pd.DataFrame({"game": game, "delta": delta, "n": 1})
        .groupby("game")
        .sum()
        .to_numpy()
    )
    draw = rng.integers(0, len(z), (1200, len(z)))
    b = z[draw].sum(1)
    return np.quantile(b[:, 0] / b[:, 1], [0.025, 0.975]).tolist()


changes = []
for t in [0, 10, 20]:
    mask = sp["test"][:, t]
    new = P["Movement recognition"][mask, t]
    base = B[mask, t]
    delta = (new.argmax(1) == Y[mask]).astype(float) - (base.argmax(1) == Y[mask])
    changes.append(
        {
            "observation": float(TIMES[t]),
            "accuracy_difference": float(delta.mean()),
            "ci95": bootstrap(D.game_id[mask], delta),
        }
    )
# All calibrated dense forecast probabilities, one row per eligible receiver/time.
i, t = np.where(sp["test"])
e = D.iloc[i][
    [
        "row_id",
        "game_id",
        "play_id",
        "nfl_id",
        "display_name",
        "route_ran",
        "nfl_position",
        "draft_year",
    ]
].reset_index(drop=True)
e["observation_seconds"] = TIMES[t]
e["unseen_player"] = unseen[i]
for k, c in enumerate(CLASSES):
    e["p_" + c] = P["Movement recognition"][i, t, k]
e.to_csv(O / "recognition-probabilities.csv.gz", index=False, compression="gzip")
del e
# Recognition event: true label is most likely above threshold for 5 successive frames (.4 seconds).
# Lack of recognition before release/end is censored, never assigned a fake delay of 3 seconds.
recognition = []
testrows = np.flatnonzero((D.split == "test").to_numpy() & V[:, 0])
pp = P["Movement recognition"]
for threshold in [0.5, 0.6, 0.7]:
    for familymode in [False, True]:
        probs = pp @ MAP if familymode else pp
        truth = FM[Y] if familymode else Y
        event = (
            (probs.argmax(2) == truth[:, None])
            & (
                np.take_along_axis(probs, truth[:, None, None], axis=2)[:, :, 0]
                >= threshold
            )
            & V
        )
        for idx in testrows:
            starts = [j for j in range(27) if event[idx, j : j + 5].all()]
            available = np.flatnonzero(V[idx])
            first = float(TIMES[starts[0]]) if starts else None
            recognition.append(
                {
                    "row_id": int(D.iloc[idx].row_id),
                    "game_id": int(D.iloc[idx].game_id),
                    "play_id": int(D.iloc[idx].play_id),
                    "nfl_id": int(D.iloc[idx].nfl_id),
                    "threshold": threshold,
                    "target": "family" if familymode else "exact route",
                    "recognized": bool(starts),
                    "recognition_onset_seconds": first,
                    "recognition_confirmed_seconds": first + 0.4
                    if first is not None
                    else None,
                    "last_eligible_seconds": float(TIMES[available[-1]]),
                }
            )
r = pd.DataFrame(recognition)
r.to_csv(O / "recognition-delay.csv.gz", index=False, compression="gzip")
summary = []
for (threshold, target), g in r.groupby(["threshold", "target"]):
    summary.append(
        {
            "threshold": threshold,
            "target": target,
            "n": len(g),
            "recognized_fraction": float(g.recognized.mean()),
            "median_onset_among_recognized": float(
                g.loc[g.recognized, "recognition_onset_seconds"].median()
            ),
            "censored_fraction": float(1 - g.recognized.mean()),
        }
    )
# Independent model checks: normalized probabilities, frozen game partitions, future-input exclusion.
assert all(np.allclose(p.sum(2), 1, atol=1e-5) for p in P.values())
games = {
    s: set(D.loc[D.split == s, "game_id"]) for s in ["train", "validation", "test"]
}
assert not games["train"] & games["test"]
assert not games["validation"] & games["test"]
# Joint model location head scored separately; it is not assumed to improve the existing location model.
paths = np.load(W / "Joint_route_and_path_paths.npy")
loc = []
rawlive = np.load(W / "sequences.npz")["live"]
for tick in [0, 10, 20]:
    for h in [1, 2]:
        end = 10 + tick + h * 10
        ok = sp["test"][:, tick] & rawlive[:, end] & np.isfinite(S[:, end, :2]).all(1)
        truth = S[ok, end, :2] - S[ok, 10 + tick, :2]
        pred = paths[ok, tick, (h - 1) * 2 : h * 2]
        cv = S[ok, 10 + tick, 4:6] * h
        loc.append(
            {
                "observation": float(TIMES[tick]),
                "horizon": h,
                "n": int(ok.sum()),
                "joint_error_yards": float(np.linalg.norm(pred - truth, axis=1).mean()),
                "constant_velocity_error_yards": float(
                    np.linalg.norm(cv - truth, axis=1).mean()
                ),
            }
        )
result = {
    "metrics": metrics,
    "selection_scores": selection,
    "chosen": chosen,
    "subgroups": subgroups,
    "improvement_intervals": changes,
    "recognition": summary,
    "joint_location": loc,
    "classes": CLASSES,
    "families": FAMILIES,
    "times": TIMES.tolist(),
    "verification": {
        "probability_normalization": True,
        "game_separation": True,
        "selection_uses_2025_labels": False,
        "history_uses_validation_or_test_labels": False,
    },
    "notes": [
        "2025 is a previously examined development holdout, not a new confirmatory test.",
        "The six families are explicitly defined modeling labels. Exact-route log loss and accuracy remain primary.",
        "Movement recognition excludes player/team historical priors. The history-aware model is a separate forecasting experiment.",
        "The sequence models are causal and trained on -1 through +3 seconds; outputs use only the available prefix. Future locations appear only as supervised training targets.",
        "Recognition requires the recorded true route/family to be the top prediction and exceed a probability threshold for five frames. It is a retrospective model-based measure, not proof of a defender reading the route.",
        "The onset time is known only after the 0.4-second confirmation window. No recognition before release/end or 3 seconds is censored, not treated as a 3-second delay.",
        "Examples retain all available cohort players; missing players are not reconstructed.",
        "No Combine features were used, preserving an independent outcome for later Combine-to-recognition analysis.",
    ],
}
(O / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
print(json.dumps(chosen, indent=2))
print("EVALUATION COMPLETE", flush=True)
