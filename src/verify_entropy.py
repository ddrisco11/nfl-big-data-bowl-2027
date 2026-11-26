"""Verify causal features, probability scores, and censor-aware response detection."""

import pathlib, json, argparse
import numpy as np, pandas as pd
from rei_core import prefix_features, shannon, integrate_entropy, response_onset

parser = argparse.ArgumentParser()
parser.add_argument("--work", default=".work/models")
parser.add_argument("--output", default=".work/rei")
args = parser.parse_args()
W = pathlib.Path(args.work)
O = pathlib.Path(args.output)
d = pd.read_parquet(W / "routes.parquet")
a = np.load(W / "sequences.npz")
s = a["motion"]
sel = np.arange(64)
x, b = prefix_features(d.iloc[sel], s[sel])
changed = s[sel].copy()
changed[:, 21:] += 100
xp, bp = prefix_features(d.iloc[sel], changed)
assert np.allclose(x[:, :11], xp[:, :11], equal_nan=True)
assert np.allclose(b[:, :11], bp[:, :11], equal_nan=True)
other = d.iloc[sel].copy()
other["route_ran"] = "FAKE"
other["nfl_id"] = -1
other["forty"] = 999
xx, _ = prefix_features(other, s[sel])
assert np.allclose(x, xx, equal_nan=True)
t = np.arange(11) * 0.1
assert abs(integrate_entropy(np.full((11, 8), 1 / 8), t) - 100) < 1e-8
assert integrate_entropy(np.tile([1, 0, 0, 0, 0, 0, 0, 0], (11, 1)), t) == 0
try:
    integrate_entropy(np.full((3, 8), 1 / 8), [0, 0.1, 0.3])
    raise AssertionError("Gap accepted")
except ValueError:
    pass
signal = np.zeros(40)
signal[15:] = 1.2
assert response_onset(signal, 12, 30, 1) == 15
assert response_onset(signal, 12, 16, 1) is None
signal = np.zeros(40)
signal[10:] = 1.2
assert response_onset(signal, 12, 30, 1) == 10
signal = np.zeros(40)
signal[15] = 2
assert response_onset(signal, 12, 30, 1) is None
# A deterministic velocity forecast is an essential stronger location baseline.
probs = np.load(O / "entropy-probabilities.npz")
valid = probs["valid"]
i, k = np.where(valid & (d.split == "test").to_numpy()[:, None])
j = k + 10
target = np.column_stack(
    [s[i, j + 5, :2] - s[i, j, :2], s[i, j + 10, :2] - s[i, j, :2]]
)
cv = np.column_stack([s[i, j, 4:6] * 0.5, s[i, j, 4:6]])
cvenergy = float(np.linalg.norm(cv - target, axis=1).mean())
summary = json.loads((O / "entropy-summary.json").read_text())
summary["constant_velocity_energy_score_four_dimensional_yards"] = cvenergy
summary["checks"].update(
    {
        "future_input_perturbation_invariance": True,
        "route_label_identifier_combine_feature_invariance": True,
        "gap_rejected": True,
        "reaction_censoring_and_anticipation_tests": True,
    }
)
(O / "entropy-summary.json").write_text(json.dumps(summary, indent=2))
print("Causality, entropy endpoints, gaps, response censoring and anticipation: PASS")
print("Constant velocity energy", cvenergy)
