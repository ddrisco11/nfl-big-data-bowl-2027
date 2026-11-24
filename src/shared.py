"""Dataset loading, hierarchy mapping, calibration, and evaluation utilities."""

import os, sys, pathlib, json, pickle

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np, pandas as pd
from common import CLASSES, CAT, make_features, temp_prob
from sklearn.metrics import log_loss

W = pathlib.Path(os.environ.get("NFL_MODEL_WORK", ".work/models"))
O = pathlib.Path(os.environ.get("NFL_OUTPUT_DIR", ".work/results"))
TIMES = np.arange(31) * 0.1
FAMILIES = json.loads((O / "family-map.json").read_text())
FN = list(FAMILIES)
FM = np.array([next(i for i, n in enumerate(FN) if c in FAMILIES[n]) for c in CLASSES])
MAP = np.eye(len(FN))[FM]


def load():
    d = pd.read_parquet(W / "routes.parquet")
    a = np.load(W / "sequences.npz")
    s = a["motion"]
    c = a["context"]
    h = a["history"]
    v = a["live"][:, 10:41] & np.logical_and.accumulate(
        np.isfinite(s[:, 10:41]).all(2), axis=1
    )
    y = d.route_ran.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    return d, s, c, h, v, y


def feature_matrix(d, s, c, h, context=False, history=False):
    static = make_features(d, s[:, 10:], 0)
    static = (
        pd.get_dummies(static, columns=CAT, dtype=float)
        .astype(float)
        .to_numpy(np.float32)
    )
    N = len(d)
    out = []
    pre = s[:, :11]
    pres = np.column_stack(
        [
            np.nanmean(pre, axis=1),
            np.nan_to_num(pre[:, 10, :2] - pre[:, 0, :2]),
            np.isfinite(pre).all(2).sum(1),
        ]
    )
    for t in range(31):
        j = t + 10
        prefix = s[:, 10 : j + 1]
        past = np.concatenate([s[:, max(0, j - lag)] for lag in [0, 2, 4, 8]], axis=1)
        means = np.nanmean(prefix, axis=1)
        spread = np.nanstd(prefix, axis=1)
        delta = s[:, j] - s[:, max(10, j - 4)]
        motion = np.diff(prefix[:, :, :2], axis=1)
        distance = np.nansum(np.linalg.norm(motion, axis=2), axis=1)
        angle = np.arctan2(prefix[:, :, 5], prefix[:, :, 4])
        turn = np.nansum(
            np.abs(
                np.arctan2(
                    np.sin(np.diff(angle, axis=1)), np.cos(np.diff(angle, axis=1))
                )
            ),
            axis=1,
        )
        parts = [
            static,
            pres,
            past,
            means,
            spread,
            delta,
            np.column_stack([distance, turn, np.full(N, t * 0.1)]),
        ]
        if context:
            parts.extend([c[:, j], np.nanmean(c[:, : j + 1], axis=1)])
        if history:
            parts.append(h)
        out.append(np.column_stack(parts))
    return np.stack(out, axis=1).astype(np.float32)


def splits(d, v):
    train = (d.split == "train").to_numpy()
    val = (d.split == "validation").to_numpy()
    sel = val & ((d.game_id // 5) % 2 == 0).to_numpy()
    cal = val & ~sel
    test = (d.split == "test").to_numpy()
    return {
        k: b[:, None] & v
        for k, b in [
            ("train", train),
            ("selection", sel),
            ("calibration", cal),
            ("test", test),
        ]
    }


def score(y, p, mask, model):
    rows = []
    for t in [0, 10, 20, 30]:
        ok = mask[:, t]
        P = p[ok, t]
        yy = y[ok]
        if not len(yy):
            continue
        conf = P.max(1)
        correct = P.argmax(1) == yy
        ece = 0
        for b in range(10):
            m = (conf >= b / 10) & (conf < (b + 1) / 10 if b < 9 else conf <= 1)
            if m.any():
                ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
        rows.append(
            {
                "model": model,
                "observation": float(TIMES[t]),
                "n": int(ok.sum()),
                "accuracy": float(correct.mean()),
                "family_accuracy": float(((P @ MAP).argmax(1) == FM[yy]).mean()),
                "log_loss": float(log_loss(yy, P, labels=range(12))),
                "brier": float(np.mean(np.sum((P - np.eye(12)[yy]) ** 2, axis=1))),
                "ece": float(ece),
            }
        )
    return rows


def calibrate(p, y, mask):
    i, t = np.where(mask)
    temp = min(
        [0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.8],
        key=lambda z: log_loss(y[i], temp_prob(p[i, t], z), labels=range(12)),
    )
    return temp_prob(p.reshape(-1, 12), temp).reshape(p.shape).astype(np.float32), temp
