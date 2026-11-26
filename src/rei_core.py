"""Causal input construction and mathematical primitives for Route Entropy Index."""

import numpy as np
import pandas as pd


def prefix_features(d, s):
    n = len(d)
    static = pd.get_dummies(
        d[
            [
                "nfl_position",
                "lined_up_position",
                "offense_formation",
                "receiver_alignment",
            ]
        ].fillna("Missing"),
        dtype=float,
    ).to_numpy(np.float32)
    static = np.column_stack(
        [d[["down", "yards_to_go", "snap_y"]].to_numpy(np.float32), static]
    )
    full = []
    baseline = []
    for t in range(31):
        j = t + 10
        hist = s[:, 10 : j + 1]
        cur = s[:, j]
        motion = np.column_stack(
            [
                s[:, j - 4],
                s[:, j - 2],
                cur,
                np.nanmean(hist, axis=1),
                np.nanstd(hist, axis=1),
                cur - s[:, j - 4],
            ]
        )
        full.append(np.column_stack([static, np.full(n, t * 0.1), motion]))
        baseline.append(np.column_stack([static, np.full(n, t * 0.1), cur[:, 2]]))
    return np.stack(full, axis=1).astype(np.float32), np.stack(baseline, axis=1).astype(
        np.float32
    )


def shannon(p):
    p = np.asarray(p)
    return -np.sum(p * np.log(np.maximum(p, 1e-12)), axis=-1)


def integrate_entropy(p, times):
    p = np.asarray(p)
    times = np.asarray(times)
    if len(times) < 2 or times[-1] <= times[0]:
        raise ValueError("Positive duration and at least two observations required")
    if not np.allclose(np.diff(times), 0.1):
        raise ValueError("Missing or non-native frames must not be bridged")
    return float(
        100
        * np.trapezoid(shannon(p), times)
        / ((times[-1] - times[0]) * np.log(p.shape[-1]))
    )


def response_onset(signal, j, last, threshold):
    """First three-frame sustained response, respecting right censoring."""
    for k in range(j - 2, min(j + 11, last - 1)):
        if np.isfinite(signal[k : k + 3]).all() and np.all(
            signal[k : k + 3] >= threshold
        ):
            return k
    return None
