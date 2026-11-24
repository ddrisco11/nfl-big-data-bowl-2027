"""Fit softly gated route-family and conditional-route gradient-boosted trees."""

import os
from shared import *
from sklearn.ensemble import HistGradientBoostingClassifier

D, S, C, H, V, Y = load()
sp = splits(D, V)
rng = np.random.default_rng(2027)
train = sp["train"].copy()
train[:, [j for j in range(31) if j % 4 and j not in [10, 20, 30]]] = False
ii, tt = np.where(train)
metrics = []
choices = []
for stage, usecontext, usehistory in [
    ("Motion families", False, False),
    ("Historical tendencies", True, True),
]:
    X = feature_matrix(D, S, C, H, usecontext, usehistory)
    Z = X[ii, tt]
    print(stage, "fit", Z.shape, flush=True)
    # Soft hierarchy: family posterior × within-family route posterior.
    fam = HistGradientBoostingClassifier(
        max_iter=130,
        max_leaf_nodes=23,
        learning_rate=0.08,
        l2_regularization=20,
        early_stopping=False,
        random_state=2027,
    )
    fam.fit(Z, FM[Y[ii]])
    predf = fam.predict_proba(X.reshape(-1, X.shape[-1]))
    P = np.zeros((len(D) * 31, 12))
    conditional = {}
    for f in range(len(FN)):
        classes = np.flatnonzero(FM == f)
        ok = FM[Y[ii]] == f
        if len(classes) == 1:
            P[:, classes[0]] = predf[:, f]
            continue
        m = HistGradientBoostingClassifier(
            max_iter=110,
            max_leaf_nodes=15,
            learning_rate=0.08,
            l2_regularization=20,
            early_stopping=False,
            random_state=2027,
        )
        m.fit(Z[ok], Y[ii[ok]])
        p = m.predict_proba(X.reshape(-1, X.shape[-1]))
        P[:, m.classes_] = predf[:, f, None] * p
        conditional[f] = m
    P = P.reshape(len(D), 31, 12)
    P, temp = calibrate(P, Y, sp["calibration"])
    i, t = np.where(sp["selection"])
    loss = log_loss(Y[i], P[i, t], labels=range(12))
    print(stage, "selection", loss, "temperature", temp, flush=True)
    np.save(W / (stage.replace(" ", "_") + ".npy"), P)
    pickle.dump(
        (fam, conditional, temp), open(W / (stage.replace(" ", "_") + ".pkl"), "wb")
    )
    metrics.extend(score(Y, P, sp["test"], stage))
    choices.append({"model": stage, "selection_log_loss": loss, "temperature": temp})
    del X, Z
(O / "tree-results.json").write_text(
    json.dumps({"metrics": metrics, "selection": choices}, indent=2)
)
print("TREE COMPARISONS COMPLETE", flush=True)
