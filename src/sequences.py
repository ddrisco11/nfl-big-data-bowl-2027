"""Train causal GRU classifiers with optional auxiliary trajectory supervision."""

import os
from shared import *
import torch, copy, time
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

# CPU execution is reproducible and avoids device-specific recurrent-kernel behavior.
torch.set_num_threads(4)
torch.manual_seed(2027)
np.random.seed(2027)
D, S, C, H, V, Y = load()
sp = splits(D, V)
trainrow = np.flatnonzero((D.split == "train").to_numpy())
selrow = np.flatnonzero(
    ((D.split == "validation") & ((D.game_id // 5) % 2 == 0)).to_numpy()
)
# Exactly -1.0 through +3.0 sec at 10 Hz; causal GRU outputs can only consume their prefix.
raw = np.concatenate([S[:, :41], C[:, :41]], axis=2)
missing = ~np.isfinite(raw)
mu = np.nanmean(raw[trainrow], axis=(0, 1))
sd = np.nanstd(raw[trainrow], axis=(0, 1))
sd = np.maximum(sd, 0.1)
seq = np.concatenate(
    [np.nan_to_num((raw - mu) / sd), missing.astype(float)], axis=2
).astype(np.float32)
seq = np.clip(seq, -12, 12)
static = make_features(D, S[:, 10:], 0)
static = pd.get_dummies(static, columns=CAT, dtype=float).to_numpy(float)
med = np.nanmedian(static[trainrow], axis=0)
static = np.where(np.isfinite(static), static, med)
scaler = StandardScaler().fit(static[trainrow])
static = scaler.transform(static).astype(np.float32)
# Future labels are auxiliary training targets only and never enter the input tensors.
future = np.zeros((len(D), 31, 4), np.float32)
futuremask = np.zeros((len(D), 31, 2), bool)
for t in range(31):
    for a, horizon in enumerate([10, 20]):
        end = 10 + t + horizon
        if end < 51:
            future[:, t, 2 * a : 2 * a + 2] = (S[:, end, :2] - S[:, 10 + t, :2]) / 10
            futuremask[:, t, a] = (
                V[:, t]
                & np.isfinite(S[:, end, :2]).all(1)
                & np.load(W / "sequences.npz")["live"][:, end]
            )
future = np.nan_to_num(future)
xs = torch.from_numpy(seq)
xt = torch.from_numpy(static)
yv = torch.tensor(Y, dtype=torch.long)
vf = torch.from_numpy(V)
ft = torch.from_numpy(future)
fm = torch.from_numpy(futuremask)
family = torch.tensor(FM, dtype=torch.long)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(seq.shape[-1], 96, num_layers=1, batch_first=True)
        self.static = nn.Sequential(nn.Linear(static.shape[-1], 32), nn.ReLU())
        self.shared = nn.Sequential(nn.Linear(128, 96), nn.ReLU(), nn.Dropout(0.15))
        self.route = nn.Linear(96, 12)
        self.family = nn.Linear(96, len(FN))
        self.path = nn.Linear(96, 4)

    def forward(self, x, z):
        hidden, _ = self.gru(x)
        hidden = hidden[:, 10:]
        z = self.static(z)[:, None, :].expand(-1, 31, -1)
        h = self.shared(torch.cat([hidden, z], dim=-1))
        return self.route(h), self.family(h), self.path(h)


def evaluate(model, indices):
    model.eval()
    loss = 0
    n = 0
    with torch.no_grad():
        for b in range(0, len(indices), 512):
            ix = indices[b : b + 512]
            logits, _, _ = model(xs[ix], xt[ix])
            ls = nn.functional.cross_entropy(
                logits.transpose(1, 2), yv[ix, None].expand(-1, 31), reduction="none"
            )
            mask = vf[ix]
            loss += (ls * mask).sum().item()
            n += mask.sum().item()
    return loss / n


metrics = []
records = []
for name, weight in [("Sequence GRU", 0.0), ("Joint route and path", 0.3)]:
    torch.manual_seed(2027)
    model = Net()
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    best = float("inf")
    patience = 0
    trace = []
    start = time.time()
    for epoch in range(1, 51):
        model.train()
        perm = np.random.default_rng(2027 + epoch).permutation(trainrow)
        total = 0
        for b in range(0, len(perm), 256):
            ix = perm[b : b + 256]
            logits, flogits, path = model(xs[ix], xt[ix])
            mask = vf[ix]
            ys = yv[ix, None].expand(-1, 31)
            ce = nn.functional.cross_entropy(
                logits.transpose(1, 2), ys, reduction="none"
            )
            fc = nn.functional.cross_entropy(
                flogits.transpose(1, 2), family[ys], reduction="none"
            )
            loss = (((ce + 0.15 * fc) * mask).sum(1) / mask.sum(1).clamp_min(1)).mean()
            if weight:
                pe = (
                    nn.functional.smooth_l1_loss(path, ft[ix], reduction="none")
                    .reshape(-1, 31, 2, 2)
                    .mean(-1)
                )
                loss += weight * (pe * fm[ix]).sum() / fm[ix].sum().clamp_min(1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            opt.step()
            total += loss.item()
        vl = evaluate(model, selrow)
        trace.append({"epoch": epoch, "selection_log_loss": vl})
        print(
            name,
            "epoch",
            epoch,
            "selection",
            round(vl, 5),
            "seconds",
            round(time.time() - start),
            flush=True,
        )
        if vl < best - 0.0005:
            best = vl
            state = copy.deepcopy(model.state_dict())
            bestepoch = epoch
            patience = 0
        else:
            patience += 1
        if patience >= 7:
            break
    model.load_state_dict(state)
    model.eval()
    P = []
    paths = []
    with torch.no_grad():
        for b in range(0, len(D), 512):
            logits, _, path = model(xs[b : b + 512], xt[b : b + 512])
            P.append(torch.softmax(logits, -1).numpy())
            paths.append(path.numpy() * 10)
    P = np.concatenate(P)
    P, temp = calibrate(P, Y, sp["calibration"])
    np.save(W / (name.replace(" ", "_") + ".npy"), P)
    np.save(W / (name.replace(" ", "_") + "_paths.npy"), np.concatenate(paths))
    torch.save(
        {
            "state": state,
            "sequence_mean": mu,
            "sequence_std": sd,
            "static_median": med,
            "static_scaler": scaler,
            "temperature": temp,
            "classes": CLASSES,
        },
        W / (name.replace(" ", "_") + ".pt"),
    )
    # Verify that future input perturbations cannot alter a past forecast.
    with torch.no_grad():
        testx = xs[:32].clone()
        a = model(testx, xt[:32])[0]
        testx[:, 21:] += 100
        b = model(testx, xt[:32])[0]
        causal = float((a[:, :11] - b[:, :11]).abs().max())
    assert causal < 1e-6
    metrics.extend(score(Y, P, sp["test"], name))
    i, t = np.where(sp["selection"])
    records.append(
        {
            "model": name,
            "best_epoch": bestepoch,
            "selection_log_loss": float(log_loss(Y[i], P[i, t], labels=range(12))),
            "temperature": temp,
            "trace": trace,
            "causal_max_logit_difference": causal,
        }
    )
    print("COMPLETE", name, bestepoch, temp, flush=True)
(O / "sequence-results.json").write_text(
    json.dumps({"metrics": metrics, "selection": records}, indent=2)
)
print("SEQUENCE COMPARISONS COMPLETE", flush=True)
