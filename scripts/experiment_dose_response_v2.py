# =============================================================================
# §11d — DOSE-RESPONSE v2:  denser alpha grid (fix 2) + 3 seeds (fix 3)
#
# WHY THIS REPLACES THE OLD FIGURE
# --------------------------------
# The v1 dose-response was 4 points from 1 seed. Pearson r = +0.939 sounds
# strong, but at n=4 the critical value for p<0.05 is 0.950 -- so it was
# NOT significant (p = 0.061). Mintun correlates across 512 augmentation
# schemes; Taori across 204 models. A reviewer checks n before reading prose.
#
# This cell fixes both problems:
#   fix 2: 7 alpha points instead of 4   (evaluation is inference-only)
#   fix 3: 3 seeds instead of 1          (gives error bars + per-seed r)
#
# PRIMARY STATISTIC is the exact monotonicity test, not Pearson r. It is
# distribution-free, assumes no normality, and does not lean on n the way r
# does: under a null of random ordering, P(strictly monotone) = 1/k!.
#
# Requires in memory: §4 core, §6 data, §7 training, §8 loaders (client_loaders,
# val_loaders, raw, val_idx, TF, CFG, device).
# Cost: 9 training runs (~2.5-3 h at 128px) + ~15 min inference.
# =============================================================================

for _n in ["client_loaders", "raw", "val_idx", "TF", "CFG", "device", "run_federated"]:
    assert _n in globals(), f"missing {_n} -- run cells 1,2,3,4,6,7,8 first"

import itertools
from math import factorial, sqrt

SEEDS_DR = [0, 1, 2]
ALPHAS_DR = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75]   # alpha=1.0 excluded: unmatchable
ROUNDS_DR = 10
ARMS_DR = {"FedAvg": ("fedavg", 0.0),
           "Random policy": ("random_policy", 0.0),
           "Learned (λ=0.001)": ("ours", 0.001)}


# --- self-contained graded-site machinery (no dependency on §11b/§11c state) --
IN_BANK_TARGET = dict(contrast=0.75, gain=0.70, gamma=1.70)


def apply_graded_skew(x, alpha, seed=0, oob_scale=1.0):
    """alpha=0: purely in-bank (contrast/gain/gamma). alpha=1: purely out-of-bank
    (bias field + motion blur + S-curve). oob_scale holds shift MAGNITUDE fixed."""
    a_oob = alpha * oob_scale
    w = 1.0 - alpha
    c = 1.0 + (IN_BANK_TARGET["contrast"] - 1.0) * w
    g = 1.0 + (IN_BANK_TARGET["gain"] - 1.0) * w
    gm = 1.0 + (IN_BANK_TARGET["gamma"] - 1.0) * w
    m = x.mean(dim=[2, 3], keepdim=True)
    y = torch.clamp(((x - m) * c + m) * g, 1e-4, 1.0) ** gm
    if a_oob > 1e-6:
        y = torch.clamp(_bias_field(y, min(0.95, 0.60 * a_oob), seed), 0, 1)
        L = int(round(13 * a_oob))
        if L >= 3:
            if L % 2 == 0:
                L += 1
            y = torch.clamp(_motion_blur(y, min(L, 21), 30.0), 0, 1)
        y = _s_curve(y, min(1.2, 0.70 * a_oob))
    return torch.clamp(y, 0, 1)


class GradedSiteDataset(Dataset):
    def __init__(self, base, gi, transform, alpha, seed=0, oob_scale=1.0):
        self.base, self.gi, self.transform = base, list(gi), transform
        self.alpha, self.seed, self.k = alpha, seed, oob_scale

    def __len__(self):
        return len(self.gi)

    def __getitem__(self, i):
        img, lab = self.base[self.gi[i]]
        return apply_graded_skew(self.transform(img).unsqueeze(0), self.alpha,
                                 self.seed, self.k)[0], lab


def calibrate_alphas(probe, alphas, seed=0):
    """Binary-search an oob_scale per alpha so MSE-from-clean is constant.
    Removes the magnitude confound: the x-axis then varies ONLY in family."""
    target = F.mse_loss(apply_graded_skew(probe, 0.0, seed), probe).item()
    out = {}
    for a in alphas:
        if a < 1e-6:
            out[a] = 1.0
            continue
        lo, hi = 0.05, 25.0
        for _ in range(26):
            mid = 0.5 * (lo + hi)
            if F.mse_loss(apply_graded_skew(probe, a, seed, mid), probe).item() < target:
                lo = mid
            else:
                hi = mid
        out[a] = 0.5 * (lo + hi)
    return target, out


def bank_reconstructability(clean, skewed, steps=400, lr=0.05, restarts=5):
    """Best-of-N optimisation of the bank's continuous parameters against clean.
    Upper bound on what any policy over this bank could achieve. Differentiable
    blur (an earlier .item() call detached it and pinned v['blur'] at init)."""
    B = skewed.shape[0]
    g = torch.Generator().manual_seed(0)
    inits = [{op: float(DEFAULT_LEVELS[op][IDENTITY_IDX[op]]) for op in OP_ORDER}]
    for _ in range(restarts - 1):
        inits.append({op: float(DEFAULT_LEVELS[op][int(torch.randint(
            0, len(DEFAULT_LEVELS[op]), (1,), generator=g))]) for op in OP_ORDER})
    residuals = []
    for init in inits:
        v = {op: torch.tensor(init[op], device=clean.device, requires_grad=True)
             for op in OP_ORDER}
        opt = torch.optim.Adam(list(v.values()), lr=lr)
        loss = None
        for _ in range(steps):
            y = skewed
            for op in OP_ORDER:
                s = v[op].clamp(-1, 3)
                s4 = s.view(1, 1, 1, 1)
                if op == "gain":
                    y = y * s4
                elif op == "bias":
                    y = y + s4
                elif op == "contrast":
                    mm = y.mean([2, 3], keepdim=True); y = (y - mm) * s4 + mm
                elif op == "gamma":
                    y = y.clamp(1e-4, 1.0) ** s4.clamp(0.1, 3)
                elif op == "saturation":
                    gg = y.mean(1, keepdim=True); y = gg + (y - gg) * s4
                elif op == "sharpen":
                    y = y + s4 * (y - _blur(y, 1.0))
                elif op == "blur":
                    y = _blur_per_sample(y, s.clamp(0.01, 2.5).expand(B))
                y = y.clamp(0, 1)
            loss = F.mse_loss(y, clean)
            opt.zero_grad(); loss.backward(); opt.step()
        residuals.append(loss.item())
    m0, m1 = F.mse_loss(skewed, clean).item(), min(residuals)
    return m0, m1, 100.0 * (m0 - m1) / max(m0, 1e-9), max(residuals) - min(residuals)


# --- 1. build the x-axis (no models involved) --------------------------------
_probe = torch.stack([TF(raw[i][0]) for i in val_idx[:12].tolist()]).to(device)
target_mse, OOB_SCALE = calibrate_alphas(_probe, ALPHAS_DR)
print(f"magnitude-matched to MSE ~= {target_mse:.5f}\n")
print(f"{'alpha':>6s} {'oob_scale':>10s} {'MSE':>9s} {'residual':>9s} {'R (%)':>8s} {'spread':>9s}")
recon = {}
for a in ALPHAS_DR:
    sk = apply_graded_skew(_probe, a, 0, OOB_SCALE[a])
    m0, m1, pct, spr = bank_reconstructability(_probe, sk)
    recon[a] = dict(mse=m0, resid=m1, pct=pct, spread=spr)
    print(f"{a:6.3f} {OOB_SCALE[a]:10.2f} {m0:9.5f} {m1:9.5f} {pct:7.1f}% {spr:9.5f}")

# drop any alpha whose magnitude could not be matched within 8%
KEEP_DR = [a for a in ALPHAS_DR
           if abs(recon[a]["mse"] - recon[0.0]["mse"]) / recon[0.0]["mse"] <= 0.08]
dropped = [a for a in ALPHAS_DR if a not in KEEP_DR]
if dropped:
    print(f"\ndropped (magnitude unmatchable): {dropped}")
xs_dr = [recon[a]["pct"] for a in KEEP_DR]
mono_x = all(xs_dr[i] >= xs_dr[i + 1] - 1.0 for i in range(len(xs_dr) - 1))
print(f"\nretained alphas: {KEEP_DR}")
print(f"R values: {[round(x,1) for x in xs_dr]}   monotonic in alpha: {mono_x}")
assert mono_x, "x-axis not monotonic -- inspect the restart spreads above"


# --- 2. train 3 methods x 3 seeds --------------------------------------------
if "dose_models" not in globals():
    dose_models = {}
for label, (method, le) in ARMS_DR.items():
    for s in SEEDS_DR:
        key = (label, s)
        if key in dose_models:
            continue
        dose_models[key] = run_federated(
            client_loaders, val_loaders, device,
            {**CFG, "method": method, "rounds": ROUNDS_DR, "lambda_ent": le, "seed": s},
            verbose=False, probe_every=0)
        print(f"trained {label:20s} seed={s}")


# --- 3. evaluate every model on every retained alpha (inference only) --------
graded_loaders = {a: DataLoader(GradedSiteDataset(raw, val_idx.tolist(), TF, a, 0,
                                                  OOB_SCALE[a]),
                                batch_size=CFG["batch_size"], num_workers=0)
                  for a in KEEP_DR}

acc = {}          # acc[(label, seed)] = list over KEEP_DR
for (label, s), o in dose_models.items():
    row = []
    for a in KEEP_DR:
        v, _, _ = evaluate(o["model"], graded_loaders[a], device, None,
                           CFG["levels"] if "levels" in CFG else DEFAULT_LEVELS,
                           CFG["nclass"])
        row.append(100 * v)
    acc[(label, s)] = row
    print(f"evaluated {label:20s} seed={s}: " + " ".join(f"{x:5.1f}" for x in row))


# --- 4. statistics ------------------------------------------------------------
adv = {}          # adv[(label, seed)] = advantage over FedAvg, same seed (paired)
for label in ARMS_DR:
    if label == "FedAvg":
        continue
    for s in SEEDS_DR:
        adv[(label, s)] = [a - b for a, b in zip(acc[(label, s)], acc[("FedAvg", s)])]

print("\n" + "=" * 78)
print("PRIMARY: exact monotonicity test (distribution-free)")
print(f"  k = {len(KEEP_DR)} points -> P(strictly decreasing by chance) = 1/{len(KEEP_DR)}! "
      f"= {1/factorial(len(KEEP_DR)):.2e}")
for label in ARMS_DR:
    if label == "FedAvg":
        continue
    hits = sum(all(adv[(label, s)][i] > adv[(label, s)][i + 1]
                   for i in range(len(KEEP_DR) - 1)) for s in SEEDS_DR)
    mean_adv = [float(np.mean([adv[(label, s)][i] for s in SEEDS_DR]))
                for i in range(len(KEEP_DR))]
    mono_mean = all(mean_adv[i] > mean_adv[i + 1] for i in range(len(KEEP_DR) - 1))
    print(f"  {label:20s} monotone in {hits}/{len(SEEDS_DR)} seeds | "
          f"seed-mean curve monotone: {mono_mean}"
          + (f"  p = {1/factorial(len(KEEP_DR)):.2e}" if mono_mean else ""))

print("\nSECONDARY: Pearson r (report WITH the point count)")
tc = None
try:
    from scipy import stats as _st
    tc = _st.t.ppf(0.975, len(KEEP_DR) - 2)
except Exception:
    pass
r_crit = (tc / sqrt(tc ** 2 + len(KEEP_DR) - 2)) if tc else float("nan")
print(f"  n = {len(KEEP_DR)} points -> critical |r| for p<0.05 is {r_crit:.3f}")
for label in ARMS_DR:
    if label == "FedAvg":
        continue
    rs = []
    for s in SEEDS_DR:
        v = adv[(label, s)]
        rs.append(float(np.corrcoef(xs_dr, v)[0, 1]) if np.std(v) > 1e-9 else float("nan"))
    rs_ok = [x for x in rs if not np.isnan(x)]
    mean_adv = [float(np.mean([adv[(label, s)][i] for s in SEEDS_DR]))
                for i in range(len(KEEP_DR))]
    r_mean = float(np.corrcoef(xs_dr, mean_adv)[0, 1])
    rng = max(mean_adv) - min(mean_adv)
    if len(rs_ok) > 1:
        rtxt = f"{np.mean(rs_ok):+.3f} ± {np.std(rs_ok, ddof=1):.3f}"
    elif rs_ok:
        rtxt = f"{rs_ok[0]:+.3f}"
    else:
        rtxt = "undefined (flat)"
    print(f"  {label:20s} r(seed-mean) = {r_mean:+.3f} | per-seed r = {rtxt}"
          f" | advantage {mean_adv[0]:+.1f} -> {mean_adv[-1]:+.1f} (range {rng:.1f})")
    if rng < 3.0:
        print(f"{'':22s} range is tiny -- FLAT, not a decay; do not report r as a trend")

print("\n" + "=" * 78)
print(f"{'R (%)':>7s} " + " ".join(f"{l:>22s}" for l in ARMS_DR))
for i, a in enumerate(KEEP_DR):
    cells = []
    for label in ARMS_DR:
        v = [acc[(label, s)][i] for s in SEEDS_DR]
        cells.append(f"{np.mean(v):9.2f}±{np.std(v, ddof=1):4.2f}       ")
    print(f"{xs_dr[i]:7.1f} " + " ".join(cells))
print("=" * 78)


# --- 5. figure ----------------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
for label in ARMS_DR:
    m = [np.mean([acc[(label, s)][i] for s in SEEDS_DR]) for i in range(len(KEEP_DR))]
    e = [np.std([acc[(label, s)][i] for s in SEEDS_DR], ddof=1) for i in range(len(KEEP_DR))]
    a1.errorbar(xs_dr, m, yerr=e, marker="o", ms=4, capsize=3, label=label)
a1.set_xlabel("reconstructable fraction of the test shift (%)")
a1.set_ylabel("held-out accuracy (%)"); a1.invert_xaxis(); a1.grid(alpha=.3); a1.legend()
a1.set_title(f"Accuracy vs. reconstructability (n={len(SEEDS_DR)} seeds)")

for label in ARMS_DR:
    if label == "FedAvg":
        continue
    m = [np.mean([adv[(label, s)][i] for s in SEEDS_DR]) for i in range(len(KEEP_DR))]
    e = [np.std([adv[(label, s)][i] for s in SEEDS_DR], ddof=1) for i in range(len(KEEP_DR))]
    a2.errorbar(xs_dr, m, yerr=e, marker="o", ms=4, capsize=3, label=f"{label} − FedAvg")
a2.axhline(0, c="k", lw=1)
a2.set_xlabel("reconstructable fraction of the test shift (%)")
a2.set_ylabel("advantage over FedAvg (pts)"); a2.invert_xaxis(); a2.grid(alpha=.3); a2.legend()
a2.set_title("Benefit decays outside the augmentation family")
plt.tight_layout(); plt.show()


# --- 6. persist ---------------------------------------------------------------
import json as _json, os as _os
_dst = "/kaggle/working" if _os.path.isdir("/kaggle/working") else "."
with open(f"{_dst}/dose_response_v2.json", "w") as f:
    _json.dump({"alphas": KEEP_DR, "R": xs_dr, "seeds": SEEDS_DR,
                "oob_scale": {str(k): v for k, v in OOB_SCALE.items()},
                "recon": {str(k): {kk: float(vv) for kk, vv in v.items()}
                          for k, v in recon.items()},
                "acc": {f"{l}|{s}": v for (l, s), v in acc.items()},
                "adv": {f"{l}|{s}": v for (l, s), v in adv.items()}}, f, indent=2)
print(f"saved -> {_dst}/dose_response_v2.json")
