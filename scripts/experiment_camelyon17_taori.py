# =============================================================================
# §15 — CAMELYON17 under Taori et al. (2020) robustness metrics
#
# WHY THIS CELL EXISTS
# --------------------
# Taori's first stated conclusion is that robustness measurements must control
# for accuracy: apparent robustness gains often just reflect a model doing
# better on the standard test set. Our Camelyon17 analysis used raw accuracy
# differences and then noticed post hoc that baseline HEADROOM predicted them
# (r = +0.93) -- i.e. we hit exactly the confound Taori warns about. Using their
# metrics turns that liability into a citation.
#
# Taori defines two quantities (their Sec. 2):
#   relative  robustness   tau(f') = acc2(f') - acc2(f)
#   effective robustness   rho(f)  = acc2(f) - beta(acc1(f))
# where acc1 = standard (in-distribution) accuracy, acc2 = shifted accuracy,
# and beta is a baseline fitted over models WITHOUT a robustness intervention.
#
# WHAT WE CAN AND CANNOT DO WITH THE DATA WE HAVE
# -----------------------------------------------
#   tau  -> COMPUTABLE NOW. Our "advantage over FedAvg" is literally tau.
#   rho  -> NOT COMPUTABLE. The fold loop set val_loaders = {"heldout": ...},
#           so acc1 was never measured. PART B below fixes that; it needs a re-run.
#
# PART A runs anywhere in seconds (numbers hardcoded from the completed run).
# PART B is optional and needs the full Camelyon pipeline (~2 h + download).
# =============================================================================

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# PART A — relative robustness (free, no GPU, no session state)
# ---------------------------------------------------------------------------
HOSP = [0, 1, 2, 3, 4]
R_hat = np.array([91.8, 96.3, 54.1, 77.3, 96.3])     # distributional reconstructability
acc2 = {                                              # held-out (shifted) accuracy
    "FedAvg":            np.array([92.70, 85.17, 72.27, 87.30, 76.67]),
    "Random policy":     np.array([89.80, 87.63, 84.17, 90.07, 92.20]),
    "Learned (λ=0.001)": np.array([85.63, 82.60, 85.53, 89.37, 82.83]),
}
INTERVENTIONS = ["Random policy", "Learned (λ=0.001)"]

tau = {m: acc2[m] - acc2["FedAvg"] for m in INTERVENTIONS}
headroom = 100.0 - acc2["FedAvg"]

print("=" * 78)
print("PART A — relative robustness tau (Taori Sec. 2), n = 5 folds")
print("=" * 78)
print(f"{'hosp':>5s} {'R (%)':>7s} {'FedAvg':>8s} {'headroom':>9s} "
      + " ".join(f"{'tau '+m.split()[0]:>14s}" for m in INTERVENTIONS))
for i, h in enumerate(HOSP):
    print(f"{h:5d} {R_hat[i]:7.1f} {acc2['FedAvg'][i]:8.2f} {headroom[i]:9.2f} "
          + " ".join(f"{tau[m][i]:+14.2f}" for m in INTERVENTIONS))
print("-" * 78)
print(f"{'mean':>5s} {'':7s} {acc2['FedAvg'].mean():8.2f} {headroom.mean():9.2f} "
      + " ".join(f"{tau[m].mean():+14.2f}" for m in INTERVENTIONS))

r_crit = (lambda t: t / np.sqrt(t ** 2 + 3))(stats.t.ppf(0.975, 3))
print(f"\ncorrelations (n=5, critical |r| for p<0.05 = {r_crit:.3f}):")
for m in INTERVENTIONS:
    rR, pR = stats.pearsonr(R_hat, tau[m])
    rH, pH = stats.pearsonr(headroom, tau[m])
    print(f"  {m:20s} tau vs R        r = {rR:+.3f}  p = {pR:.4f}"
          + ("  sig" if pR < 0.05 else "  n.s."))
    print(f"  {'':20s} tau vs headroom r = {rH:+.3f}  p = {pH:.4f}"
          + ("  sig" if pH < 0.05 else "  n.s."))

print("\nOne-sample t vs zero (is tau different from no benefit at all?):")
for m in INTERVENTIONS:
    t_, p_ = stats.ttest_1samp(tau[m], 0.0)
    print(f"  {m:20s} tau = {tau[m].mean():+6.2f} ± {tau[m].std(ddof=1):.2f}"
          f"  t = {t_:+.2f}  p = {p_:.4f}" + ("  sig" if p_ < 0.05 else "  n.s."))

print("\nSYNTHETIC vs NATURAL, same method, same op bank:")
print(f"  synthetic in-bank site (3 seeds) : tau = +51.88")
print(f"  natural Camelyon17 (5 hospitals) : tau = {tau['Random policy'].mean():+6.2f}")

print("""
READ THIS CAREFULLY BEFORE WRITING:
  tau vs headroom being the significant correlation is Taori's confound, not a
  finding of ours. They introduced effective robustness precisely because
  standard accuracy acts as a confounder on tau. Do NOT present 'headroom
  predicts benefit' as a contribution -- present it as the reason we report
  effective robustness, and cite Taori Sec. 2.
""")

# ---------------------------------------------------------------------------
# PART B — effective robustness (needs a re-run; set RUN_PART_B = True)
# ---------------------------------------------------------------------------
# The only change from the original fold loop: each fold now holds out an
# IN-DISTRIBUTION split from the TRAINING hospitals, giving acc1 per model.
# beta is then fitted over the FedAvg points (the no-intervention models) and
# rho = acc2 - beta(acc1) measures benefit BEYOND what in-distribution accuracy
# already buys.
#
# Requires: SITES, HOSPITALS, run_federated, evaluate, device  (i.e. the
# Camelyon setup cell + main cell must have run in this session).
# Cost: 15 training runs (~2 h at 96 px) if you keep 3 methods x 5 folds.

RUN_PART_B = False

if RUN_PART_B:
    assert "SITES" in globals(), "run the Camelyon17 setup + main cell first"
    from torch.utils.data import DataLoader, Subset

    CFG_ER = dict(rounds=10, local_epochs=1, batch_size=64, nclass=2,
                  lr_model=1e-3, lr_policy=1e-3, tau=1.0, lambda_ent=0.001, seed=0)
    ARMS_ER = {"FedAvg": ("fedavg", 0.0),
               "Random policy": ("random_policy", 0.0),
               "Learned (λ=0.001)": ("ours", 0.001)}
    ID_FRAC = 0.15                      # in-distribution holdout per training hospital

    er = {}
    for h_out in HOSPITALS:
        tr_h = [h for h in HOSPITALS if h != h_out]
        g = torch.Generator().manual_seed(0)
        tr_loaders, id_parts = [], []
        for h in tr_h:
            n = len(SITES[h])
            perm = torch.randperm(n, generator=g).tolist()
            cut = int(ID_FRAC * n)
            id_parts.append(Subset(SITES[h], perm[:cut]))          # in-distribution eval
            tr_loaders.append(DataLoader(Subset(SITES[h], perm[cut:]),
                                         batch_size=CFG_ER["batch_size"],
                                         shuffle=True, drop_last=True, num_workers=0))
        vl = {"in_dist": DataLoader(torch.utils.data.ConcatDataset(id_parts),
                                    batch_size=CFG_ER["batch_size"], num_workers=0),
              "heldout": DataLoader(SITES[h_out], batch_size=CFG_ER["batch_size"],
                                    num_workers=0)}
        row = {}
        for label, (method, le) in ARMS_ER.items():
            o = run_federated(tr_loaders, vl, device,
                              {**CFG_ER, "method": method, "lambda_ent": le},
                              verbose=False, probe_every=0)
            hh = o["history"][-1]
            row[label] = (hh["in_dist_acc"], hh["heldout_acc"])    # (acc1, acc2)
        er[h_out] = row
        print(f"hospital {h_out}: " + "  ".join(
            f"{k}: acc1={v[0]:5.1f} acc2={v[1]:5.1f}" for k, v in row.items()))

    # fit beta on the no-intervention (FedAvg) points, then rho = acc2 - beta(acc1)
    a1 = np.array([er[h]["FedAvg"][0] for h in HOSPITALS])
    a2 = np.array([er[h]["FedAvg"][1] for h in HOSPITALS])
    slope, icept, rb, pb, se = stats.linregress(a1, a2)
    beta = lambda x: slope * x + icept
    print(f"\nbeta fitted on FedAvg: acc2 = {slope:.3f}*acc1 + {icept:.2f} "
          f"(r={rb:+.3f}, p={pb:.4f}, n=5)")
    if pb > 0.05:
        print("  !! beta fit is not significant at n=5 -- rho below is indicative only")

    print(f"\n{'hosp':>5s} {'R (%)':>7s} " + " ".join(f"{'rho '+m.split()[0]:>14s}"
                                                     for m in INTERVENTIONS))
    rho = {m: [] for m in INTERVENTIONS}
    for i, h in enumerate(HOSPITALS):
        vals = []
        for m in INTERVENTIONS:
            v = er[h][m][1] - beta(er[h][m][0])
            rho[m].append(v); vals.append(v)
        print(f"{h:5d} {R_hat[i]:7.1f} " + " ".join(f"{v:+14.2f}" for v in vals))
    for m in INTERVENTIONS:
        v = np.array(rho[m])
        rR, pR = stats.pearsonr(R_hat, v)
        t_, p_ = stats.ttest_1samp(v, 0.0)
        print(f"\n  {m:20s} rho = {v.mean():+.2f} ± {v.std(ddof=1):.2f}"
              f"  (t={t_:+.2f}, p={p_:.4f})")
        print(f"  {'':20s} rho vs R: r = {rR:+.3f}, p = {pR:.4f}"
              + ("  sig" if pR < 0.05 else "  n.s."))

    import json as _json, os as _os
    _dst = "/kaggle/working" if _os.path.isdir("/kaggle/working") else "."
    with open(f"{_dst}/camelyon17_effective_robustness.json", "w") as f:
        _json.dump({"beta": {"slope": slope, "intercept": icept, "r": rb, "p": pb},
                    "acc": {str(h): {k: list(v) for k, v in er[h].items()} for h in er},
                    "rho": {m: list(map(float, rho[m])) for m in INTERVENTIONS},
                    "tau": {m: list(map(float, tau[m])) for m in INTERVENTIONS}}, f, indent=2)
    print(f"\nsaved -> {_dst}/camelyon17_effective_robustness.json")
else:
    print("PART B skipped (RUN_PART_B = False).")
    print("  Set it True only if you have ~2 h GPU and the Camelyon session is live.")
