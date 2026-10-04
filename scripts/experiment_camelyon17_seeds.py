# =============================================================================
# §16 — CAMELYON17 with SEEDS: per-fold error bars + variance decomposition
#
# The single-seed run gave tau = +5.95 ± 7.55 (n.s., n=5 folds). That +/- is
# BETWEEN-FOLD heterogeneity (hospital 0 = -2.90, hospital 4 = +15.53), not
# measurement noise. Seeds therefore tighten each point but should NOT be
# expected to change significance -- this cell measures that explicitly with a
# variance decomposition, so the limitation is quantified rather than asserted.
#
# What you gain: error bars per fold, per-seed correlations, and a defensible
# answer to "n=5, one run each?" in review.
#
# Requires (in a LIVE session): SITES, HOSPITALS, run_federated, device, and
# dist_reconstructability from the Camelyon device-fix cell.
# Cost: 5 folds x 3 methods x 3 seeds = 45 runs, roughly 4.5-6 h at 96 px.
#       Resumable: completed (fold, method, seed) entries are skipped on re-run.
# =============================================================================

for _n in ["SITES", "HOSPITALS", "run_federated", "device", "dist_reconstructability"]:
    assert _n in globals(), f"missing {_n} -- run the Camelyon setup + main + device-fix cells"

import json as _json, os as _os
from scipy import stats

SEEDS_MS = [0, 1, 2]          # drop to [0, 1] if you want ~3 h instead of ~5 h
CFG_MS = dict(rounds=10, local_epochs=1, batch_size=64, nclass=2,
              lr_model=1e-3, lr_policy=1e-3, tau=1.0, lambda_ent=0.001)
ARMS_MS = {"FedAvg": ("fedavg", 0.0),
           "Random policy": ("random_policy", 0.0),
           "Learned (λ=0.001)": ("ours", 0.001)}
PROBE_N_MS = 256

if "acc_ms" not in globals():
    acc_ms = {}        # acc_ms[(hosp, label, seed)] = held-out accuracy
if "R_ms" not in globals():
    R_ms = {}          # R_ms[hosp] = reconstructability (seed-independent)

# --- reconstructability once per fold (does not depend on the training seed) --
for h_out in HOSPITALS:
    if h_out in R_ms:
        continue
    tr_h = [h for h in HOSPITALS if h != h_out]
    src = torch.cat([SITES[h].x[:PROBE_N_MS // len(tr_h)].float() / 255.0
                     for h in tr_h]).to(device)
    tgt = (SITES[h_out].x[:PROBE_N_MS].float() / 255.0).to(device)
    rec = dist_reconstructability(src, tgt)
    R_ms[h_out] = rec
    print(f"hospital {h_out}: R = {rec['R']:5.1f}%  d0/noise = {rec['ratio']:6.1f}  "
          f"{'ok' if rec['valid'] else 'TOO SIMILAR'}")

# --- train --------------------------------------------------------------------
total = len(HOSPITALS) * len(ARMS_MS) * len(SEEDS_MS)
done = 0
for h_out in HOSPITALS:
    tr_h = [h for h in HOSPITALS if h != h_out]
    cl = [DataLoader(SITES[h], batch_size=CFG_MS["batch_size"], shuffle=True,
                     drop_last=True, num_workers=0) for h in tr_h]
    vl = {"heldout": DataLoader(SITES[h_out], batch_size=CFG_MS["batch_size"],
                                num_workers=0)}
    for label, (method, le) in ARMS_MS.items():
        for s in SEEDS_MS:
            done += 1
            if (h_out, label, s) in acc_ms:
                continue
            o = run_federated(cl, vl, device,
                              {**CFG_MS, "method": method, "lambda_ent": le, "seed": s},
                              verbose=False, probe_every=0)
            acc_ms[(h_out, label, s)] = o["history"][-1]["heldout_acc"]
            print(f"[{done:3d}/{total}] hosp {h_out} {label:20s} seed {s}: "
                  f"{acc_ms[(h_out, label, s)]:6.2f}")

# --- assemble -----------------------------------------------------------------
VALID = [h for h in HOSPITALS if R_ms[h]["valid"]]
INTERV = [m for m in ARMS_MS if m != "FedAvg"]
R_vec = np.array([R_ms[h]["R"] for h in VALID])

# tau is PAIRED within (fold, seed): same fold, same seed, intervention vs FedAvg
tau_ms = {m: np.array([[acc_ms[(h, m, s)] - acc_ms[(h, "FedAvg", s)]
                        for s in SEEDS_MS] for h in VALID]) for m in INTERV}
# shape (n_folds, n_seeds)

print("\n" + "=" * 82)
print(f"RELATIVE ROBUSTNESS tau, mean ± std over {len(SEEDS_MS)} seeds")
print("=" * 82)
print(f"{'hosp':>5s} {'R (%)':>7s} {'FedAvg':>14s} "
      + " ".join(f"{('tau '+m.split()[0]):>16s}" for m in INTERV))
for i, h in enumerate(VALID):
    fa = [acc_ms[(h, "FedAvg", s)] for s in SEEDS_MS]
    cells = [f"{tau_ms[m][i].mean():+9.2f}±{tau_ms[m][i].std(ddof=1):5.2f}" for m in INTERV]
    print(f"{h:5d} {R_vec[i]:7.1f} {np.mean(fa):8.2f}±{np.std(fa, ddof=1):4.2f} "
          + " ".join(f"{c:>16s}" for c in cells))
print("-" * 82)
for m in INTERV:
    fm = tau_ms[m].mean(axis=1)                     # fold means
    t_, p_ = stats.ttest_1samp(fm, 0.0)
    print(f"  {m:20s} tau = {fm.mean():+6.2f} ± {fm.std(ddof=1):5.2f} across folds"
          f"   t({len(VALID)-1}) = {t_:+.2f}  p = {p_:.4f}"
          + ("  SIGNIFICANT" if p_ < 0.05 else "  n.s."))

# --- variance decomposition: is the noise between folds, or between seeds? ----
print("\n" + "=" * 82)
print("VARIANCE DECOMPOSITION (why seeds cannot rescue significance)")
print("=" * 82)
fracs = []
for m in INTERV:
    T = tau_ms[m]
    v_between = T.mean(axis=1).var(ddof=1)          # across folds
    v_within = T.var(axis=1, ddof=1).mean()         # across seeds, averaged over folds
    frac = 100 * v_between / max(v_between + v_within, 1e-12)
    fracs.append(frac)
    print(f"  {m:20s} between-fold var = {v_between:7.2f} | "
          f"within-fold (seed) var = {v_within:6.2f} | "
          f"{frac:.0f}% of variance is BETWEEN HOSPITALS")
if np.mean(fracs) >= 60:
    print("\n  -> variance is dominated by real hospital heterogeneity, not run-to-run")
    print("     noise. More seeds shrink the error bars but not the between-fold term")
    print("     the t-test is built on. Camelyon17 has 5 hospitals; n cannot grow.")
else:
    print("\n  -> variance is dominated by RUN-TO-RUN noise, not hospital heterogeneity.")
    print("     Here more seeds DO tighten the fold means and can change significance.")
    print("     Consider adding seeds before concluding anything about tau.")

# --- does R predict tau? (per seed, and on seed means) ------------------------
r_crit = (lambda t: t / np.sqrt(t ** 2 + len(VALID) - 2))(stats.t.ppf(0.975, len(VALID) - 2))
print("\n" + "=" * 82)
print(f"DOES RECONSTRUCTABILITY PREDICT tau?  (n = {len(VALID)} folds, "
      f"critical |r| = {r_crit:.3f})")
print("=" * 82)
for m in INTERV:
    per_seed = [stats.pearsonr(R_vec, tau_ms[m][:, j])[0]
                for j in range(len(SEEDS_MS)) if np.std(tau_ms[m][:, j]) > 1e-9]
    rm, pm = stats.pearsonr(R_vec, tau_ms[m].mean(axis=1))
    print(f"  {m:20s} r(seed-mean) = {rm:+.3f}  p = {pm:.4f}"
          + ("  sig" if pm < 0.05 else "  n.s.")
          + (f"   | per-seed r = {np.mean(per_seed):+.3f} ± {np.std(per_seed, ddof=1):.3f}"
             if len(per_seed) > 1 else "   | per-seed r = undefined (flat)"))

print(f"\nSYNTHETIC vs NATURAL (same method, same op bank):")
print(f"  synthetic in-bank site, 3 seeds : tau = +51.88")
print(f"  natural Camelyon17, {len(VALID)} hospitals : "
      f"tau = {tau_ms['Random policy'].mean(axis=1).mean():+6.2f}")

# --- figure -------------------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
xs = np.arange(len(VALID))
w = 0.35
for k, m in enumerate(INTERV):
    a1.bar(xs + (k - 0.5) * w, tau_ms[m].mean(axis=1), w,
           yerr=tau_ms[m].std(axis=1, ddof=1), capsize=4, label=f"{m} − FedAvg")
a1.axhline(0, c="k", lw=1)
a1.set_xticks(xs); a1.set_xticklabels([f"H{h}\nR={R_ms[h]['R']:.0f}%" for h in VALID])
a1.set_ylabel("relative robustness τ (pts)")
a1.set_title(f"Natural site shift, per hospital ({len(SEEDS_MS)} seeds)")
a1.legend(); a1.grid(alpha=.3, axis="y")

for m in INTERV:
    a2.errorbar(R_vec, tau_ms[m].mean(axis=1), yerr=tau_ms[m].std(axis=1, ddof=1),
                fmt="o", ms=8, capsize=4, label=f"{m} − FedAvg")
a2.axhline(0, c="k", lw=1)
a2.set_xlabel("distributional reconstructability of held-out hospital (%)")
a2.set_ylabel("relative robustness τ (pts)")
a2.set_title("τ does not track R on natural shift")
a2.legend(); a2.grid(alpha=.3)
plt.tight_layout(); plt.show()

# --- persist ------------------------------------------------------------------
_dst = "/kaggle/working" if _os.path.isdir("/kaggle/working") else "."
with open(f"{_dst}/camelyon17_multiseed.json", "w") as f:
    _json.dump({"seeds": SEEDS_MS, "valid_folds": VALID,
                "R": {str(h): float(R_ms[h]["R"]) for h in HOSPITALS},
                "d0_over_noise": {str(h): float(R_ms[h]["ratio"]) for h in HOSPITALS},
                "acc": {f"{h}|{m}|{s}": float(v) for (h, m, s), v in acc_ms.items()},
                "tau": {m: tau_ms[m].tolist() for m in INTERV}}, f, indent=2)
print(f"\nsaved -> {_dst}/camelyon17_multiseed.json")
