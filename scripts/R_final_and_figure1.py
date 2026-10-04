# =============================================================================
# FINAL R LADDER + FIGURE 1, from one fixed estimator.
#
# THE BUG THIS FIXES
# ------------------
# Every previous version recorded `loss.item()` from the LAST optimization step
# of each restart, not the best loss reached during it. On a flat surface Adam
# at lr=0.05 oscillates near the optimum, so running longer could record a
# WORSE residual. That is why 20 restarts x 600 steps returned R=44.4% at
# alpha=0.75 where 5 x 400 returned 47.4% -- mathematically impossible for a
# true upper bound, since the 5 inits are a subset of the 20.
#
# Fix: track the running minimum within each restart (two lines). R is then
# monotone non-decreasing in both steps and restarts, as the definition
# requires. Also adds cosine lr decay so late steps stop oscillating.
#
# This script computes the ladder AND draws Figure 1 with the SAME estimator,
# so the figure annotation and Table 1 cannot disagree.
#
# Run after figure1_prereq.py. No training. ~5-8 min at 128px.
# =============================================================================

import torch, torch.nn.functional as F, json, os
import numpy as np
import matplotlib.pyplot as plt

for _n in ["apply_graded_skew", "OOB_SCALE", "TF", "raw", "val_idx", "device",
           "OP_ORDER", "DEFAULT_LEVELS", "IDENTITY_IDX", "_blur", "_blur_per_sample"]:
    assert _n in globals(), f"missing {_n} -- run figure1_prereq.py first"

RESTARTS = 20
STEPS = 600
PROBE_N = 12            # same probe as Table 1 -- do not change
FIG1_ALPHAS = [0.0, 0.75]
FIG1_SHOW = 3           # which of the 12 probe images to DISPLAY (0..11)
RESID_GAIN = 5.0

OLD_TABLE1 = {0.0: 99.9, 0.125: 93.8, 0.25: 83.1, 0.375: 75.3,
              0.5: 71.7, 0.625: 61.2, 0.75: 47.4}


def reconstruct(clean, skewed, steps=STEPS, lr=0.05, restarts=RESTARTS):
    """Returns (best_image, mse_shift, best_residual, R, restart_spread).

    Records the BEST loss along each restart's trajectory, not the final one."""
    B = skewed.shape[0]
    g = torch.Generator().manual_seed(0)
    inits = [{op: float(DEFAULT_LEVELS[op][IDENTITY_IDX[op]]) for op in OP_ORDER}]
    for _ in range(restarts - 1):
        inits.append({op: float(DEFAULT_LEVELS[op][int(torch.randint(
            0, len(DEFAULT_LEVELS[op]), (1,), generator=g))]) for op in OP_ORDER})

    per_restart, best_overall, best_img = [], None, None
    for init in inits:
        v = {op: torch.tensor(init[op], device=clean.device, requires_grad=True)
             for op in OP_ORDER}
        opt = torch.optim.Adam(list(v.values()), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        best_here, best_here_img = float("inf"), None
        for _ in range(steps):
            y = skewed
            for op in OP_ORDER:
                s = v[op].clamp(-1, 3); s4 = s.view(1, 1, 1, 1)
                if op == "gain":        y = y * s4
                elif op == "bias":      y = y + s4
                elif op == "contrast":
                    mm = y.mean([2, 3], keepdim=True); y = (y - mm) * s4 + mm
                elif op == "gamma":     y = y.clamp(1e-4, 1.0) ** s4.clamp(0.1, 3)
                elif op == "saturation":
                    gg = y.mean(1, keepdim=True); y = gg + (y - gg) * s4
                elif op == "sharpen":   y = y + s4 * (y - _blur(y, 1.0))
                elif op == "blur":      y = _blur_per_sample(y, s.clamp(0.01, 2.5).expand(B))
                y = y.clamp(0, 1)
            loss = F.mse_loss(y, clean)
            l = loss.item()
            if l < best_here:                      # <-- THE FIX
                best_here, best_here_img = l, y.detach().clone()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        per_restart.append(best_here)
        if best_overall is None or best_here < best_overall:
            best_overall, best_img = best_here, best_here_img

    m0 = F.mse_loss(skewed, clean).item()
    return (best_img, m0, best_overall,
            100.0 * (m0 - best_overall) / max(m0, 1e-9),
            max(per_restart) - min(per_restart))


# --- 1. the ladder ------------------------------------------------------------
probe = torch.stack([TF(raw[i][0]) for i in val_idx[:PROBE_N].tolist()]).to(device)
alphas = sorted(OOB_SCALE)
print(f"restarts={RESTARTS}  steps={STEPS}  probe={PROBE_N}  (best-along-trajectory)\n")
print(f"{'alpha':>6s} {'oob':>6s} {'MSE':>9s} {'residual':>10s} {'R new':>8s} "
      f"{'R old':>8s} {'delta':>7s} {'spread':>10s}")
ladder, cache = {}, {}
for a in alphas:
    sk = apply_graded_skew(probe, a, 0, OOB_SCALE[a])
    img, m0, m1, R, spr = reconstruct(probe, sk)
    ladder[a] = dict(oob_scale=OOB_SCALE[a], mse=m0, residual=m1, R=R, spread=spr)
    if a in FIG1_ALPHAS:
        cache[a] = dict(shifted=sk, rec=img, resid=(probe - img).abs().clamp(0, 1), R=R)
    old = OLD_TABLE1.get(a, float("nan"))
    print(f"{a:6.3f} {OOB_SCALE[a]:6.2f} {m0:9.5f} {m1:10.5f} {R:7.1f}% {old:7.1f}% "
          f"{R-old:+7.1f} {spr:10.5f}")

mono = all(ladder[alphas[i]]["R"] >= ladder[alphas[i+1]]["R"] - 0.5
           for i in range(len(alphas)-1))
bound = all(ladder[a]["R"] >= OLD_TABLE1[a] - 0.05 for a in alphas)
print(f"\nladder monotonic in alpha : {mono}")
print(f"every R >= previous value : {bound}   <-- MUST be True now")
if not bound:
    print("  !! still failing -- do not use these numbers; report back.")

_dst = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
json.dump({str(k): v for k, v in ladder.items()},
          open(f"{_dst}/R_ladder_final.json", "w"), indent=2)
print(f"saved -> {_dst}/R_ladder_final.json")

print("\nTable 1 row for the paper:")
print("  R (%)  " + "  ".join(f"{ladder[a]['R']:.1f}" for a in alphas))

# --- 2. Figure 1, same estimator ---------------------------------------------
def show(ax, t, gain=1.0):
    im = (t[FIG1_SHOW].detach().cpu().permute(1, 2, 0).numpy() * gain).clip(0, 1)
    ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_linewidth(0.6); s_.set_color("0.4")

COLS = ["clean image $x$", "shifted $x'$",
        r"bank's best $B_{\theta^*}(x')$", r"residual $|x - B_{\theta^*}(x')|$"]
ROWLAB = [r"$\bf{in\text{-}family}$" + "\nshift",
          r"$\bf{out\text{-}of\text{-}family}$" + "\nshift"]

fig, axes = plt.subplots(2, 4, figsize=(6.9, 3.7))
for r, a in enumerate(FIG1_ALPHAS):
    row = cache[a]
    show(axes[r][0], probe); show(axes[r][1], row["shifted"])
    show(axes[r][2], row["rec"]); show(axes[r][3], row["resid"], gain=RESID_GAIN)
    axes[r][0].set_ylabel(ROWLAB[r], fontsize=9, labelpad=8)
    axes[r][3].text(0.05, 0.95, rf"$R = {row['R']:.1f}\%$",
                    transform=axes[r][3].transAxes, ha="left", va="top",
                    fontsize=11, fontweight="bold",
                    color=("#1a7f37" if row["R"] > 90 else "#b3261e"),
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec="0.6", lw=0.5, alpha=0.92))
for c, t in enumerate(COLS):
    axes[0][c].set_title(t, fontsize=8.5, pad=5)
axes[1][3].text(0.5, -0.06, f"residual shown $\\times${RESID_GAIN:.0f}",
                transform=axes[1][3].transAxes, ha="center", va="top",
                fontsize=7, color="0.35")
plt.tight_layout(rect=[0.02, 0.03, 1, 1], h_pad=1.2)
plt.savefig(f"{_dst}/fig1_reconstructability.pdf", bbox_inches="tight", dpi=300)
plt.savefig(f"{_dst}/fig1_reconstructability.png", bbox_inches="tight", dpi=200)
plt.show()

ra = cache[FIG1_ALPHAS[0]]["resid"][FIG1_SHOW].mean().item()
rb = cache[FIG1_ALPHAS[1]]["resid"][FIG1_SHOW].mean().item()
print(f"\nfigure check: mean |residual| in-family {ra:.5f}  out-of-family {rb:.5f}"
      f"  ratio {rb/max(ra,1e-9):.1f}x   (want >= 10x)")
print(f"figure annotates R = {cache[FIG1_ALPHAS[0]]['R']:.1f}% and "
      f"{cache[FIG1_ALPHAS[1]]['R']:.1f}%  -- matches the ladder above by construction")
print(f"saved -> {_dst}/fig1_reconstructability.pdf")
