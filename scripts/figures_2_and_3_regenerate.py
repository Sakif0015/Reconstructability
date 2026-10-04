# =============================================================================
# FIGURES 2 & 3 — regenerated from saved JSONs. No retraining, no GPU, no torch.
# Needs only: python, numpy, matplotlib, and the two result files.
#
#   Fig 2  <- dose_response_v2.json      (dose-response, 7 alphas x 3 seeds)
#   Fig 3  <- camelyon17_multiseed.json  (per-hospital tau, with h1/h4 annotated)
#
# Run it anywhere — laptop, Colab, or a fresh Kaggle notebook with the two JSONs
# attached. Put the JSONs next to the script, or edit the two paths below.
#
# Outputs, at IEEE column widths, vector PDF + PNG preview:
#   fig2_dose_response.pdf        3.3in — single column, tau panel only (USE THIS)
#   fig2_dose_response_wide.pdf   6.9in — two panels, if you have double-column room
#   fig3_camelyon_tau.pdf         3.3in — single column, h1/h4 bracket annotated
# =============================================================================

import json, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

DOSE_JSON = "dose_response_v2.json"
LADDER_JSON = "R_ladder_final.json"   # final 20-restart ladder; overrides the R in DOSE_JSON
CAM_JSON = "camelyon17_multiseed.json"
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

# --- publication styling ------------------------------------------------------
matplotlib.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.linewidth": 0.7, "grid.linewidth": 0.5, "lines.linewidth": 1.2,
    "pdf.fonttype": 42, "ps.fonttype": 42,        # embed real fonts, not bitmaps
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
C = {"FedAvg": "#555555", "Random policy": "#1f77b4", "Learned (\u03bb=0.001)": "#ff7f0e"}
LBL = {"FedAvg": "FedAvg", "Random policy": "Random policy",
       "Learned (\u03bb=0.001)": "Learned ($\\lambda$=0.001)"}


def _save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf")
    fig.savefig(f"{OUT}/{name}.png", dpi=220)
    print(f"  saved -> {OUT}/{name}.pdf  (+ .png preview)")


# =============================================================================
# FIGURE 2 — dose-response
# =============================================================================
def figure2():
    d = json.load(open(DOSE_JSON))
    seeds = d["seeds"]; arms = list(C)

    # The dose-response JSON stores the R values from the run that produced it
    # (5-restart estimator). The final ladder supersedes them: same sites, same
    # ordering, tighter upper bound. Prefer it when present so the figure's
    # x-axis matches Table 1.
    if os.path.exists(LADDER_JSON):
        L = json.load(open(LADDER_JSON))
        R = np.array([L[str(a)]["R"] for a in d["alphas"]])
        print(f"  x-axis from {LADDER_JSON}: " + " ".join(f"{v:.1f}" for v in R))
        old = np.array(d["R"])
        if np.abs(R - old).max() > 0.05:
            print("  (superseded old ladder: " + " ".join(f"{v:.1f}" for v in old) + ")")
    else:
        R = np.array(d["R"])
        print(f"  WARNING: {LADDER_JSON} not found -- using the older R values "
              f"stored in {DOSE_JSON}. Figure 2 will not match Table 1.")
    acc = {a: np.array([d["acc"][f"{a}|{s}"] for s in seeds]) for a in arms}   # (seed, alpha)
    adv = {a: acc[a] - acc["FedAvg"] for a in arms if a != "FedAvg"}

    # ---- single-column version: the tau panel only (this is the claim) -------
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    for a in adv:
        ax.errorbar(R, adv[a].mean(0), yerr=adv[a].std(0, ddof=1),
                    marker="o", ms=3.2, capsize=2.5, capthick=0.8,
                    color=C[a], label=LBL[a])
    ax.axhline(0, color="k", lw=0.7)
    ax.invert_xaxis()
    ax.set_xlabel("reconstructability $R$ of held-out shift (%)")
    ax.set_ylabel("relative robustness $\\tau$ (pts)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="upper right")
    ax.annotate("in-family $\\longrightarrow$ out-of-family",
                xy=(0.5, -0.255), xycoords="axes fraction",
                fontsize=6.5, color="0.45", ha="center")
    _save(fig, "fig2_dose_response"); plt.show()

    # ---- wide version: accuracy + tau ---------------------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.9, 2.6))
    for a in arms:
        a1.errorbar(R, acc[a].mean(0), yerr=acc[a].std(0, ddof=1), marker="o",
                    ms=3.2, capsize=2.5, capthick=0.8, color=C[a], label=LBL[a])
    a1.invert_xaxis(); a1.grid(alpha=0.3); a1.legend(frameon=False)
    a1.set_xlabel("reconstructability $R$ (%)")
    a1.set_ylabel("held-out accuracy (%)")
    a1.set_title("(a) accuracy", loc="left")
    for a in adv:
        a2.errorbar(R, adv[a].mean(0), yerr=adv[a].std(0, ddof=1), marker="o",
                    ms=3.2, capsize=2.5, capthick=0.8, color=C[a], label=LBL[a])
    a2.axhline(0, color="k", lw=0.7)
    a2.invert_xaxis(); a2.grid(alpha=0.3); a2.legend(frameon=False)
    a2.set_xlabel("reconstructability $R$ (%)")
    a2.set_ylabel("relative robustness $\\tau$ (pts)")
    a2.set_title("(b) benefit over FedAvg", loc="left")
    fig.tight_layout()
    _save(fig, "fig2_dose_response_wide"); plt.show()

    for a in adv:
        m = adv[a].mean(0)
        print(f"  {LBL[a]:26s} tau {m[0]:+.1f} -> {m[-1]:+.1f}  (loses {100*(m[0]-m[-1])/m[0]:.0f}%)")


# =============================================================================
# FIGURE 3 — Camelyon17 per-hospital tau, with the h1/h4 pair annotated
# =============================================================================
def figure3():
    d = json.load(open(CAM_JSON))
    H = [str(h) for h in d["valid_folds"]]; seeds = d["seeds"]
    R = np.array([d["R"][h] for h in H])
    arms = ["Random policy", "Learned (\u03bb=0.001)"]
    tau = {a: np.array([[d["acc"][f"{h}|{a}|{s}"] - d["acc"][f"{h}|FedAvg|{s}"]
                         for s in seeds] for h in H]) for a in arms}

    fig, ax = plt.subplots(figsize=(3.3, 2.9))
    x = np.arange(len(H)); w = 0.36
    for k, a in enumerate(arms):
        ax.bar(x + (k - 0.5) * w, tau[a].mean(1), w,
               yerr=tau[a].std(1, ddof=1), capsize=2.5,
               error_kw=dict(lw=0.8, capthick=0.8),
               color=C[a], label=LBL[a], edgecolor="white", linewidth=0.4)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"H{h}\n$R$={R[i]:.0f}%" for i, h in enumerate(H)])
    ax.set_ylabel("relative robustness $\\tau$ (pts)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(frameon=False, ncol=2, loc="lower left",
              bbox_to_anchor=(0, 1.02, 1, 0.12), mode="expand",
              borderaxespad=0, handlelength=1.4, columnspacing=1.2)

    # --- the h1/h4 bracket: the paper's single most important visual claim ----
    i1, i4 = H.index("1"), H.index("4")
    top = max((tau[a].mean(1) + tau[a].std(1, ddof=1)).max() for a in arms)
    y = top + 1.6
    ax.plot([x[i1], x[i1], x[i4], x[i4]], [y - 0.7, y, y, y - 0.7],
            lw=0.9, color="0.25", clip_on=False)
    dt = tau["Random policy"].mean(1)[i4] - tau["Random policy"].mean(1)[i1]
    ax.text((x[i1] + x[i4]) / 2, y + 0.4,
            f"same $R$=96.3%,  $\\Delta\\tau$={dt:.1f} pts,  $p$=0.0009",
            ha="center", va="bottom", fontsize=6.8, color="0.15", clip_on=False)
    ax.set_ylim(top=y + 2.6)

    fig.tight_layout()
    _save(fig, "fig3_camelyon_tau"); plt.show()

    for a in arms:
        m = tau[a].mean(1)
        print(f"  {LBL[a]:26s} tau = {m.mean():+.2f} +- {m.std(ddof=1):.2f} across folds")
    print(f"  h1 vs h4 (Random): {tau['Random policy'].mean(1)[i1]:+.2f} vs "
          f"{tau['Random policy'].mean(1)[i4]:+.2f}  ->  delta {dt:+.2f}")


if __name__ == "__main__":
    for f, fn in [(DOSE_JSON, figure2), (CAM_JSON, figure3)]:
        if os.path.exists(f):
            print(f"\n{f}:"); fn()
        else:
            print(f"\n{f}: NOT FOUND — skipped. Put it next to this script.")
