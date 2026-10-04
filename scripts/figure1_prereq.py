# =============================================================================
# FIGURE 1 PREREQUISITE — builds apply_graded_skew + OOB_SCALE, nothing else.
#
# Run this INSTEAD of the dose-response v2 cell (§11d). It contains the exact
# top section of that cell — the graded-site machinery and the magnitude
# calibration — and stops before the training loop. No models, no GPU hours.
# Takes a few seconds.
#
# The calibration is pure binary search on MSE (no Adam, no reconstructability),
# so it is fast. Figure 1 computes R itself.
#
# Requires already in memory: TF, raw, val_idx, device, and the three
# out-of-bank corruption helpers _bias_field, _motion_blur, _s_curve.
# =============================================================================

import torch
import torch.nn.functional as F

_base = ["TF", "raw", "val_idx", "device"]
_missing = [n for n in _base if n not in globals()]
assert not _missing, f"missing {_missing} -- run imports, dataset, split, core, data, loaders"

_helpers = ["_bias_field", "_motion_blur", "_s_curve"]
_hmiss = [n for n in _helpers if n not in globals()]
assert not _hmiss, (
    f"missing {_hmiss} -- these are defined in the graded-sites cell (§11b). "
    "Run that cell first, then re-run this one.")

# --- verbatim from the dose-response v2 cell ---------------------------------
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


def calibrate_alphas(probe, alphas, seed=0):
    """Binary-search an oob_scale per alpha so MSE-from-clean is constant.
    Removes the magnitude confound: the axis then varies ONLY in family."""
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


# --- calibrate on the SAME probe the v2 cell used (val_idx[:12]) -------------
ALPHAS_DR = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75]
_probe = torch.stack([TF(raw[i][0]) for i in val_idx[:12].tolist()]).to(device)
target_mse, OOB_SCALE = calibrate_alphas(_probe, ALPHAS_DR)

print(f"magnitude-matched to MSE ~= {target_mse:.5f}\n")
print(f"{'alpha':>6s} {'oob_scale':>10s} {'MSE':>9s} {'off target':>11s}")
for a in ALPHAS_DR:
    mse = F.mse_loss(apply_graded_skew(_probe, a, 0, OOB_SCALE[a]), _probe).item()
    print(f"{a:6.3f} {OOB_SCALE[a]:10.2f} {mse:9.5f} {100*(mse-target_mse)/target_mse:+10.2f}%")

print("\nready for figure1_generate.py:",
      all(n in globals() for n in ["apply_graded_skew", "OOB_SCALE"]))
