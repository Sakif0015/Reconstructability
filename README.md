# Reconstructability

**A model-free measure of augmentation–shift compatibility for federated medical imaging benchmarks.**

Augmentation-based domain generalization in federated medical imaging is usually
evaluated on *synthetically* shifted held-out sites — and those shifts are
typically built from the same primitives (contrast, gamma, blur) the
augmentations themselves use. This repository asks how much of the reported
benefit survives when the test shift is moved *outside* the augmentation family,
and provides a measure that answers the question before any model is trained.

---

## The measure

Given a clean image `x`, its shifted counterpart `x'`, and an augmentation bank
`B` with continuous parameters θ:

```
R  =  1  −  min_θ MSE( B_θ(x'), x )  /  MSE( x', x )
```

`R = 1` means the bank can invert the shift exactly — the held-out site is inside
the augmentation family. `R = 0` means its best correction removes nothing.

Three properties:

- **Model-free.** No trained feature extractor. Computable before an experiment runs.
- **Bank-relative.** `R` is a property of the *pair* (bank, shift), not of the shift
  alone. That is the point: it asks whether *this* benchmark tests *this* method.
- **An upper bound.** Obtained by optimizing continuous parameters directly, so any
  trained policy over a discrete grid achieves less.

---

## Main results

**Synthetic (CT-Kidney, 3 clients, FedAvg, ResNet-18, 3 seeds).** Seven held-out
sites of *identical* shift magnitude (MSE 0.03538, matched by binary search) but
graded reconstructability:

| R (%) | 99.9 | 93.8 | 83.1 | 75.3 | 71.7 | 61.8 | 52.0 |
|---|---|---|---|---|---|---|---|
| τ, random policy | +51.0 | +56.5 | +45.2 | +43.0 | +31.7 | +24.6 | +19.0 |

Relative robustness falls 63% across the range. Page's trend test:
*z* = +4.04, *p* = 3×10⁻⁵. Per-seed Pearson *r* = +0.946 ± 0.014.

**46 ± 14%** of the random policy's apparent gain does not survive an
out-of-family shift of matched magnitude. The control that makes this
interpretable: FedAvg, which uses no bank augmentation, shows an in-family /
out-of-family gap of +1.81 ± 5.59 (n.s.) — the two sites are equally hard on
their own.

**Natural (Camelyon17-WILDS, 5 hospitals, leave-one-out, 3 seeds).** The same
method and bank yield τ = **+6.09 ± 6.81** (*t*(4) = 2.00, *p* = 0.116) — an 8.5×
drop. Neither `R` nor an adapted perceptual similarity measure predicts it.
Hospitals 1 and 4 share *R* = 96.3% yet differ in benefit by **12.86 points**
(7.2 pooled SD, *p* = 0.0009).

What *does* predict τ on natural data is baseline headroom (*r* = +0.914) — the
accuracy confound documented by Taori et al. (2020), reported here as such and
claimed as nothing further.

---

## Reproducing the figures (no GPU, ~2 seconds)

All results are saved as JSON, so the figures regenerate on CPU without retraining:

```bash
pip install -r requirements.txt
cd scripts
python figures_2_and_3_regenerate.py
```

This reads `results/*.json` and writes `fig2_dose_response.pdf` and
`fig3_camelyon_tau.pdf`. Figure 1 needs the notebook (it renders real
reconstructions) — see `notebooks/`.

## Reproducing the experiments (GPU)

The full pipeline lives in `notebooks/`. Approximate costs on a single T4:

| Experiment | Cost |
|---|---|
| Leakage, 3 arms × 3 seeds | ~1.5 h |
| Dose–response, 7 sites × 3 seeds | ~3 h |
| Camelyon17, 5 folds × 3 arms × 3 seeds | ~5 h |
| Perceptual comparator (MSD), 5 folds | ~3.5 h |
| R ladder + Figure 1 | ~8 min |

---

## Layout

```
paper/        LaTeX source and figures
notebooks/    full experimental pipeline
src/          the R estimator, op bank, graded-site construction
scripts/      figure generation from saved results
results/      all experimental output as JSON
```

---

## Scope and limitations

Stated plainly, because they bound what the results mean:

- The similarity–benefit relationship is **not new** — Mintun et al. (NeurIPS 2021)
  established it perceptually. This work contributes a constructive, model-free
  measure, a graded magnitude-matched axis, and the federated medical setting.
- Synthetic→natural non-transfer is **not new** either — Taori et al. (NeurIPS 2020).
- One synthetic dataset (centrally saturated; used as a controlled testbed, not a
  difficulty benchmark), one architecture, one augmentation bank.
- Camelyon17 has five hospitals and τ is not significant. **95%** of τ variance is
  between hospitals, so additional seeds cannot raise the effective sample size.
  The benefit is reported as *not detected*, not as absent.
- The learned-policy arm is nondeterministic: repeated runs at a fixed seed differ
  by 2.20 points. FedAvg and the random policy reproduce to ±0.07, and all headline
  claims rest on those two arms.
- Causal scope: in the synthetic experiment `R` was manipulated with images, task,
  and shift magnitude held fixed. Across the synthetic/natural divide, modality,
  task, baseline, and the `R` instrument all change — so the claim is only that the
  relationship the synthetic benchmark reports does not appear in natural data,
  not that the protocol causes it.

---

<!--
## Citation

```bibtex
@misc{alam2026reconstructability,
  title  = {Reconstructability: A Model-Free Measure of Augmentation--Shift
            Compatibility for Federated Medical Imaging Benchmarks},
  author = {Alam, S. M. Seefat},
  year   = {2026},
  note   = {Preprint}
}

## License

MIT (code) — see `LICENSE`. CT-Kidney and Camelyon17-WILDS retain their own licenses.
