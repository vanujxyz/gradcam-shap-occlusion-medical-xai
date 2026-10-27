# Grad-CAM vs. SHAP vs. Occlusion on lightweight medical classifiers (+ CGHA, BAFS, INT8 drift)

Working implementation of the paper *"Grad-CAM vs. SHAP vs. Occlusion Sensitivity: Efficiency–Accuracy Trade-offs for
Explainable AI in Resource-Constrained Medical Image Classification"*. Everything here was actually run and measured on
this laptop (i5-12450H CPU, RTX 3050 Laptop GPU, PyTorch 2.14). The paper's own tables are illustrative numbers, so
**these results are not expected to match them** (see "Honest comparison to the paper").

## What is implemented

| File | Content |
|---|---|
| `common.py` | data loading, backbones (MobileNetV2 / EfficientNet-B0 / ResNet-18), fake-INT8 post-training quantisation |
| `train.py` | fine-tuning (AdamW, lr 1e-4, class-weighted CE, cosine schedule, AMP, 8 epochs) + FP32/INT8 test scoring |
| `explainers.py` | Grad-CAM, Occlusion (16px, stride 8, 729 evals), KernelSHAP (Captum, 50 SLIC superpixels, n=1000), **CGHA-Occ**, **CGHA-Shap** (exact Shapley over top-M CAM regions + background player) |
| `experiments.py` | deletion/insertion AUC, F, BAFS inputs, model-randomisation sanity check, seed/noise stability, FP32→INT8 drift (SSIM, top-15% IoU), latency/memory, ablation on M and ρ |
| `report.py` | builds `results/tables.md` and the figures |
| `smoke_test.py` | quick unit checks (shapes, Shapley efficiency, INT8 wrapper) |
| `results/` | `tables.md` (all tables), `*.json` raw results, `latency_fidelity.png`, `examples_*.png` |

## Setup and reproduce

```bash
conda create -n dlproject python=3.11
conda activate dlproject
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install captum medmnist scikit-image scikit-learn scipy pandas matplotlib seaborn psutil
python train.py 8                  # ~3 min/epoch/model on an RTX 3050
python experiments.py quality      # fidelity, stability, sanity, INT8 drift
python experiments.py latency      # CPU + GPU latency / memory
python experiments.py ablation     # M and rho ablation
python report.py                   # tables + figures -> results/
```

## Deviations from the paper (forced by what is available)

1. **Datasets:** RSNA and HAM10000 need a Kaggle login. I used **MedMNIST+ at 224×224**: PneumoniaMNIST (pediatric chest
   X-ray, binary — *not RSNA*) and DermaMNIST (HAM10000-derived, 7 classes). Official MedMNIST splits, not patient-level
   re-splits.
2. **Hardware:** no Raspberry Pi / Jetson. "CPU" = this laptop CPU with 4 threads, "GPU" = RTX 3050. Absolute latencies
   are therefore ~5–10× lower than a Pi 4; only the ratios between methods are comparable. **Energy was not measured**
   (no power meter). CPU "memory" is whole-process RSS, so it is dominated by PyTorch itself.
3. **Localisation metrics (pointing game, IoU) were not computed:** MedMNIST has no bounding boxes or lesion masks.
4. **INT8** is simulated (per-channel int8 weights, per-tensor uint8 activations with percentile-calibrated ranges,
   512 calibration images), not run on an integer kernel. The paper also uses a fake-quantised copy for gradients.
5. **Scale:** fewer images than the paper's 500: 50 per method for MobileNetV2 (both datasets), 20 for EfficientNet-B0
   and ResNet-18 (pneumonia only; the derma EfficientNet/ResNet models were not trained to save time). Seed stability
   uses 3 seeds × 15 images. Sanity/noise/INT8 metrics use 15 images (6 for the other backbones).
6. **CGHA region formation:** when the selected CAM cells form fewer than M connected components (common for diffuse
   maps) the top-M individual cells are used as players, otherwise the Shapley game collapses to 2 players. CGHA-Occ's
   refined cells blend 50/50 the fine occlusion map with the CAM weights.
7. **INT8 accuracy column:** `results/classification.json` INT8 scores for the four trained models were produced
   *before* the calibration was changed to percentile clipping (EfficientNet-B0 pneumonia shows 0.880; with the final
   calibration it is 0.978 — measured separately, ~0.9-point drop). The INT8 *drift* experiments use the final calibration.

## Results (full tables in `results/tables.md`)

**Classification (test):** pneumonia AUROC 0.986 / 0.987 / 0.990 (MobileNetV2 / EfficientNet-B0 / ResNet-18);
DermaMNIST MobileNetV2 balanced accuracy 0.793.

**Cost per image (pneumonia, MobileNetV2, FP32)**

| Method | CPU (s) | GPU (s) | CPU speed-up of CGHA vs method |
|---|---|---|---|
| Grad-CAM | 0.050 | 0.020 | – |
| CGHA-Occ | 0.79 | 0.10 | – |
| CGHA-Shap (M=6) | 2.33 | 0.18 | – |
| Occlusion | 8.87 | 0.78 | CGHA-Occ 11×, CGHA-Shap 3.8× faster |
| KernelSHAP | 17.7 | 1.69 | CGHA-Occ 22×, CGHA-Shap 7.6× faster |

Grad-CAM is ~180× faster than Occlusion and ~355× faster than KernelSHAP on CPU.

**Fidelity F = ½(Ins + 1 − Del), DermaMNIST MobileNetV2 (the informative dataset):**
Grad-CAM 0.565, Occlusion 0.604, KernelSHAP 0.609, CGHA-Occ 0.588, CGHA-Shap 0.572.

**BAFS (τ=5 s):** on CPU, CGHA-Occ is best on derma (0.588 vs Grad-CAM 0.565, Occlusion 0.453, KernelSHAP 0.323); on
pneumonia the F differences are tiny so Grad-CAM wins by being fastest. On GPU every method fits the budget, so BAFS = F.

**Model-randomisation sanity check:** all methods pass on pneumonia (ρ≈0); on derma **Grad-CAM does not** (ρ=0.64 with a
randomised network), while the perturbation methods pass (ρ≤0.14).

**Stability:** KernelSHAP is seed-dependent (Spearman 0.66 ± 0.12 pneumonia, 0.21 ± 0.19 derma); the other four are
deterministic.

**INT8 drift (SSIM FP32 vs INT8):** Grad-CAM drifts most (0.03 / 0.06), Occlusion and KernelSHAP least
(0.60–0.79); CGHA is in between (0.34–0.54).

**CGHA-Shap ablation on M (pneumonia, CPU):** latency doubles per extra region (0.32 → 4.4 s for M=3 → 7); F is flat
(0.508–0.510), insensitive to ρ.

## Honest comparison to the paper

Reproduced:
- The cost hierarchy: Grad-CAM two orders of magnitude cheaper than Occlusion/KernelSHAP; CGHA sits in between and is
  roughly 4–11× cheaper than Occlusion (CGHA-Shap vs KernelSHAP 7.6×); cost of CGHA-Shap doubles per region.
- KernelSHAP is the slowest and the only seed-unstable method.
- Grad-CAM drifts most under INT8, and Occlusion is more robust; CGHA-Shap's Shapley efficiency holds exactly
  (map sums to f(x) − f(baseline)).
- BAFS changes the ranking with the time budget: slow-but-faithful methods lose on CPU, all fit on GPU.

**Not** reproduced (the paper's numbers were illustrative):
- **CGHA-Shap did not match Occlusion's fidelity** on our data: F 0.572 vs 0.604 (derma), and 0.476 vs 0.596 on
  EfficientNet-B0 pneumonia. CGHA-Occ is closer but still below.
- **KernelSHAP had the highest F**, not Occlusion. Likely cause: KernelSHAP uses a blurred baseline, the same as the
  deletion/insertion curves, so it optimises (nearly) the metric it is scored on. Treat the F ranking with that bias in mind.
- Occlusion is barely better than Grad-CAM in F.
- On **pneumonia the deletion/insertion metric is nearly degenerate**: the fine-tuned models score blank or blurred
  images as pneumonia at 0.84–0.95 probability, and the ResNet-18 model is saturated (Del ≈ Ins ≈ 1.0, F = 0.500 for every
  method). F differences on pneumonia are therefore not meaningful; use DermaMNIST for fidelity conclusions.
- INT8 drift SSIM values are much lower than the paper's (Grad-CAM 0.03 vs 0.71, Occlusion 0.60–0.78 vs 0.93), partly
  because raw SSIM on min-max-normalised, low-contrast maps is very sensitive to small shifts.

## Limitations

Small image counts (confidence intervals are wide; no significance tests), MedMNIST instead of RSNA/HAM10000, no edge
hardware, no energy, no localisation metrics, simulated INT8, deletion/insertion with a blurred baseline is
out-of-distribution (no ROAR), and results come from a single training seed per model.
