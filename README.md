# Explainable AI for Lightweight Medical Image Classifiers

A benchmark of three post-hoc explanation methods — **Grad-CAM**, **KernelSHAP** and **Occlusion Sensitivity** — on small
CNN classifiers for medical images, plus two methods and one metric introduced in this project:

- **CGHA (CAM-Gated Hierarchical Attribution)** — uses one cheap Grad-CAM pass to decide *where* to spend a perturbation
  budget, then computes either hierarchical occlusion (**CGHA-Occ**) or *exact* Shapley values over the top-M regions
  (**CGHA-Shap**) instead of sampled approximations.
- **BAFS (Budget-Adjusted Fidelity Score)** — one number that combines explanation fidelity with a soft penalty for
  exceeding a time budget.
- **INT8 explanation drift** — how much each method's heatmap changes when the model is quantised to INT8.

Everything in this repository was run and measured on a laptop (Intel i5-12450H, RTX 3050 Laptop GPU, PyTorch 2.14).

## Repository layout

| File | Purpose |
|---|---|
| `common.py` | data loading, backbones, input normalisation wrapper, simulated INT8 quantisation |
| `train.py` | fine-tunes the classifiers and scores them in FP32 and INT8 |
| `explainers.py` | Grad-CAM, Occlusion, KernelSHAP, CGHA-Occ, CGHA-Shap |
| `experiments.py` | fidelity, sanity, stability, INT8 drift, latency/memory, ablations |
| `report.py` | builds `results/tables.md` and the figures |
| `demo.py` | explains one image with all five methods and prints time and fidelity |
| `smoke_test.py` | quick checks (shapes, Shapley efficiency, quantisation wrapper) |
| `checkpoints/` | trained weights (4 models) |
| `results/` | `tables.md`, raw `*.json`, figures |

## Data and models

- **Datasets** (MedMNIST+, 224×224, downloaded automatically on first use):
  PneumoniaMNIST (chest X-ray, binary) and DermaMNIST (dermoscopy, 7 classes, derived from HAM10000).
- **Backbones** (ImageNet-pretrained, fine-tuned 8 epochs with AdamW, lr 1e-4, class-weighted cross-entropy, cosine
  schedule, mixed precision): MobileNetV2, EfficientNet-B0, ResNet-18.
- **Trained and included:** MobileNetV2 on both datasets; EfficientNet-B0 and ResNet-18 on pneumonia.

| Model | Metric (test) | FP32 |
|---|---|---|
| Pneumonia / MobileNetV2 | AUROC | 0.986 |
| Pneumonia / EfficientNet-B0 | AUROC | 0.987 |
| Pneumonia / ResNet-18 | AUROC | 0.990 |
| Derma / MobileNetV2 | Balanced accuracy | 0.793 |

## Methods

| Method | Evaluations per image | Notes |
|---|---|---|
| Grad-CAM | ≈3 forward-pass equivalents | last conv block, 7×7 upsampled to 224×224 |
| Occlusion | 729 | 16 px patch, stride 8, dataset-mean baseline |
| KernelSHAP | 1000 + SLIC | Captum, 50 SLIC superpixels, blurred baseline, stochastic |
| CGHA-Occ | ≈69 | occlude selected CAM cells, refine the top 2 with fine patches |
| CGHA-Shap | 3 + 2^(M+1) ≈ 131 for M=6 | exact Shapley over M regions + 1 background player |

**CGHA in short.** (1) Run Grad-CAM once. (2) Split the image into a 7×7 grid and keep the CAM-heaviest cells until
60% of the CAM mass is covered (max 16 cells). (3) Merge them into at most M regions; everything else is one background
player. If the cells form fewer than M connected regions, the top-M individual cells are used so the game stays
non-trivial. (4) CGHA-Shap evaluates all 2^(M+1) coalitions and computes exact Shapley values, so the attribution map sums
exactly to f(x) − f(baseline) and has no sampling noise. (5) Each region's value is spread over its pixels in proportion to
the CAM, keeping Grad-CAM's fine spatial detail.

**BAFS.** F = ½(Insertion-AUC + (1 − Deletion-AUC)); BAFS = F · min(1, (τ/T)^β), with latency T, budget τ = 5 s, β = 0.5.

**Metrics.** Deletion/insertion AUC in 1% steps with a blurred baseline; model-randomisation sanity check (Spearman
correlation with maps from a randomly initialised network; lower is better); seed stability (KernelSHAP, Spearman across
seeds); noise stability (SSIM under σ = 0.01 input noise); INT8 drift (SSIM and top-15% IoU between FP32 and INT8 maps);
latency and memory.

**INT8** is simulated: per-channel int8 weights and per-tensor uint8 activations with percentile-calibrated ranges
(512 calibration images), applied as quantise–dequantise so gradients still flow for Grad-CAM.

## Setup

```bash
conda create -n dlproject python=3.11
conda activate dlproject
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install captum medmnist scikit-image scikit-learn scipy pandas matplotlib seaborn psutil
```

## Usage

```bash
python demo.py derma mobilenet_v2          # live demo on one image (GPU if available)
python demo.py pneumonia mobilenet_v2 cpu  # force CPU
python smoke_test.py                       # quick checks
```

Full pipeline (the trained checkpoints are included, so training can be skipped):

```bash
python train.py 8                # ~3 min per epoch per model on an RTX 3050
python experiments.py quality    # fidelity, stability, sanity, INT8 drift  (~25 min)
python experiments.py latency    # CPU and GPU latency / memory
python experiments.py ablation   # number of regions M, coverage rho
python report.py                 # -> results/tables.md and figures
```

## Results

Full tables are in [`results/tables.md`](results/tables.md). "CPU" is the laptop CPU with 4 threads; "GPU" is the RTX 3050.

### Cost per image (pneumonia, MobileNetV2, FP32)

| Method | CPU (s) | GPU (s) |
|---|---|---|
| Grad-CAM | 0.050 | 0.020 |
| CGHA-Occ | 0.79 | 0.10 |
| CGHA-Shap (M=6) | 2.33 | 0.18 |
| Occlusion | 8.87 | 0.78 |
| KernelSHAP | 17.7 | 1.69 |

Grad-CAM is about 180× faster than Occlusion and 355× faster than KernelSHAP on CPU. CGHA-Occ is about 11× faster than
Occlusion; CGHA-Shap is about 3.8× faster than Occlusion and 7.6× faster than KernelSHAP.

### Fidelity F (DermaMNIST, MobileNetV2, 50 images)

| Method | Del-AUC ↓ | Ins-AUC ↑ | F ↑ |
|---|---|---|---|
| Grad-CAM | 0.332 | 0.463 | 0.565 |
| Occlusion | 0.356 | 0.564 | 0.604 |
| KernelSHAP | 0.383 | 0.601 | 0.609 |
| CGHA-Occ | 0.359 | 0.535 | 0.588 |
| CGHA-Shap | 0.384 | 0.528 | 0.572 |

Occlusion and KernelSHAP are the most faithful; both CGHA variants sit between them and Grad-CAM at a fraction of the
cost. KernelSHAP's lead should be read with care: it uses a blurred baseline, the same one the deletion/insertion curves
use, so it partly optimises the metric it is scored on.

### BAFS (τ = 5 s)

On CPU, CGHA-Occ has the best BAFS on DermaMNIST (0.588, versus Grad-CAM 0.565, Occlusion 0.453, KernelSHAP 0.323). On
the GPU every method fits inside the budget, so BAFS equals F. On pneumonia the fidelity differences are tiny, so the
fastest method (Grad-CAM) wins on CPU.

### Sanity, stability and INT8 drift (MobileNetV2)

| | Grad-CAM | Occlusion | KernelSHAP | CGHA-Occ | CGHA-Shap |
|---|---|---|---|---|---|
| Randomisation ρ, derma (↓) | **0.64** | 0.00 | 0.14 | 0.00 | 0.03 |
| Randomisation ρ, pneumonia (↓) | 0.02 | 0.00 | −0.07 | 0.00 | 0.00 |
| Seed ρ, derma / pneumonia (↑) | deterministic | deterministic | 0.21 / 0.66 | deterministic | deterministic |
| INT8 drift SSIM, derma / pneumonia (↑) | 0.06 / 0.03 | 0.78 / 0.60 | 0.79 / 0.63 | 0.46 / 0.49 | 0.34 / 0.54 |

- Grad-CAM **fails** the randomisation check on DermaMNIST (its map barely changes when the network is randomised).
- KernelSHAP is the only method whose output depends on the random seed.
- Grad-CAM drifts the most under INT8; the perturbation methods are the most robust; CGHA is in between.

### CGHA-Shap ablation on the number of regions M (pneumonia, CPU)

Latency roughly doubles for each extra region (0.32 s at M=3, 4.4 s at M=7), while F stays flat at 0.508–0.510 and is
insensitive to the coverage threshold ρ.

## Findings

1. The cost gap between Grad-CAM and perturbation methods is two orders of magnitude; CGHA recovers part of the
   fidelity at a few percent to tens of percent of the cost, with an explicit cost/detail dial (M).
2. CGHA-Shap's attributions satisfy Shapley efficiency exactly and are deterministic, removing KernelSHAP's seed
   variance.
3. CGHA did not match Occlusion's fidelity in these experiments (F 0.572–0.588 vs 0.604 on DermaMNIST); its advantage
   is speed and determinism, not top fidelity.
4. Grad-CAM is fast but the least trustworthy: it fails the randomisation check on DermaMNIST and changes most under INT8.

## Limitations

- **Pneumonia fidelity is weak evidence.** The fine-tuned pneumonia models label blank or blurred images as pneumonia
  with 0.84–0.95 probability, and the ResNet-18 model is saturated (Del ≈ Ins ≈ 1.0, F = 0.500 for every method). Use
  DermaMNIST for fidelity conclusions.
- **Small samples.** 50 images per method for MobileNetV2, 20 for EfficientNet-B0 and ResNet-18; stability, sanity and
  drift use 15 images (6 for the other backbones). No confidence intervals or significance tests; one training seed.
- **Not measured:** localisation against expert annotations (MedMNIST has no masks), energy, and behaviour on real
  edge hardware (a Raspberry Pi or Jetson would be roughly 5–10× slower than this laptop; only the ratios between methods
  carry over). CPU "memory" is whole-process RSS and is dominated by PyTorch.
- **Datasets** are small-resolution MedMNIST variants (PneumoniaMNIST is a pediatric chest X-ray set), not full-size
  clinical datasets.
- **INT8 is simulated**, not run on integer kernels. The INT8 accuracy figures in `results/classification.json` were
  computed before calibration was switched to percentile clipping (EfficientNet-B0 pneumonia shows 0.880; re-measured with
  the final calibration it is 0.978). The drift experiments use the final calibration.
- Deletion/insertion with a blurred baseline produces out-of-distribution inputs; no retraining-based check (ROAR) was done.
- No clinician evaluation, so nothing here shows that more faithful maps improve clinical decisions.
