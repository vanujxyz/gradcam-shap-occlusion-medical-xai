# Results (measured on this machine)

## Classification (test set; FP32 / fake-INT8)

| Dataset / backbone | metric | FP32 | INT8 |
|---|---|---|---|
| pneumonia/mobilenet_v2 | AUROC | 0.986 | 0.986 |
| pneumonia/efficientnet_b0 | AUROC | 0.987 | 0.880 |
| pneumonia/resnet18 | AUROC | 0.990 | 0.990 |
| derma/mobilenet_v2 | Bal. Acc. | 0.793 | 0.793 |

## Cost per image (pneumonia, FP32): CPU = 4 threads i5-12450H, GPU = RTX 3050 Laptop

| Backbone | Method | CPU latency (s) | CPU RSS (MB) | GPU latency (s) | GPU peak mem (MB) |
|---|---|---|---|---|---|
| mobilenet_v2 | Grad-CAM | 0.050 | 2171 | 0.020 | 116 |
| mobilenet_v2 | Occlusion | 8.871 | 3029 | 0.779 | 1550 |
| mobilenet_v2 | KernelSHAP | 17.743 | 2375 | 1.688 | 1209 |
| mobilenet_v2 | CGHA-Occ | 0.794 | 2486 | 0.104 | 332 |
| mobilenet_v2 | CGHA-Shap | 2.334 | 2372 | 0.183 | 860 |
| efficientnet_b0 | Grad-CAM | 0.072 | 2526 | 0.063 | 156 |
| efficientnet_b0 | Occlusion | 12.756 | 3330 | 0.952 | 1581 |
| efficientnet_b0 | KernelSHAP | 21.696 | 2684 | 1.981 | 1240 |
| efficientnet_b0 | CGHA-Occ | 1.134 | 2490 | 0.150 | 363 |
| efficientnet_b0 | CGHA-Shap | 3.021 | 2697 | 0.238 | 892 |
| resnet18 | Grad-CAM | 0.114 | 2641 | 0.015 | 196 |
| resnet18 | Occlusion | 15.070 | 3497 | 0.819 | 1373 |
| resnet18 | KernelSHAP | 25.724 | 2840 | 1.562 | 894 |
| resnet18 | CGHA-Occ | 1.679 | 2637 | 0.097 | 306 |
| resnet18 | CGHA-Shap | 3.481 | 2845 | 0.164 | 685 |

## Faithfulness (mean over images; Del-AUC lower is better, Ins-AUC / F higher)

| Dataset / backbone | Method | n | Del-AUC | Ins-AUC | F |
|---|---|---|---|---|---|
| pneumonia/mobilenet_v2 | Grad-CAM | 50 | 0.938 | 0.962 | 0.512 |
| pneumonia/mobilenet_v2 | Occlusion | 50 | 0.942 | 0.972 | 0.515 |
| pneumonia/mobilenet_v2 | KernelSHAP | 50 | 0.856 | 0.983 | 0.563 |
| pneumonia/mobilenet_v2 | CGHA-Occ | 50 | 0.945 | 0.967 | 0.511 |
| pneumonia/mobilenet_v2 | CGHA-Shap | 50 | 0.943 | 0.959 | 0.508 |
| derma/mobilenet_v2 | Grad-CAM | 50 | 0.332 | 0.463 | 0.565 |
| derma/mobilenet_v2 | Occlusion | 50 | 0.356 | 0.564 | 0.604 |
| derma/mobilenet_v2 | KernelSHAP | 50 | 0.383 | 0.601 | 0.609 |
| derma/mobilenet_v2 | CGHA-Occ | 50 | 0.359 | 0.535 | 0.588 |
| derma/mobilenet_v2 | CGHA-Shap | 50 | 0.384 | 0.528 | 0.572 |
| pneumonia/efficientnet_b0 | Grad-CAM | 20 | 0.656 | 0.853 | 0.598 |
| pneumonia/efficientnet_b0 | Occlusion | 20 | 0.674 | 0.866 | 0.596 |
| pneumonia/efficientnet_b0 | KernelSHAP | 20 | 0.570 | 0.886 | 0.658 |
| pneumonia/efficientnet_b0 | CGHA-Occ | 20 | 0.703 | 0.849 | 0.573 |
| pneumonia/efficientnet_b0 | CGHA-Shap | 20 | 0.779 | 0.732 | 0.476 |
| pneumonia/resnet18 | Grad-CAM | 20 | 0.999 | 1.000 | 0.500 |
| pneumonia/resnet18 | Occlusion | 20 | 0.999 | 1.000 | 0.500 |
| pneumonia/resnet18 | KernelSHAP | 20 | 0.999 | 1.000 | 0.501 |
| pneumonia/resnet18 | CGHA-Occ | 20 | 1.000 | 1.000 | 0.500 |
| pneumonia/resnet18 | CGHA-Shap | 20 | 1.000 | 1.000 | 0.500 |

## Sanity check, stability and INT8 drift (MobileNetV2)

| Dataset | Method | Randomised rho (lower better) | Seed rho | Noise SSIM | INT8 drift SSIM | INT8 top-15% IoU |
|---|---|---|---|---|---|---|
| pneumonia | Grad-CAM | 0.02 | 1.00 (deterministic) | 0.73 | 0.03 | 0.15 |
| pneumonia | Occlusion | 0.00 | 1.00 (deterministic) | 0.42 | 0.60 | 0.37 |
| pneumonia | KernelSHAP | -0.07 | 0.66 ± 0.12 | 0.58 | 0.63 | 0.18 |
| pneumonia | CGHA-Occ | 0.00 | 1.00 (deterministic) | 0.48 | 0.49 | 0.12 |
| pneumonia | CGHA-Shap | 0.00 | 1.00 (deterministic) | 0.55 | 0.54 | 0.16 |
| derma | Grad-CAM | 0.64 | 1.00 (deterministic) | 0.77 | 0.06 | 0.15 |
| derma | Occlusion | 0.00 | 1.00 (deterministic) | 0.57 | 0.78 | 0.57 |
| derma | KernelSHAP | 0.14 | 0.21 ± 0.19 | 0.81 | 0.79 | 0.52 |
| derma | CGHA-Occ | 0.00 | 1.00 (deterministic) | 0.54 | 0.46 | 0.13 |
| derma | CGHA-Shap | 0.03 | 1.00 (deterministic) | 0.72 | 0.34 | 0.17 |

## BAFS (tau=5.0s, beta=0.5); latency from this machine's CPU and GPU

| Dataset | Method | CPU BAFS | GPU BAFS |
|---|---|---|---|
| pneumonia | Grad-CAM | 0.512 | 0.512 |
| pneumonia | Occlusion | 0.387 | 0.515 |
| pneumonia | KernelSHAP | 0.299 | 0.563 |
| pneumonia | CGHA-Occ | 0.511 | 0.511 |
| pneumonia | CGHA-Shap | 0.508 | 0.508 |
| derma | Grad-CAM | 0.565 | 0.565 |
| derma | Occlusion | 0.453 | 0.604 |
| derma | KernelSHAP | 0.323 | 0.609 |
| derma | CGHA-Occ | 0.588 | 0.588 |
| derma | CGHA-Shap | 0.572 | 0.572 |

## Best method on CPU by time budget (pneumonia, MobileNetV2)

| Budget tau | Best | BAFS |
|---|---|---|
| 0.5 s | Grad-CAM | 0.512 |
| 1 s | Grad-CAM | 0.512 |
| 5 s | Grad-CAM | 0.512 |
| 30 s | KernelSHAP | 0.563 |
| 60 s | KernelSHAP | 0.563 |

## Ranking agreement across backbones (F, pneumonia)

| Backbone | Grad-CAM | Occlusion | KernelSHAP | CGHA-Occ | CGHA-Shap |
|---|---|---|---|---|---|
| mobilenet_v2 | 0.512 | 0.515 | 0.563 | 0.511 | 0.508 |
| efficientnet_b0 | 0.598 | 0.596 | 0.658 | 0.573 | 0.476 |
| resnet18 | 0.500 | 0.500 | 0.501 | 0.500 | 0.500 |

## Ablation on M (CGHA-Shap, pneumonia, MobileNetV2, CPU)

| M | FPE | CPU latency (s) | F | BAFS |
|---|---|---|---|---|
| 3 | 19 | 0.32 | 0.510 | 0.510 |
| 4 | 35 | 0.71 | 0.509 | 0.509 |
| 5 | 67 | 1.20 | 0.509 | 0.509 |
| 6 | 131 | 2.31 | 0.508 | 0.508 |
| 7 | 259 | 4.38 | 0.508 | 0.508 |

Sensitivity to coverage rho (F): rho=0.4: 0.510, rho=0.6: 0.510, rho=0.8: 0.510