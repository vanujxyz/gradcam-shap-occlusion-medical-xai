"""Sanity tests for the explainers using an untrained network and a random image (no data needed)."""
import numpy as np
import torch
from common import *
import explainers as E
from experiments import explain, del_ins, METHODS

torch.manual_seed(0)
m = build("mobilenet_v2", 2, pretrained=False).to(DEV).eval()
layer = target_layer(m, "mobilenet_v2")
x = torch.rand(1, 3, 224, 224, device=DEV)
base = torch.full_like(x, 0.5)
c = int(m(x).argmax())
for k in METHODS:
    a = explain(k, m, layer, x, c, base)
    d, i = del_ins(m, x, c, a)
    print(f"{k:11s} map {tuple(a.shape)} sum={a.sum().item():+.4f} del={d:.3f} ins={i:.3f}")
# Shapley efficiency: sum of CGHA-Shap map == f(x) - f(baseline)
cam = E.gradcam(m, layer, x, c)
a, phi = E.cgha_shap(m, x, c, cam, base)
fx, fb = E.score(m, x, c)[0].item(), E.score(m, base, c)[0].item()
print("efficiency:", a.sum().item(), "vs", fx - fb)
assert abs(a.sum().item() - (fx - fb)) < 1e-4
q = fake_quantize(m, x.repeat(8, 1, 1, 1))
print("int8 logits diff:", (q(x) - m(x)).abs().max().item())
print("OK")
