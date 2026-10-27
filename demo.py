"""Live demo: explain one test image with all five methods using the trained models.

usage: python demo.py [pneumonia|derma] [backbone] [device: cuda|cpu]
Saves results/demo_<dataset>.png and prints per-method time and fidelity.
"""
import sys
import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common
from common import *
import experiments as X
import explainers as E

ds = sys.argv[1] if len(sys.argv) > 1 else "derma"
bb = sys.argv[2] if len(sys.argv) > 2 else "mobilenet_v2"
dev = torch.device(sys.argv[3] if len(sys.argv) > 3 else DEV.type)
X.DEV = common.DEV = dev
model = load_model(ds, bb, dev)
layer = target_layer(model, bb)
_, base = X.dataset_stuff(ds)
base = base.to(dev)
x, c = X.pick_images(ds, model, 1, seed=1)[0]
x = x.to(dev)
print(f"{ds}/{bb} on {dev}: predicted class {c}, p={E.score(model, x, c)[0].item():.3f}")

for m in X.METHODS:  # warm-up so the timings below exclude CUDA/cuDNN start-up
    X.explain(m, model, layer, x, c, base)
fig, ax = plt.subplots(1, 6, figsize=(16, 3.2))
ax[0].imshow(x[0].permute(1, 2, 0).cpu()); ax[0].set_title("input")
print(f"{'method':11s} {'time (s)':>9s} {'Del-AUC':>8s} {'Ins-AUC':>8s} {'F':>6s}")
for j, m in enumerate(X.METHODS, 1):
    X.sync(); t = time.perf_counter()
    a = X.explain(m, model, layer, x, c, base)
    X.sync(); dt = time.perf_counter() - t
    d, i = X.del_ins(model, x, c, a)
    print(f"{m:11s} {dt:9.3f} {d:8.3f} {i:8.3f} {0.5 * (i + 1 - d):6.3f}")
    ax[j].imshow(x[0].permute(1, 2, 0).cpu()); ax[j].imshow(X.norm01(a), cmap="jet", alpha=0.5)
    ax[j].set_title(f"{m}\n{dt:.2f}s")
for a in ax:
    a.axis("off")
plt.tight_layout()
out = os.path.join(RES_DIR, f"demo_{ds}.png")
plt.savefig(out, dpi=130)
print("saved", out)
