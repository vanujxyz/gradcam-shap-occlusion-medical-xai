"""Aggregate results into tables (results/tables.md) and figures (results/*.png)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

METHODS = ["Grad-CAM", "Occlusion", "KernelSHAP", "CGHA-Occ", "CGHA-Shap"]
TAU, BETA = 5.0, 0.5
J = lambda n: json.load(open(os.path.join(RES_DIR, n)))
q, lat, cls = J("quality.json"), J("latency.json"), J("classification.json")
abl = J("ablation.json") if os.path.exists(os.path.join(RES_DIR, "ablation.json")) else None
mean = lambda v: float(np.mean(v)) if len(v) else float("nan")
std = lambda v: float(np.std(v)) if len(v) else float("nan")


def fid(ds, bb, m):
    r = q[f"{ds}/{bb}"][m]
    return 0.5 * (mean(r["ins"]) + 1 - mean(r["del"]))


def bafs(F, T, tau=TAU, beta=BETA):
    return F * min(1.0, (tau / T) ** beta)


out = []
w = out.append
w("# Results (measured on this machine)\n")
w("## Classification (test set; FP32 / fake-INT8)\n")
w("| Dataset / backbone | metric | FP32 | INT8 |\n|---|---|---|---|")
for k, v in cls.items():
    w(f"| {k} | {'AUROC' if k.startswith('pneumonia') else 'Bal. Acc.'} | {v['fp32']:.3f} | {v['int8']:.3f} |")

w("\n## Cost per image (pneumonia, FP32): CPU = 4 threads i5-12450H, GPU = RTX 3050 Laptop\n")
w("| Backbone | Method | CPU latency (s) | CPU RSS (MB) | GPU latency (s) | GPU peak mem (MB) |\n|---|---|---|---|---|---|")
for bb in BACKBONES:
    for m in METHODS:
        c, g = lat[f"cpu/{bb}"][m], lat[f"cuda/{bb}"][m]
        w(f"| {bb} | {m} | {c['median_s']:.3f} | {c['mem_mb']:.0f} | {g['median_s']:.3f} | {g['mem_mb']:.0f} |")

w("\n## Faithfulness (mean over images; Del-AUC lower is better, Ins-AUC / F higher)\n")
w("| Dataset / backbone | Method | n | Del-AUC | Ins-AUC | F |\n|---|---|---|---|---|---|")
for key in q:
    ds, bb = key.split("/")
    for m in METHODS:
        r = q[key][m]
        w(f"| {key} | {m} | {len(r['del'])} | {mean(r['del']):.3f} | {mean(r['ins']):.3f} | {fid(ds, bb, m):.3f} |")

w("\n## Sanity check, stability and INT8 drift (MobileNetV2)\n")
w("| Dataset | Method | Randomised rho (lower better) | Seed rho | Noise SSIM | INT8 drift SSIM | INT8 top-15% IoU |\n|---|---|---|---|---|---|---|")
for ds in DATASETS:
    for m in METHODS:
        r = q[f"{ds}/mobilenet_v2"][m]
        seed = f"{mean(r['seed_rho']):.2f} ± {std(r['seed_rho']):.2f}" if r["seed_rho"] else "1.00 (deterministic)"
        w(f"| {ds} | {m} | {mean(r['rand_rho']):.2f} | {seed} | {mean(r['noise_ssim']):.2f} | "
          f"{mean(r['drift_ssim']):.2f} | {mean(r['drift_iou']):.2f} |")

w(f"\n## BAFS (tau={TAU}s, beta={BETA}); latency from this machine's CPU and GPU\n")
w("| Dataset | Method | CPU BAFS | GPU BAFS |\n|---|---|---|---|")
bafs_tab = {}
for ds in DATASETS:
    for m in METHODS:
        F = fid(ds, "mobilenet_v2", m)
        bc = bafs(F, lat["cpu/mobilenet_v2"][m]["median_s"])
        bg = bafs(F, lat["cuda/mobilenet_v2"][m]["median_s"])
        bafs_tab[(ds, m)] = (bc, bg)
        w(f"| {ds} | {m} | {bc:.3f} | {bg:.3f} |")

w("\n## Best method on CPU by time budget (pneumonia, MobileNetV2)\n")
w("| Budget tau | Best | BAFS |\n|---|---|---|")
for tau in (0.5, 1, 5, 30, 60):
    sc = {m: bafs(fid("pneumonia", "mobilenet_v2", m), lat["cpu/mobilenet_v2"][m]["median_s"], tau) for m in METHODS}
    b = max(sc, key=sc.get)
    w(f"| {tau} s | {b} | {sc[b]:.3f} |")

w("\n## Ranking agreement across backbones (F, pneumonia)\n")
w("| Backbone | " + " | ".join(METHODS) + " |\n|---|" + "---|" * len(METHODS))
for bb in BACKBONES:
    w(f"| {bb} | " + " | ".join(f"{fid('pneumonia', bb, m):.3f}" for m in METHODS) + " |")

if abl:
    w("\n## Ablation on M (CGHA-Shap, pneumonia, MobileNetV2, CPU)\n")
    w("| M | FPE | CPU latency (s) | F | BAFS |\n|---|---|---|---|---|")
    for M, r in abl["M"].items():
        w(f"| {M} | {r['fpe']} | {r['cpu_latency_s']:.2f} | {r['F']:.3f} | {bafs(r['F'], r['cpu_latency_s']):.3f} |")
    w("\nSensitivity to coverage rho (F): " + ", ".join(f"rho={k}: {v:.3f}" for k, v in abl["rho"].items()))

open(os.path.join(RES_DIR, "tables.md"), "w", encoding="utf-8").write("\n".join(out))

# ------------------------------------------------------------ figures
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, dev in zip(ax, ("cpu", "cuda")):
    for m in METHODS:
        a.scatter(lat[f"{dev}/mobilenet_v2"][m]["median_s"], fid("pneumonia", "mobilenet_v2", m), s=70,
                  marker="D" if m.startswith("CGHA") else "o", label=m)
    a.set_xscale("log"); a.axvline(TAU, ls="--", c="gray"); a.set_xlabel("latency per image (s, log)")
    a.set_ylabel("fidelity F (pneumonia)"); a.set_title({"cpu": "CPU (4 threads)", "cuda": "GPU RTX 3050"}[dev])
ax[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(RES_DIR, "latency_fidelity.png"), dpi=150); plt.close()

for ds in DATASETS:
    p = os.path.join(RES_DIR, f"examples_{ds}_mobilenet_v2.npy")
    if not os.path.exists(p):
        continue
    ex = np.load(p, allow_pickle=True)
    fig, axs = plt.subplots(len(ex), 6, figsize=(14, 2.4 * len(ex)))
    for r, e in enumerate(ex):
        axs[r, 0].imshow(e["img"].transpose(1, 2, 0)); axs[r, 0].set_title("input" if r == 0 else "")
        for j, m in enumerate(METHODS):
            axs[r, j + 1].imshow(e["img"].transpose(1, 2, 0)); axs[r, j + 1].imshow(e[m], cmap="jet", alpha=0.5)
            if r == 0:
                axs[r, j + 1].set_title(m)
    for a in axs.ravel():
        a.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(RES_DIR, f"examples_{ds}.png"), dpi=130); plt.close()
