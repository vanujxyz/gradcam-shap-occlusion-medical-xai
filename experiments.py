"""Run the explanation benchmark: fidelity, stability, sanity, INT8 drift, latency, ablation.

usage: python experiments.py quality|latency|ablation
"""
import json
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity as ssim
from common import *
import explainers as E

METHODS = ["Grad-CAM", "Occlusion", "KernelSHAP", "CGHA-Occ", "CGHA-Shap"]
H = 224


def sync():
    if DEV.type == "cuda":
        torch.cuda.synchronize()


def explain(method, model, layer, x, c, base, M=6, seed=0):
    """Return a (H,H) attribution map (torch, on x's device)."""
    if method == "Grad-CAM":
        return E.gradcam(model, layer, x, c)
    if method == "Occlusion":
        return E.occlusion(model, x, c, base)
    if method == "KernelSHAP":
        return E.kernelshap(model, x, c, seed=seed)
    cam = E.gradcam(model, layer, x, c)
    if method == "CGHA-Occ":
        return E.cgha_occ(model, x, c, cam, base)[0]
    if method == "CGHA-Shap":
        return E.cgha_shap(model, x, c, cam, base, M=M)[0]
    raise ValueError(method)


# ------------------------------------------------------------------ metrics
@torch.no_grad()
def del_ins(model, x, c, amap, steps=100):
    """Deletion / insertion AUC with 1% steps and a blurred baseline."""
    blur = E.blurred(x)
    order = torch.argsort(amap.flatten(), descending=True)
    n = order.numel()
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(n, device=order.device)
    ranks = ranks.view(1, 1, H, H)
    ks = [int(round(n * t / steps)) for t in range(steps + 1)]
    rm = torch.cat([(ranks < k) for k in ks])  # (101,1,H,W) True = top-k pixels
    dele = torch.where(rm, blur, x)
    ins = torch.where(rm, x, blur)
    sd = E.score(model, dele, c)
    si = E.score(model, ins, c)
    return float(torch.trapezoid(sd, dx=1 / steps)), float(torch.trapezoid(si, dx=1 / steps))


def norm01(m):
    m = m.detach().float().cpu().numpy()
    lo, hi = m.min(), m.max()
    return (m - lo) / (hi - lo) if hi > lo else np.zeros_like(m)


def top_iou(a, b, q=0.15):
    ta = a >= np.quantile(a, 1 - q)
    tb = b >= np.quantile(b, 1 - q)
    return (ta & tb).sum() / max((ta | tb).sum(), 1)


def rank_corr(a, b):
    a = F.avg_pool2d(torch.as_tensor(np.abs(a))[None, None], 4).flatten().numpy()
    b = F.avg_pool2d(torch.as_tensor(np.abs(b))[None, None], 4).flatten().numpy()
    r = spearmanr(a, b).statistic
    return 0.0 if np.isnan(r) else float(r)


# ------------------------------------------------------------------ data
def pick_images(ds, model, n, seed=0):
    xte, yte = load_split(ds, "test")
    pos = np.nonzero(yte == 1)[0] if ds == "pneumonia" else np.nonzero(yte != 5)[0]
    pos = np.random.RandomState(seed).permutation(pos)
    out, cnt = [], 0
    with torch.no_grad():
        for i in pos:
            x = to_tensor(xte[i:i + 1]).to(DEV)
            p = model(x).argmax(1).item()
            if p == yte[i]:
                out.append((x, int(yte[i])))
                if len(out) >= n:
                    break
    return out


def dataset_stuff(ds):
    xtr, _ = load_split(ds, "train")
    sub = xtr[np.random.RandomState(0).choice(len(xtr), 512, replace=False)]
    calib = to_tensor(sub)
    base = calib[:500].mean((0, 2, 3), keepdim=True).expand(1, 3, H, H).to(DEV).contiguous()
    return calib, base


# ------------------------------------------------------------------ quality
def quality(ds, bb, n, nsub, seeds_n):
    model = load_model(ds, bb)
    layer = target_layer(model, bb)
    calib, base = dataset_stuff(ds)
    qmodel = fake_quantize(model, calib)
    qlayer = target_layer(qmodel, bb)
    rnd = build(bb, DATASETS[ds][1], pretrained=False).to(DEV).eval()
    rlayer = target_layer(rnd, bb)
    imgs = pick_images(ds, model, n)
    print(f"[{ds}/{bb}] {len(imgs)} images", flush=True)
    rec = {m: {"del": [], "ins": [], "noise_ssim": [], "drift_ssim": [], "drift_iou": [],
               "rand_rho": [], "seed_rho": [], "time_gpu": []} for m in METHODS}
    examples = []
    gen = torch.Generator(device=DEV).manual_seed(0)
    for k, (x, c) in enumerate(imgs):
        ex = {"img": x[0].cpu().numpy(), "c": c}
        for m in METHODS:
            sync(); t = time.perf_counter()
            amap = explain(m, model, layer, x, c, base)
            sync(); rec[m]["time_gpu"].append(time.perf_counter() - t)
            d, i = del_ins(model, x, c, amap)
            rec[m]["del"].append(d); rec[m]["ins"].append(i)
            a = norm01(amap)
            ex[m] = a
            if k < nsub:
                xn = (x + 0.01 * torch.randn(x.shape, device=DEV, generator=gen)).clamp(0, 1)
                an = norm01(explain(m, model, layer, xn, c, base))
                rec[m]["noise_ssim"].append(ssim(a, an, data_range=1.0))
                aq = norm01(explain(m, qmodel, qlayer, x, c, base))
                rec[m]["drift_ssim"].append(ssim(a, aq, data_range=1.0))
                rec[m]["drift_iou"].append(top_iou(a, aq))
                ar = norm01(explain(m, rnd, rlayer, x, c, base))
                rec[m]["rand_rho"].append(rank_corr(a, ar))
        if k < nsub and seeds_n:
            maps = [norm01(E.kernelshap(model, x, c, seed=s)) for s in range(seeds_n)]
            rs = [rank_corr(maps[i], maps[j]) for i in range(seeds_n) for j in range(i + 1, seeds_n)]
            rec["KernelSHAP"]["seed_rho"].append(float(np.mean(rs)))
        if k < 3:
            examples.append(ex)
        print(f"  img {k + 1}/{len(imgs)}", flush=True)
    np.save(os.path.join(RES_DIR, f"examples_{ds}_{bb}.npy"), np.array(examples, dtype=object), allow_pickle=True)
    return rec


def run_quality():
    out = {}
    # reduced scope: MobileNetV2 on both datasets, the other backbones on pneumonia only
    cfg = [("pneumonia", "mobilenet_v2", 50, 15, 3), ("derma", "mobilenet_v2", 50, 15, 3),
           ("pneumonia", "efficientnet_b0", 20, 6, 0), ("pneumonia", "resnet18", 20, 6, 0)]
    for ds, bb, n, nsub, sd in cfg:
        out[f"{ds}/{bb}"] = quality(ds, bb, n, nsub, sd)
        json.dump(out, open(os.path.join(RES_DIR, "quality.json"), "w"))


# ------------------------------------------------------------------ latency
def latency_for(model, layer, device, x_list, base, methods, M=6, warm=1):
    import psutil
    res = {m: [] for m in methods}
    mem = {m: 0.0 for m in methods}
    for m in methods:
        for k, (x, c) in enumerate(x_list):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize() if device.type == "cuda" else None
            t = time.perf_counter()
            explain(m, model, layer, x, c, base, M=M)
            torch.cuda.synchronize() if device.type == "cuda" else None
            dt = time.perf_counter() - t
            if k >= warm:
                res[m].append(dt)
        if device.type == "cuda":
            mem[m] = torch.cuda.max_memory_allocated() / 2 ** 20
        else:
            mem[m] = psutil.Process().memory_info().rss / 2 ** 20
    return {m: {"median_s": float(np.median(res[m])), "mem_mb": mem[m]} for m in methods}


def run_latency():
    import copy
    out = {}
    torch.set_num_threads(4)
    ds = "pneumonia"
    calib, base = dataset_stuff(ds)
    for bb in BACKBONES:
        gm = load_model(ds, bb)
        imgs = pick_images(ds, gm, 8)
        cm = copy.deepcopy(gm).cpu().eval()
        imgs_c = [(x.cpu(), c) for x, c in imgs]
        print(f"latency {bb}", flush=True)
        out[f"cuda/{bb}"] = latency_for(gm, target_layer(gm, bb), torch.device("cuda"), imgs, base, METHODS, warm=2)
        out[f"cpu/{bb}"] = latency_for(cm, target_layer(cm, bb), torch.device("cpu"), imgs_c[:3], base.cpu(), METHODS, warm=1)
        json.dump(out, open(os.path.join(RES_DIR, "latency.json"), "w"), indent=1)


# ------------------------------------------------------------------ ablation
def run_ablation():
    import copy
    torch.set_num_threads(4)
    ds, bb = "pneumonia", "mobilenet_v2"
    model = load_model(ds, bb)
    layer = target_layer(model, bb)
    _, base = dataset_stuff(ds)
    imgs = pick_images(ds, model, 20)
    cm = copy.deepcopy(model).cpu().eval()
    rows = {}
    for M in (3, 4, 5, 6, 7):
        fs = []
        for x, c in imgs:
            a = explain("CGHA-Shap", model, layer, x, c, base, M=M)
            d, i = del_ins(model, x, c, a)
            fs.append(0.5 * (i + 1 - d))
        lat = latency_for(cm, target_layer(cm, bb), torch.device("cpu"),
                          [(x.cpu(), c) for x, c in imgs[:4]], base.cpu(), ["CGHA-Shap"], M=M, warm=1)
        rows[M] = {"F": float(np.mean(fs)), "cpu_latency_s": lat["CGHA-Shap"]["median_s"], "fpe": 3 + 2 ** (M + 1)}
        print(f"M={M} {rows[M]}", flush=True)
    # rho sensitivity
    import explainers
    rho_rows = {}
    for rho in (0.4, 0.6, 0.8):
        fs = []
        for x, c in imgs[:10]:
            cam = E.gradcam(model, layer, x, c)
            a, _ = E.cgha_shap(model, x, c, cam, base, M=6, rho=rho)
            d, i = del_ins(model, x, c, a)
            fs.append(0.5 * (i + 1 - d))
        rho_rows[rho] = float(np.mean(fs))
    json.dump({"M": rows, "rho": rho_rows}, open(os.path.join(RES_DIR, "ablation.json"), "w"), indent=1)


if __name__ == "__main__":
    {"quality": run_quality, "latency": run_latency, "ablation": run_ablation}[sys.argv[1]]()
