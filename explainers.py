"""Grad-CAM, Occlusion, KernelSHAP (Captum) and the proposed CGHA-Occ / CGHA-Shap."""
import itertools
from math import factorial
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from skimage.segmentation import slic
from captum.attr import KernelShap

H = 224


@torch.no_grad()
def score(model, xs, c, bs=64):
    """Softmax probability of class c for a batch of images (in [0,1])."""
    out = []
    for i in range(0, len(xs), bs):
        out.append(torch.softmax(model(xs[i:i + bs]), 1)[:, c])
    return torch.cat(out)


def blurred(x, k=21, sigma=6.0):
    ax = torch.arange(k, device=x.device).float() - k // 2
    g = torch.exp(-ax ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :]).expand(3, 1, k, k)
    return F.conv2d(F.pad(x, (k // 2,) * 4, mode="reflect"), w, groups=3)


# ------------------------------------------------------------------ Grad-CAM
def gradcam(model, layer, x, c):
    """x: (1,3,H,W). Returns (H,W) ReLU'd CAM (unnormalised)."""
    feats, grads = [], []
    def fwd(m, i, o):
        feats.append(o)
        o.register_hook(lambda g: grads.append(g))
    h1 = layer.register_forward_hook(fwd)
    h2 = h1
    try:
        with torch.enable_grad():
            xx = x.clone().requires_grad_(True)
            out = model(xx)
            model.zero_grad()
            out[0, c].backward()
    finally:
        h1.remove()
        h2.remove()
    a, g = feats[0].detach(), grads[0].detach()
    w = g.mean((2, 3), keepdim=True)
    cam = F.relu((w * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, (H, H), mode="bilinear", align_corners=False)
    return cam[0, 0]


# ------------------------------------------------------------------ Occlusion
def occlusion(model, x, c, base, p=16, s=8, bs=64):
    """Score drop for a p x p patch moved with stride s; averaged over overlaps."""
    pos = list(range(0, H - p + 1, s))
    base_score = score(model, x, c)[0]
    batch, coords = [], []
    for u in pos:
        for v in pos:
            z = x.clone()
            z[:, :, u:u + p, v:v + p] = base[:, :, u:u + p, v:v + p]
            batch.append(z)
            coords.append((u, v))
    sc = score(model, torch.cat(batch), c, bs)
    drop = base_score - sc
    amap = torch.zeros(H, H, device=x.device)
    cnt = torch.zeros(H, H, device=x.device)
    for (u, v), d in zip(coords, drop):
        amap[u:u + p, v:v + p] += d
        cnt[u:u + p, v:v + p] += 1
    return amap / cnt.clamp_min(1)


# ------------------------------------------------------------------ KernelSHAP
def kernelshap(model, x, c, n=1000, d=50, seed=0, bs=100):
    img = x[0].permute(1, 2, 0).cpu().numpy().astype(np.double)
    seg = slic(img, n_segments=d, compactness=1.0, start_label=0, convert2lab=False)
    mask = torch.from_numpy(seg).to(x.device)[None, None].expand(1, 3, H, H).long()
    torch.manual_seed(seed)
    np.random.seed(seed)
    ks = KernelShap(lambda z: torch.softmax(model(z), 1))
    with torch.no_grad():
        attr = ks.attribute(x, baselines=blurred(x), target=c, feature_mask=mask,
                            n_samples=n, perturbations_per_eval=bs)
    return attr[0].sum(0)


# ------------------------------------------------------------------ CGHA
def _regions(cam, rho=0.6, kmax=16, M=6, grid=7):
    """Steps 1-3: select CAM-heavy cells, merge into <=M connected regions."""
    L = cam / cam.sum().clamp_min(1e-12)
    cell = H // grid
    mass = L.reshape(grid, cell, grid, cell).sum((1, 3)).cpu().numpy()
    order = np.argsort(-mass.ravel())
    sel, cum = [], 0.0
    for i in order:
        sel.append(i)
        cum += mass.ravel()[i]
        if cum >= rho or len(sel) >= kmax:
            break
    m = np.zeros(grid * grid, bool)
    m[sel] = True
    m = m.reshape(grid, grid)
    lab, k = ndimage.label(m)
    if k < M and M < 10 ** 6:
        # Selected cells form too few connected regions (typical for diffuse CAMs):
        # fall back to the top-M individual cells as players so the Shapley game is non-trivial.
        lab = np.zeros_like(lab)
        top = [i for i in sel][:M]
        for n, i in enumerate(top, 1):
            lab[i // grid, i % grid] = n
        return L, lab, list(range(1, len(top) + 1)), mass, cell
    comps = sorted(range(1, k + 1), key=lambda r: -mass[lab == r].sum())[:M]
    return L, lab, comps, mass, cell


def _expand(lab, comps, cell):
    """(R,H,W) boolean pixel masks for each region."""
    out = []
    for r in comps:
        out.append(np.kron((lab == r).astype(np.uint8), np.ones((cell, cell), np.uint8)).astype(bool))
    return out


def cgha_shap(model, x, c, cam, base, M=6, rho=0.6, kmax=16, bs=64):
    """Exact Shapley values over the top-M CAM regions + 1 background player."""
    L, lab, comps, mass, cell = _regions(cam, rho, kmax, M)
    masks = [torch.from_numpy(m).to(x.device) for m in _expand(lab, comps, cell)]
    bg = ~torch.stack(masks).any(0)
    players = masks + [bg]
    n = len(players)
    coal = list(itertools.product([0, 1], repeat=n))
    batch = []
    for co in coal:
        keep = torch.zeros(H, H, dtype=torch.bool, device=x.device)
        for j, on in enumerate(co):
            if on:
                keep |= players[j]
        batch.append(torch.where(keep, x[0], base[0]))
    v = dict(zip(coal, score(model, torch.stack(batch), c, bs).tolist()))
    phi = np.zeros(n)
    for i in range(n):
        for co in coal:
            if co[i] == 0:
                s = sum(co)
                w = factorial(s) * factorial(n - s - 1) / factorial(n)
                co2 = co[:i] + (1,) + co[i + 1:]
                phi[i] += w * (v[co2] - v[co])
    amap = torch.zeros(H, H, device=x.device)
    for j, pm in enumerate(players):
        w = L[pm] + 1e-12
        amap[pm] = float(phi[j]) * w / w.sum()
    return amap, phi


def cgha_occ(model, x, c, cam, base, rho=0.6, kmax=16, r=2, M=6, bs=64):
    """Hierarchical occlusion: occlude selected cells, refine the top-r cells."""
    L, lab, comps, mass, cell = _regions(cam, rho, kmax, M=10 ** 6)
    sel_mask = lab > 0
    cells = [(i, j) for i, j in zip(*np.nonzero(sel_mask))]
    b0 = score(model, x, c)[0]
    batch = []
    for (i, j) in cells:
        z = x.clone()
        sl = (slice(None), slice(None), slice(i * cell, (i + 1) * cell), slice(j * cell, (j + 1) * cell))
        z[sl] = base[sl]
        batch.append(z)
    drops = (b0 - score(model, torch.cat(batch), c, bs)).cpu().numpy()
    amap = torch.zeros(H, H, device=x.device)
    for (i, j), d in zip(cells, drops):
        ys, xs = slice(i * cell, (i + 1) * cell), slice(j * cell, (j + 1) * cell)
        w = L[ys, xs] + 1e-12
        amap[ys, xs] = float(d) * w / w.sum()
    # refine the r most important cells with 16x16 patches, stride 8, 48x48 window
    for k in np.argsort(-drops)[:r]:
        i, j = cells[k]
        y0 = int(np.clip(i * cell + cell // 2 - 24, 0, H - 48))
        x0 = int(np.clip(j * cell + cell // 2 - 24, 0, H - 48))
        batch, co = [], []
        for u in range(y0, y0 + 48 - 16 + 1, 8):
            for v in range(x0, x0 + 48 - 16 + 1, 8):
                z = x.clone()
                z[:, :, u:u + 16, v:v + 16] = base[:, :, u:u + 16, v:v + 16]
                batch.append(z)
                co.append((u, v))
        dd = b0 - score(model, torch.cat(batch), c, bs)
        fine = torch.zeros(H, H, device=x.device)
        cnt = torch.zeros(H, H, device=x.device)
        for (u, v), d in zip(co, dd):
            fine[u:u + 16, v:v + 16] += d
            cnt[u:u + 16, v:v + 16] += 1
        fine = fine / cnt.clamp_min(1)
        ys, xs = slice(i * cell, (i + 1) * cell), slice(j * cell, (j + 1) * cell)
        f = F.relu(fine[ys, xs]) + 1e-12
        f = f / f.sum()
        w = L[ys, xs] + 1e-12
        w = w / w.sum()
        amap[ys, xs] = float(drops[k]) * (0.5 * f + 0.5 * w)
    return amap, 1 + len(cells) + 0  # (map, cells used)
