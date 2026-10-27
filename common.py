"""Shared utilities: data, models, normalisation wrapper, fake-INT8 quantisation."""
import copy
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
RES_DIR = os.path.join(ROOT, "results")
for _d in (DATA_DIR, CKPT_DIR, RES_DIR):
    os.makedirs(_d, exist_ok=True)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASETS = {
    # name -> (medmnist flag, n_classes)
    "pneumonia": ("pneumoniamnist", 2),
    "derma": ("dermamnist", 7),
}
BACKBONES = ["mobilenet_v2", "efficientnet_b0", "resnet18"]
IMSIZE = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_split(ds, split):
    """Return uint8 images (N,224,224,3) and int labels (N,)."""
    import medmnist
    flag, _ = DATASETS[ds]
    cls = getattr(medmnist, medmnist.INFO[flag]["python_class"])
    d = cls(split=split, download=True, size=IMSIZE, root=DATA_DIR)
    x = d.imgs
    if x.ndim == 3:
        x = np.repeat(x[..., None], 3, axis=-1)
    return x, d.labels.reshape(-1).astype(np.int64)


def to_tensor(x_uint8):
    """uint8 NHWC -> float NCHW in [0,1]."""
    return torch.from_numpy(x_uint8).permute(0, 3, 1, 2).float() / 255.0


class Normalised(nn.Module):
    """Wraps a network so inputs are in [0,1] (perturbations stay in image space)."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("mean", MEAN.clone())
        self.register_buffer("std", STD.clone())

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def build(backbone, n_classes, pretrained=True):
    tv = torchvision.models
    if backbone == "mobilenet_v2":
        net = tv.mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
        net.classifier[1] = nn.Linear(net.last_channel, n_classes)
    elif backbone == "efficientnet_b0":
        net = tv.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, n_classes)
    elif backbone == "resnet18":
        net = tv.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        net.fc = nn.Linear(net.fc.in_features, n_classes)
    else:
        raise ValueError(backbone)
    return Normalised(net)


def target_layer(model, backbone):
    net = model.net
    return {"mobilenet_v2": lambda: net.features[-1],
            "efficientnet_b0": lambda: net.features[-1],
            "resnet18": lambda: net.layer4[-1]}[backbone]()


def ckpt_path(ds, backbone):
    return os.path.join(CKPT_DIR, f"{ds}_{backbone}.pt")


def load_model(ds, backbone, device=DEV):
    m = build(backbone, DATASETS[ds][1], pretrained=False)
    m.load_state_dict(torch.load(ckpt_path(ds, backbone), map_location="cpu"))
    return m.to(device).eval()


# ---------------------------------------------------------------- fake INT8
class _FQ(nn.Module):
    """Per-tensor asymmetric 8-bit fake-quantiser applied to a layer's input."""

    def __init__(self):
        super().__init__()
        self.lo, self.hi, self.on, self.cal = None, None, False, False

    def forward(self, x):
        if self.cal:
            # percentile clipping (robust to the outliers of SiLU / squeeze-excite layers)
            flat = x.detach().flatten()
            if flat.numel() > 2_000_000:
                flat = flat[torch.randint(0, flat.numel(), (2_000_000,), device=flat.device)]
            lo, hi = torch.quantile(flat, 1e-4), torch.quantile(flat, 1 - 1e-4)
            self.lo = lo if self.lo is None else 0.5 * (self.lo + lo)
            self.hi = hi if self.hi is None else 0.5 * (self.hi + hi)
            return x
        if not self.on:
            return x
        scale = (self.hi - self.lo).clamp_min(1e-8) / 255.0
        zp = torch.round(-self.lo / scale)
        q = torch.clamp(torch.round(x / scale) + zp, 0, 255)
        return (q - zp) * scale


def fake_quantize(model, calib):
    """Post-training static INT8 simulation (quantise-dequantise).

    Weights: per-output-channel symmetric int8. Activations: per-tensor uint8
    with min/max calibrated on `calib` images. Stays differentiable (straight
    through is unnecessary: the rounding is piecewise constant, gradients flow
    through the dequantised graph), which is how the paper obtains gradients
    for Grad-CAM on the INT8 model.
    """
    q = copy.deepcopy(model).eval()
    fqs = []
    for mod in q.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            w = mod.weight.data
            dims = tuple(range(1, w.dim()))
            s = w.abs().amax(dim=dims, keepdim=True).clamp_min(1e-8) / 127.0
            mod.weight.data = torch.clamp(torch.round(w / s), -127, 127) * s
            fq = _FQ()
            fqs.append(fq)
            mod.register_forward_pre_hook(lambda m, a, fq=fq: (fq(a[0]),))
    for fq in fqs:
        fq.cal = True
    dev = next(q.parameters()).device
    with torch.no_grad():
        for i in range(0, len(calib), 64):
            q(calib[i:i + 64].to(dev))
    for fq in fqs:
        fq.cal, fq.on = False, True
    return q
