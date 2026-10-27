"""Fine-tune the three backbones on both datasets; also evaluate FP32 vs fake-INT8."""
import json
import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from common import *

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 8


def predict(model, x_uint8, bs=128):
    model.eval()
    dev = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for i in range(0, len(x_uint8), bs):
            out.append(torch.softmax(model(to_tensor(x_uint8[i:i + bs]).to(dev)), 1).cpu())
    return torch.cat(out).numpy()


def metric(ds, probs, y):
    if ds == "pneumonia":
        return roc_auc_score(y, probs[:, 1])
    return balanced_accuracy_score(y, probs.argmax(1))


def augment(x):
    if torch.rand(1) < 0.5:
        x = x.flip(3)
    return x


def train_one(ds, bb, data):
    (xtr, ytr), (xva, yva) = data["train"], data["val"]
    nc = DATASETS[ds][1]
    torch.manual_seed(0)
    model = build(bb, nc).to(DEV)
    cnt = np.bincount(ytr, minlength=nc).astype(np.float32)
    w = torch.tensor(cnt.sum() / (nc * np.maximum(cnt, 1)), device=DEV)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    scaler = torch.amp.GradScaler()
    best, bs = -1, 64
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.RandomState(ep).permutation(len(xtr))
        tot = 0
        for i in range(0, len(perm) - bs + 1, bs):
            idx = perm[i:i + bs]
            xb = augment(to_tensor(xtr[idx])).to(DEV)
            yb = torch.from_numpy(ytr[idx]).to(DEV)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(model(xb), yb)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
        sched.step()
        m = metric(ds, predict(model, xva), yva)
        print(f"  {ds}/{bb} ep{ep + 1}/{EPOCHS} loss={tot / (len(perm) // bs):.4f} val={m:.4f}", flush=True)
        if m > best:
            best = m
            torch.save(model.state_dict(), ckpt_path(ds, bb))
    return best


def main():
    rp = os.path.join(RES_DIR, "classification.json")
    results = json.load(open(rp)) if os.path.exists(rp) else {}  # resume: skip finished runs
    for ds in DATASETS:
        data = {s: load_split(ds, s) for s in ("train", "val", "test")}
        calib = to_tensor(data["train"][0][np.random.RandomState(0).choice(len(data["train"][0]), 512, replace=False)])
        for bb in BACKBONES:
            if f"{ds}/{bb}" in results:
                continue
            print(f"== {ds} / {bb}", flush=True)
            val = train_one(ds, bb, data)
            model = load_model(ds, bb)
            xte, yte = data["test"]
            fp = metric(ds, predict(model, xte), yte)
            q = fake_quantize(model, calib)
            i8 = metric(ds, predict(q, xte), yte)
            results[f"{ds}/{bb}"] = {"val": val, "fp32": fp, "int8": i8}
            print(f"   test fp32={fp:.4f} int8={i8:.4f}", flush=True)
            json.dump(results, open(os.path.join(RES_DIR, "classification.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
