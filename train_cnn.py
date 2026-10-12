"""Train the CNN baseline, then cache 128-d features for the SNN head.

The TEST set is never evaluated here (only its features are cached).
Usage:  python train_cnn.py --data-dir data
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from dataset import get_loaders, set_seed
from models import CNN


def evaluate(model, loader, device):
    crit = nn.CrossEntropyLoss()
    model.eval()
    loss_sum, ys, ps = 0.0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += crit(out, y).item() * y.size(0)
            ys.append(y.cpu())
            ps.append(out.argmax(1).cpu())
    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()
    return loss_sum / len(y), float((y == p).mean()), float(f1_score(y, p, average="macro"))


@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    X, Y = [], []
    for x, y in loader:
        X.append(model.embed(x.to(device)).cpu())
        Y.append(y)
    return torch.cat(X), torch.cat(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--aug-copies", type=int, default=3,
                    help="augmented copies of train features to cache for the SNN head")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--feat-dir", default="features")
    ap.add_argument("--results-dir", default="results")
    a = ap.parse_args()

    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for d in (a.models_dir, a.feat_dir, a.results_dir):
        Path(d).mkdir(exist_ok=True)
    print("device:", device)

    L = get_loaders(a.data_dir, a.batch_size)
    classes = L["classes"]
    print("classes:", classes, "| train counts:", L["counts"],
          "| class weights:", [round(w, 3) for w in L["class_weights"].tolist()])

    model = CNN(n_classes=len(classes)).to(device)
    crit = nn.CrossEntropyLoss(weight=L["class_weights"].to(device))
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    best_f1, best_loss, bad = -1.0, 1e9, 0
    hist = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    ckpt = Path(a.models_dir) / "cnn_best.pt"

    for ep in range(1, a.epochs + 1):
        model.train()
        tl, n = 0.0, 0
        for x, y in L["train"]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tl += loss.item() * y.size(0)
            n += y.size(0)
        vl, va, vf = evaluate(model, L["val"], device)
        sched.step(vf)
        for k, v in zip(hist, (tl / n, vl, va, vf)):
            hist[k].append(v)
        print(f"ep {ep:02d} | train loss {tl/n:.4f} | val loss {vl:.4f} | val acc {va:.3f} | val macro-F1 {vf:.3f}")

        if vf > best_f1 + 1e-6 or (abs(vf - best_f1) <= 1e-6 and vl < best_loss):
            best_f1, best_loss, bad = vf, vl, 0
            torch.save({"state": model.state_dict(), "classes": classes}, ckpt)
        else:
            bad += 1
            if bad >= a.patience:
                print(f"early stopping at epoch {ep}")
                break

    # curves
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(hist["val_acc"], label="val acc"); ax[1].plot(hist["val_f1"], label="val macro-F1")
    ax[1].set_title("Validation"); ax[1].legend()
    plt.tight_layout(); plt.savefig(Path(a.results_dir) / "cnn_curves.png", dpi=150); plt.close()
    json.dump(hist, open(Path(a.results_dir) / "cnn_history.json", "w"), indent=2)

    # reload best, cache features
    ck = torch.load(ckpt, map_location=device)
    model.load_state_dict(ck["state"])
    vl, va, vf = evaluate(model, L["val"], device)
    print(f"\nBEST CNN  | val acc {va:.3f} | val macro-F1 {vf:.3f}  (test not touched)")

    fd = Path(a.feat_dir)
    for name, loader in [("train", L["train_clean"]), ("val", L["val"]), ("test", L["test"])]:
        X, Y = extract(model, loader, device)
        torch.save({"x": X, "y": Y, "classes": classes}, fd / f"{name}.pt")
        print(f"cached {name}: {tuple(X.shape)}")
    if a.aug_copies > 0:
        Xs, Ys = [], []
        for _ in range(a.aug_copies):
            X, Y = extract(model, L["train"], device)   # augmented transform
            Xs.append(X); Ys.append(Y)
        torch.save({"x": torch.cat(Xs), "y": torch.cat(Ys), "classes": classes}, fd / "train_aug.pt")
        print(f"cached train_aug: {a.aug_copies} copies")


if __name__ == "__main__":
    main()