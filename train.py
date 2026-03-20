
import sys
sys.path.insert(0, "/home/pesu/deepfake_project")

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import numpy as np
import json
from tqdm import tqdm

from dataset import get_dataloaders
from f3net import F3Net

# ── Config ────────────────────────────────────────────────────
PRETRAINED_PATH = "/home/pesu/deepfake_project/xception-b5690688.pth"
CHECKPOINT_DIR  = "/home/pesu/deepfake_project/checkpoints_v2"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS          = 30
LR              = 0.0001       # Lower than before
WEIGHT_DECAY    = 1e-4
MODE            = "Mix"
GRAD_CLIP       = 1.0
EARLY_STOP      = 7            # More patience than before

# Fix for class imbalance: fake:real = 4:1
# pos_weight = n_negative / n_positive = 22963 / 91814 = 0.25
# This tells the loss: penalise missing a real face 4x more
POS_WEIGHT = torch.tensor([0.25])
# ─────────────────────────────────────────────────────────────


def compute_metrics(labels, probs, threshold=0.5):
    preds    = (np.array(probs) >= threshold).astype(int)
    labels   = np.array(labels)
    auc      = roc_auc_score(labels, probs)
    bal_acc  = balanced_accuracy_score(labels, preds)
    real_acc = (preds[labels == 0] == 0).mean() if (labels == 0).any() else 0.0
    fake_acc = (preds[labels == 1] == 1).mean() if (labels == 1).any() else 0.0
    return {
        "auc":      round(float(auc),      4),
        "bal_acc":  round(float(bal_acc),  4),
        "real_acc": round(float(real_acc), 4),
        "fake_acc": round(float(fake_acc), 4),
    }


def find_best_threshold(labels, probs):
    """Find threshold that maximises balanced accuracy."""
    best_t       = 0.5
    best_bal_acc = 0.0
    for t in np.arange(0.1, 0.9, 0.01):
        preds   = (probs >= t).astype(int)
        bal_acc = balanced_accuracy_score(labels, preds)
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_t       = t
    return best_t, best_bal_acc


def train_one_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0.0
    all_labels = []
    all_probs  = []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for images, labels in pbar:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(all_labels, all_probs)
    return avg_loss, metrics


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs  = []

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        logits = model(images)
        loss   = criterion(logits, labels)
        total_loss += loss.item()
        all_probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(all_labels, all_probs)
    return avg_loss, metrics, np.array(all_labels), np.array(all_probs)


def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("Loading data...")
    loaders = get_dataloaders()

    print(f"\nBuilding F3Net (mode={MODE})...")
    model = F3Net(
        pretrained_path=PRETRAINED_PATH,
        mode=MODE,
        device=str(DEVICE)
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.1f}M")

    # Loss with pos_weight to correct class imbalance
    # pos_weight < 1 means: reduce penalty for missing fakes
    # (since we have 4x more fakes, we want to be more careful about reals)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=POS_WEIGHT.to(DEVICE)
    )

    # Separate learning rates: lower for pretrained backbone, higher for new layers
    backbone_params = (list(model.fad_backbone.parameters()) +
                       list(model.lfs_backbone.parameters()))
    new_params      = (list(model.fad.parameters()) +
                       list(model.lfs.parameters()) +
                       list(model.mixblock.parameters()) +
                       list(model.classifier.parameters()))

    optimizer = optim.Adam([
        {"params": backbone_params, "lr": LR * 0.1},   # 10x lower for backbone
        {"params": new_params,      "lr": LR},
    ], weight_decay=WEIGHT_DECAY)

    # Cosine annealing with warm restarts — avoids getting stuck
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)

    best_auc         = 0.0
    patience_counter = 0
    history          = []

    print(f"\nStarting training for {EPOCHS} epochs on {DEVICE}")
    print(f"pos_weight={POS_WEIGHT.item():.2f} | LR={LR} | backbone LR={LR*0.1}\n")
    print(f"{'Epoch':>5} | {'TrLoss':>7} | {'TrAUC':>7} | {'VaLoss':>7} | "
          f"{'VaAUC':>7} | {'BalAcc':>7} | {'RealAcc':>8} | {'FakeAcc':>8}")
    print("-" * 80)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_m = train_one_epoch(
            model, loaders["train"], optimizer, criterion, epoch)

        val_loss, val_m, val_labels, val_probs = evaluate(
            model, loaders["val"], criterion)

        scheduler.step()

        # Find best threshold on val set this epoch
        best_t, best_bal = find_best_threshold(val_labels, val_probs)

        print(f"{epoch:>5} | {train_loss:>7.4f} | {train_m['auc']:>7.4f} | "
              f"{val_loss:>7.4f} | {val_m['auc']:>7.4f} | {val_m['bal_acc']:>7.4f} | "
              f"{val_m['real_acc']:>8.4f} | {val_m['fake_acc']:>8.4f}  "
              f"[best_t={best_t:.2f} bal={best_bal:.4f}]")

        history.append({
            "epoch":           epoch,
            "train_loss":      train_loss,
            "val_loss":        val_loss,
            "best_threshold":  float(best_t),
            **{f"train_{k}": v for k, v in train_m.items()},
            **{f"val_{k}":   v for k, v in val_m.items()},
        })

        with open(os.path.join(CHECKPOINT_DIR, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        if val_m["auc"] > best_auc:
            best_auc         = val_m["auc"]
            patience_counter = 0
            torch.save({
                "epoch":           epoch,
                "model":           model.state_dict(),
                "optimizer":       optimizer.state_dict(),
                "val_auc":         best_auc,
                "val_metrics":     val_m,
                "best_threshold":  float(best_t),
            }, os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print(f"         ✓ Best saved  AUC={best_auc:.4f}  threshold={best_t:.2f}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    print(f"\nDone. Best Val AUC: {best_auc:.4f}")
    print(f"Saved to: {CHECKPOINT_DIR}/best_model.pth")


if __name__ == "__main__":
    train()
