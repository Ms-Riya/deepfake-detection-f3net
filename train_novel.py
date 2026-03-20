
import sys
sys.path.insert(0, "/home/pesu/deepfake_project")

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import numpy as np
import json
from tqdm import tqdm
from PIL import Image

from dataset import FFppDataset, get_val_transform
from compression_aug import get_train_transform_with_compression, DEFAULT_COMPRESSION_IDX
from f3net_novel import F3NetNovel

# ── Config ────────────────────────────────────────────────────
PRETRAINED_PATH = "/home/pesu/deepfake_project/xception-b5690688.pth"
CHECKPOINT_DIR  = "/home/pesu/deepfake_project/checkpoints_novel"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS          = 25
LR              = 0.0002
WEIGHT_DECAY    = 1e-4
BATCH_SIZE      = 16
NUM_WORKERS     = 4
GRAD_CLIP       = 1.0
EARLY_STOP      = 7

# Which novelties to use — set both True for full novel model
USE_MULTISCALE     = True   # Novelty A
USE_COMP_EMBEDDING = True   # Novelty B
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


def get_dataloaders_novel():
    """
    Same as get_dataloaders() but uses compression augmentation
    in training transform — this is Novelty B part 1.
    """
    # Training uses compression augmentation transform
    train_ds = FFppDataset("train", transform=get_train_transform_with_compression())
    val_ds   = FFppDataset("val",   transform=get_val_transform())
    test_ds  = FFppDataset("test",  transform=get_val_transform())

    # Weighted sampler for class balance
    weights = train_ds.get_sample_weights()
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=NUM_WORKERS,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE * 2,
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE * 2,
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=True)
    return {"train": train_loader, "val": val_loader, "test": test_loader}


def train_one_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0.0
    all_labels = []
    all_probs  = []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for images, labels in pbar:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Compression level — all c23 during training
        # (compression augmentation handles robustness, not this label)
        comp = torch.ones(images.shape[0], dtype=torch.long,
                          device=DEVICE) * DEFAULT_COMPRESSION_IDX

        optimizer.zero_grad()
        logits, scale_weights = model(images, comp)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader), compute_metrics(all_labels, all_probs)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs  = []

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        comp   = torch.ones(images.shape[0], dtype=torch.long,
                            device=DEVICE) * DEFAULT_COMPRESSION_IDX

        logits, _ = model(images, comp)
        loss       = criterion(logits, labels)

        total_loss += loss.item()
        all_probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    return total_loss / len(loader), compute_metrics(all_labels, all_probs)


def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("Loading data (with compression augmentation)...")
    loaders = get_dataloaders_novel()

    print(f"\nBuilding F3NetNovel...")
    print(f"  Novelty A (multi-scale DCT) : {USE_MULTISCALE}")
    print(f"  Novelty B (compression emb) : {USE_COMP_EMBEDDING}")

    model = F3NetNovel(
        pretrained_path=PRETRAINED_PATH,
        device=str(DEVICE),
        use_multiscale=USE_MULTISCALE,
        use_comp_embedding=USE_COMP_EMBEDDING,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {n_params:.1f}M")

    criterion = nn.BCEWithLogitsLoss()

    # Separate LRs: backbone gets 10x lower LR to preserve pretrained weights
    backbone_params = (list(model.fad_backbone.parameters()) +
                       list(model.lfs_backbone.parameters()))
    new_params = [p for p in model.parameters()
                  if not any(p is bp for bp in backbone_params)]

    optimizer = optim.Adam([
        {"params": backbone_params, "lr": LR * 0.1},
        {"params": new_params,      "lr": LR},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_auc         = 0.0
    patience_counter = 0
    history          = []

    print(f"\nStarting training — {EPOCHS} epochs on {DEVICE}\n")
    print(f"{'Ep':>3} | {'TrLoss':>7} | {'TrAUC':>7} | {'VaLoss':>7} | "
          f"{'VaAUC':>7} | {'BalAcc':>7} | {'RealAcc':>8} | {'FakeAcc':>8}")
    print("-" * 75)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_m = train_one_epoch(
            model, loaders["train"], optimizer, criterion, epoch)
        val_loss, val_m = evaluate(
            model, loaders["val"], criterion)
        scheduler.step()

        print(f"{epoch:>3} | {train_loss:>7.4f} | {train_m['auc']:>7.4f} | "
              f"{val_loss:>7.4f} | {val_m['auc']:>7.4f} | {val_m['bal_acc']:>7.4f} | "
              f"{val_m['real_acc']:>8.4f} | {val_m['fake_acc']:>8.4f}")

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss,
                        **{f"train_{k}": v for k, v in train_m.items()},
                        **{f"val_{k}":   v for k, v in val_m.items()}})

        with open(os.path.join(CHECKPOINT_DIR, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        if val_m["auc"] > best_auc:
            best_auc         = val_m["auc"]
            patience_counter = 0
            torch.save({
                "epoch":       epoch,
                "model":       model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "val_auc":     best_auc,
                "val_metrics": val_m,
            }, os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print(f"     ✓ Best saved  AUC={best_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    print(f"\nDone. Best Val AUC : {best_auc:.4f}")
    print(f"Baseline Val AUC   : 0.9008  (F3Net Run 1)")
    improvement = best_auc - 0.9008
    print(f"Improvement        : {improvement:+.4f}")
    print(f"Saved to           : {CHECKPOINT_DIR}/best_model.pth")


if __name__ == "__main__":
    train()
