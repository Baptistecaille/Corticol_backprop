"""Supervised training on CIFAR-10 for both CorticalNetwork and ViT baseline.

Usage
─────
    python -m cortical_column.cifar_train_supervised
    python -m cortical_column.cifar_train_supervised --model cortical
    python -m cortical_column.cifar_train_supervised --model vit
    python -m cortical_column.cifar_train_supervised --model both

Output checkpoints:
    cortical_cifar10_supervised.pt
    vit_cifar10_supervised.pt
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cortical_column.config_cifar import (
    CIFAR_MEAN, CIFAR_STD,
    CIFAR_BATCH_SIZE, CIFAR_VAL_BATCH_SIZE,
    CIFAR_LR, CIFAR_EPOCHS, CIFAR_WEIGHT_DECAY,
    CIFAR_N_CHANNELS, CIFAR_PATCH_SIZE, CIFAR_N_CLASSES,
)
from cortical_column.config import N_PATCHES, LATENT_DIM, NUM_WORKERS
from cortical_column.cortical_network import CorticalNetwork
from cortical_column.baselines.vit_encoder import ViTEncoder


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():   return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def get_cifar10(batch_size: int, train: bool, augment: bool = False) -> DataLoader:
    tf = [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
    if augment and train:
        tf = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ] + tf
    dataset = datasets.CIFAR10(
        root="./data", train=train, download=True, transform=transforms.Compose(tf)
    )
    n_workers = NUM_WORKERS if torch.cuda.is_available() else 0
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=train,
        num_workers=n_workers, pin_memory=torch.cuda.is_available(),
        drop_last=train,
    )


def train_one(
    model: nn.Module,
    name: str,
    output_ckpt: str,
    device: torch.device,
) -> float:
    """Train a single model on CIFAR-10. Returns best val accuracy."""
    train_loader = get_cifar10(CIFAR_BATCH_SIZE, train=True,  augment=True)
    val_loader   = get_cifar10(CIFAR_VAL_BATCH_SIZE, train=False)

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CIFAR_LR, weight_decay=CIFAR_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CIFAR_EPOCHS
    )
    loss_fn = nn.CrossEntropyLoss()

    print(f"\n{'─'*60}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model       : {name}  ({n_params:,} params)")
    print(f"Checkpoint  : {output_ckpt}")
    print(f"Epochs      : {CIFAR_EPOCHS}  |  LR: {CIFAR_LR}  |  Device: {device}")
    print(f"{'─'*60}")

    best_val_acc = 0.0

    for epoch in range(1, CIFAR_EPOCHS + 1):
        model.train()
        train_loss, train_correct, n_train = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss    += loss.item() * labels.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            n_train       += labels.size(0)
        scheduler.step()

        model.eval()
        val_correct, n_val = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits, _ = model(images)
                val_correct += (logits.argmax(1) == labels).sum().item()
                n_val       += labels.size(0)

        train_acc = train_correct / n_train
        val_acc   = val_correct   / n_val
        print(
            f"Epoch {epoch:3d}/{CIFAR_EPOCHS}"
            f" | train_loss={train_loss/n_train:.4f}"
            f" | train_acc={train_acc:.4f}"
            f" | val_acc={val_acc:.4f}",
            end="",
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_ckpt)
            print("  ✓ saved")
        else:
            print()

    print(f"\nBest val_acc [{name}]: {best_val_acc*100:.2f}%")
    return best_val_acc


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(which: str = "both"):
    device = get_device()
    results = {}

    if which in ("cortical", "both"):
        model = CorticalNetwork(
            patch_size=CIFAR_PATCH_SIZE,
            n_channels=CIFAR_N_CHANNELS,
            n_classes=CIFAR_N_CLASSES,
        )
        acc = train_one(model, "CorticalNetwork", "cortical_cifar10_supervised.pt", device)
        results["CorticalNetwork"] = acc

    if which in ("vit", "both"):
        model = ViTEncoder()
        acc = train_one(model, "ViT-Tiny", "vit_cifar10_supervised.pt", device)
        results["ViT-Tiny"] = acc

    print("\n" + "═"*40)
    print("CIFAR-10 supervised results:")
    for name, acc in results.items():
        print(f"  {name:<20} {acc*100:.2f}%")
    print("═"*40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["cortical", "vit", "both"], default="both",
        help="Which model to train."
    )
    args = parser.parse_args()
    main(args.model)
