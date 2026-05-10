"""
Training CorticalNetwork on MNIST.
Device: MPS (Apple M1) with CPU fallback.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cortical_column.config import (
    BATCH_SIZE,
    EPOCHS,
    LATENT_DIM,
    LR,
    N_CLASSES,
    N_PATCHES,
    PATCH_SIZE,
)
from cortical_column.cortical_network import CorticalNetwork


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dataloaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    """
    Download MNIST, return (train_loader, val_loader).
    Normalization: mean=0.1307, std=0.3081 (standard MNIST values).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )
    val_dataset = datasets.MNIST(
        root="data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return train_loader, val_loader


def train() -> None:
    device = get_device()
    print(f"Using device: {device}")

    model = CorticalNetwork(
        n_columns=N_PATCHES,
        patch_size=PATCH_SIZE,
        latent_dim=LATENT_DIM,
        n_classes=N_CLASSES,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss()

    train_loader, val_loader = get_dataloaders(BATCH_SIZE)

    for epoch in range(EPOCHS):
        # ---- Training ----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, latents = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += images.size(0)

        scheduler.step()

        train_loss /= train_total
        train_acc = train_correct / train_total

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        mean_active_sum = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits, latents = model(images)
                loss = loss_fn(logits, labels)

                val_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)

                # latents: [B, 16, 128] — fraction of non-zero activations
                mean_active_sum += (latents != 0).float().mean().item()
                n_val_batches += 1

        val_loss /= val_total
        val_acc = val_correct / val_total
        mean_active = mean_active_sum / n_val_batches

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
            f"mean_active={mean_active:.3f}"
        )

    # ---- Save model checkpoint ----
    torch.save(model.state_dict(), "cortical_network_mnist.pt")
    print("Model checkpoint saved to cortical_network_mnist.pt")

    # ---- Save test latents for UMAP visualization ----
    model.eval()
    all_latents = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            _, latents = model(images)
            # Move to CPU before converting to numpy
            all_latents.append(latents.cpu().numpy())
            all_labels.append(labels.numpy())

    all_latents = np.concatenate(all_latents, axis=0)  # [N_test, 16, 128]
    all_labels = np.concatenate(all_labels, axis=0)     # [N_test]

    np.save("test_latents.npy", all_latents)
    np.save("test_labels.npy", all_labels)
    print(f"Test latents saved: {all_latents.shape}, labels: {all_labels.shape}")


if __name__ == "__main__":
    train()
