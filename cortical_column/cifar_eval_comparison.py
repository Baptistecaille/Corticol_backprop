"""CIFAR-10 linear probe comparison: CorticalNetwork JEPA vs ViT (I-JEPA baseline).

Evaluates all available checkpoints and generates a comparison table including
published SSL results for context.

Usage
─────
    python -m cortical_column.cifar_eval_comparison
    python -m cortical_column.cifar_eval_comparison --n_labels 100 500 1000 5000 -1

The script evaluates:
  (1) CorticalNetwork supervised       — cortical_cifar10_supervised.pt
  (2) CorticalNetwork JEPA             — cortical_cifar10_jepa.pt
  (3) ViT supervised                   — vit_cifar10_supervised.pt
  (4) ViT JEPA  (I-JEPA baseline)      — vit_cifar10_jepa.pt

Published SSL baselines (from original papers, CIFAR-10 linear probe):
  SimCLR  (ResNet-50, 23M)  : 90.6%  — Chen et al., 2020
  BYOL    (ResNet-50, 23M)  : 91.3%  — Grill et al., 2020
  MAE     (ViT-B,    86M)  : ~86%   — He et al., 2022
  I-JEPA  (ViT-H,  632M)  : N/A on CIFAR-10; 87.5% on CIFAR-100
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cortical_column.config import LATENT_DIM, N_PATCHES, NUM_WORKERS
from cortical_column.config_cifar import (
    CIFAR_MEAN, CIFAR_STD,
    CIFAR_N_CHANNELS, CIFAR_PATCH_SIZE, CIFAR_N_CLASSES,
)
from cortical_column.cortical_network import CorticalNetwork
from cortical_column.baselines.vit_encoder import ViTEncoder


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def get_cifar10(batch_size: int, train: bool) -> DataLoader:
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    ds = datasets.CIFAR10(root="./data", train=train, download=True, transform=tf)
    n_workers = NUM_WORKERS if torch.cuda.is_available() else 0
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=n_workers, pin_memory=torch.cuda.is_available())


def _stratified_indices(dataset, n_total: int) -> list[int]:
    n_classes = 10
    per_class = n_total // n_classes
    buckets: dict[int, list[int]] = {c: [] for c in range(n_classes)}
    for idx, (_, label) in enumerate(dataset):
        if len(buckets[label]) < per_class:
            buckets[label].append(idx)
        if all(len(v) >= per_class for v in buckets.values()):
            break
    return [idx for bucket in buckets.values() for idx in bucket]


@torch.no_grad()
def extract_features(encoder: nn.Module, loader: DataLoader, device) -> tuple:
    encoder.eval()
    feats, labels = [], []
    for images, lbls in loader:
        images = images.to(device)
        _, latents = encoder(images)           # [B, 16, 128]
        feats.append(latents.mean(dim=1).cpu())
        labels.append(lbls)
    return torch.cat(feats), torch.cat(labels)


def train_linear_head(
    feats: torch.Tensor, labels: torch.Tensor,
    n_epochs: int = 200, lr: float = 1e-2, device=torch.device("cpu"),
) -> nn.Linear:
    head = nn.Linear(LATENT_DIM, CIFAR_N_CLASSES).to(device)
    opt  = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    loss_fn = nn.CrossEntropyLoss()
    f, l = feats.to(device), labels.to(device)
    for _ in range(n_epochs):
        head.train()
        opt.zero_grad()
        loss_fn(head(f), l).backward()
        opt.step()
        sched.step()
    return head


@torch.no_grad()
def accuracy(head: nn.Linear, feats, labels, device) -> float:
    head.eval()
    preds = head(feats.to(device)).argmax(1).cpu()
    return (preds == labels).float().mean().item()


def build_encoder(cls, ckpt_path: str, key: str | None, device) -> nn.Module | None:
    if not os.path.isfile(ckpt_path):
        return None
    enc = cls()
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if key and isinstance(state, dict) and key in state:
        state = state[key]
    enc.load_state_dict(state)
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Published baselines (CIFAR-10 full linear probe, full labels unless noted)
# ─────────────────────────────────────────────────────────────────────────────

PUBLISHED = {
    "SimCLR (ResNet-50, 23M)":  {"full": 0.906},
    "BYOL   (ResNet-50, 23M)":  {"full": 0.913},
    "MAE    (ViT-B, 86M)":      {"full": 0.860},
    "I-JEPA (ViT-H, 632M)*":    {"full": None},   # no CIFAR-10 result in paper
}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison(n_labels_list: list[int] | None = None):
    if n_labels_list is None:
        n_labels_list = [500, 1000, 5000, -1]

    device = get_device()
    print(f"Device: {device}\n")

    # ── Load encoders ──────────────────────────────────────────────────────────
    def make_cortical():
        return CorticalNetwork(
            patch_size=CIFAR_PATCH_SIZE, n_channels=CIFAR_N_CHANNELS,
            n_classes=CIFAR_N_CLASSES,
        )

    candidates = {
        "Cortical supervised": build_encoder(make_cortical, "cortical_cifar10_supervised.pt", None, device),
        "Cortical JEPA":       build_encoder(make_cortical, "cortical_cifar10_jepa.pt", "context_encoder", device),
        "ViT supervised":      build_encoder(ViTEncoder, "vit_cifar10_supervised.pt", None, device),
        "ViT JEPA (I-JEPA)":  build_encoder(ViTEncoder, "vit_cifar10_jepa.pt", "context_encoder", device),
    }
    encoders = {k: v for k, v in candidates.items() if v is not None}

    if not encoders:
        print("No checkpoints found. Run cifar_train_supervised.py and cifar_train_jepa.py first.")
        return

    print("Available checkpoints:")
    for name in encoders:
        n_params = sum(p.numel() for p in encoders[name].parameters())
        print(f"  {name:<28}  {n_params/1e6:.1f}M params")

    # ── Pre-extract full val features ─────────────────────────────────────────
    val_loader = get_cifar10(512, train=False)
    val_cache  = {}
    for name, enc in encoders.items():
        print(f"Extracting val features [{name}]...")
        val_cache[name] = extract_features(enc, val_loader, device)

    # ── Sweep label counts ────────────────────────────────────────────────────
    results: dict[str, list[float | None]] = {name: [] for name in encoders}
    label_counts = []

    cifar_train_ds = datasets.CIFAR10(
        root="./data", train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
        ])
    )

    # Build header
    col_w = 12
    header_names = list(encoders.keys())
    header = f"{'n_labels':>10}" + "".join(f"  {n[:col_w]:>{col_w}}" for n in header_names)
    print("\n" + header)
    print("─" * len(header))

    for n in n_labels_list:
        actual_n = len(cifar_train_ds) if n == -1 else n
        label_counts.append(actual_n)

        if n == -1:
            train_loader = get_cifar10(512, train=True)
        else:
            idx = _stratified_indices(cifar_train_ds, n)
            from torch.utils.data import Subset
            sub_ds = Subset(cifar_train_ds, idx)
            n_workers = NUM_WORKERS if torch.cuda.is_available() else 0
            train_loader = DataLoader(sub_ds, batch_size=min(512, n), shuffle=False,
                                      num_workers=n_workers)

        n_epochs = 300 if actual_n <= 1000 else 150
        row = f"{actual_n:>10}"

        for name, enc in encoders.items():
            train_feats, train_labels = extract_features(enc, train_loader, device)
            head = train_linear_head(train_feats, train_labels, n_epochs=n_epochs, device=device)
            val_feats, val_labels = val_cache[name]
            acc = accuracy(head, val_feats, val_labels, device)
            results[name].append(acc)
            row += f"  {acc*100:>{col_w}.2f}%"

        print(row)

    # ── Full-label comparison with published results ───────────────────────────
    print("\n" + "═"*70)
    print("Full-label linear probe comparison (CIFAR-10):")
    print("─"*70)
    print(f"  {'Method':<35} {'Params':>8}  {'Val acc':>8}")
    print("─"*70)

    param_counts = {
        name: f"{sum(p.numel() for p in enc.parameters())/1e6:.1f}M"
        for name, enc in encoders.items()
    }

    full_idx = n_labels_list.index(-1) if -1 in n_labels_list else None
    for name in encoders:
        acc_str = f"{results[name][full_idx]*100:.2f}%" if full_idx is not None else "—"
        print(f"  {name:<35} {param_counts[name]:>8}  {acc_str:>8}")

    print("─"*70)
    for name, data in PUBLISHED.items():
        acc_str = f"{data['full']*100:.1f}%" if data["full"] is not None else "N/A*"
        print(f"  {name:<35} {'':>8}  {acc_str:>8}")

    print("─"*70)
    print("* I-JEPA (Assran et al. 2023) reports 87.5% on CIFAR-100, not CIFAR-10.")
    print("  All published results use full training labels.")
    print("═"*70)

    return results, label_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_labels", nargs="+", type=int,
        default=[500, 1000, 5000, -1],
        help="Label counts to sweep. -1 = full training set.",
    )
    args = parser.parse_args()
    run_comparison(args.n_labels)
