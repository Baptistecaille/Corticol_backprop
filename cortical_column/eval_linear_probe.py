"""Linear probe evaluation: JEPA encoder vs supervised encoder.

Freezes the column encoder (backbone), trains only a fresh linear head
on top of the mean-pooled column latents. Compares:

    (A) Supervised encoder   : cortical_network_mnist.pt
    (B) JEPA encoder         : cortical_network_jepa.pt  (context_encoder key)

A good JEPA encoder should match or beat the supervised one, especially
in the low-label regime (few training samples).

Usage
─────
    python -m cortical_column.eval_linear_probe
    python -m cortical_column.eval_linear_probe --n_labels 100 500 1000 5000 -1

Output
──────
    eval_linear_probe.png  — accuracy vs. n_labels curve for both encoders
"""

import argparse
import os
import sys
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cortical_column.config import LATENT_DIM, NUM_WORKERS
from cortical_column.cortical_network import CorticalNetwork


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_mnist(train: bool, n_samples: int | None = None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    ds = datasets.MNIST(root="./data", train=train, download=True, transform=transform)
    if n_samples is not None and n_samples > 0 and n_samples < len(ds):
        # Stratified subset: equal samples per class
        indices = _stratified_indices(ds, n_samples)
        ds = Subset(ds, indices)
    return ds


def _stratified_indices(dataset, n_total: int) -> list[int]:
    """Return indices with ~equal representation per class."""
    n_classes = 10
    per_class = n_total // n_classes
    class_buckets: dict[int, list[int]] = {c: [] for c in range(n_classes)}
    for idx, (_, label) in enumerate(dataset):
        if len(class_buckets[label]) < per_class:
            class_buckets[label].append(idx)
        if all(len(v) >= per_class for v in class_buckets.values()):
            break
    indices = [idx for bucket in class_buckets.values() for idx in bucket]
    return indices


@torch.no_grad()
def extract_features(
    encoder: CorticalNetwork,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Forward all images through the frozen encoder, mean-pool the 16 column
    latents, return (features [N, 128], labels [N]).
    """
    encoder.eval()
    all_feats, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        _, latents = encoder(images)          # latents: [B, 16, 128]
        feats = latents.mean(dim=1).cpu()     # [B, 128]
        all_feats.append(feats)
        all_labels.append(labels)
    return torch.cat(all_feats), torch.cat(all_labels)


def train_linear_head(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    n_epochs: int = 100,
    lr: float = 1e-2,
    device: torch.device = torch.device("cpu"),
) -> nn.Linear:
    """Train a single Linear(128 → 10) on pre-extracted features."""
    head = nn.Linear(LATENT_DIM, 10).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    loss_fn = nn.CrossEntropyLoss()

    feats  = train_feats.to(device)
    labels = train_labels.to(device)

    for _ in range(n_epochs):
        head.train()
        logits = head(feats)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    return head


@torch.no_grad()
def evaluate(
    head: nn.Linear,
    feats: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> float:
    head.eval()
    logits = head(feats.to(device))
    preds = logits.argmax(dim=1).cpu()
    return (preds == labels).float().mean().item()


def load_encoder(ckpt_path: str, key: str | None, device: torch.device) -> CorticalNetwork:
    """Load a CorticalNetwork from a checkpoint file."""
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if key is not None and isinstance(state, dict) and key in state:
        state = state[key]
    encoder = CorticalNetwork().to(device)
    encoder.load_state_dict(state)
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    return encoder


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_probe(
    supervised_ckpt: str = "cortical_network_mnist.pt",
    jepa_ckpt: str = "cortical_network_jepa.pt",
    n_labels_list: list[int] | None = None,
    output_plot: str = "eval_linear_probe.png",
):
    """
    Run linear probe for both encoders across label counts.

    Args:
        n_labels_list : list of training set sizes to evaluate.
                        -1 means full training set (60,000).
    """
    if n_labels_list is None:
        n_labels_list = [100, 500, 1000, 5000, 10000, -1]

    device = get_device()
    n_workers = NUM_WORKERS if torch.cuda.is_available() else 0
    print(f"Device : {device}\n")

    # ── Load encoders ─────────────────────────────────────────────────────────
    encoders = {}
    if os.path.isfile(supervised_ckpt):
        encoders["supervised"] = load_encoder(supervised_ckpt, key=None, device=device)
        print(f"Loaded supervised  : {supervised_ckpt}")
    else:
        print(f"[WARN] supervised checkpoint not found: {supervised_ckpt}")

    if os.path.isfile(jepa_ckpt):
        encoders["JEPA"] = load_encoder(jepa_ckpt, key="context_encoder", device=device)
        print(f"Loaded JEPA        : {jepa_ckpt}")
    else:
        print(f"[WARN] JEPA checkpoint not found: {jepa_ckpt}")

    if not encoders:
        raise FileNotFoundError("No checkpoints found. Run training first.")

    # ── Pre-extract full val features (shared across all experiments) ─────────
    val_loader = DataLoader(
        get_mnist(train=False),
        batch_size=1024,
        shuffle=False,
        num_workers=n_workers,
    )

    val_feats_cache: dict[str, tuple] = {}
    for name, enc in encoders.items():
        print(f"Extracting val features [{name}]...")
        val_feats_cache[name] = extract_features(enc, val_loader, device)

    # ── Linear probe sweep ────────────────────────────────────────────────────
    results: dict[str, list[float]] = {name: [] for name in encoders}
    label_counts: list[int] = []

    print()
    header = f"{'n_labels':>10}" + "".join(f"  {name:>12}" for name in encoders)
    print(header)
    print("─" * len(header))

    for n in n_labels_list:
        actual_n = 60_000 if n == -1 else n
        label_counts.append(actual_n)

        train_ds = get_mnist(train=True, n_samples=None if n == -1 else n)
        train_loader = DataLoader(
            train_ds,
            batch_size=min(1024, actual_n),
            shuffle=False,
            num_workers=n_workers,
        )

        row = f"{actual_n:>10}"
        for name, enc in encoders.items():
            # Extract train features for this label count
            train_feats, train_labels = extract_features(enc, train_loader, device)

            # Train linear head
            n_epochs = 200 if actual_n <= 1000 else 100
            head = train_linear_head(
                train_feats, train_labels,
                n_epochs=n_epochs, lr=1e-2, device=device,
            )

            # Evaluate on full val set
            val_feats, val_labels = val_feats_cache[name]
            acc = evaluate(head, val_feats, val_labels, device)
            results[name].append(acc)
            row += f"  {acc * 100:>11.2f}%"

        print(row)

    # ── Plot ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = {"supervised": "#2196F3", "JEPA": "#FF5722"}
        markers = {"supervised": "o", "JEPA": "s"}

        for name, accs in results.items():
            ax.plot(
                label_counts, [a * 100 for a in accs],
                marker=markers.get(name, "o"),
                linewidth=2,
                markersize=7,
                label=name,
                color=colors.get(name, None),
            )

        ax.set_xscale("log")
        ax.set_xlabel("Number of labelled training samples", fontsize=12)
        ax.set_ylabel("Val accuracy (%)", fontsize=12)
        ax.set_title("Linear probe: JEPA encoder vs supervised encoder", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(label_counts)
        ax.set_xticklabels([str(n) for n in label_counts], rotation=30)
        plt.tight_layout()
        plt.savefig(output_plot, dpi=150)
        print(f"\nPlot saved → {output_plot}")
    except ImportError:
        print("\n[INFO] matplotlib not installed — skipping plot")

    return results, label_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear probe: JEPA vs supervised encoder")
    parser.add_argument("--supervised", default="cortical_network_mnist.pt")
    parser.add_argument("--jepa",       default="cortical_network_jepa.pt")
    parser.add_argument(
        "--n_labels", nargs="+", type=int,
        default=[100, 500, 1000, 5000, 10000, -1],
        help="Training set sizes to sweep. Use -1 for full 60k.",
    )
    parser.add_argument("--output", default="eval_linear_probe.png")
    args = parser.parse_args()

    run_probe(
        supervised_ckpt=args.supervised,
        jepa_ckpt=args.jepa,
        n_labels_list=args.n_labels,
        output_plot=args.output,
    )
