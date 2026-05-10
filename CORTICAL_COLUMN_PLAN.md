# Plan d'implémentation : Cortical Column Network pour MNIST → JEPA

## Contexte

Modèle computationnel d'une colonne corticale biologiquement inspirée.
- **Apprentissage** : Backpropagation standard (PyTorch)
- **Benchmark** : MNIST (28×28, 10 classes)
- **Objectif final** : remplacer le prédicteur dans une architecture I-JEPA
- **Matériel** : Apple M1 → backend MPS (`torch.backends.mps`)

---

## Architecture globale

```
Image MNIST 28×28
    ↓  découpage en 16 patches de 7×7 pixels (grille 4×4)
[CorticalColumn_0 ... CorticalColumn_15]   ← 16 colonnes en parallèle
    ↓  chaque colonne produit un latent ∈ ℝ¹²⁸
Mean pooling sur les 16 latents → ℝ¹²⁸
    ↓
Linear(128 → 10) + softmax  →  classe MNIST

# Futur JEPA :
Context columns → latents → CorticalPredictor → latent prédit
Target column   → latent réel
Loss = cosine similarity loss dans l'espace latent
```

---

## Structure des fichiers

```
cortical_column/
├── __init__.py
├── minicolumn.py         # Classe MiniColumn
├── cortical_layers.py    # Classes L1Layer … L6Layer
├── cortical_column.py    # Classe CorticalColumn (assemble L1-L6)
├── cortical_network.py   # Classe CorticalNetwork (16 colonnes + vote)
├── train.py              # Boucle d'entraînement MNIST
└── config.py             # Hyperparamètres centralisés
```

---

## Hyperparamètres (config.py)

```python
PATCH_SIZE     = 7       # pixels, patch carré
N_PATCHES      = 16      # 4×4 grille sur image 28×28
N_MINICOLUMNS  = 16      # mini-colonnes par colonne corticale
LATENT_DIM     = 128     # dimension du vecteur latent (sortie L5)
SPARSITY_K     = 4       # K-WTA : top-k activations gardées par mini-colonne
N_CLASSES      = 10      # MNIST

# Dimensions des couches internes
L4_DIM   = 64   # couche d'entrée
L23_DIM  = 128  # fusion bottom-up / top-down
L5_DIM   = 128  # = LATENT_DIM, sortie principale
L6_DIM   = 64   # attention / gating
L1_DIM   = 32   # bus signal d'erreur top-down

BATCH_SIZE = 64
LR         = 1e-3
EPOCHS     = 20
DEVICE     = "mps"   # fallback "cpu" si MPS non dispo
```

---

## 1. MiniColumn (`minicolumn.py`)

**Rôle biologique** : unité élémentaire répétée ~80 fois dans une colonne. Chaque mini-colonne traite le patch d'entrée avec son propre filtre et génère une représentation sparse.

**Implémentation** :
```python
class MiniColumn(nn.Module):
    """
    Traite un vecteur d'entrée (patch aplati) et produit
    une représentation sparse via K-Winner-Take-All (K-WTA).

    Args:
        input_dim  : taille du vecteur d'entrée (ex: 49 pour patch 7×7)
        hidden_dim : dimension de la représentation interne
        k          : nombre d'activations gardées (sparsité)
    """
    def __init__(self, input_dim: int, hidden_dim: int, k: int):
        ...

    def kwta(self, x: Tensor) -> Tensor:
        """
        K-Winner-Take-All : garde les k valeurs les plus élevées,
        met les autres à 0. Différentiable via straight-through estimator.
        """
        ...

    def forward(self, x: Tensor) -> Tensor:
        # Linear → BatchNorm → ReLU → K-WTA
        ...
```

**Notes** :
- Le K-WTA doit être différentiable : utiliser un straight-through estimator
  (gradient passe comme si toutes les unités étaient actives en backward)
- BachNorm avant K-WTA pour stabiliser l'entraînement

---

## 2. Couches corticales L1-L6 (`cortical_layers.py`)

Chaque couche est un `nn.Module` avec un rôle algorithmique précis.
Les couches reçoivent/émettent des tenseurs de shape `(batch, dim)`.

### L4Layer — Entrée normalisée (thalamus)
```python
class L4Layer(nn.Module):
    """
    Couche d'entrée primaire. Reçoit le patch aplati (49 dims),
    projette vers L4_DIM via N_MINICOLUMNS mini-colonnes en parallèle,
    agrège leurs sorties sparse.

    forward(patch: Tensor[B, 49]) -> Tensor[B, L4_DIM]
    """
```
- Instancie `N_MINICOLUMNS` objets `MiniColumn(input_dim=49, hidden_dim=L4_DIM//N_MINICOLUMNS, k=SPARSITY_K)`
- Concatène leurs sorties → projection linéaire → `L4_DIM`

### L23Layer — Fusion bottom-up / top-down
```python
class L23Layer(nn.Module):
    """
    Reçoit la sortie de L4 (bottom-up) ET un signal top-down (depuis L1).
    Calcule le résidu : différence entre prédiction top-down et réalité bottom-up.
    Propagation latérale simulée par une couche linéaire supplémentaire.

    forward(
        x_bottom_up: Tensor[B, L4_DIM],
        x_top_down:  Tensor[B, L1_DIM]   (peut être zeros au 1er forward)
    ) -> Tensor[B, L23_DIM]
    """
```
- Deux projections séparées (bottom-up et top-down) → concaténation → Linear → LayerNorm → ReLU
- Connexion latérale optionnelle (Linear L23_DIM → L23_DIM)

### L5Layer — Intégration + espace latent
```python
class L5Layer(nn.Module):
    """
    Intégration non-linéaire des signaux bottom-up (L2/3) et top-down (L6).
    Produit le vecteur latent principal de la colonne.

    forward(
        x_l23:      Tensor[B, L23_DIM],
        x_feedback: Tensor[B, L6_DIM]   (peut être zeros)
    ) -> Tensor[B, L5_DIM]   # = LATENT_DIM = 128
    """
```
- Architecture : Linear(L23_DIM + L6_DIM → L5_DIM) → LayerNorm → GELU
- C'est la sortie principale utilisée pour le vote et pour JEPA

### L6Layer — Attention / Gating
```python
class L6Layer(nn.Module):
    """
    Régulation du gain cortical. Reçoit L5 et produit un signal
    de modulation renvoyé vers L5 et vers le thalamus (L4 future step).

    forward(x_l5: Tensor[B, L5_DIM]) -> Tensor[B, L6_DIM]
    """
```
- Linear(L5_DIM → L6_DIM) → Sigmoid (gate multiplicatif)

### L1Layer — Bus d'erreur top-down
```python
class L1Layer(nn.Module):
    """
    Transporte le signal d'erreur depuis la sortie du réseau
    vers les couches superficielles (L2/3). Modélise les dendrites
    apicales des neurones pyramidaux.

    forward(error_signal: Tensor[B, N_CLASSES or LATENT_DIM]) -> Tensor[B, L1_DIM]
    """
```
- Linear → ReLU → projection vers `L1_DIM`
- En phase MNIST : le signal d'entrée est le gradient de la loss (détaché `.detach()`)
- En phase JEPA : le signal d'entrée sera l'erreur de prédiction latente

---

## 3. CorticalColumn (`cortical_column.py`)

```python
class CorticalColumn(nn.Module):
    """
    Assemble les 6 couches dans l'ordre de traitement biologique.
    Reçoit un patch 7×7 aplati, produit un vecteur latent ∈ ℝ¹²⁸.

    Flux d'information :
        patch → L4 → L2/3(+L1 feedback) → L5(+L6 feedback) → L6 → latent

    Args:
        patch_dim  : dimension du patch aplati (49)
        latent_dim : dimension du vecteur latent de sortie (128)
    """

    def __init__(self, patch_dim: int, latent_dim: int):
        # instancier L1Layer, L4Layer, L23Layer, L5Layer, L6Layer
        ...

    def forward(
        self,
        patch: Tensor,                      # [B, 49]
        top_down_signal: Tensor | None       # [B, L1_DIM] ou None
    ) -> tuple[Tensor, Tensor]:
        """
        Returns:
            latent    : Tensor[B, 128]  — représentation principale (L5)
            l6_signal : Tensor[B, 64]   — signal de feedback L6
        """
        # 1. L4  : entrée normalisée
        x_l4  = self.l4(patch)

        # 2. L1  : signal top-down (zeros si None)
        x_l1  = self.l1(top_down_signal) if top_down_signal is not None \
                else torch.zeros(B, L1_DIM)

        # 3. L2/3 : fusion
        x_l23 = self.l23(x_l4, x_l1)

        # 4. L5  : intégration + latent
        latent = self.l5(x_l23, l6_prev)   # l6_prev = zeros au 1er appel

        # 5. L6  : gating
        l6_out = self.l6(latent)

        return latent, l6_out
```

---

## 4. CorticalNetwork (`cortical_network.py`)

```python
class CorticalNetwork(nn.Module):
    """
    Orchestre 16 colonnes corticales sur les 16 patches d'une image MNIST.
    Agrège les latents par mean pooling et classifie via une tête linéaire.

    Futur : supprimer la tête de classification et utiliser les latents
    directement comme encodeur pour I-JEPA.

    Args:
        n_columns  : 16
        patch_size : 7
        latent_dim : 128
        n_classes  : 10
    """

    def __init__(self, n_columns, patch_size, latent_dim, n_classes):
        # n_columns instances de CorticalColumn (partagent-elles les poids ?)
        # → NON : chaque colonne est spécialisée sur sa région spatiale
        self.columns = nn.ModuleList([
            CorticalColumn(patch_dim=patch_size**2, latent_dim=latent_dim)
            for _ in range(n_columns)
        ])
        self.classifier = nn.Linear(latent_dim, n_classes)

    def extract_patches(self, images: Tensor) -> Tensor:
        """
        Découpe les images en 16 patches de 7×7.
        Input  : [B, 1, 28, 28]
        Output : [B, 16, 49]
        """
        # utiliser unfold ou einops.rearrange
        ...

    def voting(self, latents: Tensor) -> Tensor:
        """
        Agrège les 16 latents de colonnes en un seul vecteur.
        Stratégie : mean pooling (soft majority vote sur l'espace latent).

        Input  : [B, 16, 128]
        Output : [B, 128]
        """
        return latents.mean(dim=1)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """
        Returns:
            logits  : Tensor[B, 10]    — pour la loss MNIST
            latents : Tensor[B, 16, 128] — pour JEPA plus tard
        """
        patches = self.extract_patches(images)          # [B, 16, 49]
        latents = []
        for i, col in enumerate(self.columns):
            lat, _ = col(patches[:, i, :], top_down_signal=None)
            latents.append(lat)
        latents = torch.stack(latents, dim=1)           # [B, 16, 128]
        pooled  = self.voting(latents)                   # [B, 128]
        logits  = self.classifier(pooled)               # [B, 10]
        return logits, latents
```

---

## 5. Boucle d'entraînement (`train.py`)

```python
"""
Entraînement du CorticalNetwork sur MNIST.
Device : MPS (Apple M1) avec fallback CPU.
"""

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def train():
    device = get_device()
    model  = CorticalNetwork(...).to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss()

    # DataLoader MNIST standard (torchvision)
    # Normalisation : mean=0.1307, std=0.3081

    for epoch in range(EPOCHS):
        # train loop → logits, latents = model(images)
        # loss = loss_fn(logits, labels)
        # backward + step

        # validation loop
        # log : loss, accuracy, sparsité moyenne des mini-colonnes

    # Sauvegarder : model.state_dict() + latents de l'ensemble de test
    # (utile pour visualiser l'espace latent avec UMAP/t-SNE)
```

---

## 6. Tests à implémenter

1. **Sanity check forward pass** : une image aléatoire passe sans erreur de shape
2. **Sparsité K-WTA** : vérifier que exactement `k` neurones sont actifs par mini-colonne
3. **Gradient flow** : vérifier que toutes les couches reçoivent des gradients non-nuls
4. **MNIST accuracy** : objectif ≥ 95% à 20 epochs
5. **Espace latent** : visualisation UMAP des 16×128 latents colorés par classe

---

## 7. Métriques de suivi (à logger avec wandb ou tensorboard)

- `train/loss`, `val/loss`
- `train/accuracy`, `val/accuracy`
- `sparsity/mean_active_units` — ratio d'unités actives par mini-colonne
- `latent/inter_column_similarity` — similarité cosine entre colonnes (doit diminuer = spécialisation)
- `latent/inter_class_distance` — distance entre latents de classes différentes (doit augmenter)

---

## 8. Roadmap vers JEPA

Une fois MNIST ≥ 95% :

1. **Retirer la tête de classification** → `CorticalNetwork` devient un encodeur pur
2. **Ajouter un `CorticalPredictor`** : prend un sous-ensemble de latents (context columns) + masque de position → prédit les latents des target columns
3. **Loss JEPA** : cosine similarity loss dans l'espace latent (pas de reconstruction pixel)
4. **Masking strategy** : masquer 40-70% des colonnes (target), garder le reste (context)
5. **Pré-entraînement self-supervisé** sur MNIST ou CIFAR-10

```
# Architecture JEPA finale
Context patches → [CorticalColumn × k] → context latents
                                              ↓
                                    CorticalPredictor (ton modèle)
                                              ↓
                                    predicted target latents
                                              ↑ cosine loss
Target patches  → [CorticalColumn × (16-k)] → target latents (stop gradient)
```

---

## Notes importantes pour l'implémentation

- **MPS et in-place operations** : éviter les opérations in-place (`x += y` → `x = x + y`) car elles causent des erreurs de gradient sur MPS
- **LayerNorm vs BatchNorm** : préférer `LayerNorm` pour les petits batches et la stabilité sur MPS
- **Straight-through estimator pour K-WTA** :
  ```python
  # Forward : sparse, Backward : dense (gradient passe entier)
  output = kwta_forward(x)
  output = x + (output - x).detach()
  ```
- **Initialisation** : utiliser `nn.init.kaiming_normal_` pour les couches Linear (compatible ReLU/GELU)
- **Shape convention** : toujours `(batch, features)` — pas de dimensions spatiales résiduelles après `extract_patches`
