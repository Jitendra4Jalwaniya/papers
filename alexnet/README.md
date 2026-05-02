# AlexNet on CIFAR-10 — JAX / Flax

Reimplementation of AlexNet (Krizhevsky et al., 2012) for CIFAR-10, written in JAX and Flax Linen. Five notebooks explore the baseline and targeted variations.

---

## Architecture

The original architecture is adapted for 32×32 CIFAR-10 input. LRN is replaced with BatchNorm, the two-GPU split is removed, and the input resolution is scaled down accordingly.

```
Input  32×32×3
 │
 ├─ Conv(64,  3×3) → BN → ReLU → MaxPool(3×3/2)  → 15×15×64
 ├─ Conv(192, 3×3) → BN → ReLU → MaxPool(3×3/2)  →  7×7×192
 ├─ Conv(384, 3×3) → BN → ReLU                   →  7×7×384
 ├─ Conv(256, 3×3) → BN → ReLU                   →  7×7×256
 ├─ Conv(256, 3×3) → BN → ReLU → MaxPool(3×3/2)  →  3×3×256
 │
 ├─ Flatten → 2304
 ├─ FC(2048) → ReLU → Dropout(0.5)
 ├─ FC(2048) → ReLU → Dropout(0.5)
 └─ FC(10)   → logits
```

**Parameters:** ~11.2M (standard config)

### What is kept from the paper
- 5-conv-block → 3-FC architecture
- Max-pooling after blocks 1, 2, and 5 with overlapping 3×3 stride-2 windows
- Dropout (p = 0.5) in the FC layers

### What is modernised
| Original | Replacement | Reason |
|---|---|---|
| Local Response Normalisation | Batch Normalisation | LRN has no strong theoretical backing; BN stabilises training and subsumes its effect |
| Two-GPU model split | Single device | No longer a hardware constraint |
| SGD + hand-tuned LR schedule | AdamW + cosine decay | Less manual tuning; equally effective |
| 224×224 ImageNet input | 32×32 CIFAR-10 input | Scaled architecture to match dataset |

---

## Training Setup (shared across all notebooks)

| Setting | Value |
|---|---|
| Dataset | CIFAR-10 (50K train / 10K test) |
| Input size | 32×32×3 |
| Batch size | 128 |
| Epochs | 60 |
| Optimizer | AdamW |
| LR schedule | Linear warmup (5 ep) → cosine decay |
| Peak LR | 1e-3 |
| Weight decay | 1e-4 |
| Gradient clipping | global norm 1.0 |
| Train augmentation | Random 32×32 crop (4-px pad) + random horizontal flip |
| Normalisation | CIFAR-10 channel mean/std |

---

## Notebooks

### 1. [`alexnet_cifar10_jax.ipynb`](alexnet_cifar10_jax.ipynb) — Baseline

The plain reimplementation. Conv blocks use `Conv → BN → ReLU [→ MaxPool]`. FC layers use ReLU activations with dropout (p = 0.5).

**Test accuracy: 86.49%**

Per-class results at final epoch:

| Class | Accuracy |
|---|---|
| airplane | 87.58% |
| automobile | 93.89% |
| bird | 80.64% |
| cat | 72.82% |
| deer | 87.80% |
| dog | 79.58% |
| frog | 90.87% |
| horse | 88.40% |
| ship | 93.39% |
| truck | 89.89% |

---

### 2. [`alexnet_cifar10_jax_with_sigmoid.ipynb`](alexnet_cifar10_jax_with_sigmoid.ipynb) — Sigmoid Activation

Identical to the baseline except all ReLU activations (both in conv blocks and FC layers) are replaced with sigmoid.

**Test accuracy: 82.39%** (~4 pp below baseline)

The gap illustrates why sigmoid fell out of favour for deep networks: vanishing gradients in early layers slow convergence, and the output is not zero-centred, which hampers the optimiser. Training loss also progresses more slowly — reaching 0.011 at epoch 60 vs 0.000 for the baseline.

---

### 3. [`alexnet_cifar10_jax_without_dropout.ipynb`](alexnet_cifar10_jax_without_dropout.ipynb) — No Dropout

Identical to the baseline except the two `Dropout(0.5)` layers in the FC head are removed (commented out). Conv blocks and BN are unchanged.

**Test accuracy: 86.35%** (~0.1 pp below baseline)

Removing dropout has almost no effect here. BN already acts as an implicit regulariser, and weight decay covers the rest. The model reaches 100% training accuracy slightly faster (epoch 55 vs 60) but generalises comparably.

---

### 4. [`alexnet_cifar10_jax_optuna.ipynb`](alexnet_cifar10_jax_optuna.ipynb) — Optuna Hyperparameter Tuning

Adds an Optuna TPE search over training hyperparameters before the full 60-epoch run. The architecture is fixed (identical to the baseline).

**Search space:**

| HP | Range |
|---|---|
| `base_lr` | [1e-4, 1e-2] log-uniform |
| `weight_decay` | [1e-5, 1e-3] log-uniform |
| `dropout_rate` | [0.2, 0.6] uniform |
| `warmup_epochs` | {2, 3, 5, 8} categorical |

**Search budget:** 15 trials × 15 epochs each, with `MedianPruner` (n_startup=3, n_warmup=3). 12 of 15 trials were pruned. Search completed in ~20 min.

**Best config found:**

| HP | Value |
|---|---|
| `base_lr` | 6.07e-3 |
| `weight_decay` | 8.46e-4 |
| `dropout_rate` | 0.35 |
| `warmup_epochs` | 8 |

**Test accuracy (full run with best HPs): 86.25%**

The tuned config does not exceed the hand-picked baseline, suggesting the default HPs were already near-optimal for this fixed architecture.

---

### 5. [`alexnet_cifar10_jax_flexible_optuna.ipynb`](alexnet_cifar10_jax_flexible_optuna.ipynb) — Joint Architecture + HP Tuning

Introduces `FlexibleAlexNet`, a fully parametric variant where conv filter counts, kernel sizes, FC head depth and width are all tunable. Optuna searches architecture and training HPs jointly.

**New architectural dimensions:**

| HP | Choices | Controls |
|---|---|---|
| `conv_width` | narrow / standard / wide | All 5 conv filter counts via preset |
| `kernel_size` | 3, 5 | Receptive field size (uniform across blocks) |
| `num_dense_layers` | 1 – 3 | FC head depth |
| `dense_width` | 512, 1024, 2048 | Width of every FC layer |

**Conv width presets:**

| Preset | Filters (F0–F4) | Approx. params (2-FC-1024 head) |
|---|---|---|
| narrow | 32, 96, 192, 128, 128 | ~2.5M |
| standard | 64, 192, 384, 256, 256 | ~8.5M |
| wide | 96, 256, 512, 384, 384 | ~18M |

A parameter budget guard (20M cap) prunes architecturally large trials before any training cost is incurred. 12 of 15 trials were pruned (by budget guard or MedianPruner). Search completed in ~40 min.

**Best config found:**

| HP | Value |
|---|---|
| `conv_width` | standard |
| `kernel_size` | 3×3 |
| `num_dense_layers` | 2 |
| `dense_width` | 512 |
| `base_lr` | 7.48e-4 |
| `weight_decay` | 2.48e-4 |
| `dropout_rate` | 0.224 |
| `warmup_epochs` | 3 |
| Total params | 3.70M (~3× smaller than baseline) |

**Test accuracy (full run): 86.56%**

The search found that a 3× smaller model (3.7M vs 11.2M params) matches the baseline accuracy, pointing toward the baseline being over-parameterised for CIFAR-10.

---

## Results Summary

| Notebook | Variation | Test Accuracy |
|---|---|---|
| Baseline | ReLU + Dropout | **86.49%** |
| Without Dropout | ReLU, no Dropout | 86.35% |
| Flexible + Optuna | Joint arch + HP search (3.7M params) | 86.56% |
| Optuna HP tuning | Fixed arch, tuned HPs | 86.25% |
| Sigmoid | Sigmoid instead of ReLU | 82.39% |

---

## Key Observations

- **Sigmoid vs ReLU:** ~4 pp accuracy gap. Vanishing gradients visibly slow early training (epoch-1 test accuracy: 10% sigmoid vs 45% ReLU).
- **Dropout on CIFAR-10:** Negligible effect when BN and weight decay are present. The regularisation budget is already met without it.
- **Optuna HP tuning:** Default HPs were already near-optimal for the fixed architecture; tuning recovered equivalent performance but not higher.
- **Joint arch search:** Reveals that the standard AlexNet head (2×2048 FC) is over-sized for CIFAR-10. A 2×512 head achieves the same accuracy at 3× fewer parameters.

---

## Dependencies

```
jax
flax
optax
tensorflow-datasets
optuna          # notebooks 4 and 5 only
matplotlib
numpy
```
