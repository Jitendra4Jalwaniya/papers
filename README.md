# Paper Implementations

JAX / Flax reimplementations of classic deep learning papers, each with targeted variations that isolate the effect of specific design choices.

Every implementation trains on a standard benchmark, includes baseline results, and documents what was changed from the original paper and why.

---

## Implementations

### [AlexNet](alexnet/) — Krizhevsky et al., 2012

> *ImageNet Classification with Deep Convolutional Neural Networks*

Trained on CIFAR-10 (32×32). Five notebooks — baseline, sigmoid activation, no dropout, Optuna HP tuning, and joint architecture + HP search.

| Notebook | Variation | Test Acc |
|---|---|---|
| [`alexnet_cifar10_jax`](alexnet/alexnet_cifar10_jax.ipynb) | Baseline (ReLU + Dropout) | 86.49% |
| [`alexnet_cifar10_jax_without_dropout`](alexnet/alexnet_cifar10_jax_without_dropout.ipynb) | No dropout | 86.35% |
| [`alexnet_cifar10_jax_flexible_optuna`](alexnet/alexnet_cifar10_jax_flexible_optuna.ipynb) | Joint arch + HP search | 86.56% |
| [`alexnet_cifar10_jax_optuna`](alexnet/alexnet_cifar10_jax_optuna.ipynb) | Optuna HP tuning | 86.25% |
| [`alexnet_cifar10_jax_with_sigmoid`](alexnet/alexnet_cifar10_jax_with_sigmoid.ipynb) | Sigmoid activation | 82.39% |

See [alexnet/README.md](alexnet/README.md) for architecture details, training setup, and per-variation analysis.

---

### [Transformer](transformer/) — Vaswani et al., 2017

> *Attention Is All You Need*

Trained on WMT14 EN→DE. Full base model (d_model=512, 8 heads, 6 layers) with shared BPE, Noam LR schedule, label-smoothed loss, and beam-search decoding.

| Notebook | Description |
|---|---|
| [`transformer_wmt14_ende_jax`](transformer/transformer_wmt14_ende_jax.ipynb) | Complete implementation — data pipeline, model, training, decoding & BLEU |

See [transformer/README.md](transformer/README.md) for architecture details and training setup.

---

## Stack

All implementations use the same core stack:

| Library | Role |
|---|---|
| JAX | Automatic differentiation, JIT compilation, hardware acceleration |
| Flax Linen | Neural network modules (`nn.Module`) |
| Optax | Optimisers and LR schedules |
| TensorFlow Datasets | Dataset loading and preprocessing |
| Optuna | Hyperparameter and architecture search (selected notebooks) |

---

## Structure

```
papers/
├── alexnet/
│   ├── README.md
│   ├── alexnet_cifar10_jax.ipynb
│   ├── alexnet_cifar10_jax_with_sigmoid.ipynb
│   ├── alexnet_cifar10_jax_without_dropout.ipynb
│   ├── alexnet_cifar10_jax_optuna.ipynb
│   └── alexnet_cifar10_jax_flexible_optuna.ipynb
└── transformer/
    ├── README.md
    └── transformer_wmt14_ende_jax.ipynb
```
