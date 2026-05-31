# Transformer (base) — WMT14 EN→DE, JAX/Flax

A faithful implementation of the Transformer base model from [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017), trained on the WMT14 English→German translation task.

## What's in this directory

| File | Description |
|------|-------------|
| `transformer_wmt14_ende_jax.ipynb` | Self-contained Colab notebook: data pipeline, model, training loop, decoding, and BLEU evaluation |

## Architecture

The notebook implements the paper's base configuration:

- **d_model** = 512, **heads** = 8, **layers** = 6, **d_ff** = 2048
- Sinusoidal positional encoding
- Post-LayerNorm residual connections
- Shared input/output embedding (weight tying)
- Padding mask + causal mask in decoder self-attention

## Training setup

- **Dataset**: WMT14 EN→DE (~4.5M pairs), shared BPE vocab (~32k/37k)
- **Batching**: Length-bucketed, ~25k tokens per effective batch (gradient accumulation)
- **Optimiser**: Adam (β1=0.9, β2=0.98, ε=1e-9) with Noam warmup (4000 steps)
- **Loss**: Label-smoothed cross-entropy (ε=0.1)
- **Decoding**: Greedy and beam search (beam=4, α=0.6)
- **Checkpointing**: safetensors to Google Drive, with checkpoint averaging

## Presets

| Preset | Vocab | Max len | Steps | Use case |
|--------|-------|---------|-------|----------|
| `paper` | 37k | 256 | 100k | Full reproduction (multi-GPU) |
| `colab` | 32k | 128 | 15k | Single-GPU Colab Pro session |

Both presets use the same architecture; `colab` reduces sequence length and step budget to fit within a single session.

## Running

Open the notebook in Google Colab with a GPU runtime. The dataset and tokenizer are cached to Google Drive on first run, so subsequent sessions start quickly.

## Reference

```bibtex
@inproceedings{vaswani2017attention,
  title={Attention is All You Need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, Lukasz and Polosukhin, Illia},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
}
```
