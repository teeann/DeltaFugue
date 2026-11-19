<h1 align="center">DeltaFugue: Orchestrating Spatial and Associative Memory for Algorithmic Length Generalization</h1>

<p align="center">
  <a href="https://teeann.github.io/">Anh T Nguyen</a>,
  <a href="https://quandao10.github.io/">Quan Dao</a>,
  <a href="https://www.cs.uic.edu/~liub/">Bing Liu</a>
</p>

<h3 align="center">NeurIPS 2026 (Poster)</h3>

<p align="center">
  <a href="https://openreview.net/forum?id=twqMDZo9zz">[Paper]</a>
  <!-- <a href="[URL_TO_PROJECT_PAGE]">[Project Page]</a> -->
</p>

## Abstract

Linear attention models offer a highly efficient alternative to Transformers, leveraging fixed-size memory for robust state-tracking and retrieval. However, relying strictly on fixed-capacity associative states bottlenecks their ability to resolve precise structural dependencies that may grow unboundedly in multi-step algorithmic reasoning - especially within deterministic context-free and context-sensitive formal languages. Inspired by the decoupled computation of Turing Machines, we introduce DeltaFugue: a hardware-aware architecture that orchestrates a continuous linear attention controller alongside a differentiable 1D spatial tape. Unlike traditional memory-augmented RNNs that sacrifice sequence-level parallelism, DeltaFugue executes read and write operations natively via delta-rule updates, preserving the training parallelism of linear transformers. Theoretically, we prove that this decoupled spatial routing allows DeltaFugue to recognize regular, hierarchical, and context-sensitive formal languages. Empirically, extensive length generalization evaluations demonstrate that DeltaFugue achieves state-of-the-art accuracy among fully parallelizable models. Scaling to natural language, a 340M-parameter DeltaFugue model matches the performance of strong DeltaNet baselines while exhibiting markedly superior length extrapolation in reasoning-heavy domains like mathematics and code generation.

## Overview

DeltaFugue augments the associative memory of DeltaNet with a bounded 1D **tape**:
a second delta-rule memory with `tape_size` slots, addressed by independent
**read** and **write** pointers. Each pointer moves by a soft right / left / stay
action per token; its position is a cumulative sum of those actions, decoded into
a soft address over the slots by a Laplacian kernel,
`softmax(-softplus(pointer_gamma) * |slot - position|)`. The associative memory
then stores the tape's prediction error, `v - tape_read_{t-1}`.

Every part is parallel over the sequence: the pointers are prefix sums, and both
memories use the chunked delta-rule kernels from
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention).
**GatedDeltaFugue** pairs the same tape with Gated DeltaNet's forget-gated
associative memory.

## Installation

Requires a CUDA GPU (the kernels are written in Triton). Reference environment:
Python 3.12, PyTorch 2.9, CUDA 12.6.

```bash
pip install -r requirements.txt
pip install -e fla-latest          # vendored, trimmed flash-linear-attention with DeltaFugue
```

or `just install`. The `mamba` baseline additionally needs `mamba_ssm`.

## Formal-language experiments

Configs are JSON files in `experiment_configs/`, passed by name:

```bash
python run_experiment.py -c lab            # single GPU (`just lab`)
python run_experiment.py -c lab --multi    # 4 seeds across 4 GPUs
```

Models train on lengths 3-60 and are evaluated on 60-256 (length generalization).
Key fields: `task` (any module in `data_dir/fl_tasks/`), `model_name`
(`deltafugue`, `gated_deltafugue`, or a baseline such as `deltanet`,
`gateddeltanet`, `transformer_causal`, `lstm`), and `tape_size` /
`val_tape_size` (tape slots for training / evaluation). Results are written to
`results/`; Weights & Biases logging is used if `WANDB_API_KEY` is set.

## Language-model pretraining

After `just install_flame`, train a 340M model on fineweb-edu with 4 GPUs, then
convert and evaluate it (`lm_eval` needs `pip install lm-eval`):

```bash
just config=configs/delta_fugue_340M_ts16.json dump_folder=exp/df flame_train
just convert exp/df
just lm_eval <hf_checkpoint>
```

Configs for DeltaFugue, GatedDeltaFugue and the baselines are in `flame/configs/`.

## Repository layout

| Path | Contents |
| --- | --- |
| `fla-latest/fla/layers/delta_fugue.py`, `gated_deltafugue.py` | The DeltaFugue / GatedDeltaFugue layers |
| `fla-latest/fla/ops/deltafugue/` | Fused Triton kernel for the delayed tape read (`fused_delay_shift`) |
| `fla-latest/fla/models/delta_fugue/`, `gated_deltafugue/` | Hugging Face causal-LM wrappers |
| `fla-latest/tests/models/` | Forward/backward and cached-generation tests (`just test`) |
| `models/` | Formal-language model wrappers (`fla.py`) and RNN / Transformer baselines |
| `data_dir/` | Task generators and dataloaders |
| `train.py`, `run_experiment.py` | Formal-language training and evaluation |
| `flame/` | LM pretraining |

`fla-latest/` is a trimmed copy of flash-linear-attention (MIT): only the layers,
models and kernels used here are kept.

## Citation

```bibtex
@inproceedings{
nguyen2026deltafugue,
title={DeltaFugue: Orchestrating Spatial and Associative Memory for Algorithmic Length Generalization},
author={Anh T Nguyen and Quan Dao and Bing Liu},
booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
year={2026},
url={https://openreview.net/forum?id=twqMDZo9zz}
}
```
