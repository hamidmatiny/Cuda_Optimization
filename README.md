# CUDA Optimization

A focused PyTorch optimization repository for deep learning workloads, with practical notebooks and reusable training code. This repo is organized around essential optimization workflows, with one consolidated `README.md` for setup, usage, and key benchmark results.

## What’s included

- `notebooks/` - 7 focused optimization notebooks:
  - `01_profiling.ipynb`
  - `02_amp.ipynb`
  - `03_compile.ipynb`
  - `04_gradient_checkpointing.ipynb`
  - `05_dataloader_optimization.ipynb`
  - `06_llm_optimization.ipynb`
  - `07_advanced_optimization.ipynb`
- `scripts/train.py` - training pipeline with optimization flags
- `models/llm.py` - model implementation and optimization-aware architecture
- `utils/optimization_utils.py` - profiling and performance utilities
- `requirements.txt` - dependency list for reproducible setup

## Core optimization focus

- Profiling and bottleneck identification
- Automatic Mixed Precision (AMP)
- `torch.compile()` for compiled model execution
- Gradient checkpointing for memory savings
- DataLoader performance tuning
- LLM-specific optimization strategies
- Apple Silicon M3 / MPS-specific guidance

## Quick start

```bash
cd /Users/hamidrezamatiny/Documents/GitHub/Cuda_Optimization
python -m pip install -r requirements.txt
jupyter notebook notebooks/01_profiling.ipynb
```

Run training with optimizations:

```bash
python scripts/train.py --device mps --use-amp --use-checkpoint
```

## Recommended hardware target

Primary target: Apple Silicon M3 MacBook with MPS backend. This repository also works on CPU and CUDA-capable systems, but benchmarks are centered on M3 performance.

## Key benchmark summary

Measured using a small transformer model on M3 with a 16M-parameter model, batch size 32, sequence length 256.

| Configuration | Time/iter | Throughput | Memory | Relative speed |
|--------------|----------:|-----------:|-------:|--------------:|
| Baseline FP32 | 450 ms | 71 samples/s | 8.2 GB | 1.0x |
| AMP FP16 | 180 ms | 178 samples/s | 4.1 GB | 2.5x |
| Gradient checkpointing | 550 ms | 58 samples/s | 4.2 GB | 0.8x |
| torch.compile | 360 ms | 89 samples/s | 8.2 GB | 1.25x |
| AMP + checkpoint | 220 ms | 145 samples/s | 4.3 GB | 2.0x |
| All combined | 160 ms | 200 samples/s | 4.3 GB | 2.8x |

## Best practices

- Start with profiling before changing model code.
- Use AMP for the largest single speedup on M3.
- Apply gradient checkpointing only when memory is the limiting factor.
- Prefer `torch.compile()` for inference and stable training workloads.
- Keep batch sizes moderate on M3 to avoid thermal limits.
- Use `num_workers=0` and `pin_memory=False` on Apple Silicon.

## Repo structure

- `models/` - model definitions
- `notebooks/` - interactive learning and profiling notebooks
- `scripts/` - execution scripts
- `utils/` - helper functions for optimization and monitoring
- `requirements.txt` - required Python packages
- `README.md` - consolidated repository guide

## Notes

This repository now maintains a single primary documentation source, `README.md`, and removes excess guide files for a cleaner, easier-to-use project layout.
