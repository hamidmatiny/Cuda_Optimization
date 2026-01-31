# CUDA Optimization Course

Complete guide to optimizing deep learning models with PyTorch. Adapted for M3 MacBook with Metal Performance Shaders (MPS).

## Course Topics

### 1. **Profiling** (`01_profiling.ipynb`)
- PyTorch profiler setup
- Identifying bottlenecks
- Memory usage tracking
- Device utilization analysis

### 2. **Automatic Mixed Precision (AMP)** (`02_amp.ipynb`)
- FP16/BF16 basics
- ScalerGradScaler usage
- Performance improvements
- Numerical stability

### 3. **Model Compilation** (`03_compile.ipynb`)
- torch.compile() optimization
- Compilation modes (eager, reduce-overhead, max-autotune)
- Performance benchmarks
- M3 MacBook considerations

### 4. **Gradient Checkpointing** (`04_gradient_checkpointing.ipynb`)
- Memory-compute tradeoff
- torch.utils.checkpoint
- Activation checkpointing strategies
- Large model training

### 5. **DataLoader Optimization** (`05_dataloader_optimization.ipynb`)
- pin_memory for faster transfers
- num_workers tuning
- Prefetching strategies
- Batch size optimization

### 6. **LLM Model Optimization** (`06_llm_optimization.ipynb`)
- Small transformer implementation
- Layer-wise bottleneck identification
- Multi-technique optimization
- Before/after comparison

### 7. **Advanced Techniques** (`07_advanced_optimization.ipynb`)
- Combined optimization strategies
- Capacity utilization
- CPU-GPU coordination
- Real-world benchmarks

## Setup

```bash
pip install torch torchvision torchaudio
pip install numpy matplotlib pandas
pip install transformers datasets
```

## Hardware Detected

- **Device**: Apple Silicon (M3)
- **Compute Unit**: Metal Performance Shaders (MPS)
- **Fallback**: CPU
- **Architecture**: arm64

## Key Concepts for M3 MacBook

1. **MPS Backend**: PyTorch's Metal backend for Apple Silicon
2. **Unified Memory**: Shared memory architecture vs discrete GPUs
3. **Memory Bandwidth**: CPU-GPU transfers are faster than discrete setups
4. **Thermal Management**: Important for sustained performance
5. **Energy Efficiency**: Focus on throughput per watt

## Best Practices

- Use AMP to reduce memory and improve speed
- Gradient checkpointing for larger models
- Optimize batch size for M3 memory (8-16GB typical)
- Monitor device utilization with profiler
- Test on CPU baseline for comparison
- Use torch.compile() for inference optimization

## Running Notebooks

```bash
jupyter notebook notebooks/01_profiling.ipynb
```

## Performance Tips for M3

1. Keep batch sizes moderate (32-256)
2. Use pin_memory=False (not applicable on Mac)
3. Compile models when possible
4. Use AMP for mixed precision training
5. Profile before and after optimizations
6. Monitor thermal throttling

---

For best results, run optimizations in sequence starting with profiling.
