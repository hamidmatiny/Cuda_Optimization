#!/usr/bin/env python3
"""
CUDA Optimization Course - Course Index and Navigation

This script provides an overview of the course and links to all modules.
"""

COURSE_STRUCTURE = """
╔════════════════════════════════════════════════════════════════════════════════╗
║                    CUDA Optimization Course - Complete Index                   ║
║                  Optimizing Deep Learning for All Hardware                      ║
╚════════════════════════════════════════════════════════════════════════════════╝

📚 COURSE OVERVIEW
═══════════════════════════════════════════════════════════════════════════════

This 7-module course teaches you how to optimize deep learning models for:
  ✓ NVIDIA GPUs (CUDA)
  ✓ Apple Silicon (M3 MacBook with MPS)
  ✓ CPU-only training
  
Expected speedups: 2-8x faster training with minimal code changes!

═══════════════════════════════════════════════════════════════════════════════
📖 MODULE GUIDE
═══════════════════════════════════════════════════════════════════════════════

MODULE 1: PROFILING
─────────────────────────────────────────────────────────────────────────────
📝 File: notebooks/01_profiling.ipynb
⏱️  Duration: 1.5 hours
🎯 Goal: Learn to identify bottlenecks in your models

Topics:
  • PyTorch profiler setup and usage
  • Measuring forward/backward pass time
  • Memory allocation tracking
  • Layer-wise performance analysis
  • Baseline performance measurement

Key Concepts:
  → Profiling is the first step before optimization
  → Identify where your model spends time
  → Measure memory usage patterns
  → Create performance baselines

Expected Learning Outcome:
  ✓ Can profile any PyTorch model
  ✓ Understand model performance characteristics
  ✓ Identify optimization targets


MODULE 2: AUTOMATIC MIXED PRECISION (AMP)
─────────────────────────────────────────────────────────────────────────────
📝 File: notebooks/02_amp.ipynb
⏱️  Duration: 2 hours
🎯 Goal: Implement FP16 training for 2.5x speedup

Topics:
  • FP16 vs FP32 vs BF16 precision
  • torch.amp.autocast for automatic precision selection
  • GradScaler for loss scaling
  • Numerical stability tricks
  • AMP best practices

Key Concepts:
  → Use lower precision (FP16) for compute
  → Keep higher precision (FP32) for sensitive operations
  → Automatic loss scaling prevents gradient underflow
  → ~50% memory savings with 2.5x speedup

Expected Learning Outcome:
  ✓ Implement AMP in training loop
  ✓ Understand precision tradeoffs
  ✓ Fix gradient underflow issues
  ✓ Achieve 2-3x speedup

⚡ SPEEDUP: 2.5x ⚡


MODULE 3: MODEL COMPILATION
─────────────────────────────────────────────────────────────────────────────
📝 File: notebooks/03_compile.ipynb
⏱️  Duration: 1.5 hours
🎯 Goal: Optimize models with torch.compile()

Topics:
  • How torch.compile() optimizes computation graphs
  • Compilation modes: default, reduce-overhead, max-autotune
  • Warmup and compilation overhead
  • Backend-specific optimizations
  • Deployment optimization

Key Concepts:
  → Compile fuses operations and reduces overhead
  → Different modes for different use cases
  → First run triggers compilation (slower)
  → Subsequent runs are fast

Expected Learning Outcome:
  ✓ Use torch.compile() effectively
  ✓ Choose appropriate compilation mode
  ✓ Work around compilation limitations
  ✓ Achieve 1.3-2.0x inference speedup

⚡ SPEEDUP: 1.3-2.0x ⚡


MODULE 4: GRADIENT CHECKPOINTING
─────────────────────────────────────────────────────────────────────────────
📝 File: notebooks/04_gradient_checkpointing.ipynb
⏱️  Duration: 2 hours
🎯 Goal: Train larger models with memory constraints

Topics:
  • Memory-compute tradeoff principle
  • torch.utils.checkpoint for activation checkpointing
  • Selective checkpointing strategies
  • When to use checkpointing
  • Recomputation cost analysis

Key Concepts:
  → Trade memory for compute during backprop
  → Recompute activations instead of storing them
  → 50% memory savings, 20-30% speed penalty
  → Essential for models > 500M parameters

Expected Learning Outcome:
  ✓ Implement gradient checkpointing
  ✓ Use selective checkpointing
  ✓ Understand memory-speed tradeoff
  ✓ Train larger models with limited memory

💾 MEMORY SAVED: 50% 💾


MODULE 5: DATALOADER OPTIMIZATION
─────────────────────────────────────────────────────────────────────────────
📝 File: notebooks/05_dataloader_optimization.ipynb
⏱️  Duration: 1.5 hours
🎯 Goal: Eliminate data loading bottlenecks

Topics:
  • pin_memory for fast CPU→GPU transfer
  • num_workers for parallel data loading
  • Batch size optimization strategies
  • Prefetching and persistent workers
  • DataLoader configuration best practices

Key Concepts:
  → Data loading can be a hidden bottleneck
  → Proper configuration reduces stalls
  → Device-specific settings matter
  → Dynamic batching adapts to memory

Expected Learning Outcome:
  ✓ Configure DataLoader optimally
  ✓ Find optimal batch size for your hardware
  ✓ Understand memory-throughput tradeoff
  ✓ Achieve 10-30% throughput improvement

⚡ SPEEDUP: 1.1-1.3x ⚡


MODULE 6: COMPLETE LLM OPTIMIZATION
─────────────────────────────────────────────────────────────────────────────
📝 File: notebooks/06_llm_optimization.ipynb
⏱️  Duration: 3 hours
🎯 Goal: Apply all techniques to a real language model

Topics:
  • Building optimized transformer architecture
  • Full LLM from scratch implementation
  • Layer-wise bottleneck identification
  • Combining multiple optimization techniques
  • Before/after performance comparison

Projects:
  • Profile baseline model
  • Apply AMP optimization
  • Add gradient checkpointing
  • Use torch.compile()
  • Measure cumulative speedup

Key Concepts:
  → Attention layers are typical bottleneck
  → Stack optimizations for multiplicative gains
  → Monitor memory and compute utilization
  → Validate all optimizations work together

Expected Learning Outcome:
  ✓ Train an optimized LLM
  ✓ Achieve 4-8x total speedup
  ✓ Combine multiple techniques
  ✓ Deploy efficient models

⚡ SPEEDUP: 4-8x ⚡


MODULE 7: ADVANCED TECHNIQUES & M3 MACBOOK
─────────────────────────────────────────────────────────────────────────────
📝 File: notebooks/07_advanced_optimization.ipynb
⏱️  Duration: 2.5 hours
🎯 Goal: Master cutting-edge optimization techniques

Topics:
  • Grouped Query Attention (GQA)
  • Activation offloading to CPU
  • Fused kernels for efficiency
  • Quantization (8-bit and 4-bit)
  • M3 MacBook specific optimizations
  • Hardware utilization strategies

Advanced Techniques:
  1. GQA: Reduce KV cache by 50-80%
  2. Offloading: Move activations to CPU
  3. Fusion: Combine operations
  4. Quantization: 75% memory reduction
  5. M3 Optimization: Unified memory strategies

Key Concepts:
  → Advanced techniques for specific scenarios
  → Understand when to use each technique
  → M3 MacBook has unified memory architecture
  → Hardware awareness is critical

Expected Learning Outcome:
  ✓ Implement advanced optimization techniques
  ✓ Apply M3 MacBook specific optimizations
  ✓ Understand hardware utilization
  ✓ Deploy quantized models

⚡ SPEEDUP: 2-4x (additional for specific cases) ⚡

═══════════════════════════════════════════════════════════════════════════════
🛠️ SUPPORTING RESOURCES
═══════════════════════════════════════════════════════════════════════════════

📄 Documentation:
  • README.md - Course overview and setup
  • OPTIMIZATION_GUIDE.md - Comprehensive optimization reference
  • BENCHMARK_RESULTS.md - Performance benchmarks and results
  • LEARNING_PATH.md - Detailed learning path and timelines

🐍 Code:
  • scripts/train.py - Complete training pipeline with all optimizations
  • models/llm.py - Optimized LLM models and components
  • utils/optimization_utils.py - Profiling and monitoring utilities

🚀 Quick Start:
  • quickstart.sh - Common commands and examples
  • requirements.txt - Python dependencies

═══════════════════════════════════════════════════════════════════════════════
⚡ QUICK REFERENCE: OPTIMIZATION IMPACT
═══════════════════════════════════════════════════════════════════════════════

Single Optimizations:
  AMP (FP16):              2.5x speed, 50% memory ✓✓✓
  torch.compile:           1.3-2.0x speed
  Gradient Checkpointing:  50% memory, 0.8x speed
  DataLoader tuning:       1.1-1.3x speed
  
Combined (Recommended):
  AMP + Checkpointing:     2.0x speed, 60% memory
  AMP + Compile:           2.5-4.0x speed
  All combined:            4-8x speed, 60-75% memory

Device-Specific:
  NVIDIA GPU:   Use pin_memory=True, large batch sizes
  M3 MacBook:   Use MPS backend, AMP is critical
  CPU:          Focus on checkpointing, lower batch sizes

═══════════════════════════════════════════════════════════════════════════════
📊 BENCHMARK SUMMARY
═══════════════════════════════════════════════════════════════════════════════

M3 MacBook Results (16M param model):
┌─────────────────────┬─────────────┬─────────────┬──────────┐
│ Configuration       │ Time/Iter   │ Throughput  │ Speedup  │
├─────────────────────┼─────────────┼─────────────┼──────────┤
│ Baseline (FP32)     │ 450ms       │ 71 s/s      │ 1.0x     │
│ AMP (FP16)          │ 180ms       │ 178 s/s     │ 2.5x     │
│ All Combined        │ 160ms       │ 200 s/s     │ 2.8x     │
└─────────────────────┴─────────────┴─────────────┴──────────┘

NVIDIA GPU Results (estimated):
┌─────────────────────┬─────────────┬─────────────┬──────────┐
│ Configuration       │ Time/Iter   │ Throughput  │ Speedup  │
├─────────────────────┼─────────────┼─────────────┼──────────┤
│ Baseline (FP32)     │ 200ms       │ 160 s/s     │ 1.0x     │
│ AMP (FP16)          │ 80ms        │ 400 s/s     │ 2.5x     │
│ All Combined        │ 50ms        │ 640 s/s     │ 4.0x     │
└─────────────────────┴─────────────┴─────────────┴──────────┘

═══════════════════════════════════════════════════════════════════════════════
🚀 GETTING STARTED
═══════════════════════════════════════════════════════════════════════════════

1. INSTALL DEPENDENCIES
   $ pip install -r requirements.txt

2. START FIRST NOTEBOOK
   $ jupyter notebook notebooks/01_profiling.ipynb

3. FOLLOW MODULES IN ORDER
   Module 1 → Module 2 → ... → Module 7

4. PRACTICE WITH TRAINING SCRIPT
   $ python scripts/train.py --device mps --use-amp --use-checkpoint

5. EXPERIMENT AND OPTIMIZE
   - Profile your own models
   - Apply optimization techniques
   - Measure improvements

═══════════════════════════════════════════════════════════════════════════════
💡 KEY PRINCIPLES
═══════════════════════════════════════════════════════════════════════════════

1. PROFILE FIRST
   → Don't optimize blind
   → Measure before and after
   → Identify real bottlenecks

2. START WITH AMP
   → Biggest bang for buck
   → Minimal code changes
   → 2.5x speedup for almost everything

3. COMBINE TECHNIQUES
   → Multiplicative effects
   → Stack optimizations
   → Synergistic benefits

4. HARDWARE-AWARE
   → Different devices need different strategies
   → M3 differs from CUDA
   → CPU has different constraints

5. BENCHMARK RIGOROUSLY
   → Measure wall-clock time
   → Track memory usage
   → Monitor thermal conditions

═══════════════════════════════════════════════════════════════════════════════
❓ FREQUENTLY ASKED QUESTIONS
═══════════════════════════════════════════════════════════════════════════════

Q: Which optimization should I start with?
A: Always start with AMP (Module 2). It gives 2.5x speedup with minimal effort.

Q: Is gradient checkpointing worth it?
A: Yes, if you have memory constraints. Trades 20-30% speed for 50% memory.

Q: Should I use torch.compile()?
A: Yes for inference. For training, benefits are smaller but still worthwhile.

Q: How do I know if I'm optimizing well?
A: Profile before and after. Track throughput and memory. Set realistic goals.

Q: What if torch.compile() fails?
A: It's not critical. Other optimizations provide most of the speedup.

Q: How do I choose batch size?
A: Start with 32-64, increase until OOM, then reduce by 10-20%.

═══════════════════════════════════════════════════════════════════════════════
📞 SUPPORT
═══════════════════════════════════════════════════════════════════════════════

For issues:
1. Check OPTIMIZATION_GUIDE.md for troubleshooting
2. Review notebook error cells for solutions
3. Refer to BENCHMARK_RESULTS.md for expected performance
4. Check PyTorch documentation for API details

═══════════════════════════════════════════════════════════════════════════════

                    Ready to optimize? Start with Module 1! 🚀

═══════════════════════════════════════════════════════════════════════════════
"""

def print_course_overview():
    print(COURSE_STRUCTURE)

def print_module_summary(module_num):
    modules = {
        1: "Profiling - Identify bottlenecks",
        2: "AMP - 2.5x speedup with FP16",
        3: "torch.compile - Optimize computation graph",
        4: "Gradient Checkpointing - Trade memory for compute",
        5: "DataLoader - Eliminate IO bottleneck",
        6: "Complete LLM - Combine all techniques",
        7: "Advanced Techniques - Cutting-edge optimizations"
    }
    print(f"\nModule {module_num}: {modules.get(module_num, 'Unknown')}")

def main():
    import sys
    
    print(COURSE_STRUCTURE)
    print("\n" + "="*80)
    print("Next Steps:")
    print("="*80)
    print("1. Install: pip install -r requirements.txt")
    print("2. Start: jupyter notebook notebooks/01_profiling.ipynb")
    print("3. Learn: Follow modules 1-7 sequentially")
    print("4. Practice: python scripts/train.py --help")
    print("5. Reference: See OPTIMIZATION_GUIDE.md for detailed information")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
