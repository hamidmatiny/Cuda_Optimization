#!/bin/bash
# Quick start commands for CUDA Optimization Course

echo "CUDA Optimization Course - Quick Start Guide"
echo "=============================================="
echo ""

# Check environment
echo "1. Checking Python environment..."
python3 --version
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'MPS Available: {torch.backends.mps.is_available()}')"
echo ""

# Install requirements
echo "2. Installing requirements..."
echo "   Run: pip install -r requirements.txt"
echo ""

# Run notebooks
echo "3. Running Jupyter notebooks..."
echo "   Start Jupyter: jupyter notebook"
echo "   Notebooks to run in order:"
echo "   - notebooks/01_profiling.ipynb"
echo "   - notebooks/02_amp.ipynb"
echo "   - notebooks/03_compile.ipynb"
echo "   - notebooks/04_gradient_checkpointing.ipynb"
echo "   - notebooks/05_dataloader_optimization.ipynb"
echo "   - notebooks/06_llm_optimization.ipynb"
echo "   - notebooks/07_advanced_optimization.ipynb"
echo ""

# Training examples
echo "4. Training examples:"
echo ""
echo "   Baseline (FP32, no optimizations):"
echo "   python scripts/train.py --device mps --model-size small --epochs 2"
echo ""
echo "   With AMP only:"
echo "   python scripts/train.py --device mps --model-size small --epochs 2 --use-amp"
echo ""
echo "   With AMP + Checkpointing:"
echo "   python scripts/train.py --device mps --model-size small --epochs 2 --use-amp --use-checkpoint"
echo ""
echo "   With all optimizations (RECOMMENDED):"
echo "   python scripts/train.py --device mps --model-size small --epochs 2 --use-amp --use-checkpoint --use-compile"
echo ""
echo "   Large model on M3:"
echo "   python scripts/train.py --device mps --model-size large --batch-size 16 --seq-length 256 --use-amp --use-checkpoint"
echo ""

# Performance benchmarking
echo "5. Benchmarking different configurations:"
echo ""
echo "   python scripts/train.py --device mps --model-size small --use-amp"
echo "   python scripts/train.py --device mps --model-size small --use-checkpoint"
echo "   python scripts/train.py --device mps --model-size small --use-amp --use-checkpoint"
echo ""

# Key metrics to track
echo "6. Metrics to monitor:"
echo "   - Training time per epoch"
echo "   - Throughput (samples/sec)"
echo "   - GPU/CPU memory usage"
echo "   - Loss convergence"
echo ""

# M3 MacBook specific
echo "7. M3 MacBook recommendations:"
echo "   - Always use: --device mps --use-amp"
echo "   - Try: --batch-size 32 or 64"
echo "   - Try: --seq-length 256 (reduce if OOM)"
echo "   - Check Activity Monitor for thermal throttling"
echo ""

echo "=============================================="
echo "Happy optimizing! 🚀"
