# M3 MacBook Optimization Guide
## Complete Guide for Training Deep Learning Models on Apple Silicon

### Table of Contents
1. [M3 Hardware Overview](#m3-hardware-overview)
2. [Limitations & Workarounds](#limitations--workarounds)
3. [Unified Memory Architecture](#unified-memory-architecture)
4. [PyTorch MPS Backend](#pytorch-mps-backend)
5. [Practical Optimization Techniques](#practical-optimization-techniques)
6. [Performance Expectations](#performance-expectations)
7. [Scaling from M3 to Multi-GPU](#scaling-from-m3-to-multi-gpu)

---

## M3 Hardware Overview

### M3 Pro/Max Specifications

```python
M3_SPECS = {
    'M3': {
        'cpu_cores': 8,
        'gpu_cores': 8,
        'unified_memory': '8 GB',
        'memory_bandwidth': '100 GB/s',
        'peak_fp32_tflops': 1.4,
        'peak_fp16_tflops': 2.8,
        'typical_power': '10-15W',
    },
    'M3_Pro': {
        'cpu_cores': 12,
        'gpu_cores': 18,
        'unified_memory': '18 GB',
        'memory_bandwidth': '150 GB/s',
        'peak_fp32_tflops': 3.9,
        'peak_fp16_tflops': 7.8,
        'typical_power': '15-20W',
    },
    'M3_Max': {
        'cpu_cores': 12,
        'gpu_cores': 38,
        'unified_memory': '36-128 GB',
        'memory_bandwidth': '300 GB/s',
        'peak_fp32_tflops': 8.2,
        'peak_fp16_tflops': 16.4,
        'typical_power': '20-30W',
    }
}
```

### Comparison with NVIDIA GPUs

```
Performance per Watt (Higher is Better):
├─ M3 MacBook Pro: 0.18-0.25 TFLOPS/W (excellent efficiency!)
├─ A100 80GB: 0.07 TFLOPS/W (optimized for throughput)
├─ H100: 0.10 TFLOPS/W
└─ RTX 4090: 0.08 TFLOPS/W

Memory Bandwidth:
├─ M3 Max unified memory: 300 GB/s (shared CPU/GPU)
├─ A100 HBM2e: 2 TB/s (NVIDIA GPU only, not shared)
├─ H100 HBM3: 3.35 TB/s
└─ M3 bandwidth advantage: ❌ NVIDIA is 6-10x faster

Model Training Time (70B parameter model):
├─ M3 Max alone: Not feasible (only 38 GPU cores)
├─ Single A100: 2-3 weeks with 16x
├─ M3 Max + external GPU: Can work (need eGPU)
└─ M3 Max + cloud GPU: Recommended for 100B+ models
```

---

## Limitations & Workarounds

### 1. Limited GPU Memory

**Problem**: M3 Max has only 36-128 GB unified memory vs 80-320 GB on high-end NVIDIA GPUs

```python
# For M3 Pro (18GB): Can train up to ~3B parameter models comfortably
# For M3 Max (36-128GB): Can train up to ~7B parameter models

FEASIBLE_MODELS_M3 = {
    'M3_Pro': {
        'with_checkpoint': '2-3B parameters',
        'example': 'SmallLLM, tiny models',
        'batch_size': 2-4,
        'max_seq_length': 2048,
    },
    'M3_Max_36GB': {
        'with_checkpoint': '4-7B parameters',
        'example': 'LLaMA-7B feasible with extreme optimization',
        'batch_size': 4-8,
        'max_seq_length': 2048,
    },
    'M3_Max_128GB': {
        'with_checkpoint': '13-20B parameters',
        'example': 'LLaMA-13B with checkpointing',
        'batch_size': 8-16,
        'max_seq_length': 4096,
    }
}

# Workaround: Use CPU offloading for larger models
from torch.distributed.fsdp import FSDP, CPUOffload

model = FSDP(
    model,
    cpu_offload=CPUOffload(
        offload_params=True,      # Move params to CPU when not in use
        offload_optimizer=False,  # Keep optimizer on GPU for speed
    ),
)
```

### 2. No Tensor Parallelism Across GPUs

**Problem**: M3 Macs can't connect to external GPUs easily (Thunderbolt limited)

```python
# Not feasible on M3:
# - Splitting single model across multiple GPUs
# - FSDP across multiple machines via standard setup

# Workaround for scaling:
# Option 1: Use cloud GPU for training, M3 for development
# Option 2: Use external Thunderbolt GPU (limited by TB bandwidth)
# Option 3: Train smaller models, then distill to M3-sized model

# Example: Distillation approach
class DistilledModel(nn.Module):
    def __init__(self, teacher_model, student_size='small'):
        super().__init__()
        self.teacher = teacher_model
        
        # Small student model for M3
        self.student = SmallTransformer(
            hidden_dim=256,
            num_layers=6,  # vs teacher's 32
            num_heads=4,
        )
    
    def forward(self, x):
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        
        student_logits = self.student(x)
        
        # KL divergence loss between teacher and student
        loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1),
            reduction='batchmean'
        )
        
        return loss
```

### 3. Multiprocessing Issues

**Problem**: M3's fork-based multiprocessing can cause issues with PyTorch

```python
# ❌ Wrong (causes hangs on M3):
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # Multiprocessing issues!
    pin_memory=True,
)

# ✅ Correct (works on M3):
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=0,  # No multiprocessing
    pin_memory=False,  # Doesn't apply to MPS
)

# For faster data loading without multiprocessing:
# Use persistent_workers (newer PyTorch versions) or async data loading
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=0,  # Still required on M3
    pin_memory=False,
    prefetch_factor=2,  # Buffer next batches in memory
)
```

### 4. Reduced Kernel Coverage

**Problem**: Not all PyTorch operations have MPS implementations

```python
# Check operation support
import torch

# List unsupported operations in your model
def check_mps_support(model, sample_input):
    """Verify all operations are MPS-supported"""
    
    try:
        # Forward pass
        output = model(sample_input.to('mps'))
        
        # Backward pass
        loss = output.sum()
        loss.backward()
        
        print("✅ All operations supported on MPS")
        return True
    
    except RuntimeError as e:
        print(f"❌ MPS operation not supported: {e}")
        return False

# Common unsupported operations on M3:
UNSUPPORTED_OPS = [
    'einsum',  # Use matmul + reshape instead
    'scatter',  # Use indexing
    'gather',   # Use fancy indexing
    'unique',   # Use torch.unique (added in recent PyTorch)
    'sort',     # Some variants unsupported
]

# Workaround: Use CPU for unsupported ops
def safe_op(tensor, op_fn):
    """Run operation on CPU if MPS unsupported"""
    if tensor.is_mps:
        return op_fn(tensor.cpu()).to('mps')
    else:
        return op_fn(tensor)
```

### 5. Compilation Limitations

**Problem**: torch.compile with MPS has limited optimization

```python
# ❌ Limited benefit on M3:
model = torch.compile(model, mode='default')

# ✅ Better: Focus on other optimizations
# torch.compile provides 1.1-1.2x on M3 (vs 1.5-2x on CUDA)
# Better ROI from: AMP (2.5x), checkpointing (1.3x per layer)

model = torch.compile(model, mode='reduce-overhead')

# More important: AMP and checkpoint
with autocast(device_type='mps', dtype=torch.float16):
    loss = model(input)
```

---

## Unified Memory Architecture

### How Unified Memory Works

```python
# M3 uses unified memory: CPU and GPU see same memory space
# No need to copy tensors like CUDA (device=cuda, device=cpu)

# ✅ Simpler code on M3:
x = torch.randn(1000, 1000, device='mps')
y = torch.randn(1000, 1000, device='mps')
z = x @ y  # Automatic memory management

# Behind the scenes:
# 1. CPU writes data
# 2. GPU reads via shared memory bus
# 3. GPU computes
# 4. CPU reads result
# 5. All managed by hardware, not programmer

# Advantage: No need for pin_memory or manual management
# Disadvantage: Slower than dedicated GPU memory due to shared bus
```

### Implications for Optimization

```python
# 1. Memory pressure is higher
#    - Both CPU and GPU compete for same memory
#    - If memory pressure too high, swaps to disk (very slow!)

# 2. Data loading bottleneck is different
#    - CPU → Unified Memory is fast (100 GB/s)
#    - Unified Memory → GPU computation is the bottleneck

# Best strategy for M3:
def optimize_for_m3_memory():
    """
    1. Reduce batch size (less activations in memory)
    2. Use gradient checkpointing (80-90% memory savings)
    3. CPU offload optimizer states
    4. Mixed precision (FP16, reduces memory by 50%)
    """
    
    config = {
        'batch_size': 4,  # Small batches
        'gradient_checkpointing': True,
        'mixed_precision': True,
        'optimizer_cpu_offload': True,
        'model_cpu_offload': False,  # Keep model on GPU
    }
    
    return config
```

---

## PyTorch MPS Backend

### Enabling MPS

```python
import torch

# Check MPS availability
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# Create model on device
model = MyTransformer(config).to(device)

# All tensors automatically use MPS
input_tensor = torch.randn(batch_size, seq_len, hidden_dim, device=device)
output = model(input_tensor)
```

### MPS Synchronization

```python
# ✅ Always synchronize before timing on M3
import time

# Correct timing
torch.mps.synchronize()
start = time.time()

output = model(input)
loss = criterion(output, target)
loss.backward()

torch.mps.synchronize()
end = time.time()

elapsed = end - start  # Accurate timing

# Without synchronize():
# ❌ start = time.time()
# ❌ output = model(input)  # Returns immediately (async)
# ❌ end = time.time()
# ❌ elapsed = ~0 (not accurate, still computing on GPU!)
```

### Mixed Precision with MPS

```python
from torch.amp import autocast, GradScaler

# MPS supports FP16 but not BF16
# Always use float16 for mixed precision on M3

scaler = GradScaler(device='mps')

for epoch in range(num_epochs):
    for batch in dataloader:
        input, target = batch
        input = input.to(device)
        target = target.to(device)
        
        # Use autocast for mixed precision
        with autocast(device_type='mps', dtype=torch.float16):
            output = model(input)
            loss = criterion(output, target)
        
        # Scale loss for numerical stability
        scaler.scale(loss).backward()
        
        # Unscale before optimizer step
        scaler.unscale_(optimizer)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer step with scaling
        scaler.step(optimizer)
        scaler.update()
        
        optimizer.zero_grad()
```

### Disabling Fallback to CPU

```python
# By default, unsupported ops fallback to CPU
# For debugging: disable fallback

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'

# Now unsupported ops will raise error instead of silently using CPU
# This helps identify which ops need to be implemented differently

try:
    output = model(input)
except RuntimeError as e:
    print(f"Unsupported operation: {e}")
    # Fix the model to use supported operations
```

---

## Practical Optimization Techniques

### 1. Optimal Batch Size for M3

```python
def find_optimal_batch_size_m3(model, device, dataset):
    """
    Find largest batch size that fits in M3 unified memory
    """
    
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    
    for batch_size in batch_sizes:
        try:
            # Try forward + backward pass
            dataloader = DataLoader(dataset, batch_size=batch_size)
            
            for input, target in dataloader:
                input = input.to(device)
                target = target.to(device)
                
                output = model(input)
                loss = criterion(output, target)
                loss.backward()
                optimizer.zero_grad()
            
            print(f"✅ Batch size {batch_size} works")
            optimal_batch_size = batch_size
        
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"❌ Batch size {batch_size} exceeds memory")
                return optimal_batch_size
            else:
                raise
    
    return optimal_batch_size

# For M3 Pro (18GB): typically 4-8
# For M3 Max (36GB): typically 8-16
# For M3 Max (128GB): typically 16-32
```

### 2. Thermal-Aware Training

```python
import psutil
import time

class ThermalAwareTrainer:
    def __init__(self, model, optimizer, device, temp_threshold=85):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.temp_threshold = temp_threshold
        self.throttled = False
    
    def get_cpu_temp(self):
        """Get CPU temperature (macOS)"""
        import subprocess
        try:
            # M3 Macs report temperature via powermetrics
            output = subprocess.check_output(
                ['powermetrics', '-n', '1', '-f', 'cpu_power'],
                stderr=subprocess.DEVNULL,
                timeout=5
            ).decode()
            # Parse temperature from output
            # Note: Exact parsing depends on macOS version
            return 75  # Placeholder
        except:
            return None
    
    def get_gpu_utilization(self):
        """Check GPU utilization on M3"""
        import subprocess
        try:
            output = subprocess.check_output(
                ['powermetrics', '-n', '1'],
                stderr=subprocess.DEVNULL,
            ).decode()
            # Parse GPU utilization
            return 60  # Placeholder
        except:
            return None
    
    def train_step(self, input, target):
        """Train with thermal throttling protection"""
        
        # Check temperature before step
        temp = self.get_cpu_temp()
        
        if temp and temp > self.temp_threshold:
            print(f"⚠️ Thermal throttling active (temp: {temp}°C)")
            
            if not self.throttled:
                # Reduce batch size or learning rate
                self.throttled = True
                print("Reducing training intensity...")
            
            # Add delay to let system cool
            time.sleep(10)
        else:
            self.throttled = False
        
        # Normal training step
        output = self.model(input)
        loss = criterion(output, target)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item()
```

### 3. Profile-Guided Optimization for M3

```python
import cProfile
import pstats

def profile_m3_model(model, input_tensor):
    """Identify bottlenecks on M3"""
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Forward pass
    output = model(input_tensor)
    loss = output.sum()
    
    # Backward pass
    loss.backward()
    
    profiler.disable()
    
    # Analyze
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
    
    # Look for:
    # 1. PyTorch -> CPU fallback (slow on M3)
    # 2. Synchronization overhead
    # 3. CPU-GPU communication

# Example optimization based on profiling:
# If einsum is slow: Replace with matmul + reshape
# If gather is slow: Replace with fancy indexing
# If sort is slow: Use alternatives like argsort + index_select
```

### 4. M3 Optimal Training Config

```python
from dataclasses import dataclass

@dataclass
class M3OptimalConfig:
    # Model
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 4
    ffn_hidden: int = 1024
    
    # Training
    batch_size: int = 8  # M3 Pro: 4, M3 Max 36GB: 8, M3 Max 128GB: 16
    seq_length: int = 2048
    learning_rate: float = 1e-4
    
    # Optimizations
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    use_compile: bool = False  # Limited benefit on M3
    optimizer_cpu_offload: bool = True
    
    # Data loading (M3-specific)
    num_workers: int = 0  # ⚠️ Must be 0 on M3!
    pin_memory: bool = False  # Doesn't apply to MPS
    prefetch_factor: int = 2
    
    # Device
    device: str = 'mps'
    
    def validate_for_m3_max_36gb(self):
        """Check config fits in M3 Max 36GB"""
        
        # Estimate memory: ~4 bytes * params + optimizer + activations
        estimated_memory = (
            self.hidden_dim * self.num_layers * 2 * 4 +  # Weights
            self.batch_size * self.seq_length * self.hidden_dim * 2 +  # Activations
            self.hidden_dim * self.num_layers * 2 * 8  # Optimizer (AdamW = 8 bytes/param)
        ) / 1e9
        
        max_memory_gb = 30  # Leave 6GB buffer
        
        if estimated_memory > max_memory_gb:
            print(f"⚠️ Config uses {estimated_memory:.1f}GB (max: {max_memory_gb}GB)")
            print("Try reducing: batch_size, hidden_dim, num_layers, or seq_length")
            return False
        
        print(f"✅ Config uses {estimated_memory:.1f}GB (fits in 36GB)")
        return True

# Create optimal config for M3 Max 36GB
config = M3OptimalConfig(
    hidden_dim=768,    # Larger than M3 Pro
    num_layers=12,
    batch_size=8,
    gradient_checkpointing=True,
    mixed_precision=True,
)
config.validate_for_m3_max_36gb()
```

---

## Performance Expectations

### Realistic Benchmarks on M3

```python
# Training throughput (tokens/second) for different models
M3_BENCHMARKS = {
    'M3_Pro_18GB': {
        'SmallLLM_256d_6L': {
            'fp32': 120,  # tokens/sec
            'fp16': 250,  # ~2x with AMP
            'with_checkpoint': 200,  # ~0.8x (slower backward)
            'combined': 350,  # fp16 + checkpoint + compile ≈ 3x
        },
        'MediumLLM_768d_12L': {
            'fp32': 30,  # tokens/sec
            'fp16': 60,  # ~2x with AMP
            'combined': 90,  # ~3x
        },
    },
    'M3_Max_36GB': {
        'SmallLLM_384d_12L': {
            'fp32': 250,
            'fp16': 500,  # ~2x
            'combined': 700,  # ~2.8x
        },
        'LargeModel_1024d_24L': {
            'fp32': 60,
            'fp16': 120,
            'combined': 160,  # ~2.7x
        },
    },
    'M3_Max_128GB': {
        'LLaMA_7B': {
            'fp32': 45,  # Very slow!
            'fp16': 90,  # ~2x
            'with_checkpoint': 70,  # ~1.5x
            'combined': 110,  # ~2.4x
        }
    }
}
```

### Speedup Achieved on M3

```
M3 MacBook Optimization ROI (M3 Max 36GB):

┌─ Baseline (FP32, no optimization)
│  └─ 1.0x

├─ + AMP (FP16)
│  └─ 2.5x ⭐ Best bang for buck

├─ + Gradient Checkpointing
│  └─ 1.5-2.0x effective (slower but less memory)

├─ + AMP + Checkpointing
│  └─ 2.0-2.5x (reduced memory, good balance)

├─ + torch.compile
│  └─ 1.1-1.2x (limited benefit on M3)

└─ + AMP + Checkpointing + Compile + CPU offload
   └─ 2.3-2.8x realistic (not additive)

RECOMMENDATION FOR M3:
1. AMP: +2.5x (mandatory)
2. Gradient checkpointing: Enables larger models
3. CPU optimizer offload: Saves ~10% memory
4. torch.compile: Skip (not worth complexity on M3)
```

### Training Time Comparison

```python
# Training SmallLLM (384d, 12 layers) on 1 billion tokens

TRAINING_TIMES = {
    'M3_Pro_baseline': {
        'fp32': '4.2 hours',
        'fp16_optimized': '1.2 hours',  # 3.5x speedup
    },
    'M3_Max_36GB_baseline': {
        'fp32': '2.1 hours',
        'fp16_optimized': '0.6 hours',  # 3.5x speedup
    },
    'GPU_A100_comparison': {
        'single_a100_fp32': '15 minutes',
        'single_a100_optimized': '3 minutes',  # 5x speedup
        'vs_m3_max_ratio': '(3 min / 36 min) = 0.08x',  # A100 is 12x faster
    }
}

# Reality: M3 is 10-15x slower than high-end GPU for training
# BUT: M3 is free (you own it!), GPU time costs ~$1/hour
```

---

## Scaling from M3 to Multi-GPU

### Development on M3, Production on Cloud

```python
# Write model once, run on M3 or cloud GPU with minimal changes

class DeviceAgnosticModel(nn.Module):
    def __init__(self, config, device='auto'):
        super().__init__()
        
        # Auto-select device
        if device == 'auto':
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        self.model = MyTransformer(config)
        self.model = self.model.to(self.device)
    
    def forward(self, x):
        return self.model(x)

# Use same code for both M3 and GPU
model = DeviceAgnosticModel(config, device='auto')

# Performance automatically optimizes for device:
# - M3: MPS backend, FP16, checkpointing
# - CUDA: Tensor core, NCCL, larger batches
# - CPU: CPU-only optimizations
```

### Distributed Training Setup (Cloud)

```python
# Once your model works on M3, scale to multiple GPUs

# 1. Install torch distributed
# pip install torch torchrun

# 2. Wrap model with DDP (automatic for multiple GPUs)
from torch.nn.parallel import DistributedDataParallel

def train_distributed():
    # Local rank determined by torchrun
    local_rank = int(os.environ['LOCAL_RANK'])
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    
    model = MyTransformer(config)
    model = model.to(local_rank)
    model = DistributedDataParallel(model)
    
    # Training loop (same as M3!)
    for epoch in range(num_epochs):
        for batch in dataloader:
            output = model(batch)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

# 3. Launch with torchrun
# torchrun --nproc_per_node=8 train.py
```

### Model Distillation: M3 → Cloud → M3

```python
# Best practice: Train large model on cloud, deploy optimized version on M3

class StudentModel(nn.Module):
    """Small model optimized for M3"""
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 50257),  # vocab size
        )
    
    def forward(self, x):
        return self.model(x)

class DistillationTrainer:
    def __init__(self, teacher_model, student_model, device='mps'):
        self.teacher = teacher_model.to(device).eval()
        self.student = student_model.to(device)
        self.device = device
        self.temperature = 4.0
    
    def train_step(self, batch):
        input_ids, target = batch
        input_ids = input_ids.to(self.device)
        target = target.to(self.device)
        
        # Teacher generates soft labels
        with torch.no_grad():
            teacher_logits = self.teacher(input_ids)
        
        # Student predicts
        student_logits = self.student(input_ids)
        
        # KL divergence loss (knowledge distillation)
        kl_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction='batchmean'
        )
        
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(student_logits, target)
        
        # Combined loss (usually 0.3 * KL + 0.7 * CE)
        alpha = 0.3
        total_loss = alpha * kl_loss + (1 - alpha) * ce_loss
        
        return total_loss

# Result: 10-30x smaller student model, runs on M3!
```

---

## Summary: M3 MacBook Training Strategy

### ✅ What M3 is Good For
1. **Development & experimentation**: Small models (< 3B params)
2. **Fine-tuning**: Pre-trained models on M3
3. **Inference**: LLM deployment with quantization
4. **Prototyping**: Before scaling to cloud
5. **Quick iteration**: No cloud costs, instant access

### ❌ What M3 is NOT Good For
1. Training huge models (100B+) from scratch
2. Large-scale distributed training
3. Production inference for millions of requests
4. Real-time training with massive datasets

### 📋 Recommended Workflow
```
┌─ IDEA
│
├─ Prototype on M3 (1-2 hours)
│  └─ Test model architecture, loss, convergence
│
├─ Scale to cloud (if needed)
│  └─ Use torchrun + 8x GPU for real training
│  └─ Expected speedup: 50-100x vs M3
│
└─ Deploy on M3 (optional)
   └─ Distill/quantize for inference
   └─ Use onnx-runtime or CoreML for speed
```

### Key Settings for M3 Training
```python
M3_FINAL_CONFIG = {
    'device': 'mps',
    'batch_size': 8,  # M3 Max 36GB
    'seq_length': 2048,
    'mixed_precision': True,  # FP16
    'gradient_checkpointing': True,
    'num_workers': 0,  # MUST be 0!
    'pin_memory': False,
    'learning_rate': 1e-4 * math.sqrt(batch_size / 256),  # Scaled
    'warmup_steps': 2000,
    'optimizer_cpu_offload': True,
    'use_compile': False,  # Not worth it on M3
}
```

### Expected Results on M3
- **Training speed**: 50-100 tokens/second for small models
- **Memory efficiency**: 80-90% reduction with checkpointing + AMP
- **Speedup with optimization**: 2.5-3.5x vs baseline
- **Model size**: Up to 7-13B params with aggressive optimization
- **Time per epoch**: 5-30 minutes depending on dataset size
