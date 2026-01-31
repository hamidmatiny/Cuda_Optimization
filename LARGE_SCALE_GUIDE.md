# Large-Scale Model Training Guide
## Optimizing for 100B+ Parameter Models with Multi-GPU Setup

### Table of Contents
1. [Overview & Architecture](#overview--architecture)
2. [Scaling Laws & Resource Planning](#scaling-laws--resource-planning)
3. [Multi-GPU Training Strategies](#multi-gpu-training-strategies)
4. [Memory Management at Scale](#memory-management-at-scale)
5. [Communication & Synchronization](#communication--synchronization)
6. [Distributed Optimization](#distributed-optimization)
7. [Production Deployment](#production-deployment)

---

## Overview & Architecture

### Model Scaling Spectrum

```
Small Models (< 1B parameters)
├─ Typical use: Single GPU
├─ Memory: 4-8 GB
├─ Training time: Hours to days
└─ Techniques: Basic AMP, gradient checkpointing

Medium Models (1B - 10B parameters)
├─ Typical use: 1-4 GPUs
├─ Memory: 8-40 GB
├─ Training time: Days to weeks
└─ Techniques: Tensor parallel, pipeline parallel, gradient checkpointing

Large Models (10B - 100B parameters)
├─ Typical use: 4-64 GPUs
├─ Memory: 40-400 GB
├─ Training time: Weeks to months
└─ Techniques: Advanced distribution, activation checkpointing, expert parallelism

Massive Models (100B - 1T+ parameters)
├─ Typical use: 64-10,000+ GPUs
├─ Memory: 400GB - multiple TB
├─ Training time: Months to years
└─ Techniques: Full distributed stack, custom kernels, FSDP, DeepSpeed
```

### GPU Hardware Recommendations

#### For 100B Parameter Models
```python
RECOMMENDED_HARDWARE = {
    # 70B model (like LLaMA-70B)
    '70B': {
        'minimum': '8x A100 (80GB)',
        'recommended': '16x A100 (80GB)',
        'optimal': '32x A100 (80GB)',
        'memory_per_gpu': '40-50GB',
        'training_time': '3-6 weeks',
        'cost_estimate': '$500k-$2M'
    },
    
    # 175B model (like GPT-3)
    '175B': {
        'minimum': '16x A100 (80GB)',
        'recommended': '32x A100 (80GB)',
        'optimal': '64x A100 (80GB)',
        'memory_per_gpu': '40-60GB',
        'training_time': '2-3 months',
        'cost_estimate': '$2M-$5M'
    },
    
    # 1T parameter model
    '1T': {
        'minimum': '256x A100 (80GB)',
        'recommended': '512x A100 (80GB)',
        'optimal': '1024x A100 (80GB)',
        'memory_per_gpu': '60-80GB',
        'training_time': '3-6 months',
        'cost_estimate': '$10M-$50M'
    }
}
```

---

## Scaling Laws & Resource Planning

### Chinchilla Scaling Laws (Optimal Compute Allocation)

For a given compute budget C (in FLOPs):
- **Optimal model size**: N ≈ C / (6D)
- **Optimal dataset tokens**: D ≈ 6N
- **Rule**: Equal compute for model & data

```python
def compute_optimal_scaling(total_flops_budget):
    """
    Calculate optimal model and data size for given compute budget
    """
    # Chinchilla optimal: N ≈ C / (6D)
    # Rearranging for D: D ≈ 6N
    # Total compute: C = 6ND
    
    N_optimal = (total_flops_budget / 6) ** 0.5
    D_optimal = 6 * N_optimal
    
    return {
        'model_params': N_optimal,
        'tokens': D_optimal,
        'training_flops': total_flops_budget,
        'epochs': D_optimal / (total_flops_budget / (6 * N_optimal))
    }

# Example: 100 PetaFLOPs
result = compute_optimal_scaling(1e17)
print(f"Model params: {result['model_params']:.2e}")
print(f"Training tokens: {result['tokens']:.2e}")
```

### Training Time Estimation

```python
def estimate_training_time(
    model_params,           # Total parameters
    num_gpus,              # Number of GPUs
    tokens_per_gpu,        # Tokens processed per GPU per second
    training_tokens,       # Total tokens to train on
    seq_length=2048
):
    """
    Estimate total training time
    """
    # FLOPs per token (for transformer): 6ND (N=params, D=seq_len)
    flops_per_token = 6 * model_params * seq_length
    
    # Total FLOPs needed
    total_flops = flops_per_token * training_tokens
    
    # Peak FLOPs per GPU (A100 80GB ≈ 312 TFLOPS in fp16)
    peak_flops_per_gpu = 312e12  # FP16 peak
    
    # Effective FLOPs accounting for efficiency (60-70% typical)
    efficiency = 0.65
    effective_flops_per_gpu = peak_flops_per_gpu * efficiency
    
    # Total effective FLOPs
    total_effective_flops = effective_flops_per_gpu * num_gpus
    
    # Training time in seconds
    training_seconds = total_flops / total_effective_flops
    training_hours = training_seconds / 3600
    training_days = training_hours / 24
    
    return {
        'seconds': training_seconds,
        'hours': training_hours,
        'days': training_days,
        'cost_per_hour': num_gpus * 1.0,  # ~$1/hour per A100
        'total_cost': (training_hours * num_gpus * 1.0)
    }

# Example: Training 70B model on 16x A100s
time_est = estimate_training_time(
    model_params=70e9,
    num_gpus=16,
    tokens_per_gpu=1000,
    training_tokens=1e12,  # 1 trillion tokens
    seq_length=4096
)
print(f"Training time: {time_est['days']:.1f} days")
print(f"Total cost: ${time_est['total_cost']:.2e}")
```

### Memory Requirements

```python
def memory_calculation(
    model_params,
    batch_size,
    seq_length,
    use_gradient_checkpointing=True,
    mixed_precision=True
):
    """
    Calculate memory requirements per GPU
    """
    # Model weights in bytes (FP32 = 4 bytes, FP16 = 2 bytes)
    model_dtype = 2 if mixed_precision else 4
    model_memory = model_params * model_dtype / 1e9  # GB
    
    # Optimizer states (AdamW has 2 states per param)
    optimizer_memory = model_params * 2 * 4 / 1e9  # GB (always FP32)
    
    # Activations memory
    # Transformer: forward pass stores activations
    # batch_size * seq_length * hidden_dim * num_layers
    hidden_dim = 4096  # typical
    num_layers = 32    # typical
    dtype_bytes = 2 if mixed_precision else 4
    
    activation_memory_per_token = hidden_dim * num_layers * dtype_bytes / 1e9
    activation_memory = batch_size * seq_length * activation_memory_per_token
    
    if use_gradient_checkpointing:
        activation_memory *= 0.1  # 90% reduction
    
    # Gradients memory (same size as weights in training)
    gradient_memory = model_memory
    
    if mixed_precision:
        gradient_memory *= 0.5  # FP16 gradients
    
    # Total per GPU
    total_memory = (
        model_memory +
        optimizer_memory +
        activation_memory +
        gradient_memory
    )
    
    return {
        'model_weights_gb': model_memory,
        'optimizer_states_gb': optimizer_memory,
        'activations_gb': activation_memory,
        'gradients_gb': gradient_memory,
        'total_gb': total_memory,
        'recommended_gpu_memory': int(total_memory * 1.2)  # 20% buffer
    }

# Example: 70B model training
mem = memory_calculation(
    model_params=70e9,
    batch_size=1,  # per GPU
    seq_length=4096,
    use_gradient_checkpointing=True,
    mixed_precision=True
)
print(f"Total memory per GPU: {mem['total_gb']:.1f} GB")
print(f"Recommended GPU: {mem['recommended_gpu_memory']} GB")
```

---

## Multi-GPU Training Strategies

### 1. Data Parallelism (DP)

**Best for**: Medium models (< 10B) on few GPUs

```python
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel

# Simple data parallelism
model = MyTransformer(config)
model = DataParallel(model, device_ids=[0, 1, 2, 3])

# Forward pass splits batch across GPUs
# Backward pass aggregates gradients
output = model(input)  # input shape: (batch_size * 4, seq_len)
loss = criterion(output, target)
loss.backward()  # Synchronized gradient reduction
optimizer.step()
```

**Pros**: Simple, minimal code changes
**Cons**: Not memory efficient, high communication overhead

### 2. Distributed Data Parallelism (DDP)

**Best for**: Large models (10B-100B) on multiple nodes

```python
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group

# Initialize distributed training
init_process_group(backend='nccl')  # Use NCCL for GPU training

# Create model and wrap with DDP
model = MyTransformer(config)
model = model.to(rank)  # rank = process rank
model = DDP(model, device_ids=[rank])

# Each GPU processes its own batch
# Gradients synchronized across all GPUs via NCCL
# All-reduce operation is efficient

dataloader = get_dataloader(rank, world_size)

for epoch in range(num_epochs):
    for batch_idx, (input, target) in enumerate(dataloader):
        input = input.to(rank)
        target = target.to(rank)
        
        output = model(input)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()  # Automatically synchronized
        optimizer.step()
```

**Launch with**:
```bash
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
         --master_addr=10.0.0.1 --master_port=29500 \
         train.py
```

**Pros**: Efficient, minimal code changes, scalable
**Cons**: Still requires syncing full model gradients

### 3. Tensor Parallelism (TP)

**Best for**: Single-node, very large models

Split model layers across GPUs horizontally:

```python
class TensorParallel(nn.Module):
    def __init__(self, model, num_partitions):
        super().__init__()
        self.model = model
        self.num_partitions = num_partitions
    
    def forward(self, x):
        # Split input across tensor dimension
        x_chunks = torch.tensor_split(x, self.num_partitions, dim=-1)
        
        # Process on different GPUs
        outputs = []
        for i, chunk in enumerate(x_chunks):
            chunk = chunk.to(i)
            output = self.model(chunk)
            outputs.append(output.to(0))
        
        # Concatenate results
        return torch.cat(outputs, dim=-1)
```

**Example from Megatron-LM**:

```python
# Colossalai tensor parallelism (70B model across 8 GPUs)
from colossalai.nn import apply_1d_tensor_parallelism

class ParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        # Split weights across GPUs
        chunk_size = out_features // world_size
        self.linear = nn.Linear(in_features, chunk_size)
    
    def forward(self, x):
        # Local computation + all-gather for communication
        return all_gather(self.linear(x))
```

**Pros**: Memory efficient per GPU, good for large models
**Cons**: High communication overhead, complex implementation

### 4. Pipeline Parallelism (PP)

**Best for**: Very deep models (100+ layers)

Split model layers vertically across GPUs:

```python
import torch.distributed as dist
from typing import List

class PipelineParallel:
    def __init__(self, stages: List[nn.Module], num_gpus: int):
        self.stages = stages
        self.num_gpus = num_gpus
    
    def forward_pass(self, x):
        # Stage 0: GPU 0
        x = x.to(0)
        x = self.stages[0](x)
        
        # Stage 1: GPU 1
        x = x.to(1)
        x = self.stages[1](x)
        
        # ... continue for all stages
        return x
    
    def microbatch_forward(self, x, num_microbatches):
        """GPipe: split batch into microbatches for pipeline parallelism"""
        microbatch_size = x.shape[0] // num_microbatches
        outputs = []
        
        for i in range(num_microbatches):
            mb = x[i*microbatch_size:(i+1)*microbatch_size]
            output = self.forward_pass(mb)
            outputs.append(output)
        
        return torch.cat(outputs, dim=0)
```

**Example**: Training 175B model with 8 pipeline stages on 8 GPUs
- Each GPU holds 1 layer
- Forward pass: 8 sequential GPU-to-GPU transfers
- Backward pass: Gradient computation then all-reduce

**Pros**: Memory very efficient, scales to huge models
**Cons**: Pipeline bubbles, load imbalance, high complexity

### 5. Fully Sharded Data Parallel (FSDP)

**Best for**: Massive models (100B+) on many GPUs

```python
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

# Initialize distributed
dist.init_process_group(backend='nccl')

model = MyLargeTransformer(config)

# Wrap model with FSDP
auto_wrap_policy = size_based_auto_wrap_policy(
    min_num_params=1e8  # Wrap layers with >100M params
)

model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    sharding_strategy='FULL_SHARD',  # Each GPU holds 1/N params
    cpu_offload=CPUOffload(offload_params=True),  # Optional: CPU offload
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # Prefetch next layer
)

# Training looks normal
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for batch in dataloader:
    output = model(batch)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**How FSDP works**:
1. Each GPU stores 1/N of model parameters
2. Forward pass: All-gather required parameters
3. Backward pass: Compute gradients, then reduce-scatter

**Example**: 175B model on 64 GPUs
- Each GPU: 175B / 64 ≈ 2.7B parameters (~10 GB)
- Communication: All-gather + reduce-scatter per forward/backward
- Effective throughput: 60-70% of peak

**Pros**: Scales to massive models, good communication efficiency
**Cons**: More complex, requires careful tuning

---

## Memory Management at Scale

### Activation Checkpointing (Gradient Checkpointing)

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0:  # Checkpoint every other layer
                # Recompute during backward pass
                x = checkpoint(
                    layer,
                    x,
                    use_reentrant=False,  # For efficiency
                    preserve_rng_states=True
                )
            else:
                x = layer(x)
        return x
```

**Memory Savings**:
- Without checkpointing: O(L) memory (L = number of layers)
- With full checkpointing: O(1) memory per layer
- With selective checkpointing: O(L/2) memory
- Trade: 20-30% slower backward pass

### CPU Offloading

```python
import torch.distributed as dist
from torch.distributed.fsdp import CPUOffload

# Offload optimizer states to CPU
model = FSDP(
    model,
    cpu_offload=CPUOffload(
        offload_params=True,      # Offload parameters
        offload_optimizer=True    # Offload optimizer states
    ),
)

# With CPU offload:
# GPU memory: 70B / 8 GPUs = 8.75 GB (just parameters)
# CPU memory: 8.75B * 3 (param + 2x optimizer states) ≈ 26 GB
```

### Mixed Precision Strategy

```python
from torch.amp import autocast_mode
from torch.cuda.amp import GradScaler

# For 70B+ models, use stricter precision rules
model_fp16 = {
    'embedding', 'attention', 'linear_out'
}  # FP16 for these
model_fp32 = {
    'layer_norm', 'softmax', 'loss'
}  # FP32 for numerical stability

autocast_kwargs = {
    'dtype': torch.float16,
    'include_fp32_layers': list(model_fp32)
}

with autocast(**autocast_kwargs):
    output = model(input)
    loss = criterion(output, target)

scaler = GradScaler(
    init_scale=2**15,      # Start higher for large models
    max_scale=2**24,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000
)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```

---

## Communication & Synchronization

### Gradient Compression

For large models, gradient communication dominates training time.

```python
class CompressedGradient:
    def compress(self, gradient, sparsity=0.99):
        """
        Keep only top 1% of gradients (by magnitude)
        """
        k = int(gradient.numel() * (1 - sparsity))
        
        # Top-k selection
        topk_vals, topk_idxs = torch.topk(
            gradient.abs().flatten(),
            k,
            largest=True
        )
        
        # Create sparse tensor
        sparse_grad = torch.zeros_like(gradient)
        sparse_grad.view(-1)[topk_idxs] = gradient.view(-1)[topk_idxs]
        
        return sparse_grad
    
    def decompress(self, sparse_grad):
        """Decompress on receiving end"""
        return sparse_grad  # Already full tensor
```

**Communication Cost Reduction**:
- Full precision: 175B params * 2 bytes = 350 GB per all-reduce
- With 99% sparsity: 3.5 GB per all-reduce
- Time reduction: ~100x for communication

### Overlap Computation & Communication

```python
class OverlappedDDP(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        # Forward pass
        output = self.model(x)
        return output
    
    def backward_with_overlap(self, loss):
        """
        Start gradient reduction while still computing gradients
        """
        loss.backward()
        
        # PyTorch's DDP automatically overlaps computation/communication
        # during backward pass with find_unused_parameters=False
```

**Using PyTorch's built-in overlap**:
```python
from torch.nn.parallel import DistributedDataParallel

model = DDP(
    model,
    device_ids=[rank],
    find_unused_parameters=False,  # Enable overlap
    gradient_as_bucket_view=True,   # Reduce memory
)
```

---

## Distributed Optimization

### Learning Rate Scaling

For large batch training, scale learning rate:

```python
# Assume linear scaling rule
base_lr = 1e-4
base_batch_size = 256
actual_batch_size = 256 * num_gpus

# Linear scaling
scaled_lr = base_lr * (actual_batch_size / base_batch_size)

# Example: 8 GPUs × 256 batch = 2048
# Scaled learning rate: 1e-4 * (2048 / 256) = 8e-4

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=scaled_lr,
    betas=(0.9, 0.95),
    eps=1e-5,
    weight_decay=0.1
)
```

### Warmup Strategy

```python
def lr_lambda(step):
    """Warmup then cosine decay"""
    warmup_steps = 10000
    total_steps = 1000000
    
    if step < warmup_steps:
        return step / warmup_steps
    
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda
)
```

### Gradient Accumulation for Large Batches

```python
accumulation_steps = 4
effective_batch_size = batch_size * accumulation_steps * num_gpus

for epoch in range(num_epochs):
    for step, (input, target) in enumerate(dataloader):
        output = model(input)
        loss = criterion(output, target) / accumulation_steps
        
        loss.backward()
        
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
```

---

## Production Deployment

### Checkpoint/Resume from FSDP

```python
import torch.distributed as dist
from torch.distributed.fsdp import FSDP, FullStateDictConfig

def save_checkpoint(model, optimizer, step, path):
    """Save full model state from FSDP"""
    full_state_dict_config = FullStateDictConfig(
        offload_to_cpu=True,
        rank0_only=True,
    )
    
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        full_state_dict_config,
    ):
        cpu_state = model.state_dict()
        
    if dist.get_rank() == 0:
        checkpoint = {
            'model': cpu_state,
            'optimizer': optimizer.state_dict(),
            'step': step,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

def load_checkpoint(model, optimizer, path):
    """Load checkpoint into FSDP model"""
    checkpoint = torch.load(path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    step = checkpoint['step']
    
    return step
```

### Inference Optimization

```python
# For inference, consolidate model to single GPU
class InferenceOptimizer:
    def __init__(self, model_path, device):
        self.model = self.load_consolidated_model(model_path, device)
        self.model.eval()
    
    def load_consolidated_model(self, path, device):
        """Load FSDP checkpoint into single device"""
        checkpoint = torch.load(path, map_location=device)
        model = MyLargeTransformer()
        model.load_state_dict(checkpoint['model'])
        return model.to(device).half()  # FP16 inference
    
    @torch.no_grad()
    def generate(self, prompt, max_length=100):
        """Generate with KV cache for efficiency"""
        # Implementation with KV caching
        pass
```

---

## Summary: Best Practices for Large-Scale Training

| Model Size | Strategy | GPUs | Memory/GPU | Training Time |
|-----------|----------|------|-----------|---------------|
| 1-10B | DDP | 4-8 | 40-80 GB | Days-Weeks |
| 10-70B | DDP + Checkpointing | 8-16 | 40-60 GB | Weeks |
| 70-175B | DDP + FSDP + TP | 16-64 | 40-80 GB | Weeks-Months |
| 175B-1T | FSDP + TP + PP | 64-10K | 20-80 GB | Months |

### Key Optimizations:
1. ✅ Use FSDP for massive models
2. ✅ Enable gradient checkpointing
3. ✅ Use mixed precision (FP16)
4. ✅ Overlap computation & communication
5. ✅ Scale learning rate with batch size
6. ✅ Monitor GPU utilization (target: 70-80%)
7. ✅ Save/resume checkpoints frequently
8. ✅ Use profiler to identify bottlenecks

### Typical Speedup Breakdown (8x A100 cluster):
- **No optimization**: 1x (baseline)
- **DDP only**: 6.5x (with communication)
- **+ Mixed precision**: 13x
- **+ Gradient checkpointing**: 11x (slower but less memory)
- **+ FSDP + selective checkpointing**: 9-10x effective throughput
- **Combined optimization**: 4-8x sustained throughput (with all overhead)

This means training a 70B model that would take 6 months on 1 GPU takes ~2-3 weeks on 16 GPUs with proper optimization.
