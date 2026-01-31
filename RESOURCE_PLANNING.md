# Resource Planning Guide for Large-Scale Model Training
## Complete Framework for Estimating GPU Requirements and Training Time

---

## Quick Reference: Resource Estimation

### 1. Model Parameters vs. Memory

```
Memory Requirement = (# Params) × (Bytes per Param + Optimizer Overhead)

For training a single model on one GPU:
- FP32 training:     6x model params (4B weights + 8B optimizer + 4B gradients)
- FP16 training:     4x model params (2B weights + 8B optimizer + 2B gradients)
- With checkpointing: 3x model params (reduced activations storage)

Examples:
- 7B model in FP32:  7B × 6 = 42 GB ❌ Won't fit in single A100 (80GB)
- 7B model in FP16:  7B × 4 = 28 GB ✓ Fits with 50GB buffer
- 70B model in FP16: 70B × 4 = 280 GB → Need 4x A100s minimum
- 175B model:        175B × 4 = 700 GB → Need 8x A100s minimum
```

### 2. GPU Selection by Model Size

```
Small Models (< 5B parameters):
├─ GPU: Single A100 40GB or RTX 4090
├─ Training: Days to weeks
├─ Cost: ~$1/hour for cloud GPU
└─ Recommendation: Single GPU training on M3 or cloud

Large Models (5-70B parameters):
├─ GPU: 4-16x A100 80GB
├─ Training: Weeks to 2 months
├─ Cost: $50-200/hour for cluster
├─ Communication: ✓ Important, needs DDP
└─ Recommendation: Cloud GPU cluster

Massive Models (70B-500B parameters):
├─ GPU: 32-128x A100/H100 80GB
├─ Training: 1-4 months
├─ Cost: $500-5000/hour
├─ Communication: CRITICAL, needs advanced strategies
└─ Recommendation: FSDP + Tensor Parallelism + custom kernels

Trillion Parameter Models:
├─ GPU: 256-2048x H100 80GB
├─ Training: 3-12 months
├─ Cost: $5000-50000/hour
├─ Communication: Dominant factor in training time
└─ Recommendation: Dedicated cluster with custom networking
```

---

## Detailed Calculation Framework

### Step 1: Estimate Model Parameters

```python
def estimate_transformer_params(config):
    """
    Estimate total parameters in transformer model
    """
    
    # Token embedding + position embedding
    embedding_params = config.vocab_size * config.hidden_dim + config.max_seq_len * config.hidden_dim
    
    # Transformer layers
    params_per_layer = (
        # Self-attention
        3 * config.hidden_dim * config.hidden_dim +  # Q, K, V projections
        config.hidden_dim * config.hidden_dim +      # Output projection
        
        # Feed-forward network
        config.hidden_dim * config.ffn_hidden +
        config.ffn_hidden * config.hidden_dim +
        
        # Layer norms (usually small, ~2 * hidden_dim per layer)
        2 * config.hidden_dim
    )
    
    transformer_params = config.num_layers * params_per_layer
    
    # Output layer (token prediction)
    output_params = config.hidden_dim * config.vocab_size
    
    total = embedding_params + transformer_params + output_params
    
    return {
        'embedding': embedding_params,
        'transformer': transformer_params,
        'output': output_params,
        'total': total,
    }

# Example: LLaMA-7B config
config_7b = {
    'vocab_size': 32000,
    'hidden_dim': 4096,
    'max_seq_len': 4096,
    'num_layers': 32,
    'ffn_hidden': 11008,
}

params = estimate_transformer_params(config_7b)
print(f"LLaMA-7B estimated params: {params['total']/1e9:.2f}B")
```

### Step 2: Calculate GPU Memory Required

```python
def calculate_gpu_memory_required(
    model_params,
    batch_size,
    seq_length,
    use_gradient_checkpointing=True,
    mixed_precision=True,
):
    """
    Calculate total GPU memory needed for training
    """
    
    # 1. Model weights
    dtype = 2 if mixed_precision else 4  # FP16 or FP32
    weights_memory = model_params * dtype  # Bytes
    
    # 2. Optimizer states (AdamW: momentum + variance, always FP32)
    optimizer_states = model_params * 8  # 2 states × 4 bytes
    
    # 3. Gradients (same size as weights)
    gradient_memory = model_params * dtype
    
    # 4. Activations (biggest variable)
    # Rule of thumb: ~4 * batch_size * seq_len * hidden_dim * num_layers * dtype
    # Hidden dim ≈ model_params^0.5 for transformer
    # Simplified: activations ≈ 0.3 × (model_params in GB) × batch_size × seq_len / 4096
    
    activation_memory = 0.3 * model_params * batch_size * seq_length / 4096 * dtype
    
    if use_gradient_checkpointing:
        activation_memory *= 0.15  # 85% reduction with checkpointing
    
    # Total
    total_memory_bytes = (
        weights_memory +
        optimizer_states +
        gradient_memory +
        activation_memory
    )
    
    total_memory_gb = total_memory_bytes / 1e9
    
    # Add 10% buffer
    total_with_buffer = total_memory_gb * 1.1
    
    return {
        'weights_gb': weights_memory / 1e9,
        'optimizer_gb': optimizer_states / 1e9,
        'gradients_gb': gradient_memory / 1e9,
        'activations_gb': activation_memory / 1e9,
        'total_gb': total_memory_gb,
        'with_10pct_buffer': total_with_buffer,
        'fits_in_40gb': total_with_buffer < 40,
        'fits_in_80gb': total_with_buffer < 80,
    }

# Example: 7B model on single GPU
mem = calculate_gpu_memory_required(
    model_params=7e9,
    batch_size=8,
    seq_length=2048,
    use_gradient_checkpointing=True,
    mixed_precision=True,
)

print(f"7B model + batch_size=8:")
print(f"  Weights:     {mem['weights_gb']:.1f} GB")
print(f"  Optimizer:   {mem['optimizer_gb']:.1f} GB")
print(f"  Gradients:   {mem['gradients_gb']:.1f} GB")
print(f"  Activations: {mem['activations_gb']:.1f} GB")
print(f"  Total:       {mem['total_gb']:.1f} GB")
print(f"  Fits in 80GB GPU: {mem['fits_in_80gb']}")
```

### Step 3: Calculate Training Time

```python
def estimate_training_time(
    model_params,
    total_training_tokens,
    num_gpus=1,
    fp16_enabled=True,
    efficiency_factor=0.65,  # Realistic: 60-70% of peak
):
    """
    Estimate total training time
    
    Calculation:
    1. FLOPs per token = 6 × model_params
       (Chinchilla scaling law: 3N × D forward + backward)
    2. Total FLOPs = FLOPs_per_token × training_tokens
    3. Peak FLOPs = GPU peak × num_gpus × (2x for tensor cores if FP16)
    4. Realistic FLOPs = Peak FLOPs × efficiency_factor
    5. Training time = Total FLOPs / Realistic FLOPs
    \"\"\"\n    \n    # FLOPs per token (from Chinchilla scaling)\n    flops_per_token = 6 * model_params\n    \n    # Total FLOPs needed\n    total_flops = flops_per_token * total_training_tokens\n    \n    # GPU peak performance\n    # A100 40GB: 312 TFLOPS (FP32), 625 TFLOPS (FP16)\n    # H100 80GB: 756 TFLOPS (FP32), 1513 TFLOPS (FP16)\n    # M3 Max: 1-2 TFLOPS (much slower!)\n    \n    gpu_peak_tflops_fp32 = {\n        'A100_40GB': 312,\n        'A100_80GB': 312,\n        'H100_80GB': 756,\n        'L40S': 362,\n        'RTX_4090': 165,\n        'M3_Max': 2,\n    }\n    \n    # Default: A100 80GB\n    peak_tflops = gpu_peak_tflops_fp32['A100_80GB']\n    if fp16_enabled:\n        peak_tflops *= 2  # FP16 is 2x faster on tensor cores\n    \n    # Total peak FLOPs across all GPUs\n    total_peak_flops_per_sec = peak_tflops * 1e12 * num_gpus\n    \n    # Realistic FLOPs accounting for communication overhead\n    realistic_flops_per_sec = total_peak_flops_per_sec * efficiency_factor\n    \n    # Training time\n    training_seconds = total_flops / realistic_flops_per_sec\n    training_hours = training_seconds / 3600\n    training_days = training_hours / 24\n    \n    # Cost (rough estimate: $1/hour per A100, higher for H100)\n    cost_per_hour = num_gpus * 1.0  # $1/hour per GPU\n    total_cost = training_hours * cost_per_hour\n    \n    return {\n        'seconds': training_seconds,\n        'hours': training_hours,\n        'days': training_days,\n        'cost': total_cost,\n        'flops_per_token': flops_per_token,\n        'total_flops': total_flops,\n    }\n\n# Example 1: Training 7B model on 1 trillion tokens\ntime_7b_8gpus = estimate_training_time(\n    model_params=7e9,\n    total_training_tokens=1e12,\n    num_gpus=8,\n    fp16_enabled=True,\n)\n\nprint(f\"7B model × 1T tokens on 8x A100:\")\nprint(f\"  Time: {time_7b_8gpus['days']:.1f} days ({time_7b_8gpus['hours']:.0f} hours)\")\nprint(f\"  Cost: ${time_7b_8gpus['cost']:,.0f}\")\n\n# Example 2: Training on M3 Mac (for comparison)\ntime_7b_m3 = estimate_training_time(\n    model_params=7e9,\n    total_training_tokens=1e9,  # Much smaller for M3\n    num_gpus=1,\n    fp16_enabled=True,\n)\n\nprint(f\"\\n7B model × 1B tokens on M3 Max:\")\nprint(f\"  Time: {time_7b_m3['hours']:.1f} hours\")\nprint(f\"  (Infeasible for production, but great for development!)\")\n```

### Step 4: Determine Optimal Number of GPUs

```python
def find_optimal_gpu_count(\n    model_params,\n    total_training_tokens,\n    max_budget_dollars=100000,\n    max_training_days=30,\n):\n    \"\"\"\n    Find optimal number of GPUs based on constraints\n    \"\"\"\n    \n    results = {}\n    \n    for num_gpus in [1, 2, 4, 8, 16, 32, 64, 128]:\n        time_estimate = estimate_training_time(\n            model_params=model_params,\n            total_training_tokens=total_training_tokens,\n            num_gpus=num_gpus,\n            fp16_enabled=True,\n        )\n        \n        results[num_gpus] = {\n            'time_days': time_estimate['days'],\n            'cost': time_estimate['cost'],\n            'meets_budget': time_estimate['cost'] <= max_budget_dollars,\n            'meets_deadline': time_estimate['days'] <= max_training_days,\n        }\n    \n    # Find best option\n    valid_options = [\n        (k, v) for k, v in results.items()\n        if v['meets_budget'] and v['meets_deadline']\n    ]\n    \n    if valid_options:\n        # Choose option with fewest GPUs (cost-efficient)\n        optimal = min(valid_options, key=lambda x: x[0])\n        print(f\"✓ Feasible with {optimal[0]} GPUs\")\n        print(f\"  Time: {optimal[1]['time_days']:.1f} days\")\n        print(f\"  Cost: ${optimal[1]['cost']:,.0f}\")\n    else:\n        print(f\"✗ No feasible configuration found\")\n        print(f\"  Budget: ${max_budget_dollars:,}\")\n        print(f\"  Deadline: {max_training_days} days\")\n        print(f\"  Need: 128 GPUs for {results[128]['time_days']:.1f} days, ${results[128]['cost']:,.0f}\")\n    \n    return results\n\n# Example: Find optimal GPUs for 70B model\nresults = find_optimal_gpu_count(\n    model_params=70e9,\n    total_training_tokens=1e12,\n    max_budget_dollars=50000,\n    max_training_days=30,\n)\n```

---

## M3 MacBook Specific Planning

### What's Feasible on M3?

```python
M3_FEASIBILITY = {\n    'M3_Pro_18GB': {\n        'max_model_params': 2e9,  # 2B parameters\n        'recommended_batch_size': 4,\n        'training_tokens_realistic': 1e9,  # 1 billion tokens, takes ~24 hours\n        'use_cases': [\n            '✓ Model development & testing',\n            '✓ Fine-tuning pre-trained models',\n            '✓ Quick experiments (< 24 hours)',\n            '✓ Inference & serving',\n        ],\n        'not_recommended': [\n            '✗ Training from scratch (LLaMA-7B)',\n            '✗ Production scale training',\n            '✗ Long-running experiments',\n        ]\n    },\n    'M3_Max_36GB': {\n        'max_model_params': 7e9,  # 7B parameters with aggressive optimization\n        'recommended_batch_size': 8,\n        'training_tokens_realistic': 10e9,  # 10B tokens, takes ~3-5 days\n        'use_cases': [\n            '✓ Fine-tune LLaMA-7B',\n            '✓ Multi-day experiments',\n            '✓ Instruction tuning',\n            '✓ Domain-specific training',\n        ],\n        'not_recommended': [\n            '✗ Training 70B+ from scratch',\n            '✗ Multi-week training jobs',\n        ]\n    },\n    'M3_Max_128GB': {\n        'max_model_params': 13e9,  # 13B with optimization\n        'recommended_batch_size': 16,\n        'training_tokens_realistic': 100e9,  # 100B tokens, takes ~2-3 weeks\n        'use_cases': [\n            '✓ Train custom 13B models',\n            '✓ Multi-week experiments',\n            '✓ Instruction tuning at scale',\n            '✓ Research with moderate data',\n        ],\n        'not_recommended': [\n            '✗ Production scale (100B+ tokens)',\n            '✗ Multi-month training',\n        ]\n    }\n}\n```

### M3 Training Duration Estimates

```python\ndef estimate_m3_training_time(\n    model_params,\n    total_tokens,\n    m3_variant='M3_Max_36GB',\n):\n    \"\"\"\n    Realistic training time estimation for M3 MacBook\n    M3 is MUCH slower than NVIDIA GPUs\n    \"\"\"\n    \n    # M3 throughput (tokens/second) from experience\n    m3_throughput = {\n        'M3_Pro': 50,      # 50 tokens/sec\n        'M3_Max_36GB': 100,  # 100 tokens/sec\n        'M3_Max_128GB': 150, # 150 tokens/sec\n    }\n    \n    throughput = m3_throughput.get(m3_variant, 100)\n    \n    total_seconds = total_tokens / throughput\n    total_hours = total_seconds / 3600\n    total_days = total_hours / 24\n    \n    # Account for thermal throttling (reduce by 10%)\n    total_days *= 1.1\n    \n    return {\n        'seconds': total_seconds,\n        'hours': total_hours,\n        'days': total_days,\n        'throughput': throughput,\n        'feasible': total_days < 30,  # Don't train for > month on M3\n    }\n\n# Example timings\nprint(\"M3 Training Time Estimates:\\n\")\n\ntests = [\n    ('SmallLLM (500M)', 500e6, 1e9),\n    ('LLaMA-7B (truncated)', 7e9, 10e9),  # Only feasible with aggressive optimization\n    ('Custom 3B model', 3e9, 5e9),\n]\n\nfor name, params, tokens in tests:\n    est = estimate_m3_training_time(params, tokens, 'M3_Max_36GB')\n    print(f\"{name}: {est['days']:.1f} days ({est['hours']:.0f} hours)\")\n    if est['feasible']:\n        print(f\"  ✓ Feasible on M3\")\n    else:\n        print(f\"  ✗ NOT recommended for M3 (too long)\")\n    print()\n```

---

## Hybrid Approach: M3 Development → Cloud Production

### Recommended Workflow

```
┌─────────────────────────────────────────────────────────────────┐\n│                    DEVELOPMENT WORKFLOW                         │\n└─────────────────────────────────────────────────────────────────┘\n\n1. PROTOTYPE (on M3, 1-2 hours)\n   ├─ Write model code\n   ├─ Test on small dataset (1000 examples)\n   ├─ Verify loss convergence\n   └─ Expected time: 30 min - 2 hours\n\n2. SCALE TO CLOUD (same code, no changes!)\n   ├─ Use torchrun for multi-GPU\n   ├─ Scale batch size: 4 → 32 per GPU\n   ├─ Launch on 8x A100 cluster\n   ├─ Expected time: Days instead of weeks\n   └─ Expected cost: $100-500\n\n3. PRODUCTION DEPLOYMENT\n   ├─ Option A: Quantize (8-bit, 4-bit)\n   │  └─ Deploy back to M3 for inference\n   ├─ Option B: Distill to smaller model\n   │  └─ Train student model on cloud\n   │  └─ Student runs efficiently on M3\n   └─ Option C: Use API service\n      └─ Call cloud model as service\n\n┌─────────────────────────────────────────────────────────────────┐\n│              COST & TIME COMPARISON                              │\n├─────────────────────────────────────────────────────────────────┤\n│                                                                   │\n│ SCENARIO 1: Train 70B model from scratch                        │\n│ ❌ M3 alone: ~3 years (infeasible)                               │\n│ ✅ Cloud 64 GPUs: 2-3 weeks, ~$100k                             │\n│                                                                   │\n│ SCENARIO 2: Fine-tune 7B model (10B tokens)                     │\n│ ✅ M3 alone: 3-5 days (feasible but slow)                       │\n│ ✅ Cloud 8 GPUs: 8-12 hours, $200-400                           │\n│ → M3 for quick experiments, cloud for production                │\n│                                                                   │\n│ SCENARIO 3: Quick experimentation                               │\n│ ✅ M3 alone: Instant (no waiting for cloud)                     │\n│ → M3 is perfect for iteration speed                             │\n│                                                                   │\n└─────────────────────────────────────────────────────────────────┘\n```

---

## Scaling Calculator

```python
def complete_scaling_calculator():\n    \"\"\"\n    Interactive calculator for training resource planning\n    \"\"\"\n    \n    import json\n    \n    calc = {\n        'step_1_model': {\n            'question': 'Model size?',\n            'options': ['small (1-5B)', 'large (5-70B)', 'massive (70B+)'],\n            'params': [1e9, 30e9, 175e9],\n        },\n        'step_2_data': {\n            'question': 'Training tokens?',\n            'options': ['small (1B)', 'medium (100B)', 'large (1T+)'],\n            'tokens': [1e9, 100e9, 1e12],\n        },\n        'step_3_budget': {\n            'question': 'Max budget?',\n            'options': ['low ($1k)', 'medium ($10k)', 'high ($100k+)'],\n            'budget': [1000, 10000, 100000],\n        },\n        'step_4_timeline': {\n            'question': 'Deadline?',\n            'options': ['fast (< 1 week)', 'medium (1-4 weeks)', 'flexible (any)'],\n            'days': [7, 28, 365],\n        },\n    }\n    \n    return calc\n\nprint(\"\\n\" + \"=\"*70)\nprint(\"RESOURCE PLANNING CALCULATOR\")\nprint(\"=\"*70)\n\ncalc = complete_scaling_calculator()\nfor step, info in calc.items():\n    print(f\"\\n{step}:\")\n    print(f\"  Q: {info['question']}\")\n    for i, opt in enumerate(info['options']):\n        print(f\"    {i+1}. {opt}\")\n```

---

## Summary Table

| Model | Tokens | Single GPU | 8x GPU | 16x GPU | Cost (8x) |\n|-------|--------|-----------|---------|----------|----------|\n| 7B | 100B | 15 days | 2 days | 1 day | $2K |\n| 7B | 1T | 150 days | 20 days | 10 days | $20K |\n| 70B | 100B | infeasible | 14 days | 7 days | $20K |\n| 70B | 1T | infeasible | 140 days | 70 days | $200K |\n| 175B | 1T | infeasible | ~1 year | 150 days | $500K |\n\n**M3 MacBook Column:**\n| Task | M3 Pro | M3 Max 36GB | M3 Max 128GB |\n|------|--------|-------------|-------------|\n| Dev & test | ✓ | ✓ | ✓ |\n| Fine-tune 7B | ✗ | ✓ (slow) | ✓ |\n| Train from scratch | ✗ | ✗ | ✗ (too slow) |\n| Inference | ✓ | ✓ | ✓ |\n\n---\n\n## Conclusion\n\n**Use M3 MacBook for:**\n- Model development & testing\n- Quick experiments (< 1 day)\n- Fine-tuning pre-trained models\n- Inference & serving\n\n**Use Cloud GPUs for:**\n- Training large models from scratch\n- Production scale (100B+ tokens)\n- When deadline matters\n- When resources are justified\n\n**Cost-Efficient Workflow:**\n1. Develop on free M3 (0 cost, instant)\n2. Scale to cloud when needed ($100-1000 per experiment)\n3. Deploy optimized model back to M3 for inference\n