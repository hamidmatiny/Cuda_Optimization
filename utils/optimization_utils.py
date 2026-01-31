"""
Optimization utilities for CUDA/MPS training.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import time
from torch.profiler import profile, record_function, ProfilerActivity


class PerformanceMonitor:
    """Monitor training performance metrics."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.metrics = {}
    
    def get_device_memory(self) -> Dict[str, float]:
        """Get device memory stats in GB."""
        if self.device == 'cuda':
            return {
                'allocated': torch.cuda.memory_allocated() / 1e9,
                'reserved': torch.cuda.memory_reserved() / 1e9,
                'max_allocated': torch.cuda.max_memory_allocated() / 1e9,
            }
        else:
            return {'note': 'Memory stats not available for this device'}
    
    def measure_throughput(self, func, num_iterations: int = 100) -> float:
        """Measure throughput in iterations/sec."""
        torch.cuda.synchronize() if self.device == 'cuda' else None
        torch.mps.synchronize() if self.device == 'mps' else None
        
        start = time.perf_counter()
        for _ in range(num_iterations):
            func()
        
        torch.cuda.synchronize() if self.device == 'cuda' else None
        torch.mps.synchronize() if self.device == 'mps' else None
        
        elapsed = time.perf_counter() - start
        return num_iterations / elapsed
    
    def profile_model(self, model: nn.Module, input_data: torch.Tensor) -> str:
        """Profile model and return summary."""
        activities = [ProfilerActivity.CPU]
        if self.device == 'cuda':
            activities.append(ProfilerActivity.CUDA)
        
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True
        ) as prof:
            with record_function("model_inference"):
                with torch.no_grad():
                    _ = model(input_data)
        
        return prof.key_averages().table(
            sort_by="cpu_time_total" if self.device != 'cuda' else "cuda_time_total",
            row_limit=20
        )


class OptimizationTracker:
    """Track optimization improvements."""
    
    def __init__(self):
        self.results = {}
    
    def add_result(self, name: str, time_taken: float, throughput: float, memory_used: float = 0):
        """Record optimization result."""
        self.results[name] = {
            'time': time_taken,
            'throughput': throughput,
            'memory': memory_used
        }
    
    def get_speedup(self, baseline_name: str, optimized_name: str) -> float:
        """Calculate speedup compared to baseline."""
        baseline_time = self.results[baseline_name]['time']
        optimized_time = self.results[optimized_name]['time']
        return baseline_time / optimized_time
    
    def print_summary(self):
        """Print comparison of all optimizations."""
        print("\n" + "="*80)
        print("OPTIMIZATION SUMMARY")
        print("="*80)
        print(f"{'Optimization':<30} {'Time (s)':<12} {'Throughput':<15} {'vs Baseline':<12}")
        print("-"*80)
        
        baseline_time = None
        for name, metrics in sorted(self.results.items()):
            if baseline_time is None:
                baseline_time = metrics['time']
                speedup_str = "1.0x"
            else:
                speedup = baseline_time / metrics['time']
                speedup_str = f"{speedup:.2f}x"
            
            print(f"{name:<30} {metrics['time']:<12.2f} {metrics['throughput']:<15.0f} {speedup_str:<12}")


def estimate_training_time(
    model_size_m: float,
    num_tokens: int,
    device: str = 'cuda',
    use_amp: bool = True,
    use_checkpoint: bool = True
) -> Dict[str, float]:
    """
    Estimate training time for a given model and dataset size.
    
    Args:
        model_size_m: Model size in millions of parameters
        num_tokens: Total tokens to train on
        device: 'cuda', 'mps', or 'cpu'
        use_amp: Use automatic mixed precision
        use_checkpoint: Use gradient checkpointing
    
    Returns:
        Dictionary with estimated times
    """
    # Rough estimates based on typical performance
    flops_per_token = 6 * model_size_m * 1e6  # FLOPs per token
    
    # Peak throughput estimates (FLOPs/sec)
    if device == 'cuda':
        peak_throughput = 5e12  # Conservative for typical GPU
    elif device == 'mps':
        peak_throughput = 1e12  # M3 MacBook
    else:
        peak_throughput = 1e11  # CPU
    
    # Apply speedup factors
    if use_amp:
        peak_throughput *= 2.0
    if use_checkpoint:
        peak_throughput *= 0.7  # 30% slower due to recomputation
    
    total_flops = flops_per_token * num_tokens
    training_time_seconds = total_flops / peak_throughput
    
    return {
        'seconds': training_time_seconds,
        'minutes': training_time_seconds / 60,
        'hours': training_time_seconds / 3600,
        'days': training_time_seconds / 86400
    }


class BatchSizeOptimizer:
    """Find optimal batch size for your hardware."""
    
    @staticmethod
    def find_optimal_batch_size(
        model: nn.Module,
        device: str,
        sequence_length: int = 256,
        batch_sizes: List[int] = None
    ) -> Tuple[int, float]:
        """
        Find batch size that maximizes throughput without OOM.
        
        Returns:
            (optimal_batch_size, throughput_samples_per_sec)
        """
        if batch_sizes is None:
            batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256]
        
        best_batch_size = 2
        best_throughput = 0
        
        for bs in batch_sizes:
            try:
                # Try a forward/backward pass
                input_ids = torch.randint(0, 10000, (bs, sequence_length)).to(device)
                labels = torch.randint(0, 10000, (bs, sequence_length)).to(device)
                
                torch.cuda.synchronize() if device == 'cuda' else None
                torch.mps.synchronize() if device == 'mps' else None
                
                start = time.perf_counter()
                
                output = model(input_ids)
                loss = output.sum()
                loss.backward()
                
                torch.cuda.synchronize() if device == 'cuda' else None
                torch.mps.synchronize() if device == 'mps' else None
                
                elapsed = time.perf_counter() - start
                throughput = bs / elapsed
                
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_batch_size = bs
                
                print(f"Batch size {bs:3d}: {throughput:7.0f} samples/sec")
                
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"Batch size {bs:3d}: OOM - stopping")
                    break
                raise
        
        return best_batch_size, best_throughput
