"""
Complete training pipeline with all optimizations.
Usage: python train.py --model-size small --device mps --use-amp --use-checkpoint
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, TensorDataset
import argparse
import time
from typing import Optional
import sys

# Add models to path
sys.path.insert(0, '/Users/hamidrezamatiny/Documents/GitHub/Cuda_Optimization')
from models.llm import OptimizedLLM


class TrainingConfig:
    """Configuration for training."""
    
    def __init__(self, args):
        self.device = args.device
        self.batch_size = args.batch_size
        self.learning_rate = args.lr
        self.num_epochs = args.epochs
        self.seq_length = args.seq_length
        self.use_amp = args.use_amp
        self.use_checkpoint = args.use_checkpoint
        self.use_compile = args.use_compile
        self.model_size = args.model_size
        self.num_workers = 0 if args.device == 'mps' else args.num_workers
    
    def get_model_config(self):
        """Get model config based on size."""
        configs = {
            'tiny': {'d_model': 256, 'n_heads': 4, 'n_layers': 2},
            'small': {'d_model': 384, 'n_heads': 6, 'n_layers': 6},
            'medium': {'d_model': 768, 'n_heads': 12, 'n_layers': 12},
            'large': {'d_model': 1024, 'n_heads': 16, 'n_layers': 24},
        }\
        return configs.get(self.model_size, configs['small'])


class Trainer:
    \"\"\"Training loop with optimizations.\"\"\"
    \n    def __init__(self, config: TrainingConfig):\n        self.config = config\n        self.device = torch.device(config.device)\n        \n        # Model\n        model_config = config.get_model_config()\n        self.model = OptimizedLLM(\n            vocab_size=50257,\n            d_model=model_config['d_model'],\n            n_heads=model_config['n_heads'],\n            n_layers=model_config['n_layers'],\n            d_ff=model_config['d_model'] * 4,\n        ).to(self.device)\n        \n        # Compile if requested\n        if config.use_compile and hasattr(torch, 'compile'):\n            try:\n                self.model = torch.compile(self.model, mode='reduce-overhead')\n                print(\"✓ Model compiled with torch.compile()\")\n            except Exception as e:\n                print(f\"⚠ Compilation failed: {e}\")\n        \n        # Optimizer\n        self.optimizer = torch.optim.AdamW(\n            self.model.parameters(),\n            lr=config.learning_rate,\n            betas=(0.9, 0.95),\n            eps=1e-8\n        )\n        \n        # Scaler for AMP\n        self.scaler = GradScaler(device=config.device) if config.use_amp else None\n        \n        # Stats\n        self.train_losses = []\n        self.throughputs = []\n    \n    def create_dataloader(self):\n        \"\"\"Create dummy dataloader.\"\"\"\n        num_samples = 1000\n        input_ids = torch.randint(0, 50257, (num_samples, self.config.seq_length))\n        labels = torch.randint(0, 50257, (num_samples, self.config.seq_length))\n        \n        dataset = TensorDataset(input_ids, labels)\n        return DataLoader(\n            dataset,\n            batch_size=self.config.batch_size,\n            shuffle=True,\n            num_workers=self.config.num_workers,\n            pin_memory=self.config.device == 'cuda'\n        )\n    \n    def train_epoch(self, dataloader):\n        \"\"\"Train one epoch.\"\"\"\n        self.model.train()\n        total_loss = 0\n        num_batches = 0\n        \n        torch.cuda.synchronize() if self.config.device == 'cuda' else None\n        torch.mps.synchronize() if self.config.device == 'mps' else None\n        epoch_start = time.perf_counter()\n        \n        for batch_idx, (input_ids, labels) in enumerate(dataloader):\n            input_ids = input_ids.to(self.device, non_blocking=self.config.device == 'cuda')\n            labels = labels.to(self.device, non_blocking=self.config.device == 'cuda')\n            \n            self.optimizer.zero_grad()\n            \n            if self.config.use_amp:\n                with autocast(device_type=self.config.device, dtype=torch.float16):\n                    logits = self.model(input_ids, use_checkpoint=self.config.use_checkpoint)\n                    loss = F.cross_entropy(logits.view(-1, 50257), labels.view(-1))\n                \n                self.scaler.scale(loss).backward()\n                self.scaler.step(self.optimizer)\n                self.scaler.update()\n            else:\n                logits = self.model(input_ids, use_checkpoint=self.config.use_checkpoint)\n                loss = F.cross_entropy(logits.view(-1, 50257), labels.view(-1))\n                loss.backward()\n                self.optimizer.step()\n            \n            total_loss += loss.item()\n            num_batches += 1\n            \n            if (batch_idx + 1) % 10 == 0:\n                print(f\"  Batch {batch_idx + 1}: Loss = {loss.item():.4f}\")\n        \n        torch.cuda.synchronize() if self.config.device == 'cuda' else None\n        torch.mps.synchronize() if self.config.device == 'mps' else None\n        epoch_time = time.perf_counter() - epoch_start\n        \n        avg_loss = total_loss / num_batches\n        throughput = (num_batches * self.config.batch_size) / epoch_time\n        \n        self.train_losses.append(avg_loss)\n        self.throughputs.append(throughput)\n        \n        return avg_loss, throughput, epoch_time\n    \n    def train(self):\n        \"\"\"Full training loop.\"\"\"\n        print(f\"\"\"\n╔════════════════════════════════════════════════════════╗\n║            Training Configuration                       ║\n╠════════════════════════════════════════════════════════╣\n║ Device:              {self.config.device:<27} ║\n║ Model Size:          {self.config.model_size:<27} ║\n║ Batch Size:          {self.config.batch_size:<27} ║\n║ Sequence Length:     {self.config.seq_length:<27} ║\n║ AMP:                 {'Enabled' if self.config.use_amp else 'Disabled':<27} ║\n║ Gradient Checkpoint: {'Enabled' if self.config.use_checkpoint else 'Disabled':<27} ║\n║ torch.compile:       {'Enabled' if self.config.use_compile else 'Disabled':<27} ║\n║ Epochs:              {self.config.num_epochs:<27} ║\n╚════════════════════════════════════════════════════════╝\n        \"\"\")\n        \n        dataloader = self.create_dataloader()\n        model_params = sum(p.numel() for p in self.model.parameters())\n        print(f\"Model Parameters: {model_params / 1e6:.1f}M\")\n        print(f\"Model Size (FP32): {model_params * 4 / 1e9:.2f} GB\\n\")\n        \n        for epoch in range(self.config.num_epochs):\n            print(f\"Epoch {epoch + 1}/{self.config.num_epochs}\")\n            avg_loss, throughput, epoch_time = self.train_epoch(dataloader)\n            \n            print(f\"  Loss: {avg_loss:.4f}\")\n            print(f\"  Throughput: {throughput:.0f} samples/sec\")\n            print(f\"  Time: {epoch_time:.2f}s\\n\")\n        \n        print(\"\\nTraining Complete!\")\n        print(f\"Final Loss: {self.train_losses[-1]:.4f}\")\n        print(f\"Avg Throughput: {sum(self.throughputs) / len(self.throughputs):.0f} samples/sec\")\n\n\ndef main():\n    parser = argparse.ArgumentParser(description='Optimized LLM Training')\n    parser.add_argument('--device', choices=['cuda', 'mps', 'cpu'], default='mps',\n                        help='Device to use')\n    parser.add_argument('--model-size', choices=['tiny', 'small', 'medium', 'large'],\n                        default='small', help='Model size')\n    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')\n    parser.add_argument('--seq-length', type=int, default=256, help='Sequence length')\n    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')\n    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')\n    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader workers')\n    parser.add_argument('--use-amp', action='store_true', help='Use AMP')\n    parser.add_argument('--use-checkpoint', action='store_true', help='Use gradient checkpointing')\n    parser.add_argument('--use-compile', action='store_true', help='Use torch.compile')\n    \n    args = parser.parse_args()\n    \n    # Set device\n    if args.device == 'cuda' and not torch.cuda.is_available():\n        print(\"CUDA not available, using CPU\")\n        args.device = 'cpu'\n    elif args.device == 'mps' and not torch.backends.mps.is_available():\n        print(\"MPS not available, using CPU\")\n        args.device = 'cpu'\n    \n    config = TrainingConfig(args)\n    trainer = Trainer(config)\n    trainer.train()\n\n\nif __name__ == '__main__':\n    main()\n