"""
LLM models and utilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE)."""
    
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, t: int):
        """Generate rotary embedding for sequence position."""
        freqs = torch.einsum("i,j->ij", torch.arange(t, device=self.inv_freq.device).type_as(self.inv_freq), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]


class FlashAttention(nn.Module):
    """Optimized attention mechanism."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(context)
        
        return output


class MLP(nn.Module):
    """Feed-forward network with activation."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformer encoder block with pre-norm."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = FlashAttention(d_model, n_heads, dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Attention with residual and pre-norm\n        x = x + self.attn(self.norm1(x), mask)\n        # MLP with residual and pre-norm\n        x = x + self.mlp(self.norm2(x))\n        return x\n\n\nclass OptimizedLLM(nn.Module):\n    \"\"\"Optimized Language Model with all best practices.\"\"\"\n    \n    def __init__(\n        self,\n        vocab_size: int = 50257,\n        d_model: int = 768,\n        n_heads: int = 12,\n        n_layers: int = 12,\n        d_ff: int = 3072,\n        max_seq_len: int = 4096,\n        dropout: float = 0.1\n    ):\n        super().__init__()\n        self.vocab_size = vocab_size\n        self.d_model = d_model\n        \n        # Embeddings\n        self.token_embedding = nn.Embedding(vocab_size, d_model)\n        self.pos_embedding = nn.Embedding(max_seq_len, d_model)\n        \n        # Transformer layers\n        self.layers = nn.ModuleList([\n            TransformerBlock(d_model, n_heads, d_ff, dropout)\n            for _ in range(n_layers)\n        ])\n        \n        # Output\n        self.norm = nn.LayerNorm(d_model)\n        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)\n        \n        # Tie embeddings\n        self.lm_head.weight = self.token_embedding.weight\n    \n    def forward(\n        self,\n        input_ids: torch.Tensor,\n        use_checkpoint: bool = False,\n        use_cache: bool = False\n    ) -> torch.Tensor:\n        seq_len = input_ids.shape[1]\n        \n        # Embeddings\n        pos_ids = torch.arange(seq_len, device=input_ids.device)\n        x = self.token_embedding(input_ids) + self.pos_embedding(pos_ids)\n        \n        # Transformer layers\n        for layer in self.layers:\n            if use_checkpoint:\n                from torch.utils.checkpoint import checkpoint\n                x = checkpoint(layer, x, None, use_reentrant=False)\n            else:\n                x = layer(x, mask=None)\n        \n        # Output\n        x = self.norm(x)\n        logits = self.lm_head(x)\n        \n        return logits\n    \n    def generate(\n        self,\n        input_ids: torch.Tensor,\n        max_new_tokens: int = 50,\n        temperature: float = 1.0,\n        top_k: Optional[int] = None,\n        device: str = 'cuda'\n    ) -> torch.Tensor:\n        \"\"\"Generate tokens autoregressively.\"\"\"\n        self.eval()\n        with torch.no_grad():\n            for _ in range(max_new_tokens):\n                logits = self.forward(input_ids)\n                next_token_logits = logits[:, -1, :] / temperature\n                \n                if top_k is not None:\n                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]\n                    next_token_logits[indices_to_remove] = float('-inf')\n                \n                probs = F.softmax(next_token_logits, dim=-1)\n                next_token = torch.multinomial(probs, num_samples=1)\n                input_ids = torch.cat([input_ids, next_token], dim=1)\n        \n        return input_ids\n\n\ndef create_model_from_config(config: dict) -> OptimizedLLM:\n    \"\"\"Create model from config dictionary.\"\"\"\n    return OptimizedLLM(\n        vocab_size=config.get('vocab_size', 50257),\n        d_model=config.get('d_model', 768),\n        n_heads=config.get('n_heads', 12),\n        n_layers=config.get('n_layers', 12),\n        d_ff=config.get('d_ff', 3072),\n        max_seq_len=config.get('max_seq_len', 4096),\n        dropout=config.get('dropout', 0.1)\n    )\n