import torch
import torch.nn as nn


class GroupedMultiQueryAttention(nn.Module):
    """Grouped multi-query attention as used in Llama2.

    This module projects queries normally into n_heads, and projects keys
    and values into n_key_value_groups. Each key/value group is shared by a
    block of heads.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        max_batch_size: int,
        max_seq_len: int,
        rotary: nn.Module = None,
    ):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.heads_per_kv_group = n_heads // n_kv_heads

        self.q_proj = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(self.dim, 2 * self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)
        self.rotary = rotary

        # Tạo KV cache cho autoregressive decoding
        self.register_buffer(
            "cache_k",
            torch.zeros(max_batch_size, max_seq_len, self.n_kv_heads, self.head_dim)
        )

        self.register_buffer(
            "cache_v",
            torch.zeros(max_batch_size, max_seq_len, self.n_kv_heads, self.head_dim)
        )

    def _reshape_qkv(self, x: torch.Tensor, n_groups: int) -> torch.Tensor:
        """Reshape projected tensor into (B, seq_len, groups, head_dim)."""
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, n_groups, self.head_dim)
    
    def _inflate_kv(self, tensor: torch.Tensor) -> torch.Tensor:
        """Repeat grouped keys/values across heads in each group."""
        return tensor.repeat_interleave(self.heads_per_kv_group, dim=2)
    
    def _build_causal_mask(self, seq_len: int, total_len: int, device: torch.device) -> torch.Tensor:
        q_idx = torch.arange(total_len - seq_len, total_len, device=device).unsqueeze(1)
        k_idx = torch.arange(total_len, device=device).unsqueeze(0)
        return q_idx >= k_idx
    
    def forward(self, x: torch.Tensor, start_pos: int, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """Compute grouped multi-query attention.

        Args:
            x: input tensor of shape (B, seq_len, dim)
            start_pos: starting position for caching keys/values in autoregressive decoding
            attn_mask: optional attention mask broadcastable to (B, 1, seq_len, seq_len)
        Returns:
            output tensor of shape (B, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        q = self._reshape_qkv(q, self.n_heads)  # (B, seq_len, n_heads, head_dim)
        kv = self.kv_proj(x)
        kv = self._reshape_qkv(kv, 2 * self.n_kv_heads)  # (B, seq_len, 2*n_kv_heads, head_dim)
        k, v = kv.chunk(2, dim=2)  # each (B, seq_len, n_kv_heads, head_dim)
        if self.rotary is not None:
            q = self.rotary(q)
            k = self.rotary(k)

        # Lưu K, V vào cache (chưa inflate)
        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = k
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = v

        # Lấy toàn bộ lịch sử từ cache
        k = self.cache_k[:batch_size, : start_pos + seq_len]  # (B, total_seq_len, n_kv_heads, head_dim)
        v = self.cache_v[:batch_size, : start_pos + seq_len]  # (B, total_seq_len, n_kv_heads, head_dim)

        k = self._inflate_kv(k)  # (B, total_seq_len, n_heads, head_dim)
        v = self._inflate_kv(v)  # (B, total_seq_len, n_heads, head_dim)

        q = q.transpose(1, 2)  # (B, n_heads, seq_len, head_dim)
        k = k.transpose(1, 2)  # (B, n_heads, seq_len, head_dim)
        v = v.transpose(1, 2)  # (B, n_heads, seq_len, head_dim)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # (B, n_heads, seq_len, seq_len)
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float("-inf"))
        else:
            total_len = start_pos + seq_len
            causal_mask = self._build_causal_mask(seq_len, total_len, x.device)  # (seq_len, total_len)
            attn_scores = attn_scores.masked_fill(causal_mask == 0, float("-inf"))
        
        attn_probs = torch.softmax(attn_scores, dim=-1)  # (B, n_heads, seq_len, seq_len)
        attn_output = torch.matmul(attn_probs, v)  # (B, n_heads, seq_len, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_heads * self.head_dim)  # (B, seq_len, dim)
        output = self.out_proj(attn_output)  # (B, seq_len, dim)
        return output