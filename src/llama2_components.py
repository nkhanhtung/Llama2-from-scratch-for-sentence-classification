from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = None
    vocab_size: int = -1        # Đặt sau khi biết vocab size thực tế
    multiple_of: int = 256
    ffn_dim_multiplier: float = None
    norm_eps: float = 1e-5
    rope_base: float = 10000.0
    # Cần thiết cho cache KV
    max_batch_size: int = 32
    max_seq_len: int = 2048
    device: str = None


"""
Phần implementation cho Root Mean Square Layer Normalization (RMSNorm)
"""
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        # The gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Giữ nguyên kiểu dữ liệu đầu vào (ví dụ float16)
        input_dtype = x.dtype
        # Tính toán RMS nên thực hiện ở float32 để chính xác hơn
        x = x.to(torch.float32)
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        output = output * self.weight
        return output.to(input_dtype)
    

"""
Phần implementation cho Rotary Position Embeddings (RoPE)
"""
def precompute_frequencies(head_dim: int, seq_len: int, device: str, base: float = 10000.0) -> torch.Tensor:

    # Tính các tần số cho RoPE
    # Tham số:
    #     head_dim: Số chiều của mỗi attention head (phải chia hết cho 2)
    #     seq_len:  Độ dài tối đa của sequence
    #     device:   CPU hoặc CUDA
    #     base:     Tham số mặc định = 10000 trong công thức tính theta_i
    assert head_dim % 2 == 0, "head_dim phải chia hết cho 2"

    # Công thức tính tần số cơ bản: theta_i = base^(-2i / head_dim), i = {0, 1, ..., head_dim/2 - 1}
    # Shape: (head_dim/2,)
    i = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    theta = 1.0 / (base ** (i / head_dim))

    # Dãy vị trí m = 0, 1, ..., seq_len - 1
    # Shape: (seq_len,)
    m = torch.arange(seq_len, dtype=torch.float32, device=device)

    # Tính ma trận góc xoay m * theta_i (outer product)
    # Shape: (seq_len, head_dim/2)
    freqs = torch.outer(m, theta)

    # Chuyển sang số phức dạng polar: e^(j * freq) = cos(freq) + j*sin(freq)
    # Shape: (seq_len, head_dim/2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str) -> torch.Tensor:
    # Áp dụng RoPE lên tensor Q hoặc K
    # Tham số:
    #     x: Tensor shape (B, seq_len, n_heads, head_dim)
    #     freqs_complex: Output của precompute_frequencies, shape (seq_len, head_dim/2)
    #     device: CPU hoặc CUDA
    # Return:
    #     Tensor cùng shape với x, đã được xoay

    # Ghép từng cặp chiều liên tiếp thành số phức
    # (B, seq_len, n_heads, head_dim) -> (B, seq_len, n_heads, head_dim/2) [complex]
    x_complex = torch.view_as_complex(
        x.float().reshape(*x.shape[:-1], -1, 2)
    )

    # Broadcast freqs_complex để match shape của x_complex
    # (seq_len, head_dim/2) -> (1, seq_len, 1, head_dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

    # Nhân số phức -> xoay trong không gian 2D
    # (B, seq_len, n_heads, head_dim/2) * (1, seq_len, 1, head_dim/2)
    x_rotated = x_complex * freqs_complex

    # Chuyển về sô thực rồi reshape lại shape ban đầu
    # (B, seq_len, n_heads, head_dim/2) [complex] -> (B, seq_len, n_heads, head_dim/2, 2) [real] -> (B, seq_len, n_heads, head_dim)
    x_out = torch.view_as_real(x_rotated).reshape(*x.shape)
    # Giữ nguyên dtype gốc và đưa về đúng device
    return x_out.type_as(x).to(device)


class RotaryEncodings(nn.Module):
    # Module RoPE (Rotary Position Embedding)
    # Cách dùng: 
    #     rope = RotaryEncodings(head_dim=64, seq_len=2048, device='cuda')
    #     q = rope(q, start_pos=0)  # shape: (B, seq_len, n_heads, head_dim)
    #     k = rope(k, start_pos=0)  # shape: (B, seq_len, n_kv_heads, head_dim)

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.device = args.device
        # head_dim của mỗi attention head
        head_dim = args.dim // args.n_heads
        seq_len = args.max_seq_len
        base = args.rope_base
        # Tính frequencies và lưu vào buffer
        freqs_complex = precompute_frequencies(head_dim=head_dim, seq_len=seq_len, device=self.device, base=base)
        self.register_buffer("freqs_complex", freqs_complex, persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        # Args:
        #     x: Tensor shape (B, seq_len, n_heads, head_dim)
        # Return:
        #     Tensor cùng shape, đã áp dụng RoPE
        seq_len = x.shape[1]
        # Chỉ lấy phần frequencies cần thiết (hỗ trợ sequence ngắn hơn max)
        return apply_rotary_embeddings(x, self.freqs_complex[start_pos:start_pos + seq_len], self.device)
    

"""
Phần implementation cho Grouped Multi-Query Attention (GMQA)
"""
class GroupedMultiQueryAttention(nn.Module):
    # Grouped multi-query attention as used in Llama2.
    # This module projects queries normally into n_heads, and projects keys
    # and values into n_key_value_groups. Each key/value group is shared by a
    # block of heads.

    def __init__(self, args: ModelArgs, rotary: nn.Module = None):
        super().__init__()
        assert args.dim % args.n_heads == 0, "dim must be divisible by n_heads"
        assert args.n_heads % args.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.heads_per_kv_group = args.n_heads // args.n_kv_heads
        self.max_batch_size = args.max_batch_size
        self.max_seq_len = args.max_seq_len

        self.q_proj = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(self.dim, 2 * self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)
        self.rotary = rotary

        # Tạo KV cache cho autoregressive decoding
        self.register_buffer(
            "cache_k",
            torch.zeros(self.max_batch_size, self.max_seq_len, self.n_kv_heads, self.head_dim, device=args.device)
        )

        self.register_buffer(
            "cache_v",
            torch.zeros(self.max_batch_size, self.max_seq_len, self.n_kv_heads, self.head_dim, device=args.device)
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
    
    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
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
            q = self.rotary(q, start_pos=start_pos)
            k = self.rotary(k, start_pos=start_pos)

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

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5) 
        total_len = start_pos + seq_len
        causal_mask = self._build_causal_mask(seq_len, total_len, x.device)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, total_len)
        attn_scores = attn_scores.masked_fill(~causal_mask, float("-inf"))
        
        attn_probs = torch.softmax(attn_scores, dim=-1)  # (B, n_heads, seq_len, seq_len)
        attn_output = torch.matmul(attn_probs, v)  # (B, n_heads, seq_len, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_heads * self.head_dim)  # (B, seq_len, dim)
        output = self.out_proj(attn_output)  # (B, seq_len, dim)
        return output
    

"""
Phần implementation cho Feed Forward Network (FFN)
"""
class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.dim
        hidden_dim = 4 * dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(hidden_dim * args.ffn_dim_multiplier)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, dim) -> (B, seq_len, hidden_dim)
        swish = F.silu(self.w1(x))
        # x: (B, seq_len, dim) -> (B, seq_len, hidden_dim)
        x_V = self.w3(x)
        # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) -> (B, Seq_Len, Hidden_Dim)
        x = swish * x_V
        # (B, seq_len, hidden_dim) -> (B, seq_len, dim)
        x = self.w2(x)
        return x