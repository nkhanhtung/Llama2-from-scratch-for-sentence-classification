import torch
import torch.nn as nn


def precompute_frequencies(head_dim: int, seq_len: int, device: str, base: float = 10000.0) -> torch.Tensor:
    """
    Tính các tần số cho RoPE
    Tham số:
        head_dim: Số chiều của mỗi attention head (phải chia hết cho 2)
        seq_len:  Độ dài tối đa của sequence
        device:   CPU hoặc CUDA
        base:     Tham số mặc định = 10000 trong công thức tính theta_i
    """
    assert head_dim % 2 == 0, "head_dim phải chia hết cho 2"

    """
    Công thức tính tần số cơ bản: theta_i = base^(-2i / head_dim), i = {0, 1, ..., head_dim/2 - 1}
    Shape: (head_dim/2,)
    """
    i = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    theta = 1.0 / (base ** (i / head_dim))

    """
    Dãy vị trí m = 0, 1, ..., seq_len - 1
    Shape: (seq_len,)
    """
    m = torch.arange(seq_len, dtype=torch.float32, device=device)

    """
    Tính ma trận góc xoay m * theta_i (outer product)
    Shape: (seq_len, head_dim/2)
    """
    freqs = torch.outer(m, theta)

    """
    Chuyển sang số phức dạng polar: e^(j * freq) = cos(freq) + j*sin(freq)
    Shape: (seq_len, head_dim/2)
    """
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str) -> torch.Tensor:
    """
    Áp dụng RoPE lên tensor Q hoặc K

    Tham số:
        x: Tensor shape (B, seq_len, n_heads, head_dim)
        freqs_complex: Output của precompute_frequencies, shape (seq_len, head_dim/2)
        device: CPU hoặc CUDA

    Return:
        Tensor cùng shape với x, đã được xoay
    """
    # Code chính
    """
    Ghép từng cặp chiều liên tiếp thành số phức
    (B, seq_len, n_heads, head_dim) -> (B, seq_len, n_heads, head_dim/2) [complex]
    """
    x_complex = torch.view_as_complex(
        x.float().reshape(*x.shape[:-1], -1, 2)
    )

    """
    Broadcast freqs_complex để match shape của x_complex
    (seq_len, head_dim/2) -> (1, seq_len, 1, head_dim/2)
    """
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

    """
    Nhân số phức -> xoay trong không gian 2D
    (B, seq_len, n_heads, head_dim/2) * (1, seq_len, 1, head_dim/2)
    """
    x_rotated = x_complex * freqs_complex

    """
    Chuyển về sô thực rồi reshape lại shape ban đầu
    (B, seq_len, n_heads, head_dim/2) [complex] -> (B, seq_len, n_heads, head_dim/2, 2) [real] -> (B, seq_len, n_heads, head_dim)
    """
    x_out = torch.view_as_real(x_rotated).reshape(*x.shape)
    # Giữ nguyên dtype gốc và đưa về đúng device
    return x_out.type_as(x).to(device)


class RotaryEncodings(nn.Module):
    """
    Module RoPE (Rotary Position Embedding)
    Cách dùng: 
        rope = RotaryEncodings(head_dim=64, seq_len=2048, device='cuda')
        q = rope(q)  # shape: (B, seq_len, n_heads, head_dim)
        k = rope(k)
    """
    def __init__(self, head_dim: int, seq_len: int, device: str, base: float = 10000.0):
        super().__init__()
        self.device = device
        # Tính frequencies và lưu vào buffer
        freqs_complex = precompute_frequencies(head_dim, seq_len, device, base)
        self.register_buffer("freqs_complex", freqs_complex, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor shape (B, seq_len, n_heads, head_dim)
        Return:
            Tensor cùng shape, đã áp dụng RoPE
        """
        seq_len = x.shape[1]
        # Chỉ lấy phần frequencies cần thiết (hỗ trợ sequence ngắn hơn max)
        return apply_rotary_embeddings(x, self.freqs_complex[:seq_len], self.device)