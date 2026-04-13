import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
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
Giải thích các tham số
- dim (hidden size): kích thước của vector ẩn (embedding dimension)
- eps (epsilon): Một số cực nhỏ được cộng vào mẫu số để tránh lỗi chia cho 0 khi thực hiện phép chia
- weight: Tham số gamma cần học
"""