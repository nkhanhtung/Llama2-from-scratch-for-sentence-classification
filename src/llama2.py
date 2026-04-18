import torch
import torch.nn as nn
from llama2_components import ModelArgs, GroupedMultiQueryAttention, FeedForward, RMSNorm, RotaryEncodings


"""
Phần implementation cho một Llama2 block, bao gồm GMQA và FFN
"""
class Llama2Block(nn.Module):
    def __init__(self, args: ModelArgs, rotary: nn.Module = None):
        super().__init__()
        self.attn = GroupedMultiQueryAttention(args, rotary)
        self.ffn = FeedForward(args)
        # Chuẩn hóa tiền attention và tiền ffn
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        # Attention block
        x_attn = self.attn_norm(x)
        x_attn = self.attn(x_attn, start_pos)
        x = x + x_attn  # Residual connection

        # FFN block
        x_ffn = self.ffn_norm(x)
        x_ffn = self.ffn(x_ffn)
        x = x + x_ffn  # Residual connection
        return x


"""
Phần implementation cho lớp Llama2Model, bao gồm nhiều block và embedding layers
"""
class Llama2Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.vocab_size > 0, "Vocab size must be set in ModelArgs"
        self.args = args
        self.vocab_size = args.vocab_size
        if args.n_kv_heads is None:
            args.n_kv_heads = args.n_heads
        self.token_embedding = nn.Embedding(self.vocab_size, args.dim)
        # Rotary embeddings
        self.rotary = RotaryEncodings(args)
        # Tạo các block
        self.blocks = nn.ModuleList([Llama2Block(args, self.rotary) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, args.norm_eps)
        # Output projection 
        self.output_proj = nn.Linear(args.dim, self.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor, start_pos: int) -> torch.Tensor:
        batch_size, seq_len = tokens.size()
        x = self.token_embedding(tokens)  # (batch_size, seq_len, dim)
        for block in self.blocks:
            x = block(x, start_pos=start_pos)
        x = self.norm(x)
        logits = self.output_proj(x)  # (batch_size, seq_len, vocab_size)
        return logits