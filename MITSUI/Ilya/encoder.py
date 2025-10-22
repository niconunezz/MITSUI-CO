import torch
import torch.nn.functional as F
from torch import nn

# Hyperparameters
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape , weight=self.weight, bias=self.bias, eps=1e-5)

class AttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.qkv = nn.Linear(config.ilya.encoder.n_embd, config.ilya.encoder.n_embd * 3)
        self.proj = nn.Linear(config.ilya.encoder.n_embd, config.ilya.encoder.n_embd)
        self.n_heads = config.ilya.encoder.n_heads

    def forward(self, x):
        B, T, C = x.shape
        # chunk is the same as split, but by definition just "attempts" to split the tensor.
        qkv = self.qkv(x).chunk(3, dim=-1)
        # divide in heads and transpose to get the right shape, making each head work as a batch that multiplies parallelly
        q, k, v = map(lambda t: t.view(B, T, self.n_heads, C//self.n_heads).transpose(1,2), qkv) # (B, n_heads, T, hs)

        y = F.scaled_dot_product_attention(q, k, v)
        # recombine heads and project out
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(y)

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.ilya.encoder.n_embd, 4 * config.ilya.encoder.n_embd),
            nn.ReLU()
        )
        self.c_proj = nn.Linear(4 * config.ilya.encoder.n_embd, config.ilya.encoder.n_embd)
        self.c_proj.NANO_GPT_SCALE_INIT = 1


    def forward(self, x):
        return self.c_proj(self.net(x))

class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.ilya.encoder.n_embd, True)
        self.ln2 = LayerNorm(config.ilya.encoder.n_embd, True)
        self.attn = AttentionHead(config)
        self.ff = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


                