import copy

import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, embed_dim, heads = 8, dropout = 0.1):
        super().__init__()
        dim_head = embed_dim // heads
        assert dim_head * heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {heads}"
        project_out = not (heads == 1)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(embed_dim, embed_dim * 3)

        self.to_out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.atten_drop = nn.Dropout(dropout)

    def forward(self, x, atten_mask=None):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        A_raw = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if atten_mask is not None:
            A_raw = A_raw + atten_mask
        A = self.attend(A_raw)
        A_ = self.atten_drop(A)

        out = torch.matmul(A_, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), A, A_raw

class TransformerLayer(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout = 0.1, pre_norm=True):
        super().__init__()

        self.self_attn = Attention(dim, heads=heads, dropout=dropout)
        self.ffn = FeedForward(dim, mlp_dim, dropout = dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.pre_norm = pre_norm

    def forward(self, x, atten_mask=None):
        if self.pre_norm:
            x_, A, A_raw = self.self_attn(self.norm1(x), atten_mask)
            x = x_ + x
            x = self.ffn(self.norm2(x)) + x
        else:
            x_, A, A_raw = self.self_attn(x, atten_mask)
            x = self.norm1(x_ + x)
            x = self.norm2(self.ffn(x) + x)

        return x, A, A_raw

class Transformer(nn.Module):
    def __init__(self, layer, depth, norm=None):
        super().__init__()
        self.layers = _get_clones(layer, depth)
        self.norm = norm
    def forward(self, x, atten_mask=None):

        for layer in self.layers[:-1]:
            x, _, _ = layer(x, atten_mask)
        output, A, A_raw = self.layers[-1](x, atten_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output, A, A_raw


class BalancedSA(nn.Module):
    def __init__(self, embed_dim, num_patch, num_table, heads = 8, dropout = 0.1):
        super().__init__()
        dim_head = embed_dim // heads
        assert dim_head * heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {heads}"
        project_out = not (heads == 1)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(embed_dim, embed_dim * 3)

        self.to_out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.atten_drop = nn.Dropout(dropout)
        self.num_patch = num_patch
        self.num_table = num_table

    def attention(self, A_raw):
        A_raw_patch = A_raw[:, :, :, :self.num_patch]
        A_raw_table = A_raw[:, :, :, self.num_patch:self.num_patch+self.num_table]
        A_raw_cls = A_raw[:, :, :, self.num_patch+self.num_table:]
        A_raw_patch_mean = A_raw_patch.mean(-1, keepdim=True)
        A_raw_table_mean = A_raw_table.mean(-1, keepdim=True)
        A_raw_cls_mean = A_raw_cls.max(-1, keepdim=True)[0]

        A_patch_mean_exp = torch.exp(A_raw_patch_mean)
        A_table_mean_exp = torch.exp(A_raw_table_mean)
        A_cls_mean_exp = torch.exp(A_raw_cls_mean)
        A_exp_sum = A_patch_mean_exp + A_table_mean_exp + A_cls_mean_exp
        A_patch_mean = A_patch_mean_exp / A_exp_sum
        A_table_mean = A_table_mean_exp / A_exp_sum
        A_cls_mean = A_cls_mean_exp / A_exp_sum
        A_patch = torch.exp(A_raw_patch)/(torch.exp(A_raw_patch).sum(-1, keepdim=True)+1e-6) * A_patch_mean
        A_table = torch.exp(A_raw_table)/(torch.exp(A_raw_table).sum(-1, keepdim=True)+1e-6) * A_table_mean
        A_cls = torch.exp(A_raw_cls)/(torch.exp(A_raw_cls).sum(-1, keepdim=True)+1e-6) * A_cls_mean

        return torch.cat((A_patch, A_table, A_cls), dim=-1)


    def forward(self, x, atten_mask=None):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        A_raw = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if atten_mask is not None:
            A_raw = A_raw + atten_mask
        A = self.attention(A_raw)
        A_ = self.atten_drop(A)

        out = torch.matmul(A_, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), A, A_raw

class TransformerLayer_Balanced(nn.Module):
    def __init__(self, dim, heads, mlp_dim, num_patch, num_table, dropout = 0.1, pre_norm=True):
        super().__init__()

        self.self_attn = BalancedSA(dim, num_patch, num_table, heads=heads, dropout=dropout)
        self.ffn = FeedForward(dim, mlp_dim, dropout = dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.pre_norm = pre_norm

    def forward(self, x, atten_mask=None):
        if self.pre_norm:
            x_, A, A_raw = self.self_attn(self.norm1(x), atten_mask)
            x = x_ + x
            x = self.ffn(self.norm2(x)) + x
        else:
            x_, A, A_raw = self.self_attn(x, atten_mask)
            x = self.norm1(x_ + x)
            x = self.norm2(self.ffn(x) + x)

        return x, A, A_raw
