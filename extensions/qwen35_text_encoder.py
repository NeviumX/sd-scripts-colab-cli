"""Qwen3.5 4B の Text Encoder。Anima-3.8B が中間層を使う。

comfyui-anima-3-8B の text_encoder を、ComfyUI に頼らない形へ書き直した。帰属は THIRD_PARTY_NOTICES.md にある。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


SELF_ATTENTION_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31}
HIDDEN_SIZE = 2560
INTERMEDIATE_SIZE = 9216
OUTPUT_DIM = 1024
VOCAB_SIZE = 248320
ROTARY_DIM = 64
ROPE_THETA = 10000000.0
PAD_TOKEN_ID = 151643


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(dtype=x.dtype)


class Qwen35RMSNorm(nn.Module):
    """checkpoint の重みは 1 からの差で持つ。"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * (1.0 + self.weight.float())).to(dtype=x.dtype)


class SSMBlock(nn.Module):
    """GatedDeltaNet の線形 attention。トークンを前から順に走査する。"""

    def __init__(self, hidden_size=HIDDEN_SIZE, n_groups=32, conv_kernel=4, norm_dim=128):
        super().__init__()
        self.num_v_heads = n_groups
        self.num_k_heads = 16
        self.head_k_dim = norm_dim
        self.head_v_dim = norm_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.in_proj_qkv = nn.Linear(hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.in_proj_a = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        self.in_proj_b = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        self.conv1d = nn.Conv1d(
            self.conv_dim, self.conv_dim, conv_kernel, groups=self.conv_dim, padding=conv_kernel - 1, bias=False
        )
        self.out_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.norm = RMSNorm(self.head_v_dim)
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        self.dt_bias = nn.Parameter(torch.zeros(self.num_v_heads))

    def _delta_rule_scan(self, query, key, value, g, beta):
        initial_dtype = query.dtype
        query = l2norm(query, dim=-1).transpose(1, 2).contiguous().float()
        key = l2norm(key, dim=-1).transpose(1, 2).contiguous().float()
        value = value.transpose(1, 2).contiguous().float()
        beta = beta.transpose(1, 2).contiguous().float()
        g = g.transpose(1, 2).contiguous().float()

        batch, heads, seq_len, k_dim = key.shape
        v_dim = value.shape[-1]
        query = query * (1.0 / (k_dim**0.5))
        state = torch.zeros(batch, heads, k_dim, v_dim, device=query.device, dtype=torch.float32)
        outputs = []
        for i in range(seq_len):
            q_t = query[:, :, i]
            k_t = key[:, :, i]
            v_t = value[:, :, i]
            g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
            beta_t = beta[:, :, i].unsqueeze(-1)
            state = state * g_t
            kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * beta_t
            state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            outputs.append((state * q_t.unsqueeze(-1)).sum(dim=-2))
        return torch.stack(outputs, dim=2).transpose(1, 2).contiguous().to(initial_dtype)

    def forward(self, hidden_states, attention_mask=None):
        if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
            hidden_states = hidden_states * attention_mask[:, :, None].to(dtype=hidden_states.dtype)

        batch, seq_len, _ = hidden_states.shape
        z = self.in_proj_z(hidden_states).reshape(batch, seq_len, self.num_v_heads, self.head_v_dim)
        mixed = self.in_proj_qkv(hidden_states).transpose(1, 2)
        mixed = F.silu(self.conv1d(mixed)[:, :, :seq_len]).transpose(1, 2)
        query, key, value = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch, seq_len, self.num_v_heads, self.head_v_dim)

        beta = self.in_proj_b(hidden_states).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(self.in_proj_a(hidden_states).float() + self.dt_bias.float())
        repeat = self.num_v_heads // self.num_k_heads
        if repeat > 1:
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)

        y = self._delta_rule_scan(query, key, value, g, beta)
        y = self.norm(y.reshape(-1, self.head_v_dim))
        y = y * F.silu(z.reshape(-1, self.head_v_dim).float()).to(y.dtype)
        return self.out_proj(y.reshape(batch, seq_len, self.value_dim))


def apply_rotary(x, cos, sin):
    rotary_dim = cos.shape[-1]
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    x1 = x_rot[..., : rotary_dim // 2]
    x2 = x_rot[..., rotary_dim // 2 :]
    embedded = (x_rot * cos) + (torch.cat((-x2, x1), dim=-1) * sin)
    return torch.cat((embedded, x_pass), dim=-1)


def precompute_rotary(head_dim, seq_len, theta=ROPE_THETA, device=None, dtype=None):
    inverse = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    freqs = torch.outer(torch.arange(seq_len, device=device, dtype=torch.float32), inverse)
    cos = freqs.cos().unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 2)
    sin = freqs.sin().unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 2)
    return cos.to(dtype), sin.to(dtype)


class GatedSelfAttention(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE, num_heads=16, num_kv_heads=4, head_dim=256):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.gqa_ratio = num_heads // num_kv_heads
        self.inner_dim = num_heads * head_dim
        self.q_proj = nn.Linear(hidden_size, 2 * self.inner_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(self.inner_dim, hidden_size, bias=False)
        self.q_norm = Qwen35RMSNorm(head_dim)
        self.k_norm = Qwen35RMSNorm(head_dim)

    def forward(self, hidden_states, attention_mask, rotary):
        batch, seq_len, _ = hidden_states.shape
        qg = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim * 2)
        q, gate = qg.chunk(2, dim=-1)
        gate = gate.reshape(batch, seq_len, self.inner_dim)
        k = self.k_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        q = apply_rotary(q, *rotary)
        k = apply_rotary(k, *rotary)
        k = k.repeat_interleave(self.gqa_ratio, dim=1)
        v = v.repeat_interleave(self.gqa_ratio, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask.to(dtype=q.dtype))
        out = out.transpose(1, 2).reshape(batch, seq_len, self.inner_dim)
        return self.o_proj(out * torch.sigmoid(gate))


class MLP(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE, intermediate_size=INTERMEDIATE_SIZE):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class HybridBlock(nn.Module):
    def __init__(self, use_ssm: bool, has_mlp: bool):
        super().__init__()
        self.use_ssm = use_ssm
        self.has_mlp = has_mlp
        self.input_layernorm = Qwen35RMSNorm(HIDDEN_SIZE)
        if use_ssm:
            self.linear_attn = SSMBlock()
        else:
            self.self_attn = GatedSelfAttention()
        if has_mlp:
            self.post_attention_layernorm = Qwen35RMSNorm(HIDDEN_SIZE)
            self.mlp = MLP()

    def forward(self, x, attention_bias, rotary, linear_mask):
        normalized = self.input_layernorm(x)
        if self.use_ssm:
            x = x + self.linear_attn(normalized, attention_mask=linear_mask)
        else:
            x = x + self.self_attn(normalized, attention_bias, rotary)
        if self.has_mlp:
            x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen35HybridModel(nn.Module):
    def __init__(self, num_layers: int = 32):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.layers = nn.ModuleList(
            [HybridBlock(i not in SELF_ATTENTION_LAYERS, i != 31) for i in range(num_layers)]
        )
        self.norm = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, OUTPUT_DIM, bias=True),
            Qwen35RMSNorm(OUTPUT_DIM),
            nn.SiLU(),
            nn.Linear(OUTPUT_DIM, OUTPUT_DIM, bias=True),
        )

    @property
    def device(self):
        return self.embed_tokens.weight.device

    def hidden_states(self, input_ids, attention_mask, layer_indices):
        """指定した層の出力を float で返す。パディングは右側に置く。走査は前からなので、実トークンの結果は変わらない。"""
        x = self.embed_tokens(input_ids)
        seq_len = x.shape[1]
        rotary = precompute_rotary(ROTARY_DIM, seq_len, device=x.device, dtype=x.dtype)
        fill = torch.finfo(x.dtype).min / 4
        causal = torch.empty(seq_len, seq_len, dtype=x.dtype, device=x.device).fill_(fill).triu_(1)
        padding = 1.0 - attention_mask.to(x.dtype).reshape(attention_mask.shape[0], 1, -1, seq_len).expand(
            attention_mask.shape[0], 1, seq_len, seq_len
        )
        attention_bias = causal + padding.masked_fill(padding.to(torch.bool), fill)

        states = {}
        for index, layer in enumerate(self.layers):
            x = layer(x, attention_bias, rotary, attention_mask)
            if index in layer_indices:
                states[index] = x.float()
        return [states[index] for index in layer_indices]


def load_qwen35(path: str, dtype: torch.dtype) -> Qwen35HybridModel:
    """Text Encoder のファイルを読む。fp8 の重みは dtype にそろえる。"""
    from accelerate import init_empty_weights
    from safetensors.torch import load_file

    with init_empty_weights():
        model = Qwen35HybridModel()
    state = {key: value.to(dtype) for key, value in load_file(path).items()}
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval().requires_grad_(False)
