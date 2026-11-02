"""Anima-3.8B v1.1 の Semantic Connector v2。Qwen3.5 の中間層を timestep ごとに llm_adapter の文脈へ混ぜる。

comfyui-anima-3-8B の semantic_connector_v2.py と progressive_cross_adapter.py を sd-scripts のクラスで書き直した。
学習はしないので、推論の経路だけを持つ。帰属は THIRD_PARTY_NOTICES.md にある。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from library.anima_models import LLMAdapterAttention, LLMAdapterRMSNorm


LAYER_INDICES = (7, 15, 23, 31)
QWEN35_DIM = 2560
ADAPTER_DIM = 1024
ADAPTER_HEADS = 16
ADAPTER_BLOCKS = 6


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half, 1)
    )
    angles = timesteps.float().reshape(-1, 1) * 1_000.0 * frequencies.reshape(1, -1)
    embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
    return F.pad(embedding, (0, 1)) if dim % 2 else embedding


class MultiHeadAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.query_dim = query_dim
        self.q_proj = nn.Linear(query_dim, query_dim, bias=False)
        self.k_proj = nn.Linear(context_dim, query_dim, bias=False)
        self.v_proj = nn.Linear(context_dim, query_dim, bias=False)
        self.o_proj = nn.Linear(query_dim, query_dim, bias=False)

    def forward(self, query, context, context_mask=None):
        batch, query_tokens, _ = query.shape
        context_tokens = context.shape[1]

        def heads(value, tokens):
            return value.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

        q = heads(self.q_proj(query), query_tokens)
        k = heads(self.k_proj(context), context_tokens)
        v = heads(self.v_proj(context), context_tokens)
        mask = None
        if context_mask is not None:
            mask = context_mask.to(torch.bool).reshape(batch, 1, 1, context_tokens)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o_proj(attended.transpose(1, 2).reshape(batch, query_tokens, self.query_dim))


class TimestepModulatedNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, value, scale, shift):
        return self.norm(value) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SemanticResamplerBlock(nn.Module):
    def __init__(self, dim: int, qwen_dim: int, num_heads: int, mlp_hidden_dim: int) -> None:
        super().__init__()
        self.cross_norm = TimestepModulatedNorm(dim)
        self.self_norm = TimestepModulatedNorm(dim)
        self.mlp_norm = TimestepModulatedNorm(dim)
        self.source_norm = nn.LayerNorm(qwen_dim)
        self.cross_attention = MultiHeadAttention(dim, qwen_dim, num_heads)
        self.self_attention = MultiHeadAttention(dim, dim, num_heads)
        self.mlp_in = nn.Linear(dim, 2 * mlp_hidden_dim, bias=False)
        self.mlp_out = nn.Linear(mlp_hidden_dim, dim, bias=False)
        self.time_modulation = nn.Linear(dim, 6 * dim, bias=True)

    def forward(self, queries, qwen_features, timestep_embedding, qwen_mask):
        modulation = self.time_modulation(F.silu(timestep_embedding))
        cross_scale, cross_shift, self_scale, self_shift, mlp_scale, mlp_shift = modulation.chunk(6, dim=-1)
        queries = queries + self.cross_attention(
            self.cross_norm(queries, cross_scale, cross_shift), self.source_norm(qwen_features), qwen_mask
        )
        normalized = self.self_norm(queries, self_scale, self_shift)
        queries = queries + self.self_attention(normalized, normalized)
        normalized = self.mlp_norm(queries, mlp_scale, mlp_shift)
        gate, value = self.mlp_in(normalized).chunk(2, dim=-1)
        return queries + self.mlp_out(F.silu(gate) * value)


class TimestepAwareSemanticResampler(nn.Module):
    def __init__(
        self, qwen_dim: int, output_dim: int, num_layers: int, num_queries: int, num_blocks: int,
        model_dim: int, num_heads: int, mlp_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, model_dim))
        self.layer_embeddings = nn.Parameter(torch.zeros(num_layers, 1, qwen_dim))
        self.time_mlp = nn.Sequential(nn.Linear(model_dim, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim))
        self.blocks = nn.ModuleList(
            [SemanticResamplerBlock(model_dim, qwen_dim, num_heads, mlp_hidden_dim) for _ in range(num_blocks)]
        )
        self.output_norm = nn.LayerNorm(model_dim)
        self.output_projection = nn.Linear(model_dim, output_dim, bias=False)

    def forward(self, hidden_states: Sequence[torch.Tensor], timesteps, source_mask):
        streams = [hidden + self.layer_embeddings[i].unsqueeze(0) for i, hidden in enumerate(hidden_states)]
        features = torch.cat(streams, dim=1)
        mask = None if source_mask is None else torch.cat([source_mask] * self.num_layers, dim=1)
        time = self.time_mlp(sinusoidal_timestep_embedding(timesteps, self.model_dim).to(dtype=features.dtype))
        queries = self.query_tokens.expand(hidden_states[0].shape[0], -1, -1)
        for block in self.blocks:
            queries = block(queries, features, time, mask)
        return self.output_projection(self.output_norm(queries))


class QualityAnchor(nn.Module):
    """v1 の adapter から引き継いだ部分。checkpoint のキーに合わせてフォルダーを分ける。"""

    def __init__(self, num_blocks: int, num_layers: int) -> None:
        super().__init__()
        head_dim = ADAPTER_DIM // ADAPTER_HEADS
        self.query_norms = nn.ModuleList([LLMAdapterRMSNorm(ADAPTER_DIM) for _ in range(num_blocks)])
        self.source_norms = nn.ModuleList([LLMAdapterRMSNorm(QWEN35_DIM) for _ in range(num_blocks)])
        self.semantic_attentions = nn.ModuleList(
            [LLMAdapterAttention(ADAPTER_DIM, QWEN35_DIM, ADAPTER_HEADS, head_dim) for _ in range(num_blocks)]
        )
        self.layer_mix_logits = nn.Parameter(torch.zeros(num_blocks, num_layers))


class SemanticConnectorV2(nn.Module):
    def __init__(
        self, num_queries: int = 64, resampler_blocks: int = 6, resampler_dim: int = 2048,
        resampler_heads: int = 16, mlp_hidden_dim: int = 5632, adapter_blocks: int = ADAPTER_BLOCKS,
    ) -> None:
        super().__init__()
        head_dim = ADAPTER_DIM // ADAPTER_HEADS
        self.quality_anchor = QualityAnchor(adapter_blocks, len(LAYER_INDICES))
        self.semantic_resampler = TimestepAwareSemanticResampler(
            QWEN35_DIM, ADAPTER_DIM, len(LAYER_INDICES), num_queries, resampler_blocks,
            resampler_dim, resampler_heads, mlp_hidden_dim,
        )
        self.v2_query_norms = nn.ModuleList([LLMAdapterRMSNorm(ADAPTER_DIM) for _ in range(adapter_blocks)])
        self.v2_semantic_norms = nn.ModuleList([LLMAdapterRMSNorm(ADAPTER_DIM) for _ in range(adapter_blocks)])
        self.v2_attentions = nn.ModuleList(
            [LLMAdapterAttention(ADAPTER_DIM, ADAPTER_DIM, ADAPTER_HEADS, head_dim) for _ in range(adapter_blocks)]
        )

    @staticmethod
    def _attention_mask(mask):
        if mask is None:
            return None
        mask = mask.to(torch.bool)
        return mask.unsqueeze(1).unsqueeze(1) if mask.ndim == 2 else mask

    def forward(
        self, native_adapter, native_source, target_input_ids, semantic_states, semantic_mask, timesteps,
        target_attention_mask=None, native_source_mask=None,
    ):
        """semantic_states は Qwen3.5 の 4 層で、それぞれ (B, L, 2560)。timesteps は [0, 1]。"""
        target_attention_mask = self._attention_mask(target_attention_mask)
        native_source_mask = self._attention_mask(native_source_mask)
        semantic_attention_mask = self._attention_mask(semantic_mask)
        bank = self.semantic_resampler(semantic_states, timesteps, semantic_mask)

        x = native_adapter.in_proj(native_adapter.embed(target_input_ids))

        def rope(tokens):
            return native_adapter.rotary_emb(x, torch.arange(tokens, device=x.device).unsqueeze(0))

        query_rope = rope(x.shape[1])
        native_rope = rope(native_source.shape[1])
        anchor_rope = rope(semantic_states[0].shape[1])
        bank_rope = rope(bank.shape[1])
        anchor = self.quality_anchor
        mix = anchor.layer_mix_logits.float().softmax(dim=-1).to(x.dtype)

        for index, native_block in enumerate(native_adapter.blocks):
            x = native_block(
                x,
                native_source,
                target_attention_mask=target_attention_mask,
                source_attention_mask=native_source_mask,
                position_embeddings=query_rope,
                position_embeddings_context=native_rope,
            )
            mixed = sum(state * mix[index, layer] for layer, state in enumerate(semantic_states))
            x = x + anchor.semantic_attentions[index](
                anchor.query_norms[index](x),
                mask=semantic_attention_mask,
                context=anchor.source_norms[index](mixed),
                position_embeddings=query_rope,
                position_embeddings_context=anchor_rope,
            )
            x = x + self.v2_attentions[index](
                self.v2_query_norms[index](x),
                context=self.v2_semantic_norms[index](bank),
                position_embeddings=query_rope,
                position_embeddings_context=bank_rope,
            )
        return native_adapter.norm(native_adapter.out_proj(x))
