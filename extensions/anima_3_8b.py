"""Anima-3.8B v1.1 の読み込みと学習。52 ブロックの DiT に、同梱の Semantic Connector v2 と Qwen3.5 4B を足す。

接続部は timestep ごとに llm_adapter の代わりに動く。上流は変更せず、モデルの読み込みの間だけ接続部のキーを抜き出し、
llm_adapter と forward をインスタンスの上で差し替える。Text Encoder の出力は Anima の 4 つに 2 つを足して 6 つになる。
接続部と Qwen3.5 は固定で、LoRA は DiT のブロックだけを学習する。手順の説明は docs/extensions.md にある。
"""

from __future__ import annotations

from contextlib import contextmanager
import logging
import random

import numpy as np
import torch

from library import anima_utils, strategy_anima
from library.device_utils import clean_memory_on_device

from anima_3_8b_connector import LAYER_INDICES, SemanticConnectorV2
from anima_block_count import BlockCountLoaderMixin
from qwen35_text_encoder import PAD_TOKEN_ID, Qwen35HybridModel, load_qwen35


CONNECTOR_PREFIX = "anima_v2_connector."
QWEN35_TOKENIZER = "Qwen/Qwen3.5-4B"
QWEN35_TOKENIZER_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

logger = logging.getLogger(__name__)


# ---- Text Encoder ----


class Anima38BTokenizeStrategy(strategy_anima.AnimaTokenizeStrategy):
    def __init__(self, qwen35_tokenizer, qwen35_max_length: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.qwen35_tokenizer = qwen35_tokenizer
        self.qwen35_max_length = qwen35_max_length

    def tokenize(self, text):
        text = [text] if isinstance(text, str) else text
        tokens = super().tokenize(text)
        ids = torch.full((len(text), self.qwen35_max_length), PAD_TOKEN_ID, dtype=torch.long)
        mask = torch.zeros((len(text), self.qwen35_max_length), dtype=torch.long)
        for row, caption in enumerate(text):
            encoded = self.qwen35_tokenizer.encode(caption, add_special_tokens=False)[: self.qwen35_max_length]
            encoded = encoded or [PAD_TOKEN_ID]
            ids[row, : len(encoded)] = torch.tensor(encoded, dtype=torch.long)
            mask[row, : len(encoded)] = 1
        return tokens + [ids, mask]


class Anima38BTextEncodingStrategy(strategy_anima.AnimaTextEncodingStrategy):
    """出力は [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask, semantic_states, semantic_mask]。

    semantic_states は (B, 4, L, 2560)。Qwen3.5 は text_encoders の一覧に入れず、ここで持つ。
    """

    def __init__(self) -> None:
        super().__init__()
        self.qwen35: Qwen35HybridModel | None = None

    def encode_tokens(self, tokenize_strategy, models, tokens):
        outputs = super().encode_tokens(tokenize_strategy, models, tokens[:4])
        ids, mask = tokens[4:6]
        device = self.qwen35.device
        # 走査が遅いので、バッチの最長まで切り詰める
        length = max(int(mask.sum(dim=1).max()), 1)
        states = self.qwen35.hidden_states(ids[:, :length].to(device), mask[:, :length].to(device), LAYER_INDICES)
        stacked = torch.zeros(ids.shape[0], len(LAYER_INDICES), ids.shape[1], states[0].shape[-1], device=device)
        for layer, state in enumerate(states):
            stacked[:, layer, :length] = state
        stacked *= mask.to(device)[:, None, :, None]
        return outputs + [stacked, mask]

    def drop_cached_text_encoder_outputs(
        self, prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask, semantic_states, semantic_mask,
        caption_dropout_rates=None,
    ):
        outputs = [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask, semantic_states, semantic_mask]
        if caption_dropout_rates is None or torch.all(caption_dropout_rates == 0.0).item():
            return outputs

        outputs = [value.clone() for value in outputs]
        prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask, semantic_states, semantic_mask = outputs
        for i in range(prompt_embeds.shape[0]):
            if random.random() < caption_dropout_rates[i].item():
                prompt_embeds[i] = 0
                t5_input_ids[i, 0] = 1
                t5_input_ids[i, 1:] = 0
                t5_attn_mask[i, 0] = 1
                t5_attn_mask[i, 1:] = 0
                semantic_states[i] = 0
                # マスクがすべて偽だと attention の結果が NaN になるので、空の 1 トークンを残す
                attn_mask[i, 1:] = 0
                attn_mask[i, 0] = 1
                semantic_mask[i, 1:] = 0
                semantic_mask[i, 0] = 1
        return outputs


class Anima38BTextEncoderOutputsCachingStrategy(strategy_anima.AnimaTextEncoderOutputsCachingStrategy):
    ANIMA_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX = "_anima38b_te.npz"
    EXTRA_KEYS = ("semantic_states", "semantic_mask")

    def is_disk_cached_outputs_expected(self, npz_path: str) -> bool:
        if not super().is_disk_cached_outputs_expected(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True
        with np.load(npz_path) as npz:
            return all(key in npz for key in self.EXTRA_KEYS)

    def load_outputs_npz(self, npz_path: str):
        data = np.load(npz_path)
        return [
            data["prompt_embeds"], data["attn_mask"], data["t5_input_ids"], data["t5_attn_mask"],
            data["semantic_states"], data["semantic_mask"], data["caption_dropout_rate"],
        ]

    def cache_batch_outputs(self, tokenize_strategy, models, text_encoding_strategy, infos, captions):
        tokens = tokenize_strategy.tokenize(captions)
        with torch.no_grad():
            outputs = text_encoding_strategy.encode_tokens(tokenize_strategy, models, tokens)
        embeds, attn_mask, t5_ids, t5_mask, states, semantic_mask = [
            value.float().cpu().numpy() if value.is_floating_point() else value.cpu().numpy() for value in outputs
        ]
        for i, info in enumerate(infos):
            values = {
                "prompt_embeds": embeds[i],
                "attn_mask": attn_mask[i],
                "t5_input_ids": t5_ids[i].astype(np.int32),
                "t5_attn_mask": t5_mask[i].astype(np.int32),
                "semantic_states": states[i],
                "semantic_mask": semantic_mask[i].astype(np.int32),
                "caption_dropout_rate": np.array(info.caption_dropout_rate, dtype=np.float32),
            }
            if self.cache_to_disk:
                # 4 層を最大の長さで持つので大きい。パディングの 0 が多いので圧縮する
                np.savez_compressed(info.text_encoder_outputs_npz, **values)
            else:
                info.text_encoder_outputs = tuple(values.values())


# ---- 接続部の実行 ----


def ids_key(input_ids: torch.Tensor) -> bytes:
    return input_ids.reshape(-1).to(torch.long).cpu().numpy().tobytes()


class ConnectorRuntime:
    """llm_adapter と forward を差し替えて、接続部へ timesteps と Qwen3.5 の出力を渡す。

    学習では forward の中で llm_adapter が呼ばれるので、そのまま接続部を動かす。
    生成確認では、上流が llm_adapter を先に 1 回だけ呼び、結果を全ステップで使う。接続部は毎ステップ動かす必要があるので、
    ここでは空のテンソルを返して引数を覚えておき、そのテンソルが forward に渡ったときに計算する。
    """

    def __init__(self, dit, connector: SemanticConnectorV2, sample_semantics: dict[bytes, tuple]) -> None:
        self.dit = dit
        self.connector = connector
        self.timesteps = None
        self.training_semantic = None
        self.sample_semantics = sample_semantics
        self.deferred: dict[int, tuple] = {}
        self.original_forward = dit.forward
        # 束縛メソッドのまま入れると、accelerate.prepare が forward.__func__ を取り出して dit に束縛し直し、
        # self が ConnectorRuntime ではなく dit になる。__func__ を持たない関数で包んでおく。
        dit.llm_adapter.forward = lambda *args, **kwargs: self.adapter_forward(*args, **kwargs)
        dit.forward = lambda *args, **kwargs: self.dit_forward(*args, **kwargs)

    def run(self, source, input_ids, target_mask, source_mask, semantic, timesteps):
        states, semantic_mask = semantic
        device = input_ids.device
        dtype = self.dit.llm_adapter.embed.weight.dtype
        layers = [state.to(device=device, dtype=dtype) for state in states.unbind(dim=1)]
        with torch.no_grad():
            return self.connector(
                self.dit.llm_adapter,
                source.to(dtype=dtype),
                input_ids,
                layers,
                semantic_mask.to(device=device, dtype=torch.bool),
                timesteps.reshape(-1).to(device=device),
                target_attention_mask=target_mask,
                native_source_mask=source_mask,
            )

    def adapter_forward(self, source_hidden_states, target_input_ids, target_attention_mask=None, source_attention_mask=None):
        if self.timesteps is not None:
            if self.training_semantic is None:
                raise RuntimeError("Anima-3.8B needs Qwen3.5 outputs for this step")
            return self.run(
                source_hidden_states, target_input_ids, target_attention_mask, source_attention_mask,
                self.training_semantic, self.timesteps,
            )
        semantic = self.sample_semantics.get(ids_key(target_input_ids))
        if semantic is None:
            raise RuntimeError("Anima-3.8B has no cached Qwen3.5 output for this sample prompt")
        placeholder = source_hidden_states.new_zeros(target_input_ids.shape[0], target_input_ids.shape[1], 1024)
        self.deferred[id(placeholder)] = (
            placeholder, source_hidden_states, target_input_ids, target_attention_mask, source_attention_mask, semantic,
        )
        return placeholder

    def dit_forward(self, x, timesteps, context=None, *args, **kwargs):
        deferred = self.deferred.get(id(context))
        if deferred is not None:
            _, source, input_ids, target_mask, source_mask, semantic = deferred
            context = self.run(source, input_ids, target_mask, source_mask, semantic, timesteps)
            context[~target_mask.bool()] = 0
        self.timesteps = timesteps
        try:
            return self.original_forward(x, timesteps, context, *args, **kwargs)
        finally:
            self.timesteps = None


# ---- 読み込み ----


@contextmanager
def capture_connector_state():
    """上流は余分なキーで止まるので、読み込みの直後に接続部のキーを抜き出す。"""
    original = anima_utils.load_safetensors_with_lora_and_fp8
    captured: dict[str, torch.Tensor] = {}

    def load(*args, **kwargs):
        state = original(*args, **kwargs)
        for key in [key for key in state if key.startswith(CONNECTOR_PREFIX)]:
            captured[key[len(CONNECTOR_PREFIX) :]] = state.pop(key)
        return state

    anima_utils.load_safetensors_with_lora_and_fp8 = load
    try:
        yield captured
    finally:
        anima_utils.load_safetensors_with_lora_and_fp8 = original


def build_connector(state: dict[str, torch.Tensor], dtype) -> SemanticConnectorV2:
    from accelerate import init_empty_weights

    with init_empty_weights():
        connector = SemanticConnectorV2()
    connector.load_state_dict(state, strict=True, assign=True)
    if dtype is not None:
        connector.to(dtype)
    return connector.eval().requires_grad_(False)


class Anima38BLoaderMixin(BlockCountLoaderMixin):
    def __init__(self) -> None:
        super().__init__()
        self.text_encoding_strategy = Anima38BTextEncodingStrategy()
        self.connector_runtime: ConnectorRuntime | None = None
        self.sample_semantics: dict[bytes, tuple] = {}

    def assert_extra_args(self, args, train_dataset_group, val_dataset_group):
        if not args.qwen35:
            raise ValueError("Anima-3.8B needs qwen35 = \"qwen35_4b.safetensors\" in the training TOML")
        if not args.cache_text_encoder_outputs and not args.cache_text_encoder_outputs_to_disk:
            raise ValueError("Anima-3.8B needs cache_text_encoder_outputs = true")
        if "train_llm_adapter=true" in [str(item).replace(" ", "").lower() for item in args.network_args or []]:
            raise ValueError("Anima-3.8B cannot train the llm adapter because the connector replaces it")
        super().assert_extra_args(args, train_dataset_group, val_dataset_group)

    def get_tokenize_strategy(self, args):
        from transformers import AutoTokenizer

        base = super().get_tokenize_strategy(args)
        tokenizer = AutoTokenizer.from_pretrained(QWEN35_TOKENIZER, revision=QWEN35_TOKENIZER_REVISION)
        return Anima38BTokenizeStrategy(
            tokenizer,
            args.qwen35_max_token_length,
            qwen3_tokenizer=base.qwen3_tokenizer,
            t5_tokenizer=base.t5_tokenizer,
            qwen3_max_length=base.qwen3_max_length,
            t5_max_length=base.t5_max_length,
        )

    def get_text_encoding_strategy(self, args):
        return self.text_encoding_strategy

    def get_text_encoder_outputs_caching_strategy(self, args):
        return Anima38BTextEncoderOutputsCachingStrategy(
            args.cache_text_encoder_outputs_to_disk, args.text_encoder_batch_size, args.skip_cache_check, False
        )

    def load_target_model(self, args, weight_dtype, accelerator):
        result = super().load_target_model(args, weight_dtype, accelerator)
        logger.info("Loading Qwen3.5 text encoder...")
        self.text_encoding_strategy.qwen35 = load_qwen35(args.qwen35, weight_dtype)
        return result

    def load_unet_lazily(self, args, weight_dtype, accelerator, text_encoders):
        # 上流はキャッシュの後に DiT を読み込むので、ここまで来れば Qwen3.5 は使わない。メモリを空ける
        self.text_encoding_strategy.qwen35 = None
        clean_memory_on_device(accelerator.device)

        with capture_connector_state() as state:
            model, text_encoders = super().load_unet_lazily(args, weight_dtype, accelerator, text_encoders)
        if not state:
            raise RuntimeError("The checkpoint has no anima_v2_connector tensors")
        connector = build_connector(state, weight_dtype)
        model.add_module("anima_v2_connector", connector)
        self.connector_runtime = ConnectorRuntime(model, connector, self.sample_semantics)
        logger.info("Loaded Semantic Connector v2 with %d tensors", len(state))
        return model, text_encoders

    def cache_text_encoder_outputs_if_needed(self, args, accelerator, unet, vae, text_encoders, dataset, weight_dtype):
        # 検証用のデータセットがあると 2 回呼ばれる
        qwen35 = self.text_encoding_strategy.qwen35
        qwen35.to(accelerator.device)
        super().cache_text_encoder_outputs_if_needed(args, accelerator, unet, vae, text_encoders, dataset, weight_dtype)
        qwen35.to("cpu")
        clean_memory_on_device(accelerator.device)

        # 上流の生成確認は 4 つの出力を前提にするので、Qwen3.5 の分を T5 のトークンで引けるように分ける
        for prompt, outputs in (self.sample_prompts_te_outputs or {}).items():
            self.sample_semantics[ids_key(outputs[2])] = (outputs[4].cpu(), outputs[5].cpu())
            self.sample_prompts_te_outputs[prompt] = outputs[:4]

    def get_noise_pred_and_target(
        self, args, accelerator, noise_scheduler, latents, batch, text_encoder_conds, unet, network, weight_dtype,
        train_unet, is_train=True,
    ):
        # 上流は先頭の 4 つだけを読む
        self.connector_runtime.training_semantic = tuple(text_encoder_conds[4:6])
        return super().get_noise_pred_and_target(
            args, accelerator, noise_scheduler, latents, batch, text_encoder_conds, unet, network, weight_dtype,
            train_unet, is_train=is_train,
        )

    def sample_images(self, *args, **kwargs):
        try:
            return super().sample_images(*args, **kwargs)
        finally:
            self.connector_runtime.deferred.clear()
