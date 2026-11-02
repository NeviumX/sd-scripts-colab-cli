"""Anima の学習の入口。モデルの中身から台帳の種類を選び、学習の処理を組み立てる。

VM_ROOT/extensions に置かれ、隣の VM_ROOT/sd-scripts を読み込む。起動の手順は anima_train_network.py の末尾と同じ。
"""

from __future__ import annotations

import argparse
import importlib
import logging
from pathlib import Path
import sys

EXTENSIONS_DIR = Path(__file__).resolve().parent
SD_SCRIPTS_DIR = EXTENSIONS_DIR.parent / "sd-scripts"
for path in (EXTENSIONS_DIR, SD_SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import anima_train_network as upstream  # noqa: E402
from library import anima_train_utils  # noqa: E402
from library import args as args_util  # noqa: E402

from detail_inv_t_support import (  # noqa: E402
    DetailInvTTrainerMixin,
    add_detail_inv_t_arguments,
    show_timesteps_with_detail_inv_t,
    validate_detail_inv_t_args,
)
from anima_variants import AnimaVariant, inspect_checkpoint  # noqa: E402


logger = logging.getLogger(__name__)


class VariantMetadataMixin:
    def update_metadata(self, metadata, args):
        super().update_metadata(metadata, args)
        metadata["ss_anima_num_blocks"] = args.anima_checkpoint.variant.block_count
        metadata["ss_anima_model_variant"] = args.anima_checkpoint.variant.name


def setup_parser() -> argparse.ArgumentParser:
    parser = add_detail_inv_t_arguments(upstream.setup_parser())
    parser.add_argument("--qwen35", type=str, default=None, help="Qwen3.5 4B text encoder for Anima-3.8B")
    parser.add_argument("--qwen35_max_token_length", type=int, default=256, help="Maximum Qwen3.5 tokens for Anima-3.8B")
    return parser


def build_trainer_class(variant: AnimaVariant):
    bases = [VariantMetadataMixin, DetailInvTTrainerMixin, upstream.AnimaNetworkTrainer]
    if variant.loader:
        module_name, class_name = variant.loader.split(":")
        bases.insert(0, getattr(importlib.import_module(module_name), class_name))
    return type("AnimaVariantTrainer", tuple(bases), {})


def check_blocks_to_swap(args, variant: AnimaVariant) -> None:
    limit = variant.block_count - 2
    if (args.blocks_to_swap or 0) > limit:
        raise ValueError(f"blocks_to_swap is at most {limit} for Anima {variant.name}")


def main() -> int:
    parser = setup_parser()
    args = parser.parse_args()
    args_util.verify_command_line_training_args(args)
    args = args_util.read_config_from_file(args, parser)
    validate_detail_inv_t_args(args)

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"

    if args.show_timesteps:
        show_timesteps_with_detail_inv_t(
            args, anima_train_utils.show_timesteps, anima_train_utils, "compute_loss_weighting_for_anima"
        )
        return 0

    checkpoint = inspect_checkpoint(args.pretrained_model_name_or_path)
    check_blocks_to_swap(args, checkpoint.variant)
    args.anima_checkpoint = checkpoint
    logger.info("Detected Anima %s: %d transformer blocks", checkpoint.variant.name, checkpoint.variant.block_count)
    print(f"Anima variant: {checkpoint.variant.name}, {checkpoint.variant.block_count} blocks", flush=True)
    build_trainer_class(checkpoint.variant)().train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
