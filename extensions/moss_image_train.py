"""MOSS-Image の学習の入口。上流の学習に detail_inv_t を足す。

VM_ROOT/extensions に置かれ、隣の VM_ROOT/sd-scripts を読み込む。起動の手順は moss_image_train_network.py の末尾と同じ。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

EXTENSIONS_DIR = Path(__file__).resolve().parent
SD_SCRIPTS_DIR = EXTENSIONS_DIR.parent / "sd-scripts"
for path in (EXTENSIONS_DIR, SD_SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import moss_image_train_network as upstream  # noqa: E402
from library import args as args_util  # noqa: E402
from library import flux_train_utils, moss_image_train_utils  # noqa: E402

from detail_inv_t_support import (  # noqa: E402
    DetailInvTTrainerMixin,
    add_detail_inv_t_arguments,
    show_timesteps_with_detail_inv_t,
    validate_detail_inv_t_args,
)


class MossImageDetailInvTTrainer(DetailInvTTrainerMixin, upstream.MossImageNetworkTrainer):
    # 上流は sigma * 1000 を timesteps として渡す
    detail_inv_t_timestep_scale = float(moss_image_train_utils.NUM_TRAIN_TIMESTEPS)


def setup_parser() -> argparse.ArgumentParser:
    return add_detail_inv_t_arguments(upstream.setup_parser())


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
            args, moss_image_train_utils.show_timesteps, flux_train_utils, "compute_loss_weighting_for_sd3"
        )
        return 0

    MossImageDetailInvTTrainer().train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
