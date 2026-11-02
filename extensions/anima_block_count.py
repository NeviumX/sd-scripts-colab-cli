"""上流が固定している Anima のブロック数を変えて読み込む。"""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import logging

from library import anima_models


UPSTREAM_BLOCK_COUNT = 28

logger = logging.getLogger(__name__)


@contextmanager
def force_block_count(anima_class, block_count: int):
    """読み込みの間だけ、コンストラクターに渡るブロック数を変える。"""
    original_init = anima_class.__init__

    @wraps(original_init)
    def patched_init(self, *args, **kwargs):
        if "num_blocks" not in kwargs:
            raise RuntimeError("Upstream Anima constructor no longer passes num_blocks by keyword")
        requested = int(kwargs["num_blocks"])
        if requested == UPSTREAM_BLOCK_COUNT:
            kwargs["num_blocks"] = block_count
        elif requested != block_count:
            raise RuntimeError(f"Upstream Anima requested {requested} blocks, but the checkpoint has {block_count}")
        original_init(self, *args, **kwargs)

    anima_class.__init__ = patched_init
    try:
        yield
    finally:
        anima_class.__init__ = original_init


class BlockCountLoaderMixin:
    """台帳のブロック数でモデルを作る。sd-scripts は変更しない。"""

    def load_unet_lazily(self, args, weight_dtype, accelerator, text_encoders):
        block_count = args.anima_checkpoint.variant.block_count
        logger.info("Loading Anima with %d transformer blocks", block_count)
        with force_block_count(anima_models.Anima, block_count):
            model, text_encoders = super().load_unet_lazily(args, weight_dtype, accelerator, text_encoders)
        if len(model.blocks) != block_count:
            raise RuntimeError(f"Loaded Anima has {len(model.blocks)} blocks, expected {block_count}")
        return model, text_encoders
