"""Anima のモデルの台帳と、safetensors のヘッダーからの判定。

重みを読まないので、モデルを読み込む前に動かせる。標準ライブラリだけで動くよう、torch を読み込まない。
新しいモデルは VARIANTS に足す。手順は docs/extensions.md にある。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import struct


ANIMA_MODEL_CHANNELS = 2048
MAX_HEADER_BYTES = 100 * 1024 * 1024

_SPLIT_FILE = re.compile(r"^(.*?)(\d+)-of-(\d+)\.safetensors$", re.IGNORECASE)
_BLOCK_KEY = re.compile(r"^blocks\.(\d+)\.")
_KEY_PREFIXES = ("net.", "model.diffusion_model.")
_CHANNELS_KEY = "x_embedder.proj.1.weight"


@dataclass(frozen=True)
class AnimaVariant:
    """loader は上流と違う読み込みが必要なモデルの Mixin で、「モジュール:クラス」の形で書く。

    required_key は、そのモデルに必ずあるキーの先頭。同じブロック数の別の版を見分ける。
    """

    name: str
    block_count: int
    loader: str = ""
    required_key: str = ""


VARIANTS = (
    AnimaVariant("standard", 28),
    AnimaVariant("2.9B", 40, loader="anima_block_count:BlockCountLoaderMixin"),
    AnimaVariant("3.8B", 52, loader="anima_3_8b:Anima38BLoaderMixin", required_key="anima_v2_connector."),
)


@dataclass(frozen=True)
class AnimaCheckpoint:
    path: Path
    model_files: tuple[Path, ...]
    variant: AnimaVariant


def normalize_key(key: str) -> str:
    for prefix in _KEY_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def model_files_of(path: Path) -> tuple[Path, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Anima checkpoint was not found: {path}")
    if path.suffix.lower() != ".safetensors":
        raise ValueError(f"Anima checkpoint must be a safetensors file: {path}")

    match = _SPLIT_FILE.match(path.name)
    if match is None:
        return (path,)
    count = int(match.group(3))
    files = tuple(path.with_name(f"{match.group(1)}{i:05d}-of-{count:05d}.safetensors") for i in range(1, count + 1))
    missing = [str(file) for file in files if not file.is_file()]
    if missing:
        raise FileNotFoundError("Anima split checkpoint is incomplete. Missing: " + ", ".join(missing))
    return files


def read_header(path: Path) -> dict:
    with path.open("rb") as file:
        raw_size = file.read(8)
        if len(raw_size) != 8:
            raise ValueError(f"Not a safetensors file: {path}")
        size = struct.unpack("<Q", raw_size)[0]
        if size > MAX_HEADER_BYTES:
            raise ValueError(f"Not a safetensors file: {path}")
        try:
            header = json.loads(file.read(size))
        except ValueError:
            raise ValueError(f"Not a safetensors file: {path}") from None
    header.pop("__metadata__", None)
    return header


def inspect_checkpoint(model_path: str | Path) -> AnimaCheckpoint:
    path = Path(model_path).expanduser().resolve()
    files = model_files_of(path)
    blocks: set[int] = set()
    keys: list[str] = []
    channels = None

    for file in files:
        for original_key, tensor in read_header(file).items():
            key = normalize_key(original_key)
            keys.append(key)
            if key.endswith(".weight_scale"):
                raise ValueError(
                    "This checkpoint is pre-quantized, for example int8 ConvRot, and cannot be trained. "
                    "Use the regular safetensors file."
                )
            match = _BLOCK_KEY.match(key)
            if match is not None:
                blocks.add(int(match.group(1)))
            if key == _CHANNELS_KEY:
                channels = int(tensor["shape"][0])

    if not blocks:
        raise ValueError("The checkpoint has no Anima transformer block keys")
    missing = sorted(set(range(max(blocks) + 1)) - blocks)
    if missing:
        raise ValueError("The checkpoint has missing transformer blocks: " + ", ".join(map(str, missing)))
    if channels is None:
        raise ValueError(f"The checkpoint has no {_CHANNELS_KEY}")
    if channels != ANIMA_MODEL_CHANNELS:
        raise ValueError(f"Unsupported Anima model_channels: {channels}. Expected {ANIMA_MODEL_CHANNELS}.")

    for variant in VARIANTS:
        if variant.block_count == len(blocks):
            if variant.required_key and not any(key.startswith(variant.required_key) for key in keys):
                raise ValueError(
                    f"This {len(blocks)}-block checkpoint has no {variant.required_key} keys. "
                    f"Anima {variant.name} needs the bundled release, for example Anima-3.8B-v1.1.safetensors."
                )
            return AnimaCheckpoint(path, files, variant)
    supported = ", ".join(f"{v.block_count} ({v.name})" for v in VARIANTS)
    raise ValueError(f"Unsupported Anima block count: {len(blocks)}. Supported: {supported}.")
