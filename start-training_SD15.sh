#!/usr/bin/env bash
# SD15 の LoRA 学習を始める。TOML を省くと config/training/templates/sd15.toml に config/training/sd15.local.toml を重ねて使う。
#   ./start-training_SD15.sh [TOML] [--save-to local|drive] [--resume] [--drive|--no-drive]
exec bash "$(dirname "$0")/lib/train.sh" sd15 "$@"
