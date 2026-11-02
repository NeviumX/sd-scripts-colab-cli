#!/usr/bin/env bash
# SDXL の LoRA 学習を始める。TOML を省くと config/training/templates/sdxl.toml に config/training/sdxl.local.toml を重ねて使う。
#   ./start-training_SDXL.sh [TOML] [--save-to local|drive] [--resume] [--drive|--no-drive]
exec bash "$(dirname "$0")/lib/train.sh" sdxl "$@"
