#!/usr/bin/env bash
# MOSS の LoRA 学習を始める。TOML を省くと config/training/templates/moss.toml に config/training/moss.local.toml を重ねて使う。
#   ./start-training_MOSS.sh [TOML] [--save-to local|drive] [--resume] [--drive|--no-drive]
exec bash "$(dirname "$0")/lib/train.sh" moss "$@"
