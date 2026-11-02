#!/usr/bin/env bash
# Anima の LoRA 学習を始める。TOML を省くと config/training/templates/anima.toml に config/training/anima.local.toml を重ねて使う。
#   ./start-training_Anima.sh [TOML] [--save-to local|drive] [--resume] [--drive|--no-drive]
exec bash "$(dirname "$0")/lib/train.sh" anima "$@"
