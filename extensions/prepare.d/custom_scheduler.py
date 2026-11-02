"""LoRA Easy のスケジューラとオプティマイザに必要な値を、training.toml へ補完する。

    python custom_scheduler.py <training.toml> --sd-scripts <DIR>

first_cycle_max_steps と warmup_ratio は総ステップ数から決まるので、データセットを送った後の VM で計算する。
sd-scripts の TOML の扱いは変えない。対象の指定が無ければ、ファイルに触らずに終わる。
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import tomllib

RESTART_SCHEDULERS = {
    "LoraEasyCustomOptimizer.CosineAnnealingWarmRestarts.CosineAnnealingWarmRestarts",
    "LoraEasyCustomOptimizer.RexAnnealingWarmRestarts.RexAnnealingWarmRestarts",
}
FFTDESCENT = "LoraEasyCustomOptimizer.fftdescent.FFTDescent"


def has_arg(args, key):
    return any(str(item).split("=", 1)[0].strip() == key for item in args or [])


def import_entrypoint(sd_scripts, entrypoint):
    script = (sd_scripts / entrypoint).resolve()
    sys.path[:0] = [str(sd_scripts), str(script.parent)]
    spec = importlib.util.spec_from_file_location(f"_prepare_{script.stem}", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_args(sd_scripts, entrypoint, training):
    parser = import_entrypoint(sd_scripts, entrypoint).setup_parser()
    parsed = parser.parse_args(["--config_file", str(training)])

    from library import args as args_util

    original = sys.argv
    sys.argv = [original[0], "--config_file", str(training)]
    try:
        return args_util.read_config_from_file(parsed, parser)
    finally:
        sys.argv = original


def total_steps(sd_scripts, entrypoint, training):
    args = read_args(sd_scripts, entrypoint, training)
    if args.max_train_epochs is None:
        return int(args.max_train_steps)

    from library import accelerator_setup, config_util

    accelerator_setup.prepare_dataset_args(args, True)
    if args.dataset_config is not None:
        user_config = config_util.load_user_config(args.dataset_config)
    else:
        subsets = config_util.generate_dreambooth_subsets_config_by_subdirs(args.train_data_dir, args.reg_data_dir)
        user_config = {"datasets": [{"subsets": subsets}]}
    sanitizer = config_util.ConfigSanitizer(True, True, args.masked_loss, True)
    blueprint = config_util.BlueprintGenerator(sanitizer).generate(user_config, args)
    group, _ = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    if len(group) == 0:
        raise ValueError("no training data found while calculating custom scheduler steps")
    return int(args.max_train_epochs) * math.ceil(len(group) / int(args.gradient_accumulation_steps or 1))


def complete(config, steps_of, triton_available):
    """補完した内容の説明を返す。config は書き換える。何もしなければ空のリストを返す。"""
    changes = []
    if config.get("optimizer_type") == FFTDESCENT and not triton_available():
        optimizer_args = config.get("optimizer_args") or []
        if not has_arg(optimizer_args, "spectral_clip_compile") and not has_arg(optimizer_args, "compile_step"):
            config["optimizer_args"] = [*optimizer_args, "spectral_clip_compile=False"]
            changes.append("spectral_clip_compile=False")

    if config.get("lr_scheduler_type") not in RESTART_SCHEDULERS:
        return changes
    if config.get("lr_warmup_steps"):
        raise ValueError("lr_warmup_steps cannot be used with this lr_scheduler_type. Use warmup_ratio or warmup_steps in lr_scheduler_args.")

    scheduler_args = list(config.get("lr_scheduler_args") or [])
    cycles = int(config.get("lr_scheduler_num_cycles", 1) or 1)
    warmup_ratio = config.pop("warmup_ratio", None)
    if warmup_ratio is not None:
        changes.append("removed warmup_ratio")
    fill_cycle = not has_arg(scheduler_args, "first_cycle_max_steps")
    fill_warmup = warmup_ratio is not None and not has_arg(scheduler_args, "warmup_steps")
    if fill_cycle or fill_warmup:
        if cycles < 1:
            raise ValueError("lr_scheduler_num_cycles must be greater than zero")
        steps = steps_of()
        if fill_cycle:
            scheduler_args.append(f"first_cycle_max_steps={steps // cycles}")
            changes.append(scheduler_args[-1])
        if fill_warmup:
            scheduler_args.append(f"warmup_steps={round(steps * float(warmup_ratio)) // cycles}")
            changes.append(scheduler_args[-1])
        config["lr_scheduler_args"] = scheduler_args
    return changes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("training", type=Path)
    parser.add_argument("--sd-scripts", type=Path, required=True)
    args = parser.parse_args()

    config = tomllib.loads(args.training.read_text(encoding="utf-8"))
    entrypoint = json.loads((args.training.parent / "job.json").read_text(encoding="utf-8"))["entrypoint"]
    changes = complete(config, lambda: total_steps(args.sd_scripts, entrypoint, args.training),
                       lambda: importlib.util.find_spec("triton") is not None)
    if not changes:
        return 0

    import toml

    args.training.write_text(toml.dumps(config), encoding="utf-8")
    print("[remote] " + ", ".join(changes), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
