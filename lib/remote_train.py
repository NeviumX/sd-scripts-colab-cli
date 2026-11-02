"""Colab VM 上で学習を実行する。手元の lib/common.sh が VM へ送り、ssh 越しに呼ぶ。

    python3 remote_train.py prepare DIR --sd-scripts DIR --venv DIR 拡張による TOML の補完
    python3 remote_train.py check DIR --sd-scripts DIR [--resume]   実行前の検査
    python3 remote_train.py run DIR --sd-scripts DIR --venv DIR     学習。tmux の中で動く
    python3 remote_train.py stop DIR                                学習を止める

DIR は job.json と training.toml を置いたジョブのフォルダー。ログは run.log に、状態は status.json に書く。
tmux の中で動くので、手元の SSH が切れても学習は止まらない。
"""

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import tomllib

ADD_ARGUMENT = re.compile(r"""add_argument\(\s*['"]--([A-Za-z0-9_]+)['"]""")
STEP_STATE = re.compile(r"-step\d{8}-state$")
EXIT_NOT_NEEDED = 3
# 拡張の入口を通るルートでも、上流の入口の引数を既知として扱う
UPSTREAM_ENTRYPOINTS = {"anima": "anima_train_network.py", "moss": "moss_image_train_network.py"}


def extensions_dir(sd_scripts):
    return Path(sd_scripts).parent / "extensions"


def unknown_keys(config, sd_scripts, info):
    """sd-scripts は TOML の未知のキーを黙って無視するので、ソースの引数名と照らし合わせる。"""
    upstream = UPSTREAM_ENTRYPOINTS.get(info["route"], info["entrypoint"])
    sources = [sd_scripts / info["entrypoint"], sd_scripts / upstream, sd_scripts / "train_network.py", *(sd_scripts / "library").glob("*.py"),
               *extensions_dir(sd_scripts).glob("*.py")]
    known = set()
    for source in sources:
        known.update(ADD_ARGUMENT.findall(source.read_text(encoding="utf-8", errors="replace")))
    return sorted(set(config) - known)


def complete_states(output_dir):
    """保存が終わった状態のフォルダーを古い順に返す。"""
    states = []
    for path in Path(output_dir).glob("*-state"):
        data = path / "train_state.json"
        if data.is_file() and any(path.glob("optimizer*")):
            states.append(path)
    return sorted(states, key=lambda p: p.stat().st_mtime)


def continue_epoch(state):
    """再開するエポック。ステップの状態はエポックの途中なので、そのエポックをやり直す。"""
    epoch = json.loads((state / "train_state.json").read_text(encoding="utf-8"))["current_epoch"]
    return epoch if STEP_STATE.search(state.name) else epoch + 1


def prepare(args):
    """extensions/prepare.d のスクリプトを順に実行する。1 つでも失敗すれば学習を始めない。"""
    training = Path(args.job) / "training.toml"
    for hook in sorted((extensions_dir(args.sd_scripts) / "prepare.d").glob("*.py")):
        print(f"[remote] {hook.name}", flush=True)
        command = [str(Path(args.venv) / "bin" / "python"), str(hook), str(training), "--sd-scripts", args.sd_scripts]
        code = subprocess.run(command, cwd=args.sd_scripts).returncode
        if code != 0:
            return code
    return 0


def check(args):
    job = Path(args.job)
    info = json.loads((job / "job.json").read_text(encoding="utf-8"))
    training = job / "training.toml"
    config = tomllib.loads(training.read_text(encoding="utf-8"))
    sd_scripts = Path(args.sd_scripts)

    unknown = unknown_keys(config, sd_scripts, info)
    if unknown:
        print("UNKNOWN_KEYS " + " ".join(unknown))
        return 1
    missing = [f"{key}={value}" for key, value in config.items()
               if key in {"dataset_config", "sample_prompts"} and not Path(value).exists()]
    if missing:
        print("MISSING " + " ".join(missing))
        return 1
    if args.resume:
        states = complete_states(config["output_dir"])
        if not states:
            print("NO_STATE " + config["output_dir"])
            return 1
        state = states[-1]
        epoch = continue_epoch(state)
        if epoch > (config.get("max_train_epochs") or sys.maxsize):
            print(f"NOTHING_LEFT {epoch} {config['max_train_epochs']}")
            return EXIT_NOT_NEEDED
        with training.open("a", encoding="utf-8") as file:
            file.write(f"resume = {json.dumps(str(state), ensure_ascii=False)}\ninitial_epoch = {epoch}\n")
        print(f"RESUME {state} {epoch}")
    return 0


class Status:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.data = {"state": "running", "started_at": now(), "updated_at": "", "exit_code": None}

    def write(self, **changes):
        with self.lock:
            self.data.update(changes, updated_at=now())
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.data), encoding="utf-8")
            temporary.replace(self.path)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def token_env(job):
    """手元から送ったトークンを、値をコマンドの引数に載せずに学習のプロセスへ渡す。"""
    env = {}
    path = job.parent.parent / "state" / "tokens.json"
    if path.is_file():
        env = {k: v for k, v in json.loads(path.read_text()).items() if k in {"HF_TOKEN", "WANDB_API_KEY"}}
    # トークンが無ければ、huggingface_hub にマシンのトークンを探させない
    if "HF_TOKEN" not in env:
        env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    return env


def run(args):
    job = Path(args.job)
    info = json.loads((job / "job.json").read_text(encoding="utf-8"))
    status = Status(job / "status.json")
    (job / "pid").write_text(str(os.getpid()))
    status.write()

    stopping = threading.Event()
    finished = threading.Event()

    def heartbeat():
        while not finished.wait(30):
            status.write()

    command = [str(Path(args.venv) / "bin" / "accelerate"), "launch", "--num_cpu_threads_per_process=1", info["entrypoint"],
               f"--config_file={job / 'training.toml'}"]
    # rich のログがパイプの幅で折り返されないようにする
    env = {**os.environ, **token_env(job), "PYTHONUNBUFFERED": "1", "COLUMNS": "160"}
    process = subprocess.Popen(command, cwd=args.sd_scripts, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, start_new_session=True)
    threading.Thread(target=heartbeat, daemon=True).start()

    def on_stop(signum, frame):
        stopping.set()
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass

    signal.signal(signal.SIGTERM, on_stop)
    signal.signal(signal.SIGINT, on_stop)
    code = 1
    try:
        with (job / "run.log").open("ab") as log:
            while chunk := os.read(process.stdout.fileno(), 65536):
                log.write(chunk)
                log.flush()
        code = process.wait()
    finally:
        finished.set()
        state = "stopped" if stopping.is_set() else "succeeded" if code == 0 else "failed"
        with (job / "run.log").open("a", encoding="utf-8") as log:
            log.write(f"\n[sd-scripts-colab-cli] training {state}, exit code {code}\n")
        status.write(state=state, exit_code=code)
    return code


def stop(args):
    job = Path(args.job)
    pid = int((job / "pid").read_text())
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return 1
    for _ in range(60):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return 0
        time.sleep(1)
    return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "check", "run", "stop"])
    parser.add_argument("job")
    parser.add_argument("--sd-scripts", default="")
    parser.add_argument("--venv", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    return {"prepare": prepare, "check": check, "run": run, "stop": stop}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
