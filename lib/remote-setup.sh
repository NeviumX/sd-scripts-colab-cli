#!/usr/bin/env bash
# Colab VM 上で実行される。手元の lib/common.sh が lib/messages.sh と一緒に ssh 経由で標準入力から流し込む。
# 配置:
#   VM_ROOT/sd-scripts    sd-scripts 本体
#   VM_ROOT/sd-venv       uv で作った Python 環境
#   VM_ROOT/extensions    手元から送った拡張
#   VM_ROOT/custom_scheduler  extensions/setup.d/custom_optimizer.sh がクローンするオプティマイザ
#   VM_ROOT/models        学習に使うモデル
#   VM_ROOT/datasets      手元から送ったデータセット
#   VM_ROOT/outputs       学習結果。保存先が手元の PC のとき
#   STAGE_DIR             手元から送ったファイルと、ジョブと状態
#
# 受け取る変数: SD_SCRIPTS_REPO SD_SCRIPTS_REF SD_EXTRA_PACKAGES APT_PACKAGES VM_ROOT STAGE_DIR MSG_LANG
set -euo pipefail

SD_DIR="$VM_ROOT/sd-scripts"
VENV="$VM_ROOT/sd-venv"
VENV_PY="$VENV/bin/python"
STATE="$STAGE_DIR/state"
DONE="$STATE/setup-done"
TORCH_MIN=2.6
TORCH_STRICT=("torch==2.8.0" "torchvision==0.23.0")
TORCH_STRICT_INDEX=https://download.pytorch.org/whl/cu128
mkdir -p "$STATE" "$VM_ROOT/models" "$VM_ROOT/datasets" "$VM_ROOT/outputs"
rmsg() { echo "[remote] $(msg "$@")"; }

# 2 つのウィンドウから同時に実行されたときは、順番に待つ
exec 9>"$STATE/setup.lock"
flock -w 1800 9 || { rmsg r_lock_timeout "$STATE/setup.lock"; exit 1; }

# 拡張の setup.d の内容が変わったときも環境を作り直す
shopt -s nullglob
setup_hooks=("$VM_ROOT"/extensions/setup.d/*.sh)
shopt -u nullglob
want="$SD_SCRIPTS_REF|$SD_EXTRA_PACKAGES|$(cat /dev/null ${setup_hooks[@]+"${setup_hooks[@]}"} | sha256sum)"
if [[ -f "$DONE" && "$(cat "$DONE")" == "$want" ]]; then
    rmsg r_already_ready
    exit 0
fi

# ---- apt ----
# 別の名前のパッケージが提供していることもあるので、dpkg と提供名の一覧の両方を見る
provided="$(dpkg-query -W -f '${Provides}\n' 2>/dev/null || true)"
missing=()
for p in tmux rsync git $APT_PACKAGES; do
    dpkg -s "$p" >/dev/null 2>&1 || grep -qw -- "$p" <<<"$provided" || missing+=("$p")
done
if (( ${#missing[@]} > 0 )); then
    rmsg r_installing "${missing[*]}"
    apt-get -qq update >/dev/null
    apt-get -qq install -y "${missing[@]}" >/dev/null
fi

# ---- Python ----
# Colab の Python には CUDA 版の PyTorch が入っている。同じ版を venv にも入れる
find_py() {
    local c
    for c in $(type -ap python3) /usr/local/bin/python3 /usr/bin/python3; do
        [[ -x "$c" ]] && "$c" -c 'import torch' 2>/dev/null && { echo "$c"; return 0; }
    done
    return 1
}
PY="$(find_py)" || PY="$(command -v python3)"

if ! command -v uv >/dev/null 2>&1; then
    rmsg r_uv
    curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh >/dev/null
    export PATH="$HOME/.local/bin:$PATH"
fi

# ---- sd-scripts ----
# 履歴なしで指定のコミットだけを取る。ブランチやタグの名前も同じ手順で取れる。latest は既定のブランチの先頭
fetch_ref="$SD_SCRIPTS_REF"
[[ "$fetch_ref" == latest ]] && fetch_ref=HEAD
rmsg r_fetching "$SD_SCRIPTS_REPO" "$SD_SCRIPTS_REF"
if [[ ! -d "$SD_DIR/.git" ]]; then
    rm -rf "$SD_DIR"
    git init -q "$SD_DIR"
    git -C "$SD_DIR" remote add origin "$SD_SCRIPTS_REPO"
fi
git -C "$SD_DIR" fetch -q --depth=1 origin "$fetch_ref"
git -C "$SD_DIR" -c advice.detachedHead=false checkout -q --force FETCH_HEAD

# ---- パッケージ ----
if [[ ! -x "$VENV_PY" ]]; then
    rmsg r_venv "$PY"
    uv venv -q --python "$PY" "$VENV"
fi
uv_install() { uv pip install --python "$VENV_PY" -q "$@"; }

# Colab の torch が新しければ、同じ版を CUDA の索引から入れる。古いか無ければ、確かめられている組み合わせを入れる
if ! "$VENV_PY" -c 'import torch' 2>/dev/null; then
    pins="$("$PY" - "$TORCH_MIN" <<'PYEOF' 2>/dev/null || true
import importlib.metadata as metadata, re, sys
import torch
need = tuple(int(n) for n in sys.argv[1].split("."))
have = tuple(int(n) for n in re.findall(r"\d+", metadata.version("torch"))[:2])
cuda = torch.version.cuda or ""
if have >= need and re.fullmatch(r"\d+\.\d+", cuda):
    print("https://download.pytorch.org/whl/cu" + cuda.replace(".", ""))
    print("torch==" + metadata.version("torch"), "torchvision==" + metadata.version("torchvision"))
PYEOF
)"
    if [[ -n "$pins" ]]; then
        torch_index="$(sed -n 1p <<<"$pins")"
        read -r -a torch_pkgs <<<"$(sed -n 2p <<<"$pins")"
    else
        rmsg r_torch_strict "$TORCH_MIN"
        torch_index="$TORCH_STRICT_INDEX"
        torch_pkgs=("${TORCH_STRICT[@]}")
    fi
    rmsg r_torch "${torch_pkgs[*]}"
    uv_install --index-url "$torch_index" "${torch_pkgs[@]}"
fi
torch_before="$("$VENV_PY" -c 'import importlib.metadata as m; print(m.version("torch"))')"

# requirements.txt は相対指定の -e . を含むので、sd-scripts のフォルダーで実行する
rmsg r_requirements
(cd "$SD_DIR" && uv_install -r requirements.txt)
if [[ -n "$SD_EXTRA_PACKAGES" ]]; then
    rmsg r_extra "$SD_EXTRA_PACKAGES"
    # shellcheck disable=SC2086
    uv_install $SD_EXTRA_PACKAGES
fi

# 拡張の追加の導入。uv_install、SD_DIR、VENV_PY、VM_ROOT を使える
for hook in ${setup_hooks[@]+"${setup_hooks[@]}"}; do
    # shellcheck disable=SC1090
    source "$hook"
done

# requirements が torch を置き換えると、torchvision との組み合わせが崩れて学習が失敗する
torch_after="$("$VENV_PY" -c 'import importlib.metadata as m; print(m.version("torch"))')"
if [[ "$torch_after" != "$torch_before" ]]; then
    rmsg r_torch_replaced "$torch_before" "$torch_after"
    exit 1
fi
"$VENV_PY" -c 'import accelerate, diffusers, torch, torchvision, transformers' || { rmsg r_import_failed; exit 1; }

# accelerate の設定は 1 回だけ作る
"$VENV_PY" - <<'PYEOF'
from pathlib import Path
from accelerate.utils import write_basic_config
if not (Path.home() / ".cache/huggingface/accelerate/default_config.yaml").exists():
    write_basic_config()
PYEOF

printf '%s' "$want" > "$DONE"
rmsg r_ready "$(git -C "$SD_DIR" rev-parse --short HEAD)"
