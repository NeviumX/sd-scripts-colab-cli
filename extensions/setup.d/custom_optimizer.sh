# LoRA Easy のカスタムオプティマイザを VM へクローンして、venv へ連結する。sd-scripts は変更しない
# remote-setup.sh が requirements の後に source する。下の版を変えると環境構築をやり直す
# torchao 0.14 以降は custom_scheduler が使う定数が無く、ImportError になる
CUSTOM_OPTIMIZER_REPO=https://github.com/67372a/LoRA_Easy_Training_scripts_Backend.git
CUSTOM_OPTIMIZER_COMMIT=656b075757a525c2a1b3d47da9655875705affb6
CUSTOM_OPTIMIZER_PACKAGES=("adv-optm==2.6.0" "torchao==0.13.0")
CUSTOM_OPTIMIZER_DIR="$VM_ROOT/custom_scheduler"

rmsg r_custom_optimizer "${CUSTOM_OPTIMIZER_COMMIT:0:7}"
if [[ ! -d "$CUSTOM_OPTIMIZER_DIR/.git" ]]; then
    rm -rf "$CUSTOM_OPTIMIZER_DIR"
    git init -q "$CUSTOM_OPTIMIZER_DIR"
    git -C "$CUSTOM_OPTIMIZER_DIR" remote add origin "$CUSTOM_OPTIMIZER_REPO"
    git -C "$CUSTOM_OPTIMIZER_DIR" config extensions.partialClone origin
    git -C "$CUSTOM_OPTIMIZER_DIR" sparse-checkout set custom_scheduler
fi
git -C "$CUSTOM_OPTIMIZER_DIR" fetch -q --depth=1 --filter=blob:none origin "$CUSTOM_OPTIMIZER_COMMIT"
git -C "$CUSTOM_OPTIMIZER_DIR" -c advice.detachedHead=false checkout -q --force FETCH_HEAD

# --upgrade は torch まで上げるので付けない
uv_install "${CUSTOM_OPTIMIZER_PACKAGES[@]}"
uv_install --no-deps -e "$CUSTOM_OPTIMIZER_DIR/custom_scheduler"
"$VENV_PY" -W ignore -c 'import LoraEasyCustomOptimizer' || { rmsg r_custom_optimizer_failed; exit 1; }
