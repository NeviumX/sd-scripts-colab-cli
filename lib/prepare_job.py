"""学習の TOML を手元で調べて、Colab VM へ送るファイルを作る。

    python3 lib/prepare_job.py --training TOML --route NAME --entrypoint FILE --out DIR --root DIR
                               --vm-root DIR --stage DIR --catalog FILE [--lang ja]

VM を確保する前に動くので、ここで止めれば課金は発生しない。問題があれば理由を表示して 1 で終わる。
config/<種類>/templates の原本に config/<種類>/<名前>.local.toml があれば重ねる。プロンプトは .local.txt が全体を置き換える。
--out には training.toml、dataset.toml、sample_prompts.txt、job.json を作り、最後に output_name を標準出力に書く。
output_dir は保存先が決まってから手元の lib/train.sh が足す。
TOML の値は sd-scripts の引数のまま扱い、次の値だけを VM 上のパスに書き換える。

    モデルのキー     ファイル名だけなら VM の models フォルダー
    dataset_config   手元のファイルなら VM へ送る。省略すると config/datasets/templates/<種類>.toml
    sample_prompts   手元のファイルなら VM へ送る。省略すると config/prompts/templates/<種類>.txt
    image_dir        手元のフォルダーなら VM へ送る。画像のあるサブフォルダーごとに subset を分け、<数>_ の名前を繰り返し数にする
    train_data_dir   手元のフォルダーなら VM へ送る
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tomllib
from urllib.parse import unquote, urlsplit

MODEL_KEYS = ("pretrained_model_name_or_path", "vae", "qwen3", "qwen35", "text_encoder", "network_weights")
SECRET_KEY = re.compile(r"_token$|api_key|secret|password", re.I)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
BROKEN_SCHEDULERS = {"reduce_lr_on_plateau", "greedy", "cosine_warmup_with_min_lr"}
TE_CACHE_FORBIDS = ("shuffle_caption", "caption_tag_dropout_rate", "token_warmup_step")
SDXL_ROUTES = {"sdxl", "illustrious"}

TEXTS = {
    "ja": {
        "not_found": "ファイルが見つかりません: {path}",
        "bad_toml": "TOML を読めません: {path}: {error}",
        "no_output_name": "output_name がありません。学習の TOML に書いてください。",
        "bad_output_name": "output_name に使える文字は英数字と . _ - だけです: {name}",
        "secret": "{key} を TOML に書かないでください。トークンは config/config.local.env に書きます。",
        "no_network": "network_module がありません。このツールは LoRA の学習用です。",
        "no_dataset": "dataset_config も train_data_dir もありません。",
        "te_cache": "テキストのキャッシュと {key} は同時に使えません。どちらかを外してください。",
        "te_dropout": "SDXL ではテキストのキャッシュと caption_dropout_rate を同時に使えません。",
        "te_unet_only": "テキストのキャッシュには network_train_unet_only = true が必要です。",
        "constant_warmup": "lr_scheduler = \"constant\" では lr_warmup_steps を指定できません。外すか constant_with_warmup にしてください。",
        "bad_scheduler": "lr_scheduler = \"{name}\" は学習の開始時に失敗します。別のスケジューラにしてください。",
        "min_lr": "cosine_with_min_lr には lr_scheduler_min_lr_ratio か min_lr のどちらか片方が必要です。",
        "scheduler_args": "lr_scheduler_args は文字列ではなくリストで書いてください。",
        "piecewise": "piecewise_constant には lr_scheduler_args の step_rules が必要です。",
        "save_zero": "save_every_n_epochs = 0 は sd-scripts が 0 で割って失敗します。キーを外してください。",
        "epochs_steps": "max_train_epochs と max_train_steps の両方があります。max_train_epochs が優先されます。",
        "cycles": "lr_scheduler_num_cycles は cosine_with_restarts でないと、学習率が下がってから再び上がることがあります。",
        "quantized_model": "int8_convrot のモデルは学習できません。量子化していない safetensors を指定してください。",
        "swap_limit": "blocks_to_swap は {route} では {limit} までです。",
        "swap_offload": "blocks_to_swap と {key} は同時に使えません。",
        "h2d": "block_swap_h2d_only には gradient_checkpointing が必要です。",
        "no_images": "画像がありません: {path}。読める拡張子は .png .jpg .jpeg .webp .bmp で、すべて小文字かすべて大文字のものです。",
        "no_captions": "キャプションがありません: {path}。caption_extension の {ext} のファイルを画像の隣に置くか、caption_prefix か class_tokens を書いてください。",
        "no_folder": "フォルダーが見つかりません: {path}",
        "no_subset_dir": "subset に image_dir がありません。",
        "bad_subfolder": "{path} に <繰り返し数>_<名前> のフォルダーがありません。",
        "ext_default": "caption_extension がありません。既定値は .caption です。.txt を使うなら明示してください。",
        "placeholder": "{key} が例の値 mytrigger のままです。",
        "ok_local": "自分の設定を重ねます: {path}",
        "ok_images": "データセット {path}: 画像 {images} 枚、キャプション {captions} 件",
        "ok_repeats": "データセット {path}: 画像 {images} 枚、キャプション {captions} 件、繰り返し {repeats} 回",
        "problems": "学習の設定に問題があります。ランタイムは作っていません。",
    },
    "en": {
        "not_found": "File not found: {path}",
        "bad_toml": "Cannot read the TOML file: {path}: {error}",
        "no_output_name": "output_name is missing. Add it to the training TOML.",
        "bad_output_name": "output_name may only use letters, digits and . _ -: {name}",
        "secret": "Do not put {key} in a TOML file. Tokens go in config/config.local.env.",
        "no_network": "network_module is missing. This tool is for LoRA training.",
        "no_dataset": "Neither dataset_config nor train_data_dir is set.",
        "te_cache": "Text encoder caching cannot be used with {key}. Remove one of them.",
        "te_dropout": "SDXL cannot use text encoder caching with caption_dropout_rate.",
        "te_unet_only": "Text encoder caching needs network_train_unet_only = true.",
        "constant_warmup": "lr_scheduler = \"constant\" cannot take lr_warmup_steps. Remove it or use constant_with_warmup.",
        "bad_scheduler": "lr_scheduler = \"{name}\" fails when training starts. Choose another scheduler.",
        "min_lr": "cosine_with_min_lr needs exactly one of lr_scheduler_min_lr_ratio and min_lr.",
        "scheduler_args": "lr_scheduler_args must be a list, not a string.",
        "piecewise": "piecewise_constant needs step_rules in lr_scheduler_args.",
        "save_zero": "save_every_n_epochs = 0 makes sd-scripts divide by zero. Remove the key.",
        "epochs_steps": "Both max_train_epochs and max_train_steps are set. max_train_epochs wins.",
        "cycles": "Outside cosine_with_restarts, lr_scheduler_num_cycles can raise the learning rate again after it falls.",
        "quantized_model": "An int8_convrot model cannot be trained. Use a safetensors file that is not quantized.",
        "swap_limit": "blocks_to_swap is at most {limit} for {route}.",
        "swap_offload": "blocks_to_swap cannot be used with {key}.",
        "h2d": "block_swap_h2d_only needs gradient_checkpointing.",
        "no_images": "No images in {path}. Readable extensions are .png .jpg .jpeg .webp .bmp, all lower case or all upper case.",
        "no_captions": "No captions in {path}. Put {ext} files next to the images, or set caption_prefix or class_tokens.",
        "no_folder": "Folder not found: {path}",
        "no_subset_dir": "A subset has no image_dir.",
        "bad_subfolder": "{path} has no <repeats>_<name> folders.",
        "ext_default": "caption_extension is missing, and its default is .caption. Set it if you use .txt.",
        "placeholder": "{key} still has the example value mytrigger.",
        "ok_local": "Applying your settings: {path}",
        "ok_images": "Dataset {path}: {images} images, {captions} captions",
        "ok_repeats": "Dataset {path}: {images} images, {captions} captions, {repeats} repeats",
        "problems": "The training settings have problems. No runtime was created.",
    },
}
LANG = "en"
errors = []
warnings = []


def t(text_key, **values):
    return TEXTS[LANG][text_key].format(**values)


def fail(text_key, **values):
    errors.append(t(text_key, **values))


def warn(text_key, **values):
    warnings.append(t(text_key, **values))


# ---- TOML ----

def read_toml(path):
    try:
        with path.open("rb") as file:
            return tomllib.load(file)
    except OSError:
        raise SystemExit(t("not_found", path=path)) from None
    except tomllib.TOMLDecodeError as error:
        raise SystemExit(t("bad_toml", path=path, error=error)) from None


def flatten(data):
    """sd-scripts は TOML のセクションを 1 段だけ開いて平らなキーとして読む。"""
    flat = {}
    for key, value in data.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    return flat


def merge(base, local):
    """local の値を base に重ねる。表は中へ入り、表の並びは位置ごとに重ねる。空文字の値はキーを消す。"""
    merged = dict(base)
    for key, value in local.items():
        old = merged.get(key)
        # [[datasets]] を書かずに [[datasets.subsets]] だけを書くと、datasets は並びではなく表になる。最初の要素へ重ねる
        if isinstance(value, dict) and isinstance(old, list):
            value = [value]
        if value == "":
            merged.pop(key, None)
        elif isinstance(value, dict) and isinstance(old, dict):
            merged[key] = merge(old, value)
        elif (isinstance(value, list) and isinstance(old, list)
              and all(isinstance(item, dict) for item in value + old)):
            pairs = [merge(old[i], item) if i < len(old) else item for i, item in enumerate(value)]
            merged[key] = pairs + old[len(value):]
        else:
            merged[key] = value
    return merged


def local_file(path, root):
    """config/<種類>/templates/<名前>.<拡張子> に対応する config/<種類>/<名前>.local.<拡張子> を返す。テンプレートでなければ None。"""
    try:
        parts = path.resolve().relative_to((root / "config").resolve()).parts
    except ValueError:
        return None
    if len(parts) != 3 or parts[1] != "templates":
        return None
    candidate = root / "config" / parts[0] / f"{path.stem}.local{path.suffix}"
    if candidate.is_file():
        print(t("ok_local", path=candidate), file=sys.stderr)
        return candidate
    return None


def read_with_local(path, root):
    data = read_toml(path)
    local = local_file(path, root)
    return merge(data, read_toml(local)) if local else data


def sampling_disabled(config):
    return (config.get("sample_every_n_epochs", 1) <= 0 and config.get("sample_every_n_steps", 0) <= 0
            and not config.get("sample_at_first"))


def toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{json.dumps(k)} = {toml_value(v)}" for k, v in value.items()) + "}"
    raise TypeError(type(value))


def dump_flat(data):
    return "".join(f"{key} = {toml_value(value)}\n" for key, value in data.items())


def dump_dataset(data):
    """データセット TOML を書く。[general] と [[datasets]] と [[datasets.subsets]] の形だけを扱う。"""
    lines = []
    for key, value in data.items():
        if key not in {"general", "datasets"}:
            lines.append(f"{key} = {toml_value(value)}")
    if "general" in data:
        lines += ["[general]"] + [f"{k} = {toml_value(v)}" for k, v in data["general"].items()]
    for dataset in data.get("datasets", []):
        lines.append("[[datasets]]")
        lines += [f"{k} = {toml_value(v)}" for k, v in dataset.items() if k != "subsets"]
        for subset in dataset.get("subsets", []):
            lines.append("[[datasets.subsets]]")
            lines += [f"{k} = {toml_value(v)}" for k, v in subset.items()]
    return "\n".join(lines) + "\n"


# ---- 検査 ----

def check_training(config, route):
    for key in config:
        if SECRET_KEY.search(key):
            fail("secret", key=key)
    name = config.get("output_name", "")
    if not name:
        fail("no_output_name")
    elif not re.fullmatch(r"[A-Za-z0-9._-]+", name) or name.startswith("."):
        fail("bad_output_name", name=name)
    if not config.get("network_module"):
        fail("no_network")
    if not config.get("dataset_config") and not config.get("train_data_dir"):
        fail("no_dataset")

    scheduler = config.get("lr_scheduler", "cosine")
    if scheduler == "constant" and config.get("lr_warmup_steps"):
        fail("constant_warmup")
    if scheduler in BROKEN_SCHEDULERS:
        fail("bad_scheduler", name=scheduler)
    if scheduler == "cosine_with_min_lr" and bool("lr_scheduler_min_lr_ratio" in config) == bool("min_lr" in config):
        fail("min_lr")
    if isinstance(config.get("lr_scheduler_args"), str):
        fail("scheduler_args")
    if scheduler == "piecewise_constant" and not any("step_rules" in str(a) for a in config.get("lr_scheduler_args", [])):
        fail("piecewise")
    if config.get("save_every_n_epochs") == 0:
        fail("save_zero")
    if "max_train_epochs" in config and "max_train_steps" in config:
        warn("epochs_steps")
    if "lr_scheduler_num_cycles" in config and scheduler != "cosine_with_restarts" and not config.get("lr_scheduler_type"):
        warn("cycles")

    if route == "anima" and "int8_convrot" in str(config.get("pretrained_model_name_or_path", "")):
        fail("quantized_model")

    swap = config.get("blocks_to_swap", 0)
    # Anima の上限はモデルのブロック数で変わるので、VM でヘッダーを読んだ extensions/anima_train_dispatcher.py が調べる
    limit = {"moss": 40}.get(route)
    if swap and limit and swap > limit:
        fail("swap_limit", route=route, limit=limit)
    if swap:
        for key in ("cpu_offload_checkpointing", "unsloth_offload_checkpointing"):
            if config.get(key):
                fail("swap_offload", key=key)
    if config.get("block_swap_h2d_only") and not config.get("gradient_checkpointing"):
        fail("h2d")


def text_cached(config):
    return config.get("cache_text_encoder_outputs") or config.get("cache_text_encoder_outputs_to_disk")


def check_caption_options(config, dataset, route):
    if not text_cached(config):
        return
    if not config.get("network_train_unet_only"):
        fail("te_unet_only")
    scopes = [config, dataset.get("general", {})]
    for group in dataset.get("datasets", [{}]):
        scopes.append(group)
        scopes.extend(group.get("subsets", []))
    for key in TE_CACHE_FORBIDS:
        if any(scope.get(key) for scope in scopes):
            fail("te_cache", key=key)
    if route in SDXL_ROUTES and any(scope.get("caption_dropout_rate") for scope in scopes):
        fail("te_dropout")


def listed_images(folder):
    return [p for p in folder.iterdir() if p.is_file() and not p.name.startswith(".")
            and p.suffix.lower() in IMAGE_SUFFIXES and p.suffix in {p.suffix.lower(), p.suffix.upper()}]


def check_images(folder, extension, has_prefix, repeats=None):
    images = listed_images(folder)
    if not images:
        fail("no_images", path=folder)
        return
    captions = sum(1 for p in images if p.with_suffix(extension).is_file())
    if not captions and not has_prefix:
        fail("no_captions", path=folder, ext=extension)
    text = "ok_images" if repeats is None else "ok_repeats"
    print(t(text, path=folder, images=len(images), captions=captions, repeats=repeats), file=sys.stderr)


# ---- パス ----

class Job:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.out)
        self.root = Path(args.root)
        self.vm_root = args.vm_root.rstrip("/")
        self.stage = args.stage.rstrip("/")
        self.sync = []          # VM へ送るフォルダー

    def on_vm(self, value):
        return value == self.vm_root or value.startswith(self.vm_root + "/")

    def local_path(self, value, base):
        """手元のパスを実際の場所にする。TOML のフォルダー、プロジェクトのフォルダーの順に探す。"""
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        return next((c for c in (base / path, self.root / path) if c.exists()), base / path)

    def vm_folder(self, folder):
        """手元のフォルダーを VM へ送る指示を足して、VM 上の場所を返す。同じ名前のフォルダーが重ならないよう、場所のハッシュを足す。"""
        tag = hashlib.sha256(str(folder.resolve()).encode()).hexdigest()[:8]
        name = re.sub(r"[^A-Za-z0-9._-]", "_", folder.name) or "dataset"
        remote = f"{self.vm_root}/datasets/{name}-{tag}"
        if not any(item["local"] == str(folder) for item in self.sync):
            self.sync.append({"local": str(folder), "remote": remote})
        return remote

    def job_file(self, name):
        return f"{self.stage}/jobs/{self.name}/{name}"


def repeat_folders(folder, repeats):
    """folder とその下のフォルダーのうち、画像のあるものを (フォルダー, 繰り返し数) で返す。

    sd-scripts の train_data_dir と同じく、名前が <数>_ で始まるフォルダーはその数を繰り返し数にする。
    名前に数の無いフォルダーは親の繰り返し数を受け継ぎ、0_ で始まるフォルダーは中身ごと使わない。
    """
    found = [(folder, repeats)] if listed_images(folder) else []
    for child in sorted(folder.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        match = re.match(r"([0-9]+)_", child.name)
        if match and int(match[1]) < 1:
            continue
        found += repeat_folders(child, int(match[1]) if match else repeats)
    return found


def prepare_subset(job, subset, general, group, base):
    """subset を調べて、学習に使う subset の並びを返す。手元のフォルダーはサブフォルダーごとに分ける。"""
    value = subset.get("image_dir", "")
    if not value:
        fail("no_subset_dir")
        return [subset]
    extension = subset.get("caption_extension", group.get("caption_extension", general.get("caption_extension")))
    if extension is None:
        warn("ext_default")
        extension = ".caption"
    prefix = subset.get("caption_prefix") or subset.get("class_tokens")
    if prefix and "mytrigger" in prefix:
        warn("placeholder", key="caption_prefix")
    if job.on_vm(value):
        return [subset]
    folder = job.local_path(value, base)
    if not folder.is_dir():
        fail("no_folder", path=folder)
        return [subset]
    folders = repeat_folders(folder, subset.get("num_repeats", 1))
    if not folders:
        fail("no_images", path=folder)
        return [subset]
    # VM へはフォルダーを丸ごと送り、subset はその中の場所を指す
    remote = job.vm_folder(folder)
    subsets = []
    for found, repeats in folders:
        check_images(found, extension, bool(prefix), repeats)
        part = found.relative_to(folder).as_posix()
        subsets.append({**subset, "image_dir": remote if part == "." else f"{remote}/{part}", "num_repeats": repeats})
    return subsets


def prepare_dataset(job, config, base):
    """dataset_config を読み、手元の画像のフォルダーを VM の場所に書き換えて書き出す。読んだ内容を返す。"""
    value = config.get("dataset_config", "")
    if not value:
        return {}
    if job.on_vm(value):
        return {}
    path = job.local_path(value, base)
    if not path.is_file():
        fail("not_found", path=path)
        return {}
    dataset = read_with_local(path, job.root)
    general = dataset.get("general", {})
    for group in dataset.get("datasets", []):
        group["subsets"] = [part for subset in group.get("subsets", [])
                            for part in prepare_subset(job, subset, general, group, path.parent)]
    (job.out / "dataset.toml").write_text(dump_dataset(dataset), encoding="utf-8")
    config["dataset_config"] = job.job_file("dataset.toml")
    return dataset


def prepare_train_data_dir(job, config, base):
    value = config.get("train_data_dir", "")
    if not value or config.get("dataset_config") or job.on_vm(value):
        return
    folder = job.local_path(value, base)
    if not folder.is_dir():
        fail("no_folder", path=folder)
        return
    extension = config.get("caption_extension", ".caption")
    subsets = [d for d in folder.iterdir() if d.is_dir() and re.match(r"[1-9][0-9]*_", d.name)]
    if not subsets:
        fail("bad_subfolder", path=folder)
    for subset in subsets:
        check_images(subset, extension, bool(config.get("class_tokens") or config.get("caption_prefix")))
    config["train_data_dir"] = job.vm_folder(folder)


def prepare_prompts(job, config, base):
    value = config.get("sample_prompts", "")
    if value and sampling_disabled(config):
        config.pop("sample_prompts")
        return
    if not value or job.on_vm(value):
        return
    path = job.local_path(value, base)
    if not path.is_file():
        fail("not_found", path=path)
        return
    path = local_file(path, job.root) or path
    text = path.read_text(encoding="utf-8")
    if "mytrigger" in re.sub(r"(?m)^#.*$", "", text):
        warn("placeholder", key="sample_prompts")
    shutil.copyfile(path, job.out / "sample_prompts.txt")
    config["sample_prompts"] = job.job_file("sample_prompts.txt")


def read_catalog(path):
    """ファイル名と URL の対応を読む。1 行は「URL」か「URL ファイル名」。models.local.txt があれば後から読んで上書きする。"""
    catalog = {}
    for source in (path, path.with_name(path.stem + ".local" + path.suffix)):
        if not source.is_file():
            continue
        for line in source.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts and not parts[0].startswith("#"):
                name = parts[1] if len(parts) > 1 else unquote(urlsplit(parts[0]).path.rsplit("/", 1)[-1])
                catalog[name] = parts[0]
    return catalog


def prepare_models(job, config):
    """モデルのキーを VM のパスにして、VM に必要なファイルの一覧を返す。

    ファイル名だけの値は models フォルダーを指す。/ を含み VM のパスでないものは Hugging Face のリポジトリ名として、そのまま渡す。
    """
    catalog = read_catalog(Path(job.args.catalog))
    needed = []
    for key in MODEL_KEYS:
        value = config.get(key)
        if not isinstance(value, str) or not value:
            continue
        if "/" not in value:
            value = config[key] = f"{job.vm_root}/models/{value}"
        elif not job.on_vm(value):
            continue
        needed.append({"key": key, "path": value, "url": catalog.get(value.rsplit("/", 1)[-1], "")})
    return needed


def main():
    global LANG
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--vm-root", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--lang", default="en")
    args = parser.parse_args()
    LANG = "ja" if args.lang == "ja" else "en"

    path = Path(args.training)
    data = read_toml(path)
    local = local_file(path, Path(args.root))
    # 学習の TOML は平らなキーとして読むので、セクションが違っても同じキーなら上書きできるよう、開いてから重ねる
    config = merge(flatten(data), flatten(read_toml(local))) if local else flatten(data)
    out = Path(args.out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    job = Job(args)
    job.name = config.get("output_name") or "job"
    # 省略したときは学習の種類と同じ名前の原本を使う。local があれば、いつもどおり原本に重なる
    if not config.get("dataset_config") and not config.get("train_data_dir"):
        config["dataset_config"] = str(job.root / "config" / "datasets" / "templates" / f"{args.route}.toml")
    if not config.get("sample_prompts"):
        config["sample_prompts"] = str(job.root / "config" / "prompts" / "templates" / f"{args.route}.txt")
    check_training(config, args.route)
    dataset = prepare_dataset(job, config, path.parent)
    prepare_train_data_dir(job, config, path.parent)
    check_caption_options(config, dataset, args.route)
    prepare_prompts(job, config, path.parent)
    models = prepare_models(job, config)

    for message in warnings:
        print("  ! " + message, file=sys.stderr)
    if errors:
        print(t("problems"), file=sys.stderr)
        for message in errors:
            print("  - " + message, file=sys.stderr)
        return 1

    config["logging_dir"] = job.job_file("logs")
    for key in ("output_dir", "resume", "initial_epoch", "initial_step"):
        config.pop(key, None)
    (out / "training.toml").write_text(dump_flat(config), encoding="utf-8")
    (out / "job.json").write_text(json.dumps({
        "name": job.name, "route": args.route, "entrypoint": args.entrypoint, "sync": job.sync, "models": models,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(job.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
