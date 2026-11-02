# sd-scripts-colab-cli

[English](README.md)

[sd-scripts](https://github.com/kohya-ss/sd-scripts) の LoRA 学習を、Google Colab の GPU で動かすためのツールです。
Windows の PC から学習の設定を渡すと、Colab のランタイムの作成、データセットの送信、学習、結果の回収までを行います。

対応している学習は、Anima、MOSS-Image、SDXL、Illustrious、SD 1.5 の LoRA です。

## 必要なもの

- Windows 10 または 11 と、WSL2 の Ubuntu
- GPU を使える Colab の有料プラン
- Google アカウント

## 準備

1. `setup-wsl.bat` を実行します。WSL に必要なものと [Colab CLI](https://github.com/googlecolab/google-colab-cli) が入ります。
2. WSL で次のコマンドを実行し、表示に従って Google アカウントを認証します。

```bash
colab usage
```

設定は `config/config.env` にあります。変えたい値は、同じフォルダーに `config/config.local.env` を作って書きます。`config.env` は書き換えずに、こちらの値で上書きできます。

```bash
# config/config.local.env
GPU=A100
```

よく変える設定は次のとおりです。

| 設定 | 意味 | 既定値 |
|---|---|---|
| `GPU` | 使う GPU。`T4`、`L4`、`G4`、`H100`、`A100` | `L4` |
| `USE_DRIVE` | Google ドライブに接続するか。`ask` なら毎回尋ねる | `ask` |
| `SAVE_TO` | 学習したモデルの保存先。`ask`、`local`、`drive` | `ask` |
| `STOP_AFTER_TRAINING` | 1 にすると、学習が成功して保存を確かめた後にランタイムを削除する | `0` |

## 学習

Anima を例にします。ほかのモデルも手順は同じです。

1. 画像と、同じ名前の `.txt` のキャプションを `datasets/my-chara` に置きます。
2. 次の 3 つのファイルを作ります。

```toml
# config/training/anima.local.toml
output_name = "my-chara"
```

```toml
# config/datasets/anima.local.toml
[[datasets.subsets]]
image_dir = "datasets/my-chara"
caption_prefix = "mychara,"
```

```text
# config/prompts/anima.local.txt
masterpiece, best quality, mychara, 1girl, cowboy shot --w 832 --h 1216 --s 30 --l 5 --d 42
```

3. `start-training_Anima.bat` を実行します。

ランタイムが無ければ作るところから始めるので、先に `start.bat` を実行する必要はありません。
学習のログはウィンドウに表示され、終わると結果を回収します。

4. 使い終わったら、`stop.bat` でランタイムを削除します。

> [!IMPORTANT]
> ランタイムは学習が終わっても動き続け、そのあいだ課金されます。使い終わったら必ず `stop.bat` で削除してください。

## 操作の一覧

どの操作も、Windows の `.bat` と WSL の `.sh` の組になっています。

| 操作 | Windows | WSL |
|---|---|---|
| ランタイムを作り、sd-scripts を準備する | `start.bat` | `./up.sh` |
| Anima の LoRA を学習する | `start-training_Anima.bat` | `./start-training_Anima.sh` |
| MOSS-Image の LoRA を学習する | `start-training_MOSS.bat` | `./start-training_MOSS.sh` |
| SDXL の LoRA を学習する | `start-training_SDXL.bat` | `./start-training_SDXL.sh` |
| Illustrious の LoRA を学習する | `start-training_Illustrious.bat` | `./start-training_Illustrious.sh` |
| SD 1.5 の LoRA を学習する | `start-training_SD15.bat` | `./start-training_SD15.sh` |
| 学習のログを表示し直す | `logs.bat` | `./logs.sh` |
| 学習を止める | `stop-training.bat` | `./stop-training.sh` |
| 学習結果を手元にコピーする | `pull-outputs.bat` | `./pull-outputs.sh` |
| モデルをダウンロードする | `model-download.bat` | `./model-download.sh` |
| 状態を見る | `status.bat` | `./status.sh` |
| ランタイムを削除する | `stop.bat` | `./down.sh` |
| VM に入る | `ssh.bat` | `./ssh.sh` |

学習のウィンドウで Ctrl+C を押したりウィンドウを閉じたりしても、止まるのは表示だけです。学習は Colab で続きます。
`logs.bat` で表示し直せます。学習を止めるときは `stop-training.bat` を使います。

## 自分の設定

学習の設定、データセット、生成確認のプロンプトには、それぞれ原本があります。原本は `config/training/templates/`、`config/datasets/templates/`、`config/prompts/templates/` にあり、何も書かなければこれを使います。

原本は書き換えずに、変えたい値だけを local ファイルに書いてください。

| 原本 | 自分の設定 | 重なり方 |
|---|---|---|
| `config/training/templates/anima.toml` | `config/training/anima.local.toml` | 書いたキーだけを上書きします |
| `config/datasets/templates/anima.toml` | `config/datasets/anima.local.toml` | 書いたキーだけを上書きします。`[[datasets.subsets]]` は、上から順に原本の同じ位置の表へ重なります |
| `config/prompts/templates/anima.txt` | `config/prompts/anima.local.txt` | ファイル全体を置き換えます |

ほかのモデルでは、`anima` の部分が `moss`、`sdxl`、`illustrious`、`sd15` になります。

- キーは sd-scripts の引数の名前そのままです。
- 原本にあるキーを消すときは、値を `""` にします。例: `optimizer_args = ""`
- local ファイルでは、`[network]` などのセクションを書いても書かなくてもかまいません。

### 別の TOML を使う

学習の TOML を `.bat` にドラッグするか、`./start-training_Anima.sh my.toml` のように渡します。このときは渡したファイルだけを使い、local ファイルは重ねません。

## データセット

### 画像のフォルダー

`image_dir` に書いたフォルダーは、サブフォルダーまで探します。画像のあるフォルダーは、それぞれ別のデータセットとして学習に使います。

フォルダーの名前を `<数>_<名前>` にすると、その数が繰り返し数になります。sd-scripts の `train_data_dir` と同じ書き方です。

```text
datasets/my-chara/
  10_main/       10 回繰り返す
  3_background/  3 回繰り返す
  0_unused/      使わない
```

- 数の無いフォルダーは、親のフォルダーの繰り返し数を受け継ぎます。
- `image_dir` に直接置いた画像は、`num_repeats` の回数だけ繰り返します。
- 読める画像は `.png`、`.jpg`、`.jpeg`、`.webp`、`.bmp` です。

### Google ドライブの画像を使う

`image_dir` を `/content/drive/MyDrive/` で始めると、PC から送らずにドライブの画像を使えます。ドライブへの接続が必要です。

```toml
[[datasets.subsets]]
image_dir = "/content/drive/MyDrive/datasets/my-chara"
```

## 生成確認

学習の途中で、プロンプトのファイルから画像を生成して仕上がりを確かめます。既定では 1 エポックごとに生成し、画像は学習結果のフォルダーに保存されます。

間隔は学習の local ファイルで変えられます。

```toml
sample_every_n_epochs = 2   # 2 エポックごとに生成する
```

```toml
sample_every_n_epochs = 0   # 生成しない
```

## モデル

### 自動でダウンロードするモデル

`config/models.txt` にあるモデルは、学習を始めるときに自動でダウンロードします。学習の TOML には、ファイル名だけを書きます。

```toml
pretrained_model_name_or_path = "anima-base-v1.0.safetensors"
```

### ほかのモデルを使う

`model-download.bat` を実行して、モデルの URL を貼り付けます。Hugging Face と Civitai はページの URL をそのまま使えます。ほかのサイトも、https の URL なら使えます。

```bash
./model-download.sh https://huggingface.co/owner/repo/blob/main/model.safetensors --to-drive
```

`--to-drive` を付けると、ドライブにも保存して次のランタイムで再利用します。ダウンロードしたら、学習の TOML にファイル名を書きます。

先に TOML へファイル名を書いて学習を始めてもかまいません。モデルが見つからなければ URL を尋ねるので、貼り付けるとダウンロードして学習を続けます。入力した URL は `config/models.local.txt` に残り、次からは尋ねません。

安全のため、pickle 形式のファイルはダウンロードしません。safetensors を使ってください。

### 認証が必要なモデル

トークンを `config/config.local.env` に書きます。トークンは TOML には書かないでください。

```bash
HF_TOKEN=hf_...
CIVITAI_TOKEN=...
```

## Google ドライブと保存先

既定では、ランタイムを作る前に Google ドライブに接続するかを尋ねます。初回は認可の URL が表示されるので、ブラウザーで開いて許可し、ウィンドウに戻って Enter を押します。

学習を始める前には、学習したモデルの保存先を尋ねます。

| 保存先 | 保存の仕方 |
|---|---|
| 手元の PC | 学習中も 5 分ごとに `outputs/<output_name>` へコピーし、終わったら中身が一致するまで確かめます |
| Google ドライブ | `マイドライブ/sd-scripts-colab/outputs/<output_name>` に直接保存します。ランタイムが途中で消えても、そこまでの結果が残ります |

長い学習では、ドライブに保存するのが安全です。Colab の利用上限で、ランタイムが途中で切られることがあるからです。

- 毎回同じ答えにするときは、`config.local.env` に `USE_DRIVE=1` か `0`、`SAVE_TO=local` か `drive` を書きます。
- 1 回だけ変えるときは、学習のコマンドに `--drive`、`--no-drive`、`--save-to drive` などを付けます。
- ドライブの結果を PC にコピーするときは、`./pull-outputs.sh --drive` を実行します。

## 学習の再開

`--resume` を付けると、最後に保存した状態から続けます。

```bash
./start-training_Anima.sh --resume
```

ランタイムが変わっていても、PC かドライブに状態が残っていれば再開できます。ネットワークの形を変えた設定では再開できません。

## モデルごとの使い方

### Anima-2.9B

[Anima-2.9B](https://huggingface.co/Gazingstars123/Anima-2.9B) で学習するときは、学習の local ファイルにモデルを書きます。自動でダウンロードします。

```toml
pretrained_model_name_or_path = "Anima-2.9B-preview-v1.safetensors"
```

- `start-training_Anima.bat` のまま使えます。標準の Anima か 2.9B かは、ファイルの中身から判定します。
- Text Encoder と VAE は、標準の Anima と同じものを使います。
- `*_int8_convrot` のような量子化したモデルでは学習できません。

### Anima-3.8B

[Anima-3.8B](https://huggingface.co/lylogummy/Anima-3.8B) で学習するときは、学習の local ファイルに次の 2 行を書きます。どちらも自動でダウンロードします。

```toml
pretrained_model_name_or_path = "Anima-3.8B-v1.1.safetensors"
qwen35 = "qwen35_4b.safetensors"
```

- 対応しているのは v1.1 だけです。`Anima-3.8B.safetensors` は使えません。
- LoRA は DiT のブロックだけを学習します。
- `cache_text_encoder_outputs = true` が必要です。原本では有効になっています。
- GPU のメモリーを多く使います。L4 以上を使い、足りなければ `blocks_to_swap` を指定してください。

### detail_inv_t

Anima と MOSS-Image では、細部の学習を重視する損失の重み付けを使えます。学習の local ファイルに書きます。

```toml
loss_weighting = "detail_inv_t"
weighting_scheme = "uniform"
detail_inv_t_min = 1.0
detail_inv_t_max = 5.0
```

### カスタムオプティマイザ

[LoRA Easy Training Scripts](https://github.com/67372a/LoRA_Easy_Training_scripts_Backend) のオプティマイザとスケジューラを使えます。ランタイムを準備するときに自動で入ります。すべてのモデルで使えます。

学習の local ファイルに、大文字と小文字も含めて完全な名前を書きます。

```toml
optimizer_type = "LoraEasyCustomOptimizer.fftdescent.FFTDescent"
optimizer_args = ["weight_decay=0.04"]
lr_scheduler = "rex"
lr_scheduler_type = "LoraEasyCustomOptimizer.RexAnnealingWarmRestarts.RexAnnealingWarmRestarts"
lr_scheduler_args = ["min_lr=0.0", "gamma=0.9", "d=0.9"]
lr_scheduler_num_cycles = 3
warmup_ratio = 0.05
```

- スケジューラのステップ数は、データセットから自動で計算します。
- `lr_warmup_steps` は一緒に使えません。ウォームアップは `warmup_ratio` で指定します。

## 学習を始める前の検査

ランタイムを作る前に、PC で設定とデータセットを検査します。画像が無い、使えない組み合わせの設定がある、キーの綴りが違うといった問題があれば、課金が始まる前に理由を表示して止まります。

## 注意

- **課金:** ランタイムは学習が終わっても動き続け、そのあいだ課金されます。`stop.bat` で削除してください。`STOP_AFTER_TRAINING=1` にすると、学習が成功して保存を確かめた後に自動で削除します。
- **自動削除には PC が必要:** 自動削除は PC のウィンドウが行います。学習中に PC の電源を切ったりスリープさせたりすると、学習と保存は続きますが、ランタイムは削除されません。
- **削除の前の回収:** `stop.bat` は、ランタイムに残っている学習結果を PC にコピーしてから削除します。コピーできないときは、削除するかを尋ねます。
- **Google ドライブ:** 学習は Colab の上で管理者の権限で動き、接続した Google ドライブ全体を読み書きできます。個人のデータを置いていない Google アカウントを使ってください。
- **VM に入るとき:** Colab のランタイムには、同時に 1 本しか SSH をつなげません。`colab ssh` ではなく `ssh.bat` を使ってください。
- **ライセンス:** モデルのライセンスは、モデルごとに確かめてください。
- **表示の言語:** 日本語の環境なら日本語、それ以外では英語で表示します。`SD_COLAB_LANG` を `ja` か `en` にすると固定できます。
