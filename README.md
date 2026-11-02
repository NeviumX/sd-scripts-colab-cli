# sd-scripts-colab-cli

[日本語](README_JP.md)

A tool to run [sd-scripts](https://github.com/kohya-ss/sd-scripts) LoRA training on Google Colab GPUs.
Give it your training settings on a Windows PC, and it creates a Colab runtime, sends the dataset, trains, and fetches the results.

It trains LoRAs for Anima, MOSS-Image, SDXL, Illustrious and SD 1.5.

## Requirements

- Windows 10 or 11 with Ubuntu on WSL2
- A paid Colab plan with GPU access
- A Google account

## Setup

1. Run `setup-wsl.bat`. It installs what WSL needs, including the [Colab CLI](https://github.com/googlecolab/google-colab-cli).
2. Run this command in WSL and follow the prompts to sign in to your Google account.

```bash
colab usage
```

Settings are in `config/config.env`. To change a value, create `config/config.local.env` in the same folder and write it there. Its values override `config.env`, which you leave unchanged.

```bash
# config/config.local.env
GPU=A100
```

These are the settings you are most likely to change.

| Setting | Meaning | Default |
|---|---|---|
| `GPU` | The GPU to use: `T4`, `L4`, `G4`, `H100` or `A100` | `L4` |
| `USE_DRIVE` | Whether to connect Google Drive. `ask` asks every time | `ask` |
| `SAVE_TO` | Where to save the trained model: `ask`, `local` or `drive` | `ask` |
| `STOP_AFTER_TRAINING` | Set to 1 to delete the runtime after a successful run once the results are verified | `0` |

## Training run

This example uses Anima. The steps are the same for the other models.

1. Put your images in `datasets/my-chara`, each with a `.txt` caption of the same name.
2. Create these three files.

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

3. Run `start-training_Anima.bat`.

If there is no runtime, one is created first, so you do not need to run `start.bat` beforehand.
The training log shows in the window, and the results are fetched when training ends.

4. When you are done, delete the runtime with `stop.bat`.

> [!IMPORTANT]
> The runtime keeps running after training ends, and you are billed for it. Always delete it with `stop.bat` when you are done.

## Commands

Each command is a pair of a Windows `.bat` and a WSL `.sh`.

| Command | Windows | WSL |
|---|---|---|
| Create a runtime and prepare sd-scripts | `start.bat` | `./up.sh` |
| Train an Anima LoRA | `start-training_Anima.bat` | `./start-training_Anima.sh` |
| Train a MOSS-Image LoRA | `start-training_MOSS.bat` | `./start-training_MOSS.sh` |
| Train an SDXL LoRA | `start-training_SDXL.bat` | `./start-training_SDXL.sh` |
| Train an Illustrious LoRA | `start-training_Illustrious.bat` | `./start-training_Illustrious.sh` |
| Train an SD 1.5 LoRA | `start-training_SD15.bat` | `./start-training_SD15.sh` |
| Show the training log again | `logs.bat` | `./logs.sh` |
| Stop training | `stop-training.bat` | `./stop-training.sh` |
| Copy the results to this PC | `pull-outputs.bat` | `./pull-outputs.sh` |
| Download a model | `model-download.bat` | `./model-download.sh` |
| Show the state | `status.bat` | `./status.sh` |
| Delete the runtime | `stop.bat` | `./down.sh` |
| Open a shell on the VM | `ssh.bat` | `./ssh.sh` |

Pressing Ctrl+C or closing the training window only stops the display. Training continues on Colab.
`logs.bat` shows it again. To stop training, use `stop-training.bat`.

## Your settings

The training settings, the dataset and the sample prompts each have an original. The originals are in `config/training/templates/`, `config/datasets/templates/` and `config/prompts/templates/`, and are used when you write nothing.

Do not edit the originals. Write only the values you want to change in local files. Git ignores local files, so updating the tool keeps them.

| Original | Your settings | How it applies |
|---|---|---|
| `config/training/templates/anima.toml` | `config/training/anima.local.toml` | Overrides the keys you write |
| `config/datasets/templates/anima.toml` | `config/datasets/anima.local.toml` | Overrides the keys you write. `[[datasets.subsets]]` entries apply in order to the entry at the same position in the original |
| `config/prompts/templates/anima.txt` | `config/prompts/anima.local.txt` | Replaces the whole file |

For the other models, replace `anima` with `moss`, `sdxl`, `illustrious` or `sd15`.

- Keys are the sd-scripts argument names.
- To remove a key from the original, set it to `""`, as in `optimizer_args = ""`.
- Sections such as `[network]` are optional in local files.

### Using another TOML

Drop a training TOML on the `.bat`, or pass it as in `./start-training_Anima.sh my.toml`. Only that file is used, and local files are not applied.

## Datasets

### Image folders

The folder in `image_dir` is searched at every depth. Each folder with images is trained as its own dataset.

Name a folder `<number>_<name>` to set its number of repeats. This is the same naming as sd-scripts' `train_data_dir`.

```text
datasets/my-chara/
  10_main/       repeated 10 times
  3_background/  repeated 3 times
  0_unused/      not used
```

- A folder without a number takes its parent folder's repeats.
- Images directly in `image_dir` are repeated `num_repeats` times.
- Readable images are `.png`, `.jpg`, `.jpeg`, `.webp` and `.bmp`.

### Images on Google Drive

Start `image_dir` with `/content/drive/MyDrive/` to use images on Drive without sending them from your PC. Drive must be connected.

```toml
[[datasets.subsets]]
image_dir = "/content/drive/MyDrive/datasets/my-chara"
```

## Sample images

During training, images are generated from the prompt file so you can check progress. By default this happens every epoch, and the images are saved in the results folder.

Change the interval in your local training file.

```toml
sample_every_n_epochs = 2   # every 2 epochs
```

```toml
sample_every_n_epochs = 0   # no sample images
```

## Models

### Models downloaded automatically

Models listed in `config/models.txt` are downloaded automatically when training starts. Write only the file name in the training TOML.

```toml
pretrained_model_name_or_path = "anima-base-v1.0.safetensors"
```

### Other models

Run `model-download.bat` and paste the model URL. Hugging Face and Civitai page URLs work as they are. Other sites work with an https URL.

```bash
./model-download.sh https://huggingface.co/owner/repo/blob/main/model.safetensors --to-drive
```

With `--to-drive`, the model is also saved to Drive and reused by the next runtime. After downloading, write the file name in the training TOML.

You can also write the file name in the TOML first and start training. If the model is not found, you are asked for its URL. Paste it, and the model is downloaded and training continues. The URL is kept in `config/models.local.txt`, so you are not asked again.

For safety, pickle files are not downloaded. Use safetensors.

### Models that need a token

Put the tokens in `config/config.local.env`. Do not put them in a TOML file.

```bash
HF_TOKEN=hf_...
CIVITAI_TOKEN=...
```

## Google Drive and the save destination

By default, you are asked whether to connect Google Drive before a runtime is created. The first time, an authorization URL appears. Open it in a browser, allow access, then go back to the window and press Enter.

Before training, you are asked where to save the trained model.

| Destination | How it is saved |
|---|---|
| This PC | Copied to `outputs/<output_name>` every 5 minutes during training, then verified by content at the end |
| Google Drive | Saved directly to `My Drive/sd-scripts-colab/outputs/<output_name>`. Results so far survive if the runtime goes away |

For long runs, Drive is the safer choice, because Colab usage limits can end a runtime partway through.

- To give the same answer every time, write `USE_DRIVE=1` or `0` and `SAVE_TO=local` or `drive` in `config.local.env`.
- To change it for one run, add `--drive`, `--no-drive` or `--save-to drive` to the training command.
- To copy the results on Drive to your PC, run `./pull-outputs.sh --drive`.

## Resuming

`--resume` continues from the last saved state.

```bash
./start-training_Anima.sh --resume
```

This works on a new runtime too, as long as the state is on your PC or on Drive. Settings that change the network shape cannot resume.

## Model-specific usage

### Anima-2.9B

To train on [Anima-2.9B](https://huggingface.co/Gazingstars123/Anima-2.9B), set the model in your local training file. It downloads automatically.

```toml
pretrained_model_name_or_path = "Anima-2.9B-preview-v1.safetensors"
```

- Use `start-training_Anima.bat` as usual. Standard Anima and 2.9B are told apart from the file contents.
- It uses the same text encoder and VAE as standard Anima.
- Quantized models such as `*_int8_convrot` cannot be trained.

### Anima-3.8B

To train on [Anima-3.8B](https://huggingface.co/lylogummy/Anima-3.8B), add these two lines to your local training file. Both download automatically.

```toml
pretrained_model_name_or_path = "Anima-3.8B-v1.1.safetensors"
qwen35 = "qwen35_4b.safetensors"
```

- Only v1.1 is supported. `Anima-3.8B.safetensors` cannot be used.
- The LoRA trains only the DiT blocks.
- `cache_text_encoder_outputs = true` is required. The original already sets it.
- It uses a lot of GPU memory. Use an L4 or better, and set `blocks_to_swap` if it runs out.

### detail_inv_t

Anima and MOSS-Image can use a loss weighting that puts more weight on fine detail. Set it in your local training file.

```toml
loss_weighting = "detail_inv_t"
weighting_scheme = "uniform"
detail_inv_t_min = 1.0
detail_inv_t_max = 5.0
```

### Custom optimizers

The optimizers and schedulers from [LoRA Easy Training Scripts](https://github.com/67372a/LoRA_Easy_Training_scripts_Backend) are installed automatically when the runtime is prepared. They work with every model.

Write the full name in your local training file. Names are case-sensitive.

```toml
optimizer_type = "LoraEasyCustomOptimizer.fftdescent.FFTDescent"
optimizer_args = ["weight_decay=0.04"]
lr_scheduler = "rex"
lr_scheduler_type = "LoraEasyCustomOptimizer.RexAnnealingWarmRestarts.RexAnnealingWarmRestarts"
lr_scheduler_args = ["min_lr=0.0", "gamma=0.9", "d=0.9"]
lr_scheduler_num_cycles = 3
warmup_ratio = 0.05
```

- The scheduler step counts are computed from the dataset.
- `lr_warmup_steps` cannot be used with these schedulers. Set the warmup with `warmup_ratio`.

## Checks before training

Before a runtime is created, your settings and dataset are checked on your PC. Missing images, settings that cannot be combined and misspelled keys stop with a reason before billing starts.

## Notes

- **Billing:** The runtime keeps running after training ends, and you are billed for it. Delete it with `stop.bat`. With `STOP_AFTER_TRAINING=1`, it is deleted automatically after a successful run once the results are verified.
- **Automatic deletion needs your PC:** Your PC's window does the automatic deletion. If you turn off or sleep your PC during training, training and saving continue, but the runtime is not deleted.
- **Copying before deletion:** `stop.bat` copies the results left on the runtime to your PC before deleting it. If that fails, it asks before deleting.
- **Google Drive:** Training runs with administrator rights on Colab and can read and write all of the connected Google Drive. Use a Google account without personal data.
- **Opening a shell on the VM:** A Colab runtime accepts only one SSH connection at a time. Use `ssh.bat`, not `colab ssh`.
- **Licenses:** Check the license of each model yourself.
- **Language:** Messages are in Japanese on a Japanese system and in English otherwise. Set `SD_COLAB_LANG` to `ja` or `en` to fix the language.
