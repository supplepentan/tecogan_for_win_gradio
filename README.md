# TecoGAN-PyTorch for Windows

### Introduction

This is a reimplementation of **TecoGAN** (Temporally Coherent GAN) for Video Super-Resolution (VSR) on Windows environment,
browser app with only inference function using Gradio. Please refer to the official TensorFlow implementation [TecoGAN-TensorFlow](https://github.com/thunil/TecoGAN) and [TecoGAN-PyTorch](https://github.com/skycrapers/TecoGAN-PyTorch) for more information.

### Updates

- 11/2023: Release TecoGAN with Gradio on Windows environment.
- 09/2024: Add command mode and web mode.
- 06/2025: Add Training Module.

#### Original TecoGAN-PyTorch

- 11/2021: Supported 2x SR.
- 10/2021: Supported model training/testing on the [REDS](https://seungjunnah.github.io/Datasets/reds.html) dataset.
- 07/2021: Upgraded codebase to support multi-GPU training & testing.

## Dependencies

- Windows
- NVIDIA GPU + CUDA
- Python >= 3.10
- PyTorch >= 2.0

## Setting

### 1. Download pre-trained TecoGAN models.

Download the model from [[BD-4x-Vimeo](https://drive.google.com/file/d/13FPxKE6q7tuRrfhTE7GB040jBeURBj58/view?usp=sharing)][[BI-4x-Vimeo](https://drive.google.com/file/d/1ie1F7wJcO4mhNWK8nPX7F0LgOoPzCwEu/view?usp=sharing)][[BD-4x-REDS](https://drive.google.com/file/d/1vMvMbv_BvC2G-qCcaOBkNnkMh_gLNe6q/view?usp=sharing)][[BD-2x-REDS](https://drive.google.com/file/d/1XN5D4hjNvitO9Kb3OrYiKGjwNU0b43ZI/view?usp=sharing)], and put it under `./pretrained_models`.

These tags represent different configurations used during the training or testing of TecoGAN. Each tag specifies the dataset being used, the degradation model applied, and the scale of super-resolution.

**[BD-4x-Vimeo]**: In this configuration, the Vimeo dataset is used with a BD (Blur and Downsample) degradation model, and a 4x super-resolution is applied.

**[BI-4x-Vimeo]**: In this configuration, the Vimeo dataset is used with a BI (Bicubic Interpolation) degradation model, and a 4x super-resolution is applied.

**[BD-4x-REDS]**: In this configuration, the REDS dataset is used with a BD degradation model, and a 4x super-resolution is applied.

**[BD-2x-REDS]**: In this configuration, the REDS dataset is used with a BD degradation model, and a 2x super-resolution is applied.

These configurations can be chosen according to the specific task or requirements. For example, if a 4x super-resolution is needed and the expected degradation is blur and downsample, the [BD-4x-Vimeo] or [BD-4x-REDS] configuration would be suitable. If the model's performance on a specific dataset (Vimeo or REDS) is to be evaluated, then the respective dataset configuration is chosen.

### 2. Run Gradio.

```bash
python main.py
```

## Acknowledgements

This code is built on [TecoGAN-PyTorch](https://github.com/skycrapers/TecoGAN-PyTorch), [TecoGAN-TensorFlow](https://github.com/thunil/TecoGAN), [BasicSR](https://github.com/xinntao/BasicSR) and [LPIPS](https://github.com/richzhang/PerceptualSimilarity). We thank the authors for sharing their codes.

## Training

**Note:** Due to the inaccessibility of the VimeoTecoGAN dataset, we recommend using other public datasets, e.g., REDS, for model training. To use REDS as the training dataset, just download it from here and replace the following `VimeoTecoGAN` to `REDS`.

1. Download the official training dataset according to the instructions in TecoGAN-TensorFlow, rename to `VimeoTecoGAN/Raw`, and place under `training_module/data`.

2. Generate LMDB for GT data to accelerate IO. The LR counterpart will then be generated on the fly during training.

```bash
python training_module/scripts/create_lmdb.py --dataset VimeoTecoGAN --raw_dir training_module/data/VimeoTecoGAN/Raw --lmdb_dir training_module/data/VimeoTecoGAN/GT.lmdb
```

The following shows the dataset structure after finishing the above two steps.

```tex
data
  ├─ VimeoTecoGAN
    ├─ Raw                 # Raw dataset
      ├─ scene_2000
        └─ ***.png
      ├─ scene_2001
        └─ ***.png
      └─ ...
    └─ GT.lmdb             # LMDB dataset
      ├─ data.mdb
      ├─ lock.mdb
      └─ meta_info.pkl     # each key has format: [vid]_[total_frame]x[h]x[w]_[i-th_frame]
```

3. **(Optional, this step is only required for BI degradation)** Manually generate the LR sequences with the Matlab's imresize function, and then create LMDB for them.

```bash
# Generate the raw LR video sequences. Results will be saved at training_module/data/VimeoTecoGAN/Bicubic4xLR
matlab -nodesktop -nosplash -r "cd training_module/scripts; generate_lr_bi"

# Create LMDB for the LR video sequences
python training_module/scripts/create_lmdb.py --dataset VimeoTecoGAN --raw_dir training_module/data/VimeoTecoGAN/Bicubic4xLR --lmdb_dir training_module/data/VimeoTecoGAN/Bicubic4xLR.lmdb
```

4. Train a FRVSR model first, which can provide a better initialization for the subsequent TecoGAN training. FRVSR has the same generator as TecoGAN, but without perceptual training (GAN and perceptual losses).

```bash
bash training_module/train.sh BD FRVSR/FRVSR_VimeoTecoGAN_4xSR_2GPU
```

> You can download and use our pre-trained FRVSR models instead of training from scratch. [BD-4x-Vimeo] [BI-4x-Vimeo] [BD-4x-REDS][BD-2x-REDS]

When the training is complete, set the generator's `load_path` in `training_module/experiments_BD/TecoGAN/TecoGAN_VimeoTecoGAN_4xSR_2GPU/train.yml` to the latest checkpoint weight of the FRVSR model.

5. Train a TecoGAN model. You can specify which gpu to be used in `training_module/train.sh`. By default, the training is conducted in the background and the output info will be logged in `training_module/experiments_BD/TecoGAN/TecoGAN_VimeoTecoGAN/train/train.log`.

```bash
bash training_module/train.sh BD TecoGAN/TecoGAN_VimeoTecoGAN_4xSR_2GPU
```

6. Run the following script to monitor the training process and visualize the validation performance.

```bash
python training_module/scripts/monitor_training.py -dg BD -m TecoGAN/TecoGAN_VimeoTecoGAN_4xSR_2GPU -ds Vid4
```

> Note that the validation results are NOT exactly the same as the testing results mentioned above due to different implementation of the metrics. The differences are caused by croping policy, LPIPS version and some other issues.

<p align = "center">
    <img src="training_module/resources/losses.png" width="1080" />
    <img src="training_module/resources/metrics.png" width="1080" />
</p>

### Training with Custom Videos (Simplified)

The following steps provide a simplified, Python-only workflow for training a model on your own video files, as an alternative to the LMDB/Matlab-based method described above. This method relies on custom scripts that we have refined.

**Prerequisites:**

- Ensure all script modifications from our previous discussions have been applied.
- **Note:** All commands in this section are expected to be run from the root directory of the project (`d:\projects-d\tecogan_for_win_gradio`).

**Step 1: Prepare Your Training Videos**

1.  Create a new folder named `my_training_videos` inside the `training_module` directory.
2.  Place your training video files (e.g., `.mp4`, `.mov`) into the `training_module/my_training_videos` folder.

**Step 2: Generate the Image Dataset**

These scripts will convert your videos into paired HR (High-Resolution) and LR (Low-Resolution) image frames.

1.  **Extract HR Frames:** Run the following command to convert videos into sequences of PNG images.

    ```bash
    python training_module/scripts/video_to_frames.py --video_dir training_module/my_training_videos --out_dir training_module/datasets/MyCustomDataset/HR
    ```

2.  **Generate LR Frames:** Next, create the corresponding downscaled LR images from the HR frames.

    ```bash
    python training_module/scripts/generate_lr_bd.py --hr_dir training_module/datasets/MyCustomDataset/HR --lr_dir training_module/datasets/MyCustomDataset/LR --scale 4 --sigma 1.5
    ```

**Step 3: Configure `train.yml`**

Ensure your `train.yml` (e.g., `training_module/options/train/my_custom_train.yml`) is configured to use the custom dataset loader (`MyPairedFolder`) and points to the folders created in Step 2.

```yaml
dataset:
  degradation:
    type: BD

  train:
    name: MyPairedFolder
    hr_root: training_module/datasets/MyCustomDataset/HR
    lr_root: training_module/datasets/MyCustomDataset/LR
    # ... other settings
```

**Step 4: Start Training**

With the dataset and configuration ready, start the training process.

```bash
python training_module/codes/main.py --exp_dir training_module/experiments_BD/MyModel --mode train --opt training_module/options/train/my_custom_train.yml --gpu_ids 0
```

The trained model checkpoints will be saved in the `training_module/experiments_BD/MyModel/train/ckpt/` directory.
