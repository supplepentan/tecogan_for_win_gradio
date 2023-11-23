# TecoGAN-PyTorch for Windows

### Introduction

This is a reimplementation of **TecoGAN** (Temporally Coherent GAN) for Video Super-Resolution (VSR) on Windows environment,
browser app with only inference function using Gradio. Please refer to the official TensorFlow implementation [TecoGAN-TensorFlow](https://github.com/thunil/TecoGAN) and [TecoGAN-PyTorch](https://github.com/skycrapers/TecoGAN-PyTorch) for more information.

### Updates

- 11/2023: Release TecoGAN with Gradio on Windows environment.

#### Original TecoGAN-PyTorch

- 11/2021: Supported 2x SR.
- 10/2021: Supported model training/testing on the [REDS](https://seungjunnah.github.io/Datasets/reds.html) dataset.
- 07/2021: Upgraded codebase to support multi-GPU training & testing.

## Dependencies

- Windows
- NVIDIA GPU + CUDA 11.7
- Python >= 3.10
- PyTorch >= 2.0
- Python packages: numpy, matplotlib, opencv-python, pyyaml, lmdb, Gradio, PyYAML, scipy, scikit-image, tqdm, IPython

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
