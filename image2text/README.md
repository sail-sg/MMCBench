#  Self-Consistency Evaluation in Image-to-Text Generation

## Introduction

This section focuses on the robustness and self-consistency of image-to-text generation models, particularly in their responses to a variety of image corruptions. We analyze the resilience of these models against numerous types of image distortions, drawing from libraries such as imagecorruptions and imgaug, and assessing their performance on a curated subset of the LAION-Aesthetics dataset. Our goal is to understand how well these models maintain consistent caption generation despite image quality degradation.

#### Image Corruptions

We incorporate 29 distinct types of image corruption, sourced from the imagecorruptions library and imgaug. These corruptions are methodically categorized into several groups:

- **Noise-Related:** Gaussian Noise, Shot Noise, Impulse Noise, Speckle Noise
- **Blur-Related:** Defocus Blur, Glass Blur, Motion Blur, Zoom Blur, Gaussian Blur
- **Weather Conditions:** Snow, Frost, Fog
- **Digital:** Brightness, Contrast, Pixelate, JPEG Compression, Spatter, Saturate, Gamma Contrast
- **Arithmetic:** Cutout, Salt and Pepper, Coarse Dropout
- **Geometric:** Scale, Rotate, Shear, Piecewise Affine, Jigsaw
- **Edge:** Canny

This comprehensive selection results in a wide array of corruptions for robustness analysis.

#### Data Selection

From the LAION-Aesthetics dataset, we select 3 million images and further refine this to 1,000 images based on visual quality and challenge levels for multimodal models. The inconsistency score is determined by one minus the average cosine similarity between the text generated from original uncorrupted images and the text from images subjected to the 15 common corruptions of ImageNet-C at severity level 3. The selection prioritizes the 1,000 images with the highest inconsistency scores, utilizing outputs from baseline models: vitgpt2, blip-base, and git-base. The chosen images are maintained at a consistent size of 384 x 384 pixels.

## Usage

### Download Data

The data is avialable at [huggingface](https://huggingface.co/datasets/javyduck/MMCBench/tree/main/image2text). You can manually download the required files and place them into the `data` directory.

Alternatively, for your convenience, you can simply execute the script `bash download_data.sh`. This script will automatically download and organize the necessary data into the data directory for you.

### Define Your Model

Encapsulate your model within the `CaptionGenerator` class, as defined in `architecture.py`. This class should accept images as input and generate the corresponding text captions as output.

### Evaluation Script

Use `evaluation.py` with specific command-line arguments to evaluate the text generator's performance under various image corruptions.

#### Command Format

```
python evaluation.py --model_name [MODEL_NAME] --level [LEVEL] --degree [DEGREE] --batch_size [BATCH_SIZE] --gpu [GPU_ID] --store_output
```

- `--model_name`: Name of the model (e.g., `blip-base`).
- `--level`: Dataset level (choices: `hard_1k`, `random_1k`).
- `--degree`: Degree of image manipulation (choices: `heavy`, `light`).
- `--batch_size`: Number of images to process at a time.
- `--gpu`: GPU ID for computation.
- `--store_output`: Flag to store output text.

**Example:**

```
python evaluation.py --model_name blip-base --level hard_1k --degree heavy --batch_size 10 --gpu 0 --store_output
```