# Self-Consistency Evaluation in Text-to-Image Generation

## Introduction

This part is centered on assessing the self-consistency of text-to-image generation models, particularly in the context of handling text corruptions. We investigate the resilience of these models against a wide array of text corruption types, drawing from sources such as NlpAug, TextAugment, and NL-Augmenter. Our evaluation is based on a meticulously selected subset of the LAION-COCO dataset, emphasizing captions that pose challenges in maintaining output consistency under text corruption.

#### Text Corruptions

We incorporate a total of 23 distinct types of text corruption, sourced from various platforms including NlpAug, TextAugment, and NL-Augmenter. These corruptions are systematically categorized into three complexity levels:

- **Char Level:** Includes corruptions like Substitute Char by OCR, Substitute Char by Keyboard, and others.
- **Word Level:** Consists of Synonym Replacement, Random Deletion, Random Swap, and more.
- **Sentence Level:** Features CheckList, Back Translation, Style Paraphraser, and Paraphrase.

#### Data Selection

Our initial dataset comprises 10 million caption-image pairs from the LAION-COCO dataset. From this, we select 1,000 captions that are inherently complex and rich in description, based on four scores: inconsistency score, readability score, syntax complexity score, and description score. The final selection also undergoes a filtering process based on aesthetic scores to ensure visual relevance.

## Usage

### Define Your Model

Encapsulate your model within the `ImageGenerator` class, as defined in `architecture.py`. This class should accept captions as input and generate the corresponding images as output.

### Evaluation Script

Use `evaluation.py` with specific command-line arguments to evaluate the image generator's performance under various text corruptions.

#### Command Format

`python evaluation.py --model_name [MODEL_NAME] --level [LEVEL] --degree [DEGREE] --batch_size [BATCH_SIZE] --gpu [GPU_ID] --store_output`

- `--model_name`: Name of the model (e.g., `stable-diffusion-v1-5`).
- `--level`: Dataset level (choices: `hard_1k`, `random_1k`).
- `--degree`: Degree of data manipulation (choices: `heavy`, `light`).
- `--batch_size`: Number of samples to process at a time.
- `--gpu`: GPU ID for computation.
- `--store_output`: Flag to store output images.

**Example:**

`python evaluation.py --model_name stable-diffusion-v1-5 --level hard_1k --degree heavy --batch_size 10 --gpu 0 --store_output`