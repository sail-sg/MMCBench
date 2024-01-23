# Self-Consistency Evaluation in Text-to-Speech Generation

## Introduction

This section focuses on assessing the self-consistency of text-to-speech (TTS) generation models, particularly in the context of handling text corruptions. Our study probes the resilience of these models against various text corruption types, sourced from platforms such as NlpAug, TextAugment, and NL-Augmenter. The evaluation leverages a carefully curated subset of the Common Voice 15.0 dataset, with an emphasis on speech pairs that present significant linguistic and acoustic challenges.

#### Text Corruptions

We incorporate a total of 23 distinct types of text corruption, systematically categorized into three complexity levels:

- **Char Level:** Substitute Char by OCR, Substitute Char by Keyboard, Insert Char Randomly, Substitute Char Randomly, Swap Char Randomly, Delete Char Randomly, Uppercase Char Randomly, Repeat Characters, Leet Letters, Whitespace Perturbation, Substitute with Homoglyphs
- **Word Level:** Synonym Replacement, Random Deletion, Random Swap, Random Insertion, Misspell Word, Abbreviate Word, Multilingual Dictionary Based Code Switch, Close Homophones Swap
- **Sentence Level:** CheckList, Back Translation, Style Paraphraser, Paraphrase

#### Data Selection

From the extensive collection of 1.75 million validated text-speech pairs in the Common Voice 15.0 dataset, we select 1,000 pairs that are inherently complex. This selection is based on four scores: inconsistency score, readability score, syntax complexity score, and description score.

## Usage

### Define Your Model

Encapsulate your model within the `SpeechSynthesizer` class, as defined in `architecture.py`. This class should accept transcriptions as input and generate the corresponding speech outputs.

### Evaluation Script

Use `evaluation.py` with specific command-line arguments to evaluate the TTS model's performance under various text corruptions.

#### Command Format

```
python evaluation.py --model_name [MODEL_NAME] --level [LEVEL] --degree [DEGREE] --batch_size [BATCH_SIZE] --gpu [GPU_ID] --store_output
```

- `--model_name`: Name of the TTS model (e.g., `bark`).
- `--level`: Dataset level (e.g., `hard_1k`).
- `--degree`: Degree of data manipulation (choices: `heavy`, `light`).
- `--batch_size`: Number of samples to process at a time.
- `--gpu`: GPU ID for computation.
- `--store_output`: Flag to store output speech.

**Example:**

```
python evaluation.py --model_name bark --level hard_1k --degree heavy --batch_size 2 --gpu 0 --store_output
```

**Note:** The current evaluation is limited to unimodality similarity. We will include more comprehensive evaluations with cross-modality similarity soon.