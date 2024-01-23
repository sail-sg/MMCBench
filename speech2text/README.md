# Self-Consistency Evaluation in Speech-to-Text Generation

## Introduction

This project focuses on evaluating the self-consistency of speech-to-text (STT) generation models, particularly under various speech corruptions. We analyze the robustness of these models against a range of audio corruptions using a subset of the Common Voice 15.0 dataset. Our goal is to understand how well these models maintain transcription consistency when faced with corrupted speech.

#### Speech Corruptions

Our study includes 16 types of audio corruptions sourced from Audiomentations, categorized as follows:

- **Noise Additions and Interference:** Gaussian Noise, Short Noises, Gaussian SNR.
- **Filtering and Frequency Adjustments:** Bandpass Filter, Low Pass Filter.
- **Distortion and Audio Quality Effects:** Clipping Distortion, MP3 Compression, Tanh Distortion.
- **Temporal and Speed Modifications:** Fast Resample, Slow Resample, Time Stretch (Fast), Time Stretch (Slow).
- **Pitch and Dynamic Range Adjustments:** Pitch Shift, Gain Transition.
- **Repetitive and Temporal Effects:** Repeat Part, Time Mask.

#### Data Selection

We utilize the Common Voice 15.0 dataset, which includes approximately 1.75 million validated text-speech pairs, known for its diversity. From this, we select 1,000 speeches to highlight the challenges in STT processing. The selection is based on the average cosine similarity between texts generated from corrupted and original audio using baseline models like `speecht5_asr`, `wav2vec2-base-960h`, and `whisper-base.en`. We prioritize speeches with significant drops in text similarity post-corruption. The original audio is maintained at a sampling rate of 16 kHz.

## Usage

### Download Data

The data is avialable at [huggingface](https://huggingface.co/datasets/javyduck/MMCBench/tree/main/speech2text). You can manually download the required files and place them into the `data` directory.

Alternatively, for your convenience, you can simply execute the script `bash download_data.sh`. This script will automatically download and organize the necessary data into the data directory for you.

### Define Your Model

Incorporate your model within the `TranscriptionGenerator` class, defined in `speech_architecture.py`. This class should accept audio input and generate corresponding transcriptions.

### Evaluation Script

Execute `evaluation.py` with specific command-line arguments to assess the performance of STT models under various speech corruptions.

#### Command Format

`python evaluation.py --model [MODEL_NAME] --level [LEVEL] --degree [DEGREE] --batch_size [BATCH_SIZE] --gpu [GPU_ID] --store_output`

- `--model`: Name of the STT model (e.g., `wav2vec2-base-960h`).
- `--level`: Dataset level (choices: `hard_1k`, `random_1k`).
- `--degree`: Degree of audio manipulation (choices: `heavy`, `light`).
- `--batch_size`: Number of samples to process in a batch.
- `--gpu`: GPU ID for computation.
- `--store_output`: Flag to store output transcriptions.

**Example:**

`python evaluation.py --model wav2vec2-base-960h --level hard_1k --degree heavy --batch_size 5 --gpu 0 --store_output`