# MMCBench: Benchmarking Large Multimodal Models against Common Corruptions 🚀

## Overview

MMCBench is a comprehensive benchmarking framework designed to evaluate the robustness and self-consistency of Large Multimodal Models (LMMs) under common corruption scenarios. This framework focuses on cross-modal interactions involving text, image, and speech, covering essential generative tasks such as text-to-image, image-to-text, text-to-speech, and speech-to-text. Our benchmarking approach uses a novel methodology for selecting representative examples from large datasets and employs a consistent metric system for performance measurement across various cross-modalities.

## Benchmarking Process 📈

The selection and evaluation process for cross-modality consistency in MMCBench involves two main steps:

1. **Selection Process** 🕵️‍♂️: This step involves determining similarity based on text modality, using model-generated captions or transcriptions for non-text inputs, and directly comparing text inputs before and after corruption.

2. **Evaluation Process** 📝: This step measures self-consistency by comparing clean inputs with outputs from corrupted inputs and comparing outputs from clean and corrupted inputs against each other.

### Overview of the Selection and Evaluation Process 📌

![Selection and Evaluation Process](figs/pipeline.pdf)

## Model Resilience Analysis 🛡️

We present radar charts depicting the relative consistency scores of selected models for various corruptions across four cross-modality tasks: text-to-image 🎨, image-to-text 📜, text-to-speech 🗣️, and speech-to-text 📝. The scores are normalized with the highest scoring model set as the baseline for each type of corruption, allowing for a comparative analysis of each model's resilience.

### Radar Charts of Model Consistency Scores 🎯

![Radar Charts](figs/radar.pdf)

## Repository Structure 📂

- `MMCBench/`
  - `image2text/`: Image-to-Text generation tasks.
  - `speech2text/`: Speech-to-Text generation tasks.
  - `text2image/`: Text-to-Image generation tasks.
  - `text2speech/`: Text-to-Speech generation tasks.

## Getting Started 🚦

To begin using MMCBench, clone this repository and follow the setup instructions in each module. Detailed documentation for each step of the benchmarking process is provided.

## Contributions 👐

MMCBench is an open-source project, and contributions are welcome. If you wish to contribute, please submit a pull request or open an issue to discuss your proposed changes.

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 🎉

We thank all contributors and participants who have made MMCBench a comprehensive benchmark for evaluating large multimodal models.