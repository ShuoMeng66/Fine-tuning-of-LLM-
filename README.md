
# LLM Fine-Tuning with Qwen2
[**简体中文**](README-zh.md) | **English**

This repository provides a streamlined pipeline for downloading datasets and fine-tuning Large Language Models (LLMs), specifically using the Qwen2-1.5B-Instruct model. The project leverages LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning and integrates SwanLab for experiment tracking and visualization.

## Features
* **Automated Data Downloading**: Includes scripts to automatically fetch high-quality medical QA datasets from HuggingFace/ModelScope (e.g., Huatuo26M-Lite, DISC-Med-SFT).
* **LoRA Fine-Tuning**: Utilizes `peft` to fine-tune the Qwen2 model efficiently, reducing VRAM requirements while maintaining performance.
* **Experiment Tracking**: Seamless integration with SwanLab to monitor training loss, learning rate, and model predictions in real-time.

## Project Structure

Fine-tuning-of-LLM-/
├── data_download.py     # Script to download large medical datasets
├── train.py             # Main script for data processing, LoRA fine-tuning, and inference
├── requirements.txt     # Python dependencies
├── datasets/            # Directory for downloaded datasets (Ignored in git)
├── output/              # Directory for saved model weights (Ignored in git)
└── qwen/                # Downloaded base model weights (Ignored in git)
Installation
Ensure you have Python installed. It is recommended to use a virtual environment.

Bash
# Clone the repository
git clone [https://github.com/ShuoMeng66/Fine-tuning-of-LLM-.git](https://github.com/ShuoMeng66/Fine-tuning-of-LLM-.git)
cd Fine-tuning-of-LLM-

# Install dependencies
pip install -r requirements.txt
Usage
1. Prepare Datasets
Run the download script to fetch the datasets. The script configures a domestic mirror for faster downloading in restricted network environments.

Bash
python data_download.py
Note: Make sure your train.jsonl and test.jsonl files are placed in the correct directory as expected by train.py.

2. Start Fine-Tuning
The training script will automatically download the Qwen2-1.5B-Instruct base model via ModelScope, apply LoRA configuration, start training, and log the process to SwanLab.

Bash
python train.py
Once completed, the fine-tuned model weights and the tokenizer will be saved in the ./output/Qwen2 directory.

Tracking with SwanLab
During or after training, you can view your training metrics and prediction results on the SwanLab dashboard. The script automatically creates a project named Qwen2-fintune.
