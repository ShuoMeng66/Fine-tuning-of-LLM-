# 基于 Qwen2 的大语言模型微调指南
**简体中文** | [**English**](README.md)

本项目提供了一套完整的数据集下载与大语言模型 (LLM) 微调流程，核心基于 Qwen2-1.5B-Instruct 模型。项目采用 LoRA (低秩微调) 技术实现参数高效微调，并接入了 SwanLab 用于训练过程的监控与可视化。

## 主要特性
* **自动化数据下载**：内置脚本自动从 HuggingFace/ModelScope 拉取高质量医疗问答数据集（如华佗 Lite、DISC-Med-SFT）。
* **LoRA 高效微调**：使用 `peft` 库对 Qwen2 进行微调，大幅降低显存占用，同时保证模型效果。
* **可视化实验跟踪**：深度集成 SwanLab，实时记录训练 Loss、学习率变化，并可视化模型推理结果。

## 目录结构

Fine-tuning-of-LLM-/
├── data_download.py     # 用于下载大型医疗数据集的脚本
├── train.py             # 数据预处理、LoRA 微调与推理主脚本
├── requirements.txt     # Python 依赖清单
├── datasets/            # 存放下载的数据集 (不在 Git 版本控制中)
├── output/              # 存放微调后的模型权重 (不在 Git 版本控制中)
└── qwen/                # 下载的基础模型权重存放目录 (不在 Git 版本控制中)
环境安装
请确保已安装 Python，建议在虚拟环境中运行：

Bash
# 克隆仓库
git clone [https://github.com/ShuoMeng66/Fine-tuning-of-LLM-.git](https://github.com/ShuoMeng66/Fine-tuning-of-LLM-.git)
cd Fine-tuning-of-LLM-

# 安装依赖项
pip install -r requirements.txt
快速使用
1. 准备数据集
运行数据下载脚本。该脚本已配置国内镜像（hf-mirror），以保证网络不佳情况下的下载速度。

Bash
python data_download.py
提示：请确保你的 train.jsonl 和 test.jsonl 文件已放置在 train.py 期望的读取路径下。

2. 启动模型微调
训练脚本会自动通过 ModelScope 下载 Qwen2-1.5B-Instruct 基础模型，应用 LoRA 配置，开始训练并将日志推送到 SwanLab。

Bash
python train.py
训练完成后，微调后的模型权重及 Tokenizer 将被保存在 ./output/Qwen2 目录下。

SwanLab 实验面板
在训练期间或训练完成后，你可以前往 SwanLab 面板查看训练指标与测试集的推理生成结果。脚本会自动创建一个名为 Qwen2-fintune 的项目。


---

需要我为你直接生成那个用来屏蔽大文件的 `.gitignore` 代码，或者协助你调整 `train.py` 中读取数
