import os
import json
from datasets import load_dataset
from modelscope.msdatasets import MsDataset

# 设置国内 HuggingFace 镜像（可选，但推荐，尤其是下载大型数据集时）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 创建一个专门存放数据的文件夹
save_dir = "./datasets"
os.makedirs(save_dir, exist_ok=True)

def save_to_jsonl(dataset, filename):
    """通用保存函数：将数据集保存为 JSONL 格式"""
    filepath = os.path.join(save_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"成功保存: {filepath} (共 {len(dataset)} 条)")



#1. 下载 华佗 Huatuo-Lite (Lite版本是经过清洗的高质量QA,包含了海量的疾病百科、用药指南以及真实的医患问答数据。)
print("正在下载 华佗 Huatuo26M-Lite...")
try:
    ds_huatuo = load_dataset("FreedomIntelligence/Huatuo26M-Lite", split="train")
    save_to_jsonl(ds_huatuo, "huatuo_lite.jsonl")
except Exception as e:
    print(f"华佗下载失败: {e}")

#2. 下载 DISC-Med-SFT (复旦大学开源的高质量医疗数据集,里面的回答专业且非常注重“AI 医生的语气”和“多轮对话逻辑”。)
print("正在下载 DISC-Med-SFT...")
try:
    ds_disc = load_dataset("Flmc/DISC-Med-SFT", split="train")
    save_to_jsonl(ds_disc, "disc_med_sft.jsonl")
except Exception as e:
    print(f"DISC下载失败: {e}")

# 3. 下载 CMedQA V2.0 (基于真实的在线医疗问诊平台（如寻医问药、好大夫在线）抓取并清洗的数据。)
print("正在下载 CMedQA V2.0...")
try:
    ds_cmed = load_dataset("wangrongsheng/cMedQA-V2.0", split="train")
    save_to_jsonl(ds_cmed, "cmedqa_v2.jsonl")
except Exception as e:
    print(f"CMedQA下载失败: {e}")

print("所有数据集下载任务执行完毕！请前往 ./datasets 文件夹查看。")