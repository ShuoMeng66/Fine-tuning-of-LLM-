import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.transformers import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
import os
import swanlab

def dataset_jsonl_transfer(origin_path, new_path):
    
    messages = []
    with open(origin_path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            query = data.get("query", data.get("instruction", data.get("question", data.get("text", ""))))
            response = data.get("response", data.get("output", data.get("answer", "")))

            if not query or not response:
                continue

            message = {
                "instruction": "你现在是一个经验丰富、专业且富有同理心的AI医疗助手。请根据患者的描述，给出科学、准确的医疗科普和建议。请注意，你的建议不能替代专业医生的面诊。",
                "input": query,
                "output": response,
            }
            messages.append(message)

    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")


def process_func(example):
    
    MAX_LENGTH = 1024 
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|im_start|>system\n{example['instruction']}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = (
        instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    )
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = (
        [-100] * len(instruction["input_ids"])
        + response["input_ids"]
        + [tokenizer.pad_token_id]
    )
    if len(input_ids) > MAX_LENGTH: #截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def predict(messages, model, tokenizer):
    
    device = model.device # 动态获取模型所在设备
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(response)

    return response


# 在modelscope上下载Qwen模型到本地目录下
model_dir = snapshot_download(
    "qwen/Qwen2-1.5B-Instruct", cache_dir="./", revision="master"
)

# Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained(
    "./qwen/Qwen2-1___5B-Instruct/", use_fast=False, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    "./qwen/Qwen2-1___5B-Instruct/", device_map="auto", torch_dtype=torch.bfloat16
)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

# 加载、处理数据集和测试集
train_dataset_path = "train.jsonl"
test_dataset_path = "test.jsonl"

train_jsonl_new_path = "new_train.jsonl"
test_jsonl_new_path = "new_test.jsonl"

if not os.path.exists(train_jsonl_new_path):
    dataset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)
if not os.path.exists(test_jsonl_new_path):
    dataset_jsonl_transfer(test_dataset_path, test_jsonl_new_path)

train_df = pd.read_json(train_jsonl_new_path, lines=True)
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.5,  # Dropout 比例
)

model = get_peft_model(model, config)
print(model)#打印模型信息

args = TrainingArguments(
    output_dir="./output/Qwen2-1.5B-Instruct",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=3,           # 分类任务通常 3 轮效果更好
    save_steps=100,
    learning_rate=5e-5,           # LoRA 学习率通常设在 5e-5 到 1e-4
    save_on_each_node=True,
    gradient_checkpointing=True,
    bf16=True,                    # 如果是 Ampere 架构显卡必开
    lr_scheduler_type="cosine",    # 余弦退火学习率
    warmup_ratio=0.1,             # 预热步数
    weight_decay=0.01,
    report_to="none",
)

swanlab_callback = SwanLabCallback(
    project="Qwen2-fintune",
    experiment_name="Qwen2-1.5B-Instruct",
    description="使用通义千问Qwen2-1.5B-Instruct模型在zh_cls_fudan-news数据集上微调。",
    config={
        "model": "qwen/Qwen2-1.5B-Instruct",
        "dataset": "huangjintao/zh_cls_fudan-news",
    },
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)

trainer.train()
trainer.save_model("./output/Qwen2")
tokenizer.save_pretrained("./output/Qwen2")
print("模型已保存至 ./output/Qwen2")



test_df = pd.read_json(test_jsonl_new_path, lines=True)[:10]

test_text_list = []
for index, row in test_df.iterrows():
    instruction = row["instruction"]
    input_value = row["input"]

    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{input_value}"},
    ]

    response = predict(messages, model, tokenizer)
    messages.append({"role": "assistant", "content": f"{response}"})
    result_text = f"{messages[0]}\n\n{messages[1]}\n\n{messages[2]}"
    test_text_list.append(swanlab.Text(result_text, caption=response))

swanlab.log({"Prediction": test_text_list})
swanlab.finish()
