# ============================================================
# 实验 1：在 GoEmotions（折叠为 Ekman 7 类）上微调 RoBERTa-base
# 用法：整段复制到 Kaggle Notebook 的一个代码单元格里运行
# 前提：右侧 Settings 里 Accelerator 选 GPU，Internet 选 On
# 预计耗时：T4 显卡约 20-30 分钟
# 产出（在 /kaggle/working/ 下）：
#   test_probs.npy   测试集每条样本的 7 类概率（后续分析要用）
#   val_probs.npy    验证集概率
#   test_labels.npy / val_labels.npy  真实标签
#   test_predictions.csv  方便人看的表格
#   metrics.json     准确率和 Macro-F1
# ============================================================
import json
import os
import random
import subprocess

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

# ---------- 1. 设置（与 Selection Effect 论文附录 C.3 保持一致）----------
SEED = 42            # 论文用了 42、43、44 三个种子；先跑 42
MODEL_NAME = "roberta-base"
EPOCHS = 3
BATCH_SIZE = 32
LR = 2e-5
MAX_LEN = 128
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("运行设备:", device, "（如果显示 cpu，说明没有打开 GPU，会非常慢）")

# ---------- 2. 标签：GoEmotions 28 类 -> Ekman 7 类 ----------
EKMAN = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
GO_TO_EKMAN = {
    "anger": "anger", "annoyance": "anger", "disapproval": "anger",
    "disgust": "disgust",
    "fear": "fear", "nervousness": "fear",
    "joy": "joy", "amusement": "joy", "approval": "joy", "excitement": "joy",
    "gratitude": "joy", "love": "joy", "optimism": "joy", "relief": "joy",
    "pride": "joy", "admiration": "joy", "desire": "joy", "caring": "joy",
    "sadness": "sadness", "disappointment": "sadness",
    "embarrassment": "sadness", "grief": "sadness", "remorse": "sadness",
    "surprise": "surprise", "realization": "surprise",
    "confusion": "surprise", "curiosity": "surprise",
    "neutral": "neutral",
}

# ---------- 3. 读数据 ----------
ds = load_dataset("google-research-datasets/go_emotions", "simplified")
go_names = ds["train"].features["labels"].feature.names
print("各部分样本数:", {k: len(v) for k, v in ds.items()})  # 应为 43410 / 5426 / 5427


def to_ekman(example):
    # 一条样本可能有多个标签；与论文一致，取第一个标签
    first = go_names[example["labels"][0]]
    return {"label": EKMAN.index(GO_TO_EKMAN[first])}


ds = ds.map(to_ekman)
tok = AutoTokenizer.from_pretrained(MODEL_NAME)


def collate(batch):
    enc = tok([b["text"] for b in batch], truncation=True, max_length=MAX_LEN,
              padding=True, return_tensors="pt")
    enc["labels"] = torch.tensor([b["label"] for b in batch])
    return enc


g = torch.Generator()
g.manual_seed(SEED)
train_dl = DataLoader(ds["train"], batch_size=BATCH_SIZE, shuffle=True,
                      collate_fn=collate, generator=g)
val_dl = DataLoader(ds["validation"], batch_size=128, collate_fn=collate)
test_dl = DataLoader(ds["test"], batch_size=128, collate_fn=collate)

# ---------- 4. 模型与训练 ----------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(EKMAN)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_dl) * EPOCHS
sched = get_linear_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)
use_amp = device == "cuda"
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)


@torch.no_grad()
def predict(dl):
    """返回 (logits, labels)，都是 numpy 数组"""
    model.eval()
    all_logits, all_labels = [], []
    for batch in dl:
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(**batch).logits
        all_logits.append(logits.float().cpu())
        all_labels.append(labels)
    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()


def report(name, logits, labels):
    pred = logits.argmax(1)
    acc = accuracy_score(labels, pred)
    f1 = f1_score(labels, pred, average="macro")
    print(f"{name}: 准确率={acc:.4f}  Macro-F1={f1:.4f}")
    return {"accuracy": float(acc), "macro_f1": float(f1)}


for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    for step, batch in enumerate(train_dl, 1):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", enabled=use_amp):
            loss = model(**batch).loss
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        running += loss.item()
        if step % 200 == 0:
            print(f"  第 {epoch} 轮 第 {step}/{len(train_dl)} 步  平均损失={running / step:.4f}")
    v_logits, v_labels = predict(val_dl)
    report(f"第 {epoch} 轮结束，验证集", v_logits, v_labels)

# ---------- 5. 最终评估并保存 ----------
v_logits, v_labels = predict(val_dl)
t_logits, t_labels = predict(test_dl)
metrics = {"seed": SEED, "model": MODEL_NAME,
           "val": report("最终 验证集", v_logits, v_labels),
           "test": report("最终 测试集", t_logits, t_labels)}
print("论文报告的测试集 Macro-F1 约为 0.623 ± 0.005，相差不大即为正常。")


def softmax(x):
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


t_probs, v_probs = softmax(t_logits), softmax(v_logits)
np.save(f"{OUT_DIR}/test_logits.npy", t_logits)
np.save(f"{OUT_DIR}/val_logits.npy", v_logits)
np.save(f"{OUT_DIR}/test_probs.npy", t_probs)
np.save(f"{OUT_DIR}/val_probs.npy", v_probs)
np.save(f"{OUT_DIR}/test_labels.npy", t_labels)
np.save(f"{OUT_DIR}/val_labels.npy", v_labels)

df = pd.DataFrame({"text": ds["test"]["text"],
                   "gold": [EKMAN[i] for i in t_labels],
                   "pred": [EKMAN[i] for i in t_probs.argmax(1)],
                   "confidence": t_probs.max(1)})
for i, name in enumerate(EKMAN):
    df[f"p_{name}"] = t_probs[:, i]
df.to_csv(f"{OUT_DIR}/test_predictions.csv", index=False)

# ---------- 6. 与论文作者公开的缓存对齐检查 ----------
# 目的：确认我们的测试集顺序和标签与作者的一致，这样才能把我们的
# 编码器概率和他们缓存的大模型预测逐条对上。
try:
    repo = f"{OUT_DIR}/selective-emotion-classification"
    if not os.path.isdir(repo):
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/tikhomirovd/selective-emotion-classification.git",
                        repo], check=True)
    with open(f"{repo}/data/cache/verifier/llm_zs_ge.json") as f:
        cached = json.load(f)
    gold_cached = cached["gold_labels"]
    ours = [EKMAN[i] for i in t_labels]
    same = sum(a == b for a, b in zip(ours, gold_cached))
    rate = same / len(ours)
    print(f"对齐检查：{same}/{len(ours)} 条真实标签一致（{rate:.2%}）")
    print("100% 表示完全对齐；如果明显低于 100%，把这一行输出发给 Claude。")
    metrics["alignment_with_author_cache"] = rate
except Exception as e:  # 网络没开或仓库有变化时不影响前面的结果
    print("对齐检查没有完成（不影响已保存的文件）：", repr(e))

with open(f"{OUT_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
print("全部完成。文件保存在", OUT_DIR)
