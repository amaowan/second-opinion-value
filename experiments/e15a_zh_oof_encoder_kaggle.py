# ============================================================
# 实验 15a：中文编码器的"折外预测"（给实验 15b 的互补 LoRA 用）
#
# 为什么要做：实验 15b 要让 LoRA 多学编码器容易错的句子。可编码器在自己训练过的句子上几乎不会错，
#   直接拿它在训练集上的预测，会把"编码器的弱点"估计得太少（Jeffares 等 2023 说的假多样性）。
#   所以把训练集随机分成两半：用 A 半训练、去预测 B 半，再用 B 半训练、去预测 A 半。
#   这样训练集里每一条都有一个"没见过它的编码器"给出的预测，这就是折外预测。
# 设置与实验 8 相同：chinese-roberta-wwm-ext，3 轮，batch 32，学习率 2e-5，种子 42。
#   训练集的清理方式与实验 8、14b 完全相同（去掉与评测集重复的、去掉内部重复），顺序也相同。
#
# 运行前：
#   1. 新建 Notebook，Add Input 添加 EWECT 数据集（和以前一样）。
#   2. Settings：Accelerator 选 GPU T4 x2，Internet 选 On。
#   3. 粘贴本脚本，Save Version -> Save & Run All。预计 40 到 60 分钟。
#
# 产出（/kaggle/working/ 下）：
#   zh_oof/train_oof_probs.npy    训练集每一条的折外 6 类概率（顺序与清理后的训练集一致）
#   zh_oof/train_md5.npy          每一条训练文本的 md5，用来在实验 15b 里核对顺序
#   zh_oof/train_labels.npy、zh_oof/train_fold.npy
#   zh_oof_metrics.json           折外预测的准确率等（发给 Claude 的就是这个）
# ============================================================
import glob
import hashlib
import json
import os
import random
import zipfile

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

MODEL_NAME = os.environ.get("ZH_MODEL", "hfl/chinese-roberta-wwm-ext")
EPOCHS = int(os.environ.get("ZH_EPOCHS", "3"))
BATCH_SIZE = 32
LR = 2e-5
MAX_LEN = 192
SEED = 42
QUICK = int(os.environ.get("ZH_QUICK", "0"))      # 仅供本地冒烟测试用，Kaggle 上保持 0
ROOT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
INPUT_ROOTS = ["/kaggle/input", os.environ.get("ZH_DATA", ROOT)]
ZH_LABELS = ["angry", "fear", "happy", "sad", "surprise", "neutral"]
EVAL_SETS = ["usual_eval", "usual_test", "virus_eval", "virus_test"]
NEU = ZH_LABELS.index("neutral")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("运行设备:", device, "（如果显示 cpu，说明没有打开 GPU）")
if device == "cpu" and not os.environ.get("ALLOW_CPU"):
    raise SystemExit("没有检测到 GPU。请在右侧 Settings 里把 Accelerator 设为 GPU，再重新运行。")


def candidates(basename):
    out = []
    for root in INPUT_ROOTS:
        if os.path.isdir(root):
            out += glob.glob(os.path.join(root, "**", basename), recursive=True)
    return sorted(set(out))


if not candidates("usual_train.txt"):
    for root in INPUT_ROOTS:
        for z in glob.glob(os.path.join(root, "**", "*.zip"), recursive=True):
            with zipfile.ZipFile(z) as zf:
                zf.extractall(f"{ROOT}/ewect_data")
    INPUT_ROOTS.append(f"{ROOT}/ewect_data")


def load_split(name):
    base = f"{name}.txt" if name.endswith("train") else f"{name}_labeled.txt"
    best = None
    for p in candidates(base):
        with open(p, "rb") as f:
            d = json.loads(f.read().decode("utf-8-sig"))
        if all(x.get("label") in ZH_LABELS for x in d) and (best is None or len(d) > len(best)):
            best = d
    if best is None:
        raise SystemExit(f"没有找到 {base}。请确认已在 Add Input 里添加 EWECT 数据集。")
    return [{"text": str(x["content"]), "label": ZH_LABELS.index(x["label"])} for x in best]


# ---------- 1. 训练集清理（与实验 8、14b 完全相同） ----------
DATA = {n: load_split(n) for n in ["usual_train"] + EVAL_SETS}
held = set(x["text"] for n in EVAL_SETS for x in DATA[n])
seen, train = set(), []
for x in DATA["usual_train"]:
    if x["text"] in held or x["text"] in seen:
        continue
    seen.add(x["text"]); train.append(x)
print(f"训练集：清理后 {len(train)} 条（实验 14b 是 26660 条，两者应当相同）")
if QUICK:
    train = train[:QUICK]
    DATA = {n: (d[:QUICK] if n in EVAL_SETS else d) for n, d in DATA.items()}

md5 = np.array([hashlib.md5(x["text"].encode("utf-8")).hexdigest() for x in train])
labels = np.array([x["label"] for x in train])
fold = np.random.default_rng(SEED).permutation(len(train)) % 2      # 随机对半分

# ---------- 2. 两折训练 ----------
tok = AutoTokenizer.from_pretrained(MODEL_NAME)


def collate(batch):
    enc = tok([b["text"] for b in batch], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
    enc["labels"] = torch.tensor([b["label"] for b in batch])
    return enc


def softmax(x):
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


use_amp = device == "cuda"
oof = np.zeros((len(train), 6), dtype=np.float32)
eval_probs = {n: [] for n in EVAL_SETS}
metrics = {"model": MODEL_NAME, "epochs": EPOCHS, "train_size": len(train), "folds": {}}

for f in (0, 1):
    random.seed(SEED + f); np.random.seed(SEED + f); torch.manual_seed(SEED + f); torch.cuda.manual_seed_all(SEED + f)
    tr_idx, te_idx = np.where(fold != f)[0], np.where(fold == f)[0]
    g = torch.Generator(); g.manual_seed(SEED + f)
    train_dl = DataLoader([train[i] for i in tr_idx], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate, generator=g)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=6).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total = len(train_dl) * EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    @torch.no_grad()
    def predict(items):
        model.eval()
        out = []
        for batch in DataLoader(items, batch_size=128, collate_fn=collate):
            batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", enabled=use_amp):
                out.append(model(**batch).logits.float().cpu())
        return softmax(torch.cat(out).numpy())

    print(f"\n===== 第 {f + 1} 折：用 {len(tr_idx)} 条训练，预测另外 {len(te_idx)} 条 =====")
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
            scaler.step(opt); scaler.update(); sched.step()
            running += loss.item()
            if step % 200 == 0:
                print(f"  第 {epoch} 轮 第 {step}/{len(train_dl)} 步  平均损失={running / step:.4f}")
        print(f"  第 {epoch} 轮结束")
    p = predict([train[i] for i in te_idx])
    oof[te_idx] = p
    pred = p.argmax(1)
    m = {"accuracy": float(accuracy_score(labels[te_idx], pred)),
         "macro_f1": float(f1_score(labels[te_idx], pred, average="macro", labels=list(range(6)), zero_division=0))}
    for n in EVAL_SETS:
        pe = predict(DATA[n])
        eval_probs[n].append(pe)
        y = np.array([x["label"] for x in DATA[n]])
        m[n] = {"accuracy": float(accuracy_score(y, pe.argmax(1)))}
    metrics["folds"][f"fold{f}"] = m
    print(f"  折外准确率 {m['accuracy']:.4f}  Macro-F1 {m['macro_f1']:.4f}；usual_test 准确率 {m['usual_test']['accuracy']:.4f}")
    del model, opt, sched, scaler
    if device == "cuda":
        torch.cuda.empty_cache()

# ---------- 3. 保存 ----------
os.makedirs(f"{ROOT}/zh_oof", exist_ok=True)
np.save(f"{ROOT}/zh_oof/train_oof_probs.npy", oof)
np.save(f"{ROOT}/zh_oof/train_md5.npy", md5)
np.save(f"{ROOT}/zh_oof/train_labels.npy", labels)
np.save(f"{ROOT}/zh_oof/train_fold.npy", fold)
for n in EVAL_SETS:
    np.save(f"{ROOT}/zh_oof/{n}_foldavg_probs.npy", np.mean(eval_probs[n], 0))
pt = oof[np.arange(len(labels)), labels]
metrics["oof"] = {"accuracy": float((oof.argmax(1) == labels).mean()),
                  "macro_f1": float(f1_score(labels, oof.argmax(1), average="macro", labels=list(range(6)), zero_division=0)),
                  "p_true_mean": float(pt.mean()), "share_p_true_below_0.5": float((pt < 0.5).mean())}
with open(f"{ROOT}/zh_oof_metrics.json", "w") as f_:
    json.dump(metrics, f_, indent=2, ensure_ascii=False)
print(f"\n折外预测：准确率 {metrics['oof']['accuracy']:.4f}，正确类别的平均概率 {metrics['oof']['p_true_mean']:.3f}，"
      f"正确类别概率低于 0.5 的句子占 {metrics['oof']['share_p_true_below_0.5']:.1%}")
print("全部完成。请把 zh_oof_metrics.json 发给 Claude，然后去跑实验 15b。")
