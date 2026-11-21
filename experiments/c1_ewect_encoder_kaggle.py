# ============================================================
# 实验 8（C1）：在 SMP2020-EWECT 上微调中文编码器
#
# 做什么：分别在"通用微博"(usual) 和"疫情微博"(virus) 两个训练集上微调 chinese-roberta-wwm-ext，
#         每个各跑 3 个种子（42/43/44），设置与英文实验一致（3 轮、batch 32、学习率 2e-5）。
#         每个模型都在 4 个有标签的集合上预测：usual_eval / usual_test / virus_eval / virus_test。
#         这样既有同领域结果，也顺带得到跨领域结果。
#
# 运行前要做的三件事：
#   1. 把"评测数据集.zip"上传成一个 Kaggle Dataset（设为 Private），
#      然后在 Notebook 右侧 Add Input 里把它加进来。
#   2. 右侧 Settings：Accelerator 选 GPU T4，Internet 选 On。
#   3. 用 Save Version -> Save & Run All 在后台跑。预计 60 到 90 分钟。
#
# 产出（/kaggle/working/zh/ 下）：
#   {usual,virus}/seed{42,43,44}/{集合名}_probs.npy   每条样本的 6 类概率
#   labels/{集合名}_labels.npy、labels/{集合名}_ids.npy 标签和原始 id
#   zh_metrics.json                                    全部指标（发给 Claude 的就是这个）
# ============================================================
import glob
import json
import os
import random
import zipfile
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

SEEDS = [42, 43, 44]
TRAIN_DOMAINS = ["usual", "virus"]
MODEL_NAME = os.environ.get("ZH_MODEL", "hfl/chinese-roberta-wwm-ext")
EPOCHS = 3
BATCH_SIZE = 32
LR = 2e-5
MAX_LEN = 192                     # 95% 的微博在 150 字以内；更长的会被截断
QUICK = int(os.environ.get("ZH_QUICK", "0"))      # 仅供本地冒烟测试用，Kaggle 上保持 0
ROOT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUT_DIR = f"{ROOT}/zh"
INPUT_ROOTS = ["/kaggle/input", os.environ.get("ZH_DATA", ROOT)]
ZH_LABELS = ["angry", "fear", "happy", "sad", "surprise", "neutral"]     # 中性放最后，与英文实验的习惯一致
EVAL_SETS = ["usual_eval", "usual_test", "virus_eval", "virus_test"]

device = "cuda" if torch.cuda.is_available() else "cpu"
print("运行设备:", device, "（如果显示 cpu，说明没有打开 GPU，会非常慢）")


# ---------- 1. 找文件、读数据 ----------
def candidates(basename):
    out = []
    for root in INPUT_ROOTS:
        if os.path.isdir(root):
            out += glob.glob(os.path.join(root, "**", basename), recursive=True)
    return sorted(set(out))


if not candidates("usual_train.txt"):          # Kaggle 没有自动解压时，自己解压
    for root in INPUT_ROOTS:
        for z in glob.glob(os.path.join(root, "**", "*.zip"), recursive=True):
            print("正在解压", z)
            with zipfile.ZipFile(z) as zf:
                zf.extractall(f"{ROOT}/ewect_data")
    INPUT_ROOTS.append(f"{ROOT}/ewect_data")
if not candidates("usual_train.txt"):
    raise SystemExit("没有找到 usual_train.txt。请确认已在右侧 Add Input 里添加了你上传的数据集。")


def read_json(path):
    with open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8-sig"))


def load_split(name):
    """name 如 usual_train / usual_eval / usual_test。测试集有两个版本，取不含混淆数据（没有 None 标签）的那个"""
    base = f"{name}.txt" if name.endswith("train") else f"{name}_labeled.txt"
    best = None
    for p in candidates(base):
        d = read_json(p)
        if all(x.get("label") in ZH_LABELS for x in d) and (best is None or len(d) > len(best)):
            best = d
    if best is None:
        raise SystemExit(f"没有找到带完整标签的 {base}。请把这行输出发给 Claude。")
    return [{"id": x["id"], "text": str(x["content"]), "label": ZH_LABELS.index(x["label"])} for x in best]


DATA = {n: load_split(n) for n in ["usual_train", "virus_train"] + EVAL_SETS}
expected = {"usual_train": 27768, "virus_train": 8606, "usual_eval": 2000, "usual_test": 5000,
            "virus_eval": 2000, "virus_test": 3000}
for n, d in DATA.items():
    dist = Counter(ZH_LABELS[x["label"]] for x in d)
    flag = "" if len(d) == expected[n] else f"  <-- 与说明文档的 {expected[n]} 条不一致，请告诉 Claude"
    print(f"{n:12s} {len(d):6d} 条  " + "  ".join(f"{k}={dist[k]}" for k in ZH_LABELS) + flag)

# ---------- 2. 清理训练集：去掉与评测集重复的微博，再去掉训练集内部的重复 ----------
held_out = set(x["text"] for n in EVAL_SETS for x in DATA[n])
clean_info = {}
for n in ("usual_train", "virus_train"):
    seen, kept, leak, dup, conflict = {}, [], 0, 0, 0
    for x in DATA[n]:
        if x["text"] in held_out:
            leak += 1
        elif x["text"] in seen:
            dup += 1
            conflict += int(seen[x["text"]] != x["label"])
        else:
            seen[x["text"]] = x["label"]
            kept.append(x)
    clean_info[n] = {"original": len(DATA[n]), "removed_overlap_with_eval_or_test": leak,
                     "removed_duplicates": dup, "duplicates_with_conflicting_labels": conflict, "kept": len(kept)}
    print(f"{n}：原有 {len(DATA[n])} 条；与评测集重复 {leak} 条（已去掉，防止泄漏）；"
          f"内部重复 {dup} 条（其中标签互相矛盾 {conflict} 条）；保留 {len(kept)} 条")
    DATA[n] = kept
if QUICK:
    for n in DATA:
        DATA[n] = DATA[n][:QUICK]

os.makedirs(f"{OUT_DIR}/labels", exist_ok=True)
for n in EVAL_SETS:
    np.save(f"{OUT_DIR}/labels/{n}_labels.npy", np.array([x["label"] for x in DATA[n]]))
    np.save(f"{OUT_DIR}/labels/{n}_ids.npy", np.array([x["id"] for x in DATA[n]]))

# ---------- 3. 训练和预测 ----------
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
lens = [len(tok(x["text"])["input_ids"]) for x in DATA["usual_test"][:2000]]
print(f"usual_test 前 2000 条的 token 长度：中位数 {int(np.median(lens))}，95 分位 {int(np.percentile(lens, 95))}，"
      f"超过 {MAX_LEN} 被截断的占 {np.mean(np.array(lens) > MAX_LEN):.2%}")


def collate(batch):
    enc = tok([b["text"] for b in batch], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
    enc["labels"] = torch.tensor([b["label"] for b in batch])
    return enc


def softmax(x):
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


def score(probs, labels):
    pred = probs.argmax(1)
    return {"accuracy": float(accuracy_score(labels, pred)),
            "macro_f1": float(f1_score(labels, pred, average="macro", labels=list(range(len(ZH_LABELS))), zero_division=0)),
            "pred_neutral_rate": float((pred == ZH_LABELS.index("neutral")).mean()),
            "gold_neutral_rate": float((labels == ZH_LABELS.index("neutral")).mean())}


use_amp = device == "cuda"
metrics = {"model": MODEL_NAME, "labels": ZH_LABELS, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
           "max_len": MAX_LEN, "train_cleaning": clean_info, "runs": {}}
eval_dls = {n: DataLoader(DATA[n], batch_size=128, collate_fn=collate) for n in EVAL_SETS}
eval_labels = {n: np.array([x["label"] for x in DATA[n]]) for n in EVAL_SETS}

for domain in TRAIN_DOMAINS:
    for seed in SEEDS:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        g = torch.Generator(); g.manual_seed(seed)
        train_dl = DataLoader(DATA[f"{domain}_train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate, generator=g)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(ZH_LABELS)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        total = len(train_dl) * EPOCHS
        sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        except Exception:                      # 旧版本 PyTorch
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        @torch.no_grad()
        def predict(dl):
            model.eval()
            out = []
            for batch in dl:
                batch.pop("labels")
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    out.append(model(**batch).logits.float().cpu())
            return softmax(torch.cat(out).numpy())

        print(f"\n===== 训练集 {domain}，种子 {seed}（{len(train_dl)} 步/轮）=====")
        per_epoch = []
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
            m = score(predict(eval_dls[f"{domain}_eval"]), eval_labels[f"{domain}_eval"])
            per_epoch.append(m)
            print(f"  第 {epoch} 轮结束，{domain}_eval：准确率={m['accuracy']:.4f}  Macro-F1={m['macro_f1']:.4f}")

        run = {"eval_per_epoch": per_epoch}
        os.makedirs(f"{OUT_DIR}/{domain}/seed{seed}", exist_ok=True)
        for n in EVAL_SETS:
            probs = predict(eval_dls[n])
            np.save(f"{OUT_DIR}/{domain}/seed{seed}/{n}_probs.npy", probs)
            run[n] = score(probs, eval_labels[n])
            tag = "同领域" if n.startswith(domain) else "跨领域"
            print(f"  {n:11s}（{tag}）准确率={run[n]['accuracy']:.4f}  Macro-F1={run[n]['macro_f1']:.4f}  "
                  f"预测为中性的比例={run[n]['pred_neutral_rate']:.3f}（金标 {run[n]['gold_neutral_rate']:.3f}）")
        metrics["runs"][f"{domain}|seed{seed}"] = run
        with open(f"{ROOT}/zh_metrics.json", "w") as f:           # 每跑完一个就存一次，中途断了也不白跑
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        del model, opt, sched, scaler
        if device == "cuda":
            torch.cuda.empty_cache()

# ---------- 4. 汇总 ----------
print("\n" + "=" * 70)
print("汇总（3 个种子的平均 ± 标准差）")
print("=" * 70)
metrics["summary"] = {}
for domain in TRAIN_DOMAINS:
    for n in EVAL_SETS:
        f1 = [metrics["runs"][f"{domain}|seed{s}"][n]["macro_f1"] for s in SEEDS]
        acc = [metrics["runs"][f"{domain}|seed{s}"][n]["accuracy"] for s in SEEDS]
        metrics["summary"][f"{domain}->{n}"] = {"macro_f1_mean": float(np.mean(f1)), "macro_f1_std": float(np.std(f1)),
                                                "accuracy_mean": float(np.mean(acc)), "accuracy_std": float(np.std(acc))}
        print(f"  在 {domain} 上训练 -> {n:11s} Macro-F1={np.mean(f1):.4f} ± {np.std(f1):.4f}   准确率={np.mean(acc):.4f} ± {np.std(acc):.4f}")
for n in EVAL_SETS:
    maj = Counter(eval_labels[n].tolist()).most_common(1)[0][1] / len(eval_labels[n])
    metrics["summary"][f"majority_baseline|{n}"] = float(maj)
    print(f"  参照：{n} 全猜最多的一类，准确率={maj:.4f}")
with open(f"{ROOT}/zh_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
print("\n全部完成。请把 zh_metrics.json 发给 Claude。概率文件留在这个 Notebook 的输出里，后面的分析会用到。")
