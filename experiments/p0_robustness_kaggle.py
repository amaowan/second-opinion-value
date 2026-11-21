# ============================================================
# 实验 3（稳健性检查）：三个随机种子 + 置信区间 + 第二编码器对照 + 可选的真实 VA 距离
#
# 用法：新建一个 Notebook（或在原 Notebook 里新建单元格），整段粘贴。
#       Settings 里 Accelerator 选 GPU，Internet 打开。
#       建议点右上角 Save Version -> Save & Run All，在后台运行，约 60-80 分钟。
#       如果当前会话里还留着实验 1 的 test_probs.npy，种子 42 会直接复用，不再重训。
#
# 做四件事：
#   1. 用种子 42、43、44 各训练一个 RoBERTa 编码器（已有结果的会跳过）
#   2. 三个种子上重复"置信度门 vs 严重度门"的比较，报告均值和标准差
#   3. 用配对自助法（bootstrap）给出差值的 95% 置信区间
#   4. 对照：把"另一个种子的编码器"当作验证器，看严重度上的好处是不是大模型特有的
#
# 产出：/kaggle/working/seed42|43|44/ 下的概率文件，以及 robustness_results.json
# ============================================================
import json
import os
import shutil
import subprocess

import numpy as np

SEEDS = [42, 43, 44]
MODEL_NAME = "roberta-base"
EPOCHS, BATCH_SIZE, LR, MAX_LEN = 3, 32, 2e-5, 128
N_BOOT = 1000                      # 自助法重采样次数
BUDGETS = [0.10, 0.20, 0.40]
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
REPO = f"{OUT_DIR}/selective-emotion-classification"
EKMAN = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

# ------------------------------------------------------------
# 可选：真实的效价(valence)、唤醒度(arousal)数值
# 现在是 None，脚本只用"效价正负"这把粗尺子。
# 想启用真实数值：到 NRC VAD Lexicon 官网下载词表（研究用途免费），查出下面 7 个词的
# valence 和 arousal（都是 0 到 1 之间的小数），填进去，把 None 那一行删掉即可。
# 例：VA = {"anger": (0.xxx, 0.xxx), ...}，括号里先 valence 后 arousal。
# 不要凭印象填，必须照词表抄，并在论文里注明来源和版本。
# ------------------------------------------------------------
VA = None
# VA = {
#     "anger":    (0.0, 0.0),
#     "disgust":  (0.0, 0.0),
#     "fear":     (0.0, 0.0),
#     "joy":      (0.0, 0.0),
#     "sadness":  (0.0, 0.0),
#     "surprise": (0.0, 0.0),
#     "neutral":  (0.0, 0.0),
# }

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


# ============================================================
# 第一部分：训练（和实验 1 相同，只是包成函数，方便换种子）
# ============================================================
def train_one(seed, save_dir):
    import random

    import torch
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer, get_linear_schedule_with_warmup)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n===== 训练种子 {seed}，设备 {device} =====")

    ds = load_dataset("google-research-datasets/go_emotions", "simplified")
    go_names = ds["train"].features["labels"].feature.names
    ds = ds.map(lambda ex: {"label": EKMAN.index(GO_TO_EKMAN[go_names[ex["labels"][0]]])})
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def collate(batch):
        enc = tok([b["text"] for b in batch], truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([b["label"] for b in batch])
        return enc

    g = torch.Generator()
    g.manual_seed(seed)
    train_dl = DataLoader(ds["train"], batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate, generator=g)
    val_dl = DataLoader(ds["validation"], batch_size=128, collate_fn=collate)
    test_dl = DataLoader(ds["test"], batch_size=128, collate_fn=collate)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(EKMAN)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total = len(train_dl) * EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    @torch.no_grad()
    def predict(dl):
        model.eval()
        outs, labs = [], []
        for batch in dl:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(**batch).logits
            outs.append(logits.float().cpu())
            labs.append(labels)
        return torch.cat(outs).numpy(), torch.cat(labs).numpy()

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
            if step % 400 == 0:
                print(f"  种子 {seed} 第 {epoch} 轮 {step}/{len(train_dl)} 步  平均损失={running / step:.4f}")

    def softmax(x):
        x = x - x.max(1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(1, keepdims=True)

    os.makedirs(save_dir, exist_ok=True)
    for name, dl in (("val", val_dl), ("test", test_dl)):
        logits, labels = predict(dl)
        np.save(f"{save_dir}/{name}_probs.npy", softmax(logits))
        np.save(f"{save_dir}/{name}_labels.npy", labels)
        pred = logits.argmax(1)
        print(f"  种子 {seed} {name}: 准确率={(pred == labels).mean():.4f}  Macro-F1={macro_f1(pred, labels):.4f}")
    del model, opt
    if device == "cuda":
        torch.cuda.empty_cache()


# ============================================================
# 第二部分：分析用的小工具
# ============================================================
def macro_f1(pred, labels, k=7):
    """7 类的 Macro-F1，用 numpy 实现，便于在自助法里重复上千次"""
    pred = np.clip(pred, 0, k - 1)
    cm = np.bincount(labels * k + pred, minlength=k * k).reshape(k, k).astype(float)
    tp = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return float(f1.mean())


def build_distance():
    """返回 {名字: (7x7 距离矩阵, 远错阈值)}"""
    val = np.array([-1, -1, -1, +1, -1, 0, 0])
    out = {"效价正负(粗略)": (np.abs(val[:, None] - val[None, :]).astype(float), 2.0)}
    if VA is not None:
        pts = np.array([VA[name] for name in EKMAN], dtype=float)
        d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
        d = d / d.max()                       # 归一化到 0-1
        off = d[~np.eye(7, dtype=bool)]
        out["VA欧氏距离(常模)"] = (d, float(np.median(off)))   # 距离超过中位数算"远错"
        print("已启用真实 VA 距离。类别间距离矩阵：")
        print(np.round(d, 2))
    else:
        print("提示：VA 还没填，本次只用'效价正负'这把粗尺子。")
    return out


def metrics(pred, labels, DIST, far_thr):
    p = np.clip(pred, 0, 6)
    d = np.where(pred >= 0, DIST[p, labels], 1.0)
    return np.array([(pred == labels).mean(), macro_f1(pred, labels),
                     d.mean(), (d >= far_thr - 1e-9).mean() if far_thr > 0 else 0.0])


METRIC_NAMES = ["准确率", "Macro-F1", "平均距离", "远错率"]


def top(score, budget):
    return np.argsort(-score, kind="stable")[:int(len(score) * budget)]


def load_verifier(path):
    with open(f"{REPO}/{path}") as f:
        d = json.load(f)
    return np.array([EKMAN.index(p) if p in EKMAN else -1 for p in d["predictions"]])


# ============================================================
# 主流程
# ============================================================
# 1) 训练（已有结果则跳过）
for s in SEEDS:
    sd = f"{OUT_DIR}/seed{s}"
    if os.path.exists(f"{sd}/test_probs.npy"):
        print(f"种子 {s} 已有结果，跳过训练")
        continue
    if s == 42 and os.path.exists(f"{OUT_DIR}/test_probs.npy"):
        os.makedirs(sd, exist_ok=True)
        for fn in ("test_probs.npy", "test_labels.npy", "val_probs.npy", "val_labels.npy"):
            if os.path.exists(f"{OUT_DIR}/{fn}"):
                shutil.copy(f"{OUT_DIR}/{fn}", f"{sd}/{fn}")
        print("种子 42 复用实验 1 的结果")
        continue
    train_one(s, sd)

# 2) 作者公开的大模型预测
if not os.path.isdir(REPO):
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/tikhomirovd/selective-emotion-classification.git", REPO],
                   check=True)
LORA = load_verifier("data/cache/verifier/llm_lora_ge.json")
ZSMAX = load_verifier("data/cache/ladder/qwen3.7-max_ge_v6_n5427_s42.json")

probs = {s: np.load(f"{OUT_DIR}/seed{s}/test_probs.npy") for s in SEEDS}
labels = np.load(f"{OUT_DIR}/seed{SEEDS[0]}/test_labels.npy")
for s in SEEDS:
    assert (np.load(f"{OUT_DIR}/seed{s}/test_labels.npy") == labels).all()
n = len(labels)
enc_pred = {s: probs[s].argmax(1) for s in SEEDS}
# 第二编码器对照：种子 s 的"验证器"是下一个种子训练出的编码器
other = {s: enc_pred[SEEDS[(i + 1) % len(SEEDS)]] for i, s in enumerate(SEEDS)}

rng = np.random.default_rng(0)
rand_score = rng.random(n)
boot_idx = rng.integers(0, n, size=(N_BOOT, n))
results = {"seeds": SEEDS, "n_boot": N_BOOT}

for dname, (DIST, far_thr) in build_distance().items():
    print("\n" + "#" * 78)
    print(f"# 严重度尺子：{dname}")
    print("#" * 78)

    def gate_scores(s):
        sp = np.sort(probs[s], axis=1)
        return {"置信度门": -(sp[:, -1] - sp[:, -2]),
                "严重度门": (probs[s] * DIST[enc_pred[s]]).sum(1),
                "随机": rand_score}

    def hybrid(s, gate, budget, verifier):
        S = top(gate_scores(s)[gate], budget)
        h = enc_pred[s].copy()
        h[S] = verifier(s)[S]
        return h

    VERIFS = {"LoRA-4B(有监督大模型)": lambda s: LORA,
              "零样本-Max": lambda s: ZSMAX,
              "第二编码器(对照)": lambda s: other[s]}

    base = np.array([metrics(enc_pred[s], labels, DIST, far_thr) for s in SEEDS])
    print("\n只用编码器（3 个种子 均值±标准差）")
    print("   " + "  ".join(f"{m}={base[:, i].mean():.3f}±{base[:, i].std():.3f}" for i, m in enumerate(METRIC_NAMES)))
    results[f"{dname}|encoder_only"] = base.mean(0).tolist()

    # ---- 表 1：各系统的整体结果 ----
    for b in BUDGETS:
        print(f"\n转交预算 {b:.0%}（整体结果，均值±标准差）")
        for vname, vf in VERIFS.items():
            for gate in ("置信度门", "严重度门", "随机"):
                arr = np.array([metrics(hybrid(s, gate, b, vf), labels, DIST, far_thr) for s in SEEDS])
                print(f"   {vname:16s} {gate:5s} " +
                      "  ".join(f"{m}={arr[:, i].mean():.3f}±{arr[:, i].std():.3f}" for i, m in enumerate(METRIC_NAMES)))
                results[f"{dname}|{b}|{vname}|{gate}"] = {"mean": arr.mean(0).tolist(), "std": arr.std(0).tolist()}

    # ---- 表 2：关键差值的 95% 置信区间（配对自助法，对三个种子取平均）----
    def diff_ci(make_a, make_b):
        """make_a/make_b: 种子 -> 预测。返回 (点估计, 下界, 上界)，各含 4 个指标"""
        A = {s: make_a(s) for s in SEEDS}
        B = {s: make_b(s) for s in SEEDS}
        point = np.mean([metrics(A[s], labels, DIST, far_thr) - metrics(B[s], labels, DIST, far_thr)
                         for s in SEEDS], axis=0)
        boots = np.empty((N_BOOT, 4))
        for r in range(N_BOOT):
            ix = boot_idx[r]
            boots[r] = np.mean([metrics(A[s][ix], labels[ix], DIST, far_thr) -
                                metrics(B[s][ix], labels[ix], DIST, far_thr) for s in SEEDS], axis=0)
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        return point, lo, hi

    print("\n关键比较的差值与 95% 置信区间（区间不包含 0 才算有统计意义）")
    print("   准确率、Macro-F1：正数 = 前者更好；平均距离、远错率：负数 = 前者更好")
    enc = lambda s: enc_pred[s]
    for b in BUDGETS:
        comps = {
            "LoRA+严重度门  减  只用编码器": (lambda s: hybrid(s, "严重度门", b, VERIFS["LoRA-4B(有监督大模型)"]), enc),
            "LoRA+置信度门  减  只用编码器": (lambda s: hybrid(s, "置信度门", b, VERIFS["LoRA-4B(有监督大模型)"]), enc),
            "LoRA+严重度门  减  LoRA+置信度门": (lambda s: hybrid(s, "严重度门", b, VERIFS["LoRA-4B(有监督大模型)"]),
                                              lambda s: hybrid(s, "置信度门", b, VERIFS["LoRA-4B(有监督大模型)"])),
            "第二编码器+严重度门  减  只用编码器": (lambda s: hybrid(s, "严重度门", b, VERIFS["第二编码器(对照)"]), enc),
            "LoRA+严重度门  减  第二编码器+严重度门": (lambda s: hybrid(s, "严重度门", b, VERIFS["LoRA-4B(有监督大模型)"]),
                                                  lambda s: hybrid(s, "严重度门", b, VERIFS["第二编码器(对照)"])),
            "零样本Max+严重度门  减  只用编码器": (lambda s: hybrid(s, "严重度门", b, VERIFS["零样本-Max"]), enc),
        }
        print(f"\n  预算 {b:.0%}")
        for cname, (fa, fb) in comps.items():
            p, lo, hi = diff_ci(fa, fb)
            cells = []
            for i, m in enumerate(METRIC_NAMES):
                star = "*" if (lo[i] > 0 or hi[i] < 0) else " "
                cells.append(f"{m} {p[i]:+.4f} [{lo[i]:+.4f},{hi[i]:+.4f}]{star}")
            print(f"    {cname}\n        " + "  ".join(cells))
            results[f"{dname}|{b}|CI|{cname}"] = {"point": p.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}

with open(f"{OUT_DIR}/robustness_results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("\n完成。带 * 的差值表示 95% 置信区间不包含 0。结果已保存到 robustness_results.json。")
print("请把'只用编码器'那一行开始往下的全部输出复制给 Claude。")
