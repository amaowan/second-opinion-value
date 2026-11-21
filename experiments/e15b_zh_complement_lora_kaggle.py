# ============================================================
# 实验 15b：与编码器互补的中文 LoRA 验证器（两种训练方式，同一个脚本）
#
# 和实验 14b 的区别只有训练损失，其余（模型、LoRA 设置、学习率、3 轮、数据、预测方式）完全相同。
#   VARIANT = "weighted"   加权版：编码器越没把握的句子（折外预测里正确类别的概率越低），损失里的分量越重。
#                          权重 = (1 - 编码器给正确类别的概率) + 0.25，再整体缩放到平均为 1。
#   VARIANT = "complement" 互补版：损失 = -log(0.5 × 编码器概率 + 0.5 × LoRA 概率)（正确类别上）
#                          + 0.25 × 普通交叉熵。编码器已经很有把握答对的句子，第一项几乎不给 LoRA 压力，
#                          LoRA 的力气主要花在编码器答不对的句子上；第二项保证它单独使用时仍是一个分类器。
#   VARIANT = "standard6"  对照版（可选，不用跑）：普通交叉熵，只是只在 6 个类别字之间算。
#   这里的"编码器"是实验 15a 的折外预测，不是实验 8 的编码器，防止把训练集上的假弱点学进去。
#
# 运行前：
#   1. 先跑完实验 15a。
#   2. 新建 Notebook，Add Input 添加 EWECT 数据集（Datasets），再添加实验 15a 那个 Notebook 的输出（Notebooks）。
#   3. Settings：Accelerator 选 GPU T4 x2（只会用到其中一块），Internet 选 On。
#   4. 下面 VARIANT 那一行写 "weighted"，Save Version -> Save & Run All。约 6 小时。
#      再新建一个 Notebook（或把这个 Notebook 复制一份），VARIANT 改成 "complement"，同样跑一次。
#
# 产出（/kaggle/working/ 下）：
#   zh_llm/Qwen3-4B-LoRA{W|C}-ep{1,2,3}/usual/{集合名}_probs.npy   每一轮结束时的概率文件
#   lora_{variant}_ep{1,2,3}/                                     每一轮结束时的 LoRA 参数
#   zh_lora_{variant}_metrics.json                                指标（发给 Claude 的就是这个）
# ============================================================
import glob
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch

try:
    import peft  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft"], check=True)
# Kaggle 预装的 torchao 版本太旧，新版 peft 一检查到它就报错。我们用不到 torchao，
# 所以告诉 peft "没有 torchao"，让它跳过这一步。
try:
    import importlib
    import peft.import_utils as _piu
    _piu.is_torchao_available = lambda: False
    for _m in ("peft.tuners.lora.torchao", "peft.tuners.lora.model", "peft.tuners.tuners_utils"):
        try:
            _mod = importlib.import_module(_m)
            if hasattr(_mod, "is_torchao_available"):
                _mod.is_torchao_available = lambda: False
        except Exception:
            pass
except Exception as _ex:
    print("（跳过 torchao 检查时出了点问题，可以忽略：", repr(_ex), "）")
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

# ★ 第一次请把下面这行改成 True，用"Run All"在编辑器里直接跑（约 5 到 10 分钟）。
#   确认最后打印出"全部完成"之后，再改回 False，用 Save Version 正式跑。
QUICK_TEST = False

# ★ 每次运行前确认这一行：第一次写 "weighted"，第二次写 "complement"
VARIANT = os.environ.get("LORA_VARIANT", "weighted")
assert VARIANT in ("weighted", "complement", "standard6"), "VARIANT 只能是 weighted、complement 或 standard6"
TAG = {"weighted": "W", "complement": "C", "standard6": "S6"}[VARIANT]
W_FLOOR = 0.25          # 加权版：权重的下限，防止编码器很有把握的句子被完全忽略
MIX_BETA = 0.5          # 互补版：编码器在混合里的比重
CE_LAMBDA = 0.25        # 互补版：普通交叉熵的比重

MODEL_NAME = os.environ.get("ZH_LLM", "Qwen/Qwen3-4B-Instruct-2507")
LORA_TARGETS = os.environ.get("LORA_TARGETS", "q_proj,v_proj").split(",")
TIME_BUDGET_HOURS = float(os.environ.get("TIME_BUDGET_HOURS", "10.0"))  # 训练加预测的时间上限
EPOCHS = int(os.environ.get("LORA_EPOCHS", "3"))
MICRO_BATCH = 8
GRAD_ACCUM = 4                     # 8 x 4 = 等效 batch 32，与原论文相同
LR = 2e-4
MAX_CHARS = 200
SEED = 42
QUICK = int(os.environ.get("ZH_QUICK", "300" if QUICK_TEST else "0"))
ROOT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
INPUT_ROOTS = ["/kaggle/input", os.environ.get("ZH_DATA", ROOT)]
ZH_LABELS = ["angry", "fear", "happy", "sad", "surprise", "neutral"]
EVAL_SETS = ["usual_eval", "usual_test", "virus_eval", "virus_test"]
NEU = ZH_LABELS.index("neutral")
WORDS = ["愤怒", "恐惧", "开心", "悲伤", "惊讶", "中性"]        # 与实验 11 的措辞 v1 相同
SYSTEM = ("你是一名情绪分类专家。请判断这条微博作者的主要情绪，只能从给定的类别中选一个。"
          "只输出类别名称，不要解释，不要标点，不要多余的字。")
USER = "类别：愤怒 | 恐惧 | 开心 | 悲伤 | 惊讶 | 中性\n\n微博：{text}\n情绪："

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
T_START = time.time()
import transformers  # noqa: E402
print(f"PyTorch {torch.__version__}，transformers {transformers.__version__}，peft {peft.__version__ if 'peft' in dir() else '?'}")
if device == "cuda":
    for i in range(torch.cuda.device_count()):
        pr = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {pr.name}，显存 {pr.total_memory / 2**30:.1f} GB")
elif not os.environ.get("ALLOW_CPU"):
    raise SystemExit("没有检测到 GPU。请在右侧 Settings 里把 Accelerator 设为 GPU T4 x2，再重新运行。")
if QUICK_TEST:
    print("★ 当前是快速测试模式（只用 300 条），结果没有意义，只用来确认能跑通。")


# ---------- 1. 读数据（与实验 8 相同：去掉与评测集重复的训练样本） ----------
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


DATA = {n: load_split(n) for n in ["usual_train"] + EVAL_SETS}
held = set(x["text"] for n in EVAL_SETS for x in DATA[n])
seen, train = set(), []
for x in DATA["usual_train"]:
    if x["text"] in held or x["text"] in seen:
        continue
    seen.add(x["text"]); train.append(x)
print(f"训练集：原有 {len(DATA['usual_train'])} 条，去掉与评测集重复和内部重复后 {len(train)} 条（实验 14b 是 26660 条）")
if QUICK:
    train = train[:QUICK]
    DATA = {n: (d[:QUICK] if n in EVAL_SETS else d) for n, d in DATA.items()}


# ---------- 1b. 读实验 15a 的折外预测，并按文本 md5 对齐 ----------
def find_npy(suffix):
    for root in INPUT_ROOTS + [ROOT]:
        if not os.path.isdir(root):
            continue
        for d, _, files in os.walk(root, followlinks=True):
            for fn in files:
                p = os.path.join(d, fn).replace("\\", "/")
                if p.endswith(suffix):
                    return p
    return None


P_OOF = None
if VARIANT != "standard6":
    pf, mf = find_npy("zh_oof/train_oof_probs.npy"), find_npy("zh_oof/train_md5.npy")
    if pf is None or mf is None:
        raise SystemExit("没有找到实验 15a 的输出（zh_oof 文件夹）。请在 Add Input 里添加实验 15a 那个 Notebook 的输出。")
    oof_all, md5_all = np.load(pf), np.load(mf)
    pos = {h: i for i, h in enumerate(md5_all)}
    my_md5 = [hashlib.md5(x["text"].encode("utf-8")).hexdigest() for x in train]
    missing = sum(h not in pos for h in my_md5)
    if missing:
        raise SystemExit(f"有 {missing} 条训练文本在实验 15a 的输出里找不到。请把这行发给 Claude。")
    P_OOF = torch.tensor(np.stack([oof_all[pos[h]] for h in my_md5]), dtype=torch.float32)
    y_all = np.array([x["label"] for x in train])
    pt = P_OOF.numpy()[np.arange(len(train)), y_all]
    print(f"折外预测已对齐：{len(train)} 条；编码器给正确类别的平均概率 {pt.mean():.3f}，"
          f"折外准确率 {(P_OOF.numpy().argmax(1) == y_all).mean():.4f}")
    W_ALL = torch.tensor((1.0 - pt) + W_FLOOR, dtype=torch.float32)
    W_ALL = W_ALL / W_ALL.mean()
    print(f"加权版的权重：最小 {W_ALL.min():.2f}，中位数 {W_ALL.median():.2f}，最大 {W_ALL.max():.2f}（平均为 1）")


def clean(text):
    return re.sub(r"\s+", " ", text).strip()[:MAX_CHARS]


# ---------- 2. 模型 ----------
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
FIRST_IDS = [tok(w, add_special_tokens=False)["input_ids"][0] for w in WORDS]
assert len(set(FIRST_IDS)) == 6, "两个类别名称的第一个 token 相同，请把这行发给 Claude"
dt = torch.float16
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dt, low_cpu_mem_usage=True)
model = model.half()                         # 确保是 16 位；32 位的 4B 模型放不进一块 T4
print(f"模型参数类型 {next(model.parameters()).dtype}，大小约 {sum(p.numel() for p in model.parameters()) * 2 / 2**30:.1f} GB")
model.to(device)
model.config.use_cache = False
try:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
except TypeError:
    model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=LORA_TARGETS,
                                         bias="none", task_type="CAUSAL_LM"))
for n_, p in model.named_parameters():
    if p.requires_grad:
        p.data = p.data.float()            # LoRA 参数用 32 位，避免半精度训练出错
model.print_trainable_parameters()


def build(text):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER.format(text=clean(text))}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def last_logits(texts):
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
        try:
            out = model(**enc, logits_to_keep=1)      # 只算最后一个位置的输出，省显存、省时间
        except TypeError:
            out = model(**enc)
    return out.logits[:, -1, :].float()


# ---------- 3. 训练 ----------
prompts = [build(x["text"]) for x in train]
targets = torch.tensor([x["label"] for x in train])          # 类别序号 0..5
FIRST_T = torch.tensor(FIRST_IDS)


def batch_loss(logits, idx):
    """idx 是这一批在训练集里的位置；返回这一批的平均损失。只在 6 个类别字之间计算。"""
    lg6 = logits[:, FIRST_T.to(logits.device)]
    logp = torch.log_softmax(lg6, -1)
    y = targets[idx].to(logits.device)
    nll = -logp.gather(1, y[:, None])[:, 0]
    if VARIANT == "standard6":
        return nll.mean()
    if VARIANT == "weighted":
        return (W_ALL[idx].to(logits.device) * nll).mean()
    pe = P_OOF[idx].to(logits.device).gather(1, y[:, None])[:, 0]
    mix = MIX_BETA * pe + (1 - MIX_BETA) * torch.exp(-nll)
    return (-torch.log(mix.clamp_min(1e-8)) + CE_LAMBDA * nll).mean()
lengths = np.array([len(p) for p in prompts])


def make_batches(epoch):
    rng = np.random.default_rng(SEED + epoch)
    idx = rng.permutation(len(prompts))
    batches = []
    for c0 in range(0, len(idx), MICRO_BATCH * 32):           # 每 256 条内部按长度排序，减少补齐浪费
        chunk = idx[c0:c0 + MICRO_BATCH * 32]
        chunk = chunk[np.argsort(lengths[chunk])]
        batches += [chunk[i:i + MICRO_BATCH] for i in range(0, len(chunk), MICRO_BATCH)]
    rng.shuffle(batches)
    return batches


params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
micro_total = EPOCHS * math.ceil(len(prompts) / MICRO_BATCH)
steps_total = math.ceil(micro_total / GRAD_ACCUM)
sched = get_linear_schedule_with_warmup(opt, int(0.03 * steps_total), steps_total)
try:
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
except Exception:
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")

# ---------- 4. 预测函数（与实验 11 相同的读打分方式） ----------
def score(probs, labels):
    pred = probs.argmax(1)
    return {"accuracy": float(accuracy_score(labels, pred)),
            "macro_f1": float(f1_score(labels, pred, average="macro", labels=list(range(6)), zero_division=0)),
            "pred_neutral_rate": float((pred == NEU).mean()), "gold_neutral_rate": float((labels == NEU).mean()),
            "detection_hit_rate": float(((pred != NEU) & (labels != NEU)).sum() / max((labels != NEU).sum(), 1)),
            "detection_false_alarm_rate": float(((pred != NEU) & (labels == NEU)).sum() / max((labels == NEU).sum(), 1))}


metrics = {"model": MODEL_NAME, "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "targets": LORA_TARGETS, "lr": LR,
                                         "effective_batch": MICRO_BATCH * GRAD_ACCUM, "epochs_planned": EPOCHS},
           "train_size": len(prompts), "loss_log": [], "checkpoints": {},
           "variant": VARIANT, "w_floor": W_FLOOR, "mix_beta": MIX_BETA, "ce_lambda": CE_LAMBDA}


def evaluate(tag, epochs_done):
    """在 4 个集合上预测并保存；tag 例如 ep1、ep2、ep3"""
    was_training = model.training
    model.eval()
    t0 = time.time()
    out_dir = f"{ROOT}/zh_llm/Qwen3-4B-LoRA{TAG}-{tag}/usual"
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(f"{ROOT}/lora_{VARIANT}_{tag}")
    res = {"epochs_done": epochs_done, "examples_seen": seen_examples, "sets": {}}
    print(f"\n=== 第 {epochs_done:.2f} 轮结束，开始预测（{tag}）===")
    for n in EVAL_SETS:
        texts = [build(x["text"]) for x in DATA[n]]
        labels = np.array([x["label"] for x in DATA[n]])
        order = np.argsort([len(t) for t in texts])
        probs = np.zeros((len(texts), 6), dtype=np.float32)
        mass = np.zeros(len(texts), dtype=np.float32)
        with torch.no_grad():
            for b0 in range(0, len(texts), 32):
                idx = order[b0:b0 + 32]
                lg = last_logits([texts[i] for i in idx])
                mass[idx] = torch.softmax(lg, -1)[:, FIRST_IDS].sum(1).cpu().numpy()
                probs[idx] = torch.softmax(lg[:, FIRST_IDS], -1).cpu().numpy()
        np.save(f"{out_dir}/{n}_probs.npy", probs)
        m = score(probs, labels)
        m["label_token_mass_mean"] = float(mass.mean())
        res["sets"][n] = m
        print(f"  {n}: 准确率={m['accuracy']:.4f}  Macro-F1={m['macro_f1']:.4f}  预测为中性={m['pred_neutral_rate']:.3f}"
              f"（金标 {m['gold_neutral_rate']:.3f}）  类别词概率合计 {mass.mean():.3f}")
    res["predict_minutes"] = (time.time() - t0) / 60
    metrics["checkpoints"][tag] = res
    metrics["loss_log"] = log
    metrics["train_and_predict_minutes_so_far"] = (time.time() - t_train) / 60
    with open(f"{ROOT}/zh_lora_{VARIANT}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  预测用时 {res['predict_minutes']:.0f} 分钟；结果已写入 zh_lora_{VARIANT}_metrics.json")
    if was_training:
        model.train()


# ---------- 5. 训练（每轮结束后预测一次） ----------
model.train()
t_train, micro, seen_examples, stopped_early, running, log = time.time(), 0, 0, False, [], []
oom_splits = 0
print(f"\n开始训练：{len(prompts)} 条 × {EPOCHS} 轮，共 {steps_total} 个更新步；时间上限 {TIME_BUDGET_HOURS} 小时")
for epoch in range(EPOCHS):
    for bi, b in enumerate(make_batches(epoch)):
        try:
            logits = last_logits([prompts[i] for i in b])
            loss = batch_loss(logits, b) / GRAD_ACCUM
            scaler.scale(loss).backward()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()                 # 显存不够：把这一批拆成两半各算一次
            for half in (b[:len(b) // 2], b[len(b) // 2:]):
                lg = last_logits([prompts[i] for i in half])
                ls = batch_loss(lg, half) / GRAD_ACCUM * len(half) / len(b)
                scaler.scale(ls).backward()
            loss = ls * len(b) / len(half)
            oom_splits += 1
            print(f"  （显存不足，已把一批拆开计算；累计 {oom_splits} 次）")
        if not torch.isfinite(loss):
            print("  警告：损失变成了 nan/inf，跳过这一批")
            opt.zero_grad(set_to_none=True)
            continue
        running.append(loss.item() * GRAD_ACCUM)
        micro += 1
        seen_examples += len(b)
        if micro % GRAD_ACCUM == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        elapsed = time.time() - t_train
        if micro == 1 and device == "cuda":
            print(f"  第一批完成，显存峰值 {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
        if micro == 40:
            per = elapsed / micro
            print(f"  速度：每个小批次 {per:.2f} 秒；{EPOCHS} 轮训练预计 {per * micro_total / 3600:.1f} 小时（另加每轮约 15 分钟预测）")
        if micro % 200 == 0:
            avg = float(np.mean(running[-200:]))
            log.append({"micro_step": micro, "epoch": epoch + 1, "examples": seen_examples, "loss": avg, "minutes": elapsed / 60})
            print(f"  第 {epoch + 1} 轮  第 {micro}/{micro_total} 个小批次  最近平均损失 {avg:.4f}  已用 {elapsed / 60:.0f} 分钟")
        if elapsed > TIME_BUDGET_HOURS * 3600:
            stopped_early = True
            print(f"  到达时间上限，停止训练。共看过 {seen_examples} 条（{seen_examples / len(prompts):.2f} 轮）")
            break
    if micro % GRAD_ACCUM:                        # 把这一轮最后不满一组的梯度也更新掉
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update()
        opt.zero_grad(set_to_none=True)
    if stopped_early:
        evaluate(f"partial{seen_examples / len(prompts):.2f}", seen_examples / len(prompts))
        break
    evaluate(f"ep{epoch + 1}", epoch + 1)

metrics["stopped_by_time_budget"] = stopped_early
metrics["oom_splits"] = oom_splits
with open(f"{ROOT}/zh_lora_{VARIANT}_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
print(f"\n全部完成（{VARIANT}），总用时 {(time.time() - T_START) / 3600:.1f} 小时。请把 zh_lora_{VARIANT}_metrics.json 发给 Claude。")
print("对照（usual_test）：编码器 Macro-F1 = 0.762；普通 LoRA 第 3 轮 = 0.788。互补版单独用时比普通 LoRA 低一些是正常的。")
