# ============================================================
# 实验 G1：通用任务流水线（一个任务一个 Notebook，三个任务用同一个脚本）
#
# 只需要改下面一行 TASK：
#   "nli"       英文自然语言推理（MNLI；ChaosNLI 每句 100 位标注者的分布用来分层）
#   "toxicity"  英文毒性检测（Civil Comments；每条有"认为有毒的标注者比例"）
#   "cold"      中文冒犯语言检测（COLD）
# 三个任务都整理成"检测"问题：正类 = 蕴含 / 有毒 / 冒犯。
#
# 依次做四件事（与中文情绪实验 17a、17b、11 的设计相同）：
#   1. 整理数据：训练集最多 30000 条，去掉与验证集、测试集重复的文本；再随机对半分成 A、B 两半（种子 42）
#   2. 编码器：A、B 两半各训练 2 个（种子 42、43），3 轮，batch 32，学习率 2e-5
#   3. 零样本大模型：Qwen3-4B-Instruct-2507 和 Qwen2.5-3B-Instruct，读"是 / 否"两个词的概率
#   4. LoRA：Qwen3-4B-Instruct-2507 在 A、B 两半上各训练 1 轮（其余设置与实验 14b 相同）
#   5. 第二个大模型家族（FAMILY_MODEL，按实验 X0 的推荐填写）：零样本 + A、B 两半 LoRA，设置与 3、4 相同
#
# PART 一行决定跑哪些部分：
#   "all"     1 到 5 全跑（cold 用这个，约 5 到 6 小时）
#   "x_only"  只跑 1 和 5（nli、toxicity 用这个，Qwen 和编码器之前已经跑过了，约 2 到 3 小时）
#
# 运行前：
#   1. 新建 Notebook。不需要添加任何 Input（数据会自动从 Hugging Face 下载）。
#   2. Settings：Accelerator 选 GPU T4 x2，Internet 选 On。
#   3. 改好 TASK、PART、FAMILY_MODEL，Save Version -> Save & Run All。
#      用 Gemma 或 Llama 时，在 Add-ons -> Secrets 里打开 HF_TOKEN（与实验 X0 相同）。
#
# 产出（/kaggle/working/ 下）：
#   tasks/{TASK}/labels/{dev,test}_labels.npy、tasks/{TASK}/agree/{dev,test}_agree.npy
#   tasks/{TASK}/enc/half{0,1}/seed{42,43}/{dev,test}_probs.npy
#   tasks/{TASK}/llm/{zs_qwen3_4b, zs_qwen25_3b, lora_half0, lora_half1}/{dev,test}_probs.npy
#   tasks/{TASK}/llm/{zs_x, lora_x_half0, lora_x_half1}/{dev,test}_probs.npy 以及 llm/x_model.json
#   task_{TASK}_metrics.json（PART = "all"）或 task_{TASK}_x_metrics.json（PART = "x_only"），发给 Claude 的就是这个
# ============================================================
import gc
import glob
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import urllib.request
import zipfile

import numpy as np
import torch

# ★ 每个 Notebook 改这一行："nli"、"toxicity" 或 "cold"
TASK = os.environ.get("G_TASK", "nli")
# ★ 想先试跑（约 15 分钟）就改成 True；确认最后打印"全部完成"后改回 False 正式跑
QUICK_TEST = False
# ★ cold 写 "all"；nli、toxicity 写 "x_only"
PART = os.environ.get("G_PART", "all")
# 第二家族：冻结方案不跑，留空即可
FAMILY_MODEL = os.environ.get("G_XLLM", "")      # 2026-11-19 冻结方案：不跑第二家族，保持为空
assert PART in ("all", "x_only"), "PART 只能是 all 或 x_only"

assert TASK in ("nli", "toxicity", "cold"), "TASK 只能是 nli、toxicity 或 cold"
QUICK = int(os.environ.get("G_QUICK", "200" if QUICK_TEST else "0"))
FAKE = bool(os.environ.get("G_FAKE"))                     # 仅本地冒烟测试用
ROOT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUT = f"{ROOT}/tasks/{TASK}"
LANG = "zh" if TASK == "cold" else "en"
ENC_MODEL = os.environ.get("G_ENC", "hfl/chinese-roberta-wwm-ext" if LANG == "zh" else "roberta-base")
LORA_MODEL = os.environ.get("G_LLM", "Qwen/Qwen3-4B-Instruct-2507")
ZS_MODELS = {"zs_qwen3_4b": LORA_MODEL, "zs_qwen25_3b": os.environ.get("G_LLM2", "Qwen/Qwen2.5-3B-Instruct")}
LORA_TARGETS = os.environ.get("LORA_TARGETS", "q_proj,v_proj").split(",")
TRAIN_MAX, DEV_N = 30000, 2000
SEED = 42
MAX_LEN = 192
MAX_CHARS = 200 if LANG == "zh" else 600
T_START = time.time()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"任务 {TASK}；设备 {device}；PyTorch {torch.__version__}")
if device == "cpu" and not os.environ.get("ALLOW_CPU"):
    raise SystemExit("没有检测到 GPU。请在 Settings 里把 Accelerator 设为 GPU T4 x2。")
if QUICK:
    print(f"★ 快速测试模式：每个集合只用 {QUICK} 条，结果没有意义")
METRICS = {"task": TASK, "part": PART, "encoder": ENC_MODEL, "lora_model": LORA_MODEL, "zero_shot": ZS_MODELS,
           "family_model": FAMILY_MODEL}
MFILE = f"{ROOT}/task_{TASK}_metrics.json" if PART == "all" else f"{ROOT}/task_{TASK}_x_metrics.json"


def save_metrics():
    METRICS["minutes_so_far"] = (time.time() - T_START) / 60
    with open(MFILE, "w") as f:
        json.dump(METRICS, f, indent=2, ensure_ascii=False)


def clean(t):
    return re.sub(r"\s+", " ", str(t)).strip()


# ============================================================
# 1. 整理数据。每条：{"a": 文本, "b": 第二段文本或 None, "y": 0/1, "f": 认为是正类的标注者比例（没有就是 nan）}
# ============================================================
def load_nli():
    from datasets import load_dataset
    ds = load_dataset("nyu-mll/multi_nli")
    def conv(split):
        return [{"a": clean(x["premise"]), "b": clean(x["hypothesis"]), "y": int(x["label"] == 0), "f": float("nan")}
                for x in split if x["label"] in (0, 1, 2)]
    train, val = conv(ds["train"]), conv(ds["validation_matched"])
    # ChaosNLI：MNLI 验证集里 1599 句，每句 100 位标注者
    chaos = None
    for pat in ("/kaggle/input/**/chaosNLI_mnli_m.jsonl", f"{ROOT}/**/chaosNLI_mnli_m.jsonl"):
        for p in glob.glob(pat, recursive=True):
            chaos = p
    if chaos is None:
        try:
            z = f"{ROOT}/chaosnli.zip"
            urllib.request.urlretrieve("https://www.dropbox.com/s/h4j7dqszmpt2679/chaosNLI_v1.0.zip?dl=1", z)
            with zipfile.ZipFile(z) as zf:
                zf.extractall(f"{ROOT}/chaosnli")
            chaos = (glob.glob(f"{ROOT}/chaosnli/**/chaosNLI_mnli_m.jsonl", recursive=True) or [None])[0]
        except Exception as ex:
            print("ChaosNLI 从 Dropbox 下载失败，改试 Hugging Face 镜像：", repr(ex))
    dist = {}
    if chaos:
        with open(chaos) as f:
            for line in f:
                d = json.loads(line)
                dist[(clean(d["example"]["premise"]), clean(d["example"]["hypothesis"]))] = float(d["label_dist"][0])
    else:                                             # Dropbox 不通时，改用 Hugging Face 上的镜像（字段相同）
        try:
            cm = load_dataset("metaeval/chaos-mnli-ambiguity")
            for sp in cm:
                for d in cm[sp]:
                    dist[(clean(d["premise"]), clean(d["hypothesis"]))] = float(d["label_dist"][0])
            print(f"已从 Hugging Face 镜像 metaeval/chaos-mnli-ambiguity 读到 ChaosNLI-MNLI {len(dist)} 句")
        except Exception as ex:
            print("Hugging Face 镜像也没读到（不影响主实验，只是少了按标注者一致程度分层）：", repr(ex))
    if dist:
        hit = 0
        for x in val:
            k = (x["a"], x["b"])
            if k in dist:
                x["f"] = dist[k]; hit += 1
        print(f"ChaosNLI：{len(dist)} 句，其中 {hit} 句在 MNLI 验证集里找到了（有 100 位标注者的分布）")
    rng = np.random.default_rng(SEED)
    has = np.array([not math.isnan(x["f"]) for x in val])
    pool = np.where(~has)[0]
    dev_idx = set(rng.choice(pool, DEV_N, replace=False).tolist())
    dev = [val[i] for i in sorted(dev_idx)]
    test = [x for i, x in enumerate(val) if i not in dev_idx]
    return train, dev, test


def load_toxicity():
    from datasets import load_dataset
    ds = load_dataset("google/civil_comments")
    rng = np.random.default_rng(SEED)
    def sample(split, n_pos, n_neg):
        tox = np.array(split["toxicity"], dtype=float)
        pos, neg = np.where(tox > 0.5)[0], np.where(tox < 0.5)[0]
        idx = np.concatenate([rng.choice(pos, min(n_pos, len(pos)), replace=False),
                              rng.choice(neg, min(n_neg, len(neg)), replace=False)])
        idx = np.sort(idx)
        sub = split.select(idx.tolist())
        return [{"a": clean(t), "b": None, "y": int(v > 0.5), "f": float(v)} for t, v in zip(sub["text"], sub["toxicity"])]
    return sample(ds["train"], 9000, 21000), sample(ds["validation"], 600, 1400), sample(ds["test"], 1800, 4200)


def load_cold():
    import pandas as pd
    from huggingface_hub import hf_hub_download
    def read(name):
        df = pd.read_csv(hf_hub_download("thu-coai/cold", name, repo_type="dataset"))
        tcol = [c for c in df.columns if c.lower() == "text"][0]
        return [{"a": clean(t), "b": None, "y": int(l), "f": float("nan")} for t, l in zip(df[tcol], df["label"]) if int(l) in (0, 1)]
    train, dev_all, test = read("train.csv"), read("dev.csv"), read("test.csv")
    rng = np.random.default_rng(SEED)
    dev = [dev_all[i] for i in sorted(rng.choice(len(dev_all), min(DEV_N, len(dev_all)), replace=False))]
    return train, dev, test


def load_fake():
    rng = np.random.default_rng(0)
    words = ["好", "坏", "great", "awful", "maybe", "fine", "hate", "love"]
    def make(n):
        out = []
        for _ in range(n):
            y = int(rng.random() < 0.35)
            a = " ".join(rng.choice(words, 8)) + (" love" if y else " fine")
            out.append({"a": a, "b": ("hyp " + a[:10]) if TASK == "nli" else None, "y": y,
                        "f": float(np.clip(y * 0.8 + rng.normal(0, 0.2), 0, 1)) if TASK != "cold" else float("nan")})
        return out
    return make(400), make(150), make(200)


print("\n===== 1. 整理数据 =====")
if FAKE:
    train, dev, test = load_fake()
else:
    train, dev, test = {"nli": load_nli, "toxicity": load_toxicity, "cold": load_cold}[TASK]()
key = lambda x: (x["a"], x["b"])  # noqa: E731
held = set(key(x) for x in dev + test)
seen, tr = set(), []
for x in train:
    if key(x) in held or key(x) in seen or not x["a"]:
        continue
    seen.add(key(x)); tr.append(x)
train = [tr[i] for i in sorted(np.random.default_rng(SEED).permutation(len(tr))[:TRAIN_MAX])] if len(tr) > TRAIN_MAX else tr
fold = np.random.default_rng(SEED).permutation(len(train)) % 2
if QUICK:
    keep = np.concatenate([np.where(fold == h)[0][:QUICK // 2] for h in (0, 1)])
    train, fold = [train[i] for i in keep], fold[keep]
    dev, test = dev[:QUICK], test[:QUICK]
SPLITS = {"dev": dev, "test": test}
for d in ("labels", "agree"):
    os.makedirs(f"{OUT}/{d}", exist_ok=True)
for n, d in SPLITS.items():
    np.save(f"{OUT}/labels/{n}_labels.npy", np.array([x["y"] for x in d]))
    np.save(f"{OUT}/agree/{n}_agree.npy", np.array([x["f"] for x in d], dtype=float))
if TASK == "nli":                                   # 保存前提编号（同一前提的几个假设在分析时按前提聚类重抽样）
    import hashlib
    os.makedirs(f"{OUT}/meta", exist_ok=True)
    for n, d in SPLITS.items():
        np.save(f"{OUT}/meta/{n}_premise.npy",
                np.array([int(hashlib.md5(x["a"].encode("utf-8")).hexdigest()[:15], 16) for x in d], dtype=np.int64))
METRICS["sizes"] = {"train": len(train), "half_A": int((fold == 0).sum()), "half_B": int((fold == 1).sum()),
                    "dev": len(dev), "test": len(test)}
METRICS["positive_rate"] = {n: float(np.mean([x["y"] for x in d])) for n, d in [("train", train), ("dev", dev), ("test", test)]}
METRICS["test_items_with_rater_distribution"] = int(np.sum([not math.isnan(x["f"]) for x in test]))
save_metrics()
print(f"训练 {len(train)}（A 半 {(fold == 0).sum()}，B 半 {(fold == 1).sum()}）；验证 {len(dev)}；测试 {len(test)}；"
      f"测试集正类比例 {METRICS['positive_rate']['test']:.3f}；有标注者分布的测试句 {METRICS['test_items_with_rater_distribution']}")


def auc(score, y):
    o = np.argsort(score, kind="stable")
    r = np.empty(len(score)); r[o] = np.arange(1, len(score) + 1)
    s = score[o]
    i = 0
    while i < len(s):                                         # 并列取平均秩
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[o[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def report(name, probs):
    out = {}
    for n, d in SPLITS.items():
        y = np.array([x["y"] for x in d])
        p = probs[n]
        out[n] = {"accuracy": float((p.argmax(1) == y).mean()), "auc": auc(np.log(p[:, 1] + 1e-9) - np.log(p[:, 0] + 1e-9), y),
                  "hit_rate": float((p.argmax(1)[y == 1] == 1).mean()), "false_alarm_rate": float((p.argmax(1)[y == 0] == 1).mean())}
    print(f"  {name}: 测试集 准确率 {out['test']['accuracy']:.4f}  AUC {out['test']['auc']:.4f}  "
          f"命中 {out['test']['hit_rate']:.3f}  虚报 {out['test']['false_alarm_rate']:.3f}")
    return out


# ============================================================
# 2. 编码器
# ============================================================
from torch.utils.data import DataLoader  # noqa: E402
from transformers import (AutoModelForCausalLM, AutoModelForSequenceClassification,  # noqa: E402
                          AutoTokenizer, get_linear_schedule_with_warmup)

print("\n===== 2. 编码器（A、B 两半 × 种子 42、43）=====")
etok = AutoTokenizer.from_pretrained(ENC_MODEL)


def ecollate(batch):
    if batch[0]["b"] is not None:
        enc = etok([x["a"] for x in batch], [x["b"] for x in batch], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
    else:
        enc = etok([x["a"] for x in batch], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
    enc["labels"] = torch.tensor([x["y"] for x in batch])
    return enc


use_amp = device == "cuda"
METRICS["encoders"] = {}
for h in ((0, 1) if PART == "all" else ()):
    for seed in (42, 43):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        g = torch.Generator(); g.manual_seed(seed)
        dl = DataLoader([train[i] for i in np.where(fold == h)[0]], batch_size=32, shuffle=True, collate_fn=ecollate, generator=g)
        model = AutoModelForSequenceClassification.from_pretrained(ENC_MODEL, num_labels=2, ignore_mismatched_sizes=True).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(opt, int(0.06 * len(dl) * 3), len(dl) * 3)
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        t0 = time.time()
        for ep in range(3):
            model.train()
            for batch in dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    loss = model(**batch).loss
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step()
        model.eval()
        probs = {}
        d_ = f"{OUT}/enc/half{h}/seed{seed}"
        os.makedirs(d_, exist_ok=True)
        with torch.no_grad():
            for n, d in SPLITS.items():
                out = []
                for batch in DataLoader(d, batch_size=128, collate_fn=ecollate):
                    batch.pop("labels")
                    with torch.autocast(device_type="cuda", enabled=use_amp):
                        out.append(torch.softmax(model(**{k: v.to(device) for k, v in batch.items()}).logits.float(), -1).cpu())
                probs[n] = torch.cat(out).numpy()
                np.save(f"{d_}/{n}_probs.npy", probs[n])
        METRICS["encoders"][f"half{h}|seed{seed}"] = report(f"编码器 {'AB'[h]} 半 种子 {seed}（{(time.time() - t0) / 60:.0f} 分钟）", probs)
        save_metrics()
        del model, opt, sched, scaler
        gc.collect(); torch.cuda.empty_cache() if device == "cuda" else None

# ============================================================
# 3、4. 大模型：零样本 + LoRA
# ============================================================
try:
    import peft  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft"], check=True)
try:                                                   # 与实验 14b 相同：绕开 Kaggle 上过旧的 torchao
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
from peft import LoraConfig, get_peft_model  # noqa: E402
def hf_login():
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        try:
            from kaggle_secrets import UserSecretsClient
            tok = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            tok = None
    if tok:
        from huggingface_hub import login
        login(token=tok, add_to_git_credential=False)
        print("已用 HF_TOKEN 登录 Hugging Face")
    else:
        print("没有找到 HF_TOKEN（不影响不需要许可的模型）")


def chat(tok, system, user):
    """有的模型（如 Gemma 2）的对话模板不接受 system 角色，这时把 system 并进 user。"""
    try:
        return tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                       tokenize=False, add_generation_prompt=True)
    except Exception:
        return tok.apply_chat_template([{"role": "user", "content": system + "\n\n" + user}],
                                       tokenize=False, add_generation_prompt=True)


def load_lm(name):
    """Gemma 3 的 4B 版本带视觉部分，旧版 transformers 不能用 AutoModelForCausalLM 加载，就换一个类。"""
    try:
        m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        m = AutoModelForImageTextToText.from_pretrained(name, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    return m.half()


def lora_targets(model, names):
    """带视觉部分的模型只在语言部分加 LoRA。"""
    if any("vision" in n for n, _ in model.named_modules()):
        return r".*language_model.*\.(" + "|".join(names) + r")$"
    return names



hf_login()

if TASK == "nli":
    SYSTEM = "You are an expert annotator for natural language inference. Answer with Yes or No only."
    USER = ("Premise: {a}\nHypothesis: {b}\n\n"
            "Given the premise, is the hypothesis definitely true? Answer Yes or No.")
elif TASK == "toxicity":
    SYSTEM = "You are an expert content moderator. Answer with Yes or No only."
    USER = ("Comment: {a}\n\nIs this comment toxic, that is, rude, disrespectful, or unreasonable enough "
            "that it would likely make someone leave a discussion? Answer Yes or No.")
else:
    SYSTEM = "你是一名内容审核专家。只回答“是”或“否”。"
    USER = "评论：{a}\n\n这条评论是否具有冒犯性（例如侮辱、攻击个人或群体、带有歧视或偏见）？只回答“是”或“否”。"
WORDS = [["否"], ["是"]] if LANG == "zh" else [["No", "no", " No", " no"], ["Yes", "yes", " Yes", " yes"]]


def run_llm(name, model_name, lora_half=None):
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ids = []
    for ws in WORDS:
        s = []
        for w in ws:
            t = tok(w, add_special_tokens=False)["input_ids"]
            if t and t[0] not in s:
                s.append(t[0])
        ids.append(s)
    assert not set(ids[0]) & set(ids[1]), "是 / 否 两个词的第一个 token 重合，请把这行发给 Claude"
    target_id = [ids[0][0], ids[1][0]]
    model = load_lm(model_name).to(device)

    def build(x):
        u = USER.format(a=x["a"][:MAX_CHARS], b=(x["b"] or "")[:MAX_CHARS])
        return chat(tok, SYSTEM, u)

    def last_logits(texts):
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            try:
                out = model(**enc, logits_to_keep=1)
            except TypeError:
                out = model(**enc)
        return out.logits[:, -1, :].float()

    log = []
    if lora_half is not None:
        model.config.use_cache = False
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=lora_targets(model, LORA_TARGETS),
                                                 bias="none", task_type="CAUSAL_LM"))
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        items = [train[i] for i in np.where(fold == lora_half)[0]]
        prompts = [build(x) for x in items]
        targets = torch.tensor([target_id[x["y"]] for x in items])
        lengths = np.array([len(p) for p in prompts])
        rng = np.random.default_rng(SEED)
        idx = rng.permutation(len(prompts))
        batches = []
        for c0 in range(0, len(idx), 256):
            ch = idx[c0:c0 + 256]; ch = ch[np.argsort(lengths[ch])]
            batches += [ch[i:i + 8] for i in range(0, len(ch), 8)]
        rng.shuffle(batches)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=2e-4, weight_decay=0.0)
        steps = math.ceil(len(batches) / 4)
        sched = get_linear_schedule_with_warmup(opt, int(0.03 * steps), steps)
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")
        model.train()
        running, nan_skips = [], 0
        print(f"  LoRA {'AB'[lora_half]} 半：{len(items)} 条 × 1 轮，{len(batches)} 个小批次")
        for bi, b in enumerate(batches, 1):
            try:
                loss = torch.nn.functional.cross_entropy(last_logits([prompts[i] for i in b]), targets[b].to(device)) / 4
                scaler.scale(loss).backward()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                for hb in (b[:len(b) // 2], b[len(b) // 2:]):
                    ls = torch.nn.functional.cross_entropy(last_logits([prompts[i] for i in hb]), targets[hb].to(device)) / 4 * len(hb) / len(b)
                    scaler.scale(ls).backward()
                loss = ls * len(b) / len(hb)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                nan_skips += 1
                if nan_skips >= 20 and nan_skips > 0.05 * bi:
                    raise RuntimeError("超过 5% 的批次损失是 nan/inf（16 位精度不稳定）")
                continue
            running.append(loss.item() * 4)
            if bi % 4 == 0 or bi == len(batches):
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            if bi == 40:
                print(f"  速度：每个小批次 {(time.time() - t0) / 40:.2f} 秒，预计训练 {(time.time() - t0) / 40 * len(batches) / 60:.0f} 分钟")
            if bi % 200 == 0:
                log.append({"micro_step": bi, "loss": float(np.mean(running[-200:])), "minutes": (time.time() - t0) / 60})
                print(f"  第 {bi}/{len(batches)} 个小批次  最近平均损失 {log[-1]['loss']:.4f}  已用 {log[-1]['minutes']:.0f} 分钟")
        model.eval()
    probs, mass = {}, {}
    d_ = f"{OUT}/llm/{name}"
    os.makedirs(d_, exist_ok=True)
    with torch.no_grad():
        for n, d in SPLITS.items():
            texts = [build(x) for x in d]
            order = np.argsort([len(t) for t in texts])
            p = np.zeros((len(texts), 2), dtype=np.float32); m = np.zeros(len(texts), dtype=np.float32)
            for b0 in range(0, len(texts), 16):
                ii = order[b0:b0 + 16]
                lg = last_logits([texts[i] for i in ii])
                if not torch.isfinite(lg).all():
                    raise RuntimeError("预测时模型输出出现 nan/inf（16 位精度溢出）")
                sm = torch.softmax(lg, -1)
                s0, s1 = sm[:, ids[0]].sum(1), sm[:, ids[1]].sum(1)
                p[ii, 0] = (s0 / (s0 + s1 + 1e-12)).cpu().numpy(); p[ii, 1] = (s1 / (s0 + s1 + 1e-12)).cpu().numpy()
                m[ii] = (s0 + s1).cpu().numpy()
            probs[n], mass[n] = p, float(m.mean())
            np.save(f"{d_}/{n}_probs.npy", p)
    res = report(f"{name}（{(time.time() - t0) / 60:.0f} 分钟）", probs)
    res["answer_token_mass"] = mass
    res["loss_log"] = log
    print(f"    是 / 否 两个词的概率合计（测试集平均）{mass['test']:.3f}")
    del model
    gc.collect(); torch.cuda.empty_cache() if device == "cuda" else None
    return res


METRICS["llm"] = {}
if PART == "all":
    print("\n===== 3. 零样本大模型 =====")
    for name, mn in ZS_MODELS.items():
        METRICS["llm"][name] = run_llm(name, mn); save_metrics()
    print("\n===== 4. LoRA（A、B 两半各 1 轮）=====")
    for h in (0, 1):
        METRICS["llm"][f"lora_half{h}"] = run_llm(f"lora_half{h}", LORA_MODEL, lora_half=h); save_metrics()

if FAMILY_MODEL:
    print(f"\n===== 5. 第二个大模型家族：{FAMILY_MODEL}（零样本 + A、B 两半 LoRA）=====")
    os.makedirs(f"{OUT}/llm", exist_ok=True)
    with open(f"{OUT}/llm/x_model.json", "w") as f:
        json.dump({"model": FAMILY_MODEL}, f)
for name, half in ((("zs_x", None), ("lora_x_half0", 0), ("lora_x_half1", 1)) if FAMILY_MODEL else ()):
    try:
        METRICS["llm"][name] = run_llm(name, FAMILY_MODEL, lora_half=half)
    except Exception as ex:                  # 第二个家族出错时不影响前面已经算好的结果
        METRICS["llm"][name] = {"error": repr(ex)[:800]}
        print(f"  {name} 出错（前面的结果不受影响），请把这行发给 Claude：", repr(ex)[:300])
        gc.collect(); torch.cuda.empty_cache() if device == "cuda" else None
    save_metrics()

save_metrics()
print(f"\n全部完成，总用时 {(time.time() - T_START) / 60:.0f} 分钟。请把 {os.path.basename(MFILE)} 发给 Claude。")
