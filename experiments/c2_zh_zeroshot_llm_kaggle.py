# ============================================================
# 实验 11（C2）：中文零样本大模型（开源模型，在 Kaggle 的 GPU 上跑）
#
# 模型：Qwen/Qwen3-4B-Instruct-2507（Apache-2.0，约 40 亿参数，纯文本，不带"思考"模式）
# 做法：不让模型自由生成，而是直接读出它对 6 个类别名称的打分，取最高的作为预测。
#       好处：(1) 每条只算一次，快；(2) 得到的是连续的概率，可以画出大模型自己的 ROC 曲线，
#       英文实验里的大模型只有"有/无"二值结果，做不到这一点。
#       脚本会抽 200 条让模型真的自由生成，核对两种方式的结果是否一致。
# 两种提示措辞（v1 简洁版；v2 带类别定义版），用来检查结论是否依赖措辞。
#
# 运行前：
#   1. 新建一个 Notebook，右侧 Add Input 添加你上传的 EWECT 数据集（和实验 8 用的是同一个）。
#   2. 右侧 Settings：Accelerator 选 GPU T4 x2，Internet 选 On。
#   3. 粘贴本脚本，Save Version -> Save & Run All。预计 30 到 60 分钟。
#
# 产出（/kaggle/working/ 下）：
#   zh_llm/{模型名}/{v1,v2}/{集合名}_probs.npy   每条样本的 6 类概率（格式与编码器的相同）
#   zh_llm_metrics.json                          指标（发给 Claude 的就是这个）
# ============================================================
import glob
import json
import os
import re
import time
import zipfile

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = os.environ.get("ZH_LLM", "Qwen/Qwen3-4B-Instruct-2507").split(",")
BATCH_SIZE = 16
MAX_CHARS = 300                       # 超长微博截断到 300 字（95% 的微博在 150 字以内）
N_FREE_CHECK = 200                    # 抽多少条做自由生成核对
QUICK = int(os.environ.get("ZH_QUICK", "0"))          # 仅供本地冒烟测试
ROOT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
INPUT_ROOTS = ["/kaggle/input", os.environ.get("ZH_DATA", ROOT)]
ZH_LABELS = ["angry", "fear", "happy", "sad", "surprise", "neutral"]      # 与实验 8 的顺序一致
EVAL_SETS = ["usual_eval", "usual_test", "virus_eval", "virus_test"]
NEU = ZH_LABELS.index("neutral")

PROMPTS = {
    "v1": {
        "words": ["愤怒", "恐惧", "开心", "悲伤", "惊讶", "中性"],
        "system": "你是一名情绪分类专家。请判断这条微博作者的主要情绪，只能从给定的类别中选一个。"
                  "只输出类别名称，不要解释，不要标点，不要多余的字。",
        "user": "类别：愤怒 | 恐惧 | 开心 | 悲伤 | 惊讶 | 中性\n\n微博：{text}\n情绪：",
    },
    "v2": {
        "words": ["愤怒", "恐惧", "开心", "悲伤", "惊讶", "无情绪"],
        "system": "你是一名情绪分类专家。请判断这条微博作者的主要情绪，只能从给定的类别中选一个。"
                  "只输出类别名称，不要解释，不要标点，不要多余的字。",
        "user": "类别及含义：\n"
                "愤怒：生气、不满、厌烦、指责\n"
                "恐惧：害怕、担心、紧张、焦虑\n"
                "开心：高兴、喜欢、感激、期待、祝福\n"
                "悲伤：难过、失望、遗憾、想念、无奈\n"
                "惊讶：意外、吃惊、难以置信\n"
                "无情绪：客观陈述或转述信息，看不出作者的情绪\n\n"
                "微博：{text}\n情绪：",
    },
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print("运行设备:", device, f"（GPU 数量 {torch.cuda.device_count()}）" if device == "cuda" else "（没有打开 GPU，会非常慢）")


# ---------- 1. 读数据（与实验 8 相同的找法） ----------
def candidates(basename):
    out = []
    for root in INPUT_ROOTS:
        if os.path.isdir(root):
            out += glob.glob(os.path.join(root, "**", basename), recursive=True)
    return sorted(set(out))


if not candidates("usual_eval_labeled.txt"):
    for root in INPUT_ROOTS:
        for z in glob.glob(os.path.join(root, "**", "*.zip"), recursive=True):
            with zipfile.ZipFile(z) as zf:
                zf.extractall(f"{ROOT}/ewect_data")
    INPUT_ROOTS.append(f"{ROOT}/ewect_data")


def load_split(name):
    best = None
    for p in candidates(f"{name}_labeled.txt"):
        with open(p, "rb") as f:
            d = json.loads(f.read().decode("utf-8-sig"))
        if all(x.get("label") in ZH_LABELS for x in d) and (best is None or len(d) > len(best)):
            best = d
    if best is None:
        raise SystemExit(f"没有找到带完整标签的 {name}_labeled.txt。请确认已在 Add Input 里添加 EWECT 数据集。")
    return [{"id": x["id"], "text": str(x["content"]), "label": ZH_LABELS.index(x["label"])} for x in best]


DATA = {n: load_split(n) for n in EVAL_SETS}
if QUICK:
    DATA = {n: d[:QUICK] for n, d in DATA.items()}
for n, d in DATA.items():
    print(f"{n}: {len(d)} 条")


def clean(text):
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_CHARS]


def score(probs, labels):
    pred = probs.argmax(1)
    return {"accuracy": float(accuracy_score(labels, pred)),
            "macro_f1": float(f1_score(labels, pred, average="macro", labels=list(range(6)), zero_division=0)),
            "pred_neutral_rate": float((pred == NEU).mean()), "gold_neutral_rate": float((labels == NEU).mean()),
            "detection_hit_rate": float(((pred != NEU) & (labels != NEU)).sum() / max((labels != NEU).sum(), 1)),
            "detection_false_alarm_rate": float(((pred != NEU) & (labels == NEU)).sum() / max((labels == NEU).sum(), 1))}


# ---------- 2. 逐个模型、逐种措辞预测 ----------
metrics = {"models": MODELS, "labels": ZH_LABELS, "max_chars": MAX_CHARS, "runs": {}}
for model_name in MODELS:
    slug = model_name.split("/")[-1]
    print(f"\n===== 加载模型 {model_name}（第一次要下载约 8 GB，几分钟）=====")
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        kw = {"device_map": "auto"} if device == "cuda" else {}
        dt = torch.float16 if device == "cuda" else torch.float32
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dt, **kw)
        except TypeError:                               # 旧版本 transformers 用的参数名不同
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dt, **kw)
    except Exception as ex:
        print("模型加载失败：", repr(ex)[:500])
        print("如果提示不认识这个模型结构，请在脚本最前面加一个单元格运行：!pip install -q -U transformers accelerate ，然后重启再跑。")
        raise
    model.eval()
    tok.padding_side = "left"                       # 批量时在左边补齐，这样每条的最后一个位置都是"该输出答案"的位置
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    in_dev = model.get_input_embeddings().weight.device

    def build(prompt, text):
        msgs = [{"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"].format(text=clean(text))}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    for pname, prompt in PROMPTS.items():
        first_ids = [tok(w, add_special_tokens=False)["input_ids"][0] for w in prompt["words"]]
        print(f"\n--- 措辞 {pname}：6 个类别名称的第一个 token = {[tok.decode([i]) for i in first_ids]} ---")
        if len(set(first_ids)) < 6:
            raise SystemExit("有两个类别名称的第一个 token 相同，无法区分。请把这行输出发给 Claude。")
        run = {}
        os.makedirs(f"{ROOT}/zh_llm/{slug}/{pname}", exist_ok=True)
        for n in EVAL_SETS:
            texts = [build(prompt, x["text"]) for x in DATA[n]]
            labels = np.array([x["label"] for x in DATA[n]])
            order = np.argsort([len(t) for t in texts])           # 长度相近的放一批，省时间
            probs = np.zeros((len(texts), 6), dtype=np.float32)
            mass = np.zeros(len(texts), dtype=np.float32)
            t0 = time.time()
            for b0 in range(0, len(texts), BATCH_SIZE):
                idx = order[b0:b0 + BATCH_SIZE]
                enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(in_dev)
                with torch.no_grad():
                    logits = model(**enc).logits[:, -1, :].float()
                full = torch.softmax(logits, dim=-1)[:, first_ids]                 # 模型在整个词表上给这 6 个词的概率
                mass[idx] = full.sum(1).cpu().numpy()
                probs[idx] = torch.softmax(logits[:, first_ids], dim=-1).cpu().numpy()   # 只在这 6 个词之间归一化
                if (b0 // BATCH_SIZE) % 50 == 0:
                    print(f"  {n}: {b0}/{len(texts)}  已用 {time.time() - t0:.0f} 秒")
            np.save(f"{ROOT}/zh_llm/{slug}/{pname}/{n}_probs.npy", probs)
            run[n] = score(probs, labels)
            run[n]["label_token_mass_mean"] = float(mass.mean())
            run[n]["label_token_mass_below_half"] = float((mass < 0.5).mean())
            print(f"  {n}: 准确率={run[n]['accuracy']:.4f}  Macro-F1={run[n]['macro_f1']:.4f}  "
                  f"预测为中性={run[n]['pred_neutral_rate']:.3f}（金标 {run[n]['gold_neutral_rate']:.3f}）  "
                  f"6 个类别词占全部概率的平均 {mass.mean():.3f}")

        # 核对：抽一部分让模型真的自由生成，看和"读打分"的结果是否一致
        n = "usual_test"
        k = min(N_FREE_CHECK, len(DATA[n]))
        sub = np.random.default_rng(0).choice(len(DATA[n]), k, replace=False)
        saved = np.load(f"{ROOT}/zh_llm/{slug}/{pname}/{n}_probs.npy")
        agree, parsed_ok = 0, 0
        for b0 in range(0, k, BATCH_SIZE):
            idx = sub[b0:b0 + BATCH_SIZE]
            enc = tok([build(prompt, DATA[n][i]["text"]) for i in idx], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(in_dev)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
            for j, i in enumerate(idx):
                txt = tok.decode(out[j, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                hit = [c for c, w in enumerate(prompt["words"]) if txt.startswith(w)] or \
                      [c for c, w in enumerate(prompt["words"]) if w in txt]
                if hit:
                    parsed_ok += 1
                    agree += int(hit[0] == saved[i].argmax())
        run["free_generation_check"] = {"n": int(k), "parsed_rate": parsed_ok / k, "agreement_among_parsed": agree / max(parsed_ok, 1)}
        print(f"  自由生成核对（{k} 条）：能解析出类别的占 {parsed_ok / k:.3f}；其中与读打分结果一致的占 {agree / max(parsed_ok, 1):.3f}")
        metrics["runs"][f"{slug}|{pname}"] = run
        with open(f"{ROOT}/zh_llm_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

print("\n全部完成。请把 zh_llm_metrics.json 发给 Claude。概率文件留在这个 Notebook 的输出里，后面的分析会用到。")
