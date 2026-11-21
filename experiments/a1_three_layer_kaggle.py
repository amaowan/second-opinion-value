# ============================================================
# 实验 A1：主分析（取代 P1 与 G2）。论文：When Is a Second Opinion Worth Asking For?
#
# 每个任务、每对模型（A = 给出完整概率的编码器，B = 第二判断者），在任务原本的标签空间里算三层：
#   1. 同预算理想挑选：恰好调用 k 次、调用后用 B 覆盖 A，知道真值时最多能多做对多少
#   2. 实际兑现：置信度门控（调用 A 最没把握的 k 条）与学习路由（验证集拟合、测试集锁定），拆成改对与改坏
#   3. 附加信息：已知 A 的完整概率后，B 的一票能否改善留出集 log loss（两个多项逻辑回归，同样的正则化选择流程）
# 另外：E15 四格配对表；ChaosNLI 分半稳健性；次要问题（能否事前估计，只比恒零与直接测量）。
# 英文情绪的大模型只有测试集输出，附加信息与学习路由改用测试集两折交叉拟合（正则化也在各自的训练半折内选）。
# 中文疫情迁移：一切拟合都在通用微博验证集上完成，冻结后直接用于疫情测试集。
# 先为每个任务写一张统一逐题表（a1_items_*.npz），所有指标都从同一批预测算出。
#
# 需要的输入（Add Input → Notebooks）：
#   中文：实验 8（编码器）、实验 11（零样本）、实验 14b（普通 LoRA）、实验 15b 的两个（加权、互补 LoRA）
#   英文：实验 3（GoEmotions 编码器 3 个种子）；原作者缓存会自动从 GitHub 下载（需要 Internet On）
#   新任务：G1 nli、G1 toxicity（旧版即可）、G1 cold（新跑的）
# Accelerator 选 None，Internet 选 On。约 1 到 2 小时。输出：a1_results.json、a1_items_*.npz
# ============================================================
import re
import time
import json
import math
import os
from collections import defaultdict
from statistics import NormalDist

import numpy as np

Z = NormalDist().inv_cdf
PHI = NormalDist().cdf
GL_X, GL_W = np.polynomial.legendre.leggauss(64)

try:
    from scipy.special import ndtr as ncdf
except Exception:  # 没有 scipy 时的后备
    _erf = np.vectorize(math.erf)

    def ncdf(x):
        return 0.5 * (1.0 + _erf(np.asarray(x, float) / math.sqrt(2.0)))


def clipz(p):
    return Z(min(max(float(p), 1e-6), 1 - 1e-6))


def sdt(resp, truth, w):
    """返回 (命中率, 虚报率, d', c)，log-linear 校正"""
    s, nz = truth == 1, truth == 0
    H = (w[s & (resp == 1)].sum() + 0.5) / (w[s].sum() + 1.0)
    F = (w[nz & (resp == 1)].sum() + 0.5) / (w[nz].sum() + 1.0)
    return H, F, Z(H) - Z(F), -0.5 * (Z(H) + Z(F))


def roc_curve(score, truth, w):
    order = np.argsort(-score, kind="stable")
    t, ww = truth[order], w[order]
    tp = np.concatenate([[0.0], np.cumsum(ww * (t == 1))])
    fp = np.concatenate([[0.0], np.cumsum(ww * (t == 0))])
    return fp / max(fp[-1], 1e-12), tp / max(tp[-1], 1e-12)


def auc_of(score, truth, w):
    fpr, tpr = roc_curve(score, truth, w)
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))


def bvn_upper(h, k, rho):
    """二元标准正态 P(X>h, Y>k)，相关为 rho。用恒等式 dP/d rho = 二元密度，对 rho 做数值积分"""
    r = 0.5 * rho * (GL_X + 1.0)
    dens = np.exp(-(h * h - 2 * r * h * k + k * k) / (2 * (1 - r * r))) / (2 * np.pi * np.sqrt(1 - r * r))
    return PHI(-h) * PHI(-k) + 0.5 * rho * float(np.sum(GL_W * dens))


def tetrachoric(n11, n10, n01, n00):
    """四分相关：假设两个二值反应各自来自一个正态的内部变量过阈值，求这两个内部变量的相关"""
    n11, n10, n01, n00 = n11 + 0.5, n10 + 0.5, n01 + 0.5, n00 + 0.5
    N = n11 + n10 + n01 + n00
    h, k, target = clipz(1 - (n11 + n10) / N), clipz(1 - (n11 + n01) / N), n11 / N
    lo, hi = -0.995, 0.995
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        if bvn_upper(h, k, mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def dvc(ra, rb, truth, w):
    """决策变量相关：分别在"有情绪"(truth=1) 和"中性"(truth=0) 的句子内部估计"""
    out = []
    for cls in (1, 0):
        m = truth == cls
        a, b, ww = ra[m] == 1, rb[m] == 1, w[m]
        out.append(tetrachoric(ww[a & b].sum(), ww[a & ~b].sum(), ww[~a & b].sum(), ww[~a & ~b].sum()))
    return out[0], out[1]


def d_from_auc(a):
    return math.sqrt(2.0) * clipz(a)


def sim_fused_uv(muA, muB, sigma, rho_sig, rho_noise, t, n=300000, seed=0):
    """不等方差版本的理论预测：有情绪的句子上两个内部变量的标准差为 sigma（中性为 1）。返回 (合并后 AUC, A 单独 AUC)"""
    rng = np.random.default_rng(seed)
    r1, r0 = min(max(rho_sig, -0.99), 0.99), min(max(rho_noise, -0.99), 0.99)
    u1 = rng.standard_normal(n); v1 = r1 * u1 + math.sqrt(1 - r1 * r1) * rng.standard_normal(n)
    u0 = rng.standard_normal(n); v0 = r0 * u0 + math.sqrt(1 - r0 * r0) * rng.standard_normal(n)
    x = np.concatenate([muA + sigma * u1, u0]); b = np.concatenate([(muB + sigma * v1) > t, v0 > t])
    y = np.concatenate([np.ones(n), np.zeros(n)]).astype(int)
    p1 = np.clip(ncdf((r1 * (x - muA) / sigma - (t - muB) / sigma) / math.sqrt(1 - r1 * r1)), 1e-12, 1 - 1e-12)
    p0 = np.clip(ncdf((r0 * x - t) / math.sqrt(1 - r0 * r0)), 1e-12, 1 - 1e-12)
    llr = -0.5 * ((x - muA) / sigma) ** 2 - math.log(sigma) + 0.5 * x * x + np.where(b, np.log(p1 / p0), np.log((1 - p1) / (1 - p0)))
    one = np.ones(2 * n)
    return auc_of(llr, y, one), auc_of(np.where(sigma == 1.0, x, -0.5 * ((x - muA) / sigma) ** 2 + 0.5 * x * x), y, one)


SEEDS = [42, 43, 44]
N_BOOT = 300
K_GRID = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
SEARCH_ROOTS = [os.environ.get("A1_INPUT", "/kaggle/input"), OUT_DIR]   # outside Kaggle: A1_INPUT=../predictions
ZH_LABELS = ["angry", "fear", "happy", "sad", "surprise", "neutral"]
NEU = ZH_LABELS.index("neutral")
DOMAINS = ["usual", "virus"]
OTHER = {"usual": "virus", "virus": "usual"}
ZGRID = np.linspace(0.03, 0.70, 28)
ZGRID_Z = np.array([Z(g) for g in ZGRID])


def find_file(suffix):
    for root in SEARCH_ROOTS:
        if not os.path.isdir(root):
            continue
        for d, _, files in os.walk(root, followlinks=True):
            for fn in files:
                p = os.path.join(d, fn)
                if p.replace("\\", "/").endswith(suffix):
                    return p
    return None


def need(suffix):
    p = find_file(suffix)
    if p is None:
        raise SystemExit(f"没有找到 {suffix}。请在右侧 Add Input 里添加实验 8（中文编码器）那个 Notebook 的输出。")
    return p


def zroc_fit(fpr, tpr):
    hh = np.clip(np.interp(ZGRID, fpr, tpr), 1e-4, 1 - 1e-4)
    slope, icpt = np.polyfit(ZGRID_Z, [Z(h) for h in hh], 1)
    return float(slope), float(icpt)


def best_threshold(score, truth, w):
    """在给定数据上找让检测准确率最高的阈值"""
    order = np.argsort(-score, kind="stable")
    s, t, ww = score[order], truth[order], w[order]
    acc = (np.concatenate([[0.0], np.cumsum(ww * (t == 1))]) + (ww * (t == 0)).sum()
           - np.concatenate([[0.0], np.cumsum(ww * (t == 0))])) / ww.sum()
    i = int(np.argmax(acc))
    return float("inf") if i == 0 else (float(s[i - 1]) if i == len(s) else float(0.5 * (s[i - 1] + s[i])))


def score_of(p):
    return np.log(np.clip(1 - p[:, NEU], 1e-7, 1) / np.clip(p[:, NEU], 1e-7, 1))


def crossfit_gain(x, b, truth, w, fold):
    fa, fb = [], []
    for f in (0, 1):
        tr, te = fold != f, fold == f
        best = K_GRID[int(np.argmax([auc_of(x[tr] + k * b[tr], truth[tr], w[tr]) for k in K_GRID]))]
        fa.append(auc_of(x[te] + best * b[te], truth[te], w[te])); fb.append(auc_of(x[te], truth[te], w[te]))
    return float(np.mean(fa) - np.mean(fb))


def pair_params(pa, pb, truth, w):
    """预测所需的全部参数：基础模型的 z-ROC（斜率、截距），第二个模型的命中率和虚报率，两者的证据相关"""
    x, ra, rb = score_of(pa), (pa.argmax(1) != NEU).astype(int), (pb.argmax(1) != NEU).astype(int)
    sl, ic = zroc_fit(*roc_curve(x, truth, w))
    Hb, Fb, dB, _ = sdt(rb, truth, w)
    _, _, dA, _ = sdt(ra, truth, w)
    r1, r0 = dvc(ra, rb, truth, w)
    return {"slope": sl, "icpt": ic, "H_B": Hb, "F_B": Fb, "dA": dA, "dB": dB, "dvc_sig": r1, "dvc_noise": r0}


def predict_gain(pm, independent=False):
    sigma, muA, t = 1.0 / pm["slope"], pm["icpt"] / pm["slope"], -Z(pm["F_B"])
    muB = t + sigma * Z(pm["H_B"])
    a_f, a_s = sim_fused_uv(muA, muB, sigma, 0.0 if independent else pm["dvc_sig"], 0.0 if independent else pm["dvc_noise"], t)
    return a_f - a_s


# ============================================================
# 读取全部设置：中文情绪（通用、疫情）+ G1 的三个任务。每个判断者都换成"检测"二分类概率 [负类, 正类]


# ============================================================
# 以上是共用的纯计算函数（与实验 12、P1 相同）。下面是 A1 的内容。
# ============================================================
import traceback
import warnings
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*converge.*")

N_BOOT_A = int(os.environ.get("A_BOOT", "2000"))
N_SIM = int(os.environ.get("A_NSIM", "60000"))
QS = [0.1, 0.2, 0.4]
CS = [0.01, 0.1, 1.0, 10.0]
EKMAN = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
GO_TO_EKMAN = {"admiration": "joy", "amusement": "joy", "anger": "anger", "annoyance": "anger", "approval": "joy",
               "caring": "joy", "confusion": "surprise", "curiosity": "surprise", "desire": "joy", "disappointment": "sadness",
               "disapproval": "anger", "disgust": "disgust", "embarrassment": "sadness", "excitement": "joy", "fear": "fear",
               "gratitude": "joy", "grief": "sadness", "joy": "joy", "love": "joy", "nervousness": "fear", "optimism": "joy",
               "pride": "joy", "realization": "surprise", "relief": "joy", "remorse": "sadness", "sadness": "sadness",
               "surprise": "surprise", "neutral": "neutral"}
T0 = time.time()


def walk():
    for root in SEARCH_ROOTS:
        if os.path.isdir(root):
            for d, _, files in os.walk(root, followlinks=True):
                yield d.replace("\\", "/"), set(files)


DIRS = list(walk())


def J(test_prob=None, test_pred=None, dev_prob=None, dev_pred=None):
    """一个判断者：概率（有的话）和预测类别；只有标签的判断者（英文缓存）概率为 None"""
    tp = test_pred if test_pred is not None else test_prob.argmax(1)
    dp = dev_pred if dev_pred is not None else (dev_prob.argmax(1) if dev_prob is not None else None)
    return {"tp": test_prob, "tpred": np.asarray(tp), "dp": dev_prob, "dpred": None if dp is None else np.asarray(dp)}


SETTINGS = {}

# ---------- 中文情绪：实验 8 的编码器、实验 11 的零样本、14b 和 15b 的 LoRA ----------
zl = [d for d, f in DIRS if d.endswith("/zh/labels") and "usual_test_labels.npy" in f]
if zl:
    zroot = zl[0][: -len("/labels")]
    L = lambda n: np.load(f"{zroot}/labels/{n}_labels.npy")  # noqa: E731
    E = {(dom, s): {n: np.load(f"{zroot}/{dom}/seed{s}/{n}_probs.npy") for n in ("usual_eval", "usual_test", "virus_eval", "virus_test")}
         for dom in ("usual", "virus") for s in (42, 43, 44) if os.path.exists(f"{zroot}/{dom}/seed{s}/usual_test_probs.npy")}
    LLM = {}
    want = {"Qwen3-4B-LoRA-ep3/usual": "LoRA", "Qwen3-4B-LoRAW-ep3/usual": "LoRA（加权损失）",
            "Qwen3-4B-LoRAC-ep3/usual": "LoRA（互补损失）", "Qwen3-4B-Instruct-2507/v1": "零样本Qwen3"}
    for d, f in DIRS:
        for key, nm in want.items():
            if d.endswith("/zh_llm/" + key) and "usual_test_probs.npy" in f and nm not in LLM:
                LLM[nm] = {n: np.load(f"{d}/{n}_probs.npy") for n in ("usual_eval", "usual_test", "virus_eval", "virus_test")}
    print("中文：编码器 %d 个；大模型 %s" % (len(E), sorted(LLM)))
    for dom, sname in (("usual", "中文情绪-通用"), ("virus", "中文情绪-疫情（迁移）")):
        # 迁移设定：路由、融合模型和事前预测都只用通用微博的验证集拟合，原样用到疫情测试集上（规则冻结后直接迁移）
        ev, te = "usual_eval", f"{dom}_test"
        S = {"K": 6, "y_dev": L(ev), "y_test": L(te), "neutral": 5, "judges": {}, "pairs": [], "dev_mode": "dev"}
        for s in (42, 43, 44):
            if ("usual", s) in E:   # 基础判断者：在通用微博上训练的编码器（疫情集上即为迁移）
                S["judges"][f"编码器{s}"] = J(E[("usual", s)][te], dev_prob=E[("usual", s)][ev])
            if ("virus", s) in E:
                S["judges"][f"疫情编码器{s}"] = J(E[("virus", s)][te], dev_prob=E[("virus", s)][ev])
        for nm, v in LLM.items():
            S["judges"][nm] = J(v[te], dev_prob=v[ev])
        for s in (42, 43, 44):
            a = f"编码器{s}"
            if a not in S["judges"]:
                continue
            for nm in ("LoRA", "零样本Qwen3", "LoRA（加权损失）", "LoRA（互补损失）"):
                if nm in LLM:
                    S["pairs"].append((a, nm, nm))
            o = f"编码器{[42, 43, 44][([42, 43, 44].index(s) + 1) % 3]}"
            if o in S["judges"]:
                S["pairs"].append((a, o, "另一种子编码器"))
            if f"疫情编码器{s}" in S["judges"]:
                S["pairs"].append((a, f"疫情编码器{s}", "疫情训练的编码器"))
        if S["pairs"]:
            SETTINGS[sname] = S

# ---------- 英文情绪：实验 3 的编码器（只有测试集与验证集概率），原作者缓存的大模型（只有测试集） ----------
enc_en = {}
for d, f in DIRS:
    for s in (42, 43, 44):
        if d.endswith(f"/seed{s}") and "test_probs.npy" in f and "test_labels.npy" in f and s not in enc_en:
            enc_en[s] = (np.load(f"{d}/test_probs.npy"), np.load(f"{d}/test_labels.npy"))
repo = [d for d, f in DIRS if d.endswith("/data/cache/verifier") and "llm_lora_ge.json" in f]
if enc_en and not repo and not os.environ.get("A_NO_CLONE"):
    try:
        import subprocess
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/tikhomirovd/selective-emotion-classification.git",
                        f"{OUT_DIR}/sec_repo"], check=True)
        repo = [f"{OUT_DIR}/sec_repo/data/cache/verifier"]
    except Exception as ex:
        print("英文缓存下载失败，跳过英文：", repr(ex)[:200])
if enc_en and repo:
    R = repo[0][: -len("/data/cache/verifier")]
    def lab(rel):
        with open(f"{R}/{rel}") as f_:
            return np.array([EKMAN.index(p) if p in EKMAN else -1 for p in json.load(f_)["predictions"]])
    yE = list(enc_en.values())[0][1]
    S = {"K": 7, "y_dev": None, "y_test": yE, "neutral": 6, "judges": {}, "pairs": [], "dev_mode": "crossfit"}
    for s, (p, y_) in enc_en.items():
        assert (y_ == yE).all(), "英文编码器的测试集标签不一致"
        S["judges"][f"编码器{s}"] = J(p)
    for nm, rel in (("LoRA（Qwen3.5-4B）", "data/cache/verifier/llm_lora_ge.json"), ("零样本Qwen3.5-4B", "data/cache/verifier/llm_zs_ge.json"),
                    ("零样本Qwen3.7-Max", "data/cache/ladder/qwen3.7-max_ge_v6_n5427_s42.json"),
                    ("零样本Gemma3-4B", "data/cache/ladder/gemma3_4b_ge_v6_n5427_s42.json")):
        v = lab(rel)
        assert len(v) == len(yE), f"{nm} 条数不符"
        S["judges"][nm] = J(test_pred=v)
    for s in enc_en:
        for nm in ("LoRA（Qwen3.5-4B）", "零样本Qwen3.5-4B", "零样本Qwen3.7-Max", "零样本Gemma3-4B"):
            S["pairs"].append((f"编码器{s}", nm, nm))
        o = [t for t in enc_en if t != s][0] if len(enc_en) > 1 else None
        if o:
            S["pairs"].append((f"编码器{s}", f"编码器{o}", "另一种子编码器"))
    SETTINGS["英文情绪"] = S

# ---------- NLI 前提编号：旧版 G1 没有保存时，按 G1 的划分规则从 MNLI 原始数据重建，并逐项核对 ----------
def rebuild_nli_premise(y_test, f_test, dev_n=2000, seed=42):
    """完全照 G1 load_nli 的步骤：MNLI matched 验证集 → 用 ChaosNLI（HF 镜像）标出 1599 句 →
    在其余句子里用种子 42 抽 2000 句做验证集 → 剩下的按原顺序是测试集。
    核对测试集长度、标签、ChaosNLI 位置和比例四项全部一致才返回；任何一项不一致返回 None（退回逐题重抽）。"""
    import hashlib
    try:
        from datasets import load_dataset
        cl = lambda t: re.sub(r"\s+", " ", str(t)).strip()  # noqa: E731
        vm = load_dataset("nyu-mll/multi_nli", split="validation_matched")
        val = [(cl(a), cl(b), int(l == 0)) for a, b, l in zip(vm["premise"], vm["hypothesis"], vm["label"]) if l in (0, 1, 2)]
        dist = {}
        cm = load_dataset("metaeval/chaos-mnli-ambiguity")
        for sp in cm:
            for d in cm[sp]:
                dist[(cl(d["premise"]), cl(d["hypothesis"]))] = float(d["label_dist"][0])
        fv = np.array([dist.get((a, b), np.nan) for a, b, _ in val])
        has = ~np.isnan(fv)
        rng = np.random.default_rng(seed)
        dev_idx = set(rng.choice(np.where(~has)[0], dev_n, replace=False).tolist())
        te = [i for i in range(len(val)) if i not in dev_idx]
        ok = (len(te) == len(y_test)
              and (np.array([val[i][2] for i in te]) == y_test).all()
              and (has[te] == ~np.isnan(f_test)).all()
              and np.allclose(fv[te][has[te]], f_test[~np.isnan(f_test)]))
        if not ok:
            print("   NLI 前提重建：核对不一致，退回逐题重抽")
            return None
        print(f"   NLI 前提重建：{len(te)} 句全部核对一致，前提 {len(set(val[i][0] for i in te))} 个")
        return np.array([int(hashlib.md5(val[i][0].encode("utf-8")).hexdigest()[:15], 16) for i in te], dtype=np.int64)
    except Exception as ex:
        print("   NLI 前提重建失败，退回逐题重抽：", repr(ex)[:200])
        return None


# ---------- G1 的三个任务（二分类；同一任务的多个输出会合并，并核对标签一致） ----------
TB = defaultdict(list)
for d, f in DIRS:
    m = re.search(r"/tasks/(nli|toxicity|cold)/labels$", d)
    if m and "test_labels.npy" in f:
        TB[m.group(1)].append(d[: -len("/labels")])
AGREE = {}
for task, bases in sorted(TB.items()):
    def ld(p_, req=True):
        for b in bases:
            if os.path.exists(f"{b}/{p_}"):
                return np.load(f"{b}/{p_}")
        if req:
            raise FileNotFoundError(p_)
    ys = [np.load(f"{b}/labels/test_labels.npy") for b in bases]
    if any(len(y_) != len(ys[0]) or (y_ != ys[0]).any() for y_ in ys):
        raise SystemExit(f"{task} 的几个 G1 输出标签不一致，数据划分不同。请把这行发给 Claude。")
    try:
        S = {"K": 2, "y_dev": ld("labels/dev_labels.npy"), "y_test": ld("labels/test_labels.npy"), "neutral": 0,
             "judges": {}, "pairs": [], "dev_mode": "dev"}
        encs = []
        for h, hn in ((0, "A"), (1, "B")):
            for s in (42, 43):
                nm = f"编码器{hn}{s}"
                S["judges"][nm] = J(ld(f"enc/half{h}/seed{s}/test_probs.npy"), dev_prob=ld(f"enc/half{h}/seed{s}/dev_probs.npy"))
                encs.append(nm)
            S["judges"][f"LoRA_{hn}"] = J(ld(f"llm/lora_half{h}/test_probs.npy"), dev_prob=ld(f"llm/lora_half{h}/dev_probs.npy"))
        for nm, dn in (("零样本Qwen3", "zs_qwen3_4b"), ("零样本Qwen2.5", "zs_qwen25_3b")):
            S["judges"][nm] = J(ld(f"llm/{dn}/test_probs.npy"), dev_prob=ld(f"llm/{dn}/dev_probs.npy"))
    except FileNotFoundError as ex:
        print(f"{task} 缺少 {ex}，跳过")
        continue
    for a in encs:
        for o in ("LoRA_A", "LoRA_B"):
            S["pairs"].append((a, o, "LoRA"))
        for o in ("零样本Qwen3", "零样本Qwen2.5"):
            S["pairs"].append((a, o, o))
        other = a[:-2].replace("A", "#").replace("B", "A").replace("#", "B") + a[-2:]
        S["pairs"].append((a, other, "另一半数据的编码器"))
    ag = ld("agree/test_agree.npy", req=False)
    if ag is not None:
        AGREE[task] = ag
    pr = ld("meta/test_premise.npy", req=False)     # NLI：同一前提的几个假设按前提整体重抽样
    if pr is None and task == "nli" and ag is not None and not os.environ.get("A_SMOKE"):
        pr = rebuild_nli_premise(S["y_test"], ag)
    if pr is not None and len(pr) == len(S["y_test"]):
        S["cluster"] = np.unique(pr, return_inverse=True)[1]
    SETTINGS[task] = S

if not SETTINGS:
    raise SystemExit("一个设置都没找到。请检查 Add Input。")
for sn, S in SETTINGS.items():                      # 自检：基础编码器的准确率要像样
    a0 = S["pairs"][0][0]
    acc = float((S["judges"][a0]["tpred"] == S["y_test"]).mean())
    print(f"  自检 {sn}：{a0} 测试集准确率 {acc:.3f}（{S['K']} 类，测试 {len(S['y_test'])} 条，配对 {len(S['pairs'])} 个）")
    if acc < 0.4 and not os.environ.get("A_SMOKE"):
        raise SystemExit(f"{sn} 的编码器准确率只有 {acc:.3f}，标签或概率很可能对错了。请把这几行发给 Claude。")


# ============================================================
# 三层分析的函数
# ============================================================
def feats(p):
    """调用前能看到的量：A 的完整对数概率、最大概率、前两名之差、熵"""
    p = np.clip(p, 1e-7, 1)
    s = np.sort(p, 1)
    return np.column_stack([np.log(p), s[:, -1], s[:, -1] - s[:, -2], -(p * np.log(p)).sum(1)])


def fit_lr(X, y):
    """验证集上用 5 折交叉验证选 L2 强度，指标为 log loss；两个模型用完全相同的流程"""
    if len(np.unique(y)) < 2:
        return None
    try:
        return LogisticRegressionCV(Cs=CS, cv=5, scoring="neg_log_loss", max_iter=3000).fit(X, y)
    except ValueError:                               # 某类在某一折里太少时退回固定强度
        return LogisticRegression(C=1.0, max_iter=3000).fit(X, y)


def onehot(pred, K):
    M = np.zeros((len(pred), K + 1))
    M[np.arange(len(pred)), np.where(pred < 0, K, pred)] = 1     # 最后一列 = 缓存里无法解析的回答
    return M


def ll_items(model, X, y):
    P = np.full((len(y), int(max(y.max(), model.classes_.max())) + 1), 1e-7)
    P[:, model.classes_] = np.clip(model.predict_proba(X), 1e-7, 1)
    P /= P.sum(1, keepdims=True)
    return -np.log(P[np.arange(len(y)), y]), P.argmax(1), P


def layer2(S, a, b):
    """附加信息：M0 = 只用 A 的对数概率，M1 = 再加 B 投的类别（one-hot）。验证集拟合，测试集逐条 log loss 差（正 = B 有额外信息）"""
    K, yt = S["K"], S["y_test"]
    A, B = S["judges"][a], S["judges"][b]
    Xt0 = np.log(np.clip(A["tp"], 1e-7, 1)); Xt1 = np.column_stack([Xt0, onehot(B["tpred"], K)])
    if S["dev_mode"] == "dev":
        yd = S["y_dev"]
        Xd0 = np.log(np.clip(A["dp"], 1e-7, 1)); Xd1 = np.column_stack([Xd0, onehot(B["dpred"], K)])
        m0, m1 = fit_lr(Xd0, yd), fit_lr(Xd1, yd)
        l0, p0, P0 = ll_items(m0, Xt0, yt); l1, p1, P1 = ll_items(m1, Xt1, yt)
    else:                                            # 英文：大模型只有测试集输出，改用测试集两折交叉拟合
        fold = np.random.default_rng(17).permutation(len(yt)) % 2
        l0, l1, p0, p1 = (np.zeros(len(yt)) for _ in range(4))
        P0 = P1 = None                               # 英文没有 ChaosNLI，不需要保存概率
        for f in (0, 1):
            tr, tt = fold != f, fold == f
            m0, m1 = fit_lr(Xt0[tr], yt[tr]), fit_lr(Xt1[tr], yt[tr])
            l0[tt], p0[tt], _ = ll_items(m0, Xt0[tt], yt[tt]); l1[tt], p1[tt], _ = ll_items(m1, Xt1[tt], yt[tt])
    return {"dl": l0 - l1, "changed": (p0 != p1).astype(float), "dacc": ((p1 == yt).astype(float) - (p0 == yt).astype(float)),
            "P0": P0, "P1": P1}


def router_scores(S, a, b):
    """学出来的路由：只用调用前能看到的量，预测 D = 1[B 对] − 1[A 对]，得分 = P(D=+1) − P(D=−1)。验证集拟合，测试集锁定"""
    A, B = S["judges"][a], S["judges"][b]
    yt = S["y_test"]
    Ft = feats(A["tp"])
    Dt = (B["tpred"] == yt).astype(int) - (A["tpred"] == yt).astype(int)
    def score(m, F):
        pr = m.predict_proba(F); c = list(m.classes_)
        return (pr[:, c.index(1)] if 1 in c else 0) - (pr[:, c.index(-1)] if -1 in c else 0)
    if S["dev_mode"] == "dev":
        yd = S["y_dev"]
        Dd = (B["dpred"] == yd).astype(int) - (A["dpred"] == yd).astype(int)
        m = fit_lr(feats(A["dp"]), Dd)
        return score(m, Ft) if m is not None else np.zeros(len(yt))
    out = np.zeros(len(yt)); fold = np.random.default_rng(19).permutation(len(yt)) % 2
    for f in (0, 1):
        m = fit_lr(Ft[fold != f], Dt[fold != f])
        out[fold == f] = score(m, Ft[fold == f]) if m is not None else 0
    return out


def boot_index(rng, S, n):
    """自助法下标：有聚类（NLI 前提）时整簇重抽，否则逐题重抽"""
    if "cluster" not in S:
        return rng.integers(0, n, n)
    if "_members" not in S:
        cl = S["cluster"]; o = np.argsort(cl, kind="stable")
        S["_members"] = np.split(o, np.flatnonzero(np.diff(cl[o])) + 1)
    m = S["_members"]
    return np.concatenate([m[j] for j in rng.integers(0, len(m), len(m))])


def budget_lines(D, conf, rs, idx=None):
    """恰好调用 k 次、调用后用 B 覆盖 A。返回每个预算下：理想挑选、置信门控、学习路由的增益（相对 A），以及门控和路由的改对、改坏"""
    if idx is not None:
        D, conf, rs = D[idx], conf[idx], rs[idx]
    n = len(D); out = {}
    srt = np.sort(D)[::-1]
    og, orr = np.argsort(conf, kind="stable"), np.argsort(-rs, kind="stable")
    for q in QS + [0.0, 1.0]:
        k = int(np.floor(q * n))
        g, r = og[:k], orr[:k]
        out[q] = {"oracle": srt[:k].sum() / n,
                  "gate": D[g].sum() / n, "gate_fix": (D[g] == 1).sum() / n, "gate_harm": (D[g] == -1).sum() / n,
                  "router": D[r].sum() / n, "router_fix": (D[r] == 1).sum() / n, "router_harm": (D[r] == -1).sum() / n}
    return out


# ============================================================
# 主循环：先写统一逐题表，再算三层
# ============================================================
A1 = {"n_boot": N_BOOT_A, "budgets": QS, "settings": {}}
try:
    for sn, S in SETTINGS.items():
        t0 = time.time()
        yt = S["y_test"]; n = len(yt)
        # 统一逐题表：真值、各判断者的完整概率（有的话）和预测类别；门控信号 = 基础判断者的最大概率
        tab = {"y_test": yt}
        for nm, j in S["judges"].items():
            tab[f"pred|{nm}"] = j["tpred"]
            if j["tp"] is not None:
                tab[f"prob|{nm}"] = j["tp"]
        fname = {"中文情绪-通用": "zh_usual", "中文情绪-疫情（迁移）": "zh_virus_shift", "英文情绪": "en_emotion"}.get(sn, sn)
        np.savez_compressed(f"{OUT_DIR}/a1_items_{fname}.npz", **tab)
        rows = []
        for a, b, typ in S["pairs"]:
            A, B = S["judges"][a], S["judges"][b]
            D = (B["tpred"] == yt).astype(int) - (A["tpred"] == yt).astype(int)
            conf = A["tp"].max(1)
            rs = router_scores(S, a, b)
            bl = budget_lines(D, conf, rs)
            allB = float(D.mean())
            for line in ("oracle", "gate", "router"):                       # 自检：预算两端
                assert abs(bl[0.0][line]) < 1e-12, f"{sn} {a}/{b} k=0 时 {line} 不为 0"
                assert abs(bl[1.0][line] - allB) < 1e-9, f"{sn} {a}/{b} k=n 时 {line} 不等于 Acc_B − Acc_A"
            L2 = layer2(S, a, b)
            rows.append({"base": a, "second": b, "type": typ, "D": D, "conf": conf, "rs": rs, "L2": L2,
                         "acc_A": float((A["tpred"] == yt).mean()), "acc_B": float((B["tpred"] == yt).mean()),
                         "oracle_unbudgeted": float(((A["tpred"] == yt) | (B["tpred"] == yt)).mean())})
        # 自助法：所有配对共用同一组重抽样；按类型取平均
        types = sorted(set(r["type"] for r in rows), key=lambda t: [r["type"] for r in rows].index(t))
        rng = np.random.default_rng(23)
        keys = [(q, ln) for q in QS for ln in ("oracle", "gate", "gate_fix", "gate_harm", "router", "router_fix", "router_harm")]
        boot = {t: defaultdict(list) for t in types}
        point = {t: defaultdict(list) for t in types}
        for r in rows:
            bl = budget_lines(r["D"], r["conf"], r["rs"])
            for q, ln in keys:
                point[r["type"]][(q, ln)].append(bl[q][ln])
            for kx, v in (("dll", r["L2"]["dl"].mean()), ("changed", r["L2"]["changed"].mean()), ("dacc", r["L2"]["dacc"].mean()),
                          ("acc_A", r["acc_A"]), ("acc_B", r["acc_B"]), ("allB", r["D"].mean()), ("oracle_unbudgeted", r["oracle_unbudgeted"])):
                point[r["type"]][kx].append(v)
        for bi in range(N_BOOT_A):
            ii = boot_index(rng, S, n)
            acc_t = defaultdict(lambda: defaultdict(list))
            for r in rows:
                bl = budget_lines(r["D"], r["conf"], r["rs"], ii)
                for q, ln in keys:
                    acc_t[r["type"]][(q, ln)].append(bl[q][ln])
                acc_t[r["type"]]["dll"].append(r["L2"]["dl"][ii].mean())
            for t in types:
                for kx, v in acc_t[t].items():
                    boot[t][kx].append(float(np.mean(v)))
            if bi == 9:
                print(f"   {sn}：自助法预计 {(time.time() - t0) / 10 * N_BOOT_A / 60:.0f} 分钟")
        res = {"n_test": n, "K": S["K"], "dev_mode": S["dev_mode"], "types": {}}
        print(f"\n=== {sn}（{S['K']} 类，测试 {n} 条）===")
        for t in types:
            P_ = {k: float(np.mean(v)) for k, v in point[t].items()}
            B_ = boot[t]
            ci = lambda k: [float(x) for x in np.percentile(B_[k], [2.5, 97.5])]  # noqa: E731
            dif = lambda k1, k2: [float(x) for x in np.percentile(np.array(B_[k1]) - np.array(B_[k2]), [2.5, 97.5])]  # noqa: E731
            tr = {"n_pairs": len(point[t]["dll"]), "acc_A": P_["acc_A"], "acc_B": P_["acc_B"], "allB": P_["allB"],
                  "oracle_unbudgeted_minus_B": P_["oracle_unbudgeted"] - P_["acc_B"],
                  "logloss_gain": P_["dll"], "logloss_gain_ci": ci("dll"),
                  "fusion_changed_decisions": P_["changed"], "fusion_acc_change": P_["dacc"], "budgets": {}}
            for q in QS:
                g = {ln: P_[(q, ln)] for ln in ("oracle", "gate", "gate_fix", "gate_harm", "router", "router_fix", "router_harm")}
                g.update({f"{ln}_ci": ci((q, ln)) for ln in ("oracle", "gate", "router")})
                g["oracle_minus_gate_ci"] = dif((q, "oracle"), (q, "gate"))
                g["router_minus_gate_ci"] = dif((q, "router"), (q, "gate"))
                tr["budgets"][str(q)] = g
            res["types"][t] = tr
            print(f"  第二判断者 = {t}（{tr['n_pairs']} 对）：A {tr['acc_A']:.3f}  B {tr['acc_B']:.3f}  全用 B {tr['allB'] * 100:+.1f}  "
                  f"任一对比 B 高 {tr['oracle_unbudgeted_minus_B'] * 100:.1f} 个百分点")
            for q in QS:
                g = tr["budgets"][str(q)]
                print(f"     预算 {int(q * 100):2d}%：理想 {g['oracle'] * 100:+.2f}  置信门控 {g['gate'] * 100:+.2f}（改对 {g['gate_fix'] * 100:.2f} 改坏 {g['gate_harm'] * 100:.2f}）"
                      f"  学习路由 {g['router'] * 100:+.2f}（改对 {g['router_fix'] * 100:.2f} 改坏 {g['router_harm'] * 100:.2f}）  个百分点")
            print(f"     附加信息：留出 log loss 改善 {tr['logloss_gain']:+.4f} [{tr['logloss_gain_ci'][0]:+.4f}, {tr['logloss_gain_ci'][1]:+.4f}]；"
                  f"加入这一票后改变判定的比例 {tr['fusion_changed_decisions'] * 100:.1f}%，准确率变化 {tr['fusion_acc_change'] * 100:+.2f}")
        res["pairs"] = [{"base": r["base"], "second": r["second"], "type": r["type"], "acc_A": r["acc_A"], "acc_B": r["acc_B"],
                         "logloss_gain": float(r["L2"]["dl"].mean())} for r in rows]
        res["bootstrap_unit"] = "premise" if "cluster" in S else "item"
        A1["settings"][sn] = res
        S["_rows"] = rows
        with open(f"{OUT_DIR}/a1_results.json", "w") as f_:
            json.dump(A1, f_, indent=1, ensure_ascii=False)

    # ============================================================
    # E15 四格配对表：固定 A、样本和调用名单，比较普通 LoRA 与改损失的 LoRA
    # ============================================================
    A1["e15_four_cells"] = {}
    for sn in ("中文情绪-通用", "中文情绪-疫情（迁移）"):
        if sn not in SETTINGS:
            continue
        S = SETTINGS[sn]; yt = S["y_test"]; n = len(yt)
        variants = [v for v in ("LoRA（加权损失）", "LoRA（互补损失）") if v in S["judges"]]
        if "LoRA" not in S["judges"] or not variants:
            continue
        if len(variants) < 2:
            print(f"   提醒：{sn} 只找到 {variants}，另一个 15b 版本没有加进 Input")
        out = {}
        for new in variants:
            for q in QS + [1.0]:
                cells = defaultdict(lambda: [0, 0.0])
                tot = 0.0
                for s in (42, 43, 44):
                    a = f"编码器{s}"
                    if a not in S["judges"]:
                        continue
                    pa = S["judges"][a]["tpred"]; conf = S["judges"][a]["tp"].max(1)
                    call = np.argsort(conf, kind="stable")[: int(np.floor(q * n))]
                    bs, bn = S["judges"]["LoRA"]["tpred"][call], S["judges"][new]["tpred"][call]
                    ya, yy = pa[call], yt[call]
                    ds = (bs == yy).astype(int) - (ya == yy).astype(int)
                    dn = (bn == yy).astype(int) - (ya == yy).astype(int)
                    das, dan = bs != ya, bn != ya
                    for nm, m in (("始终一致", ~das & ~dan), ("新增分歧", ~das & dan), ("消失分歧", das & ~dan), ("始终分歧", das & dan)):
                        cells[nm][0] += int(m.sum()); cells[nm][1] += float((dn[m] - ds[m]).sum())
                    tot += float(dn.sum() - ds.sum())
                assert abs(sum(v[1] for v in cells.values()) - tot) < 1e-9, "四格相加没有还原准确率变化"
                out[f"{new}|{q}"] = {nm: {"items": v[0], "net_fix_change": v[1]} for nm, v in cells.items()}
                out[f"{new}|{q}"]["total_net_fix_change"] = tot
        A1["e15_four_cells"][sn] = out
        print(f"\n=== E15 四格配对表（{sn}；三个编码器种子合计；净改对数变化 = 改损失版 − 普通版）===")
        for key, v in out.items():
            print(f"   {key:22s} " + "  ".join(f"{nm} {v[nm]['items']}条/{v[nm]['net_fix_change']:+.0f}" for nm in ("始终一致", "新增分歧", "消失分歧", "始终分歧"))
                  + f"  合计 {v['total_net_fix_change']:+.0f}")

    # ============================================================
    # 稳健性：ChaosNLI（一半票定参考标签，另一半定共识，交换角色，重复 20 次）
    # ============================================================
    if "nli" in SETTINGS and "nli" in AGREE:
        S = SETTINGS["nli"]; f_ = AGREE["nli"]; has = ~np.isnan(f_)
        e_all = np.round(100 * f_[has]).astype(int)
        rr = np.random.default_rng(29)
        acc = defaultdict(list)
        for rep in range(20):
            e1 = rr.hypergeometric(e_all, 100 - e_all, 50); e2 = e_all - e1
            for ref_e, con_e in ((e1, e2), (e2, e1)):
                keep = ref_e != 25
                ref = (ref_e > 25).astype(int)
                cons = np.where(ref == 1, con_e, 50 - con_e) / 50   # 另一半里同意参考标签的比例
                for r in S["_rows"]:
                    if r["type"] not in ("LoRA", "零样本Qwen3"):
                        continue
                    A_, B_ = S["judges"][r["base"]], S["judges"][r["second"]]
                    pa, pb = A_["tpred"][has], B_["tpred"][has]
                    for subset, m in (("全部 ChaosNLI", keep), ("高共识（另一半 ≥ 0.8）", keep & (cons >= 0.8))):
                        D = (pb[m] == ref[m]).astype(int) - (pa[m] == ref[m]).astype(int)
                        bl = budget_lines(D, A_["tp"][has][m].max(1), np.zeros(m.sum()))
                        acc[(r["type"], subset, "gate20")].append(bl[0.2]["gate"])
                        acc[(r["type"], subset, "oracle20")].append(bl[0.2]["oracle"])
                        # 附加信息也按分半参考标签重算（融合模型仍是在 MNLI 验证集上拟合的那两个）
                        P0h, P1h = r["L2"]["P0"][has][m], r["L2"]["P1"][has][m]
                        ii_ = np.arange(m.sum()); rm = ref[m]
                        acc[(r["type"], subset, "logloss_gain")].append(float(np.mean(np.log(P1h[ii_, rm]) - np.log(P0h[ii_, rm]))))
                        acc[(r["type"], subset, "n")].append(int(m.sum()))
        A1["chaosnli_split_half"] = {"|".join(k): float(np.mean(v)) for k, v in acc.items()}
        print(f"\n=== 稳健性：ChaosNLI（{has.sum()} 句，参考标签来自一半票）===")
        for t in ("LoRA", "零样本Qwen3"):
            for subset in ("全部 ChaosNLI", "高共识（另一半 ≥ 0.8）"):
                g = lambda x: A1["chaosnli_split_half"].get(f"{t}|{subset}|{x}", float("nan"))  # noqa: E731
                print(f"   {t:10s} {subset:16s}（约 {g('n'):.0f} 句）：预算 20% 理想 {g('oracle20') * 100:+.2f}  置信门控 {g('gate20') * 100:+.2f}  log loss 改善 {g('logloss_gain'):+.4f}")
        print("   （log loss 改善沿用 MNLI 金标拟合与评分，只按子集重算平均，供参照）")
except Exception:
    print("\nA1 主分析出错，请把下面的报错发给 Claude：")
    traceback.print_exc()


# ============================================================
# 次要问题：第二票的价值能否在部署前估计（只在二分类检测上、只比恒零与直接测量）
#   验证集分两半：一半选 k，另一半分别用"直接测量"和"本文的双正态模型"预测同一个 k 在测试集上的 AUC 提高量
# ============================================================
def det_score(p, neu):
    return np.log(np.clip(1 - p[:, neu], 1e-7, 1) / np.clip(p[:, neu], 1e-7, 1)) if p.shape[1] > 2 else \
        np.log(np.clip(p[:, 1], 1e-7, 1) / np.clip(p[:, 0], 1e-7, 1))


def make_g(x_neg):
    xs, cnt = np.unique(np.sort(x_neg), return_counts=True)
    m = cnt.sum(); cum = (np.cumsum(cnt) - 0.5 * cnt) / m
    zq = np.array([Z(float(min(max(c, 0.5 / m), 1 - 0.5 / m))) for c in cum])
    def sd(lo, hi):
        a_, b_ = np.quantile(x_neg, [lo, hi]); return max((b_ - a_) / (Z(hi) - Z(lo)), 1e-6)
    s_hi, s_lo = sd(0.90, 0.99), sd(0.01, 0.10)
    def g(x):
        x = np.asarray(x, float); o = np.interp(x, xs, zq)
        o = np.where(x > xs[-1], zq[-1] + (x - xs[-1]) / s_hi, o)
        return np.where(x < xs[0], zq[0] + (x - xs[0]) / s_lo, o)
    return g


SIM_U = np.random.default_rng(2026).standard_normal((4, N_SIM))


def auc1(s, y):
    return auc_of(s, y, np.ones(len(y)))


def sim_rule_gain(gx, ra, b, y, k, independent=False):
    sl, ic = zroc_fit(*roc_curve(gx, y, np.ones(len(y))))
    sl = min(max(sl, 0.2), 5.0); sig, muA = 1.0 / sl, ic / sl
    Hb, Fb, _, _ = sdt(b, y, np.ones(len(y)))
    t = -Z(Fb); muB = t + sig * Z(Hb)
    r1, r0 = (0.0, 0.0) if independent else dvc(ra, b, y, np.ones(len(y)))
    r1, r0 = min(max(r1, -0.99), 0.99), min(max(r0, -0.99), 0.99)
    u1, e1, u0, e0 = SIM_U
    xs_ = np.concatenate([muA + sig * u1, u0])
    bs_ = np.concatenate([muB + sig * (r1 * u1 + math.sqrt(1 - r1 * r1) * e1) > t, r0 * u0 + math.sqrt(1 - r0 * r0) * e0 > t]).astype(float)
    ys_ = np.concatenate([np.ones(N_SIM, int), np.zeros(N_SIM, int)])
    return auc1(xs_ + k * bs_, ys_) - auc1(xs_, ys_)


try:
    print("\n" + "=" * 100 + "\n次要问题：能否用验证集事前估计第二票的价值（二分类检测，锁定同一条规则）\n" + "=" * 100)
    SEC = {}
    GROUP = {"中文情绪-通用": "域内", "中文情绪-疫情（迁移）": "迁移", "nli": "域内", "toxicity": "域内", "cold": "域内"}
    BOOT = defaultdict(list)
    for sn, S in SETTINGS.items():
        if S["dev_mode"] != "dev":
            continue
        neu, yd, yt = S["neutral"], S["y_dev"], S["y_test"]
        td, tt = (yd != neu).astype(int), (yt != neu).astype(int)
        half = np.random.default_rng(7).permutation(len(yd)) % 2
        rows = []
        for a, b, typ in S["pairs"]:
            A, B = S["judges"][a], S["judges"][b]
            xd, xt = det_score(A["dp"], neu), det_score(A["tp"], neu)
            bd, bt = (B["dpred"] != neu).astype(int), (B["tpred"] != neu).astype(int)
            rad = (xd > 0).astype(int)   # A 的二分类决定：后验 > 0.5
            hs = []
            for sel in (0, 1):
                S_, T_ = half == sel, half != sel
                g = make_g(xd[S_ & (td == 0)]); gd, gt = g(xd), g(xt)   # 变换只用选权重的那一半拟合
                k = K_GRID[int(np.argmax([auc1(gd[S_] + kk * bd[S_], td[S_]) for kk in K_GRID]))]
                hs.append({"k": k, "直接测量": auc1(gd[T_] + k * bd[T_], td[T_]) - auc1(gd[T_], td[T_]),
                           "本文": sim_rule_gain(gd[T_], rad[T_], bd[T_], td[T_], k),
                           "若独立": sim_rule_gain(gd[T_], rad[T_], bd[T_], td[T_], k, independent=True), "恒为0": 0.0,
                           "obs": auc1(gt + k * bt, tt) - auc1(gt, tt), "gt": gt})
            rows.append({"type": typ, "halves": hs, "bt": bt})
        meths = ["本文", "直接测量", "若独立", "恒为0"]
        mae = {m: float(np.mean([np.mean([abs(h[m] - h["obs"]) for h in r["halves"]]) for r in rows])) for m in meths}
        rng = np.random.default_rng(31); bm = {m: [] for m in meths}
        for bi in range(N_BOOT_A):
            ii = boot_index(rng, S, len(tt)); e = {m: [] for m in meths}
            for r in rows:
                for h in r["halves"]:
                    o = auc1(h["gt"][ii] + h["k"] * r["bt"][ii], tt[ii]) - auc1(h["gt"][ii], tt[ii])
                    for m in meths:
                        e[m].append(abs(h[m] - o))
            for m in meths:
                bm[m].append(float(np.mean(e[m])))
        for m in meths:
            BOOT[(GROUP.get(sn, "其他"), m, sn)] = bm[m]
        SEC[sn] = {"MAE": mae, "MAE_ci": {m: [float(x) for x in np.percentile(bm[m], [2.5, 97.5])] for m in meths}, "n_pairs": len(rows)}
        print(f"   {sn}（{len(rows)} 对）：平均绝对误差  " + "  ".join(f"{m} {mae[m]:.4f}" for m in meths))
    PRIMARY = {"H1 域内：本文 − 直接测量": ("域内", "本文", "直接测量"), "H2 迁移：本文 − 直接测量": ("迁移", "本文", "直接测量"),
               "H3 全部：本文 − 恒为0": (None, "本文", "恒为0")}
    pv, prim = {}, {}
    for hn, (grp, m1, m2) in PRIMARY.items():
        sets = [k for k in BOOT if k[1] == m1 and (grp is None or k[0] == grp)]
        if not sets:
            continue
        d = np.mean([np.array(BOOT[s]) for s in sets], 0) - np.mean([np.array(BOOT[(s[0], m2, s[2])]) for s in sets], 0)
        pv[hn] = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
        prim[hn] = {"diff": float(d.mean()), "ci95": [float(x) for x in np.percentile(d, [2.5, 97.5])], "p_boot": pv[hn]}
    order = sorted(pv, key=pv.get)
    for i, hn in enumerate(order):
        prim[hn]["p_holm"] = float(min(1.0, max(pv[h] * (len(order) - j) for j, h in enumerate(order[: i + 1]))))
    print("   预先指定的主比较（平均绝对误差之差，负数 = 本文更准；Holm 校正）：")
    for hn, v in prim.items():
        print(f"      {hn}：{v['diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]  Holm p = {v['p_holm']:.3f}")
    A1["secondary_prediction"] = {"settings": SEC, "primary": prim}
except Exception:
    print("\n次要问题部分出错（主分析结果不受影响），请把报错发给 Claude：")
    traceback.print_exc()

with open(f"{OUT_DIR}/a1_results.json", "w") as f_:
    json.dump(A1, f_, indent=1, ensure_ascii=False)
print(f"\n全部完成（{(time.time() - T0) / 60:.0f} 分钟）。请把 a1_results.json 发给 Claude。")
