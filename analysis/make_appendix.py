"""Appendix numbers: label-token mass of the binary-task LLM judges and the label-flip simulation.

Inputs : inputs/task_{nli,toxicity,cold}_metrics.json (G1 outputs), inputs/c2_zh_zeroshot_metrics.json (experiment 11)
Outputs: numbers_appendix.tex (\\app{key}), tab_labelflip.tex
The simulation reuses the tetrachoric estimator of a1_three_layer_kaggle.py.
Usage  : python make_appendix.py
"""
import json
import math
from statistics import NormalDist

import numpy as np

Z, PHI = NormalDist().inv_cdf, NormalDist().cdf
GL_X, GL_W = np.polynomial.legendre.leggauss(64)


def clipz(p):
    return Z(min(max(float(p), 1e-6), 1 - 1e-6))


def bvn_upper(h, k, rho):
    r = 0.5 * rho * (GL_X + 1.0)
    dens = np.exp(-(h * h - 2 * r * h * k + k * k) / (2 * (1 - r * r))) / (2 * np.pi * np.sqrt(1 - r * r))
    return PHI(-h) * PHI(-k) + 0.5 * rho * float(np.sum(GL_W * dens))


def tetrachoric(n11, n10, n01, n00):
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


def within_class_rho(ra, rb, y):
    out = []
    for c in (1, 0):
        m = y == c
        a, b = ra[m] == 1, rb[m] == 1
        out.append(tetrachoric((a & b).sum(), (a & ~b).sum(), (~a & b).sum(), (~a & ~b).sum()))
    return float(np.mean(out))


A = {}

# ---------- label-token mass (mean probability on the label tokens, test split) ----------
masses = []
for t in ("nli", "toxicity", "cold"):
    M = json.load(open(f"inputs/task_{t}_metrics.json", encoding="utf-8"))
    for name, v in M["llm"].items():
        m = v.get("answer_token_mass", {}).get("test")
        if m is not None:
            masses.append(m)
A["mass-min"] = f"{100 * min(masses):.2f}"
A["mass-n"] = str(len(masses))

# ---------- free-generation check, Chinese emotion zero-shot judge (prompt v1, used in the paper) ----------
# inputs/c2_zh_zeroshot_metrics.json holds the fields of the experiment-11 metrics file that are used here.
C2 = json.load(open("inputs/c2_zh_zeroshot_metrics.json", encoding="utf-8"))["runs"]["Qwen3-4B-Instruct-2507|v1"]
fg = C2["free_generation_check"]
n_parsed = round(fg["n"] * fg["parsed_rate"])
A["fg-n"] = str(fg["n"])
A["fg-parsed"] = str(n_parsed)
A["fg-agree"] = str(round(n_parsed * fg["agreement_among_parsed"]))
A["fg-mass-min"] = f"{100 * min(C2[s]['label_token_mass_mean'] for s in ('usual_test', 'virus_test')):.1f}"

# ---------- ChaosNLI: why the fine-tuned judge adds negative information against crowd labels ----------
# Refits f0 and f1 exactly as a1_three_layer_kaggle.py does (validation split, LogisticRegressionCV) for the 8 fine-tuned
# pairs and the 4 zero-shot Qwen3-4B pairs, then looks at the ChaosNLI items whose 100 ratings have a majority below 80%.
import warnings
from sklearn.linear_model import LogisticRegressionCV
warnings.filterwarnings("ignore")
NL = lambda p: np.load(f"inputs/nli/{p}")   # noqa: E731
yd, yt = NL("labels/dev_labels.npy").astype(int), NL("labels/test_labels.npy").astype(int)
fr = NL("agree/test_agree.npy"); has = ~np.isnan(fr)
fr, gold = fr[has], yt[has]
maj, tie = (fr > 0.5).astype(int), fr == 0.5
low = (np.abs(fr - 0.5) < 0.3) & ~tie          # majority of the 100 ratings below 80%


def lrcv(X, y):
    return LogisticRegressionCV(Cs=[0.01, 0.1, 1, 10], cv=5, scoring="neg_log_loss", max_iter=3000).fit(X, y)


def oh(p):
    M = np.zeros((len(p), 3)); M[np.arange(len(p)), p] = 1   # third column: unparseable answer (never used here)
    return M


diag = {}
for jkey, judges in (("lora", ("lora_half0", "lora_half1")), ("zs", ("zs_qwen3_4b",))):
    w, dl_low, dl_all, ag_maj, ag_gold = [], [], [], [], []
    for j in judges:
        bd, bt = NL(f"llm/{j}/dev_probs.npy").argmax(1), NL(f"llm/{j}/test_probs.npy").argmax(1)
        ag_maj.append(np.mean(bt[has][low] == maj[low])); ag_gold.append(np.mean(bt[has][low] == gold[low]))
        for h in (0, 1):
            for sd in (42, 43):
                Xd = np.log(np.clip(NL(f"enc/half{h}/seed{sd}/dev_probs.npy"), 1e-7, 1))
                Xt = np.log(np.clip(NL(f"enc/half{h}/seed{sd}/test_probs.npy"), 1e-7, 1))
                m0, m1 = lrcv(Xd, yd), lrcv(np.column_stack([Xd, oh(bd)]), yd)
                P0 = np.clip(m0.predict_proba(Xt)[has], 1e-7, 1)
                P1 = np.clip(m1.predict_proba(np.column_stack([Xt, oh(bt)]))[has], 1e-7, 1)
                ii = np.arange(len(maj))
                d = np.log(P1[ii, maj]) - np.log(P0[ii, maj])
                dl_low.append(d[low].mean()); dl_all.append(d[~tie].mean())
                w.append(m1.coef_[0][3] - m1.coef_[0][2])   # change in log-odds when the vote flips
    diag[jkey] = dict(w=np.mean(w), dl_low=np.mean(dl_low), dl_all=np.mean(dl_all), ag_maj=np.mean(ag_maj), ag_gold=np.mean(ag_gold))
A["cn-low-n"] = f"{int(low.sum()):,}".replace(",", "{,}")
A["cn-n"] = f"{int(has.sum()):,}".replace(",", "{,}")
for jk, v in diag.items():
    A[f"cn-{jk}-w"] = f"{v['w']:.1f}"
    A[f"cn-{jk}-dllow"] = f"{v['dl_low']:.2f}".replace("-", "$-$")
    A[f"cn-{jk}-dllall"] = f"{v['dl_all']:.3f}".replace("-", "$-$")
    A[f"cn-{jk}-agmaj"] = f"{100 * v['ag_maj']:.0f}"
    A[f"cn-{jk}-aggold"] = f"{100 * v['ag_gold']:.0f}"
print("ChaosNLI diagnosis:", {k: {a: round(float(b), 3) for a, b in v.items()} for k, v in diag.items()})

# ---------- binary tasks: fine-tuned LLM trained on the same half as A versus the other half ----------
# Gated gain at 20% from the item tables of the main analysis (inputs/a1_items_{task}.npz).
hd = []
for t in ("nli", "toxicity", "cold"):
    d = np.load(f"inputs/a1_items_{t}.npz")
    y = d["y_test"]; n = len(y); k = int(np.floor(0.2 * n))
    same, other = [], []
    for h in "AB":
        for sd in ("42", "43"):
            pa = d[f"prob|编码器{h}{sd}"]; a = pa.argmax(1); call = np.argsort(pa.max(1), kind="stable")[:k]
            for L in "AB":
                b = d[f"pred|LoRA_{L}"]
                (same if L == h else other).append(((b[call] == y[call]).sum() - (a[call] == y[call]).sum()) / n)
    hd.append(abs(np.mean(same) - np.mean(other)))
A["halfdiff-max"] = f"{100 * max(hd):.2f}"

# ---------- label-flip simulation ----------
# Two equal-variance judges with true within-class correlation rho; 80% positive items;
# each judge thresholds its decision variable at d'/2; a share eta of reference labels is flipped.
rng = np.random.default_rng(3)
N = 200_000
ETAS = (0.0, 0.02, 0.05, 0.10, 0.15)
CASES = ((2.75, 2.84, 0.0), (2.75, 2.84, 0.5), (2.7, 2.0, 0.0), (1.5, 1.5, 0.0))
rows = []
for dA, dB, rho in CASES:
    y = (rng.random(N) < 0.8).astype(int)
    u, e = rng.standard_normal(N), rng.standard_normal(N)
    xa = u + dA * y
    xb = rho * u + math.sqrt(1 - rho ** 2) * e + dB * y
    ra, rb = (xa > dA / 2).astype(int), (xb > dB / 2).astype(int)
    vals = []
    for eta in ETAS:
        yo = y.copy()
        fl = rng.random(N) < eta
        yo[fl] = 1 - yo[fl]
        vals.append(within_class_rho(ra, rb, yo))
    assert abs(vals[0] - rho) < 0.03, (dA, dB, rho, vals[0])   # no flips: the estimator recovers the true value
    rows.append((dA, dB, rho, vals))
    key = f"d{dA}-{dB}-r{rho}".replace(".", "")
    for eta, v in zip(ETAS, vals):
        A[f"flip-{key}-e{int(round(100 * eta))}"] = f"{v:.2f}"


def fmt(v):
    return f"{v:.2f}".replace("-", "$-$")


with open("tab_labelflip.tex", "w", encoding="utf-8") as f:
    f.write("""%% Generated by make_appendix.py
\\begin{table}[t]
\\centering
\\caption{Apparent within-class correlation of two judges when a share $\\eta$ of reference labels is flipped. Decision variables are equal-variance normal with true correlation $\\rho$. Of the items, 80\\%% are positive, as in Chinese emotion detection, and each judge thresholds at $d'/2$. The first two rows use the discriminability of the Chinese encoder and fine-tuned LLM on that task. Each row uses %s simulated items.}
\\label{tab:labelflip}
\\small
\\begin{tabular}{@{}rrr%s@{}}
\\toprule
$d'_A$ & $d'_B$ & $\\rho$ & %s \\\\
\\midrule
""" % (f"{N:,}".replace(",", "{,}"), "r" * len(ETAS), " & ".join(f"$\\eta={int(round(100 * e))}\\%$" for e in ETAS)))
    for dA, dB, rho, vals in rows:
        f.write(f"{dA:.2f} & {dB:.2f} & {rho:.1f} & " + " & ".join(fmt(v) for v in vals) + " \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

with open("numbers_appendix.tex", "w", encoding="utf-8") as f:
    f.write("% Generated by make_appendix.py. Do not edit by hand.\n\\makeatletter\n")
    f.write("\\newcommand{\\app}[1]{\\ifcsname app@#1\\endcsname\\csname app@#1\\endcsname\\else\\textbf{\\textcolor{red}{??#1}}\\fi}\n")
    for k, v in sorted(A.items()):
        f.write(f"\\expandafter\\def\\csname app@{k}\\endcsname{{{v}}}\n")
    f.write("\\makeatother\n")
print(A)
