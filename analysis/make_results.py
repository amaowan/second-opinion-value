"""Generate every result number, table and figure of Section 5 from a1_results.json.

Outputs (in this directory):
  numbers_results.tex   \\res{key} values used in the text (rule R1-8: no hand-typed results)
  tab_fourcell.tex      Table: four-cell decomposition of the retrained judge
  tab_secondary.tex     Table: estimating the detection gain in advance
  tab_allpairs.tex      Appendix table: all three layers for every pair type
  fig_overview.pdf      Figure 2: oracle, gate and router at q = 20% for every pair type
  fig_layers.pdf        Figure 3: realised gain against each layer
Usage: python make_results.py a1_results.json
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr

src = sys.argv[1] if len(sys.argv) > 1 else "a1_results.json"
D = json.load(open(src, encoding="utf-8"))
S = D["settings"]

SETTINGS = [("中文情绪-通用", "zhgen", "Chinese emotion"),
            ("中文情绪-疫情（迁移）", "zhshift", "Chinese emotion, shift"),
            ("英文情绪", "en", "English emotion"),
            ("nli", "nli", "NLI"),
            ("toxicity", "tox", "Toxicity"),
            ("cold", "cold", "COLD")]
# (raw type name, key, label, kind)  kind: ft = fine-tuned LLM, zs = zero-shot LLM, twin = encoder twin, epi = epidemic encoder
TYPES = {
    "LoRA": ("lora", "Fine-tuned Qwen3-4B", "ft"),
    "LoRA（加权损失）": ("loraw", "Fine-tuned Qwen3-4B, weighted", "ft"),
    "LoRA（Qwen3.5-4B）": ("lora", "Fine-tuned Qwen3.5-4B", "ft"),
    "零样本Qwen3": ("zsqwen3", "Zero-shot Qwen3-4B", "zs"),
    "零样本Qwen2.5": ("zsqwen25", "Zero-shot Qwen2.5-3B", "zs"),
    "零样本Qwen3.5-4B": ("zsqwen35", "Zero-shot Qwen3.5-4B", "zs"),
    "零样本Qwen3.7-Max": ("zsmax", "Zero-shot Qwen3.7-Max", "zs"),
    "零样本Gemma3-4B": ("zsgemma", "Zero-shot Gemma3-4B", "zs"),
    "另一种子编码器": ("twin", "Encoder, other seed", "twin"),
    "另一半数据的编码器": ("twin", "Encoder, other half", "twin"),
    "疫情训练的编码器": ("epi", "Encoder, epidemic posts", "epi"),
}
ORDER = ["LoRA", "LoRA（Qwen3.5-4B）", "LoRA（加权损失）", "零样本Qwen3", "零样本Qwen2.5", "零样本Qwen3.5-4B",
         "零样本Qwen3.7-Max", "零样本Gemma3-4B", "疫情训练的编码器", "另一种子编码器", "另一半数据的编码器"]

R = {}  # key -> formatted string


def pts(x, nd=1):
    """accuracy share -> accuracy points, with a typeset minus sign"""
    v = round(100 * x, nd)
    if v == 0:
        v = 0.0
    s = f"{abs(v):.{nd}f}"
    return ("$-$" if v < 0 else "") + s


def num(x, nd=3):
    v = round(x, nd)
    if v == 0:
        v = 0.0
    return ("$-$" if v < 0 else "") + f"{abs(v):.{nd}f}"


# ---------- rows: one per (setting, pair type) ----------
rows = []
for sraw, skey, slab in SETTINGS:
    T = S[sraw]["types"]
    for traw in ORDER:
        if traw not in T:
            continue
        v = T[traw]
        tkey, tlab, kind = TYPES[traw]
        b = v["budgets"]
        r = dict(setting=skey, slab=slab, type=tkey, tlab=tlab, kind=kind,
                 bma=v["acc_B"] - v["acc_A"], dll=v["logloss_gain"], dlo=v["logloss_gain_ci"][0], dhi=v["logloss_gain_ci"][1],
                 fus=v["fusion_acc_change"], chg=v["fusion_changed_decisions"], npairs=v["n_pairs"])
        for q, qs in (("0.1", "10"), ("0.2", "20"), ("0.4", "40")):
            for m in ("oracle", "gate", "router", "gate_fix", "gate_harm", "router_fix", "router_harm"):
                r[f"{m}{qs}"] = b[q][m]
            for m in ("oracle_ci", "gate_ci", "router_ci", "oracle_minus_gate_ci", "router_minus_gate_ci"):
                r[f"{m}{qs}"] = b[q][m]
        rows.append(r)
        k = f"{skey}-{tkey}"
        R[f"{k}-bma"] = pts(r["bma"])
        R[f"{k}-bmaabs"] = pts(abs(r["bma"]))
        R[f"{k}-dll"] = num(r["dll"])
        R[f"{k}-dllci"] = f"[{num(r['dlo'])}, {num(r['dhi'])}]"
        R[f"{k}-fus"] = pts(r["fus"])
        R[f"{k}-chg"] = pts(r["chg"])
        for qs in ("10", "20", "40"):
            for m in ("oracle", "gate", "router", "gate_fix", "gate_harm", "router_fix", "router_harm"):
                R[f"{k}-{m.replace('_', '')}{qs}"] = pts(r[f"{m}{qs}"])
            lo, hi = r[f"router_minus_gate_ci{qs}"]
            R[f"{k}-rmgci{qs}"] = f"[{pts(lo, 2)}, {pts(hi, 2)}]"
            R[f"{k}-rmg{qs}"] = pts(r[f"router{qs}"] - r[f"gate{qs}"], 2)
            R[f"{k}-rmgabs{qs}"] = pts(abs(r[f"router{qs}"] - r[f"gate{qs}"]), 2)
            lo, hi = r[f"gate_ci{qs}"]
            R[f"{k}-gateci{qs}"] = f"[{pts(lo)}, {pts(hi)}]"
        R[f"{skey}-acca"] = pts(v["acc_A"])
        R[f"{k}-share20"] = f"{r['gate20'] / r['oracle20']:.2f}" if r["oracle20"] > 0 else "--"

# ---------- summary numbers used in 5.1 ----------
useful = [r for r in rows if r["kind"] in ("ft", "epi") and r["gate20"] > 0 and r["bma"] > 0]
shares = [r["gate20"] / r["oracle20"] for r in useful]
R["share-max"] = f"{max(shares):.2f}"
R["share-min-useful"] = f"{min(shares):.2f}"
R["share-max-all20"] = f"{max(r['gate20'] / r['oracle20'] for r in rows if r['oracle20'] > 0):.2f}"
R["share-max-all40"] = f"{max(r['gate40'] / r['oracle40'] for r in rows if r['oracle40'] > 0):.2f}"
assert R["share-max-all20"] == R["share-max"], "the largest share at 20% must belong to a useful pair"
# the oracle never falls when the budget grows, so no budget forces a call on a harmful item
assert all(r["oracle40"] >= r["oracle20"] - 1e-12 and r["oracle20"] >= r["oracle10"] - 1e-12 for r in rows)
R["n-types"] = str(len(rows))
twins = {r["setting"]: r for r in rows if r["kind"] == "twin"}
tw_or = [r["oracle20"] for r in twins.values()]
R["twin-oracle-min"], R["twin-oracle-max"] = pts(min(tw_or)), pts(max(tw_or))
tw_g = [r["gate20"] for r in twins.values()]
R["twin-gate-min"], R["twin-gate-max"] = pts(min(tw_g)), pts(max(tw_g))
tw_d = [r["dll"] for r in twins.values()]
R["twin-dll-min"], R["twin-dll-max"] = num(min(tw_d)), num(max(tw_d))
ratios = []
for s_, t in twins.items():
    ft = [r for r in rows if r["setting"] == s_ and r["type"] == "lora"]
    if ft:
        ratios.append(t["oracle20"] / ft[0]["oracle20"])
R["twin-ratio-min"], R["twin-ratio-max"] = f"{100 * min(ratios):.0f}", f"{100 * max(ratios):.0f}"
allor = [r["oracle20"] for r in rows]
R["oracle-min"], R["oracle-max"] = pts(min(allor)), pts(max(allor))

# ---------- 5.2 association across pair types (descriptive) ----------
y = np.array([r["gate20"] for r in rows])
yr = np.array([r["router20"] for r in rows])
X = {"oracle": np.array([r["oracle20"] for r in rows]), "bma": np.array([r["bma"] for r in rows]),
     "dll": np.array([r["dll"] for r in rows]), "fus": np.array([r["fus"] for r in rows])}


def resid(a, b):
    A = np.column_stack([np.ones(len(b)), b])
    return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]


for nm, x in X.items():
    R[f"rho-{nm}-gate"] = num(spearmanr(x, y)[0], 2)
    R[f"rho-{nm}-router"] = num(spearmanr(x, yr)[0], 2)
for nm in ("oracle", "dll"):
    rp = np.corrcoef(resid(rankdata(X[nm]), rankdata(X["bma"])), resid(rankdata(y), rankdata(X["bma"])))[0, 1]
    R[f"prho-{nm}-gate"] = num(rp, 2)

# largest router advantage over the gate, for second judges more accurate than A and for the rest
adv = lambda rs: max(r[f"router{q}"] - r[f"gate{q}"] for r in rs for q in ("10", "20", "40"))  # noqa: E731
R["radv-better"] = pts(adv([r for r in rows if r["bma"] > 0]), 2)
R["radv-rest"] = pts(adv([r for r in rows if r["bma"] <= 0]), 1)
# second judges at most COMP points less accurate than A (descriptive check that weak judges do not drive the ranks)
COMP = 0.05
cm = X["bma"] >= -COMP
R["rhoc-n"] = str(int(cm.sum()))
R["rhoc-nweak"] = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve"][int((~cm).sum())]
R["rhoc-thr"] = f"{100 * COMP:.0f}"
for nm, x in X.items():
    R[f"rhoc-{nm}-gate"] = num(spearmanr(x[cm], y[cm])[0], 2)

# ---------- 5.3 four cells ----------
FC = D["e15_four_cells"]
CELLS = [("始终一致", "agree"), ("新增分歧", "new"), ("消失分歧", "gone"), ("始终分歧", "both")]
fc_rows = []
for sraw, skey, slab in SETTINGS[:2]:
    n = S[sraw]["n_test"]
    for q, ql in (("0.1", "10\\%"), ("0.2", "20\\%"), ("0.4", "40\\%"), ("1.0", "100\\%")):
        c = FC[sraw][f"LoRA（加权损失）|{q}"]
        # identity check: cells add up to the change in gated gain (or in B accuracy for q = 1)
        T = S[sraw]["types"]
        if q == "1.0":
            g = T["LoRA（加权损失）"]["allB"] - T["LoRA"]["allB"]
        else:
            g = T["LoRA（加权损失）"]["budgets"][q]["gate"] - T["LoRA"]["budgets"][q]["gate"]
        assert abs(c["total_net_fix_change"] / (3 * n) - g) < 1e-9, (sraw, q)
        fc_rows.append((slab, ql, [(c[a]["items"], c[a]["net_fix_change"]) for a, _ in CELLS], c["total_net_fix_change"]))
        qk = q.replace("0.", "").replace("1.0", "100") if q != "1.0" else "100"
        qk = {"0.1": "10", "0.2": "20", "0.4": "40", "1.0": "100"}[q]
        for a, ak in CELLS:
            R[f"fc-{skey}-{qk}-{ak}-n"] = f"{c[a]['items']:,}".replace(",", "{,}")
            R[f"fc-{skey}-{qk}-{ak}-d"] = num(c[a]["net_fix_change"], 0)
        R[f"fc-{skey}-{qk}-tot"] = num(c["total_net_fix_change"], 0)
c = FC["中文情绪-通用"]["LoRA（加权损失）|1.0"]
R["fc-zhgen-100-new-perit"] = num(c["新增分歧"]["net_fix_change"] / c["新增分歧"]["items"], 2)
R["fc-zhgen-100-newshare"] = f"{100 * c['新增分歧']['net_fix_change'] / c['total_net_fix_change']:.0f}"

# ---------- 5.4 ChaosNLI ----------
C = D["chaosnli_split_half"]
for b_, bk in (("LoRA", "lora"), ("零样本Qwen3", "zsqwen3")):
    for sub, sk in (("全部 ChaosNLI", "all"), ("高共识（另一半 ≥ 0.8）", "high")):
        R[f"chaos-{bk}-{sk}-gate"] = pts(C[f"{b_}|{sub}|gate20"])
        R[f"chaos-{bk}-{sk}-oracle"] = pts(C[f"{b_}|{sub}|oracle20"])
        R[f"chaos-{bk}-{sk}-dll"] = num(C[f"{b_}|{sub}|logloss_gain"])
        R[f"chaos-{sk}-n"] = f"{C[f'{b_}|{sub}|n']:,.0f}".replace(",", "{,}")

# ---------- 5.5 secondary ----------
P = D["secondary_prediction"]
SEC_SET = [("中文情绪-通用", "Chinese emotion"), ("中文情绪-疫情（迁移）", "Chinese emotion, shift"),
           ("nli", "NLI"), ("toxicity", "Toxicity"), ("cold", "COLD")]
EST = [("本文", "model"), ("直接测量", "direct"), ("若独立", "indep"), ("恒为0", "zero")]
for sraw, _ in SEC_SET:
    sk = dict((a, b) for a, b, _c in SETTINGS)[sraw]
    for e, ek in EST:
        R[f"sec-{sk}-{ek}"] = f"{1000 * P['settings'][sraw]['MAE'][e]:.2f}"
    R[f"sec-{sk}-n"] = str(P["settings"][sraw]["n_pairs"])
for h, hk in (("H1 域内：本文 − 直接测量", "hone"), ("H2 迁移：本文 − 直接测量", "htwo"), ("H3 全部：本文 − 恒为0", "hthree")):
    v = P["primary"][h]
    grp = {"hone": ("中文情绪-通用", "nli", "toxicity", "cold"), "htwo": ("中文情绪-疫情（迁移）",),
           "hthree": ("中文情绪-通用", "中文情绪-疫情（迁移）", "nli", "toxicity", "cold")}[hk]
    m2 = "恒为0" if hk == "hthree" else "直接测量"
    point = np.mean([P["settings"][g_]["MAE"]["本文"] - P["settings"][g_]["MAE"][m2] for g_ in grp])
    R[f"{hk}-d"] = num(1000 * point, 2)
    R[f"{hk}-ci"] = f"[{num(1000 * v['ci95'][0], 2)}, {num(1000 * v['ci95'][1], 2)}]"
    R[f"{hk}-p"] = "$<$0.001" if v["p_holm"] < 0.001 else f"{v['p_holm']:.3f}"

# ---------- write macros ----------
with open("numbers_results.tex", "w", encoding="utf-8") as f:
    f.write("%% Generated by make_results.py from %s. Do not edit by hand.\n" % src)
    f.write("\\makeatletter\n\\newcommand{\\res}[1]{\\ifcsname res@#1\\endcsname\\csname res@#1\\endcsname\\else\\textbf{\\textcolor{red}{??#1}}\\fi}\n")
    for k, v in sorted(R.items()):
        f.write(f"\\expandafter\\def\\csname res@{k}\\endcsname{{{v}}}\n")
    f.write("\\makeatother\n")

# ---------- tables ----------
with open("tab_fourcell.tex", "w", encoding="utf-8") as f:
    f.write("""% Generated by make_results.py
\\begin{table*}[t]
\\centering
\\caption{Retraining the fine-tuned LLM with item weights, decomposed item by item under confidence gating. Each cell gives the number of queried items and the change in net fixes (weighted minus standard judge), summed over three encoder seeds. Agreement is with the label of $A$. The last column divided by $3n$ equals the change in realised gain.}
\\label{tab:fourcell}
\\small
\\begin{tabular}{@{}llrrrrr@{}}
\\toprule
 & & Always & Newly & No longer & Always & \\\\
Setting & $q$ & agree & disagree & disagree & disagree & Total \\\\
\\midrule
""")
    last = None
    for slab, ql, cells, tot in fc_rows:
        lab = slab if slab != last else ""
        if last is not None and slab != last:
            f.write("\\addlinespace\n")
        last = slab
        cs = " & ".join(f"{n:,} / {num(d, 0)}".replace(",", "{,}", 1) if n >= 1000 else f"{n} / {num(d, 0)}" for n, d in cells)
        f.write(f"{lab} & {ql} & {cs} & {num(tot, 0)} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

with open("tab_secondary.tex", "w", encoding="utf-8") as f:
    f.write("""% Generated by make_results.py
\\begin{table}[t]
\\centering
\\caption{Estimating the AUC gain of the locked rule $g(x)+w\\,b$ from half of the validation split. Entries are mean absolute errors against the gain observed on the test set, in units of $10^{-3}$ AUC.}
\\label{tab:secondary}
\\small
\\begin{tabular}{@{}lrrrrr@{}}
\\toprule
Setting & Pairs & Binormal model & Direct measurement & Independence & Zero gain \\\\
\\midrule
""")
    for sraw, slab in SEC_SET:
        sk = dict((a, b) for a, b, _c in SETTINGS)[sraw]
        f.write(f"{slab} & {R[f'sec-{sk}-n']} & " + " & ".join(R[f"sec-{sk}-{ek}"] for _, ek in EST) + " \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\\\\[2pt]\n"
            "\\parbox{\\linewidth}{\\footnotesize Pairs count every combination of a repetition of $A$ and a second judge. "
            "On the binary tasks the two fine-tuned LLMs, one per training half, are each paired with all four encoders, "
            "so a task has $8+3\\times4=20$ pairs. On Chinese emotion five second judges meet three encoder seeds.}\n\\end{table}\n")

with open("tab_allpairs.tex", "w", encoding="utf-8") as f:
    f.write("""% Generated by make_results.py
\\begin{sidewaystable}
\\centering
\\caption{All pair types. Accuracy quantities are in points and averaged over the repetitions of $A$; $\\Delta\\mathcal{L}$ is in nats per item with its 95\\% bootstrap interval. Oracle, gate and router are accuracy gains over $A$ at the given budget, the oracle being an upper bound rather than a realised gain. Fixes and harms split the gated gain at 20\\%, and their difference equals it up to rounding. Fusion is the accuracy of $f_1$ minus that of $f_0$, with the share of test items whose predicted class changes in parentheses. The oracle is shown at 20\\% only, because at 10\\% and 40\\% it takes the same value for every pair type except the epidemic encoder under shift (10.0 points at 10\\%).}
\\label{tab:allpairs}
\\footnotesize
\\setlength{\\tabcolsep}{2.5pt}
\\begin{tabular}{@{}llrrrrrrrrrrrl@{}}
\\toprule
 & & & Oracle & \\multicolumn{5}{c}{Gate} & \\multicolumn{3}{c}{Router} & & \\\\
\\cmidrule(lr){5-9}\\cmidrule(lr){10-12}
Setting & Second judge $B$ & $B{-}A$ & 20\\% & 10\\% & 20\\% & 40\\% & Fixes & Harms & 10\\% & 20\\% & 40\\% & Fusion & $\\Delta\\mathcal{L}$ [95\\% CI] \\\\
\\midrule
""")
    last = None
    for r in rows:
        lab = r["slab"] if r["slab"] != last else ""
        if last is not None and r["slab"] != last:
            f.write("\\addlinespace\n")
        last = r["slab"]
        f.write(f"{lab} & {r['tlab']} & {pts(r['bma'])} & {pts(r['oracle20'])} & {pts(r['gate10'])} & {pts(r['gate20'])} & {pts(r['gate40'])} & {pts(r['gate_fix20'])} & {pts(r['gate_harm20'])} & "
                f"{pts(r['router10'])} & {pts(r['router20'])} & {pts(r['router40'])} & {pts(r['fus'])} ({pts(r['chg'])}) & {num(r['dll'])} [{num(r['dlo'])}, {num(r['dhi'])}] \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{sidewaystable}\n")

# ---------- figures ----------
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e"})
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

# Figure 2: dumbbell, oracle vs gate vs router at q = 20%
SHORT = {"Fine-tuned Qwen3-4B, weighted": "Fine-tuned Qwen3-4B, weighted loss"}
fig, ax = plt.subplots(figsize=(6.3, 7.2))
ypos, labels, heads = [], [], []
yv = 0
for sraw, skey, slab in SETTINGS:
    heads.append((yv, slab)); yv += 1
    for r in [r for r in rows if r["setting"] == skey]:
        ypos.append(yv); labels.append(SHORT.get(r["tlab"], r["tlab"])); yv += 1
    yv += 0.4
ypos = np.array(ypos)
O = np.array([100 * r["oracle20"] for r in rows]); G = np.array([100 * r["gate20"] for r in rows]); Rt = np.array([100 * r["router20"] for r in rows])
ax.axvline(0, color="#52514e", lw=0.8, zorder=1)
for yy, o, g in zip(ypos, O, G):
    ax.plot([g, o], [yy, yy], color="#c3c2b7", lw=2, zorder=2, solid_capstyle="round")
ax.scatter(O, ypos, s=34, facecolor="white", edgecolor=BLUE, linewidth=1.6, zorder=4, label="Budgeted oracle")
ax.scatter(G, ypos, s=34, color=ORANGE, marker="o", zorder=5, edgecolor="white", linewidth=0.8, label="Confidence gate")
ax.scatter(Rt, ypos, s=36, color=AQUA, marker="D", zorder=5, edgecolor="white", linewidth=0.8, label="Learned router")
tick_y = list(ypos) + [h[0] for h in heads]
tick_l = labels + [h[1] for h in heads]
ax.set_yticks(tick_y); ax.set_yticklabels(tick_l)
for t, yy in zip(ax.get_yticklabels(), tick_y):
    if yy in [h[0] for h in heads]:
        t.set_fontweight("bold"); t.set_color("#0b0b0b")
ax.set_ylim(yv - 0.6, -0.8)
ax.set_xlabel("Accuracy gain over $A$ at a 20% call budget (points)")
ax.grid(axis="x", color="#e6e5e0", lw=0.6, zorder=0)
ax.legend(loc="lower left", frameon=False, ncol=3, bbox_to_anchor=(0, 1.0), handletextpad=0.3, columnspacing=1.2)
ax.tick_params(axis="y", length=0)
fig.subplots_adjust(left=0.36, right=0.98, top=0.95, bottom=0.07)
fig.savefig("fig_overview.pdf"); fig.savefig("fig_overview.png", dpi=150)
plt.close(fig)

# Figure 3: realised gain (gate 20%) against each layer
KIND = {"ft": ("Fine-tuned LLM", BLUE, "o"), "zs": ("Zero-shot LLM", ORANGE, "s"),
        "twin": ("Encoder twin", AQUA, "^"), "epi": ("Epidemic encoder", YELLOW, "D")}
fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.6), sharey=True)
panels = [("oracle", "Budgeted oracle at 20% (points)", 100), ("bma", "Accuracy of $B$ minus $A$ (points)", 100),
          ("dll", r"Added information $\Delta\mathcal{L}$ (nats)", 1)]
for ax, (nm, xl, sc) in zip(axs, panels):
    ax.axhline(0, color="#52514e", lw=0.8, zorder=1)
    for kind, (lab, col, mk) in KIND.items():
        m = np.array([r["kind"] == kind for r in rows])
        ax.scatter(sc * X[nm][m], 100 * y[m], s=22, color=col, marker=mk, edgecolor="white", linewidth=0.6, zorder=3, label=lab)
    ax.set_xlabel(xl)
    ax.set_title(r"Spearman $r_s$ = " + R[f"rho-{nm}-gate"].replace("$-$", "\u2212"), fontsize=8)
    ax.grid(color="#eeede8", lw=0.5, zorder=0)
axs[0].set_ylabel("Gated gain at 20% (points)")
fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False, fontsize=7.5, handletextpad=0.2, bbox_to_anchor=(0.5, 1.0))
fig.subplots_adjust(left=0.09, right=0.99, top=0.80, bottom=0.2, wspace=0.08)
fig.savefig("fig_layers.pdf"); fig.savefig("fig_layers.png", dpi=150)
plt.close(fig)
print(f"{len(R)} macros, {len(rows)} pair types written")
