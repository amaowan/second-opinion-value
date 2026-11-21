# When Is a Second Opinion Worth Asking For?

Code and results for the paper *When Is a Second Opinion Worth Asking For? Measuring the Value of Model Disagreement in Text Classification*.

The paper looks at a cheap first classifier *A* that hands the items it is unsure about to a second model *B*, often a large language model. It separates three quantities:

1. **Complementarity:** a budgeted oracle that knows which items *B* would fix.
2. **Realised gain:** what a confidence gate or a learned router gets when it has to choose items before the call. The gain is split into fixes and harms.
3. **Added information:** the held-out log-loss gain when *B*'s label is added to *A*'s probabilities.

The study covers five tasks: English and Chinese emotion classification, natural language inference, toxicity and Chinese offensive language.

## Repository layout

```
experiments/   Kaggle scripts that train the judges and run the main analysis (comments are in Chinese)
analysis/      Scripts that turn the analysis output into every number, table and figure in the paper
  inputs/      Small result files the appendix scripts read (metrics JSONs, NLI prediction arrays)
results/       a1_results.json, the output of the main analysis
predictions/   Per-item probabilities of every judge (see predictions/MANIFEST.md)
```

## Reproduce the paper's numbers (no GPU, a few minutes)

```bash
pip install -r requirements.txt
cd analysis
python make_results.py ../results/a1_results.json   # Section 5: numbers_results.tex, tables, fig_overview, fig_layers
python make_appendix.py                             # Appendices: numbers_appendix.tex, tab_labelflip.tex, ChaosNLI check
```

Every number in the paper is a LaTeX macro written by one of these two scripts. Nothing is typed by hand.

## Re-run the main analysis from the stored predictions (no GPU, 1 to 2 hours)

```bash
cd experiments
A1_INPUT=../predictions python a1_three_layer_kaggle.py   # writes a1_results.json and a1_items_*.npz
```

This step needs an internet connection. The script downloads the cached English LLM predictions from the repository of Tikhomirov and Savchenko (2026), and it reloads MNLI and ChaosNLI from Hugging Face to rebuild the premise clusters used in the bootstrap.

## Re-run everything from scratch (Kaggle, T4 GPU)

Each script runs as a single Kaggle notebook cell. Run them in this order. The settings each script needs are in its header comment.

| Step | Script | What it does | GPU time | Paper |
|---|---|---|---|---|
| 1 | `p0_robustness_kaggle.py` | Trains RoBERTa-base on GoEmotions (Ekman 7) with seeds 42, 43 and 44 and writes `seed42/`, `seed43/`, `seed44/`. Only this training part is used in the paper. If the output of `goemotions_roberta_kaggle.py` (same settings, seed 42) is present in the session, seed 42 is reused from it | 60–80 min | Sec. 4.1 |
| 2 | `c1_ewect_encoder_kaggle.py` | Cleans SMP2020-EWECT and trains chinese-roberta-wwm-ext on general and epidemic posts, 3 seeds each | 60–90 min | Sec. 4.1–4.2 |
| 3 | `c2_zh_zeroshot_llm_kaggle.py` | Zero-shot Qwen3-4B on Chinese emotion and the free-generation check | 30–60 min | Sec. 4.2, App. A |
| 4 | `e14_zh_lora_verifier_kaggle.py` | LoRA Qwen3-4B on Chinese emotion, 3 epochs | about 5.5 h | Sec. 4.2 |
| 5 | `e15a_zh_oof_encoder_kaggle.py` | Out-of-fold encoder probabilities for step 6 | 40–60 min | Sec. 4.2 |
| 6 | `e15b_zh_complement_lora_kaggle.py` | The weighted LoRA judge (`LORA_VARIANT=weighted`). The paper uses only this variant | about 6 h | Sec. 5.3 |
| 7 | `g1_task_pipeline_kaggle.py` | NLI, toxicity and COLD: data, encoders, zero-shot LLMs and LoRA judges. Run once per task with `G_TASK` = nli, toxicity, cold, `G_PART=all` and `G_XLLM` empty | 2–6 h per task | Sec. 4 |
| 8 | `a1_three_layer_kaggle.py` | Main analysis on the outputs of steps 1–7 | none (1–2 h CPU) | Sec. 5 |

In Kaggle, the outputs of steps 1–7 are attached to step 8 with **Add Input → Notebooks**. The analysis finds each judge by folder name, as listed in `predictions/MANIFEST.md`. Step 1 already writes the English encoders into folders named `seed42`, `seed43` and `seed44`.

## Data

All data are public. The scripts download them, except SMP2020-EWECT.

| Task | Source |
|---|---|
| English emotion | `google-research-datasets/go_emotions` (Hugging Face) |
| Chinese emotion | SMP2020-EWECT, <https://smp2020ewect.github.io/>. Download `评测数据集.zip` from the organisers and add it to Kaggle as a private dataset |
| NLI | `nyu-mll/multi_nli` and `metaeval/chaos-mnli-ambiguity` (Hugging Face) |
| Toxicity | `google/civil_comments` (Hugging Face) |
| Chinese offence | `thu-coai/cold` (Hugging Face) |

SMP2020-EWECT is released without a stated licence, so this repository does not contain any of its posts. The files in `predictions/` hold only model probabilities, labels and the post identifiers of the official release. Please follow the terms of each dataset when you use it.

## Environment

The final analysis and the last training runs used the Kaggle image of 19 November 2026 with these versions:

- Python 3.13
- PyTorch 2.11
- Transformers 5.20
- PEFT 0.21
- scikit-learn 1.6
- NumPy 2.1

Earlier training runs used earlier Kaggle images. The analysis reads only the stored predictions, so it does not depend on the training environment.

## Scope of this repository

The repository contains the experiments reported in the paper. Exploratory runs from earlier stages of the project, which the paper does not use, are not included.

## Citation

```bibtex
@article{TODO,
  title  = {When Is a Second Opinion Worth Asking For? Measuring the Value of Model Disagreement in Text Classification},
  author = {TODO},
  year   = {2027}
}
```

## License

The code is released under the MIT License (see `LICENSE`). The datasets keep their own licences.
