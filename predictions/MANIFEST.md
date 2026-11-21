# What goes in this folder

This folder holds the per-item predictions that `experiments/a1_three_layer_kaggle.py` reads. Copy the output folders of the Kaggle notebooks here, keeping the folder names below. The analysis finds files by these relative paths, so the folder names matter. Where you put them inside `predictions/` does not.

The files contain probabilities, labels and post identifiers only, never the text of a post.

| Folder (inside `predictions/`) | Kaggle notebook (script) | Files |
|---|---|---|
| `zh/labels/` | step 2 (`c1_ewect_encoder_kaggle.py`) | `{usual_eval,usual_test,virus_eval,virus_test}_labels.npy`, `*_ids.npy` |
| `zh/{usual,virus}/seed{42,43,44}/` | step 2 | `{usual_eval,usual_test,virus_eval,virus_test}_probs.npy` |
| `zh_llm/Qwen3-4B-Instruct-2507/v1/` | step 3 (`c2_zh_zeroshot_llm_kaggle.py`) | the four `*_probs.npy` files |
| `zh_llm/Qwen3-4B-LoRA-ep3/usual/` | step 4 (`e14_zh_lora_verifier_kaggle.py`) | the four `*_probs.npy` files |
| `zh_llm/Qwen3-4B-LoRAW-ep3/usual/` | step 6 (`e15b_zh_complement_lora_kaggle.py`, weighted) | the four `*_probs.npy` files |
| `en/seed{42,43,44}/` | step 1 (`goemotions_roberta_kaggle.py`), one run per seed | `test_probs.npy`, `test_labels.npy` (the script writes them to its output root, so move each run's files into its seed folder) |
| `tasks/{nli,toxicity,cold}/` | step 7 (`g1_task_pipeline_kaggle.py`) | `labels/`, `agree/`, `enc/half{0,1}/seed{42,43}/`, `llm/{zs_qwen3_4b,zs_qwen25_3b,lora_half0,lora_half1}/` |
