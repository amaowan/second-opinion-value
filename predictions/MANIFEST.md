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
| `en/seed{42,43,44}/` | step 1 (`p0_robustness_kaggle.py`) | `test_probs.npy`, `test_labels.npy` |
| `tasks/{nli,toxicity,cold}/` | step 7 (`g1_task_pipeline_kaggle.py`) | `labels/`, `agree/`, `enc/half{0,1}/seed{42,43}/`, `llm/{zs_qwen3_4b,zs_qwen25_3b,lora_half0,lora_half1}/` |

`metrics/` keeps the summary JSON written by each notebook, for reference. The analysis does not read it.

The test-split probabilities of every judge in this folder were checked against the item tables `results/a1_items_*.npz` written by the main analysis, and they agree exactly.

Do not include:

- LoRA adapter folders (`lora_adapter_ep*`, `lora_weighted_ep*`). They are large, and the analysis does not need them.
- Probabilities from LoRA epochs 1 and 2 (`*-ep1`, `*-ep2`).
- Outputs of the `complement` LoRA variant, the second-family runs (`zs_x`, `lora_x_*`) and any exploratory notebook.
- Any copy of the SMP2020-EWECT archive or its text files.

GitHub rejects files larger than 100 MB. All prediction arrays in this project are far smaller than that, but check the total size before you upload.
