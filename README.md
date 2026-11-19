# pi-detect — Lightweight prompt-injection detectors vs. perplexity and LLM judges

A complete, reproducible pipeline that trains and compares prompt-injection detectors on
**accuracy, out-of-source generalisation, false positives (over-defense) and latency**,
runs on a SLURM cluster, and regenerates every table and figure of the accompanying paper
(`paper/main.tex`).

| Family | Detector(s) | Training |
|---|---|---|
| Classical | TF-IDF (char + word n-grams) + logistic regression | supervised |
| Embedding + LR | all-MiniLM-L6-v2 / BGE-small embeddings + logistic regression | supervised probe |
| Fine-tuned | DistilBERT-base-uncased (3 seeds) | full fine-tuning |
| Perplexity | GPT-2 mean PPL, windowed PPL (Jain et al.), PPL-features + LR (Alon & Kamfonas) | zero-shot / light |
| LLM-as-judge | Qwen2.5-1.5B / 7B-Instruct (Llama-3.1-8B optional), P(Yes)/(P(Yes)+P(No)) | zero-shot |
| Reference | ProtectAI DeBERTa-v3 injection classifier | off-the-shelf |

All detectors share **one threshold rule** (max-F1 on validation). Zero-shot judges are also
reported at their default 0.5 threshold.

---

## 1. Layout

```
pi-detect/
├── configs/default.yaml        # datasets, detectors, hyper-params, eval + latency settings
├── src/pidetect/
│   ├── data.py                 # HF loading, dedup, leakage removal, OOD + synthetic stress sets
│   ├── metrics.py              # threshold rules, metrics, bootstrap CIs, paired bootstrap test
│   ├── utils.py
│   └── detectors/              # base.py, classical.py, finetune.py, perplexity.py, llm_judge.py
├── scripts/
│   ├── setup_env.sh            # create conda env / venv
│   ├── download_assets.py      # pre-cache datasets + models (needs internet)
│   ├── prepare_data.py         # build data/processed/*.parquet + manifest.json
│   ├── run_detector.py         # fit + calibrate + score all splits for one detector/seed
│   ├── benchmark_latency.py    # controlled bs=1 / bs=32 latency on GPU or CPU
│   ├── aggregate.py            # metrics, CIs, significance, cascade, LaTeX tables, figures
│   ├── make_smoke_assets.py    # tiny random models + synthetic data (offline test)
│   └── run_smoke.sh            # end-to-end smoke test
├── slurm/                      # *.job files + common.sh + submit_all.sh
├── paper/                      # main.tex, references.bib, generated/, figures/
└── tests/test_core.py          # fast unit tests (pytest)
```

## 2. Quick start on the cluster

```bash
# 0) unzip into a project/scratch directory and enter it
cd pi-detect

# 1) edit the SITE SETTINGS block in slurm/common.sh (modules, conda, HF_HOME on scratch)

# 2) create the environment (login node, with internet)
bash scripts/setup_env.sh                 # conda env "pidetect"   (or USE_VENV=1 ...)
#    if your CUDA needs a specific torch build, install it first, e.g.
#    pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3) validate env + GPU in ~5 minutes with tiny models (no downloads)
sbatch slurm/smoke.job                    # or: bash scripts/run_smoke.sh

# 4) pre-download data + models (login node if compute nodes are offline)
huggingface-cli login                     # only needed for gated Llama (disabled by default)
ALLOW_DOWNLOAD=1 bash slurm/00_download.job

# 5) submit everything with dependencies
ACCOUNT=<your_account> GPU_PARTITION=<gpu_part> CPU_PARTITION=<cpu_part> SKIP_DOWNLOAD=1 \
  bash slurm/submit_all.sh

# 6) results
cat results/findings.md                   # rankings, OOD, FPR, latency, cascade
ls paper/generated paper/figures          # LaTeX tables + PDF figures used by the paper
cd paper && latexmk -pdf main.tex         # build the paper (or upload paper/ to Overleaf)
```

Submit from the repository root (the jobs `source slurm/common.sh` relative to it).

## 3. Job graph

```
00_download ─► 01_prepare_data ─┬─► 02_tfidf_lr      (CPU)
                                ├─► 03_embed_lr      (GPU, array: MiniLM, BGE)
                                ├─► 04_distilbert    (GPU, array: 3 seeds)
                                ├─► 05_perplexity    (GPU)
                                ├─► 06_llm_judge     (GPU, array: Qwen 1.5B, 7B [, Llama 8B])
                                └─► 07_reference     (GPU)
                                          │ (afterany)
                         08_latency_gpu (exclusive) + 09_latency_cpu (exclusive)
                                          │
                                   10_aggregate  ─► tables, figures, macros, paper PDF
```

Rough wall-clock on one A100-40GB: data < 5 min, TF-IDF < 5 min, embeddings ~5 min each,
DistilBERT ~10–20 min per seed, perplexity ~10 min, Qwen-1.5B judge ~15 min,
Qwen-7B judge ~45–90 min, latency jobs ~1–2 h. Set `--time` higher if your GPUs are older.

GPU memory: everything except the 7B/8B judges fits in 16 GB. The 7B judge in bf16 needs
~20–30 GB at `batch_size: 16`; lower `batch_size` in the config for 24 GB cards.

## 4. Outputs

| Path | Content |
|---|---|
| `data/processed/*.parquet`, `manifest.json` | splits, counts, sources, leakage removed |
| `results/runs/<variant>/seed<k>/scores_<split>.parquet` | per-example scores |
| `results/runs/<variant>/seed<k>/meta.json` | threshold, fit/score time, params, env, training curve |
| `results/latency/<detector>__<device>.json` | bs=1 p50/p95/p99, bs=32 throughput, peak memory |
| `results/tables/*.csv` | long + summary metrics, CIs, significance, cascade, latency |
| `results/findings.md` | auto-generated rankings — use it to verify the paper's claims |
| `paper/generated/*.tex` | LaTeX tables and `macros.tex` (`\res{variant}{split}{metric}`) |
| `paper/figures/*.pdf` | ROC, latency–F1 Pareto, benign FPR, OOD heatmap |

## 5. The paper

`paper/main.tex` pulls **every number** from `generated/macros.tex` and the
`generated/tab_*.tex` tables, so it updates automatically after `aggregate.py`.
Interpretive sentences written before the full run are wrapped in `\claim{...}` and
render **blue** while `\drafttrue` is set. After your run:

1. read `results/findings.md`;
2. check each blue sentence against it and edit any that the data do not support;
3. set `\draftfalse` and rebuild.

A missing number renders as a red **??**; this means a detector/split was not run.

## 6. Customising

* **Add a detector**: implement `Detector.fit/score/save/load` in `src/pidetect/detectors/`,
  register its `type` in `detectors/__init__.py`, add an entry in the config.
* **Add a dataset**: add a source under `data.sources` (`role: id` or `role: ood`,
  `text_cols`, `label_col`/`label_map` or `fixed_label`). Optional sources that fail to load
  are skipped with a warning; `required: true` makes failure fatal.
* **Threshold rule**: `eval.threshold_rule: max_f1 | youden | fpr@0.01`.
* **Cascade**: `eval.cascade.cheap/expensive/fractions`.
* **Llama-3.1-8B judge**: accept the licence on HF, set `enabled: true`, use `--array=0-2`
  in `06_llm_judge.job`.

## 7. Dataset notes

The HF dataset IDs and column names in `configs/default.yaml` were set from the dataset
cards. `download_assets.py` prints the actual column names of every dataset it fetches. If
a dataset has been renamed or restructured, adjust `text_cols` / `label_col` there; loaders
accept a list of candidate column names. Check each dataset's licence before redistributing
derived data.

## 8. Tests

```bash
pytest -q tests          # unit tests (seconds, no downloads)
bash scripts/run_smoke.sh  # full pipeline with tiny models (minutes, no downloads)
```
