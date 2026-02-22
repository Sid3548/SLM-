# SLM- (Small Language Model from Scratch)

This repository trains a **100M+ parameter** decoder-only language model (default ~113M params) with long-context configs, plus a support-ticket/human-handoff focused data pipeline.

## What you get

- From-scratch transformer model in PyTorch (RMSNorm, RoPE, GQA, SwiGLU).  
- Data pipeline for fetch → clean → support-focused corpus curation → tokenization.  
- Training script with:
  - build-data-only mode,
  - step-based training,
  - epoch-based training (`--epochs`),
  - 100M minimum parameter guard.
- Sampling script to validate behavior from checkpoints.

---

## Step-by-step: run this on your personal machine

## 0) Hardware and OS assumptions

- NVIDIA GPU (you mentioned 4060 Ti 16GB).
- Linux recommended (Ubuntu 22.04/24.04).
- Python 3.10+.
- CUDA driver installed and working (`nvidia-smi`).

## 1) Clone and enter repo

```bash
git clone <your-fork-or-repo-url>
cd SLM-
```

## 2) Create environment and install dependencies

### Option A (recommended): use helper script

```bash
./scripts/personal_machine_setup.sh
source .venv/bin/activate
```

### Option B (manual)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

## 3) Quick sanity checks

```bash
python -m py_compile train.py sample.py slm/*.py data_pipeline/*.py tools/*.py
python tools/estimate_params.py --config configs/support_12288.yaml
```

Expected: around `113.27M` parameters.

## 4) Build support-focused training data (ticket + human handoff)

This creates a support-specific corpus and tokenizer artifacts:

```bash
python data_pipeline/run_support_pipeline.py \
  --config configs/support_12288.yaml \
  --max-rows 300000 \
  --min-chars 40 \
  --synthetic-samples 25000
```

Outputs:
- `artifacts/clean/support_only_corpus.txt`
- `artifacts/tokenizer_support.json`
- `artifacts/support_train_tokens.npy`
- `artifacts/logs/data_pipeline.jsonl`

## 5) Train for exactly 1 epoch (local validation)

### Fast small local validation (easier to finish)

```bash
python train.py --config configs/support_one_epoch_local.yaml --build-data-only
python train.py --config configs/support_one_epoch_local.yaml --epochs 1
```

Or use:

```bash
./scripts/train_support_one_epoch.sh configs/support_one_epoch_local.yaml
```

### Larger support run (closer to real training)

```bash
python train.py --config configs/support_12288.yaml --epochs 1
```

> Note: `support_12288.yaml` can be heavy; if OOM or too slow, lower `max_seq_len`, increase `grad_accum_steps`, or start from `support_smoke_epoch.yaml`.

## 6) Evaluate a trained checkpoint

```bash
python sample.py \
  --config configs/support_one_epoch_local.yaml \
  --ckpt checkpoints/epoch_1.pt \
  --prompt "I need to raise a support ticket and talk to a human." \
  --max-new 80
```

## 7) Recommended tuning for your 4060 Ti (16GB)

- Keep `batch_size: 1`.
- Use `precision: bf16` when CUDA supports it.
- Increase `grad_accum_steps` instead of batch size.
- If OOM:
  - reduce `max_seq_len` first,
  - then reduce `d_model` or `n_layers`.

## 8) Files you will edit most often

- `configs/support_12288.yaml` (main support training config)
- `configs/support_one_epoch_local.yaml` (quick 1-epoch local run)
- `data_pipeline/build_support_corpus.py` (relevance filtering/synthetic support dialogues)
- `train.py` (training behavior)

---

## Support-ticket / human-handoff objective

The support corpus pipeline is intentionally scoped for:
- one-on-one support communication,
- ticket creation/case management,
- escalation and handoff to human agent.

Use `data_pipeline/build_support_corpus.py` to adjust inclusion/exclusion patterns as your domain evolves.
