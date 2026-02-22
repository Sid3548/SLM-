# SLM- (Small Language Model from Scratch)

This repo contains an end-to-end training stack for a **100M+ parameter decoder-only LM** with a **12,288 token context window**.

## What is included

- From-scratch transformer model (PyTorch):
  - RMSNorm
  - RoPE
  - GQA
  - SwiGLU
  - tied embeddings
- Default config around **~112M params** and **12k context**.
- New **`data_pipeline/` folder** for:
  - fetching approved data,
  - cleaning/filtering raw text,
  - logging every pipeline step to JSONL.
- Tokenizer training + `.npy` token packing.
- Training + sampling scripts.
- Parameter floor guard: training stops if model is below 100M.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Data flow (new)

1) Review/approve datasets in `data/approved_datasets.yaml`.
2) Run fetch + clean + tokenize pipeline:

```bash
python data_pipeline/run_pipeline.py --config configs/base_12288.yaml --min-chars 40 --max-rows 0
```

Use `--max-rows` > 0 for fast dry-runs; keep `0` for full dataset.

Pipeline artifacts:
- raw text: `artifacts/raw/train_raw.txt`
- cleaned text: `artifacts/clean/train_clean.txt`
- logs: `artifacts/logs/data_pipeline.jsonl`
- tokenizer: `artifacts/tokenizer.json`
- token ids: `artifacts/train_tokens.npy`

## Train

```bash
python train.py --config configs/base_12288.yaml
```

The trainer prints parameter count and raises if `< 100M`.

For a quick CPU smoke test (1 step, still 100M+ params):

```bash
python train.py --config configs/smoke_train_cpu.yaml
```

## Sample

```bash
python sample.py \
  --config configs/base_12288.yaml \
  --ckpt checkpoints/step_500.pt \
  --prompt "Explain how transformers use attention" \
  --max-new 120
```
