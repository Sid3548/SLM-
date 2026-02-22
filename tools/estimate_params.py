from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slm.config import load_config



def estimate_params(vocab_size: int, d_model: int, n_layers: int, ffn_hidden: int, n_kv_heads: int, n_heads: int):
    head_dim = d_model // n_heads
    tok = vocab_size * d_model
    per_layer_attn = d_model * (n_heads * head_dim) + d_model * (n_kv_heads * head_dim) * 2 + d_model * d_model
    per_layer_ffn = d_model * ffn_hidden * 2 + ffn_hidden * d_model
    per_layer_norm = d_model * 2
    final_norm = d_model
    return tok + n_layers * (per_layer_attn + per_layer_ffn + per_layer_norm) + final_norm



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base_12288.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    est = estimate_params(
        cfg.model.vocab_size,
        cfg.model.d_model,
        cfg.model.n_layers,
        cfg.model.ffn_hidden,
        cfg.model.n_kv_heads,
        cfg.model.n_heads,
    )
    print(f"Estimated params: {est} ({est / 1e6:.2f}M)")
    if est / 1e6 < cfg.train.min_params_million:
        raise SystemExit(
            f"Estimated model too small: {est / 1e6:.2f}M < {cfg.train.min_params_million:.2f}M"
        )


if __name__ == "__main__":
    main()
