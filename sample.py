from __future__ import annotations

import argparse

import torch
from tokenizers import Tokenizer

from slm.config import load_config
from slm.model import TinyLM



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base_12288.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new", type=int, default=128)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tok = Tokenizer.from_file(cfg.data.tokenizer_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyLM(cfg.model).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ids = tok.encode(args.prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=args.max_new)
    text = tok.decode(out[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
