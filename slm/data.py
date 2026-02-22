from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from slm.config import DataConfig



def assert_dataset_is_approved(cfg: DataConfig):
    registry_path = Path(cfg.approved_registry)
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Dataset approval registry missing at {registry_path}. "
            "Create it and explicitly approve sources before training."
        )
    with open(registry_path, "r", encoding="utf-8") as f:
        approved = yaml.safe_load(f) or {}

    key = f"{cfg.dataset_name}:{cfg.dataset_config}"
    allowed = set(approved.get("approved", []))
    if key not in allowed:
        raise PermissionError(
            f"Dataset {key} is not approved. Add it to {registry_path} before tokenization/training."
        )



def text_iter_from_config(cfg: DataConfig):
    local_path = Path(cfg.local_text_path)
    if local_path.exists():
        with open(local_path, "r", encoding="utf-8") as f:
            for line in f:
                txt = line.strip()
                if txt:
                    yield txt
        return

    assert_dataset_is_approved(cfg)
    ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.split)
    for row in ds:
        txt = (row.get(cfg.text_key, "") or "").strip()
        if txt:
            yield txt



def build_tokenizer(cfg: DataConfig, vocab_size: int = 32768) -> Tokenizer:
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
    )
    tok.train_from_iterator(text_iter_from_config(cfg), trainer=trainer)
    Path(cfg.tokenizer_path).parent.mkdir(parents=True, exist_ok=True)
    tok.save(cfg.tokenizer_path)
    return tok



def tokenize_to_npy(cfg: DataConfig, tokenizer: Tokenizer | None = None) -> Path:
    tok = tokenizer or Tokenizer.from_file(cfg.tokenizer_path)

    all_tokens = []
    bos_id = tok.token_to_id("<bos>") or 2
    eos_id = tok.token_to_id("<eos>") or 3

    for txt in text_iter_from_config(cfg):
        ids = tok.encode(txt).ids
        all_tokens.extend([bos_id, *ids, eos_id])

    arr = np.array(all_tokens, dtype=np.int32)
    out_path = Path(cfg.tokenized_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr)
    return out_path


class PackedTokenDataset(torch.utils.data.Dataset):
    def __init__(self, npy_path: str, seq_len: int):
        self.tokens = np.load(npy_path, mmap_mode="r")
        self.seq_len = seq_len
        self.n = len(self.tokens) - seq_len - 1

    def __len__(self):
        return max(0, self.n)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.tokens[idx : idx + self.seq_len].astype(np.int64))
        y = torch.from_numpy(self.tokens[idx + 1 : idx + self.seq_len + 1].astype(np.int64))
        return x, y
