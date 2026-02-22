from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 32768
    d_model: int = 768
    n_layers: int = 14
    n_heads: int = 12
    n_kv_heads: int = 4
    ffn_hidden: int = 2048
    max_seq_len: int = 12288
    rope_theta: float = 1000000.0
    dropout: float = 0.0


@dataclass
class TrainConfig:
    batch_size: int = 1
    grad_accum_steps: int = 64
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 1000
    max_steps: int = 50000
    max_grad_norm: float = 1.0
    precision: str = "bf16"
    compile: bool = False
    checkpoint_every: int = 1000
    log_every: int = 20
    min_params_million: float = 100.0


@dataclass
class DataConfig:
    approved_registry: str = "data/approved_datasets.yaml"
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-v1"
    split: str = "train"
    text_key: str = "text"
    tokenizer_path: str = "artifacts/tokenizer.json"
    tokenized_path: str = "artifacts/train_tokens.npy"
    num_proc: int = 4
    local_text_path: str = "artifacts/clean/train_clean.txt"


@dataclass
class FullConfig:
    model: ModelConfig
    train: TrainConfig
    data: DataConfig



def _load_dataclass(cls: Any, data: dict[str, Any]) -> Any:
    return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})



def load_config(path: str | Path) -> FullConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return FullConfig(
        model=_load_dataclass(ModelConfig, raw.get("model", {})),
        train=_load_dataclass(TrainConfig, raw.get("train", {})),
        data=_load_dataclass(DataConfig, raw.get("data", {})),
    )
