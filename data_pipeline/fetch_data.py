from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import load_dataset

from slm.config import load_config
from slm.data import assert_dataset_is_approved
from data_pipeline.logging_utils import build_event, write_jsonl_record



def main():
    ap = argparse.ArgumentParser(description="Fetch approved dataset split and save as raw text.")
    ap.add_argument("--config", default="configs/base_12288.yaml")
    ap.add_argument("--output", default="artifacts/raw/train_raw.txt")
    ap.add_argument("--log", default="artifacts/logs/data_pipeline.jsonl")
    ap.add_argument("--max-rows", type=int, default=0, help="0 means no cap")
    args = ap.parse_args()

    cfg = load_config(args.config)
    assert_dataset_is_approved(cfg.data)

    ds = load_dataset(cfg.data.dataset_name, cfg.data.dataset_config, split=cfg.data.split)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            if args.max_rows and i >= args.max_rows:
                break
            text = (row.get(cfg.data.text_key, "") or "").strip()
            if not text:
                continue
            f.write(text + "\n")
            kept += 1

    write_jsonl_record(
        args.log,
        build_event(
            "fetch_complete",
            dataset=cfg.data.dataset_name,
            dataset_config=cfg.data.dataset_config,
            split=cfg.data.split,
            output=str(out_path),
            rows_written=kept,
            max_rows=args.max_rows,
        ),
    )
    print(f"Wrote {kept} rows to {out_path}")


if __name__ == "__main__":
    main()
