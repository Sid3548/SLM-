from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.logging_utils import build_event, write_jsonl_record


SPACE_RE = re.compile(r"\s+")


def clean_line(text: str, min_chars: int) -> str | None:
    text = SPACE_RE.sub(" ", text).strip()
    if len(text) < min_chars:
        return None
    return text



def main():
    ap = argparse.ArgumentParser(description="Clean raw text corpus and keep quality lines.")
    ap.add_argument("--input", default="artifacts/raw/train_raw.txt")
    ap.add_argument("--output", default="artifacts/clean/train_clean.txt")
    ap.add_argument("--min-chars", type=int, default=40)
    ap.add_argument("--log", default="artifacts/logs/data_pipeline.jsonl")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    with open(in_path, "r", encoding="utf-8") as src, open(out_path, "w", encoding="utf-8") as dst:
        for line in src:
            total += 1
            cleaned = clean_line(line, args.min_chars)
            if cleaned is None:
                continue
            dst.write(cleaned + "\n")
            kept += 1

    write_jsonl_record(
        args.log,
        build_event(
            "clean_complete",
            input=str(in_path),
            output=str(out_path),
            total_lines=total,
            kept_lines=kept,
            dropped_lines=total - kept,
            min_chars=args.min_chars,
        ),
    )
    print(f"Cleaned lines kept: {kept}/{total}")


if __name__ == "__main__":
    main()
