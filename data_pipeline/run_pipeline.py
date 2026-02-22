from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]



def run_step(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)



def main():
    ap = argparse.ArgumentParser(description="Run fetch + clean + tokenize pipeline.")
    ap.add_argument("--config", default="configs/base_12288.yaml")
    ap.add_argument("--min-chars", type=int, default=40)
    ap.add_argument("--max-rows", type=int, default=0, help="0 means full split")
    args = ap.parse_args()

    run_step([sys.executable, "data_pipeline/fetch_data.py", "--config", args.config, "--max-rows", str(args.max_rows)])
    run_step([sys.executable, "data_pipeline/clean_data.py", "--min-chars", str(args.min_chars)])
    run_step([sys.executable, "train.py", "--config", args.config, "--build-data-only"])


if __name__ == "__main__":
    main()
