from __future__ import annotations

import argparse
import subprocess
import sys



def run_step(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)



def main():
    ap = argparse.ArgumentParser(description="Run support-focused data pipeline: fetch -> clean -> support-corpus -> tokenize")
    ap.add_argument("--config", default="configs/support_12288.yaml")
    ap.add_argument("--min-chars", type=int, default=40)
    ap.add_argument("--max-rows", type=int, default=300000)
    ap.add_argument("--synthetic-samples", type=int, default=25000)
    args = ap.parse_args()

    run_step(
        [
            sys.executable,
            "-m",
            "data_pipeline.fetch_data",
            "--config",
            args.config,
            "--max-rows",
            str(args.max_rows),
        ]
    )
    run_step([sys.executable, "-m", "data_pipeline.clean_data", "--min-chars", str(args.min_chars)])
    run_step(
        [
            sys.executable,
            "-m",
            "data_pipeline.build_support_corpus",
            "--synthetic-samples",
            str(args.synthetic_samples),
        ]
    )
    run_step([sys.executable, "train.py", "--config", args.config, "--build-data-only"])


if __name__ == "__main__":
    main()
