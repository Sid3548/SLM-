from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

from data_pipeline.logging_utils import build_event, write_jsonl_record


SUPPORT_PATTERNS = [
    r"\bticket\b",
    r"\bcase\b",
    r"\bissue\b",
    r"\berror\b",
    r"\brefund\b",
    r"\bcancel",
    r"\bsubscription\b",
    r"\bbilling\b",
    r"\bpayment\b",
    r"\baccount\b",
    r"\blogin\b",
    r"\bpassword\b",
    r"\bcan't\s+access\b",
    r"\bcannot\s+access\b",
    r"\bhuman\b",
    r"\bagent\b",
    r"\brepresentative\b",
    r"\bescalat",
    r"\bmanager\b",
    r"\bcall\s+me\b",
]

NEGATIVE_PATTERNS = [
    r"\bpoem\b",
    r"\bstory\b",
    r"\bcode\b",
    r"\brecipe\b",
    r"\bmath\b",
    r"\bjoke\b",
]


def compile_patterns(raw_patterns: list[str]):
    return [re.compile(p, flags=re.IGNORECASE) for p in raw_patterns]


def is_support_relevant(text: str, pos: list[re.Pattern], neg: list[re.Pattern]) -> bool:
    low = text.lower()
    if any(rx.search(low) for rx in neg):
        return False
    return any(rx.search(low) for rx in pos)


def synth_dialogues(n: int) -> list[str]:
    intents = [
        "I need to raise a support ticket for a failed payment.",
        "Please connect me to a human support representative.",
        "My account is locked and I need an agent to help.",
        "I want to escalate this issue to a manager.",
        "Please create a case and share the ticket ID.",
        "I cannot access my account after password reset.",
        "I need billing support and a human callback.",
        "The service outage is critical; escalate to human support.",
    ]
    policies = [
        "Assistant policy: acknowledge issue, collect contact details, create ticket, offer human handoff.",
        "Assistant policy: do not answer unrelated tasks; only support troubleshooting and escalation.",
        "Assistant policy: prioritize ticket creation, severity tagging, and ETA updates.",
        "Assistant policy: if user asks for human, confirm and route to live agent queue.",
    ]

    out = []
    for i in range(n):
        u = random.choice(intents)
        p = random.choice(policies)
        out.append(
            "User: " + u + "\n"
            "Assistant: I can help with support only. I will open a ticket and route you to a human agent. "
            "Please share your registered email, a short issue summary, and urgency level.\n"
            + p
        )
    return out


def main():
    ap = argparse.ArgumentParser(description="Create support-only corpus for ticket/human-handoff model behavior.")
    ap.add_argument("--input", default="artifacts/clean/train_clean.txt")
    ap.add_argument("--output", default="artifacts/clean/support_only_corpus.txt")
    ap.add_argument("--synthetic-samples", type=int, default=25000)
    ap.add_argument("--max-real", type=int, default=300000)
    ap.add_argument("--min-chars", type=int, default=40)
    ap.add_argument("--log", default="artifacts/logs/data_pipeline.jsonl")
    args = ap.parse_args()

    pos = compile_patterns(SUPPORT_PATTERNS)
    neg = compile_patterns(NEGATIVE_PATTERNS)

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept_real = 0

    with open(out_path, "w", encoding="utf-8") as dst:
        if in_path.exists():
            with open(in_path, "r", encoding="utf-8") as src:
                for line in src:
                    total += 1
                    text = line.strip()
                    if len(text) < args.min_chars:
                        continue
                    if not is_support_relevant(text, pos, neg):
                        continue
                    dst.write(text + "\n")
                    kept_real += 1
                    if kept_real >= args.max_real:
                        break

        synthetic = synth_dialogues(args.synthetic_samples)
        for row in synthetic:
            dst.write(row + "\n")

    total_out = kept_real + len(synthetic)

    write_jsonl_record(
        args.log,
        build_event(
            "support_corpus_complete",
            input=str(in_path),
            output=str(out_path),
            total_input_lines=total,
            real_lines_kept=kept_real,
            synthetic_lines_added=len(synthetic),
            total_output_lines=total_out,
        ),
    )

    print(
        f"Support corpus ready: real_kept={kept_real}, synthetic_added={len(synthetic)}, total={total_out}"
    )


if __name__ == "__main__":
    main()
