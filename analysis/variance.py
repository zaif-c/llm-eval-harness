"""
Sample variance, standalone
===========================
Point this at a scored CSV from an --n-samples run and get the variance report
without re-running anything.

    python analysis/variance.py results/<run_id>_scored.csv
    python analysis/variance.py results/<run_id>_scored.csv --score-col judge_score
    python analysis/variance.py results/<run_id>_scored.csv --show-outputs

The question this answers: would the headline number survive a rerun? Until it
is measured, "model A scored 0.72, model B scored 0.75" is an unfalsifiable
claim. After it is measured, that 0.03 gap is either outside the run-to-run
spread or it is noise, and which one it is changes the conclusion entirely.

Unlike judge agreement, variance cannot be recovered after the fact from a
single-sample run — the extra calls have to have been made. This script is for
re-reading a sampled run from a different angle, not for creating one.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_eval import (  # noqa: E402
    find_unstable_prompts,
    print_variance,
    variance_report,
)


def main():
    parser = argparse.ArgumentParser(
        description="Score variance across repeated samples of the same prompts"
    )
    parser.add_argument("results", help="Path to a *_scored.csv from an --n-samples run")
    parser.add_argument(
        "--score-col", default="exact_match",
        help="Metric to rank unstable prompts by. Default: exact_match.",
    )
    parser.add_argument(
        "--top", type=int, default=10,
        help="How many unstable prompts to list. Default: 10.",
    )
    parser.add_argument(
        "--show-outputs", action="store_true",
        help="Print the differing outputs for each unstable prompt side by side. "
             "This is where the reason for the instability usually becomes obvious.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    print(f"Loaded {len(df)} rows from {args.results}")

    report = variance_report(df)
    if not report:
        print(
            "\nNothing to measure. Variance needs more than one sample per prompt:\n"
            "  python run_eval.py --dataset <your data> --n-samples 3\n"
        )
        return

    print_variance(report)

    unstable = find_unstable_prompts(df, score_col=args.score_col, top_n=args.top)
    if unstable.empty:
        print(f"No prompt's {args.score_col} moved across samples.")
        return

    print(f"{'-'*50}")
    print(f"LEAST STABLE PROMPTS (by {args.score_col} spread)")
    print(f"{'-'*50}")
    print(unstable.to_string(index=False))
    print()

    # The qualitative half. An unstable prompt is nearly always unstable for a
    # nameable reason, and the reason is visible in the differing text rather
    # than in the standard deviation.
    if args.show_outputs and "output" in df.columns:
        for prompt_id in unstable["prompt_id"]:
            rows = df[df["prompt_id"] == prompt_id]
            print(f"{'='*50}")
            print(f"{prompt_id}")
            if "expected" in rows.columns:
                print(f"  expected: {rows['expected'].iloc[0]}")
            print(f"{'='*50}")
            for _, row in rows.iterrows():
                text = str(row.get("output", "")).replace("\n", " ")[:160]
                score = row.get(args.score_col)
                print(f"  [sample {row.get('sample_index')}] {args.score_col}={score}")
                print(f"    {text}")
            print()


if __name__ == "__main__":
    main()
