"""
Judge agreement, standalone
===========================
Point this at any scored CSV and get the full judge-vs-ground-truth report,
without re-running inference or paying for a single API call.

    python analysis/judge_agreement.py results/<run_id>_scored.csv
    python analysis/judge_agreement.py results/<run_id>_scored.csv --top 15
    python analysis/judge_agreement.py results/<run_id>_scored.csv --gt-col token_f1

run_eval.py already prints this at the end of a labeled --judge run. This script
exists for the iteration loop afterwards: changing which ground-truth column to
compare against, pulling more disagreement rows, or slicing by category are all
things worth doing several times while reading a result, and none of them should
cost another run.

Everything numeric here comes from llm_eval.analysis. This file only handles
argument parsing, the per-category slice, and output, so there is exactly one
implementation of the statistics and it is the same one the pipeline uses.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allows `python analysis/judge_agreement.py` from the repo root without
# installing the package or setting PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_eval import (  # noqa: E402
    find_disagreements,
    judge_agreement,
    print_agreement,
    print_disagreements,
)


def main():
    parser = argparse.ArgumentParser(
        description="Judge vs ground truth agreement on a scored results CSV"
    )
    parser.add_argument("results", help="Path to a *_scored.csv from a --judge run")
    parser.add_argument(
        "--gt-col",
        default="exact_match",
        help="Ground-truth column to rank disagreements against. Default: exact_match.",
    )
    parser.add_argument(
        "--top", type=int, default=10,
        help="How many disagreement rows to show. Default: 10.",
    )
    parser.add_argument(
        "--by-category", action="store_true",
        help="Also report agreement separately within each category. "
             "Agreement that holds overall can still break down on one slice.",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Write the disagreement rows next to the input CSV.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    print(f"Loaded {len(df)} rows from {args.results}")

    report = judge_agreement(df)
    if not report:
        # The common causes are all user-fixable, so name them rather than
        # failing with an empty report.
        print(
            "\nNo agreement to compute. This needs a run where BOTH scorers "
            "produced columns:\n"
            "  - a judge_score column (run with --judge)\n"
            "  - at least one ground-truth column (dataset needs `expected`)\n"
        )
        return

    print_agreement(report)

    disagreements = find_disagreements(df, gt_col=args.gt_col, top_n=args.top)
    if disagreements.empty:
        print(f"No disagreements against {args.gt_col}: the two scorers agree on every row.")
    else:
        print_disagreements(disagreements)
        if args.save:
            out = Path(args.results).with_name(
                Path(args.results).stem.replace("_scored", "") + "_disagreements.csv"
            )
            disagreements.to_csv(out, index=False)
            print(f"Saved to: {out}")

    # Per-category agreement. An overall AUC of 0.9 can hide a category where
    # the judge is at chance, and that breakdown is usually the actual finding
    # rather than the headline number.
    if args.by_category and "category" in df.columns:
        print(f"\n{'='*50}")
        print("AGREEMENT BY CATEGORY")
        print(f"{'='*50}")
        for category in df["category"].unique():
            slice_df = df[df["category"] == category]
            slice_report = judge_agreement(slice_df)
            if not slice_report:
                print(f"\n  {category}: not computable (n={len(slice_df)})")
                continue
            # Reads the same default column the pipeline leads with, so the
            # per-category line is comparable to the headline number.
            entry = slice_report["comparisons"].get("exact_match", {})
            bits = [f"n={len(slice_df)}"]
            if entry.get("separation") is not None:
                bits.append(f"separation={entry['separation']:+.2f}")
            if entry.get("auc") is not None:
                bits.append(f"auc={entry['auc']:.3f}")
            if entry.get("spearman") is not None:
                bits.append(f"spearman={entry['spearman']:+.3f}")
            print(f"  {category}: {', '.join(bits)}")


if __name__ == "__main__":
    main()
