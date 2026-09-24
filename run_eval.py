"""
Run Evaluation Pipeline
=======================
Main entry point for running LLM evaluation experiments.

WHAT THIS FILE IS AND IS NOT
----------------------------
This is orchestration only. It parses flags, loads the dataset, decides which
scoring layers apply, and writes the artifacts. Every actual behavior lives in
llm_eval/: inference in harness.py, reference-based metrics in scoring.py,
reference-free metrics in judge.py. Keeping the logic out of here is what lets
the same library be driven from a notebook when the CLI shape does not fit.

THE CENTRAL BRANCH
------------------
One decision drives the whole run, made at `ran_ground_truth` below:

    dataset has a usable `expected` column?
        yes -> scoring.score_all         (exact match, F1, BLEU, ROUGE, ...)
        no  -> nothing, unless --judge
    --judge passed?
        yes -> judge.judge_batch         (rubric + logprob-weighted score)

The two are not exclusive. On a labeled dataset with --judge, both run, and the
judge additionally receives the gold answer as a reference. That combination is
what makes judge-vs-ground-truth agreement measurable.

ARTIFACTS WRITTEN (all prefixed with the run id)
------------------------------------------------
    <run_id>_raw.csv            inference output, written by batch_run before
                                any scoring, so a scoring crash never costs calls
    <run_id>_raw.csv.jsonl      per-row inference checkpoint (crash/resume)
    <run_id>_judge.jsonl        per-row judgment checkpoint (crash/resume)
    <run_id>_scored.csv         raw plus every score column
    <run_id>_metrics.json       the aggregate dict, written on every run

Usage:
    python run_eval.py
    python run_eval.py --dataset datasets/ground_truth_demo.json --model gpt-4o-mini

    # With different scoring options:
    python run_eval.py --dataset data.json --model gpt-4o --no-semantic

    # Resume from previous run (scoring only):
    python run_eval.py --results results/run_123.csv --score-only

    # Open-ended judge (no ground truth required):
    python run_eval.py --dataset datasets/open_ended_demo.json --judge
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Called before the llm_eval import below so that module-level environment
# reads inside the package see a populated os.environ.
load_dotenv()

from llm_eval import (
    HarnessConfig,
    JudgeConfig,
    RUBRIC_ACCURACY,
    RUBRIC_COHERENCE,
    RUBRIC_HELPFULNESS,
    RUBRIC_SAFETY,
    batch_run,
    compute_metrics,
    find_disagreements,
    find_unstable_prompts,
    judge_agreement,
    judge_batch,
    print_agreement,
    print_disagreements,
    print_metrics,
    print_variance,
    score_all,
    variance_report,
)

# Maps the --judge-rubric flag value to the Rubric object. The dict is also the
# source of the argparse `choices` list, so adding a rubric here automatically
# makes it selectable and validated without a second edit.
RUBRICS = {
    "helpfulness": RUBRIC_HELPFULNESS,
    "accuracy": RUBRIC_ACCURACY,
    "coherence": RUBRIC_COHERENCE,
    "safety": RUBRIC_SAFETY,
}


def _get_default_model() -> str:
    """Get default model from environment."""
    return os.getenv("EVAL_MODEL", os.getenv("DEFAULT_MODEL", ""))


def load_dataset(path: str) -> list[dict]:
    """Load dataset from JSON or CSV file.

    Both formats produce the same thing: a list of dicts, one per prompt. Each
    needs `prompt_id` and `input`; anything else (`expected`, `category`, or
    arbitrary metadata) is carried through by batch_run onto the result row.

    Two formats because the dataset arrives in whatever shape it arrives in, and
    converting it by hand under time pressure is exactly the kind of avoidable
    step this harness exists to remove.
    """
    path = Path(path)

    if path.suffix == ".json":
        with open(path) as f:
            return json.load(f)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
        return df.to_dict(orient="records")
    else:
        # Explicit rather than silently trying JSON, so a .jsonl or .tsv file
        # fails with a clear message instead of a confusing parse error.
        raise ValueError(f"Unsupported file format: {path.suffix}")


def generate_run_id() -> str:
    """Generate a unique run ID based on timestamp.

    Timestamp rather than a random id so runs sort chronologically in the
    results directory. Overridable with --run-id, which is what makes --resume
    usable: resuming requires naming the same run whose checkpoints you want.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main():
    parser = argparse.ArgumentParser(description="Run LLM evaluation pipeline")

    # ---- Data options -------------------------------------------------------
    parser.add_argument(
        "--dataset",
        type=str,
        # Defaulted so a bare `python run_eval.py` is a working smoke test.
        default="datasets/ground_truth_demo.json",
        help="Path to dataset file (JSON or CSV). Default: datasets/ground_truth_demo.json",
    )
    parser.add_argument("--results", type=str, help="Path to existing results CSV (for score-only mode)")

    # ---- Model options ------------------------------------------------------
    parser.add_argument(
        "--model",
        type=str,
        # None, not "", so the env-var fallback inside HarnessConfig owns the
        # default rather than it being duplicated here.
        default=None,
        help="OpenRouter model id (default: EVAL_MODEL). "
             "Examples: openai/gpt-4o-mini, anthropic/claude-sonnet-4, google/gemini-2.5-flash",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens per response")
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-request timeout in seconds. A timeout is retried like any transient error.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Concurrent inference calls. 1 runs sequentially. Default: 8.",
    )
    parser.add_argument("--system-prompt", type=str, default=None, help="System prompt to use")
    parser.add_argument(
        "--cot",
        action="store_true",
        help="Ask the model to reason step by step and end with 'ANSWER: ...'. "
             "Metrics compare that line to the expected answer, not the reasoning.",
    )

    # ---- Scoring options ----------------------------------------------------
    parser.add_argument("--no-semantic", action="store_true", help="Skip semantic similarity scoring")
    parser.add_argument("--score-only", action="store_true", help="Only run scoring on existing results")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse rows already in the run's .jsonl checkpoint instead of re-calling the API. "
             "Requires the same --run-id as the interrupted run.",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Score each output with the LLM judge. Works with or without an expected column.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="OpenRouter id for the judge (default: JUDGE_MODEL, else EVAL_MODEL).",
    )
    parser.add_argument(
        "--judge-rubric",
        # Derived from the RUBRICS dict, so argparse rejects an unknown name up
        # front rather than KeyError-ing after inference has already been paid for.
        choices=sorted(RUBRICS),
        default="helpfulness",
        help="Prebuilt rubric. Default: helpfulness.",
    )
    parser.add_argument(
        "--no-logprobs",
        action="store_true",
        help="Judge from the parsed SCORE line only, skipping the probability-weighted score.",
    )

    # ---- Output options -----------------------------------------------------
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--run-id", type=str, default=None, help="Custom run ID (default: timestamp)")

    args = parser.parse_args()

    # Validate arguments. Checked here rather than later because --score-only
    # without --results has nothing to score, and failing at once is cheaper
    # than failing after setup.
    if args.score_only and not args.results:
        parser.error("--results is required when using --score-only")

    # Setup output directory. parents=True so a nested --output-dir works;
    # exist_ok=True so a second run into the same directory is not an error.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id or generate_run_id()

    print(f"\n{'='*60}")
    print(f"LLM EVALUATION RUN: {run_id}")
    print(f"{'='*60}\n")

    # =========================================================================
    # STEP 1: get a DataFrame, either by loading one or by running inference
    # =========================================================================
    if args.score_only:
        # Skips inference entirely and rescores a finished run. This is the loop
        # to use when iterating on a metric or a rubric: the expensive part is
        # already on disk, so iteration costs nothing.
        print(f"Loading existing results from: {args.results}")
        df = pd.read_csv(args.results)
        print(f"Loaded {len(df)} results\n")
    else:
        # Load dataset
        print(f"Dataset: {args.dataset}")
        prompts = load_dataset(args.dataset)
        print(f"Loaded {len(prompts)} prompts\n")

        # Configure harness (model falls back to EVAL_MODEL env var if not provided)
        config = HarnessConfig(
            model=args.model or "",  # Empty string triggers env var fallback
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            max_workers=args.max_workers,
            system_prompt=args.system_prompt,
            chain_of_thought=args.cot,
            # Setting output_csv also derives the checkpoint path
            # (<output_csv>.jsonl) inside HarnessConfig.__post_init__.
            output_csv=str(output_dir / f"{run_id}_raw.csv"),
            resume=args.resume,
        )

        # Echoed back before any calls are made. Under time pressure this is the
        # cheapest guard against running 200 prompts against the wrong model or
        # with --cot silently off. Reads config.*, not args.*, so it shows the
        # resolved values including the env-var fallback.
        print(f"Model: {config.model}")
        print(f"Temperature: {config.temperature}")
        print(f"Max tokens: {config.max_tokens}")
        print(f"Timeout: {config.timeout}s")
        print(f"Max workers: {config.max_workers}")
        if config.system_prompt:
            print(f"System prompt: {config.system_prompt[:50]}...")
        print(f"Chain of thought: {config.chain_of_thought}")
        print()

        # Run inference. Writes <run_id>_raw.csv itself before returning.
        df = batch_run(prompts, config)

    # =========================================================================
    # STEP 2: ground-truth scoring, if the dataset is labeled
    # =========================================================================
    # Both halves of this condition matter. The column can exist but be entirely
    # null (a CSV with an empty `expected` column, or an open-ended set that was
    # given the column for consistency), and scoring against nulls would produce
    # a frame of confident zeros rather than an obvious absence.
    ran_ground_truth = "expected" in df.columns and df["expected"].notna().any()
    if ran_ground_truth:
        print("\n" + "="*60)
        print("SCORING")
        print("="*60 + "\n")
        df = score_all(df, include_semantic=not args.no_semantic)
    elif not args.judge:
        # Only warned when nothing else will score either. With --judge this is
        # the normal open-ended path, not a problem worth flagging.
        print("\nNo 'expected' column found - skipping ground-truth scoring.")
        print("Pass --judge to score open-ended outputs.")

    # =========================================================================
    # STEP 3: LLM judge, if asked for
    # =========================================================================
    if args.judge:
        print("\n" + "="*60)
        print("LLM JUDGE")
        print("="*60 + "\n")
        judge_config = JudgeConfig(
            model=args.judge_model or "",
            rubric=RUBRICS[args.judge_rubric],
            use_logprobs=not args.no_logprobs,
            # Shares --max-workers and --timeout with the inference pass, since
            # both are talking to the same provider under the same rate limits.
            max_workers=args.max_workers,
            timeout=args.timeout,
            # A separate checkpoint file from the inference one: they hold
            # different row shapes and are resumed independently.
            checkpoint_path=str(output_dir / f"{run_id}_judge.jsonl"),
            resume=args.resume,
        )
        # Hand the judge the gold answer when there is one. This is the
        # reference-assisted mode; on open-ended data it stays None and the
        # judge grades on the rubric alone.
        reference_col = "expected" if ran_ground_truth else None
        df = judge_batch(df, judge_config, reference_col=reference_col)

    # =========================================================================
    # STEP 4: persist and report
    # =========================================================================
    # The scored CSV only exists if something actually scored. The raw CSV was
    # already written by batch_run, so nothing is lost when this is skipped.
    if ran_ground_truth or args.judge:
        scored_path = output_dir / f"{run_id}_scored.csv"
        df.to_csv(scored_path, index=False)
        print(f"\nScored results saved to: {scored_path}")

    # Operational metrics (failure rate, truncation, tokens, latency) do not
    # depend on scoring, so report them even for an unscored run.
    metrics = compute_metrics(df)

    # Meta-evaluation: only possible when both scorers ran on the same rows.
    # Computed before the JSON is written so the agreement report ships inside
    # the metrics file rather than existing only as terminal output.
    # No flag guards this — when both scorers have already run, the comparison
    # is free, and it is the single most useful output of the whole pipeline to
    # have by default rather than behind a flag someone forgets.
    agreement = None
    if ran_ground_truth and args.judge:
        agreement = judge_agreement(df)
        if agreement:
            metrics["judge_agreement"] = agreement

    # Variance no-ops on this branch: --n-samples was cut, so frames have no
    # sample_index column and variance_report returns None. Left in place until
    # the analysis module is reviewed.
    variance = variance_report(df)
    if variance:
        metrics["variance"] = variance

    print_metrics(metrics)

    # No numpy conversion step here: compute_metrics returns plain Python types
    # so the dict is serializable on its own.
    metrics_path = output_dir / f"{run_id}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    if variance:
        print_variance(variance)
        # The specific prompts that moved. An aggregate spread says the run is
        # unstable; this says which questions to go read.
        score_col = "exact_match" if "exact_match" in df.columns else "judge_score"
        unstable = find_unstable_prompts(df, score_col=score_col)
        if not unstable.empty:
            print(f"{'-'*50}")
            print(f"LEAST STABLE PROMPTS (by {score_col} spread)")
            print(f"{'-'*50}")
            print(unstable.to_string(index=False))
            print()

    if agreement:
        print_agreement(agreement)
        # The individual conflicting rows, printed for immediate reading and
        # written out in full because the printed table truncates long text and
        # these rows are the ones worth reading verbatim.
        disagreements = find_disagreements(df)
        if not disagreements.empty:
            print_disagreements(disagreements)
            disagreements_path = output_dir / f"{run_id}_disagreements.csv"
            disagreements.to_csv(disagreements_path, index=False)
            print(f"Disagreements saved to: {disagreements_path}")

    # =========================================================================
    # STEP 5: per-category breakdown, when the dataset provides categories
    # =========================================================================
    # Gated on scoring having run, since there would be nothing to break down
    # otherwise. This is the cheapest real analysis in the pipeline: an aggregate
    # score of 70% means something quite different if it is uniform across
    # categories versus 100% on three of them and 10% on a fourth.
    if (ran_ground_truth or args.judge) and "category" in df.columns:
        print("\n" + "-"*50)
        print("PER-CATEGORY BREAKDOWN")
        print("-"*50)
        for category in df["category"].unique():
            cat_df = df[df["category"] == category]
            # compute_metrics is reused on the slice rather than recomputing
            # means by hand, so a category figure is defined identically to the
            # overall figure, including the exclusion of failed rows.
            cat_metrics = compute_metrics(cat_df)
            parts = [f"n={len(cat_df)}"]
            # Chained .get() so a run missing either scorer still prints the
            # other instead of raising.
            exact = cat_metrics.get("exact_match", {}).get("mean")
            judged = cat_metrics.get("judge_score", {}).get("mean")
            if exact is not None:
                parts.append(f"exact_match={exact:.2%}")
            if judged is not None:
                parts.append(f"judge_score={judged:.2f}")
            print(f"  {category}: {', '.join(parts)}")


    print(f"\n{'='*60}")
    print("RUN COMPLETE")
    print(f"{'='*60}\n")

    # Returned so this is importable and callable from a notebook, where having
    # the frame back matters more than the printed output.
    return df


if __name__ == "__main__":
    main()
