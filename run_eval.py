"""
Run Evaluation Pipeline
=======================
Main entry point for running LLM evaluation experiments.

Usage:
    python run_eval.py --dataset datasets/ground_truth_demo.json --model gpt-4o-mini

    # With different scoring options:
    python run_eval.py --dataset data.json --model gpt-4o --no-semantic
    
    # Resume from previous run (scoring only):
    python run_eval.py --results results/run_123.csv --score-only
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from llm_eval import HarnessConfig, batch_run, score_all, compute_metrics, print_metrics


def _get_default_model() -> str:
    """Get default model from environment."""
    return os.getenv("EVAL_MODEL", os.getenv("DEFAULT_MODEL", ""))


def load_dataset(path: str) -> list[dict]:
    """Load dataset from JSON or CSV file."""
    path = Path(path)
    
    if path.suffix == ".json":
        with open(path) as f:
            return json.load(f)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
        return df.to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def generate_run_id() -> str:
    """Generate a unique run ID based on timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main():
    parser = argparse.ArgumentParser(description="Run LLM evaluation pipeline")
    
    # Data options
    parser.add_argument("--dataset", type=str, help="Path to dataset file (JSON or CSV)")
    parser.add_argument("--results", type=str, help="Path to existing results CSV (for score-only mode)")
    
    # Model options
    parser.add_argument("--model", type=str, default=None, help="Model to evaluate (default: from EVAL_MODEL env var)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens per response")
    parser.add_argument("--system-prompt", type=str, default=None, help="System prompt to use")
    
    # Scoring options
    parser.add_argument("--no-semantic", action="store_true", help="Skip semantic similarity scoring")
    parser.add_argument("--score-only", action="store_true", help="Only run scoring on existing results")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--run-id", type=str, default=None, help="Custom run ID (default: timestamp)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.score_only:
        if not args.results:
            parser.error("--results is required when using --score-only")
    else:
        if not args.dataset:
            parser.error("--dataset is required")
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    run_id = args.run_id or generate_run_id()
    
    print(f"\n{'='*60}")
    print(f"LLM EVALUATION RUN: {run_id}")
    print(f"{'='*60}\n")
    
    # Either load existing results or run inference
    if args.score_only:
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
            system_prompt=args.system_prompt,
            output_csv=str(output_dir / f"{run_id}_raw.csv"),
        )
        
        print(f"Model: {config.model}")
        print(f"Temperature: {config.temperature}")
        print(f"Max tokens: {config.max_tokens}")
        if config.system_prompt:
            print(f"System prompt: {config.system_prompt[:50]}...")
        print()
        
        # Run inference
        df = batch_run(prompts, config)
    
    # Check if we have expected values for scoring
    if "expected" in df.columns:
        print("\n" + "="*60)
        print("SCORING")
        print("="*60 + "\n")
        
        # Apply scoring
        include_semantic = not args.no_semantic
        df = score_all(df, include_semantic=include_semantic)
        
        # Save scored results
        scored_path = output_dir / f"{run_id}_scored.csv"
        df.to_csv(scored_path, index=False)
        print(f"\nScored results saved to: {scored_path}")
        
        # Compute and display metrics
        metrics = compute_metrics(df)
        print_metrics(metrics)
        
        # Save metrics to JSON (convert numpy types to native Python)
        def convert_numpy(obj):
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            return obj
        
        metrics_path = output_dir / f"{run_id}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(convert_numpy(metrics), f, indent=2)
        print(f"Metrics saved to: {metrics_path}")
        
        # Per-category breakdown if 'category' column exists
        if "category" in df.columns:
            print("\n" + "-"*50)
            print("PER-CATEGORY BREAKDOWN")
            print("-"*50)
            for category in df["category"].unique():
                cat_df = df[df["category"] == category]
                cat_metrics = compute_metrics(cat_df)
                exact_mean = cat_metrics.get("exact_match", {}).get("mean", "N/A")
                print(f"  {category}: exact_match={exact_mean:.2%} (n={len(cat_df)})")
    else:
        print("\nNo 'expected' column found - skipping scoring.")
        print("Raw results saved.")
    
    print(f"\n{'='*60}")
    print("RUN COMPLETE")
    print(f"{'='*60}\n")
    
    return df


if __name__ == "__main__":
    main()
