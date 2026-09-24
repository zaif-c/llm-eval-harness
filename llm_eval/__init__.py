"""
LLM Evaluation Framework
========================
A lightweight framework for evaluating LLM outputs.

Modules:
    harness: Batch inference runner with retry/backoff
    scoring: Ground-truth scoring (exact, fuzzy, semantic, contains)
    judge: LLM-as-judge scoring with logprob weighting

Configuration:
    Set these in your .env file:
    - OPENAI_API_KEY: Your OpenAI API key
    - EVAL_MODEL: Default model for evals (e.g., gpt-4o-mini, gpt-4o)
    - JUDGE_MODEL: Model for LLM-judge scoring (optional, falls back to EVAL_MODEL)
    - EMBEDDING_MODEL: Model for semantic similarity (default: all-MiniLM-L6-v2)

Quick start:
    from llm_eval import HarnessConfig, batch_run, score_all, compute_metrics
    
    config = HarnessConfig()  # Uses EVAL_MODEL from .env
    results = batch_run(prompts, config)
    scored = score_all(results)
    metrics = compute_metrics(scored)
"""

# Harness exports
from llm_eval.harness import (
    HarnessConfig,
    batch_run,
    run_single,
)

# Scoring exports
from llm_eval.scoring import (
    score_exact,
    score_fuzzy,
    score_semantic,
    score_contains,
    score_all,
    compute_metrics,
    print_metrics,
    normalize_text,
)

# Judge exports
from llm_eval.judge import (
    JudgeConfig,
    Rubric,
    judge_batch,
    judge_single,
    RUBRIC_HELPFULNESS,
    RUBRIC_ACCURACY,
    RUBRIC_COHERENCE,
    RUBRIC_SAFETY,
)

__version__ = "0.1.0"
