"""
LLM Evaluation Framework
========================
A lightweight framework for evaluating LLM outputs.

This file is the public surface of the package. Everything re-exported below
can be imported as `from llm_eval import X` regardless of which module it
actually lives in, so callers (run_eval.py, notebooks) never depend on the
internal file layout. Anything not listed here is an implementation detail and
can move without breaking a caller.

Modules, in pipeline order:
    harness:  Batch inference runner with retry/backoff
    scoring:  Ground-truth scoring (exact, token F1, BLEU, ROUGE, char similarity, semantic)
    judge:    LLM-as-judge scoring with logprob weighting

The dependency direction is one-way: judge.py imports from harness.py (for the
shared retry policy and checkpoint), and nothing imports back down into
harness.py. That is what keeps the inference layer unaware of how its output
will be scored.

Configuration:
    Set these in your .env file:
    - OPENROUTER_API_KEY: Your OpenRouter API key
    - EVAL_MODEL: OpenRouter model id. The prefix selects the provider
      (openai/gpt-4o-mini, anthropic/claude-sonnet-4, google/gemini-2.5-flash)
    - JUDGE_MODEL: Model for LLM-judge scoring (optional, falls back to EVAL_MODEL)
    - EMBEDDING_MODEL: Model for semantic similarity (default: all-MiniLM-L6-v2)

Quick start:
    from llm_eval import HarnessConfig, batch_run, score_all, compute_metrics

    config = HarnessConfig()  # Uses EVAL_MODEL from .env
    results = batch_run(prompts, config)
    scored = score_all(results)
    metrics = compute_metrics(scored)
"""

# Harness exports.
# Checkpoint, call_with_retry, and classify_error are exported not because the
# CLI needs them but because they are the pieces most likely to be poked at
# directly when debugging a run or writing a one-off script.
from llm_eval.harness import (
    HarnessConfig,
    Checkpoint,
    batch_run,
    call_with_retry,
    classify_error,
    run_single,
)

# Scoring exports. Each score_* function is exported individually as well as via
# score_all, so a single metric can be applied on its own when only one is
# relevant to the task at hand.
from llm_eval.scoring import (
    score_exact,
    score_fuzzy,
    score_semantic,
    score_contains,
    score_tokens,
    score_bleu,
    score_rouge,
    score_char,
    score_all,
    compute_metrics,
    print_metrics,
    score_histogram,
    normalize_text,
)

# Judge exports. Rubric is exported alongside the prebuilt ones because writing
# a task-specific rubric is the expected customization, not an advanced case.
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
