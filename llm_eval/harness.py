"""
LLM Evaluation Harness
======================
Scoring-agnostic batch runner with retry/backoff.
Logs every call to a pandas DataFrame.

Usage:
    from llm_eval import HarnessConfig, batch_run
    
    # Model comes from EVAL_MODEL env var, or specify explicitly
    config = HarnessConfig()  # uses env var
    config = HarnessConfig(model="gpt-4o")  # explicit override
    
    prompts = [
        {"prompt_id": "q1", "input": "What is 2+2?"},
        {"prompt_id": "q2", "input": "Capital of France?"},
    ]
    results_df = batch_run(prompts, config)
"""

import os
import time
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, APIError, RateLimitError, APIConnectionError

load_dotenv()


def _get_default_model() -> str:
    """Get default model from environment or fall back to None (must be specified)."""
    return os.getenv("EVAL_MODEL", os.getenv("DEFAULT_MODEL", ""))


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class HarnessConfig:
    """Configuration for the inference harness."""
    model: str = ""  # Set from env or must be provided
    temperature: float = 0.0          # Deterministic by default for evals
    max_tokens: int = 1024
    system_prompt: Optional[str] = None
    
    # Retry settings
    max_retries: int = 5
    base_delay: float = 1.0           # Base delay in seconds
    max_delay: float = 60.0           # Cap on exponential backoff
    jitter: float = 0.5               # Random jitter factor (0-1)
    
    # Output settings
    output_csv: Optional[str] = None  # If set, writes results to this path
    
    def __post_init__(self):
        if not self.model:
            self.model = _get_default_model()
        if not self.model:
            raise ValueError(
                "Model must be specified via HarnessConfig(model=...) or "
                "EVAL_MODEL/DEFAULT_MODEL environment variable"
            )


# =============================================================================
# Single-call inference with retry
# =============================================================================

def run_single(
    client: OpenAI,
    prompt_id: str,
    input_text: str,
    config: HarnessConfig,
) -> dict:
    """
    Run inference on a single prompt with exponential backoff retry.
    
    Returns dict with: prompt_id, input, output, latency_ms, timestamp, 
                       model, error (None if success)
    """
    messages = []
    if config.system_prompt:
        messages.append({"role": "system", "content": config.system_prompt})
    messages.append({"role": "user", "content": input_text})
    
    last_error = None
    
    for attempt in range(config.max_retries):
        try:
            start_time = time.perf_counter()
            
            response = client.chat.completions.create(
                model=config.model,
                messages=messages,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            return {
                "prompt_id": prompt_id,
                "input": input_text,
                "output": response.choices[0].message.content,
                "latency_ms": round(latency_ms, 2),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": response.model,
                "error": None,
            }
            
        except (RateLimitError, APIConnectionError, APIError) as e:
            last_error = str(e)
            
            if attempt < config.max_retries - 1:
                # Exponential backoff: base * 2^attempt, capped at max_delay
                delay = min(config.base_delay * (2 ** attempt), config.max_delay)
                # Add jitter: delay * (1 ± jitter/2)
                jitter_range = delay * config.jitter
                delay += random.uniform(-jitter_range / 2, jitter_range / 2)
                delay = max(0.1, delay)  # Floor at 100ms
                
                print(f"  [Retry {attempt + 1}/{config.max_retries - 1}] "
                      f"{prompt_id}: {type(e).__name__}, waiting {delay:.1f}s")
                time.sleep(delay)
    
    # All retries exhausted
    return {
        "prompt_id": prompt_id,
        "input": input_text,
        "output": None,
        "latency_ms": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": config.model,
        "error": last_error,
    }


# =============================================================================
# Batch runner
# =============================================================================

def batch_run(
    prompts: list[dict],
    config: HarnessConfig,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run inference on a batch of prompts.
    
    Args:
        prompts: List of dicts, each with 'prompt_id' and 'input' keys.
                 Can include extra keys (e.g., 'expected') - they'll pass through.
        config: HarnessConfig instance.
        verbose: Print progress if True.
    
    Returns:
        DataFrame with columns: prompt_id, input, output, latency_ms, 
                                timestamp, model, error, plus any extra keys from input.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    results = []
    total = len(prompts)
    
    if verbose:
        print(f"Running {total} prompts with model={config.model}")
        print("-" * 50)
    
    for i, prompt in enumerate(prompts):
        prompt_id = prompt["prompt_id"]
        input_text = prompt["input"]
        
        if verbose:
            preview = input_text[:50] + "..." if len(input_text) > 50 else input_text
            print(f"[{i + 1}/{total}] {prompt_id}: {preview}")
        
        result = run_single(client, prompt_id, input_text, config)
        
        # Carry forward any extra keys from the input (e.g., 'expected', 'category')
        for key, value in prompt.items():
            if key not in result:
                result[key] = value
        
        results.append(result)
    
    df = pd.DataFrame(results)
    
    # Reorder columns: standard cols first, then extras
    standard_cols = ["prompt_id", "input", "output", "latency_ms", "timestamp", "model", "error"]
    extra_cols = [c for c in df.columns if c not in standard_cols]
    df = df[standard_cols + extra_cols]
    
    if verbose:
        print("-" * 50)
        success_count = df["error"].isna().sum()
        print(f"Complete: {success_count}/{total} succeeded")
        if df["latency_ms"].notna().any():
            print(f"Latency: mean={df['latency_ms'].mean():.0f}ms, "
                  f"p50={df['latency_ms'].median():.0f}ms, "
                  f"p95={df['latency_ms'].quantile(0.95):.0f}ms")
    
    # Write to CSV if configured
    if config.output_csv:
        df.to_csv(config.output_csv, index=False)
        if verbose:
            print(f"Saved to {config.output_csv}")
    
    return df


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    # Minimal smoke test (uses EVAL_MODEL from .env)
    config = HarnessConfig(
        output_csv="test_run.csv",
    )
    
    test_prompts = [
        {"prompt_id": "test_1", "input": "What is 2+2? Answer with just the number."},
        {"prompt_id": "test_2", "input": "Capital of France? Answer with just the city name."},
    ]
    
    df = batch_run(test_prompts, config)
    print("\nResults preview:")
    print(df[["prompt_id", "output", "latency_ms"]].to_string(index=False))
