"""
Ground-Truth Scoring Layer
==========================
Scoring functions that compare model outputs to expected answers.
Works with DataFrames produced by harness.py.

Usage:
    from scoring import score_exact, score_fuzzy, score_semantic, compute_metrics
    
    # After running harness.batch_run() with prompts that have 'expected' key:
    df = score_exact(df, normalize=True)
    df = score_fuzzy(df)
    df = score_semantic(df)
    
    metrics = compute_metrics(df)
"""

import os
import re
from typing import Optional
from functools import lru_cache

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# Text Normalization
# =============================================================================

def normalize_text(text: str, lower: bool = True, strip_punct: bool = True) -> str:
    """
    Normalize text for comparison.
    - Lowercases (optional)
    - Strips leading/trailing whitespace
    - Collapses multiple spaces to one
    - Removes punctuation (optional)
    """
    if text is None:
        return ""
    
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)  # Collapse whitespace
    
    if lower:
        text = text.lower()
    
    if strip_punct:
        text = re.sub(r'[^\w\s]', '', text)
    
    return text.strip()


# =============================================================================
# Exact Match Scoring
# =============================================================================

def score_exact(
    df: pd.DataFrame,
    output_col: str = "output",
    expected_col: str = "expected",
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Add exact match scores to DataFrame.
    
    Adds column: exact_match (1 if exact match after normalization, else 0)
    
    Args:
        df: DataFrame with output and expected columns.
        normalize: If True, normalizes both strings before comparing.
    """
    df = df.copy()
    
    def compare(row):
        output = row.get(output_col)
        expected = row.get(expected_col)
        
        if output is None or expected is None:
            return 0
        
        if normalize:
            output = normalize_text(output)
            expected = normalize_text(expected)
        
        return 1 if output == expected else 0
    
    df["exact_match"] = df.apply(compare, axis=1)
    return df


# =============================================================================
# Fuzzy Match Scoring (Token Overlap / Jaccard)
# =============================================================================

def tokenize(text: str) -> set[str]:
    """Simple whitespace tokenizer after normalization."""
    return set(normalize_text(text).split())


def jaccard_similarity(set1: set, set2: set) -> float:
    """Jaccard similarity: |intersection| / |union|"""
    if not set1 and not set2:
        return 1.0  # Both empty = perfect match
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def score_fuzzy(
    df: pd.DataFrame,
    output_col: str = "output",
    expected_col: str = "expected",
) -> pd.DataFrame:
    """
    Add fuzzy match scores to DataFrame using Jaccard similarity.
    
    Adds column: fuzzy_score (0.0 to 1.0)
    """
    df = df.copy()
    
    def compute_fuzzy(row):
        output = row.get(output_col)
        expected = row.get(expected_col)
        
        if output is None:
            return 0.0
        
        output_tokens = tokenize(str(output))
        expected_tokens = tokenize(str(expected))
        
        return jaccard_similarity(output_tokens, expected_tokens)
    
    df["fuzzy_score"] = df.apply(compute_fuzzy, axis=1)
    return df


# =============================================================================
# Semantic Similarity Scoring (Sentence Transformers)
# =============================================================================

# Lazy-load the model to avoid slow import at startup
_semantic_model = None

def _get_embedding_model_name() -> str:
    """Get embedding model from environment or use default."""
    return os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

def _get_semantic_model():
    """Lazy-load sentence transformer model."""
    global _semantic_model
    if _semantic_model is None:
        from sentence_transformers import SentenceTransformer
        model_name = _get_embedding_model_name()
        _semantic_model = SentenceTransformer(model_name)
    return _semantic_model


def score_semantic(
    df: pd.DataFrame,
    output_col: str = "output",
    expected_col: str = "expected",
) -> pd.DataFrame:
    """
    Add semantic similarity scores using sentence embeddings.
    
    Adds column: semantic_score (0.0 to 1.0, cosine similarity)
    
    Note: First call loads the model (~90MB), subsequent calls are fast.
    """
    df = df.copy()
    model = _get_semantic_model()
    
    # Batch encode for efficiency
    outputs = df[output_col].fillna("").astype(str).tolist()
    expecteds = df[expected_col].fillna("").astype(str).tolist()
    
    output_embeddings = model.encode(outputs, convert_to_numpy=True)
    expected_embeddings = model.encode(expecteds, convert_to_numpy=True)
    
    # Cosine similarity (embeddings are already normalized by default)
    # cos_sim = dot(a, b) / (||a|| * ||b||)
    # For normalized vectors: cos_sim = dot(a, b)
    similarities = np.sum(output_embeddings * expected_embeddings, axis=1)
    
    # Clip to [0, 1] (can be slightly negative for very dissimilar texts)
    similarities = np.clip(similarities, 0.0, 1.0)
    
    df["semantic_score"] = similarities
    return df


# =============================================================================
# Contains/Substring Scoring
# =============================================================================

def score_contains(
    df: pd.DataFrame,
    output_col: str = "output",
    expected_col: str = "expected",
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Check if expected answer is contained in output (useful for free-form answers).
    
    Adds column: contains_expected (1 if expected in output, else 0)
    """
    df = df.copy()
    
    def check_contains(row):
        output = row.get(output_col)
        expected = row.get(expected_col)
        
        if output is None or expected is None:
            return 0
        
        if normalize:
            output = normalize_text(output)
            expected = normalize_text(expected)
        
        return 1 if expected in output else 0
    
    df["contains_expected"] = df.apply(check_contains, axis=1)
    return df


# =============================================================================
# Aggregate Metrics
# =============================================================================

def compute_metrics(
    df: pd.DataFrame,
    score_cols: Optional[list[str]] = None,
) -> dict:
    """
    Compute aggregate metrics from scored DataFrame.
    
    Args:
        df: DataFrame with score columns.
        score_cols: Columns to aggregate. If None, auto-detects score columns.
    
    Returns:
        Dict with metrics for each score column (mean, std, min, max, count).
    """
    if score_cols is None:
        # Auto-detect: columns with 'score', 'match', or 'contains' in name
        score_cols = [c for c in df.columns 
                      if any(x in c.lower() for x in ['score', 'match', 'contains'])]
    
    metrics = {
        "total_samples": len(df),
        "successful_calls": df["error"].isna().sum() if "error" in df.columns else len(df),
    }
    
    for col in score_cols:
        if col in df.columns:
            values = df[col].dropna()
            metrics[col] = {
                "mean": round(values.mean(), 4),
                "std": round(values.std(), 4),
                "min": round(values.min(), 4),
                "max": round(values.max(), 4),
                "count": len(values),
            }
    
    return metrics


def print_metrics(metrics: dict) -> None:
    """Pretty-print metrics dict."""
    print(f"\n{'='*50}")
    print("EVALUATION METRICS")
    print(f"{'='*50}")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Successful API calls: {metrics['successful_calls']}")
    print()
    
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"{key}:")
            print(f"  mean: {value['mean']:.4f}  (std: {value['std']:.4f})")
            print(f"  range: [{value['min']:.4f}, {value['max']:.4f}]")
            print()


# =============================================================================
# Convenience: Score All
# =============================================================================

def score_all(
    df: pd.DataFrame,
    include_semantic: bool = True,
) -> pd.DataFrame:
    """
    Apply all scoring methods at once.
    
    Args:
        include_semantic: If True, includes semantic scoring (slower, loads model).
    """
    df = score_exact(df)
    df = score_fuzzy(df)
    df = score_contains(df)
    
    if include_semantic:
        df = score_semantic(df)
    
    return df


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    # Test with mock data (no API calls needed)
    test_data = pd.DataFrame([
        {"prompt_id": "q1", "output": "4", "expected": "4", "error": None},
        {"prompt_id": "q2", "output": "Paris", "expected": "paris", "error": None},
        {"prompt_id": "q3", "output": "The answer is 42.", "expected": "42", "error": None},
        {"prompt_id": "q4", "output": "I don't know", "expected": "London", "error": None},
        {"prompt_id": "q5", "output": None, "expected": "test", "error": "API Error"},
    ])
    
    print("Input data:")
    print(test_data[["prompt_id", "output", "expected"]].to_string(index=False))
    
    # Apply all scoring
    scored = score_all(test_data, include_semantic=True)
    
    print("\nScored data:")
    score_cols = ["exact_match", "fuzzy_score", "contains_expected", "semantic_score"]
    print(scored[["prompt_id", "output", "expected"] + score_cols].to_string(index=False))
    
    # Compute metrics
    metrics = compute_metrics(scored)
    print_metrics(metrics)
