"""
Ground-Truth Scoring Layer
==========================
Scoring functions that compare model outputs to expected answers.
Works with DataFrames produced by harness.py.

WHERE THIS FILE SITS IN THE PIPELINE
------------------------------------
    harness.py  ->  [DataFrame]  ->  scoring.py  ->  [same DataFrame + score columns]

This is the *reference-based* scoring layer: every metric in here needs a known
correct answer to compare against, which it reads from the `expected` column
that harness.py carried through from the dataset. The sibling layer, judge.py,
is the reference-free path for open-ended prompts where no `expected` exists.

Two conventions hold throughout this file and are worth internalizing:

  1. Every `score_*` function takes a DataFrame and returns a *new* DataFrame
     with one or more columns added. They never mutate the caller's frame
     (hence `df = df.copy()` at the top of each) and they never drop rows. That
     makes them composable in any order, which is all `score_all` is doing.

  2. Every metric lands on a 0-1 scale where 1 is better. Mixing scales would
     make the summary block unreadable and would make a mean across metrics
     meaningless. This is why negative cosine is floored at 0 in score_semantic.

Why hand-rolled BLEU/ROUGE rather than sacrebleu or rouge-score: the exact
variant of these metrics matters when comparing to a published number, and
having the implementation in the repo means the variant is knowable rather than
assumed. The tradeoff is stated plainly in FRAMEWORK.md.

Usage:
    from scoring import score_exact, score_fuzzy, score_semantic, compute_metrics

    # After running harness.batch_run() with prompts that have 'expected' key:
    df = score_exact(df, normalize=True)
    df = score_fuzzy(df)
    df = score_semantic(df)

    metrics = compute_metrics(df)
"""

import math
import os
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Optional
from functools import lru_cache

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Needed for EMBEDDING_MODEL, read lazily in _get_embedding_model_name.
load_dotenv()


# =============================================================================
# Text Normalization
#
# Every string metric below runs through this first. Centralizing it means the
# metrics differ only in *how they compare*, not in what they consider the same
# string, so a disagreement between two metrics is a real signal rather than an
# artifact of one of them being case-sensitive.
# =============================================================================

def normalize_text(text: str, lower: bool = True, strip_punct: bool = True) -> str:
    """
    Normalize text for comparison.
    - Lowercases (optional)
    - Strips leading/trailing whitespace
    - Collapses multiple spaces to one
    - Removes punctuation (optional)

    The order of operations matters. Whitespace is collapsed *before* punctuation
    is stripped, and the result is stripped again at the end, because removing
    punctuation can leave behind new leading/trailing spaces ("Paris." -> "Paris"
    is clean, but "- Paris" -> " Paris" is not).

    This normalization is deliberately aggressive. It will call "4" and "4."
    equal, and "Paris" and "paris" equal, which is almost always what you want
    when grading a short answer. It will also call "$100" and "100" equal, which
    occasionally is not — worth knowing before trusting exact_match on a dataset
    where punctuation is semantically load-bearing.
    """
    # None-safe rather than raising, because a failed API call leaves a null in
    # the output column and every scorer would otherwise need its own guard.
    if text is None:
        return ""

    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)  # Collapse whitespace (incl. newlines/tabs)

    if lower:
        text = text.lower()

    if strip_punct:
        # \w is [a-zA-Z0-9_], so this keeps letters, digits, and underscores
        # and drops everything else that is not whitespace.
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

    This is the strictest metric here and the one to trust most when the dataset
    has short, unambiguous answers. Its mean is a pass rate, not an average —
    worth saying that way in a presentation, since "mean exact_match 0.7" and
    "70% correct" are the same number but the second is the honest phrasing.

    Args:
        df: DataFrame with output and expected columns.
        normalize: If True, normalizes both strings before comparing.
    """
    df = df.copy()

    def compare(row):
        output = row.get(output_col)
        expected = row.get(expected_col)

        # A failed call (output None) or an unlabeled row (expected None) scores
        # 0 here. For failed calls that 0 is later overwritten with NaN by
        # score_all, so it does not drag the mean down.
        if output is None or expected is None:
            return 0

        if normalize:
            output = normalize_text(output)
            expected = normalize_text(expected)

        return 1 if output == expected else 0

    # axis=1 applies the function per row rather than per column.
    df["exact_match"] = df.apply(compare, axis=1)
    return df


# =============================================================================
# Fuzzy Match Scoring (Token Overlap / Jaccard)
# =============================================================================

def tokenize(text: str) -> set[str]:
    """Simple whitespace tokenizer after normalization.

    Returns a *set*, so duplicates and word order are both discarded. That is
    what makes the Jaccard score below order-insensitive, and it is also its
    main limitation versus score_tokens, which uses a Counter and keeps counts.
    """
    return set(normalize_text(text).split())


def jaccard_similarity(set1: set, set2: set) -> float:
    """Jaccard similarity: |intersection| / |union|"""
    if not set1 and not set2:
        return 1.0  # Both empty = perfect match
    # Handled separately from the division because the union would be non-zero
    # while the intersection is empty; the formula would return 0.0 anyway, but
    # the early return documents the intent.
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

    Useful as the "partially right" companion to exact_match: a model that says
    "The capital is Paris" against gold "Paris" scores exact_match 0 but
    fuzzy_score 0.25. Note that Jaccard penalizes extra words symmetrically with
    missing ones, which is why score_tokens (precision/recall split) is usually
    the more diagnostic of the two.
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
#
# The only metric in this file that is not pure string manipulation. It is also
# the only one with a real startup cost, which is why it is lazy-loaded and why
# --no-semantic exists as an escape hatch for a fast iteration loop.
# =============================================================================

# Lazy-load the model to avoid slow import at startup. Module-level so the
# ~90MB model is loaded once per process and reused across every call.
_semantic_model = None

def _get_embedding_model_name() -> str:
    """Get embedding model from environment or use default."""
    return os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

def _get_semantic_model():
    """Lazy-load sentence transformer model.

    The import itself is inside the function, not at module top level. That is
    deliberate: importing sentence_transformers pulls in torch and costs a few
    seconds, and a run with --no-semantic should never pay it.
    """
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

    Adds column: semantic_score (0.0 to 1.0). This is cosine similarity of
    L2-normalized embeddings, with negative cosine floored at 0.

    This is the metric that credits a correct answer phrased differently from
    the gold string, which every lexical metric above scores as a miss. The
    price is that it is a model judging a model: MiniLM's notion of similarity
    is itself approximate, and it will happily score a fluent wrong answer as
    similar to the right one. Read it alongside exact_match, not instead of it.

    Note: First call loads the model (~90MB), subsequent calls are fast.
    """
    df = df.copy()
    model = _get_semantic_model()

    # Batch encode for efficiency. SentenceTransformer.encode does not
    # normalize unless asked, so request unit vectors here. Dot product of
    # L2-normalized vectors is cosine similarity, in [-1, 1].
    #
    # This whole function is vectorized rather than row-by-row like the others,
    # because a single encode() call over N strings is dramatically faster than
    # N calls of 1 — the GPU/SIMD batching is the entire point of the library.
    # fillna("") keeps failed rows in position so the output array still lines
    # up with the DataFrame index.
    outputs = df[output_col].fillna("").astype(str).tolist()
    expecteds = df[expected_col].fillna("").astype(str).tolist()

    output_embeddings = model.encode(
        outputs, convert_to_numpy=True, normalize_embeddings=True
    )
    expected_embeddings = model.encode(
        expecteds, convert_to_numpy=True, normalize_embeddings=True
    )

    # Row-wise dot product: multiply elementwise, then sum along the embedding
    # dimension (axis=1). Because both sides are unit vectors, this *is* cosine
    # similarity — no division by magnitudes needed.
    similarities = np.sum(output_embeddings * expected_embeddings, axis=1)

    # Floor negatives at 0 so this column stays on the same 0–1 scale as
    # exact_match / fuzzy_score. Opposite-meaning pairs collapse to 0.
    # The upper clip to 1.0 guards floating-point overshoot (a self-similarity
    # can come back as 1.0000001), which would otherwise look like a bug.
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

    This is the right tool when the model is correct but verbose: "The capital
    of France is Paris" contains "Paris", so this scores 1 where exact_match
    scores 0. The failure mode to know about is short gold answers — gold "4"
    is a substring of "the answer is 42", so this over-credits on numeric or
    single-character answers. Cross-check against exact_match before quoting it.
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

        # Direction matters: gold inside prediction, not the reverse.
        return 1 if expected in output else 0

    df["contains_expected"] = df.apply(check_contains, axis=1)
    return df


def prediction_column(df: pd.DataFrame) -> str:
    """Column scoring should read. Chain-of-thought runs store the final answer separately.

    This one-liner is the whole integration between --cot and this file. On a
    CoT run, harness.run_single adds an `answer` column holding just the
    extracted ANSWER: line; on a plain run that column does not exist. Every
    scorer routes through here, so switching --cot on redirects all of them at
    once instead of each needing its own branch.
    """
    if "answer" in df.columns:
        return "answer"
    return "output"


def _token_counts(text: str) -> Counter:
    """Multiset of normalized tokens. Unlike `tokenize`, repeats are kept."""
    return Counter(normalize_text(text).split())


def _prf(overlap: int, pred_total: int, gold_total: int) -> tuple[float, float, float]:
    """Precision, recall, and F1 from an overlap count. Both empty counts as a match.

    Shared by score_tokens, _rouge_n, and _rouge_l, because all three are the
    same arithmetic over different definitions of "overlap" — n-gram matches for
    ROUGE-N, LCS length for ROUGE-L, token multiset intersection for token F1.
    Factoring it out is what keeps those three consistent with each other.

      precision = overlap / |prediction|   "how much of what I said was right"
      recall    = overlap / |gold|         "how much of the answer did I get"
      F1        = harmonic mean of the two

    The harmonic mean is used rather than the arithmetic mean because it
    punishes imbalance: 1.0 precision with 0.1 recall gives F1 0.18, not 0.55.
    """
    if pred_total == 0 and gold_total == 0:
        return 1.0, 1.0, 1.0
    # Guard each division separately: only one side may be empty.
    precision = overlap / pred_total if pred_total else 0.0
    recall = overlap / gold_total if gold_total else 0.0
    # Avoids 0/0 in the F1 formula when there is no overlap at all.
    if precision + recall == 0:
        return precision, recall, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def score_tokens(
    df: pd.DataFrame,
    output_col: Optional[str] = None,
    expected_col: str = "expected",
) -> pd.DataFrame:
    """
    Token precision, recall, and F1.

    Counts are multisets of normalized whitespace tokens, same idea as SQuAD F1.
    A repeated word counts once per occurrence. This is the metric that stays
    informative on short answers, where BLEU and ROUGE are noisy.

    Keeping precision and recall as separate columns rather than only F1 is the
    point: they diagnose opposite failure modes. High recall with low precision
    is a model that padded a correct answer with filler. Low recall with high
    precision is a model that answered too tersely or omitted part of the answer.
    F1 alone collapses those two into the same number.
    """
    df = df.copy()
    # `or prediction_column(df)` is how every metric below picks up the CoT
    # answer column automatically while still allowing an explicit override.
    output_col = output_col or prediction_column(df)

    def score_row(row):
        output = row.get(output_col)
        expected = row.get(expected_col)
        if output is None or expected is None:
            return 0.0, 0.0, 0.0
        pred = _token_counts(str(output))
        gold = _token_counts(str(expected))
        # Counter & Counter is multiset intersection: per token, the min of the
        # two counts. Summing the values gives the total matched token count,
        # so a word said twice against a gold that says it once counts once.
        overlap = sum((pred & gold).values())
        return _prf(overlap, sum(pred.values()), sum(gold.values()))

    # result_type="expand" turns the returned 3-tuples into a 3-column frame,
    # which is then unpacked into named columns below.
    scored = df.apply(score_row, axis=1, result_type="expand")
    df["token_precision"] = scored[0]
    df["token_recall"] = scored[1]
    df["token_f1"] = scored[2]
    return df


def _ngram_counts(tokens: list[str], n: int) -> Counter:
    """Multiset of contiguous n-grams.

    Tuples (not strings) as keys so that ["new", "york"] and ["newyork"] cannot
    collide. The range stops at len(tokens) - n + 1, which correctly yields an
    empty Counter when the text is shorter than n.
    """
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def sentence_bleu(prediction: str, reference: str, max_n: int = 4) -> float:
    """
    Sentence BLEU with add-one smoothing.

    N-gram order is capped by the shorter text, so a one-token answer is BLEU-1
    instead of an automatic 0 from missing 4-grams. Add-one smoothing keeps a
    single missing n-gram from zeroing the whole score.

    BLEU is precision-oriented and was designed for corpus-level MT evaluation,
    so applying it per sentence is already a compromise; the two adaptations
    above are what make it behave sanely on short QA answers. Both are
    deviations from textbook BLEU and should be stated as such rather than
    compared against a published BLEU number.
    """
    pred = normalize_text(prediction).split()
    gold = normalize_text(reference).split()
    if not pred or not gold:
        return 0.0

    # Cap the n-gram order. Textbook BLEU-4 on a 2-token answer would find zero
    # 4-grams and return 0 for a perfect match; capping at len makes it BLEU-2.
    order = min(max_n, len(pred), len(gold))
    precisions = []
    for n in range(1, order + 1):
        pred_ng = _ngram_counts(pred, n)
        gold_ng = _ngram_counts(gold, n)
        # Clipped precision: an n-gram can only match as many times as it
        # appears in the reference, so repeating a correct word cannot inflate
        # the score. gold_ng[ng] is 0 for a missing key (Counter default).
        overlap = sum(min(count, gold_ng[ng]) for ng, count in pred_ng.items())
        total = sum(pred_ng.values())
        # Add-one (Laplace) smoothing. Without the +1s, a single order with zero
        # matches makes one precision 0, and the geometric mean below drives the
        # whole score to 0 regardless of how good the other orders were.
        precisions.append((overlap + 1) / (total + 1))

    # Geometric mean of the precisions, computed in log space to avoid
    # underflow from multiplying several small numbers.
    log_avg = sum(math.log(p) for p in precisions) / order
    # Brevity penalty. BLEU is precision-only, so without this a one-word answer
    # that happens to be in the reference would score 1.0. The penalty only
    # applies when the prediction is shorter; a too-long prediction is already
    # punished by precision.
    if len(pred) >= len(gold):
        brevity = 1.0
    else:
        brevity = math.exp(1 - len(gold) / len(pred))
    return brevity * math.exp(log_avg)


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence. Used by ROUGE-L.

    Standard dynamic-programming LCS, but kept to two rows instead of a full
    len(a) x len(b) table, since only the previous row is ever read. That makes
    it O(len(b)) memory instead of O(len(a) * len(b)) — irrelevant for short
    answers, but it keeps a long-form response from allocating a huge table.

    `prev[j-1] + 1` on a match extends the subsequence diagonally;
    `max(prev[j], curr[-1])` on a mismatch carries forward the better of
    "skip a token from a" and "skip a token from b".

    Subsequence, not substring: the matched tokens need not be adjacent, only in
    order. That is what makes ROUGE-L credit a correct answer with extra words
    interleaved.
    """
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token in a:
        curr = [0]  # column 0 is always 0 (empty prefix of b)
        for j, other in enumerate(b, start=1):
            if token == other:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def _rouge_n(prediction: str, reference: str, n: int) -> float:
    """ROUGE-N F1: overlap of n-grams between prediction and reference.

    Split out from _rouge_l after a bug where ROUGE-1 was routed through the LCS
    path and therefore was not actually measuring unigram overlap. They are
    genuinely different metrics and need separate implementations.
    """
    pred = normalize_text(prediction).split()
    gold = normalize_text(reference).split()
    if not pred and not gold:
        return 1.0
    pred_counts = _ngram_counts(pred, n)
    gold_counts = _ngram_counts(gold, n)
    # No n-gram on either side (text shorter than n) is not a match.
    # This is why rouge2 is 0 on every single-word answer, even a correct one —
    # expected behavior, not a bug, but it makes rouge2 useless on short QA.
    if not pred_counts or not gold_counts:
        return 0.0
    overlap = sum(min(count, gold_counts[ng]) for ng, count in pred_counts.items())
    _, _, f1 = _prf(overlap, sum(pred_counts.values()), sum(gold_counts.values()))
    return f1


def _rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1: longest common subsequence, so shared order still counts.

    Unlike _rouge_n there is no empty-input guard here, because _prf already
    handles both-empty (returns 1.0) and one-empty (returns 0.0) correctly.
    """
    pred = normalize_text(prediction).split()
    gold = normalize_text(reference).split()
    _, _, f1 = _prf(_lcs_length(pred, gold), len(pred), len(gold))
    return f1


def score_bleu(
    df: pd.DataFrame,
    output_col: Optional[str] = None,
    expected_col: str = "expected",
) -> pd.DataFrame:
    """Add sentence BLEU. Most useful when answers are a sentence or longer."""
    df = df.copy()
    output_col = output_col or prediction_column(df)

    def score_row(row):
        output = row.get(output_col)
        expected = row.get(expected_col)
        if output is None or expected is None:
            return 0.0
        return sentence_bleu(str(output), str(expected))

    df["bleu"] = df.apply(score_row, axis=1)
    return df


def score_rouge(
    df: pd.DataFrame,
    output_col: Optional[str] = None,
    expected_col: str = "expected",
) -> pd.DataFrame:
    """
    Add ROUGE F1 scores.

    rouge1 and rouge2 are n-gram overlap. rougeL is longest common subsequence,
    so it still credits answers that share order without being identical.

    All three are computed in one pass rather than three, since they share the
    same normalization and row iteration and .apply is the expensive part.
    """
    df = df.copy()
    output_col = output_col or prediction_column(df)

    def score_row(row):
        output = row.get(output_col)
        expected = row.get(expected_col)
        if output is None or expected is None:
            return 0.0, 0.0, 0.0
        output = str(output)
        expected = str(expected)
        return (
            _rouge_n(output, expected, 1),
            _rouge_n(output, expected, 2),
            _rouge_l(output, expected),
        )

    scored = df.apply(score_row, axis=1, result_type="expand")
    df["rouge1"] = scored[0]
    df["rouge2"] = scored[1]
    df["rougeL"] = scored[2]
    return df


def score_char(
    df: pd.DataFrame,
    output_col: Optional[str] = None,
    expected_col: str = "expected",
) -> pd.DataFrame:
    """
    Character similarity in [0, 1] from difflib.SequenceMatcher.

    Catches near-misses token F1 treats as a total miss, such as a one-character
    typo inside a single token.

    Every metric above operates on whole tokens, so "Pari" versus "Paris" is a
    complete miss for all of them. This is the only metric that sees inside a
    token, which makes it the one that distinguishes a spelling slip from a
    wrong answer. From the standard library, so it costs no dependency.
    """
    df = df.copy()
    output_col = output_col or prediction_column(df)

    def score_row(row):
        output = row.get(output_col)
        expected = row.get(expected_col)
        if output is None or expected is None:
            return 0.0
        # First arg is `isjunk`; None means treat no character as ignorable.
        return SequenceMatcher(
            None, normalize_text(str(output)), normalize_text(str(expected))
        ).ratio()

    df["char_similarity"] = df.apply(score_row, axis=1)
    return df


# =============================================================================
# Aggregate Metrics
#
# Everything above produces per-row columns. This section collapses them into
# the single dict that gets printed to the terminal and written to
# <run_id>_metrics.json. It is called by run_eval.py on every run, including
# runs with no scoring at all, because the operational half of what it reports
# (failures, truncation, latency, tokens) does not need a gold answer.
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

    The returned dict has two kinds of entries, and print_metrics below relies
    on being able to tell them apart:
      - run-level scalars and small dicts (total_samples, failure_rate, tokens)
      - one nested dict per score column, each containing a "mean" and "std" key

    Every value is a plain Python type, never a numpy scalar, so the dict is
    JSON-serializable on its own without the caller having to convert it.
    """
    if score_cols is None:
        # Auto-detect: columns with 'score', 'match', or 'contains' in name.
        # Substring matching on column names is fragile by nature — it is what
        # once picked up `score_method`, a string column, and crashed .mean().
        # The is_numeric_dtype guard in the loop below is the real defense.
        markers = (
            "score", "match", "contains", "precision", "recall", "f1",
            "bleu", "rouge", "similarity",
        )
        score_cols = [c for c in df.columns if any(x in c.lower() for x in markers)]

    total = len(df)
    # `if "error" in df.columns` lets this function work on an arbitrary frame
    # (e.g. a per-category slice, or a hand-built test frame) that may not carry
    # harness columns at all.
    successful = int(df["error"].isna().sum()) if "error" in df.columns else total
    metrics = {
        "total_samples": total,
        "successful_calls": successful,
        # Guarded against an empty frame, where the division would be 0/0.
        "failure_rate": round(1 - successful / total, 4) if total else 0.0,
    }

    # Failures grouped by cause. A run that is 10% rate-limited needs a
    # different response than one that is 10% content-filtered.
    # Omitted entirely when there are no failures, so a clean run's JSON does
    # not carry an empty dict that reads as missing data.
    if "error_type" in df.columns and df["error_type"].notna().any():
        counts = df["error_type"].dropna().value_counts()
        # str()/int() casts because value_counts returns numpy types as values
        # and may return non-str keys.
        metrics["failures_by_type"] = {str(k): int(v) for k, v in counts.items()}

    # Truncation is not failure: the call succeeded and returned a cut-off
    # answer, which scores like a wrong answer unless it is reported.
    if "truncated" in df.columns and total:
        # fillna(0) covers resumed rows loaded from an older checkpoint that
        # predates the column.
        truncated = int(df["truncated"].fillna(0).sum())
        # The count is only included when non-zero, but the rate is always
        # included, so "truncation_rate: 0.0" is an explicit statement that this
        # was checked rather than an absence that could mean either.
        if truncated:
            metrics["truncated_count"] = truncated
        metrics["truncation_rate"] = round(truncated / total, 4)

    if "latency_ms" in df.columns and df["latency_ms"].notna().any():
        latency = df["latency_ms"].dropna()
        metrics["latency_ms"] = {
            "mean": round(float(latency.mean()), 1),
            "p50": round(float(latency.median()), 1),
            # p95 rather than max: max is a single outlier, p95 is the tail you
            # would actually feel across a batch.
            "p95": round(float(latency.quantile(0.95)), 1),
        }

    # Token totals, for cost and for "do longer answers score better".
    token_cols = [c for c in ("prompt_tokens", "completion_tokens", "total_tokens") if c in df.columns]
    token_stats = {}
    for col in token_cols:
        # to_numeric(errors="coerce") turns anything unparseable into NaN rather
        # than raising. This matters on the --score-only path, where the column
        # was round-tripped through CSV and may come back as strings.
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if not values.empty:
            token_stats[col] = {"total": int(values.sum()), "mean": round(float(values.mean()), 1)}
    if token_stats:
        metrics["tokens"] = token_stats

    # Format compliance for --cot runs: what fraction of responses actually
    # produced the ANSWER: line. Distinct from answer quality, and a low value
    # here invalidates the quality numbers rather than explaining them.
    if "answer_extracted" in df.columns:
        metrics["answer_extracted_rate"] = round(float(df["answer_extracted"].mean()), 4)

    # ---- Per-score-column aggregates ---------------------------------------
    for col in score_cols:
        # Two guards, both load-bearing. The membership check covers a caller
        # passing an explicit score_cols list with a typo. The dtype check
        # covers the auto-detect above matching a string column such as
        # `score_method`, whose .mean() raises a TypeError — and it would raise
        # here, after the scored CSV has already been written, which is the
        # worst possible place to lose a run.
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        # dropna() is what implements "quality means are over successful calls
        # only": score_all writes NaN into these columns for failed rows.
        values = df[col].dropna()
        if values.empty:
            continue
        # Cast to native Python so the metrics dict is JSON-serializable on its
        # own, rather than relying on the caller to convert numpy scalars.
        std = values.std()
        metrics[col] = {
            "mean": round(float(values.mean()), 4),
            # pandas .std() is NaN for a single observation (zero degrees of
            # freedom). NaN is not valid JSON, so it is reported as 0.0.
            "std": round(float(std), 4) if pd.notna(std) else 0.0,
            "min": round(float(values.min()), 4),
            "p25": round(float(values.quantile(0.25)), 4),
            "median": round(float(values.median()), 4),
            "p75": round(float(values.quantile(0.75)), 4),
            "max": round(float(values.max()), 4),
            # The denominator for `mean`. Printed by print_metrics, because with
            # failed rows excluded this is not necessarily total_samples.
            "count": int(len(values)),
            # Where the mass actually sits. A mean of 4.8 with 80% of rows at
            # the ceiling is a saturated rubric, not a strong model.
            "histogram": score_histogram(values),
        }

    return metrics


def score_histogram(values: pd.Series, bins: int = 5) -> dict:
    """Counts per bin, as a plain dict so it survives the JSON round-trip.

    Discrete scores with few distinct values (exact_match, a 1-5 judge scale)
    are counted per value instead of binned, since binning them just blurs
    the thing worth seeing.

    Two modes, chosen automatically:
      - few distinct values -> exact counts, keyed by the value itself
      - many distinct values -> `bins` equal-width ranges, keyed "lo-hi"
    """
    distinct = sorted(values.unique())
    # `bins + 1` rather than `bins` so a full 1-5 integer judge scale (five
    # distinct values) still gets the per-value treatment.
    if len(distinct) <= bins + 1:
        return {str(round(float(v), 4)): int((values == v).sum()) for v in distinct}

    lo, hi = float(values.min()), float(values.max())
    # All values identical but numerous: the edge arithmetic below would produce
    # zero-width bins, so short-circuit.
    if lo == hi:
        return {str(round(lo, 4)): len(values)}
    edges = [lo + (hi - lo) * i / bins for i in range(bins + 1)]
    out = {}
    for i in range(bins):
        left, right = edges[i], edges[i + 1]
        # Last bin is closed on the right so the max value is counted.
        # Every other bin is half-open [left, right) so the shared edge between
        # two adjacent bins is not double-counted.
        mask = (values >= left) & (values <= right) if i == bins - 1 else (values >= left) & (values < right)
        out[f"{left:.2f}-{right:.2f}"] = int(mask.sum())
    return out


def print_metrics(metrics: dict) -> None:
    """Pretty-print metrics dict.

    Deliberately ordered operational-first, quality-second. Failure rate,
    truncation, and format compliance all determine whether the quality numbers
    below them mean anything, so they are printed where they will be read first.
    """
    print(f"\n{'='*50}")
    print("EVALUATION METRICS")
    print(f"{'='*50}")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Successful API calls: {metrics['successful_calls']}"
          f"  (failure rate {metrics.get('failure_rate', 0):.1%})")

    # .get() throughout this block: every one of these keys is conditional in
    # compute_metrics, and a KeyError here would lose the whole summary.
    if metrics.get("failures_by_type"):
        causes = ", ".join(f"{k}={v}" for k, v in metrics["failures_by_type"].items())
        print(f"Failures by type: {causes}")

    # Falsy check means a 0.0 rate prints nothing, keeping the common clean-run
    # case quiet. The number is still in the JSON either way.
    if metrics.get("truncation_rate"):
        print(f"Truncated (hit max_tokens): {metrics.get('truncated_count', 0)} "
              f"({metrics['truncation_rate']:.1%})")

    if "answer_extracted_rate" in metrics:
        print(f"ANSWER: line extracted: {metrics['answer_extracted_rate']:.1%}")

    if "latency_ms" in metrics and isinstance(metrics["latency_ms"], dict):
        latency = metrics["latency_ms"]
        print(f"Latency ms: mean={latency['mean']:.0f}  p50={latency['p50']:.0f}  p95={latency['p95']:.0f}")

    if "tokens" in metrics:
        parts = [f"{k.replace('_tokens','')}={v['total']}" for k, v in metrics["tokens"].items()]
        print(f"Tokens: {'  '.join(parts)}")
    print()

    # Per-score-column block. The `"mean" in value and "std" in value` test is
    # how a score-column entry is distinguished from the run-level dicts
    # (`tokens`, `latency_ms`, `failures_by_type`) that share the same top-level
    # namespace — a structural check rather than a hardcoded key list, so a new
    # score column needs no change here.
    for key, value in metrics.items():
        if isinstance(value, dict) and "mean" in value and "std" in value:
            print(f"{key}:")
            # n= is the denominator. With failed rows blanked by score_all, this
            # can be lower than total_samples, and a reader who does not see it
            # would take the mean as a whole-dataset figure.
            print(f"  mean: {value['mean']:.4f}  (std: {value['std']:.4f}, "
                  f"n={value['count']})")
            print(f"  range: [{value['min']:.4f}, {value['max']:.4f}]  "
                  f"p25={value['p25']:.4f}  median={value['median']:.4f}  p75={value['p75']:.4f}")
            if value.get("histogram"):
                spread = "  ".join(f"{k}:{v}" for k, v in value["histogram"].items())
                print(f"  distribution: {spread}")
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

    This is the single entry point run_eval.py uses. Running every metric rather
    than picking one is deliberate for an unknown task: which metric is the
    right one depends on the shape of the answers, and that is not known until
    the dataset arrives. Computing all of them costs nothing (they are local
    string operations on data already paid for) and lets the choice be made
    after looking at the numbers.

    Args:
        include_semantic: If True, includes semantic scoring (slower, loads model).
    """
    # Resolved once and passed explicitly to each scorer, rather than letting
    # each re-resolve it, so all metrics are guaranteed to read the same column.
    output_col = prediction_column(df)
    # Snapshot of the pre-existing columns, used at the bottom to identify which
    # columns this function added.
    before = set(df.columns)
    df = score_exact(df, output_col=output_col)
    df = score_fuzzy(df, output_col=output_col)
    df = score_contains(df, output_col=output_col)
    df = score_tokens(df, output_col=output_col)
    df = score_bleu(df, output_col=output_col)
    df = score_rouge(df, output_col=output_col)
    df = score_char(df, output_col=output_col)

    # Last, and optional, because it is the only one with a model-loading cost.
    if include_semantic:
        df = score_semantic(df, output_col=output_col)

    # A row whose API call failed has no output to score. Leaving it in as a 0
    # would blend two different things into one number: a model that answered
    # wrong and a request that never returned. Blanking it makes every quality
    # mean "of the calls that succeeded", with failure_rate reported alongside.
    #
    # Scoped to the added columns via the `before` snapshot, so this never
    # touches the harness's own columns or the dataset's — blanking `error`
    # itself would erase the very thing that identifies these rows as failures.
    if "error" in df.columns:
        failed = df["error"].notna()
        if failed.any():
            df.loc[failed, list(set(df.columns) - before)] = np.nan

    return df


# =============================================================================
# Quick test
#
# `python -m llm_eval.scoring` exercises every metric on hand-written rows with
# no API calls. Fast enough to run after any edit in this file, and the five
# rows are chosen to cover the cases that matter: exact, case-differing,
# verbose-but-correct, wrong, and failed.
# =============================================================================

if __name__ == "__main__":
    # Test with mock data (no API calls needed)
    test_data = pd.DataFrame([
        {"prompt_id": "q1", "output": "4", "expected": "4", "error": None},
        {"prompt_id": "q2", "output": "Paris", "expected": "paris", "error": None},
        {"prompt_id": "q3", "output": "The answer is 42.", "expected": "42", "error": None},
        {"prompt_id": "q4", "output": "I don't know", "expected": "London", "error": None},
        # The failed row: every score column should come back NaN, not 0.
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
