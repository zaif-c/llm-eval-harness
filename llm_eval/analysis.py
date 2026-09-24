"""
Meta-Evaluation Layer
=====================
Compares the LLM judge against ground truth to answer one question:
can the judge's scores be trusted?

WHERE THIS FILE SITS IN THE PIPELINE
------------------------------------
    harness.py -> scoring.py  (ground truth)  -\\
                                                >-- analysis.py
               -> judge.py    (rubric score)  -/

scoring.py grades the model. judge.py grades the model. This file grades the
*judge*, which is why it is a layer above both rather than a function inside
either. It only has anything to do when both scorers ran on the same rows,
which happens on a labeled dataset with --judge.

WHY THIS MATTERS MORE THAN IT LOOKS
-----------------------------------
The judge exists to score open-ended data, where by definition there is no gold
answer to check it against. So the judge's credibility can never be established
on the data it is actually used for. The only place it *can* be established is a
labeled set, where both a rubric score and a known-correct answer exist for the
same row. Running both there and comparing is the only evidence that the numbers
the judge produces elsewhere mean anything.

Stated the other way: without this, "the judge gave it 4.2" is an assertion.
With it, it is a measurement with a known error characteristic.

WHAT IS REPORTED, AND WHY EACH ONE
----------------------------------
  separation  mean judge score on correct answers minus mean on incorrect ones.
              The most intuitive number and the one to lead with. If this is
              near zero the judge is not tracking correctness at all, and no
              correlation coefficient will rescue it.

  ROC-AUC     probability that a randomly chosen correct answer receives a
              higher judge score than a randomly chosen incorrect one. 0.5 is
              chance, 1.0 is perfect. Used for binary ground truth because it
              needs no assumption that the 1-5 rubric scale and the 0/1 label
              live on comparable scales — it only uses the ordering.

  Spearman    rank correlation. The primary correlation here because it assumes
              only a monotonic relationship. A 1-5 rubric and a 0-1 F1 are not
              linearly related and there is no reason to expect them to be.

  Pearson     linear correlation. Reported alongside because it is what people
              expect to see, but it is the weaker claim: it assumes a linearity
              that the scales do not have. When ground truth is binary, Pearson
              is exactly the point-biserial correlation.

  disagreements  the individual rows where the two scorers most disagree. The
              only qualitative output here, and usually the most informative:
              aggregate agreement of 0.8 does not tell you *which* answers the
              judge misreads, and the direction of the disagreement matters.

No scipy or scikit-learn. pandas provides the correlations natively, and AUC is
five lines via its Mann-Whitney relationship (see roc_auc), so this adds no
dependency and nothing here is a black box during a presentation.
"""

from typing import Optional

import numpy as np
import pandas as pd

# Ground-truth columns compared against the judge by default.
#
# score_all produces a dozen columns, and reporting all of them buries the
# finding in noise. These three are chosen because they represent genuinely
# different notions of "correct", so agreement with each answers a different
# question:
#   exact_match       strict string identity -> tracks hard correctness AND format
#   contains_expected gold answer appears    -> correctness, tolerant of verbosity
#   token_f1          graded lexical overlap -> partial correctness
#   semantic_score    graded meaning overlap -> meaning rather than wording
#
# Comparing agreement ACROSS these is itself the diagnostic, and is usually more
# informative than any single number. A judge that agrees with semantic_score
# but not exact_match is rewarding answers that sound right. A judge that agrees
# with contains_expected but not exact_match is not wrong at all — it means
# exact_match is penalizing verbosity, and the judge is the one reading the
# answer correctly. See the format-mismatch warning in judge_agreement.
DEFAULT_GT_COLS = ("exact_match", "contains_expected", "token_f1", "semantic_score")

# Below this many usable rows, correlations are too unstable to quote. At n=10
# the 95% confidence interval on a correlation spans most of the range, so a
# reported 0.7 is not meaningfully different from 0.2.
SMALL_SAMPLE_N = 30


def roc_auc(scores, labels) -> Optional[float]:
    """Area under the ROC curve, computed from ranks rather than a curve.

    Interpretation: the probability that a randomly chosen positive outranks a
    randomly chosen negative. 0.5 is chance, 1.0 is perfect separation, and
    below 0.5 means the score is inversely related to the label.

    This uses the identity between AUC and the Mann-Whitney U statistic:

        AUC = (sum of positive ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    The subtracted term is the minimum possible rank sum for n_pos items (that
    is, 1 + 2 + ... + n_pos), so the numerator counts how far above the
    worst case the positives actually ranked, and dividing by n_pos * n_neg
    normalizes by the number of positive/negative pairs.

    method="average" gives tied scores their midrank, which is what makes a tie
    contribute exactly 0.5 — the correct treatment, since a tie is genuinely no
    evidence either way. This is also why AUC is preferable to eyeballing means
    on a coarse 1-5 scale, where ties are common.

    Returns None when one class is missing, because AUC is undefined without
    both: with no incorrect answers there are no pairs to rank against.
    """
    scores = pd.Series(scores).astype(float)
    labels = pd.Series(labels).astype(float)
    # Both must be present. A row with a failed judge call or an unscored
    # ground truth carries no information about their agreement.
    mask = scores.notna() & labels.notna()
    scores, labels = scores[mask], labels[mask]

    positive = labels == 1
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return None

    ranks = scores.rank(method="average")
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _is_binary(values: pd.Series) -> bool:
    """True when the column only ever takes the values 0 and 1.

    Drives the choice of statistics: a binary column gets the separation and
    AUC treatment, a continuous one gets correlations only. Detected from the
    data rather than from a hardcoded column-name list, so a custom 0/1 metric
    added later is handled without touching this file.

    A column that is all 1s (or all 0s) still counts as binary; the degenerate
    single-class case is caught later, where it can be reported as a warning
    rather than silently reclassified as continuous.
    """
    distinct = set(values.dropna().unique())
    return distinct.issubset({0, 0.0, 1, 1.0})


def _correlations(judge: pd.Series, truth: pd.Series) -> dict:
    """Spearman and Pearson between two aligned series.

    Both are None when either side has zero variance. pandas returns NaN there,
    and NaN is not valid JSON and prints as a confusing "nan" — whereas None
    reads correctly as "not computable", which is exactly what a constant column
    means. This is not a rare edge case: a strong model on an easy dataset
    scores 1.0 on every row, and then correctness has no variance to correlate
    against.
    """
    # nunique() < 2 is the zero-variance test. Cheaper and more direct than
    # computing std and comparing to zero.
    if judge.nunique() < 2 or truth.nunique() < 2:
        return {"spearman": None, "pearson": None}
    spearman = judge.corr(truth, method="spearman")
    pearson = judge.corr(truth, method="pearson")
    return {
        "spearman": round(float(spearman), 4) if pd.notna(spearman) else None,
        "pearson": round(float(pearson), 4) if pd.notna(pearson) else None,
    }


def judge_agreement(
    df: pd.DataFrame,
    judge_col: str = "judge_score",
    gt_cols: Optional[tuple] = None,
) -> Optional[dict]:
    """Compare the judge's scores against each ground-truth metric.

    Returns a report dict, or None when the comparison is impossible (no judge
    column, no ground-truth columns, or no row with both). Returning None rather
    than an empty report lets the caller skip the whole section cleanly, since
    "we did not measure this" and "we measured it and found nothing" should not
    print the same way.

    Every value in the returned dict is a plain Python type, matching the
    convention in scoring.compute_metrics, so the report can be dropped straight
    into the metrics JSON.
    """
    if judge_col not in df.columns:
        return None

    gt_cols = gt_cols or DEFAULT_GT_COLS
    # Filtered against what is actually present and numeric: a run with
    # --no-semantic has no semantic_score column, and a CSV round-trip can turn
    # an all-null column into object dtype.
    available = [
        c for c in gt_cols
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not available:
        return None

    judge_all = pd.to_numeric(df[judge_col], errors="coerce")
    warnings: list[str] = []

    comparisons = {}
    for col in available:
        truth_all = pd.to_numeric(df[col], errors="coerce")
        # Pairwise-complete: a row is used for this comparison only if both
        # scores exist. Done per column rather than dropping rows once up front,
        # so one metric with extra nulls does not shrink the sample for the
        # others.
        mask = judge_all.notna() & truth_all.notna()
        judge = judge_all[mask]
        truth = truth_all[mask]
        if judge.empty:
            continue

        entry: dict = {"n": int(len(judge))}
        entry.update(_correlations(judge, truth))

        if _is_binary(truth):
            entry["type"] = "binary"
            correct = truth == 1
            n_correct = int(correct.sum())
            n_incorrect = int((~correct).sum())
            entry["n_correct"] = n_correct
            entry["n_incorrect"] = n_incorrect
            # Both classes are required for every statistic in this block. The
            # single-class case is common enough to deserve an explicit warning
            # rather than a row of nulls: it means the dataset was too easy (or
            # too hard) to say anything about the judge at all.
            if n_correct and n_incorrect:
                mean_correct = float(judge[correct].mean())
                mean_incorrect = float(judge[~correct].mean())
                entry["mean_judge_when_correct"] = round(mean_correct, 4)
                entry["mean_judge_when_incorrect"] = round(mean_incorrect, 4)
                entry["separation"] = round(mean_correct - mean_incorrect, 4)
                auc = roc_auc(judge, truth)
                entry["auc"] = round(auc, 4) if auc is not None else None
            else:
                entry["auc"] = None
                only = "correct" if n_correct else "incorrect"
                warnings.append(
                    f"{col}: every row is {only}, so there is nothing to "
                    f"separate. Agreement is unmeasurable on this dataset."
                )
        else:
            entry["type"] = "continuous"

        if entry.get("spearman") is None and entry["n"] >= 2:
            warnings.append(
                f"{col}: one side has no variance, so correlation is undefined."
            )

        comparisons[col] = entry

    if not comparisons:
        return None

    # Format-mismatch diagnostic.
    #
    # If the gold answer appears inside the output far more often than it equals
    # the output, exact_match is measuring output format, not correctness, and
    # every agreement statistic computed against it is measuring the wrong
    # thing. This is worth detecting automatically because the symptom is
    # counterintuitive: it shows up as the judge appearing to *disagree* with
    # ground truth, which reads as a broken judge rather than a mismatched
    # metric. Observed live on a set where the model answered all 16 correctly
    # but exact_match scored 9 of them zero.
    if "exact_match" in df.columns and "contains_expected" in df.columns:
        strict = pd.to_numeric(df["exact_match"], errors="coerce")
        loose = pd.to_numeric(df["contains_expected"], errors="coerce")
        both = strict.notna() & loose.notna()
        if both.any():
            gap = float(loose[both].mean() - strict[both].mean())
            # A fifth of the dataset is enough to change which metric should be
            # believed; below that it is noise not worth flagging.
            if gap >= 0.2:
                n_affected = int(((loose == 1) & (strict == 0))[both].sum())
                warnings.append(
                    f"format mismatch: {n_affected} rows contain the gold answer "
                    f"but do not equal it (exact_match {strict[both].mean():.2f} "
                    f"vs contains_expected {loose[both].mean():.2f}). exact_match "
                    f"is scoring output format here, not correctness — prefer "
                    f"contains_expected as the reference, or constrain the prompt."
                )

    # The sample size that actually backed the comparisons, used for the
    # small-sample warning. Taken as the max across columns since pairwise
    # completeness can differ between them.
    n_used = max(entry["n"] for entry in comparisons.values())
    if n_used < SMALL_SAMPLE_N:
        warnings.append(
            f"n={n_used} is small. Correlations at this size are unstable; "
            f"read them as directional, not as point estimates."
        )

    return {
        "judge_col": judge_col,
        "n": n_used,
        "comparisons": comparisons,
        "warnings": warnings,
    }


def find_disagreements(
    df: pd.DataFrame,
    gt_col: str = "exact_match",
    judge_col: str = "judge_score",
    scale: tuple = (1, 5),
    top_n: int = 5,
) -> pd.DataFrame:
    """The rows where the judge and ground truth disagree most.

    The two scores are not comparable as-is, so the judge score is first mapped
    onto the ground-truth metric's 0-1 range using the *nominal* rubric scale,
    not the observed min and max. Using the observed range would rescale
    relative to the run itself: on a run where every score landed between 4 and
    5, the 4s would be stretched down to 0 and look like harsh failures.

    `gap` is signed, and the sign narrows down the cause without settling it.
    Each direction has two readings, and only the text of the row distinguishes
    them — which is the reason this returns rows to read rather than a number:

      gap > 0   judge scored it higher than ground truth did. Either
                (a) the judge over-credited a wrong answer, which is the
                    dangerous case, because on open-ended data that error is
                    invisible; or
                (b) the ground-truth metric is too strict for this answer
                    format — a correct answer wrapped in a sentence scores
                    exact_match 0, and the judge is the one reading it right.
                Case (b) is common enough that judge_agreement detects it
                separately and warns.

      gap < 0   judge scored it lower than ground truth did. Either the judge
                caught a real problem the lexical metric could not see
                (unsupported reasoning, a confident hedge), or the rubric is
                asking for something the gold answer does not supply.

    Sorted by absolute gap, so both directions compete for the top slots rather
    than one drowning out the other.
    """
    if judge_col not in df.columns or gt_col not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    judge = pd.to_numeric(work[judge_col], errors="coerce")
    truth = pd.to_numeric(work[gt_col], errors="coerce")

    lo, hi = scale
    # Guard a degenerate scale (min == max) that would divide by zero.
    span = (hi - lo) or 1
    # Clipped because a judge can return a score marginally outside its nominal
    # scale, which would otherwise produce a gap above 1 and outrank real
    # disagreements.
    judge_norm = ((judge - lo) / span).clip(0.0, 1.0)

    work["judge_normalized"] = judge_norm.round(4)
    work["gap"] = (judge_norm - truth).round(4)
    work["direction"] = np.where(
        work["gap"] > 0, "judge_generous",
        np.where(work["gap"] < 0, "judge_harsh", "agree"),
    )

    work = work[work["gap"].notna()]
    # Perfect agreement is not a disagreement; excluding it keeps the list from
    # being padded with zeros when there are fewer than top_n real conflicts.
    work = work[work["gap"] != 0]
    if work.empty:
        return pd.DataFrame()

    work = work.reindex(work["gap"].abs().sort_values(ascending=False).index)

    # Only the columns needed to judge the disagreement by eye. The full row is
    # always available in the scored CSV if more context is needed.
    preferred = [
        "prompt_id", "input", "output", "answer", "expected",
        gt_col, judge_col, "judge_normalized", "gap", "direction",
    ]
    cols = [c for c in preferred if c in work.columns]
    return work[cols].head(top_n)


# =============================================================================
# Variance across repeated samples
#
# Answers a question the rest of the pipeline quietly assumes: is temperature 0
# actually deterministic, and would the headline number survive a rerun?
# Requires a run with --n-samples > 1, which sends each prompt several times as
# independent requests.
# =============================================================================

def variance_report(
    df: pd.DataFrame,
    score_cols: Optional[list] = None,
) -> Optional[dict]:
    """Measure how much scores move when the same prompt is asked repeatedly.

    Returns None on a single-sample frame, since there is nothing to compare.

    Two different variances are reported, and they answer different questions:

      within-prompt   Does the model give the same answer to the same question?
                      Measured as the standard deviation of a score across one
                      prompt's samples, then averaged over prompts. This is a
                      property of the model.

      run-level       Would the headline number move on a rerun? Each
                      sample_index is treated as one complete replicate run of
                      the whole dataset, and the aggregate metric is computed
                      per replicate. The spread across those is the error bar on
                      the number that goes in the presentation.

    The second is the decision-relevant one and the reason this exists. Without
    it, "model A scored 0.72 and model B scored 0.75" is an unfalsifiable claim;
    with it, that gap is either outside the run-to-run spread or it is noise.

    `output_identical_rate` is reported separately from any score, because it is
    the stricter and more direct test: byte-identical outputs mean determinism
    regardless of what the metrics say. A model can be perfectly stable in score
    while varying its wording, and those are different findings.
    """
    if "sample_index" not in df.columns or "prompt_id" not in df.columns:
        return None
    n_samples = int(pd.to_numeric(df["sample_index"], errors="coerce").nunique())
    if n_samples < 2:
        return None

    # Only successful rows. A failed call has no output and no score, and
    # counting it as "different" would report retry luck as model variance.
    work = df[df["error"].isna()] if "error" in df.columns else df

    if score_cols is None:
        markers = ("score", "match", "contains", "f1", "bleu", "rouge", "similarity")
        score_cols = [
            c for c in work.columns
            if any(m in c.lower() for m in markers)
            and pd.api.types.is_numeric_dtype(work[c])
        ]

    report: dict = {
        "n_samples": n_samples,
        "n_prompts": int(work["prompt_id"].nunique()),
        "per_metric": {},
    }

    # Exact output stability. The strongest statement available: if every
    # prompt returned byte-identical text across all samples, the run is
    # reproducible and every metric below is trivially stable.
    if "output" in work.columns:
        grouped = work.groupby("prompt_id")["output"]
        # nunique(dropna=False) so that two nulls count as identical rather
        # than as a difference.
        varies = grouped.nunique(dropna=False) > 1
        n_varying = int(varies.sum())
        n_total = int(len(varies))
        report["n_prompts_with_varying_output"] = n_varying
        report["output_identical_rate"] = round(1 - n_varying / n_total, 4) if n_total else 1.0
        # Named so they can be pulled up and read; a handful of unstable
        # prompts is far more useful than the rate alone.
        report["unstable_prompt_ids"] = [str(p) for p in varies[varies].index[:10]]

    for col in score_cols:
        values = pd.to_numeric(work[col], errors="coerce")
        if values.notna().sum() == 0:
            continue
        frame = pd.DataFrame({
            "prompt_id": work["prompt_id"],
            "sample_index": work["sample_index"],
            "value": values,
        }).dropna(subset=["value"])
        if frame.empty:
            continue

        # Within-prompt: std across the samples of each prompt. A prompt with
        # only one surviving sample has std NaN (zero degrees of freedom) and is
        # dropped rather than counted as perfectly stable.
        per_prompt_std = frame.groupby("prompt_id")["value"].std().dropna()
        # Run-level: one aggregate per replicate, then the spread across them.
        per_run_mean = frame.groupby("sample_index")["value"].mean()

        entry = {
            "mean_within_prompt_std": round(float(per_prompt_std.mean()), 4) if not per_prompt_std.empty else 0.0,
            "max_within_prompt_std": round(float(per_prompt_std.max()), 4) if not per_prompt_std.empty else 0.0,
            "prompts_with_variation": int((per_prompt_std > 0).sum()),
        }
        if len(per_run_mean) >= 2:
            entry["run_means"] = [round(float(v), 4) for v in per_run_mean]
            # Spread (max - min) rather than std, because with 3 replicates a
            # standard deviation is barely meaningful and the range is what a
            # reader actually wants: "the number landed between X and Y".
            entry["run_spread"] = round(float(per_run_mean.max() - per_run_mean.min()), 4)
        report["per_metric"][col] = entry

    return report


def find_unstable_prompts(
    df: pd.DataFrame,
    score_col: str = "exact_match",
    top_n: int = 5,
) -> pd.DataFrame:
    """Prompts whose score moved most across samples, worst first.

    One row per prompt, not per sample: the columns summarize the spread so the
    instability is visible at a glance, and the full per-sample rows are in the
    scored CSV for anything that needs reading in detail.

    These are the prompts to look at first, because an unstable prompt is
    usually unstable for a nameable reason — an ambiguous question, an answer
    sitting exactly on a scoring threshold, or a genuine coin-flip in the model.
    All three are findings; none of them are visible in an aggregate.
    """
    if score_col not in df.columns or "prompt_id" not in df.columns:
        return pd.DataFrame()

    work = df[df["error"].isna()] if "error" in df.columns else df
    values = pd.to_numeric(work[score_col], errors="coerce")
    frame = pd.DataFrame({"prompt_id": work["prompt_id"], "value": values}).dropna()
    if frame.empty:
        return pd.DataFrame()

    grouped = frame.groupby("prompt_id")["value"]
    summary = pd.DataFrame({
        "n_samples": grouped.count(),
        "mean": grouped.mean().round(4),
        "std": grouped.std().round(4),
        "min": grouped.min().round(4),
        "max": grouped.max().round(4),
    })
    summary["spread"] = (summary["max"] - summary["min"]).round(4)
    summary = summary[summary["spread"] > 0]
    if summary.empty:
        return pd.DataFrame()
    return summary.sort_values("spread", ascending=False).head(top_n).reset_index()


def print_variance(report: Optional[dict], score_col: str = "exact_match") -> None:
    """Print the variance report.

    Leads with output stability because it is the clearest statement, then the
    run-level spread, which is the number that qualifies every other number in
    the run.
    """
    if not report:
        return

    print(f"\n{'='*50}")
    print("VARIANCE ACROSS REPEATED SAMPLES")
    print(f"{'='*50}")
    print(f"{report['n_prompts']} prompts x {report['n_samples']} samples each")

    if "output_identical_rate" in report:
        rate = report["output_identical_rate"]
        print(f"Byte-identical outputs across samples: {rate:.1%} of prompts "
              f"({report['n_prompts_with_varying_output']} varied)")
        if report.get("unstable_prompt_ids"):
            print(f"  varying: {', '.join(report['unstable_prompt_ids'])}")
    print()

    for col, entry in report["per_metric"].items():
        # A metric that never moved is reported in one line rather than five.
        # On a deterministic run that is most of them, and the interesting ones
        # should not be buried.
        if entry["prompts_with_variation"] == 0 and not entry.get("run_spread"):
            print(f"  {col}: stable (no variation across samples)")
            continue
        print(f"  {col}")
        print(f"    within-prompt std:  mean {entry['mean_within_prompt_std']:.4f}"
              f"  worst {entry['max_within_prompt_std']:.4f}"
              f"  ({entry['prompts_with_variation']} prompts varied)")
        if "run_spread" in entry:
            means = "  ".join(f"{v:.4f}" for v in entry["run_means"])
            print(f"    per-run means:      {means}")
            print(f"    run-to-run spread:  {entry['run_spread']:.4f}"
                  f"   <- differences smaller than this are noise")
    print()


def print_agreement(report: Optional[dict]) -> None:
    """Print the agreement report.

    Ordered to match how the numbers should be reasoned about: separation first
    because it is the sanity check, then AUC as the assumption-free version of
    the same claim, then the correlations, then the caveats. A reader who stops
    after the first line still leaves with the right conclusion.
    """
    if not report:
        return

    print(f"\n{'='*50}")
    print("JUDGE vs GROUND TRUTH AGREEMENT")
    print(f"{'='*50}")
    print(f"Comparing {report['judge_col']} against "
          f"{len(report['comparisons'])} ground-truth metrics (n={report['n']})")
    print()

    for col, entry in report["comparisons"].items():
        if entry["type"] == "binary":
            header = (f"{col} (binary: {entry.get('n_correct', 0)} correct / "
                      f"{entry.get('n_incorrect', 0)} incorrect)")
        else:
            header = f"{col} (continuous)"
        print(f"  {header}")

        if "separation" in entry:
            print(f"    mean judge score:  correct {entry['mean_judge_when_correct']:.2f}"
                  f"  |  incorrect {entry['mean_judge_when_incorrect']:.2f}"
                  f"  |  separation {entry['separation']:+.2f}")
        if entry.get("auc") is not None:
            print(f"    ROC-AUC:           {entry['auc']:.3f}"
                  f"   (0.5 = chance, 1.0 = perfect separation)")

        # "n/a" rather than omitting the line, so an undefined correlation is
        # visibly undefined instead of looking like it was never computed.
        spearman = f"{entry['spearman']:+.3f}" if entry["spearman"] is not None else "n/a"
        pearson = f"{entry['pearson']:+.3f}" if entry["pearson"] is not None else "n/a"
        print(f"    Spearman: {spearman}     Pearson: {pearson}")
        print()

    for warning in report["warnings"]:
        print(f"  ! {warning}")
    if report["warnings"]:
        print()


def print_disagreements(disagreements: pd.DataFrame, max_chars: int = 60) -> None:
    """Print the disagreement table with long text fields truncated.

    Truncation is presentation-only; the full text is in the scored CSV and in
    the disagreements CSV this is printed alongside. Without it a single verbose
    response wraps across the terminal and makes the table unreadable.
    """
    if disagreements is None or disagreements.empty:
        return

    print(f"{'-'*50}")
    print("TOP DISAGREEMENTS")
    print("  gap > 0: judge scored higher — over-crediting, OR a too-strict metric")
    print("  gap < 0: judge scored lower  — caught something, OR rubric mismatch")
    print("  read the row text to tell which")
    print(f"{'-'*50}")

    display = disagreements.copy()
    for col in ("input", "output", "answer", "expected"):
        if col in display.columns:
            display[col] = (
                display[col].astype(str).str.replace(r"\s+", " ", regex=True)
                .str.slice(0, max_chars)
            )
    print(display.to_string(index=False))
    print()
