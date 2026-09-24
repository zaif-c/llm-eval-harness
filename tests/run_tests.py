#!/usr/bin/env python3
"""
End-to-end test suite for the eval harness.
=========================================================================
One command, one report:

    python tests/run_tests.py              # everything (offline + live API)
    python tests/run_tests.py --offline    # no API calls, ~15s
    python tests/run_tests.py -k resume    # only tests matching "resume"

WHAT THESE TESTS ASSERT, AND WHAT THEY DELIBERATELY DO NOT
----------------------------------------------------------
They assert *pipeline mechanics*: row counts, column schemas, resume
arithmetic, ordering, error handling, and the specific metric behaviors that
were chosen deliberately and would be easy to regress.

They do NOT assert that the model answers correctly. "exact_match == 1.0 on
all 10 prompts" is a fact about gpt-4o-mini on a Tuesday, not about this code.
A suite that fails because the model phrased something differently is a suite
you learn to ignore, which is worse than no suite. The one place model
behavior is checked at all is a loose floor (>0 rows scored, >80% CoT format
compliance), because a hard zero there means the plumbing broke, not that the
model got unlucky.

LAYOUT
------
Two groups. OFFLINE tests need no network and cover the pure functions.
LIVE tests shell out to run_eval.py and make real OpenRouter calls; they are
the ones that catch integration breaks the unit tests cannot see.

Every test is a function registered with @test(...). Failures are collected
rather than raised, so one broken test does not hide the other twelve.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# The repo root is the parent of tests/. Inserted on sys.path so `import
# llm_eval` resolves without the suite having to be installed as a package.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Populated by the @test decorator at import time, run by main() at the bottom.
TESTS = []
# Scratch directory for artifacts, created in main() and deleted afterwards so
# a test run never leaves files in results/ that look like a real run.
TMP = None


def test(name, live=False):
    """Register a test function.

    live=True marks a test that makes real API calls, so --offline can skip it.
    """
    def decorator(fn):
        TESTS.append({"name": name, "fn": fn, "live": live})
        return fn
    return decorator


def cli(*args, expect_ok=True):
    """Run run_eval.py as a subprocess and return its combined output.

    A subprocess rather than calling main() directly, because the CLI is the
    interface that actually gets used under time pressure. Importing and
    calling main() would not catch an argparse mistake or an import-time error.

    sys.executable, not "python", so the venv interpreter is used regardless of
    what is on PATH.
    """
    cmd = [sys.executable, str(REPO / "run_eval.py"), *map(str, args)]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    if expect_ok and proc.returncode != 0:
        raise AssertionError(f"exit {proc.returncode} from: {' '.join(args)}\n{out[-2000:]}")
    return out


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def ensure_run(run_id, *cli_args):
    """Make sure a named run's artifacts exist, producing them only if missing.

    Several live tests need a finished run to work from: the resume tests need
    a checkpoint to resume, and --score-only needs a raw CSV to rescore.

    Without this helper those tests would depend on an earlier test having run
    first, which works on a full pass and breaks under `-k resume`. Inter-test
    ordering dependencies are the wrong thing to debug under time pressure, so
    each test declares what it needs and this creates it on demand. On a full
    run the artifact already exists and no calls are made.
    """
    if not (TMP / f"{run_id}_raw.csv").exists():
        cli(*cli_args, "--output-dir", TMP, "--run-id", run_id)
    return TMP / f"{run_id}_raw.csv"


def gt_run():
    """The shared 10-row ground-truth run."""
    return ensure_run("live_gt", "--dataset", "datasets/ground_truth_demo.json")


def judge_run():
    """The shared 16-row hard-set run with judge scores."""
    return ensure_run("live_judge", "--dataset", "datasets/ground_truth_hard.json",
                      "--judge", "--judge-rubric", "accuracy")


# =============================================================================
# OFFLINE TESTS — pure functions, no network
# =============================================================================

@test("imports: package surface is importable")
def t_imports():
    """The public API in __init__.py must actually resolve.

    Catches the common breakage after a deletion: a name removed from a module
    but still listed in __init__.py, which raises only on first import.
    """
    import llm_eval
    for name in ("HarnessConfig", "batch_run", "run_single", "score_all",
                 "compute_metrics", "judge_batch", "JudgeConfig", "Rubric"):
        assert hasattr(llm_eval, name), f"missing export: {name}"


@test("cut: --n-samples is fully removed")
def t_n_samples_gone():
    """Guards the cut from being partially reverted.

    All four places the feature used to live are checked, because removing it
    from one and not the others is the failure mode that produces a confusing
    half-state (e.g. a config field nothing reads).
    """
    import inspect
    from llm_eval import HarnessConfig
    from llm_eval.harness import run_single
    import llm_eval.harness as h
    import llm_eval.judge as j

    assert "n_samples" not in HarnessConfig.__dataclass_fields__
    assert "sample_index" not in inspect.signature(run_single).parameters
    assert not hasattr(h, "row_key"), "row_key should be gone from harness"
    assert not hasattr(j, "row_key"), "row_key should not be imported by judge"
    # The CLI flag itself must be gone, not merely ignored.
    out = cli("--help")
    assert "--n-samples" not in out


@test("cut: --cot survived")
def t_cot_kept():
    """The counterpart to the test above: CoT was deliberately kept."""
    from llm_eval import HarnessConfig
    from llm_eval.harness import extract_final_answer, COT_INSTRUCTION
    assert "chain_of_thought" in HarnessConfig.__dataclass_fields__
    assert "ANSWER:" in COT_INSTRUCTION
    assert "--cot" in cli("--help")


@test("extract_final_answer: last ANSWER: wins")
def t_extract_answer():
    """The last match wins, not the first.

    A reasoning model routinely mentions the format mid-thought ("...so the
    ANSWER: should be the capital..."), and the real answer is the final line.
    Also checks the not-followed case returns the full text with extracted=False
    so a non-compliant response is still scoreable rather than dropped.
    """
    from llm_eval.harness import extract_final_answer

    ans, ok = extract_final_answer("thinking...\nANSWER: Paris")
    assert (ans, ok) == ("Paris", True)

    # Two markers: the last one is the real answer.
    ans, ok = extract_final_answer("the ANSWER: might be X\nANSWER: Y")
    assert (ans, ok) == ("Y", True), ans

    # Case-insensitive and tolerant of leading whitespace.
    ans, ok = extract_final_answer("  answer:   Paris  ")
    assert (ans, ok) == ("Paris", True), ans

    # No marker: full text back, flagged as not-extracted.
    ans, ok = extract_final_answer("just Paris")
    assert (ans, ok) == ("just Paris", False)

    assert extract_final_answer(None) == (None, False)
    assert extract_final_answer("") == (None, False)


@test("classify_error: buckets by cause")
def t_classify_error():
    """Each bucket implies a different fix, which is the whole point.

    rate_limit means lower --max-workers, model_not_found means a typo,
    content_filter means the dataset is tripping a safety layer. Without the
    bucketing every failure reads as "the API flaked".
    """
    from llm_eval import classify_error
    cases = {
        "429 rate limit exceeded": "rate_limit",
        "Request timed out": "timeout",
        "content policy violation": "content_filter",
        "is not a valid model id": "model_not_found",
        None: None,
    }
    for text, expected in cases.items():
        got = classify_error(text)
        assert got == expected, f"{text!r} -> {got!r}, expected {expected!r}"


@test("bleu: max_order capping (the key metric decision)")
def t_bleu_capping():
    """The single most important metric behavior on this branch.

    evaluate's BLEU defaults to order 4. A correct one-word answer has no
    4-grams, so "Paris" vs "Paris" scores 0 — a perfect answer reported as a
    total miss. score_bleu caps max_order at the shorter text to fix it.

    The wrong-answer case is checked in the same test on purpose: capping must
    not turn into a participation trophy. If a future change enables smoothing,
    "London" vs "Paris" jumps to ~0.84 and this assertion catches it.
    """
    import pandas as pd
    from llm_eval.scoring import score_bleu

    df = pd.DataFrame({
        "output":   ["Paris", "London", "New York"],
        "expected": ["Paris", "Paris",  "New York"],
    })
    df = score_bleu(df, output_col="output")
    got = list(df["bleu"])
    assert approx(got[0], 1.0), f"correct 1-word answer should be 1.0, got {got[0]}"
    assert approx(got[1], 0.0), f"wrong 1-word answer should be 0.0, got {got[1]}"
    assert approx(got[2], 1.0), f"correct 2-word answer should be 1.0, got {got[2]}"


@test("bleu: empty prediction does not raise")
def t_bleu_empty():
    """bleu.compute raises ZeroDivisionError on an empty prediction.

    Empty outputs are real: a truncated generation or a refusal produces one.
    Without the guard in score_bleu, one empty row kills the whole scoring pass
    after the API calls have already been paid for.
    """
    import pandas as pd
    from llm_eval.scoring import score_bleu
    df = score_bleu(pd.DataFrame({"output": [""], "expected": ["Paris"]}),
                    output_col="output")
    assert approx(df["bleu"].iloc[0], 0.0)


@test("rouge: per-row scores, not a corpus average")
def t_rouge_per_row():
    """score_rouge batches one call with use_aggregator=False.

    The risk in batching is getting back a single corpus-level number broadcast
    to every row, which would look plausible and be wrong. This asserts the
    rows actually differ.
    """
    import pandas as pd
    from llm_eval.scoring import score_rouge
    df = pd.DataFrame({
        "output":   ["Paris", "something else entirely"],
        "expected": ["Paris", "Paris"],
    })
    df = score_rouge(df, output_col="output")
    assert approx(df["rouge1"].iloc[0], 1.0)
    assert approx(df["rouge1"].iloc[1], 0.0)
    for col in ("rouge1", "rouge2", "rougeL"):
        assert df[col].dtype == float, f"{col} should be float, got {df[col].dtype}"


@test("scoring: degenerate frames do not crash")
def t_scoring_edge_cases():
    """An empty frame and a frame with no `expected` column.

    Both happen for real: an empty frame after filtering, and an unlabeled
    dataset being run through the ground-truth path by mistake.
    """
    import pandas as pd
    from llm_eval.scoring import score_bleu, score_rouge

    empty = pd.DataFrame({"output": [], "expected": []})
    empty = score_rouge(score_bleu(empty, output_col="output"), output_col="output")
    assert len(empty) == 0
    for col in ("bleu", "rouge1", "rouge2", "rougeL"):
        assert col in empty.columns

    noref = score_rouge(pd.DataFrame({"output": ["Paris"]}), output_col="output")
    assert approx(noref["rouge1"].iloc[0], 0.0)


@test("compute_metrics: JSON-serializable without a converter")
def t_metrics_json():
    """pandas aggregations return numpy scalars, which json.dumps rejects.

    compute_metrics casts at the source instead of relying on a convert_numpy
    walker in the caller. This test is what keeps that property from silently
    regressing the next time a statistic is added.
    """
    import pandas as pd
    from llm_eval.scoring import score_all, compute_metrics
    df = pd.DataFrame({
        "output":   ["Paris", "London"],
        "expected": ["Paris", "Paris"],
    })
    metrics = compute_metrics(score_all(df, include_semantic=False))
    json.dumps(metrics)  # raises TypeError if a numpy type leaked through


@test("scoring: failed rows are NaN, not 0")
def t_failed_rows_nan():
    """A row whose API call failed has no output, so it has no score.

    Scoring it 0 would blend "the model was wrong" with "the request never
    returned", and a rate-limit storm would then read as a quality drop.
    Blanking to NaN makes every mean "of the calls that succeeded".
    """
    import pandas as pd
    from llm_eval.scoring import score_all

    df = pd.DataFrame({
        "output":   ["Paris", None],
        "expected": ["Paris", "London"],
        "error":    [None, "429 rate limit"],
    })
    scored = score_all(df, include_semantic=False)
    assert approx(scored["exact_match"].iloc[0], 1.0)
    assert pd.isna(scored["exact_match"].iloc[1]), "failed row should be NaN"
    # And the surviving mean must ignore it entirely.
    assert approx(scored["exact_match"].mean(), 1.0)


@test("checkpoint: round-trip and torn final line")
def t_checkpoint():
    """JSONL survives a crash mid-write; a CSV or a JSON array does not.

    The torn-line case is the reason for the format: the process dying halfway
    through a write must not cost the rows already on disk.
    """
    from llm_eval import Checkpoint

    path = TMP / "ckpt.jsonl"
    ck = Checkpoint(str(path))
    ck.append({"prompt_id": "a", "output": "x"})
    ck.append({"prompt_id": "b", "output": "y"})
    assert len(ck.load()) == 2

    # Simulate a process killed mid-write: append a partial line.
    with open(path, "a") as f:
        f.write('{"prompt_id": "c", "outp')
    rows = Checkpoint(str(path)).load()
    assert len(rows) == 2, "torn final line should be dropped, earlier rows kept"
    assert [r["prompt_id"] for r in rows] == ["a", "b"]

    # A None path disables checkpointing without the caller branching on it.
    noop = Checkpoint(None)
    noop.append({"x": 1})
    assert noop.load() == []


@test("analysis: roc_auc matches known values")
def t_roc_auc():
    """Hand-rolled AUC via the Mann-Whitney identity.

    Checked against cases with an answer by inspection: perfect separation,
    perfect inversion, and an all-tied input where every pair contributes 0.5.
    Single-class input returns None rather than a misleading number.
    """
    from llm_eval import roc_auc
    assert approx(roc_auc([1, 2, 3, 4], [0, 0, 1, 1]), 1.0)
    assert approx(roc_auc([4, 3, 2, 1], [0, 0, 1, 1]), 0.0)
    assert approx(roc_auc([1, 1, 1, 1], [0, 0, 1, 1]), 0.5)  # midranks
    assert roc_auc([1, 2, 3], [1, 1, 1]) is None, "single class -> None"


@test("analysis: agreement + disagreements on a synthetic frame")
def t_agreement():
    """The meta-evaluation layer, exercised without paying for a judge run.

    The frame is built so the judge tracks exact_match perfectly, which should
    give separation > 0 and AUC 1.0. find_disagreements must then return the
    one row that was planted to conflict.
    """
    import pandas as pd
    from llm_eval import judge_agreement, find_disagreements

    df = pd.DataFrame({
        "prompt_id":   [f"q{i}" for i in range(6)],
        "input":       ["q"] * 6,
        "output":      ["a"] * 6,
        "expected":    ["a"] * 6,
        "exact_match": [1, 1, 1, 0, 0, 0],
        "judge_score": [5.0, 5.0, 4.0, 2.0, 1.0, 5.0],  # last row conflicts
    })
    rep = judge_agreement(df, gt_cols=("exact_match",))
    assert rep is not None
    # Per-metric stats are nested under "comparisons"; the top level holds
    # judge_col, n, and the warnings list.
    em = rep["comparisons"]["exact_match"]
    assert em["n_correct"] == 3 and em["n_incorrect"] == 3
    # Correct rows average 5,5,4 and incorrect 2,1,5 -> separation exactly 2.0.
    assert approx(em["separation"], 2.0, tol=1e-3), em["separation"]
    assert em["separation"] > 0, "judge should track correctness here"
    # One of the three incorrect rows outranks a correct one, so AUC is 7/9.
    assert approx(em["auc"], 7 / 9, tol=1e-3), em["auc"]
    # Small-n always warns; that guardrail should not silently disappear.
    assert any("small" in w for w in rep["warnings"])

    dis = find_disagreements(df, gt_col="exact_match", top_n=3)
    assert not dis.empty
    # The planted row: exact_match 0 but judge gave a 5, so it must rank first
    # and be labelled as the judge being generous.
    assert dis.iloc[0]["prompt_id"] == "q5", dis[["prompt_id", "gap"]].to_dict("records")
    assert dis.iloc[0]["direction"] == "judge_generous"
    assert approx(dis.iloc[0]["gap"], 1.0, tol=1e-3)


@test("analysis: variance no-ops without sample_index")
def t_variance_noop():
    """--n-samples was cut, so no frame has sample_index any more.

    variance_report must return None rather than raising, since run_eval.py
    still calls it unconditionally.
    """
    import pandas as pd
    from llm_eval import variance_report
    df = pd.DataFrame({"prompt_id": ["a", "b"], "exact_match": [1, 0]})
    assert variance_report(df) is None


# =============================================================================
# LIVE TESTS — real OpenRouter calls through the CLI
# =============================================================================

@test("live: ground truth pipeline end to end", live=True)
def t_live_ground_truth():
    """The main path: inference -> ground-truth metrics -> artifacts.

    Asserts the plumbing (row count, no failures, artifacts on disk), not the
    score. A model having a bad day must not fail this.
    """
    import pandas as pd
    raw = pd.read_csv(gt_run())
    scored = pd.read_csv(TMP / "live_gt_scored.csv")
    assert len(raw) == 10, f"expected 10 rows, got {len(raw)}"
    assert raw["error"].isna().all(), "no row should have errored"
    assert "sample_index" not in raw.columns, "sample_index should be gone"
    for col in ("exact_match", "contains_expected", "token_f1", "bleu", "rouge1"):
        assert col in scored.columns, f"missing score column: {col}"
    assert (TMP / "live_gt_metrics.json").exists()


@test("live: judge + agreement on the hard set", live=True)
def t_live_judge_agreement():
    """The hard set produces both correct and incorrect rows, which is what
    makes agreement measurable at all. The easy demo set is answered 10/10 and
    gives ground truth zero variance.

    Also covers the judge writing a disagreements CSV.
    """
    import pandas as pd
    judge_run()
    scored = pd.read_csv(TMP / "live_judge_scored.csv")
    assert len(scored) == 16
    assert "judge_score" in scored.columns
    assert scored["judge_score"].notna().any(), "judge produced no scores at all"

    metrics = json.loads((TMP / "live_judge_metrics.json").read_text())
    assert "judge_agreement" in metrics, "agreement should run automatically"
    assert "variance" not in metrics, "variance should never appear now"
    # The disagreement rows are the artifact worth reading by hand.
    assert (TMP / "live_judge_disagreements.csv").exists()


@test("live: judge with no ground truth", live=True)
def t_live_open_ended():
    """The reference-free path. No `expected` column exists, so score_all must
    be skipped and the judge must still run.
    """
    import pandas as pd
    cli("--dataset", "datasets/open_ended_demo.json", "--judge",
        "--output-dir", TMP, "--run-id", "live_open")
    scored = pd.read_csv(TMP / "live_open_scored.csv")
    assert len(scored) == 5
    assert "judge_score" in scored.columns
    assert "exact_match" not in scored.columns, "no ground truth to score against"


@test("live: --cot extracts the ANSWER: line", live=True)
def t_live_cot():
    """CoT is the flag that was kept, so it needs a real test.

    The floor is 80% format compliance rather than 100%: an occasional model
    slip is a model fact, but a collapse to zero means extract_final_answer or
    the prompt wiring broke.
    """
    import pandas as pd
    cli("--dataset", "datasets/ground_truth_hard.json", "--cot",
        "--output-dir", TMP, "--run-id", "live_cot")
    df = pd.read_csv(TMP / "live_cot_scored.csv")
    assert "answer" in df.columns and "answer_extracted" in df.columns
    rate = df["answer_extracted"].mean()
    assert rate >= 0.8, f"ANSWER: extraction collapsed to {rate:.0%}"
    # The scorers must read `answer`, not the full reasoning text.
    from llm_eval.scoring import prediction_column
    assert prediction_column(df) == "answer"


@test("live: partial resume calls only what is missing", live=True)
def t_live_resume_partial():
    """The keying that changed when row_key was removed.

    Seeds a checkpoint with 4 of 10 rows and asserts exactly 6 calls are made,
    the final frame is in dataset order, and nothing is duplicated. Order is
    the subtle one: results come back out of order from the thread pool, and
    since the gold answer rides on each row, a reordering bug would silently
    pair outputs with the wrong expected answer instead of crashing.
    """
    import pandas as pd

    # Reuse rows already paid for by the shared ground-truth run.
    gt_run()
    src = TMP / "live_gt_raw.csv.jsonl"
    dst = TMP / "resume_raw.csv.jsonl"
    lines = src.read_text().splitlines(keepends=True)
    dst.write_text("".join(lines[:4]))

    out = cli("--dataset", "datasets/ground_truth_demo.json", "--resume",
              "--output-dir", TMP, "--run-id", "resume")
    assert "Resuming: 4 rows" in out, out[:600]
    assert "Running 6/10 prompts" in out, out[:600]

    df = pd.read_csv(TMP / "resume_raw.csv")
    expected_order = [r["prompt_id"] for r in
                      json.loads((REPO / "datasets/ground_truth_demo.json").read_text())]
    assert list(df["prompt_id"]) == expected_order, "dataset order not preserved"
    assert not df["prompt_id"].duplicated().any()
    assert df["output"].notna().all()


@test("live: full resume makes zero calls", live=True)
def t_live_resume_full():
    """A complete checkpoint means nothing to do.

    Cheap but important: this is the behavior that stops a re-run from silently
    paying for the whole dataset twice.
    """
    # Seed a complete checkpoint from the shared run, so this test does not
    # depend on the partial-resume test having run first.
    gt_run()
    shutil.copyfile(TMP / "live_gt_raw.csv.jsonl", TMP / "fullresume_raw.csv.jsonl")

    out = cli("--dataset", "datasets/ground_truth_demo.json", "--resume",
              "--output-dir", TMP, "--run-id", "fullresume")
    assert "Resuming: 10 rows" in out, out[:600]
    assert "Running 0/10 prompts" in out, out[:600]


@test("live: integer prompt_id survives the checkpoint round-trip", live=True)
def t_live_int_ids():
    """The reason resume keys are str(prompt_id).

    A prompt_id that is an int in the source JSON comes back from JSONL as
    whatever json wrote. Without the str() coercion the resume map misses on
    every row and silently re-pays for the entire dataset — no error, just a
    doubled bill. Asserted by resuming and requiring zero calls.
    """
    fixture = TMP / "int_ids.json"
    fixture.write_text(json.dumps([
        {"prompt_id": 1, "input": "What is 2+2? Answer with just the number.", "expected": "4"},
        {"prompt_id": 2, "input": "Capital of France? Just the city name.", "expected": "Paris"},
    ]))

    cli("--dataset", fixture, "--output-dir", TMP, "--run-id", "intids")
    out = cli("--dataset", fixture, "--resume", "--output-dir", TMP, "--run-id", "intids")
    assert "Running 0/2 prompts" in out, out[:600]


@test("live: judge resume reuses judgments", live=True)
def t_live_judge_resume():
    """The judge has its own checkpoint and its own keying, changed alongside
    the harness. Judgments are as expensive as inference, so they resume too.
    """
    raw = judge_run()
    cli("--score-only", "--results", raw, "--judge",
        "--output-dir", TMP, "--run-id", "jresume")
    out = cli("--score-only", "--results", raw, "--judge", "--resume",
              "--output-dir", TMP, "--run-id", "jresume")
    assert "Resuming: 16 judgments" in out, out[:600]
    assert "Running LLM judge on 0/16" in out, out[:600]


@test("live: --score-only rescores with no inference", live=True)
def t_live_score_only():
    """The iteration loop for metric and rubric work.

    The expensive part is already on disk, so tuning a metric should cost
    nothing. Asserted by requiring no inference banner in the output.
    """
    import pandas as pd
    out = cli("--score-only", "--results", judge_run(),
              "--output-dir", TMP, "--run-id", "scoreonly")
    assert "Running" not in out.split("STEP 2")[0], "should not have run inference"
    df = pd.read_csv(TMP / "scoreonly_scored.csv")
    assert len(df) == 16 and "exact_match" in df.columns


@test("live: no sample_index in any artifact", live=True)
def t_live_no_sample_index():
    """A schema sweep over everything the suite just wrote.

    Cheap insurance that the cut did not leave the column alive in one code
    path (e.g. only on the judge checkpoint) where a targeted test would miss it.
    """
    import pandas as pd
    offenders = []
    for path in sorted(TMP.glob("*.csv")):
        if "sample_index" in pd.read_csv(path).columns:
            offenders.append(path.name)
    for path in sorted(TMP.glob("*.jsonl")):
        if "sample_index" in path.read_text():
            offenders.append(path.name)
    assert not offenders, f"sample_index still present in: {offenders}"


# =============================================================================
# Runner
# =============================================================================

def main():
    global TMP

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--offline", action="store_true",
                        help="Skip tests that make real API calls.")
    parser.add_argument("-k", metavar="SUBSTRING", default=None,
                        help="Only run tests whose name contains this.")
    parser.add_argument("--keep", action="store_true",
                        help="Keep the temp artifact directory for inspection.")
    args = parser.parse_args()

    selected = [
        t for t in TESTS
        if not (args.offline and t["live"])
        and (args.k is None or args.k.lower() in t["name"].lower())
    ]
    if not selected:
        print("No tests matched.")
        return 1

    # A live run needs a key. Failing here is far friendlier than 10 identical
    # auth errors surfacing one test at a time.
    if any(t["live"] for t in selected) and not os.getenv("OPENROUTER_API_KEY"):
        from dotenv import load_dotenv
        load_dotenv(REPO / ".env")
        if not os.getenv("OPENROUTER_API_KEY"):
            print("OPENROUTER_API_KEY is not set; use --offline or set it in .env")
            return 1

    TMP = Path(tempfile.mkdtemp(prefix="evaltest_"))
    n_live = sum(1 for t in selected if t["live"])
    print(f"\n{'='*64}")
    print(f"HARNESS TEST SUITE  ({len(selected)} tests, {n_live} live)")
    print(f"artifacts: {TMP}")
    print(f"{'='*64}\n")

    failures = []
    started = time.time()
    for t in selected:
        tag = "live" if t["live"] else "    "
        print(f"  [{tag}] {t['name']:<52}", end="", flush=True)
        t0 = time.time()
        try:
            t["fn"]()
            print(f" PASS  {time.time()-t0:5.1f}s")
        except Exception as e:
            print(f" FAIL  {time.time()-t0:5.1f}s")
            failures.append((t["name"], e, traceback.format_exc()))

    print(f"\n{'='*64}")
    if failures:
        print(f"{len(failures)} FAILED / {len(selected)} run   ({time.time()-started:.1f}s)")
        print(f"{'='*64}")
        for name, err, tb in failures:
            print(f"\n--- {name} ---")
            # The assertion message is the useful part; the traceback is noise
            # unless the failure was an unexpected exception rather than an
            # assert, so only the last few frames are shown.
            print("\n".join(tb.strip().splitlines()[-12:]))
    else:
        print(f"ALL {len(selected)} PASSED   ({time.time()-started:.1f}s)")
    print(f"{'='*64}\n")

    if args.keep:
        print(f"Artifacts kept at {TMP}\n")
    else:
        shutil.rmtree(TMP, ignore_errors=True)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
