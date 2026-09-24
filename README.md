# llm-eval-harness

A small, readable harness for evaluating LLM outputs. Runs a dataset of prompts against any model and scores the results two independent ways — against ground-truth answers, and with an LLM judge that reads the probability distribution over its own score token rather than just parsing a digit.

Built to be understood end to end rather than to be comprehensive. Every metric is either a few lines you can read or a documented call into a standard library, and every non-obvious decision carries an inline comment explaining the failure mode it prevents.

```
harness.py ──> DataFrame ──┬──> scoring.py   (exact match, token F1, BLEU/ROUGE, semantic)
  one row per call         └──> judge.py     (CoT rubric + logprob-weighted score)
```

## Why another one

Most eval libraries ask you to express your task in their abstractions. This one is about 1,500 lines of plain Python over a pandas DataFrame — 3,100 counting the comments and docstrings, which is the intended ratio — so adapting it to an unusual task means editing a function rather than finding the right subclass. Three things it does that are easy to get wrong:

**You never pay for the same token twice.** Raw outputs are written before anything is scored, every row is checkpointed as it lands, and `--score-only` rescores a finished run offline. A scoring bug, a changed metric, or a crash at row 190 of 200 all cost zero API calls to recover from — which matters most on exactly the runs that are expensive enough to care about.

**Failed calls are `NaN`, not `0`.** Scoring an errored row as zero blends "the model was wrong" with "the request never returned," so a rate-limit storm reads as a quality drop. Every mean here is "of the calls that succeeded," printed with its `n`, with `failure_rate` reported separately.

**Metric defaults are checked, not trusted.** BLEU at its default order of 4 scores a *correct* one-word answer `0.0000`, because there are no 4-grams to match. With smoothing enabled it scores a *wrong* one-word answer `0.8409`. Both are documented, and `score_bleu` is configured around them.

## Quickstart

```bash
git clone https://github.com/zaif-c/llm-eval-harness.git
cd llm-eval-harness
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env     # then add your OpenRouter key
python run_eval.py       # runs the bundled 10-question demo set
```

Everything goes through [OpenRouter](https://openrouter.ai), so one API key reaches every provider and the model id picks which one answers:

```bash
python run_eval.py --model openai/gpt-4o-mini
python run_eval.py --model anthropic/claude-sonnet-4
python run_eval.py --model google/gemini-2.5-flash
```

## Usage

```bash
# Ground truth only — dataset has an `expected` column
python run_eval.py --dataset datasets/ground_truth_hard.json

# Add the LLM judge — both score families land on the same rows
python run_eval.py --dataset datasets/ground_truth_hard.json --judge

# Open-ended data, no `expected` column — judge is the whole score
python run_eval.py --dataset datasets/open_ended_demo.json --judge

# Reasoning tasks: model explains, metrics score only the final ANSWER: line
python run_eval.py --dataset datasets/ground_truth_hard.json --cot

# Rescore a finished run without paying for inference again
python run_eval.py --score-only --results results/<run_id>_raw.csv

# Continue an interrupted run from its checkpoint
python run_eval.py --dataset data.json --run-id myrun --resume
```

A dataset is JSON or CSV. Records need `prompt_id` and `input`; add `expected` for ground-truth scoring and `category` for a per-category breakdown. Any other column rides along onto the output.

```json
[{"prompt_id": "q1", "input": "What is the capital of France?", "expected": "Paris", "category": "geography"}]
```

## What you get

Each run writes `<run_id>_raw.csv` (before scoring, so a scoring bug never costs API calls), `<run_id>_scored.csv`, `<run_id>_metrics.json`, and a JSONL checkpoint per row for `--resume`.

```
exact_match:
  mean: 0.4375  (std: 0.5123, n=16)
  range: [0.0000, 1.0000]  p25=0.0000  median=0.0000  p75=1.0000
  distribution: 0.0:9  1.0:7

contains_expected:
  mean: 0.8750  (std: 0.3416, n=16)
  range: [0.0000, 1.0000]  p25=1.0000  median=1.0000  p75=1.0000
  distribution: 0.0:2  1.0:14
```

Those two lines together are worth more than either alone. A large gap between `exact_match` and `contains_expected` means the model is answering correctly but in a sentence, so exact match is measuring *output format* rather than correctness — the fix is `--cot` or a tighter prompt, not a different model. On this run the model was right far more often than 44%.

Distributions print alongside every mean, because the two most common outcomes here are bimodal (exact match is really a pass rate) and saturated (a helpfulness judge giving everything a 5). Both look unremarkable as a mean and obvious as a histogram.

## Metrics

| Metric | Column | Notes |
| --- | --- | --- |
| Exact match | `exact_match` | Normalized string equality. Strict, and sensitive to answer format. |
| Contains | `contains_expected` | Gold answer appears in the output. The right call when the model wraps a short answer in prose. |
| Token F1 | `token_precision/recall/f1` | Multiset overlap, the SQuAD measure. |
| Jaccard | `fuzzy_score` | Token-set overlap, order and duplicates ignored. |
| BLEU | `bleu` | Via `evaluate`, `max_order` capped at the shorter text. |
| ROUGE | `rouge1/2/L` | Via `evaluate`, one batched call for per-row scores. |
| Character | `char_similarity` | `difflib` ratio — catches a one-character miss. |
| Semantic | `semantic_score` | Cosine over `all-MiniLM-L6-v2`, negatives clipped to 0. |
| LLM judge | `judge_score` | Rubric + chain of thought, scored from logprobs. |

The judge reads the probability distribution over the score token rather than just parsing the digit, so a 60/40 split between `4` and `3` becomes **3.6** instead of a flat 4. `judge_raw_score` keeps the parsed value so the gap is visible. On a provider without logprob support it falls back to text parsing and records which method was used.

## Reliability

Timed and retried per request (60s default, exponential backoff with jitter), concurrent by default (8 threads), and checkpointed to JSONL as each row completes — a crash at row 190 of 200 does not discard 190 paid calls. Errors that will fail identically on every retry (bad key, bad model id) skip the backoff entirely and fail in seconds rather than minutes. Failures are bucketed by cause, because a rate limit and a typo'd model id call for different fixes.

## Configuration

`.env`, or the environment:

| Variable | Required | Purpose |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | yes | OpenRouter key. |
| `EVAL_MODEL` | yes | Model under test, e.g. `openai/gpt-4o-mini`. |
| `JUDGE_MODEL` | no | Judge model. Falls back to `EVAL_MODEL`. |
| `EMBEDDING_MODEL` | no | Defaults to `all-MiniLM-L6-v2`. |

Run `python run_eval.py --help` for the full flag list. Python 3.11+.

## Layout

```
llm_eval/
  harness.py    inference: retry, concurrency, checkpointing. Knows nothing about scoring.
  scoring.py    reference-based metrics. Never calls the API.
  judge.py      LLM-as-judge with logprob-weighted scoring.
run_eval.py     CLI. The only place that orders the layers.
datasets/       demo datasets.
```

The dependency direction is one-way: `judge.py` imports the retry policy and checkpoint from `harness.py`, and nothing imports back down into the inference layer. Inference bugs and metric bugs fail in different places.

[`FRAMEWORK.md`](FRAMEWORK.md) is the deep version — architecture, execution order, and the reasoning behind each decision, including the ones that turned out to be wrong.

## License

MIT
