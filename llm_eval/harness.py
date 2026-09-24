"""
LLM Evaluation Harness
======================
Scoring-agnostic batch runner with retry/backoff.
Logs every call to a pandas DataFrame.

WHERE THIS FILE SITS IN THE PIPELINE
------------------------------------
    run_eval.py  ->  harness.py  ->  scoring.py   (ground-truth metrics)
                                 \\-> judge.py     (LLM-as-judge metrics)

This file is the *inference* layer and nothing else. It sends prompts, collects
responses, and records what happened. It deliberately does not know what a
correct answer looks like. That separation is the reason the same harness can
be pointed at a labeled QA set (scored by scoring.py) or a set of open-ended
prompts (scored by judge.py) without changing a line in here.

The unit of work is one prompt -> one row. The unit of output is a pandas
DataFrame, one row per prompt, in dataset order. Everything downstream consumes
that DataFrame, so the column names in `batch_run`'s `standard_cols` are the
contract between this file and the rest of the repo.

Two pieces in this file are shared upward rather than being inference-specific:
`call_with_retry` (the retry policy) and `Checkpoint` (crash-survival). judge.py
imports both so there is exactly one implementation of each, not two that drift.

Usage:
    from llm_eval import HarnessConfig, batch_run

    # Model is an OpenRouter id. The prefix selects the provider.
    config = HarnessConfig()  # EVAL_MODEL from the environment
    config = HarnessConfig(model="anthropic/claude-sonnet-4")
    config = HarnessConfig(model="google/gemini-2.5-flash")

    prompts = [
        {"prompt_id": "q1", "input": "What is 2+2?"},
        {"prompt_id": "q2", "input": "Capital of France?"},
    ]
    results_df = batch_run(prompts, config)
"""

import json
import os
import re
import threading
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from openai import (
    OpenAI,
    # The exception hierarchy matters here, not just the names. Every one of
    # these subclasses APIError, which is why TRANSIENT_ERRORS and
    # PERMANENT_ERRORS below have to be checked in a specific order.
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)

# Reads .env into os.environ. Called at import time so that a bare
# `from llm_eval import ...` in a notebook or a script picks up the API key
# without the caller having to remember to load it first.
load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def make_client() -> OpenAI:
    """One OpenAI-compatible client for every provider OpenRouter serves.

    This is the whole provider-agnostic story: OpenRouter speaks the OpenAI
    wire format, so the official OpenAI SDK works against it unchanged. Point
    the SDK at a different base URL and the model id string ("openai/...",
    "anthropic/...", "google/...") selects which provider actually runs the
    request. No per-provider client classes, no per-provider response parsing
    beyond what `message_text` handles below.

    Raises rather than returning None on a missing key, because every caller
    needs a working client and a None here would surface as a confusing
    AttributeError several frames later.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        # Fail here instead of later to avoid silent failures.
        raise ValueError("OPENROUTER_API_KEY is not set")
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


# Appended to the user message only when chain-of-thought is on (see
# `_user_message`). The "ANSWER:" sentinel is the contract with
# `extract_final_answer`: the model writes its reasoning freely, then marks the
# final answer on its own line, and scoring compares only that line to the gold
# answer. Without the sentinel, exact match would compare a paragraph of
# reasoning to the word "Paris" and score a correct answer as wrong.
COT_INSTRUCTION = (
    "Think step by step before answering. "
    "Write the reasoning first, then end with a final line in exactly this format:\n"
    "ANSWER: <answer>"
)


# =============================================================================
# Response parsing
#
# These three functions exist because "get the text out of the response" is not
# uniform across providers once OpenRouter is in the middle. The OpenAI shape is
# `message.content` as a plain string, but some providers return a list of
# content parts, and reasoning models may put their output in a separate
# `reasoning` field with `content` left empty.
# =============================================================================

def _parts_to_text(content) -> str:
    """Normalize provider content that is either a string or a list of parts.

    Three cases, in the order they are checked:
      1. A plain string (the common OpenAI shape) -> return it stripped.
      2. Anything that is not a list -> return "" rather than str()-ing it,
         because str(None) would put the literal text "None" in the output
         column and it would be scored as if the model had said "None".
      3. A list of parts -> concatenate the text of each part. Parts may
         themselves be raw strings, dicts with a "text" key, or objects with a
         `.text` attribute, depending on the provider, so all three are handled.
    """
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    chunks = []
    for part in content:
        if isinstance(part, str):
            chunks.append(part)
        elif isinstance(part, dict):
            # `or ""` guards against a present-but-null "text" key, which would
            # otherwise stringify to "None".
            chunks.append(str(part.get("text") or ""))
        else:
            chunks.append(str(getattr(part, "text", "") or ""))
    return "".join(chunks).strip()


def message_reasoning(message) -> Optional[str]:
    """Provider-side reasoning, when the model returns it separately from content.

    Different providers name this field differently, so both spellings are
    tried. This is stored in its own `reasoning` column rather than being merged
    into `output`, so that a reasoning model's scratchpad never gets compared to
    the gold answer by accident.
    """
    for attr in ("reasoning", "reasoning_content"):
        value = getattr(message, attr, None)
        # Checks the type as well as truthiness: a provider returning a
        # structured reasoning object rather than a string should fall through
        # to the next attribute rather than be returned as-is.
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def message_text(message) -> Optional[str]:
    """Assistant text. Uses content, then provider reasoning if content is empty.

    The fallback ordering is the important part. Content wins when it exists,
    because that is the actual answer. Reasoning is used only when content came
    back empty, which happens with some reasoning models; without the fallback
    those rows would silently record a null output and look like failures.
    """
    text = _parts_to_text(getattr(message, "content", None))
    if text:
        return text
    return message_reasoning(message)


def extract_final_answer(text: Optional[str]) -> tuple[Optional[str], bool]:
    """Pull the last ANSWER: line out of a chain-of-thought response.

    Returns (answer, extracted). extracted is False when the model did not
    follow the format; the caller then scores the full output.

    Regex flags:
      (?i) case-insensitive, so "Answer:" and "ANSWER:" both match.
      (?m) multiline, so ^ anchors to the start of each line rather than the
           start of the string. This is what makes "a line beginning with
           ANSWER:" expressible at all.
      [ \\t]* allows the model to indent the line without breaking the match.

    The last match wins, not the first, because a model asked to reason will
    often mention the format mid-reasoning ("...so the ANSWER: should be the
    capital..."). The real answer is the final one.

    Falling back to the full text on a miss, rather than returning None, keeps a
    non-compliant response scoreable: a model that just answers "Paris" without
    the sentinel still gets credit. The boolean is what separates "the model got
    it wrong" from "the model ignored the output format", and it is aggregated
    as `answer_extracted_rate` in scoring.compute_metrics.
    """
    if not text:
        return None, False
    matches = re.findall(r"(?im)^[ \t]*ANSWER:[ \t]*(.+)$", text)
    if not matches:
        return text, False
    return matches[-1].strip(), True


def token_usage(response) -> dict:
    """Prompt/completion token counts, when the provider reports them.

    Not every OpenRouter provider returns a usage block, so missing values stay
    None rather than being coerced to 0, which would understate real usage in
    the aggregate. Downstream, pandas reads None as NaN and drops it from the
    mean, so a provider that omits usage shows up as a smaller sample size
    rather than as a pile of free calls.

    `getattr` with a default is used twice over: once in case `usage` itself is
    absent (leaving `usage = None`), and once per field, since getattr(None, x,
    None) safely returns None rather than raising.
    """
    usage = getattr(response, "usage", None)
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _user_message(input_text: str, config: "HarnessConfig") -> str:
    """Build the user-turn content, appending the CoT instruction if enabled.

    Note what is *not* happening: the harness records the original `input_text`
    on the row, not this modified string. The CoT instruction exists only on the
    wire. That keeps the logged dataset comparable between a `--cot` run and a
    plain run, so the two are diffable on the same prompt column.
    """
    if not config.chain_of_thought:
        return input_text
    return f"{input_text.rstrip()}\n\n{COT_INSTRUCTION}"


def _get_default_model() -> str:
    """Model id from the environment: EVAL_MODEL, then DEFAULT_MODEL, then "".

    Returns an empty string rather than raising, so that HarnessConfig can own
    the error message (it knows about the `model=` kwarg; this function does not).
    """
    return os.getenv("EVAL_MODEL", os.getenv("DEFAULT_MODEL", ""))


# =============================================================================
# Configuration
#
# One dataclass holding every knob, passed down through batch_run -> run_single.
# The alternative, threading a dozen keyword arguments through each layer, makes
# adding a setting a three-file change; here it is a one-line change.
# =============================================================================

@dataclass
class HarnessConfig:
    """Configuration for the inference harness."""
    model: str = ""  # Set from env or must be provided
    temperature: float = 0.0          # Deterministic by default for evals
    max_tokens: int = 1024
    system_prompt: Optional[str] = None
    chain_of_thought: bool = False    # Ask for reasoning, score the ANSWER: line

    # Per-request timeout in seconds. The SDK default is 600s, long enough for
    # one hung call to stall a timed run. A timeout raises APITimeoutError,
    # which the retry loop treats as transient.
    timeout: float = 60.0

    # Retry settings. Worst case for a single row is roughly
    # timeout * max_retries plus the accumulated backoff, so these numbers set
    # the ceiling on how long one bad prompt can hold up a batch.
    max_retries: int = 5
    base_delay: float = 1.0           # Base delay in seconds
    max_delay: float = 60.0           # Cap on exponential backoff
    jitter: float = 0.5               # Random jitter factor (0-1)

    # Concurrency. 1 keeps the batch sequential. Higher values overlap the
    # waiting, which is nearly all of a batch's wall clock. Too high and the
    # provider rate-limits, which costs more time than it saves.
    max_workers: int = 8

    # Output settings
    output_csv: Optional[str] = None  # If set, writes results to this path

    # Incremental persistence. Each finished row is appended here as JSONL so a
    # crash mid-batch does not discard completed calls. Defaults to
    # <output_csv>.jsonl. With resume=True, prompt_ids already in the file are
    # skipped instead of being paid for twice.
    checkpoint_path: Optional[str] = None
    resume: bool = False

    def __post_init__(self):
        """Runs automatically after the dataclass __init__.

        Two jobs: resolve the model from the environment when the caller did not
        pass one, and derive the checkpoint path from the CSV path. Doing both
        here rather than in batch_run means the resolved values are visible on
        the config object itself, which is what run_eval.py prints back to the
        user before the run starts.
        """
        if not self.model:
            self.model = _get_default_model()
        # Checked again after the env lookup, since the env lookup may also have
        # produced "". Failing here is deliberate: a run with no model is
        # unrecoverable, and failing at construction is far cheaper than failing
        # once per row, N rows deep, after paying for nothing.
        if not self.model:
            raise ValueError(
                "Model must be specified via HarnessConfig(model=...) or "
                "EVAL_MODEL/DEFAULT_MODEL environment variable"
            )
        # `is None` rather than falsy, so an explicit checkpoint_path="" can
        # disable checkpointing without being silently overwritten here.
        if self.checkpoint_path is None and self.output_csv:
            self.checkpoint_path = f"{self.output_csv}.jsonl"


# =============================================================================
# Retry policy, shared by the harness and the judge
#
# This section is imported by judge.py. There is one definition of "what is
# worth retrying" and one backoff implementation in the repo, because two copies
# drift and "how does retry work here" should have a single answer.
# =============================================================================

# Transient API failures. APITimeoutError subclasses APIConnectionError; it is
# listed for clarity. Anything outside this tuple is a bug in our code and is
# allowed to crash rather than be retried five times and reported as flakiness.
# APIError sits at the bottom of the SDK hierarchy, so listing it here means
# "retry anything the API raised" — which is why PERMANENT_ERRORS below has to
# be carved back out of it.
TRANSIENT_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError, APIError)

# Errors that will fail identically on every retry: a bad key, a model id that
# does not exist, a malformed request. These are subclasses of APIError, so
# without this they would be caught as transient and cost a full backoff
# sequence per prompt before surfacing a typo.
PERMANENT_ERRORS = (
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    BadRequestError,
)


def classify_error(error: Optional[str]) -> Optional[str]:
    """Bucket an error string so failures can be counted by cause.

    Matched on the message rather than the exception class because the string
    is what survives into the CSV and the checkpoint; a reloaded run has no
    exception object left to inspect. Order matters: "rate limit" is checked
    before the generic connection bucket, since a 429 often mentions both.

    The result is stored per row as `error_type` and aggregated by
    scoring.compute_metrics into `failures_by_type`. The point of the bucketing
    is that the response differs by cause: rate limits mean lower --max-workers,
    a model_not_found means a typo in the model id, content_filter means the
    dataset itself is tripping a safety layer.

    Returns None for a non-error so that successful rows get a null error_type
    rather than a bogus "other", which would corrupt the failure counts.
    """
    if not error:
        return None
    text = str(error).lower()
    # Each check below is ordered by specificity. The first match wins, so a
    # more specific pattern must come before a more general one that would also
    # match the same message.
    if "rate limit" in text or "429" in text or "quota" in text:
        return "rate_limit"
    if "timed out" in text or "timeout" in text:
        return "timeout"
    # Requires both halves, since "content" alone appears in far too many
    # unrelated messages to be a reliable signal on its own.
    if "content" in text and ("filter" in text or "policy" in text):
        return "content_filter"
    if "auth" in text or "api key" in text or "401" in text or "403" in text:
        return "auth"
    # Checked before the generic 400 bucket: a bad model id comes back as a 404
    # from OpenAI but as a 400 from OpenRouter.
    if ("not found" in text or "404" in text or "no endpoints" in text
            or "not a valid model" in text or "invalid model" in text
            or "unknown model" in text):
        return "model_not_found"
    if "connection" in text or "network" in text:
        return "connection"
    # Deliberately last of the real buckets: "400" and "invalid" are broad
    # enough to swallow several of the cases above if checked earlier.
    if "400" in text or "invalid" in text:
        return "bad_request"
    return "other"


def ensure_unique_prompt_ids(prompt_ids) -> None:
    """Raise if any prompt_id repeats. Called before any call is paid for.

    Both batch_run and judge_batch file results into a dict keyed by
    str(prompt_id). A duplicate id therefore overwrites rather than collides:
    two rows end up sharing one result. Nothing raises, the frame keeps its
    shape and its row count, and the scores are wrong — which is what makes it
    worth a guard rather than a comment.

    Compared as str() for the same reason the result dicts are keyed that way:
    an int 1 and a string "1" are the same key downstream, so they are the same
    duplicate here.

    Rejects rather than de-duplicating or renaming. A dataset with repeated ids
    is a dataset bug, and silently picking one of the two rows would be the same
    class of quiet wrongness this exists to prevent.
    """
    seen: set[str] = set()
    dupes: list[str] = []
    for prompt_id in prompt_ids:
        key = str(prompt_id)
        if key in seen and key not in dupes:
            dupes.append(key)
        seen.add(key)
    if dupes:
        shown = ", ".join(dupes[:5])
        more = f" (+{len(dupes) - 5} more)" if len(dupes) > 5 else ""
        raise ValueError(
            f"Duplicate prompt_id values: {shown}{more}. "
            "Ids must be unique: results are keyed by prompt_id, so duplicates "
            "would silently share one result row."
        )


def backoff_delay(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: float = 0.5,
) -> float:
    """Exponential backoff with jitter: min(base * 2^attempt, max_delay) ± jitter/2.

    Jitter matters under concurrency. Without it, workers that hit the same
    rate limit retry in lockstep and trigger it again together.

    Walking the three lines:
      - `min(...)` caps unbounded doubling. At base 1.0 the raw sequence is
        1, 2, 4, 8, 16, 32, 64...; the cap stops it at max_delay.
      - `spread` is the jitter *window*, expressed as a fraction of the delay,
        so the randomness scales with the wait rather than being a fixed ±0.25s
        that is meaningless at a 60s delay.
      - `max(0.1, ...)` floors the result. With jitter=1.0 the random term could
        otherwise produce a near-zero or negative delay, which would turn the
        backoff into a hot loop.
    """
    delay = min(base_delay * (2 ** attempt), max_delay)
    spread = delay * jitter
    return max(0.1, delay + random.uniform(-spread / 2, spread / 2))


def call_with_retry(
    send,
    *,
    label: str,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: float = 0.5,
    verbose: bool = True,
):
    """Run `send()` until it succeeds or transient retries are exhausted.

    Returns (result, error). On success error is None; on failure result is
    None and error is the last exception's message, so the caller can record
    the failure on its row and let the batch continue. Permanent errors skip
    the backoff entirely and return on the first attempt.

    Returning instead of raising is the key design choice. A raise would
    propagate out of the worker thread and kill the whole batch over one dead
    prompt; returning lets `run_single` write a failure row and lets the other
    N-1 prompts finish. The cost is that every caller must check the error half
    of the tuple, which both `run_single` and `judge_single` do.

    `send` is a zero-argument callable rather than a request payload so that
    this function stays completely ignorant of what is being sent. That is what
    lets judge.py reuse it for a differently-shaped request, and it is also what
    lets `run_single` start its latency timer inside the closure.

    `label` is only used for the progress line, so a retry message says which
    prompt is retrying rather than just "retrying".
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            # The happy path returns straight out of the loop.
            return send(), None
        except PERMANENT_ERRORS as e:
            # Ordered before the transient clause on purpose. These are all
            # subclasses of APIError, so if TRANSIENT_ERRORS were checked first
            # it would match them and we would back off five times over a typo.
            return None, str(e)
        except TRANSIENT_ERRORS as e:
            last_error = str(e)
            # Skips the sleep after the final attempt. Without this check the
            # function would wait out a full backoff delay and then immediately
            # give up, which is pure wasted wall clock.
            if attempt < max_retries - 1:
                delay = backoff_delay(attempt, base_delay, max_delay, jitter)
                if verbose:
                    print(f"  [Retry {attempt + 1}/{max_retries - 1}] "
                          f"{label}: {type(e).__name__}, waiting {delay:.1f}s")
                time.sleep(delay)
        # Note there is no bare `except Exception`. Anything not in either tuple
        # is a bug in our own code (a KeyError, a TypeError) and is allowed to
        # propagate. Swallowing it here would retry it five times and then
        # report our own bug as an API failure.
    return None, last_error


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

    This always returns a row, never raises. The two exit paths below (success
    and exhausted-retries) are deliberately kept to the same column set so the
    resulting DataFrame has no ragged rows; a missing key in one branch would
    show up as NaN in a column that is otherwise meaningful.
    """
    # Message assembly. The system prompt is optional and goes first when set.
    messages = []
    if config.system_prompt:
        messages.append({"role": "system", "content": config.system_prompt})
    messages.append({"role": "user", "content": _user_message(input_text, config)})

    # Timed inside send() so latency_ms is the successful attempt only,
    # excluding backoff sleep and failed attempts.
    #
    # The dict is a workaround for closure scoping: `send` needs to write a
    # value that the enclosing function can read afterwards, and mutating a dict
    # does that without a `nonlocal` declaration. Timing around
    # `call_with_retry` instead would fold retry sleep into the number and
    # quietly corrupt every latency conclusion drawn from the run.
    timing = {}

    def send():
        start = time.perf_counter()  # monotonic; unaffected by clock changes
        response = client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
        )
        # Only reached when the call returned without raising, so a failed
        # attempt never overwrites the timing of a later successful one.
        timing["latency_ms"] = (time.perf_counter() - start) * 1000
        return response

    response, last_error = call_with_retry(
        send,
        label=prompt_id,
        max_retries=config.max_retries,
        base_delay=config.base_delay,
        max_delay=config.max_delay,
        jitter=config.jitter,
    )

    # ---- Success path -------------------------------------------------------
    if response is not None:
        latency_ms = timing["latency_ms"]
        choice = response.choices[0]  # n=1 is assumed; see ROADMAP variance item
        message = choice.message
        output = message_text(message)
        # "length" means max_tokens cut the answer off. Without this, a
        # truncated answer scores exactly like a wrong one.
        finish_reason = getattr(choice, "finish_reason", None)
        row = {
            "prompt_id": prompt_id,
            # The *original* input, not the CoT-augmented one actually sent.
            "input": input_text,
            "output": output,
            "reasoning": message_reasoning(message),
            "finish_reason": finish_reason,
            # Stored as int rather than bool so it sums and means directly in
            # pandas, which is what gives truncation_rate for free.
            "truncated": int(finish_reason == "length"),
            "latency_ms": round(latency_ms, 2),
            # UTC and ISO-8601, so rows from different runs sort correctly and
            # survive the CSV round-trip as text.
            "timestamp": datetime.now(timezone.utc).isoformat(),
            # response.model, not config.model: OpenRouter may route to a
            # specific versioned snapshot, and that is what actually answered.
            "model": response.model,
            "error": None,
            "error_type": None,
        }
        # Merged rather than inlined so the None-preserving logic in
        # token_usage lives in one place, shared with nothing else to go wrong.
        row.update(token_usage(response))
        # Only CoT runs get these two columns. scoring.prediction_column keys
        # off the presence of "answer" to decide what to score, so adding them
        # unconditionally would make every run look like a CoT run.
        if config.chain_of_thought:
            answer, extracted = extract_final_answer(output)
            row["answer"] = answer
            row["answer_extracted"] = int(extracted)
        return row

    # ---- Failure path: all retries exhausted --------------------------------
    # Same keys as the success row, all nulled. `truncated` is 0 rather than
    # None because no response means definitively not truncated, and keeping it
    # numeric avoids an object-dtype column that would break .sum().
    row = {
        "prompt_id": prompt_id,
        "input": input_text,
        "output": None,
        "reasoning": None,
        "finish_reason": None,
        "truncated": 0,
        "latency_ms": None,  # None, not 0: a failed call has no latency to report
        "timestamp": datetime.now(timezone.utc).isoformat(),
        # config.model here, since there is no response to read the real id from.
        "model": config.model,
        "error": last_error,
        "error_type": classify_error(last_error),
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }
    if config.chain_of_thought:
        row["answer"] = None
        row["answer_extracted"] = 0
    return row


# =============================================================================
# Checkpointing
# =============================================================================

class Checkpoint:
    """Append-only JSONL log of finished rows, written as each call returns.

    A batch is the expensive part of a run: a crash at row 190 of 200 should
    not cost 190 paid API calls. JSONL because one line per row means a
    partially written file is still readable up to the last complete line,
    which a partially written CSV or JSON array is not.

    Also used by judge.py for the same reason, with judgment dicts instead of
    inference rows. The class does not care what the dict contains.
    """

    def __init__(self, path: Optional[str]):
        # A None path makes every method a no-op, so callers never have to
        # branch on "is checkpointing enabled".
        self.path = Path(path) if path else None
        # One lock per Checkpoint instance, shared by all worker threads that
        # hold a reference to it.
        self._lock = threading.Lock()
        if self.path:
            # Created up front so the first append cannot fail on a missing
            # directory after the API call has already been paid for.
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: dict) -> None:
        """Write one row. Locked because worker threads share the handle."""
        if not self.path:
            return
        with self._lock:
            # Reopened per write rather than held open. Slower, but it means no
            # file handle is left dangling if the process dies mid-batch, and
            # append mode makes concurrent correctness the lock's only job.
            with open(self.path, "a") as f:
                # default=str so a value json cannot serialize natively (a
                # datetime, a numpy scalar) degrades to its string form instead
                # of raising and losing the row we just paid for.
                f.write(json.dumps(row, default=str) + "\n")
                # flush() pushes Python's buffer to the OS; fsync() pushes the
                # OS buffer to disk. Both are needed, since the failure being
                # defended against is the process dying with data still buffered.
                f.flush()
                os.fsync(f.fileno())

    def load(self) -> list[dict]:
        """Read finished rows. A truncated final line is dropped."""
        if not self.path or not self.path.exists():
            return []
        rows = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # tolerate a stray blank line
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    break  # interrupted mid-write; everything before is good
        return rows


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
        Rows are in dataset order regardless of which call finished first.

    Execution order inside this function:
        1. build the client and the checkpoint
        2. work out what is already done (resume) and what is pending
        3. run the pending calls, sequentially or across a thread pool
        4. reassemble everything in dataset order
        5. shape the DataFrame, print the summary, write the CSV
    """
    # Checked first because it is free and because the failure it prevents is
    # invisible: duplicate ids would produce a full-size frame with duplicated
    # results rather than an error.
    ensure_unique_prompt_ids(p["prompt_id"] for p in prompts)

    # One client for the whole batch, shared across threads. The OpenAI SDK
    # client is thread-safe and holds a connection pool, so reusing it is both
    # correct and faster than constructing one per call.
    client = make_client()
    checkpoint = Checkpoint(config.checkpoint_path)

    total = len(prompts)

    # Resume: reuse rows already on disk, only call for what is missing.
    # Failed rows are retried, since a row with an error cost a call but has
    # no usable output. Keys are str() so an int prompt_id in the source JSON
    # still matches the same id written as a string in the checkpoint.
    done: dict[str, dict] = {}
    if config.resume:
        for row in checkpoint.load():
            if row.get("error") is None:
                done[str(row.get("prompt_id"))] = row

    pending = [p for p in prompts if str(p["prompt_id"]) not in done]
    # Never spin up more threads than there is work: a 3-row smoke test should
    # not create 8 threads. The outer max(1, ...) guards the empty-pending case,
    # where min() would yield 0 and ThreadPoolExecutor would reject it.
    workers = max(1, min(config.max_workers, len(pending))) if pending else 1

    if verbose:
        if done:
            print(f"Resuming: {len(done)} rows already in {config.checkpoint_path}")
        print(f"Running {len(pending)}/{total} prompts with model={config.model} "
              f"({'sequential' if workers == 1 else f'{workers} workers'})")
        print("-" * 50)

    # Progress counter. A dict rather than a plain int because it is mutated
    # from inside the nested `run_one` closure. The lock keeps the increment and
    # the read atomic, so two threads finishing at once cannot both print the
    # same index.
    counter = {"n": 0}
    counter_lock = threading.Lock()

    def run_one(prompt: dict) -> dict:
        """One unit of work, as executed by a worker thread."""
        result = run_single(client, prompt["prompt_id"], prompt["input"], config)
        # Carry forward any extra keys from the input (e.g., 'expected', 'category')
        # so the gold answer travels on the same row as the output. `if key not
        # in result` protects the harness's own columns from being clobbered by
        # a dataset that happens to have a column called "model" or "output".
        for key, value in prompt.items():
            if key not in result:
                result[key] = value
        # Persist before returning, so an interrupt after this point still
        # leaves the row on disk. Deliberately after the extra-key merge, so the
        # checkpointed row is the complete one and a resumed run does not lose
        # the gold answer.
        checkpoint.append(result)
        if verbose:
            with counter_lock:
                counter["n"] += 1
                n = counter["n"]  # copy inside the lock; print outside it
            # A terse ok/FAILED rather than a response preview: 8 threads
            # interleaving multi-line output is unreadable, and the numbering is
            # completion order, not dataset order, anyway.
            status = "ok" if result["error"] is None else "FAILED"
            print(f"[{n}/{len(pending)}] {prompt['prompt_id']}: {status}")
        return result

    # Results are collected into a dict keyed by prompt_id rather than appended
    # to a list, because `as_completed` yields in completion order. Appending
    # would interleave rows by whichever call returned first, and since the gold
    # answer rides along on each row, that would not crash — it would silently
    # pair each output with another prompt's expected answer.
    fresh: dict[str, dict] = {}
    if workers == 1:
        # Plain loop with no pool at all. This is the path to use when debugging
        # a provider error, since a traceback is not interleaved with 7 others.
        for prompt in pending:
            fresh[str(prompt["prompt_id"])] = run_one(prompt)
    else:
        # The dict maps future -> key, which is how the result gets filed under
        # the right prompt when it completes out of order.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(run_one, p): str(p["prompt_id"])
                for p in pending
            }
            for future in as_completed(futures):
                # .result() re-raises anything the worker raised. That is
                # intentional: run_single already converts API failures into
                # rows, so anything still raising here is a real bug.
                fresh[futures[future]] = future.result()

    # Rebuild in dataset order from resumed and fresh rows, so completion
    # order never reorders the frame. `done.get(...) or fresh[...]` reads as
    # "prefer the resumed row, else the one just computed"; every prompt is in
    # exactly one of the two, so the fresh lookup cannot KeyError.
    results = [
        done.get(str(p["prompt_id"])) or fresh[str(p["prompt_id"])]
        for p in prompts
    ]

    df = pd.DataFrame(results)

    # Reorder columns: standard cols first, then extras.
    # Purely cosmetic for the CSV, but it is what makes the raw output readable
    # at a glance during a timed run: identity, then answer, then diagnostics,
    # then the dataset's own columns (expected, category) at the end.
    standard_cols = [
        "prompt_id", "input", "output", "answer", "answer_extracted", "reasoning",
        "finish_reason", "truncated", "prompt_tokens", "completion_tokens",
        "total_tokens", "latency_ms", "timestamp", "model", "error", "error_type",
    ]
    # Filtered against the actual columns, since "answer"/"answer_extracted"
    # only exist on CoT runs and indexing on a missing column would raise.
    standard_cols = [c for c in standard_cols if c in df.columns]
    extra_cols = [c for c in df.columns if c not in standard_cols]
    df = df[standard_cols + extra_cols]

    if verbose:
        print("-" * 50)
        # Counted against `total`, not `len(pending)`, so a resumed run reports
        # the whole dataset rather than just this session's share of it.
        success_count = df["error"].isna().sum()
        print(f"Complete: {success_count}/{total} succeeded")
        # Guarded: an all-failed run has an all-NaN latency column, and .mean()
        # on it would print nan.
        if df["latency_ms"].notna().any():
            # Per-call latency, unaffected by concurrency. Batch wall clock is
            # NOT the sum of these once max_workers > 1 — worth saying out loud
            # before anyone adds them up.
            print(f"Latency: mean={df['latency_ms'].mean():.0f}ms, "
                  f"p50={df['latency_ms'].median():.0f}ms, "
                  f"p95={df['latency_ms'].quantile(0.95):.0f}ms")

    # Write to CSV if configured. This is the raw, unscored artifact, written
    # before any scoring runs, so a crash in scoring.py never costs the calls.
    # It is also the file --score-only reads back.
    if config.output_csv:
        df.to_csv(config.output_csv, index=False)
        if verbose:
            print(f"Saved to {config.output_csv}")

    return df


# =============================================================================
# Quick test
#
# `python -m llm_eval.harness` runs two real calls end to end. This is the
# fastest way to confirm the key, the model id, and the network path all work
# before trusting a full dataset to them.
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
