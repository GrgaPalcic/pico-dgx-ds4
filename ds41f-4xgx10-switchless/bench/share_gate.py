#!/usr/bin/env python3
"""The served exactness gate for the shared KV pool: resumed, fork, concurrent = alone (two engines), perf.

Run it against a live dsfront lane (dsfront's OpenAI API) before and after each engine change of the
shared-KV port; the JSON report it writes is what decides whether a build ships. Deterministic filler
prompts from a fixed seed, temperature 0 and thinking off, streaming so time to the first token is
observed; every comparison is on sha256 of the streamed content, and no prompt, reply or key text ever
reaches the report. The prompt sizing is calibrated against the live tokenizer first (two small probes)
and the context limit comes from /v1/models, so an oversized request is a listed failure, not an
exception. Standard library only, so it runs on a bare host.

  python3 bench/share_gate.py run --base http://127.0.0.1:18890/v1 --out report.json
  python3 bench/share_gate.py run --base ... --key-file key.txt --sizes 128,1024 --levels 1,2 --out t.json
  python3 bench/share_gate.py run --base ... --cases fork --nonce N --sizes 4096,131072,600000 --out cold.json
  python3 bench/share_gate.py run --base ... --dry-run
  python3 bench/share_gate.py compare before.json after.json

``compare`` matches case ids across two reports and lists the ones whose hashes differ. Its uses: drafting
off against the production lane, the flag off against the base build, or a fork on a warm engine against
the same case run cold. Give both runs the same ``--nonce``, or the prompts differ and so will every hash.

The cases of ``run``, every one at temperature 0 with ``max_tokens`` 64 unless stated and
``stream_options.include_usage`` on (TTFT is the time to the first content delta):

- ``resumed-<size>``: per ``--sizes`` entry, a cold request, the same request again and the same messages
  plus one appended user sentence (reported, not compared). A warm reply that differs from the cold one is
  a note, not a failure: the warm request reruns its last positions through the decode step, which matches
  the prefill path to numerics, not bits (docs/engine.md "Serving", #9).
- ``fork-<size>``: the first quarter of the largest ``--sizes`` prompt plus a distinct filler tail, so it
  shares only the first 25 % of that earlier prompt; recorded under its own case id.
- ``concurrent-<level>``: per ``--levels`` entry, that many distinct prompts sent at once while nothing of
  theirs is in the engine (their prefills overlap: the path the pool stresses), then one prompt sent that
  many times at once (all those copies must hash equal to each other: every copy runs cold). The replies
  are compared with the ``alone`` case's, run against a freshly started engine (below).
- ``alone``: not in the default list. Per ``--levels`` entry, the same prompts as ``concurrent-<level>``,
  one at a time, recorded under that case id as ``alone-<seed>`` (seed 0 is the prompt the copies share).
  Nothing is checked in the run; ``compare`` matches them with the concurrent replies.
- ``grow16-<i>``: with ``--grow16 N``, 16 sessions extended turn by turn (append-only messages), all 16 at
  once per turn, until each passes N prompt tokens; each final reply is then compared with the same final
  messages sent alone, one at a time.
- ``perf-<level>``: per ``--perf`` entry, that many streams at once with ``max_tokens`` 512; aggregate and
  per-stream decode tok/s and TTFT per stream are reported.

Every prompt's header names the case and its size (``resumed-4096: ...``) and its words follow that name, so
no case's prompt is a prefix of another case's: two of them share the nonce line and at most the tags' own
common prefix, never a line of the filler body. Each case's first request is therefore cold and reports
``cached_tokens`` of the nonce line's tokens and a few more, not thousands. The fork prompt is the exception
by design — it starts with its parent's own bytes — and the perf prompts belong to their own level and
stream, so both perf levels measure cold prefills.

``--cases`` names which of those kinds to send (default all but ``alone``); the calibration always runs,
and naming it is allowed. The fork case's prompt is built from the nonce, the sizes and the calibrated
ratio alone, so it is byte for byte the same whether or not its parent prompt was sent in the same run.
**fork = cold**, then, is two runs:

  python3 bench/share_gate.py run --base URL --nonce N --sizes ... --out warm.json    # the parent is sent
  python3 bench/share_gate.py run --base URL --nonce N --sizes ... --cases fork --out cold.json
  python3 bench/share_gate.py compare warm.json cold.json                            # the fork hashes must agree

Run the second against a freshly started engine, so nothing of the fork prompt's prefix is in it.

**concurrent = alone** is two runs too. An alone request sent after its concurrent twin in one engine would
resume the twin's stored state, a regenerate path that matches the cold one to numerics, not bits; so the
alone replies come cold from a freshly started engine:

  python3 bench/share_gate.py run --base URL --nonce N --levels ... --out run.json      # the batches
  (restart the engine)
  python3 bench/share_gate.py run --base URL --nonce N --levels ... --cases alone --out alone.json
  python3 bench/share_gate.py compare run.json alone.json    # distinct-<s> vs alone-<s>, same-<c> vs alone-0

At most ``max(max(levels), max(perf), 16)`` requests are ever in flight. The run stops at the first HTTP
error or broken stream, writes the report so far with ``pass: false`` and exits 1.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib

TOOL = "share_gate"
VERSION = 1
HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")

# 4 words plus a fixed-width line number: a filler line is a stable number of tokens for any tokenizer
WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november "
         "oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu").split()

CASE_MAX_TOKENS = 64            # every case's reply length, except perf
PERF_MAX_TOKENS = 512
PROBE_SIZES = (256, 1024)       # two probes: their slope gives tokens per filler line, the header cancels
DEFAULT_PER_LINE = 4.0          # tokens per filler line before the probes have run
CONCURRENT_TOKENS = 4096        # the prompts of the concurrent = alone case
PERF_TOKENS = 4096
GROW16_SESSIONS = 16
GROW16_CHUNKS = 8               # grow16 appends about N / 8 tokens a turn, so about 8 turns reach N
GROW16_TURNS_MAX = 32           # a safety cap on grow16's turns
FORK_SHARE = 0.25               # the fork prompt shares this much of the parent prompt
SIZE_TOLERANCE = 0.20           # a measured prompt within +-20 % of the asked size is the right size
CASE_KINDS = ("resumed", "fork", "concurrent", "alone", "grow16", "perf")
CASE_DEFAULT = "resumed,fork,concurrent,grow16,perf"     # alone runs on its own, against a fresh engine
CASE_NAMES = ("calibration", *CASE_KINDS)                # calibration always runs; naming it is allowed
PER_REQUEST_KINDS = ("calibration", "resumed", "fork")   # the summary prints a row a request for these
APPENDED = "One more sentence, please: count the lines again and reply with the number only."
MIN_PER_LINE = 1.0
MAX_PER_LINE = 64.0


class Abort(RuntimeError):
    """A request failed (an HTTP error or a broken stream): the run stops and the report stands as it is."""


class Reply:
    """One streamed reply: its hash and timings for the report, its text kept in memory only."""

    def __init__(self, text: str, sha256: str, ttft_s: float | None, total_s: float, usage: dict,
                 started: float, first: float | None, ended: float):
        self.text = text
        self.sha256 = sha256
        self.ttft_s = ttft_s
        self.total_s = total_s
        self.usage = usage
        self.started = started
        self.first = first
        self.ended = ended

    def record(self, case: str, role: str, size: int) -> dict:
        """The report's record of this request. The reply's text is never part of it."""
        return {"case": case, "role": role, "size": size, "sha256": self.sha256, "ttft_s": self.ttft_s,
                "total_s": self.total_s, "usage": self.usage}

    def empty(self) -> bool:
        """True when the stream carried no content delta at all."""
        return self.first is None


# ------------------------------------------------------------------ prompts

def words_for(tag: str, line: int) -> str:
    """The four filler words of one body line, fixed by the tag and the line number."""
    seed = zlib.crc32(tag.encode("utf-8"))
    return " ".join(WORDS[(seed + line * 7 + j * 3) % len(WORDS)] for j in range(4))


def filler(tag: str, tokens: int, per_line: float = DEFAULT_PER_LINE, nonce: str = "") -> str:
    """A deterministic prompt of about ``tokens`` tokens: the nonce line, a header naming ``tag``, word lines.

    The nonce leads every prompt so a run never resumes what an earlier run kept, and the header names the
    case and its size, so two prompts of different cases share the nonce line and, at most, the tags' own
    common prefix: no case's prompt is a prefix of another case's, so a case's first request is cold and its
    ``cached_tokens`` are the nonce line's and a few more. The words follow the tag too, so even the bodies
    differ. The line width is fixed, so the tokens per line do not grow with the count and one measured
    ratio sizes every prompt.
    """
    lines = max(1, round(tokens / per_line))
    body = "\n".join(f"{words_for(tag, i)} line {i:07d}" for i in range(lines))
    return f"{nonce}\n{tag}: count the lines below and reply with the number only.\n{body}"


def filler_lines(prompt: str) -> int:
    """The filler body's line count: what a calibration probe's prompt_tokens are proportional to."""
    return prompt.count("\n") - 1


def fork_prompt(parent: str, tag: str, per_line: float, tokens: int) -> str:
    """The fork case's prompt: the parent's first quarter, then filler of another tag.

    The head is the parent's own bytes, so the engine can find that prefix; the tail's tag gives it other
    words and its lines are numbered from zero, so the two prompts share exactly the quarter, no more.
    """
    lines = parent.split("\n")
    body = lines[2:]
    keep = max(1, round(len(body) * FORK_SHARE))
    head = "\n".join(lines[:2] + body[:keep])
    tail_lines = max(1, round(tokens / per_line * (1.0 - FORK_SHARE)))
    tail = "\n".join(f"{words_for(tag, i)} line {i:07d}" for i in range(tail_lines))
    return head + "\n" + tail


def user(text: str) -> dict:
    return {"role": "user", "content": text}


def assistant(text: str) -> dict:
    return {"role": "assistant", "content": text}


# ------------------------------------------------------------------ HTTP

def auth_headers(key: str | None, json_body: bool = False) -> dict:
    """The headers of every request. Without a key no Authorization header is sent at all."""
    headers = {"Content-Type": "application/json"} if json_body else {}
    if key:
        headers["Authorization"] = "Bearer " + key
    return headers


def usage_numbers(usage: dict) -> dict:
    """The numeric usage fields only: no text a server chose to put in usage reaches the report."""
    out: dict = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if isinstance(usage.get(name), int):
            out[name] = usage[name]
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict) and isinstance(details.get("cached_tokens"), int):
        out["cached_tokens"] = details["cached_tokens"]
    return out


def chat(base: str, model: str, messages: list[dict], *, key: str | None, timeout: float,
         max_tokens: int) -> Reply:
    """One streamed chat completion at temperature 0 with thinking off; the reply is hashed as it arrives."""
    body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": True,
            "chat_template_kwargs": {"thinking": False}}
    request = urllib.request.Request(base + "/chat/completions", method="POST",
                                     data=json.dumps(body).encode(),
                                     headers=auth_headers(key, json_body=True))
    started = time.perf_counter()
    pieces: list[str] = []
    digest = hashlib.sha256()
    usage: dict = {}
    first = None
    done = False
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    done = True
                    break
                try:
                    chunk = json.loads(payload)
                except ValueError as exc:
                    raise Abort(f"unparseable stream chunk: {exc}") from exc
                if chunk.get("error"):
                    raise Abort("the stream carried an error chunk")
                if isinstance(chunk.get("usage"), dict):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        if first is None:
                            first = time.perf_counter()
                        pieces.append(piece)
                        digest.update(piece.encode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        exc.close()
        raise Abort(f"HTTP {exc.code} {exc.reason}") from exc
    except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError) as exc:
        raise Abort(f"the request failed: {type(exc).__name__}") from exc
    ended = time.perf_counter()
    if not done:
        raise Abort("the stream ended without [DONE]")
    return Reply("".join(pieces), digest.hexdigest(), None if first is None else round(first - started, 4),
                 round(ended - started, 4), usage_numbers(usage), started, first, ended)


def context_limit(base: str, key: str | None, timeout: float) -> tuple[int | None, str | None]:
    """The server's context limit from /v1/models' ``max_model_len`` (dsfront's /health is plain text)."""
    request = urllib.request.Request(base + "/models", headers=auth_headers(key))
    try:
        with urllib.request.urlopen(request, timeout=min(timeout, 60.0)) as response:
            data = json.loads(response.read().decode("utf-8", "replace"))
    except (ValueError, urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    for model in data.get("data") or []:
        length = model.get("max_model_len") if isinstance(model, dict) else None
        if isinstance(length, (int, float)) and length > 0:
            return int(length), None
    return None, "no max_model_len in /v1/models"


# ------------------------------------------------------------------ one run

class Run:
    """One gate run: where to send, how prompts are sized, and the report being built."""

    def __init__(self, base: str, model: str, key: str | None, timeout: float, nonce: str, cap: int):
        self.base = base
        self.model = model
        self.key = key
        self.timeout = timeout
        self.nonce = nonce
        self.cap = cap
        self.per_line = DEFAULT_PER_LINE
        self.limit: int | None = None    # the server's context limit, from /v1/models
        self.requests: list[dict] = []
        self.failures: list[dict] = []
        self.notes: list[dict] = []     # what is reported but does not fail the run
        self.perf_cases: list[dict] = []
        self.case = "run"           # the case in progress: an aborted request is reported under it

    def fail(self, why: str, case: str | None = None) -> None:
        self.failures.append({"case": case or self.case, "why": why})

    def send(self, case: str, role: str, size: int, messages: list[dict],
             max_tokens: int = CASE_MAX_TOKENS) -> Reply:
        """One request, recorded under the case in the report."""
        reply = chat(self.base, self.model, messages, key=self.key, timeout=self.timeout,
                     max_tokens=max_tokens)
        self.requests.append(reply.record(case, role, size))
        return reply

    def batch(self, items: list[tuple[str, list[dict]]],
              max_tokens: int = CASE_MAX_TOKENS) -> list[Reply]:
        """``items`` = [(role, messages)]: all sent at once behind a barrier, replies in that order.

        The caller records them, so a case that learns from the batch (grow16's last turn) can name the
        roles afterwards.
        """
        gate = threading.Barrier(len(items))

        def one(messages: list[dict]) -> Reply:
            gate.wait(timeout=120)      # every request of the batch leaves together
            return chat(self.base, self.model, messages, key=self.key, timeout=self.timeout,
                        max_tokens=max_tokens)

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(items), self.cap)) as pool:
            futures = [pool.submit(one, messages) for _, messages in items]
            return [future.result() for future in futures]

    # ---- calibration

    def calibrate(self) -> None:
        """Two small probes: the slope of prompt_tokens against filler lines is tokens per filler line."""
        self.case = "calibration"
        probes = []
        for tokens in PROBE_SIZES:
            prompt = filler(f"calibration-{tokens}", tokens, DEFAULT_PER_LINE, self.nonce)
            reply = self.send("calibration", f"probe-{tokens}", tokens, [user(prompt)])
            probes.append((filler_lines(prompt), reply.usage.get("prompt_tokens")))
        (lines_a, tokens_a), (lines_b, tokens_b) = probes
        if not tokens_a or not tokens_b:
            raise Abort("the calibration probes carried no usage.prompt_tokens")
        slope = (tokens_b - tokens_a) / (lines_b - lines_a) if lines_b != lines_a and tokens_b > tokens_a \
            else tokens_a / lines_a
        if not MIN_PER_LINE <= slope <= MAX_PER_LINE:
            raise Abort(f"the calibration slope {slope:.2f} tokens a line is not believable")
        self.per_line = slope

    # ---- the cases

    def resumed(self, size: int) -> str:
        """cold, warm (must equal cold) and appended (one sentence more, reported only) runs of one prompt."""
        case = f"resumed-{size}"
        self.case = case
        if self.limit is not None and size + CASE_MAX_TOKENS > self.limit:
            self.fail(f"size_over_context (context limit {self.limit})")
            return filler(f"resumed-{size}", size, self.per_line, self.nonce)
        prompt = filler(f"resumed-{size}", size, self.per_line, self.nonce)
        cold = self.send(case, "cold", size, [user(prompt)])
        warm = self.send(case, "warm", size, [user(prompt)])
        self.send(case, "appended", size, [user(prompt), user(APPENDED)])
        if cold.empty() or warm.empty():
            self.fail("a reply carried no content")
        elif cold.sha256 != warm.sha256:
            self.notes.append({"case": case, "warm_equals_cold": False})
        measured = cold.usage.get("prompt_tokens")
        if measured and abs(measured - size) > SIZE_TOLERANCE * size:
            self.fail(f"the prompt measured {measured} tokens, not the asked {size}")
        return prompt

    def fork(self, size: int) -> None:
        """A prompt sharing only the first quarter of the largest size prompt, under its own case id.

        The parent prompt is rebuilt here from the nonce, the size and the calibrated ratio, never taken
        from the resumed case, so the fork prompt is byte for byte the same whether or not that parent was
        sent in this run: ``--cases fork`` is the cold half of the fork = cold comparison.
        """
        case = f"fork-{size}"
        self.case = case
        if self.limit is not None and size + CASE_MAX_TOKENS > self.limit:
            self.fail(f"size_over_context (context limit {self.limit})")
            return
        parent = filler(f"resumed-{size}", size, self.per_line, self.nonce)
        reply = self.send(case, "fork", size, [user(fork_prompt(parent, f"fork-{size}", self.per_line, size))])
        if reply.empty():
            self.fail("the reply carried no content")

    def concurrent_prompts(self, level: int) -> dict[int, str]:
        """The prompts of ``concurrent-<level>``: seed 0 is the one the copies share, 1..level the distinct."""
        return {seed: filler(f"concurrent-{level}-{seed}", CONCURRENT_TOKENS, self.per_line, self.nonce)
                for seed in range(0, level + 1)}

    def concurrent(self, level: int) -> None:
        """``level`` distinct prompts at once (all cold, so their prefills overlap), then one prompt ``level``
        times at once; ``compare`` against an ``alone`` run on a fresh engine checks the replies."""
        case = f"concurrent-{level}"
        self.case = case
        prompts = self.concurrent_prompts(level)
        distinct = self.batch([(f"distinct-{seed}", [user(prompts[seed])]) for seed in range(1, level + 1)])
        for seed, reply in zip(range(1, level + 1), distinct):
            self.requests.append(reply.record(case, f"distinct-{seed}", CONCURRENT_TOKENS))
        same = self.batch([(f"same-{copy}", [user(prompts[0])]) for copy in range(1, level + 1)])
        for copy, reply in zip(range(1, level + 1), same):
            self.requests.append(reply.record(case, f"same-{copy}", CONCURRENT_TOKENS))
        if any(reply.empty() for reply in [*distinct, *same]):
            self.fail("a reply carried no content")
        if len({reply.sha256 for reply in same}) != 1:
            self.fail(f"the {level} copies of one prompt do not all match each other")

    def alone(self, level: int) -> None:
        """``concurrent-<level>``'s prompts one at a time, for ``compare``; nothing is checked here."""
        case = f"concurrent-{level}"
        self.case = case
        for seed, prompt in self.concurrent_prompts(level).items():
            self.send(case, f"alone-{seed}", CONCURRENT_TOKENS, [user(prompt)])

    def grow16(self, target: int) -> None:
        """16 sessions grown turn by turn, all 16 at once, then each final prompt again alone, one at a time."""
        chunk = max(1, round(target / GROW16_CHUNKS / self.per_line))
        sessions = [{"name": seed,
                     "messages": [user(filler(f"grow16-{seed}-1", chunk * self.per_line, self.per_line,
                                              self.nonce))],
                     "case": f"grow16-{seed}", "tokens": 0, "final": None}
                    for seed in range(1, GROW16_SESSIONS + 1)]
        turns = 0
        while turns < GROW16_TURNS_MAX:
            turns += 1
            self.case = f"grow16-turn-{turns}"
            replies = self.batch([(session["case"], session["messages"]) for session in sessions])
            grown = True
            for session, reply in zip(sessions, replies):
                session["tokens"] = reply.usage.get("prompt_tokens") or session["tokens"]
                session["final"] = reply
                if session["tokens"] < target:
                    grown = False
            roles = [f"turn-{turns}"] * len(sessions) if not grown else ["final"] * len(sessions)
            for session, reply, role in zip(sessions, replies, roles):
                self.requests.append(reply.record(session["case"], role, target))
            if grown:
                break
            for session, reply in zip(sessions, replies):
                session["messages"] = session["messages"] + [
                    assistant(reply.text),
                    user(filler(f"grow16-{session['name']}-{turns + 1}", chunk * self.per_line,
                                self.per_line, self.nonce)),
                ]
        for session in sessions:
            self.case = session["case"]
            alone = self.send(session["case"], "final-alone", target, session["messages"])
            final = session["final"]
            if alone.empty() or final is None or alone.sha256 != final.sha256:
                self.fail("the final reply differs from the same final messages sent alone")

    def perf(self, level: int) -> None:
        """``level`` streams at once with max_tokens 512: the decode speed and TTFT of each are reported."""
        case = f"perf-{level}"
        self.case = case
        items = [(f"stream-{stream}",
                  [user(filler(f"perf-{level}-{stream}", PERF_TOKENS, self.per_line, self.nonce))])
                 for stream in range(1, level + 1)]
        replies = self.batch(items, max_tokens=PERF_MAX_TOKENS)
        for (role, _), reply in zip(items, replies):
            self.requests.append(reply.record(case, role, PERF_TOKENS))
            if reply.empty():
                self.fail(f"{role} carried no content")
        completions = [reply.usage.get("completion_tokens") or 0 for reply in replies]
        firsts = [reply.first for reply in replies if reply.first is not None]
        span = max(reply.ended for reply in replies) - min(firsts) if firsts else 0.0
        self.perf_cases.append({
            "case": case, "streams": level,
            "aggregate_decode_tok_s": round(sum(completions) / span, 2) if span > 0 else None,
            "ttft_s": [reply.ttft_s for reply in replies],
            "decode_tok_s": [round((reply.usage.get("completion_tokens") or 0) / (reply.total_s - reply.ttft_s), 2)
                             if reply.ttft_s is not None and reply.total_s > reply.ttft_s else None
                             for reply in replies]})


def run(base: str, model: str, *, sizes: list[int], levels: list[int], perf: list[int], grow16: int | None,
        cases: list[str], nonce: str, key: str | None, timeout: float, argv: list[str]) -> dict:
    """The whole gate: calibrate, then the case kinds ``cases`` names (the fork one always needs sizes)."""
    cap = max([*levels, *perf, GROW16_SESSIONS])
    gate = Run(base, model, key, timeout, nonce, cap)
    report: dict = {"tool": TOOL, "version": VERSION, "argv": list(argv), "base": base, "model": model,
                    "nonce": nonce, "cases": cases, "sizes": sizes, "levels": levels, "perf": perf,
                    "grow16": grow16, "context_limit": None, "context_limit_error": None,
                    "calibration": None, "requests": gate.requests, "perf_cases": gate.perf_cases,
                    "failures": gate.failures, "notes": gate.notes, "pass": False}
    try:
        limit, limit_error = context_limit(base, key, timeout)
        gate.limit = report["context_limit"] = limit
        report["context_limit_error"] = limit_error
        gate.calibrate()
        report["calibration"] = {"tokens_per_line": round(gate.per_line, 4), "probes": list(PROBE_SIZES)}
        if "resumed" in cases:
            for size in sizes:
                gate.resumed(size)
        if "fork" in cases:
            gate.fork(max(sizes))
        if "concurrent" in cases:
            for level in levels:
                gate.concurrent(level)
        if "alone" in cases:
            for level in levels:
                gate.alone(level)
        if "grow16" in cases and grow16:
            gate.grow16(grow16)
        if "perf" in cases:
            for level in perf:
                gate.perf(level)
    except Abort as exc:
        gate.fail(str(exc))
    report["pass"] = not report["failures"]
    return report


def plan(sizes: list[int], levels: list[int], perf: list[int], grow16: int | None,
         cases: list[str]) -> dict:
    """``--dry-run``: the cases and their request counts, with no HTTP."""
    planned = [{"id": "calibration", "kind": "calibration", "size": PROBE_SIZES[0],
                "requests": len(PROBE_SIZES)}]
    if "resumed" in cases:
        for size in sizes:
            planned.append({"id": f"resumed-{size}", "kind": "resumed", "size": size, "requests": 3})
    if "fork" in cases:
        planned.append({"id": f"fork-{max(sizes)}", "kind": "fork", "size": max(sizes), "requests": 1})
    if "concurrent" in cases:
        for level in levels:
            planned.append({"id": f"concurrent-{level}", "kind": "concurrent", "size": CONCURRENT_TOKENS,
                            "requests": 2 * level})
    if "alone" in cases:
        for level in levels:
            planned.append({"id": f"concurrent-{level}", "kind": "alone", "size": CONCURRENT_TOKENS,
                            "requests": level + 1})
    if "grow16" in cases and grow16:
        # the estimate: a turn adds about grow16 / GROW16_CHUNKS prompt tokens and one 64-token reply
        per_turn = grow16 / GROW16_CHUNKS + CASE_MAX_TOKENS
        turns = min(GROW16_TURNS_MAX, max(1, -(-grow16 // int(per_turn))))
        for seed in range(1, GROW16_SESSIONS + 1):
            planned.append({"id": f"grow16-{seed}", "kind": "grow16", "size": grow16, "requests": turns + 1})
    if "perf" in cases:
        for level in perf:
            planned.append({"id": f"perf-{level}", "kind": "perf", "size": PERF_TOKENS, "requests": level})
    return {"tool": TOOL, "version": VERSION, "kind": "plan", "cases": cases, "sizes": sizes,
            "levels": levels, "perf": perf, "grow16": grow16, "planned": planned,
            "requests_total": sum(case["requests"] for case in planned)}


# ------------------------------------------------------------------ compare and output

def load_report(path: str) -> dict:
    """A share_gate report: its requests each carry a case id, a role and a sha256."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
    if not isinstance(report, dict) or not isinstance(report.get("requests"), list):
        raise ValueError(f"{path}: not a share_gate report")
    for record in report["requests"]:
        if not isinstance(record, dict) or not isinstance(record.get("case"), str) \
                or not HEX64.fullmatch(record.get("sha256") or ""):
            raise ValueError(f"{path}: a request record has no case id and sha256")
    return report


def by_case(report: dict) -> dict:
    """The report's hashes per case id and role: {case: {role: sha256}}."""
    out: dict = {}
    for record in report["requests"]:
        out.setdefault(record["case"], {})[str(record.get("role"))] = record["sha256"]
    return out


def counterpart(role: str, right: dict) -> str | None:
    """The right report's role a left role is compared with: itself, or the alone reply of its prompt.

    A concurrent run's ``distinct-<s>`` meets ``alone-<s>`` and its ``same-<c>`` meets ``alone-0`` when the
    right report is an ``alone`` run of the same nonce and levels.
    """
    if role in right:
        return role
    kind, _, index = role.partition("-")
    alone = {"distinct": f"alone-{index}", "same": "alone-0"}.get(kind)
    return alone if alone in right else None


def compare_main(first: str, second: str) -> int:
    """Case ids whose hashes differ between two reports: 0 all equal, 1 any difference, 2 unusable input."""
    try:
        reports = [by_case(load_report(first)), by_case(load_report(second))]
    except ValueError as exc:
        print(f"compare: unusable input: {exc}", file=sys.stderr)
        return 2
    left, right = reports
    shared = [case for case in left if case in right]
    if not shared:
        print(f"compare: no case id in common between {first} and {second}")
        return 0
    changed: dict = {}
    for case in shared:
        pairs = [(role, counterpart(role, right[case])) for role in sorted(left[case])]
        roles = [role for role, other in pairs if other is not None and left[case][role] != right[case][other]]
        if roles:
            changed[case] = roles
    print(f"compare: {len(shared)} case ids in common, {len(changed)} differing")
    for case, roles in changed.items():
        print(f"  {case}: " + ", ".join(roles))
    for case in sorted(set(left) - set(right)):
        print(f"  only in {first}: {case}")
    for case in sorted(set(right) - set(left)):
        print(f"  only in {second}: {case}")
    return 1 if changed else 0


def cases_in_order(requests: list[dict]) -> dict:
    """The report's requests grouped by case id, in the order the cases ran."""
    out: dict = {}
    for record in requests:
        out.setdefault(record["case"], []).append(record)
    return out


def fmt_seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def fmt_timings(values: list[float | None]) -> str:
    """``min/median/max`` of the timings that are there; one number when there is only one."""
    got = sorted(value for value in values if value is not None)
    if not got:
        return "-"
    if len(got) == 1:
        return fmt_seconds(got[0])
    return f"{got[0]:.2f}/{statistics.median(got):.2f}/{got[-1]:.2f}"


def largest(values: list) -> str:
    got = [value for value in values if isinstance(value, int)]
    return str(max(got)) if got else "-"


def summary(report: dict) -> str:
    """The one-screen table: a row a request for the plain cases, a row a case for the batched ones.

    ``calibration``, ``resumed`` and ``fork`` print each request's own prompt tokens, cached tokens, TTFT
    and total, so no row mixes two requests. ``concurrent``, ``grow16`` and ``perf`` print one row a case
    with its requests' min/median/max TTFT, the largest prompt and the most-cached reply among them; a perf
    row also carries the aggregate and the median per-stream decode rate.
    """
    failed = {failure["case"] for failure in report["failures"]}
    perf_by_case = {perf["case"]: perf for perf in report["perf_cases"]}
    rows = [f"{'case':<18} {'role':<11} {'size':>7} {'tokens':>8} {'cached':>7} "
            f"{'ttft s':>17} {'total s':>8}  verdict"]
    for case, records in cases_in_order(report["requests"]).items():
        verdict = "FAIL" if case in failed else "ok"
        if case.split("-", 1)[0] in PER_REQUEST_KINDS:
            for record in records:
                usage = record.get("usage") or {}
                rows.append(f"{case:<18} {record['role']:<11} {record.get('size') or 0:>7} "
                            f"{str(usage.get('prompt_tokens', '-')):>8} "
                            f"{str(usage.get('cached_tokens', '-')):>7} "
                            f"{fmt_seconds(record.get('ttft_s')):>17} "
                            f"{record.get('total_s') or 0:>8.2f}  {verdict}")
            continue
        usages = [record["usage"] for record in records if isinstance(record.get("usage"), dict)]
        totals = [record.get("total_s") for record in records]
        perf = perf_by_case.get(case)
        note = verdict
        if perf is not None:
            rates = [rate for rate in perf["decode_tok_s"] if rate is not None]
            note = f"{perf['aggregate_decode_tok_s']} tok/s, median " \
                   f"{round(statistics.median(rates), 1) if rates else '-'} tok/s/stream" \
                   f"{' FAIL' if case in failed else ''}"
        rows.append(f"{case:<18} {f'{len(records)} reqs':<11} {records[0].get('size') or 0:>7} "
                    f"{largest([usage.get('prompt_tokens') for usage in usages]):>8} "
                    f"{largest([usage.get('cached_tokens') for usage in usages]):>7} "
                    f"{fmt_timings([record.get('ttft_s') for record in records]):>17} "
                    f"{round(statistics.median(totals), 2) if totals else 0:>8.2f}  {note}")
    rows.append(f"pass: {report['pass']}")
    for failure in report["failures"]:
        rows.append(f"  {failure['case']}: {failure['why']}")
    for note in report.get("notes") or []:
        rows.append(f"  note {note['case']}: warm differs from cold (a regenerate path; docs/engine.md \"Serving\")")
    return "\n".join(rows)


def redact(text: str, key: str | None) -> str:
    """Nothing the tool prints or writes may carry the key, whatever else happened."""
    return text.replace(key, "<key>") if key else text


def write_report(path: str, report: dict, key: str | None) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(redact(json.dumps(report, indent=2) + "\n", key))


# ------------------------------------------------------------------ CLI

def positive_ints(value: str) -> list[int]:
    """A comma list of positive integers, or a usage error."""
    try:
        out = [int(part) for part in value.split(",") if part.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError("must be a comma list of integers")
    if not out or min(out) < 1:
        raise argparse.ArgumentTypeError("must be a non-empty list of positive integers")
    return out


def case_kinds(value: str) -> list[str]:
    """A comma list of case names, or a usage error. The calibration always runs whatever the list says."""
    out = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [part for part in out if part not in CASE_NAMES]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown {', '.join(unknown)}; known: {', '.join(CASE_NAMES)}")
    if not out:
        raise argparse.ArgumentTypeError("must name at least one case")
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="share_gate.py", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_parser = sub.add_parser("run", help="run the gate against a served lane and write a report")
    run_parser.add_argument("--base", help="the /v1 base URL, for example http://127.0.0.1:18890/v1")
    run_parser.add_argument("--key-file", metavar="F", help="a file holding the Bearer API key")
    run_parser.add_argument("--model", default="deepseek-v4.1-flash")
    run_parser.add_argument("--out", metavar="R.json", help="write the JSON report here")
    run_parser.add_argument("--nonce", help="the fixed first line of every prompt (default: random)")
    run_parser.add_argument("--cases", default=CASE_DEFAULT, type=case_kinds,
                            help=f"which cases to send, a comma list ({CASE_DEFAULT})")
    run_parser.add_argument("--sizes", default="4096,131072,600000", type=positive_ints,
                            help="prompt sizes in tokens, a comma list")
    run_parser.add_argument("--levels", default="1,4,8,16", type=positive_ints,
                            help="concurrent requests per cell, a comma list")
    run_parser.add_argument("--perf", default="1,16", type=positive_ints,
                            help="perf streams per cell, a comma list")
    run_parser.add_argument("--grow16", type=int, metavar="N",
                            help="grow 16 sessions turn by turn until each passes N prompt tokens")
    run_parser.add_argument("--timeout", type=float, default=3600.0, help="seconds per request")
    run_parser.add_argument("--dry-run", action="store_true", help="print the planned cases, send nothing")
    compare_parser = sub.add_parser("compare", help="compare two reports by case id and hash")
    compare_parser.add_argument("first")
    compare_parser.add_argument("second")
    return parser


def run_main(args: argparse.Namespace, argv: list[str]) -> int:
    if args.grow16 is not None and args.grow16 < 0:
        print("share_gate.py: --grow16 must not be negative", file=sys.stderr)
        return 2
    if args.dry_run:
        print(json.dumps(plan(args.sizes, args.levels, args.perf, args.grow16, args.cases), indent=2))
        return 0
    if not args.base or not args.out:
        print("share_gate.py: run needs --base and --out (or --dry-run)", file=sys.stderr)
        return 2
    key = None
    if args.key_file:
        try:
            with open(args.key_file, "r", encoding="utf-8") as handle:
                key = handle.read().strip()
        except OSError as exc:
            print(f"share_gate.py: cannot read the key file: {exc}", file=sys.stderr)
            return 2
        if not key:
            print("share_gate.py: the key file is empty", file=sys.stderr)
            return 2
    report = run(args.base.rstrip("/"), args.model, sizes=args.sizes, levels=args.levels, perf=args.perf,
                 grow16=args.grow16, cases=args.cases, nonce=args.nonce or os.urandom(8).hex(), key=key,
                 timeout=args.timeout, argv=argv)
    write_report(args.out, report, key)
    print(redact(summary(report), key))
    return 0 if report["pass"] else 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    if args.cmd == "compare":
        return compare_main(args.first, args.second)
    return run_main(args, argv)


if __name__ == "__main__":
    sys.exit(main())
