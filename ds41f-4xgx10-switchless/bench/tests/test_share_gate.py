"""Tests for bench/share_gate.py against a fake streaming OpenAI server in a thread.

Run from ``ds41f-4xgx10-switchless``: ``python3 -m unittest bench/tests/test_share_gate.py`` (or
``python3 -m pytest bench/tests/test_share_gate.py -q``).

The fake server answers ``/v1/models``, ``/health`` and a streamed ``/v1/chat/completions``; its reply is a
hash of the messages, so equal messages give equal replies, and it can be told to answer a warm prompt
differently or to break under concurrency. Its usage.prompt_tokens is proportional to the prompt's length,
which is what the gate's calibration measures.
"""

import hashlib
import io
import json
import os
import re
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bench import share_gate  # noqa: E402  (the tool under test)

CONTEXT = 1 << 20       # the fake server's context limit: every size these tests ask for fits
CHARS_PER_TOKEN = 4     # the fake's usage.prompt_tokens: a quarter of the rendered prompt's length
SIZE = 512              # the size these tests ask the gate to send


class FakeServer:
    """A streaming OpenAI-compatible server: the reply is a deterministic function of the messages."""

    def __init__(self, *, break_warm=False, break_concurrent=False, fail_chat=False, truncate_stream=False,
                 context=CONTEXT):
        self.break_warm = break_warm
        self.break_concurrent = break_concurrent
        self.fail_chat = fail_chat          # every chat request answers HTTP 500
        self.truncate_stream = truncate_stream  # the stream stops before its [DONE]
        self.context = context
        self.seen = set()           # the messages already answered once: the next one is a warm request
        self.requests = []          # every chat body the server saw
        self.digests = []           # every reply's content, so a test can prove none reached the report
        self.auth = []              # the Authorization header of every request, None when absent
        self.in_flight = 0          # requests being answered at once: concurrent copies overlap
        self.lock = threading.Lock()

    def handle(self, handler):
        if handler.path == "/v1/models":
            body = json.dumps({"data": [{"id": "test", "max_model_len": self.context}]}).encode()
            self.answer(handler, 200, body, "application/json")
            return
        if handler.path == "/health":
            self.answer(handler, 200, b"ok\n", "text/plain")
            return
        if handler.path != "/v1/chat/completions":
            handler.send_response(404)
            handler.end_headers()
            return
        length = int(handler.headers.get("Content-Length", 0))
        body = json.loads(handler.rfile.read(length))
        with self.lock:
            self.requests.append(body)
            self.auth.append(handler.headers.get("Authorization"))
            self.in_flight += 1
            concurrent = self.in_flight > 1
        try:
            if self.fail_chat:
                self.answer(handler, 500, b'{"error": {"message": "the engine fell over"}}', "application/json")
            else:
                self.stream(handler, body, concurrent)
        finally:
            with self.lock:
                self.in_flight -= 1

    def stream(self, handler, body, concurrent):
        """The reply's four content deltas, its usage and [DONE], as the front end streams them."""
        text = json.dumps(body["messages"], sort_keys=True)
        # proportional to the prompt's own length, as a real tokenizer's count is
        prompt_tokens = max(1, round(sum(len(message.get("content") or "") for message in body["messages"])
                                     / CHARS_PER_TOKEN))
        digest = hashlib.sha256(text.encode()).hexdigest()[:32]
        warm = text in self.seen
        self.seen.add(text)
        self.digests.append(digest)
        if self.break_warm and warm:
            digest = "0" * 32
        if self.break_concurrent and concurrent:
            digest = "2" * 32
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.end_headers()
        for piece in (digest[:8], digest[8:16], digest[16:24], digest[24:]):
            chunk = {"id": "c", "object": "chat.completion.chunk", "choices": [{"delta": {"content": piece}}]}
            handler.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": 4, "total_tokens": prompt_tokens + 4,
                 "prompt_tokens_details": {"cached_tokens": 0 if not warm else max(0, prompt_tokens - 4)}}
        if self.truncate_stream:
            handler.wfile.flush()
            return
        final = {"id": "c", "object": "chat.completion.chunk", "choices": [], "usage": usage}
        handler.wfile.write(b"data: " + json.dumps(final).encode() + b"\n\n")
        handler.wfile.write(b"data: [DONE]\n\n")
        handler.wfile.flush()

    def answer(self, handler, status, body, content_type):
        handler.send_response(status)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)


class FakeHTTPServer(ThreadingHTTPServer):
    """The stdlib threaded server with room for the gate's batches: its default backlog of 5 is too small."""

    request_queue_size = 128
    daemon_threads = True       # the gate's connections can outlive shutdown


def serve(fake):
    """The fake server on a free port, in a thread; returns the server and its /v1 base URL."""
    handler = type("Handler", (BaseHTTPRequestHandler,), {
        "do_GET": lambda self: fake.handle(self),
        "do_POST": lambda self: fake.handle(self),
        "log_message": lambda *args: None})
    server = FakeHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}/v1"


def prompt_of(body):
    """The text the server received: the messages' contents, which for one user message is the prompt."""
    return "\n".join(message.get("content") or "" for message in body["messages"])


def tag_of(prompt):
    """The case name the prompt's header carries, or the empty string."""
    match = re.search(r"\n([^:\n]*): count the lines below", prompt)
    return match.group(1) if match else ""


def common_prefix(left, right):
    """How many leading bytes two prompts share."""
    shared = 0
    for a, b in zip(left, right):
        if a != b:
            break
        shared += 1
    return shared


def header_end(prompt):
    """The offset just past a prompt's header: its nonce line and the line naming its case."""
    first = prompt.index("\n")
    return prompt.index("\n", first + 1) + 1


class ShareGateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def run_gate(self, fake, argv, out="report.json"):
        """The gate's main against the fake server: (exit code, report or None, stdout)."""
        server, base = serve(fake)

        def stop():
            server.shutdown()
            server.server_close()

        self.addCleanup(stop)
        path = os.path.join(self.tmp.name, out)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = share_gate.main(["run", "--base", base, "--out", path, *argv])
        report = None
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                report = json.load(handle)
        return code, report, stdout.getvalue()

    def write_report(self, report, name):
        path = os.path.join(self.tmp.name, name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle)
        return path

    def test_equal_replies_pass(self):
        code, report, _ = self.run_gate(FakeServer(),
                                        ["--sizes", str(SIZE), "--levels", "2", "--perf", "2"])
        self.assertEqual(code, 0)
        self.assertTrue(report["pass"])
        self.assertEqual(report["failures"], [])
        cases = {record["case"] for record in report["requests"]}
        for case in ("calibration", f"resumed-{SIZE}", f"fork-{SIZE}", "concurrent-2", "perf-2"):
            self.assertIn(case, cases)
        # resumed = fresh: the warm reply hashes the same as the cold one; the appended one differs
        resumed = {record["role"]: record for record in report["requests"] if record["case"] == f"resumed-{SIZE}"}
        self.assertEqual(resumed["warm"]["sha256"], resumed["cold"]["sha256"])
        self.assertNotEqual(resumed["appended"]["sha256"], resumed["cold"]["sha256"])
        # the calibration fixed the sizing and the measured prompt is the asked one
        self.assertIn("tokens_per_line", report["calibration"])
        self.assertLess(abs(resumed["cold"]["usage"]["prompt_tokens"] - SIZE), 0.2 * SIZE)
        self.assertGreater(resumed["warm"]["usage"]["cached_tokens"], 0)
        # every request carries a case id, a size, a hash, timings and usage
        for record in report["requests"]:
            self.assertRegex(record["sha256"], r"\A[0-9a-f]{64}\Z")
            self.assertIsInstance(record["size"], int)
            self.assertIsInstance(record["total_s"], float)
            self.assertIn("usage", record)
        # the perf case reports the aggregate and per-stream decode rates
        perf = report["perf_cases"][0]
        self.assertEqual(perf["case"], "perf-2")
        self.assertEqual(len(perf["decode_tok_s"]), 2)
        self.assertEqual(len(perf["ttft_s"]), 2)
        self.assertIsNotNone(perf["aggregate_decode_tok_s"])

    def test_no_prompt_or_reply_text_in_the_report(self):
        fake = FakeServer()
        code, _, stdout = self.run_gate(fake, ["--sizes", str(SIZE), "--levels", "1", "--perf", "1"])
        self.assertEqual(code, 0)
        with open(os.path.join(self.tmp.name, "report.json"), encoding="utf-8") as handle:
            body = handle.read()
        # the filler's header and a line of it are nowhere in the report or stdout, and neither is one
        # byte of any reply the server streamed
        self.assertNotIn("count the lines below", body)
        self.assertNotIn("count the lines below", stdout)
        self.assertNotIn("line 0000000", body)
        self.assertNotIn("line 0000000", stdout)
        self.assertTrue(fake.digests)
        for digest in fake.digests:
            self.assertNotIn(digest, body)
            self.assertNotIn(digest, stdout)

    def test_warm_reply_differing_is_a_note(self):
        code, report, stdout = self.run_gate(FakeServer(break_warm=True),
                                             ["--sizes", str(SIZE), "--levels", "1", "--perf", "1"])
        # a resumed request takes a regenerate path that matches the cold one to numerics, not bits
        self.assertEqual(code, 0)
        self.assertTrue(report["pass"])
        self.assertEqual(report["notes"], [{"case": f"resumed-{SIZE}", "warm_equals_cold": False}])
        self.assertIn(f"note resumed-{SIZE}: warm differs from cold", stdout)

    def test_concurrent_compared_with_alone_on_a_fresh_engine(self):
        args = ["--sizes", str(SIZE), "--levels", "2", "--perf", "1", "--nonce", "alone-nonce"]
        code, batches, _ = self.run_gate(FakeServer(), args, out="run.json")
        self.assertEqual(code, 0)
        concurrent = [record for record in batches["requests"] if record["case"] == "concurrent-2"]
        self.assertEqual(sorted(record["role"] for record in concurrent),
                         ["distinct-1", "distinct-2", "same-1", "same-2"])
        # the distinct batch goes while nothing of its prompts is in the engine: every one is cold
        self.assertTrue(all(record["usage"]["cached_tokens"] == 0 for record in concurrent
                            if record["role"].startswith("distinct-")))
        # the alone run, against a fresh engine, sends each prompt once and cold
        fresh = FakeServer()
        code, alone, _ = self.run_gate(fresh, [*args, "--cases", "alone"], out="alone.json")
        self.assertEqual(code, 0)
        lone = [record for record in alone["requests"] if record["case"] == "concurrent-2"]
        self.assertEqual(sorted(record["role"] for record in lone), ["alone-0", "alone-1", "alone-2"])
        self.assertTrue(all(record["usage"]["cached_tokens"] == 0 for record in lone))
        left = self.write_report(batches, "left.json")
        right = self.write_report(alone, "right.json")
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(share_gate.main(["compare", left, right]), 0)
        self.assertIn("0 differing", stdout.getvalue())
        # one distinct reply that differs from its prompt's alone reply: the case is listed, exit 1
        next(record for record in batches["requests"]
             if record["case"] == "concurrent-2" and record["role"] == "distinct-2")["sha256"] = "f" * 64
        left = self.write_report(batches, "left.json")
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(share_gate.main(["compare", left, right]), 1)
        self.assertIn("concurrent-2: distinct-2", stdout.getvalue())
        # copies that equal each other but not the alone reply of their prompt fail the compare too
        for record in batches["requests"]:
            if record["case"] == "concurrent-2" and record["role"].startswith("same-"):
                record["sha256"] = "e" * 64
        left = self.write_report(batches, "left.json")
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(share_gate.main(["compare", left, right]), 1)
        self.assertIn("concurrent-2: distinct-2, same-1, same-2", stdout.getvalue())

    def test_fork_only_is_cold_and_matches_the_full_run(self):
        args = ["--sizes", str(SIZE), "--levels", "1", "--perf", "1", "--nonce", "fork-cold"]
        code, full, _ = self.run_gate(FakeServer(), args, out="full.json")
        self.assertEqual(code, 0)
        code, cold, _ = self.run_gate(FakeServer(), [*args, "--cases", "fork"], out="cold.json")
        self.assertEqual(code, 0)
        # the fork-only run sends the calibration probes and the fork prompt, nothing else
        self.assertEqual({record["case"] for record in cold["requests"]}, {"calibration", f"fork-{SIZE}"})
        fork = next(record for record in cold["requests"] if record["case"] == f"fork-{SIZE}")
        # nothing of the fork prompt's prefix is in the engine: this is the cold half of the comparison
        self.assertEqual(fork["usage"]["cached_tokens"], 0)
        # its prompt is byte for byte the full run's, so the replies hash the same
        self.assertEqual(fork["sha256"], next(record["sha256"] for record in full["requests"]
                                              if record["case"] == f"fork-{SIZE}"))
        left = self.write_report(full, "full2.json")
        right = self.write_report(cold, "cold2.json")
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(share_gate.main(["compare", left, right]), 0)
        self.assertIn("0 differing", stdout.getvalue())

    def test_no_two_cases_share_a_prompt_prefix(self):
        """No case's prompt is a prefix of another case's: the sharing never reaches the filler body.

        The fork case is left out here (its prompt starts with its parent's bytes by design); its own test
        holds it to exactly the parent's quarter.
        """
        nonce = "prefix-check"
        fake = FakeServer()
        code, _, _ = self.run_gate(fake, ["--nonce", nonce, "--sizes", f"{SIZE},1024", "--levels", "2",
                                          "--perf", "2", "--cases", "calibration,resumed,concurrent,perf"])
        self.assertEqual(code, 0)
        prompts = [prompt_of(body) for body in fake.requests]
        tags = [tag_of(prompt) for prompt in prompts]
        self.assertTrue(all(tags), tags)
        for i, (left, tag) in enumerate(zip(prompts, tags)):
            for right, other in zip(prompts[i + 1:], tags[i + 1:]):
                if other == tag:
                    continue        # cold/warm/appended, or a batch and its own alone run: meant to match
                shared = common_prefix(left, right)
                # the shared bytes are the nonce line and, at most, the two tags' own common prefix: they
                # end inside the two header lines, so no body line is ever shared
                self.assertLessEqual(shared, header_end(left), f"{tag} and {other} share a body line")
                self.assertLessEqual(shared, header_end(right), f"{tag} and {other} share a body line")
                self.assertLessEqual(shared, len(nonce) + 1 + common_prefix(tag + ":", other + ":"),
                                     f"{tag} and {other} share more than their tags")
        # the case ids the headers name are the ones the report has
        self.assertEqual({tag.split("-")[0] for tag in tags}, {"calibration", "resumed", "concurrent", "perf"})
        # and the two sizes of one kind, the pair that used to nest, share only their kind's name
        small = next(prompt for prompt, tag in zip(prompts, tags) if tag == f"resumed-{SIZE}")
        large = next(prompt for prompt, tag in zip(prompts, tags) if tag == "resumed-1024")
        self.assertEqual(common_prefix(small, large), len(nonce) + 1 + len("resumed-"))

    def test_fork_prompt_shares_exactly_its_parents_quarter(self):
        nonce = "fork-quarter"
        fake = FakeServer()
        code, report, _ = self.run_gate(fake, ["--nonce", nonce, "--sizes", str(SIZE), "--levels", "1",
                                               "--perf", "1", "--cases", "fork"])
        self.assertEqual(code, 0)
        prompts = [prompt_of(body) for body in fake.requests]
        fork = next(prompt for prompt in prompts if tag_of(prompt) == f"resumed-{SIZE}")
        # the parent the fork-only run never sent, rebuilt from the nonce, the size and the ratio
        parent = share_gate.filler(f"resumed-{SIZE}", SIZE, report["calibration"]["tokens_per_line"], nonce)
        shared = common_prefix(fork, parent)
        self.assertGreaterEqual(shared, 0.2 * len(parent))
        self.assertLessEqual(shared, 0.3 * len(parent))
        # nothing else of the run shares more than the nonce line with the fork prompt
        for prompt in prompts:
            if prompt == fork:
                continue
            self.assertLessEqual(common_prefix(fork, prompt), len(nonce) + 1)

    def test_perf_prompts_are_cold_and_their_own(self):
        fake = FakeServer()
        code, report, _ = self.run_gate(fake, ["--nonce", "perf-cold", "--sizes", str(SIZE), "--levels", "2",
                                               "--perf", "1,2"])
        self.assertEqual(code, 0)
        prompts = [prompt_of(body) for body in fake.requests]
        tags = [tag_of(prompt) for prompt in prompts]
        seen = set()
        for i, (prompt, tag) in enumerate(zip(prompts, tags)):
            # the first request of any case is cold: nothing before it carried the same text
            if tag not in seen:
                self.assertNotIn(prompt, prompts[:i], f"{tag} was already sent before its first request")
                seen.add(tag)
            if tag.startswith("perf-"):
                # the perf prompts belong to their level and stream, so no earlier request carried them
                self.assertNotIn(prompt, prompts[:i], f"{tag} reused an earlier prompt")
                self.assertEqual(seen & {tag}, {tag})
        perf_tags = {tag for tag in tags if tag.startswith("perf-")}
        self.assertEqual(perf_tags, {"perf-1-1", "perf-2-1", "perf-2-2"})
        perf_records = [record for record in report["requests"] if record["case"].startswith("perf-")]
        self.assertTrue(all(record["usage"]["cached_tokens"] == 0 for record in perf_records),
                        [record["usage"] for record in perf_records])

    def test_case_ids_are_stable_across_runs(self):
        args = ["--sizes", f"{SIZE},1024", "--levels", "1", "--perf", "1"]
        _, first, _ = self.run_gate(FakeServer(), [*args, "--nonce", "one"])
        _, second, _ = self.run_gate(FakeServer(), [*args, "--nonce", "two"])
        cases = lambda report: sorted({record["case"] for record in report["requests"]})
        self.assertEqual(cases(first), cases(second))
        # the nonce leads every prompt, so the two runs' hashes differ
        self.assertNotEqual(first["requests"][0]["sha256"], second["requests"][0]["sha256"])

    def test_compare(self):
        args = ["--sizes", str(SIZE), "--levels", "1", "--perf", "1", "--nonce", "compare-nonce"]
        _, first, _ = self.run_gate(FakeServer(), args, out="first.json")
        _, second, _ = self.run_gate(FakeServer(), args, out="second.json")
        left = self.write_report(first, "left.json")
        right = self.write_report(second, "right.json")
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(share_gate.main(["compare", left, right]), 0)
        self.assertIn("0 differing", stdout.getvalue())
        # one hash changed in one report: the case id is listed and the exit code is 1
        record = next(record for record in second["requests"]
                      if record["case"] == f"resumed-{SIZE}" and record["role"] == "warm")
        record["sha256"] = "f" * 64
        right = self.write_report(second, "right.json")
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(share_gate.main(["compare", left, right]), 1)
        self.assertIn(f"resumed-{SIZE}", stdout.getvalue())
        # unusable input is 2: a missing file, a file that is not JSON, a report without requests
        missing = os.path.join(self.tmp.name, "missing.json")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(share_gate.main(["compare", left, missing]), 2)
        not_json = os.path.join(self.tmp.name, "not.json")
        with open(not_json, "w", encoding="utf-8") as handle:
            handle.write("{")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(share_gate.main(["compare", left, not_json]), 2)
        empty = os.path.join(self.tmp.name, "empty.json")
        with open(empty, "w", encoding="utf-8") as handle:
            handle.write("{}")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(share_gate.main(["compare", left, empty]), 2)

    def test_grow16_grows_and_compares(self):
        code, report, _ = self.run_gate(FakeServer(), ["--sizes", str(SIZE), "--levels", "1", "--perf", "1",
                                                       "--grow16", "2048"])
        self.assertEqual(code, 0)
        self.assertTrue(report["pass"])
        for seed in range(1, 17):
            case = f"grow16-{seed}"
            records = [record for record in report["requests"] if record["case"] == case]
            roles = [record["role"] for record in records]
            self.assertIn("final", roles)
            self.assertIn("final-alone", roles)
            final = next(record for record in records if record["role"] == "final")
            alone = next(record for record in records if record["role"] == "final-alone")
            self.assertEqual(final["sha256"], alone["sha256"])
            # each session passed the asked prompt size, grown turn by turn
            self.assertGreaterEqual(alone["usage"]["prompt_tokens"], 2048)
            self.assertGreater(len(roles), 2)

    def test_http_error_stops_the_run(self):
        fake = FakeServer(fail_chat=True)
        code, report, _ = self.run_gate(fake, ["--sizes", str(SIZE), "--levels", "1", "--perf", "1"])
        self.assertEqual(code, 1)
        self.assertFalse(report["pass"])
        self.assertIn("HTTP 500", json.dumps(report["failures"]))
        self.assertEqual(len(fake.requests), 1)     # the first request failed: nothing after it was sent

    def test_broken_stream_stops_the_run(self):
        fake = FakeServer(truncate_stream=True)
        code, report, _ = self.run_gate(fake, ["--sizes", str(SIZE), "--levels", "1", "--perf", "1"])
        self.assertEqual(code, 1)
        self.assertFalse(report["pass"])
        self.assertIn("without [DONE]", json.dumps(report["failures"]))
        self.assertEqual(len(fake.requests), 1)

    def test_dry_run_sends_nothing(self):
        fake = FakeServer()
        server, base = serve(fake)

        def stop():
            server.shutdown()
            server.server_close()

        self.addCleanup(stop)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = share_gate.main(["run", "--base", base, "--dry-run", "--sizes", str(SIZE),
                                    "--levels", "2", "--perf", "2", "--grow16", "1024"])
        self.assertEqual(code, 0)
        plan = json.loads(stdout.getvalue())
        self.assertEqual(plan["cases"], share_gate.CASE_DEFAULT.split(","))
        self.assertNotIn("alone", plan["cases"])
        ids = [case["id"] for case in plan["planned"]]
        for case in ("calibration", f"resumed-{SIZE}", f"fork-{SIZE}", "concurrent-2", "grow16-16", "perf-2"):
            self.assertIn(case, ids)
        self.assertGreater(plan["requests_total"], 0)
        # a case list names only its own cases; the calibration always runs
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = share_gate.main(["run", "--base", base, "--dry-run", "--sizes", str(SIZE), "--cases", "fork"])
        self.assertEqual(code, 0)
        self.assertEqual([case["id"] for case in json.loads(stdout.getvalue())["planned"]],
                         ["calibration", f"fork-{SIZE}"])
        self.assertEqual(fake.requests, [])
        self.assertEqual(fake.auth, [])

    def test_key_file(self):
        key = "sk-" + hashlib.sha256(b"the gate's key").hexdigest()[:24]
        key_file = os.path.join(self.tmp.name, "api-key")
        with open(key_file, "w", encoding="utf-8") as handle:
            handle.write(key + "\n")
        fake = FakeServer()
        code, _, stdout = self.run_gate(fake, ["--sizes", str(SIZE), "--levels", "1", "--perf", "1",
                                               "--key-file", key_file])
        self.assertEqual(code, 0)
        self.assertTrue(fake.auth)
        self.assertEqual(set(fake.auth), {"Bearer " + key})
        with open(os.path.join(self.tmp.name, "report.json"), encoding="utf-8") as handle:
            body = handle.read()
        self.assertNotIn(key, body)
        self.assertNotIn(key, stdout)
        # without the flag no Authorization header is sent at all
        plain = FakeServer()
        code, _, _ = self.run_gate(plain, ["--sizes", str(SIZE), "--levels", "1", "--perf", "1"])
        self.assertEqual(code, 0)
        self.assertTrue(plain.auth)
        self.assertEqual(set(plain.auth), {None})


if __name__ == "__main__":
    unittest.main()
