#!/usr/bin/env python3
"""The shared KV pool's defined outcome (docs/shared-kv.md §2.5), through rank 0's socket
(serve_test.py's client). A server started with --shared-kv and a pool of a few pages
(--session-gib 0.02: about 5 pages, ~20,000 positions):

  1. a request whose prompt fits but whose generation runs past the pool ends with reason 'error' and
     the pool's text ("the shared KV pool is full ("; the numbers in it are the run's, not checked);
  2. the tokens it sent before the error stand: the same prompt, with a shorter max_new that fits, run
     afterwards in the same engine (in another session), gives them as its first tokens (the common
     prefix of the two replies);
  3. the engine stays up: a following small request ends with 'length'.

The prompt of check 1 must need nearly the whole pool, so that the error comes after a few positions
(the pool holds about 5 pages; the prompt is --prompt positions, 20,300 by default). The test reads
only rank 0's replies; the ranks' logs (no crash) are the orchestrator's.

  python3 engine/server/tests/pool_test.py [--socket /tmp/dsgb10-engine.sock] [--tokens tokens.npy]
  python3 engine/server/tests/pool_test.py --prompt 16000 --max-new 20000

The engine: dsgb10_rank --mode serve --shared-kv --session-gib 0.02 --slots 2 --disk-gib 0
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_test import Engine, load_npy_i32, show  # noqa: E402

# The pool's text (docs/shared-kv.md §2.5), matched by its stable prefix: the numbers in it are the
# run's (the bytes in use and the position the page was needed for).
POOL_TEXT = 'the shared KV pool is full ('
# the boxes' token files, in the order to try when --tokens is not given
DEFAULT_TOKENS = ['/data/dsgb10/golden/synth270k/tokens.npy',
                  os.path.expanduser('~/golden/synth/tokens.npy'),
                  os.path.expanduser('~/golden/long-tokens/tokens.npy')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', default='/tmp/dsgb10-engine.sock')
    ap.add_argument('--tokens', default=None, help='a token file (default: the first of the box\'s that exists)')
    ap.add_argument('--prompt', type=int, default=20300, help='the failing request\'s prompt, positions')
    ap.add_argument('--max-new', type=int, default=20000, help='the failing request\'s max_new_tokens')
    ap.add_argument('--short', type=int, default=32, help='the reference request\'s max_new_tokens')
    a = ap.parse_args()
    fn = a.tokens or next((p for p in DEFAULT_TOKENS if os.path.exists(p)), DEFAULT_TOKENS[0])
    T = load_npy_i32(fn)
    if len(T) < a.prompt + 1:
        print('pool_test: %s holds %d tokens, the prompt needs %d' % (fn, len(T), a.prompt + 1))
        return 1
    e = Engine(a.socket)
    print('engine: %s, context %d; tokens %s (%d)' % (e.name, e.max_ctx, fn, len(T)))
    S = 0x9001000000000000
    res = []

    # 1. the generation runs past the pool: reason error and the pool's text
    x = e.run(S + 1, T[:a.prompt], max_new=a.max_new, stop=(), keep=False)
    show('past the pool', x)
    print('  the pool\'s text: %s' % (x['error'] or '(none)'))
    res.append(x['reason'] == 'error' and x['error'].startswith(POOL_TEXT))
    # 2. the tokens sent before the error stand: the same prompt with a shorter max_new, another session
    r = e.run(S + 2, T[:a.prompt], max_new=a.short, stop=(), keep=False)
    show('the same prompt, short', r)
    n = min(len(x['ids']), len(r['ids']))
    res.append(n > 0 and x['ids'][:n] == r['ids'][:n])
    print('  the %d tokens sent before the error: the reference\'s first %d, %s' % (
        len(x['ids']), n, 'equal' if n and x['ids'][:n] == r['ids'][:n] else 'DIFFER'))
    # 3. the engine stays up: a small request goes through
    y = e.run(S + 3, T[:64], max_new=8, stop=(), keep=False)
    show('after it', y)
    res.append(y['reason'] in ('length', 'stop') and y['completion'] == 8)
    print('pool_test: %d of %d checks pass' % (sum(res), len(res)))
    return 0 if all(res) else 1


if __name__ == '__main__':
    sys.exit(main())
