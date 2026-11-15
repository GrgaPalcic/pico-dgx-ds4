#!/usr/bin/env python3
"""Sampling's tail, through rank 0's socket (serve_test.py's client): --prompts prompts cut from a tokens file,
each sampled for --max-new tokens at temperature 1.0 with no top-k/top-p cut (the whole-vocabulary Gumbel-max
path), and every sampled token's log-probability under the engine's own distribution counted below 1e-2, 1e-4
and 1e-6. A faithful sampler draws a token under 1e-6 only as often as the distribution's mass there allows (a
few in ten thousand on this text); a sampler whose noise can reach +inf picks such tokens at a fixed extra rate
(about one in 130 at 129,280 ids).

  python3 engine/server/tests/tail_test.py --socket /tmp/dsgb10-engine.sock --tokens tokens.npy [--prompts 16]

Passes when at most --max-rare-per-1000 (1) of every 1,000 sampled tokens is under 1e-6. The engine:
dsgb10_rank --mode serve.
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_test import Engine, load_npy_i32  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', default='/tmp/dsgb10-engine.sock')
    ap.add_argument('--tokens', required=True)
    ap.add_argument('--prompts', type=int, default=16)
    ap.add_argument('--prompt-len', type=int, default=2000)
    ap.add_argument('--max-new', type=int, default=512)
    ap.add_argument('--top-p', type=float, default=1.0)
    ap.add_argument('--max-rare-per-1000', type=float, default=1.0, help='sampled tokens under 1e-6 allowed')
    a = ap.parse_args()
    T = load_npy_i32(a.tokens)
    e = Engine(a.socket)
    cuts = (math.log(1e-2), math.log(1e-4), math.log(1e-6))
    n = 0
    under = [0, 0, 0]
    for i in range(a.prompts):
        off = (7919 * i * 13) % max(1, len(T) - a.prompt_len)
        r = e.run(0x7461696c00000000 + i, T[off:off + a.prompt_len], max_new=a.max_new, temp=1.0, top_p=a.top_p,
                  seed=1000 + i, keep=False, n_top=1)
        lps = r['lp']
        n += len(lps)
        for j, c in enumerate(cuts):
            under[j] += sum(1 for lp in lps if lp < c)
        print(f'prompt {i}: {len(lps)} tokens; under 1e-2 {sum(1 for lp in lps if lp < cuts[0])}, '
              f'under 1e-6 {sum(1 for lp in lps if lp < cuts[2])}')
    print(f'tail_test: {n} sampled tokens (temperature 1.0, top_p {a.top_p}); under 1e-2: {under[0]}, under 1e-4: '
          f'{under[1]}, under 1e-6: {under[2]}')
    ok = under[2] <= a.max_rare_per_1000 * n / 1000
    print('tail_test: pass' if ok else 'tail_test: FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
