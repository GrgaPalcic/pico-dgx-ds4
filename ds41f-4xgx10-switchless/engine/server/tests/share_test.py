#!/usr/bin/env python3
"""What --shared-kv (docs/shared-kv.md) adds, through rank 0's socket (serve_test.py's client, its
cold-comparison helpers and its CED turn rule: a turn of at least 128 new positions replays the decoder
and must equal the same prompt prefilled cold, bit for bit; a shorter one continues exactly and must
equal the same history replayed). A page holds 4,096 positions, so every scenario runs contexts of
several pages and compares each reuse turn with the same prompt prefilled cold in a fresh session.

  1. a live session restarts below its last save (a divergence 9,000 positions back, inside its pages,
     far above the save the session never took): the pages below the restart must stay, so the turn
     equals cold (state_drop's first bug);
  2. a base rewrites a page its forks share: forks at 25,000 and 24,576 share the base's pages, the
     base then diverges at 19,000 (restart 18,432, mid-page: the copy-on-write) and at 18,000
     (restart 16,384, page-aligned: the drop), both inside page 4, which both forks hold, and the
     forks' turns after the rewrite must equal cold, as must a fork of a fork and its turn after the
     base is freed (state_drop's copy-on-write bug);
  3. slot moves: --slots N sessions, N + 2 of them of 12,000 positions, round-robin for three rounds
     of +300 turns, so sessions leave and take slots; every turn reuses its own state (only the last
     round is compared with cold, to keep the run short);
  4. a session growing past several pages while another runs: two sessions alternating +4,000-position
     turns up to 40,000 each; the last turn of each equals cold.

  python3 engine/server/tests/share_test.py [--socket /tmp/dsgb10-engine.sock] [--tokens tokens.npy]
  python3 engine/server/tests/share_test.py --dry-run     # the request plan, no socket
  python3 engine/server/tests/share_test.py --slots 4     # the server's --slots (scenario 3)

The engine: dsgb10_rank --mode serve --shared-kv --max-ctx 262144 --slots 4 (the same without
--shared-kv is the control run).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_test import Engine, close, load_npy_i32, same, show  # noqa: E402

MAX_NEW = 16   # a turn's decoded tokens
TURN = 300     # a compared turn's new positions (at least 128: CED's replay, engine.md Serving)
BLOCK = 4000   # scenario 4's growth step
DRY = False
# the boxes' token files, in the order to try when --tokens is not given (pool_test.py's list)
DEFAULT_TOKENS = ['/data/dsgb10/golden/synth270k/tokens.npy',
                  os.path.expanduser('~/golden/synth/tokens.npy'),
                  os.path.expanduser('~/golden/long-tokens/tokens.npy')]
NEED = 215000  # the tokens the scenarios' regions and blocks span


class Client:
    """rank 0's socket: one line a request (serve_test.Engine)."""
    def __init__(self, e):
        self.e = e

    def turn(self, name, sid, prompt, **kw):
        kw.setdefault('max_new', MAX_NEW)
        kw.setdefault('n_top', 2)
        o = self.e.run(sid, prompt, **kw)
        show(name, o)
        return o

    def cold(self, name, sid, prompt, **kw):
        kw['keep'] = False
        return self.turn('  ' + name, sid, prompt, **kw)

    def release(self, sid):
        self.e.release(sid)


class Dry:
    """--dry-run: the request plan, no socket."""
    def __init__(self, T):
        self.T = T
        self.n = 0

    def _one(self, tag, name, sid, prompt, kw):
        self.n += 1
        print('%-5s %-38s session %016x prompt %6d max_new %2d%s' % (
            tag, name, sid, len(prompt), kw.get('max_new', MAX_NEW),
            (' base %016x' % kw['base']) if kw.get('base') else ''))
        sys.stdout.flush()
        return {'ids': [self.T[0]] * kw.get('max_new', MAX_NEW), 'lp': [0.0], 'top': [], 'cached': 0,
                'prompt': len(prompt), 'completion': kw.get('max_new', MAX_NEW), 'reason': 'length',
                'error': '', 't_first': 0.0}

    def turn(self, name, sid, prompt, **kw):
        return self._one('turn', name, sid, prompt, kw)

    def cold(self, name, sid, prompt, **kw):
        return self._one('cold', name, sid, prompt, kw)

    def release(self, sid):
        print('release session %016x' % sid)


def blk(T, off, n, avoid=()):
    """`n` tokens at or after `off` whose first is none of `avoid`: the prompt diverges exactly there."""
    while off + n <= len(T) and T[off] in avoid:
        off += 1
    assert off + n <= len(T), 'the token file is too short for the block at %d' % off
    return T[off:off + n], off + n


def chk(name, ok, note=''):
    """One line a check (--dry-run prints the plan and counts nothing); the caller counts it."""
    if DRY:
        print('  %-38s (dry)' % name)
        return True
    print('  %-38s %s%s' % (name, 'ok' if ok else 'FAILED', note))
    return bool(ok)


def cmp(name, a, b):
    """A reuse turn against its reference (the same prompt prefilled cold, or the same history
    replayed): bit for bit, the tokens and the first token's logprob."""
    if DRY:
        print('  %-38s (dry)' % name)
        return True
    return same(name, a, b)


def clos(name, a, b):
    """A regenerate (its last positions through the step): the same first token, its logprob close."""
    if DRY:
        print('  %-38s (dry)' % name)
        return True
    return close(name, a, b)


def scenario1(c, T, res):
    """1. A live session restarts below its last save: the pages below the restart must stay."""
    S = 0x5601000000000000
    A, R, C = S + 1, S + 2, S + 3
    p1 = T[:20000]
    a = c.turn('1. A cold 20000 (+64)', A, p1, max_new=64)
    res.append(chk('1. the first turn is cold', a['cached'] == 0, ' (cached %d)' % a['cached']))
    # the same request in a fresh session: the +40 turn is an exact continuation (engine.md Serving)
    r1 = c.turn('1. the same in a fresh session', R, p1, max_new=64)
    res.append(chk('1. the fresh session agrees', r1['ids'] == a['ids']))
    b40, _ = blk(T, 60000, 40, a['ids'][:1])
    p2 = p1 + list(a['ids']) + b40
    b = c.turn('1. A +40 (exact)', A, p2)
    r2 = c.turn('1. the same +40 in the fresh session', R, p1 + list(r1['ids']) + b40)
    res.append(cmp('1. +40 vs the same history replayed', b, r2))  # shorter than 128: exact
    # a divergence 9,000 positions back, far above the save the session never took: +300 positions
    q = 11000
    d300, _ = blk(T, 70000, TURN, (T[q],))
    p3 = T[:q] + d300
    d = c.turn('1. A diverge at %d (+%d)' % (q, TURN), A, p3)
    cold = c.cold('1. the same prompt cold', C, p3)
    res.append(cmp('1. the divergence vs cold', d, cold))
    res.append(chk('1. the divergence reused its pages', 4096 <= d['cached'] <= q,
                   ' (cached %d)' % d['cached']))
    # the same prompt again: a regenerate, its last positions through the step (numerics, not bits)
    g = c.turn('1. A regenerate', A, p3)
    res.append(clos('1. the regenerate vs cold', g, cold))
    for s in (A, R, C):
        c.release(s)
    return True


def scenario2(c, T, res):
    """2. A base rewrites a page its forks share: the forks' entries must survive (copy-on-write)."""
    S = 0x5602000000000000
    B, F1, F2, F3, C = S + 1, S + 2, S + 3, S + 4, S + 5
    b = c.turn('2. B cold 30000', B, T[:30000], max_new=24)
    res.append(chk('2. the base is cold', b['cached'] == 0, ' (cached %d)' % b['cached']))
    # F1 at 25,000 (mid-page) and F2 at 24,576 (page-aligned): each shares the base's full pages
    f1b, _ = blk(T, 40000, TURN, (T[25000],))
    p1 = T[:25000] + f1b
    f1 = c.turn('2. F1 fork of B at 25000', F1, p1, base=B)
    res.append(cmp('2. the fork at 25000 vs cold', f1, c.cold('2. the same prompt cold', C, p1)))
    res.append(chk("2. the fork shared the base's pages", f1['cached'] >= 16384,
                   ' (cached %d)' % f1['cached']))
    f2b, _ = blk(T, 45000, TURN, (T[24576],))
    p2 = T[:24576] + f2b
    f2 = c.turn('2. F2 fork of B at 24576', F2, p2, base=B)
    res.append(cmp('2. the fork at 24576 vs cold', f2, c.cold('2. the same prompt cold', C, p2)))
    # the base rewrites the page its forks share, twice: at 19,000 (restart 18,432, mid-page, inside
    # page 4: the copy-on-write) and at 18,000 (restart 16,384, page-aligned: the drop)
    for q, offb, mid in ((19000, 50000, True), (18000, 51000, False)):
        bdb, _ = blk(T, offb, TURN, (T[q],))
        p3 = T[:q] + bdb
        d = c.turn('2. B diverge at %d (+%d)' % (q, TURN), B, p3)
        res.append(cmp("2. the base's divergence at %d vs cold" % q, d,
                       c.cold('2. the same prompt cold', C, p3)))
        ok = 16384 <= d['cached'] <= q and (bool(d['cached'] % 4096) if mid else not d['cached'] % 4096)
        res.append(chk('2. the divergence at %d restarted in page 4' % q, ok, ' (cached %d)' % d['cached']))
    # the forks' turns after the base's rewrite
    c1b, _ = blk(T, 55000, TURN, f1['ids'][:1])
    p4 = p1 + c1b
    g1 = c.turn('2. F1 continue (+%d)' % TURN, F1, p4)
    res.append(cmp('2. F1 after the rewrite vs cold', g1, c.cold('2. the same prompt cold', C, p4)))
    c2b, _ = blk(T, 60000, TURN, f2['ids'][:1])
    p5 = p2 + c2b
    g2 = c.turn('2. F2 continue (+%d)' % TURN, F2, p5)
    res.append(cmp('2. F2 after the rewrite vs cold', g2, c.cold('2. the same prompt cold', C, p5)))
    # a fork of a fork, and the base freed before it continues
    f3b, _ = blk(T, 65000, TURN, (T[20000],))
    p6 = T[:20000] + f3b
    f3 = c.turn('2. F3 fork of F1 at 20000', F3, p6, base=F1)
    res.append(cmp('2. the fork of a fork vs cold', f3, c.cold('2. the same prompt cold', C, p6)))
    c.release(B)
    c3b, _ = blk(T, 70000, TURN, f3['ids'][:1])
    p7 = p6 + c3b
    g3 = c.turn('2. F3 continue (B freed) (+%d)' % TURN, F3, p7)
    res.append(cmp('2. F3 after B freed vs cold', g3, c.cold('2. the same prompt cold', C, p7)))
    for s in (B, F1, F2, F3, C):
        c.release(s)
    return True


def scenario3(c, T, res, nslots):
    """3. Slot moves: N + 2 sessions round-robin, every turn from the session's own state."""
    S = 0x5603000000000000
    sess = [S + 1 + i for i in range(nslots + 2)]
    cold = S + 0xff
    base = {s: T[12000 * i:12000 * (i + 1)] for i, s in enumerate(sess)}
    off = {s: 90000 + 1000 * i for i, s in enumerate(sess)}
    last = {}
    for rnd in range(3):
        for i, s in enumerate(sess):
            if rnd == 0:
                o = c.turn('3. %03x cold 12000' % s, s, base[s])
                res.append(chk('3. %03x cold' % s, o['cached'] == 0, ' (cached %d)' % o['cached']))
                # the history's token at 12,000 is the first decoded id: the next turn must not match it
                last[s] = o['ids'][0] if o['ids'] else T[12000 * (i + 1)]
                continue
            b, off[s] = blk(T, off[s], TURN, (last[s],))
            last[s] = b[0]
            p = base[s] + b
            o = c.turn('3. %03x round %d (+%d)' % (s, rnd + 1, TURN), s, p)
            if rnd == 2:
                res.append(cmp('3. %03x round 3 vs cold' % s, o, c.cold('3. the same prompt cold', cold, p)))
            else:
                res.append(chk('3. %03x round %d reused 12000' % (s, rnd + 1), o['cached'] >= 12000 - 1024,
                               ' (cached %d)' % o['cached']))
    for s in sess + [cold]:
        c.release(s)
    return True


def scenario4(c, T, res):
    """4. A session grows past several pages while another runs: the last turn of each equals cold."""
    S = 0x5604000000000000
    A, B, CA, CB = S + 1, S + 2, S + 3, S + 4
    pa, pb = T[120000:124000], T[170000:174000]
    oa, ob = 124000, 174000
    for k in range(10):
        a = c.turn('4. A %d (+%d)' % (len(pa), BLOCK), A, pa)
        b = c.turn('4. B %d (+%d)' % (len(pb), BLOCK), B, pb)
        if k == 0:
            res.append(chk('4. both sessions start cold', a['cached'] == 0 and b['cached'] == 0,
                           ' (cached %d, %d)' % (a['cached'], b['cached'])))
        elif k == 9:
            res.append(cmp("4. A's last turn vs cold", a, c.cold('4. the same prompt cold', CA, pa)))
            res.append(cmp("4. B's last turn vs cold", b, c.cold('4. the same prompt cold', CB, pb)))
        else:
            res.append(chk('4. A %d reused its end' % len(pa), a['cached'] >= len(pa) - BLOCK - 1024,
                           ' (cached %d)' % a['cached']))
            res.append(chk('4. B %d reused its end' % len(pb), b['cached'] >= len(pb) - BLOCK - 1024,
                           ' (cached %d)' % b['cached']))
        ba, oa = blk(T, oa, BLOCK, a['ids'][:1])
        pa = pa + ba
        bb, ob = blk(T, ob, BLOCK, b['ids'][:1])
        pb = pb + bb
    for s in (A, B, CA, CB):
        c.release(s)
    return True


def main():
    global DRY
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', default='/tmp/dsgb10-engine.sock')
    ap.add_argument('--tokens', default=None, help="a token file (default: the first of the box's that exists)")
    ap.add_argument('--slots', type=int, default=4, help="the server's --slots (scenario 3 runs N + 2 sessions)")
    ap.add_argument('--dry-run', action='store_true', help='print the request plan, no socket')
    a = ap.parse_args()
    fn = a.tokens or next((p for p in DEFAULT_TOKENS if os.path.exists(p)), DEFAULT_TOKENS[0])
    T = load_npy_i32(fn)
    DRY = a.dry_run
    res = []
    if DRY:
        c = Dry(T)
        print('share_test --dry-run: %s (%d tokens)' % (fn, len(T)))
    else:
        if len(T) < NEED:
            print('share_test: %s holds %d tokens, the scenarios need %d' % (fn, len(T), NEED))
            return 1
        e = Engine(a.socket)
        print('engine: %s, context %d; tokens %s (%d)' % (e.name, e.max_ctx, fn, len(T)))
        c = Client(e)
    scenario1(c, T, res)
    scenario2(c, T, res)
    scenario3(c, T, res, a.slots)
    scenario4(c, T, res)
    if DRY:
        print('share_test: dry run, %d requests planned' % c.n)
        return 0
    print('share_test: %d of %d checks pass' % (sum(res), len(res)))
    return 0 if all(res) else 1


if __name__ == '__main__':
    sys.exit(main())
