"""Step 2 -- align syntax fields to one byte and build the RBSP-byte mask.

``rbsp_mask(parser)`` returns a 256-bit int: bit ``b`` set iff RBSP byte ``b``
keeps the parser legal.  It walks *fields*, not bits:

1. look up ``code.step(code_state, r)`` (precompiled table, ``codes.py``);
2. OR the ``partial`` chunks (field still open at the byte boundary) directly;
3. for each completion, apply the handler once on a clone (the O(1) state
   update + next-field selection of Step 1) and recurse with ``r - length``
   bits.  Completions whose futures are provably identical (``vindep`` /
   ``vclass``) share one recursion; their chunks are spread over its result.

``reference_mask`` is the naive bit-by-bit definition of the same mask (clone
per bit, feed through ``Parser.feed_bit``).  It shares the grammar but none of
the tables or grouping, so equality of the two checks Step 2 exactly.
"""

from __future__ import annotations

from .grammar import Illegal, Parser


def _spread(chunks: int, sub: int, k: int) -> int:
    """OR ``sub`` into every ``k``-bit slot selected by the chunk bitmask."""
    acc = 0
    while chunks:
        low = chunks & -chunks
        acc |= sub << ((low.bit_length() - 1) << k)
        chunks ^= low
    return acc


def _child(p: Parser, value, length: int):
    c = p.clone()
    c.finish(value, length)
    return None if c.dead else c


def _sub(c: Parser, k: int) -> int:
    if k == 0:
        return 1
    if c.done:  # every RBSP ends byte-aligned; cannot happen mid-byte
        return 0
    return solve(c, k)


def solve(p: Parser, r: int) -> int:
    """Mask over the ``2**r`` suffixes of the current byte (MSB first)."""
    code = p.code
    cs = p.cs
    st = code.step(cs, r)
    acc = st.partial
    rel_prefix = cs[1] if code.rel else None

    if p.vindep:
        for L, chunks, v in st.by_len:
            if rel_prefix is not None:
                v = (rel_prefix << L) | v
            c = _child(p, v, L)
            if c is not None:
                sub = _sub(c, r - L)
                if sub:
                    acc |= _spread(chunks, sub, r - L)
        return acc

    vclass = p.vclass
    if vclass is None:
        for v, L, ch in st.comps:
            if rel_prefix is not None:
                v = (rel_prefix << L) | v
            c = _child(p, v, L)
            if c is not None:
                sub = _sub(c, r - L)
                if sub:
                    acc |= sub << (ch << (r - L))
        return acc

    groups = {}
    for v, L, ch in st.comps:
        if rel_prefix is not None:
            v = (rel_prefix << L) | v
        key = (L, vclass(p, v))
        g = groups.get(key)
        if g is None:
            groups[key] = [v, 1 << ch]
        else:
            g[1] |= 1 << ch
    for (L, _), (v, chunks) in groups.items():
        c = _child(p, v, L)
        if c is not None:
            sub = _sub(c, r - L)
            if sub:
                acc |= _spread(chunks, sub, r - L)
    return acc


def rbsp_mask(p: Parser) -> int:
    return solve(p, 8)


def reference_mask(p: Parser) -> int:
    """Bit-by-bit exploration of the same grammar (slow; for verification)."""
    acc = 0
    stack = [(p, 0, 0)]
    while stack:
        node, depth, prefix = stack.pop()
        for b in (0, 1):
            c = node.clone()
            try:
                c.feed_bit(b)
            except Illegal:
                continue
            nxt = (prefix << 1) | b
            if depth + 1 == 8:
                acc |= 1 << nxt
            elif not c.done:
                stack.append((c, depth + 1, nxt))
    return acc
