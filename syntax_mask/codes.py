"""Step 1b -- precompiled codeword transition tables.

Every H.264 syntax field is read with one *code*: fixed width ``u(n)``,
Exp-Golomb ``ue/se/me/te``, a CAVLC table, or the unary ``level_prefix``.  A
code is a small bit-level DFA (``bit``) whose states are *code states* (how much
of the current codeword has been read).  A code rejects a bit as soon as no
legal value can still be reached, so the legal-value set is folded into the
DFA itself.

The mask builder never walks that DFA per corpus byte.  ``Code.step(cs, r)``
returns, for a code state and the ``r`` bits left in the current byte:

* ``partial`` -- bitmask over the ``2**r`` possible ``r``-bit chunks that keep
  the field open (a legal proper prefix spanning the rest of the byte), and
* ``comps`` / ``by_len`` -- every legal codeword that *finishes* inside the
  byte: ``(value, length, chunk)``, and the same grouped by length.

``step`` results are memoized per interned code, so after warm-up this is a
table lookup.  Codes are interned by their parameters (kind, width, legal set),
which makes the cache a process-wide precompiled table.
"""

from __future__ import annotations

from . import cavlc_tables as T

MORE, DONE, INV = 0, 1, 2
MAX_CODENUM = (1 << 32) - 2  # largest ue(v) codeNum (31 leading zeros)
_STEP_CACHE_CAP = 200_000

# --------------------------------------------------------------------------
# legal sets: sorted tuple of disjoint inclusive intervals, or None (= any)
# --------------------------------------------------------------------------


def iv(lo: int, hi: int):
    return ((lo, hi),) if lo <= hi else ()


def ivset(values) -> tuple:
    """Merge an iterable of ints into interval form."""
    out = []
    for v in sorted(set(values)):
        if out and out[-1][1] + 1 == v:
            out[-1][1] = v
        else:
            out.append([v, v])
    return tuple((a, b) for a, b in out)


def iv_union(*sets) -> tuple:
    pts = sorted(p for s in sets for p in s)
    out = []
    for a, b in pts:
        if out and a <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return tuple((a, b) for a, b in out)


def iv_minus_point(legal, x):
    out = []
    for a, b in legal:
        if a <= x <= b:
            if a < x:
                out.append((a, x - 1))
            if x < b:
                out.append((x + 1, b))
        else:
            out.append((a, b))
    return tuple(out)


def _hits(legal, lo, hi) -> bool:
    for a, b in legal:
        if a > hi:
            return False
        if b >= lo:
            return True
    return False


def se_to_codenum(v: int) -> int:
    return 2 * v - 1 if v > 0 else -2 * v


def codenum_to_se(k: int) -> int:
    return (k + 1) >> 1 if k & 1 else -(k >> 1)


def se_legal(lo: int, hi: int) -> tuple:
    """Signed interval -> codeNum intervals (se(v) interleaves signs)."""
    if lo > hi:
        return ()
    if lo <= 0 <= hi:
        m = min(hi, -lo)
        parts = [(0, 2 * m)]
        # one-sided tails: odd codeNums (positive) or even codeNums (negative)
        parts += [(k, k) for k in range(2 * m + 1, 2 * hi, 2)] if hi > m else []
        parts += [(k, k) for k in range(2 * m + 2, -2 * lo + 1, 2)] if -lo > m else []
        return iv_union(parts)
    return ivset(se_to_codenum(v) for v in range(lo, hi + 1))


# --------------------------------------------------------------------------
# codes
# --------------------------------------------------------------------------


class Step:
    __slots__ = ("partial", "comps", "by_len")

    def __init__(self, partial, comps, by_len):
        self.partial = partial
        self.comps = comps
        self.by_len = by_len


class Code:
    """Bit-level DFA for one coding + legal-value set."""

    rel = False  # values in step tables are relative to the state's prefix

    def __init__(self, key):
        self.key = key
        self._steps = {}
        self.empty = False

    def bit(self, cs, b):  # -> (MORE, cs') | (DONE, value) | (INV, None)
        raise NotImplementedError

    def step_key(self, cs):
        return cs

    def step(self, cs, r: int) -> Step:
        k = (self.step_key(cs), r)
        s = self._steps.get(k)
        if s is None:
            s = self._build(k[0], r)
            if len(self._steps) >= _STEP_CACHE_CAP:
                self._steps.clear()
            self._steps[k] = s
        return s

    def _build(self, cs, r):
        partial = 0
        comps = []
        frontier = [(cs, 0)]
        for depth in range(1, r + 1):
            nxt = []
            for state, chunk in frontier:
                for b in (0, 1):
                    st, x = self.bit(state, b)
                    ch = (chunk << 1) | b
                    if st == MORE:
                        if depth == r:
                            partial |= 1 << ch
                        else:
                            nxt.append((x, ch))
                    elif st == DONE:
                        comps.append((x, depth, ch))
            frontier = nxt
            if not frontier:
                break
        groups = {}
        for v, L, ch in comps:
            m, rep = groups.get(L, (0, v))
            groups[L] = (m | (1 << ch), rep)
        by_len = tuple((L, m, rep) for L, (m, rep) in sorted(groups.items()))
        return Step(partial, tuple(comps), by_len)


class UCode(Code):
    """u(n) / f(n); ``legal`` over the n-bit value, None = any."""

    def __init__(self, key, n, legal):
        super().__init__(key)
        self.n = n
        self.legal = legal
        self.rel = legal is None
        self.init = (0, 0)
        self.empty = legal is not None and not legal

    def step_key(self, cs):
        return (cs[0], 0) if self.rel else cs

    def bit(self, cs, b):
        k, p = cs
        k += 1
        p = (p << 1) | b
        if self.legal is not None:
            rem = self.n - k
            if not _hits(self.legal, p << rem, ((p + 1) << rem) - 1):
                return INV, None
        if k == self.n:
            return DONE, p
        return MORE, (k, p)


class UECode(Code):
    """ue(v)-framed code; ``legal`` over codeNum; ``vmap`` maps codeNum."""

    def __init__(self, key, legal, vmap):
        super().__init__(key)
        self.legal = legal
        self.vmap = vmap  # None | "se" | tuple table
        self.init = (0, -1, 0, 0)  # (zeros, Z or -1 while in prefix, s, sv)
        self.empty = not legal

    def _out(self, k):
        vm = self.vmap
        if vm is None:
            return k
        if vm == "se":
            return codenum_to_se(k)
        return vm[k]

    def bit(self, cs, b):
        z, Z, s, sv = cs
        legal = self.legal
        if Z < 0:
            if b == 0:
                z += 1
                if z > 31 or not _hits(legal, (1 << z) - 1, MAX_CODENUM):
                    return INV, None
                return MORE, (z, -1, 0, 0)
            if z == 0:
                return (DONE, self._out(0)) if _hits(legal, 0, 0) else (INV, None)
            base = (1 << z) - 1
            if not _hits(legal, base, base + (1 << z) - 1):
                return INV, None
            return MORE, (z, z, 0, 0)
        s += 1
        sv = (sv << 1) | b
        span = 1 << (Z - s)
        lo = (1 << Z) - 1 + sv * span
        if not _hits(legal, lo, lo + span - 1):
            return INV, None
        if s == Z:
            return DONE, self._out(lo)
        return MORE, (z, Z, s, sv)


class InvBitCode(Code):
    """te(v) with cMax == 1: a single inverted bit (optionally only value 0)."""

    def __init__(self, key, only_zero=False):
        super().__init__(key)
        self.init = 0
        self.only_zero = only_zero

    def bit(self, cs, b):
        if self.only_zero and b == 0:
            return INV, None
        return DONE, 1 - b


class VLCCode(Code):
    """CAVLC table restricted to an allowed value set."""

    def __init__(self, key, label, allowed):
        super().__init__(key)
        self.codes = {}
        self.prefixes = set()
        for code, value in T.code_map(label).items():
            if allowed is not None and value not in allowed:
                continue
            n = len(code)
            self.codes[(n, int(code, 2))] = value
            for i in range(1, n):
                self.prefixes.add((i, int(code[:i], 2)))
        self.init = (0, 0)
        self.empty = not self.codes

    def bit(self, cs, b):
        nb = (cs[0] + 1, (cs[1] << 1) | b)
        v = self.codes.get(nb)
        if v is not None:
            return DONE, v
        if nb in self.prefixes:
            return MORE, nb
        return INV, None


class UnaryCode(Code):
    """level_prefix: ``z`` zeros then a one, value z <= maxz."""

    def __init__(self, key, maxz):
        super().__init__(key)
        self.maxz = maxz
        self.init = 0

    def bit(self, cs, b):
        if b:
            return DONE, cs
        return (MORE, cs + 1) if cs + 1 <= self.maxz else (INV, None)


# --------------------------------------------------------------------------
# interned constructors
# --------------------------------------------------------------------------

_CODES: dict = {}
ANY_CN = ((0, MAX_CODENUM),)


def _intern(key, factory):
    c = _CODES.get(key)
    if c is None:
        c = factory()
        _CODES[key] = c
    return c


def u(n: int, legal=None) -> Code:
    key = ("u", n, legal)
    return _intern(key, lambda: UCode(key, n, legal))


def ue(legal=ANY_CN) -> Code:
    legal = tuple((a, min(b, MAX_CODENUM)) for a, b in legal if a <= MAX_CODENUM)
    key = ("ue", legal)
    return _intern(key, lambda: UECode(key, legal, None))


_UE_RANGE = {}


def ue_range(lo: int, hi: int) -> Code:
    code = _UE_RANGE.get((lo, hi))
    if code is None:
        code = _UE_RANGE[(lo, hi)] = ue(iv(lo, hi))
    return code


def se(lo: int = -(1 << 31) + 1, hi: int = (1 << 31) - 1) -> Code:
    key = ("se", lo, hi)
    return _intern(key, lambda: UECode(key, se_legal(lo, hi), "se"))


def me(intra: bool) -> Code:
    table = tuple(T.GOLOMB_TO_INTRA_CBP if intra else T.GOLOMB_TO_INTER_CBP)
    key = ("me", intra)
    return _intern(key, lambda: UECode(key, iv(0, 47), table))


def te(xmax: int, vmax: int | None = None) -> Code:
    """te(v) with range cMax = ``xmax``; legal values restricted to 0..vmax."""
    vmax = xmax if vmax is None else min(vmax, xmax)
    if xmax == 1:
        key = ("te1", vmax)
        return _intern(key, lambda: InvBitCode(key, only_zero=vmax == 0))
    return ue_range(0, vmax)


def vlc(label: str, allowed=None) -> Code:
    allowed = None if allowed is None else frozenset(allowed)
    key = ("vlc", label, allowed)
    return _intern(key, lambda: VLCCode(key, label, allowed))


def unary(maxz: int) -> Code:
    key = ("unary", maxz)
    return _intern(key, lambda: UnaryCode(key, maxz))


def cache_stats() -> dict:
    return {
        "codes": len(_CODES),
        "step_entries": sum(len(c._steps) for c in _CODES.values()),
    }
