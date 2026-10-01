"""Parser-side pieces of the slice-end regeneration eval (no torch).

Two interchangeable syntax masks with one interface, seeded from the original
stream bytes before the cut:

  OldMask  litgpt/byte/h264_mask (bit-tree search through the automaton)
  NewMask  syntax_mask.MaskStream (precomputed transition tables)

``slice_complete()`` is true once the cut slice has emitted its last macroblock
and rbsp_trailing_bits; the next byte would open a new start code. With one
slice per frame, the bytes after the hole then begin at a start code, so any
complete legal slice reconnects and EOS is forced there.

Also: Annex-B NAL bookkeeping for frame type / frame index, and the model-free
random-legal-bytes generator used as the content floor.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from litgpt.byte import h264_mask as HM
from syntax_mask.grammar import Illegal, Profile
from syntax_mask.stream import MaskStream, mask_to_list

VCL_TYPES = (1, 5)


# ---------------------------------------------------------------------------
# Annex-B bookkeeping
# ---------------------------------------------------------------------------
def nal_units(data: bytes) -> list[tuple[int, int]]:
    """``(start, nal_type)`` per NAL; ``start`` is the first zero of its start code."""
    out = []
    i = data.find(b"\x00\x00\x01")
    while i >= 0:
        start = i - 1 if i > 0 and data[i - 1] == 0 else i
        header = i + 3
        if header >= len(data):
            break
        out.append((start, data[header] & 0x1F))
        i = data.find(b"\x00\x00\x01", header)
    return out


def frame_type_at(window: bytes, frame_lo: int) -> str:
    """Frame class of the VCL NAL starting at ``frame_lo`` (baseline: IDR or P)."""
    for start, nal_type in nal_units(window[frame_lo:]):
        if start != 0:
            break
        if nal_type == 5:
            return "idr"
        if nal_type == 1:
            return "p"
    return "other"


def frame_index_at(window: bytes, offset: int) -> int:
    """Number of VCL NALs (frames, one slice each) starting before ``offset``."""
    return sum(
        1 for start, nal_type in nal_units(window) if start < offset and nal_type in VCL_TYPES
    )


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------
class MaskRejected(Exception):
    """A committed byte was illegal under this mask."""


class NewMask:
    name = "new"

    def __init__(self, prefix: bytes) -> None:
        self.ms = MaskStream(Profile(slice_layout="frame"))
        for byte in prefix:
            self.ms.advance(byte)

    def allowed(self) -> list[bool]:
        return mask_to_list(self.ms.mask())

    def advance(self, byte: int) -> None:
        try:
            self.ms.advance(byte)
        except Illegal as exc:
            raise MaskRejected(str(exc)) from exc

    def slice_complete(self) -> bool:
        return self.ms.p is None

    def where(self) -> str:
        return self.ms.where()


class OldMask:
    name = "old"

    def __init__(self, prefix: bytes) -> None:
        self.state = HM.MaskState(
            slice_max_mbs=HM.slice_max_mbs_for_layout(HM.SLICE_LAYOUT_FRAME),
            fail_closed=True,
        )
        for byte in prefix:
            HM.advance(self.state, byte)
        self.state.generation_started = True
        self.last_strict = True

    def allowed(self) -> list[bool]:
        before = self.state.strict_mask_calls
        mask = HM.get_valid_byte_mask(self.state)
        self.last_strict = self.state.strict_mask_calls > before
        return mask

    def advance(self, byte: int) -> None:
        HM.advance(self.state, byte)
        if self.state.automaton_unknown:
            raise MaskRejected(self.state.failure_reason or "automaton_unknown")

    def slice_complete(self) -> bool:
        return HM.can_append_bytes(self.state, b"", require_complete=True)

    def where(self) -> str:
        auto = self.state.automaton
        return str(getattr(auto, "ae_tag", "unknown")) if auto is not None else "nal"


def make_mask(kind: str, prefix: bytes):
    if kind == "new":
        return NewMask(prefix)
    if kind == "old":
        return OldMask(prefix)
    raise ValueError(f"unknown mask {kind!r}")


@dataclass
class CrossCheck:
    """Step-by-step agreement of a shadow mask with the mask driving generation."""

    shadow: object | None
    steps: int = 0
    disagree_steps: int = 0
    only_driver: int = 0  # bytes the driving mask allows and the shadow rejects
    only_shadow: int = 0
    shadow_permissive_steps: int = 0  # old-mask steps outside its strict region
    first: dict | None = None
    shadow_lost: str | None = None  # shadow rejected a committed byte

    def compare(self, step: int, driver_allowed: list[bool], driver_where: str) -> None:
        if self.shadow is None:
            return
        shadow_allowed = self.shadow.allowed()
        self.steps += 1
        if isinstance(self.shadow, OldMask) and not self.shadow.last_strict:
            self.shadow_permissive_steps += 1
        a = sum(1 for d, s in zip(driver_allowed, shadow_allowed) if d and not s)
        b = sum(1 for d, s in zip(driver_allowed, shadow_allowed) if s and not d)
        if a or b:
            self.disagree_steps += 1
            self.only_driver += a
            self.only_shadow += b
            if self.first is None:
                self.first = {
                    "step": step,
                    "where": driver_where,
                    "only_driver": a,
                    "only_shadow": b,
                }

    def advance(self, byte: int) -> None:
        if self.shadow is None:
            return
        try:
            self.shadow.advance(byte)
        except MaskRejected as exc:
            self.shadow_lost = str(exc)[:200]
            self.shadow = None

    def report(self) -> dict:
        return {
            "steps": self.steps,
            "disagree_steps": self.disagree_steps,
            "only_driver": self.only_driver,
            "only_shadow": self.only_shadow,
            "shadow_permissive_steps": self.shadow_permissive_steps,
            "first": self.first,
            "shadow_lost": self.shadow_lost,
        }


# ---------------------------------------------------------------------------
# post-hoc checks and the model-free baseline
# ---------------------------------------------------------------------------
def replay_legality(prefix: bytes, generated: bytes) -> dict:
    """Run generated bytes through the new mask: legal prefix length and end state."""
    mask = NewMask(prefix)
    complete_at = None
    for i, byte in enumerate(generated):
        if mask.slice_complete() and complete_at is None:
            complete_at = i
        if not mask.allowed()[byte]:
            return {
                "legal_bytes": i,
                "all_legal": False,
                "illegal_where": mask.where(),
                "slice_complete_at": complete_at,
            }
        mask.advance(byte)
    if mask.slice_complete() and complete_at is None:
        complete_at = len(generated)
    return {
        "legal_bytes": len(generated),
        "all_legal": True,
        "illegal_where": None,
        "slice_complete_at": complete_at,
        "ends_at_slice_end": complete_at == len(generated),
    }


@dataclass
class RandomLegalResult:
    data: bytes
    stop_reason: str
    steps: int
    cross_check: dict = field(default_factory=dict)


def random_legal_completion(
    prefix: bytes,
    rng: random.Random,
    *,
    max_bytes: int,
    mask_kind: str = "new",
) -> RandomLegalResult:
    """Uniform choice among legal bytes until the slice completes (forced EOS)."""
    mask = make_mask(mask_kind, prefix)
    out = bytearray()
    while len(out) < max_bytes:
        if mask.slice_complete():
            return RandomLegalResult(bytes(out), "slice_end", len(out))
        legal = [b for b, ok in enumerate(mask.allowed()) if ok]
        if not legal:
            return RandomLegalResult(bytes(out), "mask_boxed_in", len(out))
        byte = rng.choice(legal)
        mask.advance(byte)
        out.append(byte)
    reason = "slice_end" if mask.slice_complete() else "budget"
    return RandomLegalResult(bytes(out), reason, len(out))
