"""Step 3 -- Annex-B framing and the RBSP -> EBSP emulation-prevention filter.

``MaskStream`` is the public streaming interface::

    ms = MaskStream(Profile(slice_layout="frame"))
    for byte in data:
        m = ms.mask()        # 256-bit int, bit b set iff emitted byte b is legal
        ms.advance(byte)     # raises Illegal on a grammar violation

Framing states:

* start code: ``00`` until two zeros were seen, then ``00`` or ``01``
  (leading/trailing zero bytes are allowed; the NAL starts after ``01``);
* NAL payload: the grammar's RBSP mask filtered by constant-time EBSP rules:
  after two emitted zeros ``00..02`` cannot be emitted, ``03`` is legal iff some
  RBSP byte ``00..03`` is legal (it is an emulation-prevention byte and does not
  advance the parser), and the byte following an EPB must be ``00..03``.
* after the RBSP ends (rbsp_trailing_bits / empty payload) the next byte opens
  the next start code.
"""

from __future__ import annotations

from .compile import rbsp_mask, reference_mask
from .grammar import Ctx, Illegal, NAL_PPS, NAL_SPS, VCL, Parser, Profile

LOW4 = 0xF
START_ZERO = 1 << 0x00
START_ONE = 1 << 0x01


def mask_to_list(m: int) -> list[bool]:
    return [bool((m >> b) & 1) for b in range(256)]


def mask_to_bytes(m: int) -> bytes:
    """32-byte little-endian packing (bit b of the int = byte value b)."""
    return m.to_bytes(32, "little")


class MaskStream:
    def __init__(self, profile: Profile | None = None):
        self.ctx = Ctx(profile or Profile())
        self.p = None  # Parser of the open NAL, None while in a start code
        self.zeros = 0  # start-code zeros seen
        self.zr = 0  # consecutive zero bytes emitted inside the payload
        self.epb = False  # previous emitted byte was an emulation-prevention 03
        self.nal_count = 0
        self.offset = 0

    # ---- query -------------------------------------------------------------
    def where(self) -> str:
        if self.p is None:
            return "start_code"
        if self.epb:
            return "after_epb:" + self.p.field
        return self.p.field

    def _filter(self, r: int) -> int:
        if self.epb:
            return r & LOW4
        if self.zr >= 2:
            e = r & ~LOW4
            if r & LOW4:
                e |= 1 << 3
            return e
        return r

    def mask(self) -> int:
        if self.p is None:
            return START_ZERO | (START_ONE if self.zeros >= 2 else 0)
        return self._filter(rbsp_mask(self.p))

    def mask_reference(self) -> int:
        """Same as ``mask`` but with the bit-by-bit Step 2 (verification)."""
        if self.p is None:
            return self.mask()
        return self._filter(reference_mask(self.p))

    # ---- commit ------------------------------------------------------------
    def advance(self, byte: int) -> None:
        self.offset += 1
        p = self.p
        if p is None:
            if byte == 0:
                self.zeros += 1
            elif byte == 1 and self.zeros >= 2:
                self.p = Parser(self.ctx)
                self.zr = 0
                self.epb = False
            else:
                raise Illegal(f"byte 0x{byte:02x} in start code")
            return

        if self.epb:
            if byte > 3:
                raise Illegal(f"byte 0x{byte:02x} after emulation prevention")
            self.epb = False
        elif self.zr >= 2:
            if byte == 3:
                self.epb = True
                self.zr = 0
                return
            if byte < 3:
                raise Illegal(f"byte 0x{byte:02x} after 00 00 in payload")
        p.feed_byte(byte)
        self.zr = self.zr + 1 if byte == 0 else 0
        if p.done:
            self._commit(p)
            self.p = None
            self.zeros = 0

    def _commit(self, p: Parser) -> None:
        self.nal_count += 1
        t = p.nal_type
        if t == NAL_SPS:
            self.ctx.commit_sps(p.rec)
        elif t == NAL_PPS:
            self.ctx.pps[p.rec["id"]] = p.rec
        elif t in VCL:
            self.ctx.commit_slice(p.rec)


def iter_masks(data: bytes, profile: Profile | None = None):
    """Yield ``(offset, mask_int, where)`` for every byte of a GT stream,
    checking that the true byte is legal before committing it."""
    ms = MaskStream(profile)
    for i, byte in enumerate(data):
        m = ms.mask()
        where = ms.where()
        if not (m >> byte) & 1:
            raise Illegal(f"GT byte 0x{byte:02x} masked out at offset {i} ({where})")
        yield i, m, where
        ms.advance(byte)
