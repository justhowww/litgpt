"""Parser-state labels at every byte boundary of an H.264 window, for linear probes.

A byte usually holds bits of several CAVLC syntax elements, so labels describe the
parser state *between* bytes: the element the next bit belongs to, how many of its
bits are already consumed, and the macroblock it belongs to. ``labels[k]`` is the
state before on-disk byte ``k`` (equivalently, after byte ``k - 1``), for
``k = 0 .. len(window)``. Torch-free.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass

from litgpt.byte import h264_syntax as HS

# Element classes, grouped from the parser's span names / categories.
CLASSES = (
    "start_code",
    "nal_header",
    "param_set_sei",
    "slice_header",
    "mb_skip_run",
    "mb_type",
    "sub_mb_type",
    "ref_idx",
    "mvd",
    "intra_mode",
    "cbp",
    "mb_qp_delta",
    "coeff_token",
    "trailing_ones",
    "total_zeros",
    "run_before",
    "level",
    "rbsp_trailing",
    "other",
)
CLASS_ID = {name: i for i, name in enumerate(CLASSES)}
# Classes that belong to macroblock-layer data (the part content prediction is about).
MB_CLASSES = frozenset(
    CLASS_ID[c]
    for c in (
        "mb_skip_run", "mb_type", "sub_mb_type", "ref_idx", "mvd", "intra_mode",
        "cbp", "mb_qp_delta", "coeff_token", "trailing_ones", "total_zeros",
        "run_before", "level",
    )
)
MAX_BITS_INTO = 15  # bits-into-element labels are capped here

_INDEX = re.compile(r"\[\d+\]")


def element_class(span: HS.SyntaxSpan) -> int:
    """Map one parser span to a probe class id."""
    cat = span.category
    name = _INDEX.sub("", span.name)
    if cat == HS.Category.START_CODE:
        return CLASS_ID["start_code"]
    if cat == HS.Category.NAL_HEADER:
        return CLASS_ID["nal_header"]
    if cat == HS.Category.SLICE_HEADER:
        return CLASS_ID["slice_header"]
    if cat.value in ("sps", "pps", "sei"):
        return CLASS_ID["param_set_sei"]
    if cat == HS.Category.RBSP_TRAILING:
        return CLASS_ID["rbsp_trailing"]
    for key, label in (
        ("coeff_token", "coeff_token"),
        ("trailing_ones", "trailing_ones"),
        ("total_zeros", "total_zeros"),
        ("run_before", "run_before"),
        ("mvd", "mvd"),
        ("ref_idx", "ref_idx"),
        ("sub_mb_type", "sub_mb_type"),
        ("mb_skip_run", "mb_skip_run"),
        ("mb_type", "mb_type"),
        ("intra", "intra_mode"),
        ("coded_block_pattern", "cbp"),
        ("mb_qp_delta", "mb_qp_delta"),
    ):
        if key in name:
            return CLASS_ID[label]
    if name.endswith("level"):
        return CLASS_ID["level"]
    return CLASS_ID["other"]


@dataclass
class BoundaryLabels:
    """Per-boundary labels; index k = state before on-disk byte k. -1 = undefined."""

    element: list[int]
    bits_into: list[int]
    mb_x: list[int]
    mb_y: list[int]

    def in_mb_data(self, k: int) -> bool:
        return self.element[k] in MB_CLASSES


def parser_state_labels(window: bytes) -> BoundaryLabels:
    """Label the parser state at every byte boundary of ``window`` (Annex-B bytes)."""
    n = len(window) + 1
    element = [CLASS_ID["other"]] * n
    bits_into = [-1] * n
    mb_x = [-1] * n
    mb_y = [-1] * n
    parse = HS.parse_stream(window, parse_slice_data=True)
    sps = next(iter(parse.sps.values()), None)
    width = sps.pic_width_in_mbs if sps is not None else None

    for nal_parse in parse.nals:
        nal = nal_parse.nal
        rbsp_spans = []
        for span in nal_parse.spans:
            if span.bit_start < 0:  # byte-space spans: start code, NAL header, EPB
                cls = element_class(span)
                for k in range(span.byte_start, min(span.byte_end, n - 1)):
                    element[k] = cls
                    bits_into[k] = min(MAX_BITS_INTO, 8 * (k - span.byte_start))
            else:
                rbsp_spans.append(span)
        if not rbsp_spans:
            continue
        rbsp_spans.sort(key=lambda s: s.bit_start)
        starts = [s.bit_start for s in rbsp_spans]
        _, byte_map, _ = HS.unescape_rbsp(window, nal.payload_start + 1, nal.payload_end)
        for i, offset in enumerate(byte_map):
            bit = 8 * i
            j = bisect.bisect_right(starts, bit) - 1
            if j < 0 or bit >= rbsp_spans[j].bit_end:
                continue  # between elements (e.g. after a desync): leave "other"
            span = rbsp_spans[j]
            element[offset] = element_class(span)
            bits_into[offset] = min(MAX_BITS_INTO, bit - span.bit_start)
            if span.mb_addr is not None and width:
                mb_x[offset] = span.mb_addr % width
                mb_y[offset] = span.mb_addr // width
    return BoundaryLabels(element, bits_into, mb_x, mb_y)
