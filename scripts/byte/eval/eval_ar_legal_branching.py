#!/usr/bin/env python3
"""Controlled legal-branch evaluation for AR H.264 byte models.

This is the explanation experiment paired with ``eval_ar_continuation.py``.
Starting from a ground-truth stream, it stops at a complete H.264 syntax decision,
forces a different value that is legal in the same parser state, and then lets the
model continue without constraints. Every checkpoint receives the same probes.

The central measurement is whether the model completes the current NAL after the
same legal branch. Longer-horizon measurements report the next NAL, next frame
boundary, and a fixed free-run horizon. Ground-truth equality is intentionally not
used after the branch because a variable-length alternative changes all later bit
positions.

Supported branch types:
  mb_type, sub_mb_type, coded_block_pattern, coeff_token, mvd_l0, mb_qp_delta

Optional ``--coverage-source LABEL=MANIFEST`` inputs count how often each forced
syntax transition occurs naturally in a training corpus and across how many videos.
Use ``--coverage-train-split LABEL=SPLIT.json`` to restrict a source to its exact
training videos.

Example:
    python scripts/byte/eval/eval_ar_legal_branching.py DATA/manifest.jsonl \
        --nal-index-path DATA/nal_index.sqlite \
        --checkpoint-dirs PHASE2/step-XXXX PHASE3/step-YYYY \
        --checkpoint-labels phase2 phase3 \
        --out-dir OUT/legal_branching \
        --prefix-frames 8 --cont-frames 4 --num-clips 100 \
        --probes-per-element 40 --horizon-frames 1 \
        --coverage-source phase2=DATA/phase2_manifest.jsonl \
        --coverage-source phase3=DATA/phase3_manifest.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from litgpt.byte import h264_encode as HE  # noqa: E402
from litgpt.byte import h264_mask as HM  # noqa: E402
from litgpt.byte import h264_syntax as HS  # noqa: E402
from litgpt.byte.data import (  # noqa: E402
    VCL_NAL_TYPES,
    default_nal_index_path,
    load_manifest_rows,
    load_nal_index,
)
from litgpt.byte.free_run_eval import free_run_rollout  # noqa: E402
from scripts.byte.eval.eval_ar_continuation import (  # noqa: E402
    _concat_nals,
    _leading_param_bytes,
    model_max_gen,
    select_continuation_clips,
)
from scripts.byte.eval.helpers.checkpoint_eval_helpers import (  # noqa: E402
    jsonable,
    load_model,
)
from scripts.byte.eval.helpers.free_run_rescue_helpers import (  # noqa: E402
    _init_offset,
    _prompt_from_stream,
)


_INDEX_RE = re.compile(r"\[\d+\]")
_DEFAULT_ELEMENTS = (
    "mb_type",
    "sub_mb_type",
    "coded_block_pattern",
    "coeff_token",
    "mvd_l0",
    "mb_qp_delta",
)


@dataclass(frozen=True)
class BranchProbe:
    probe_id: int
    clip_index: int
    h264_path: str
    element_group: str
    element_name: str
    category: str
    mb_addr: int | None
    branch_byte: int
    branch_bit: int
    bit_offset: int
    gt_value: Any
    forced_value: Any
    gt_codeword: tuple[int, ...]
    forced_codeword: tuple[int, ...]
    forced_bits: tuple[int, ...]
    analysis_prefix: bytes
    model_prompt: bytes
    syntax_state: dict[str, Any]
    transition_signature: dict[str, Any]
    coverage: dict[str, dict[str, int]]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("manifest", type=Path)
    p.add_argument("--nal-index-path", type=Path, default=None)
    p.add_argument("--checkpoint-dirs", type=Path, nargs="+", required=True)
    p.add_argument(
        "--checkpoint-labels",
        nargs="+",
        default=None,
        help="Optional unique labels aligned with --checkpoint-dirs, e.g. phase2 phase3.",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-manifest-rows", type=int, default=0)
    p.add_argument("--train-split-file", type=Path, default=None)
    p.add_argument("--num-clips", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prefix-frames", type=int, default=8)
    p.add_argument("--cont-frames", type=int, default=4)
    p.add_argument("--max-window-bytes", type=int, default=16384)
    p.add_argument(
        "--elements",
        default=",".join(_DEFAULT_ELEMENTS),
        help="Comma-separated controlled branch groups.",
    )
    p.add_argument(
        "--probes-per-element",
        type=int,
        default=40,
        help="Maximum shared probes for each syntax-element group.",
    )
    p.add_argument(
        "--horizon-frames",
        type=int,
        default=1,
        help="Free-run horizon after the branch, measured by generated frame boundaries.",
    )
    p.add_argument("--max-gen-bytes", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy.")
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=0.0)
    p.add_argument(
        "--no-gt-control",
        dest="gt_control",
        action="store_false",
        help="Skip the matched rollout that forces the original GT value at each branch.",
    )
    p.set_defaults(gt_control=True)
    p.add_argument(
        "--exclude-param-sets",
        action="store_true",
        help="Drop leading SPS/PPS/SEI from the model prompt, matching IDR-start training. "
        "They remain in the analysis stream used by the parser and syntax mask.",
    )
    p.add_argument("--ffmpeg-binary", default="ffmpeg")
    p.add_argument("--timeout-sec", type=int, default=60)
    p.add_argument(
        "--skip-ffmpeg",
        action="store_true",
        help="Skip independent strict decoding; parser measurements still run.",
    )
    p.add_argument(
        "--coverage-source",
        action="append",
        default=[],
        metavar="LABEL=MANIFEST",
        help="Optional corpus used to count forced transition support; repeatable.",
    )
    p.add_argument(
        "--coverage-train-split",
        action="append",
        default=[],
        metavar="LABEL=SPLIT.json",
        help="Optional exact train split for a matching coverage source.",
    )
    p.add_argument("--coverage-max-rows", type=int, default=0)
    return p.parse_args()


def _parse_named_paths(items: list[str], option: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"{option} expects LABEL=PATH, got {item!r}")
        label, raw_path = item.split("=", 1)
        if not label or not raw_path:
            raise SystemExit(f"{option} expects LABEL=PATH, got {item!r}")
        if label in out:
            raise SystemExit(f"duplicate {option} label: {label}")
        out[label] = Path(raw_path)
    return out


def _normal_name(name: str) -> str:
    return _INDEX_RE.sub("", name)


def _element_group(span: HS.SyntaxSpan) -> str | None:
    name = _normal_name(span.name)
    if span.name == "mb_type":
        return "mb_type"
    if name == "sub_mb_type":
        return "sub_mb_type"
    if span.name == "coded_block_pattern":
        return "coded_block_pattern"
    if name.endswith(".coeff_token"):
        return "coeff_token"
    if name in ("mvd_l0.x", "mvd_l0.y"):
        return "mvd_l0"
    if span.name == "mb_qp_delta":
        return "mb_qp_delta"
    return None


def _max_coeff(name: str) -> int:
    if name.startswith("chroma_dc["):
        return 4
    if name.startswith("luma_ac[") or name.startswith("chroma_ac["):
        return 15
    return 16


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in sorted(value.items())}
    if isinstance(value, tuple):
        return [_json_value(v) for v in value]
    if isinstance(value, list):
        return [_json_value(v) for v in value]
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return str(value)


def _field_value(span: HS.SyntaxSpan) -> Any:
    if _element_group(span) == "coeff_token" and isinstance(span.value, dict):
        return {
            "total_coeff": int(span.value["total_coeff"]),
            "trailing_ones": int(span.value["trailing_ones"]),
        }
    return _json_value(span.value)


def _prior_value(nal: HS.NALParse, span: HS.SyntaxSpan, name: str) -> Any:
    found = None
    for prior in nal.spans:
        if prior.bit_start >= span.bit_start:
            break
        if prior.name == name and (
            span.mb_addr is None or prior.mb_addr is None or prior.mb_addr == span.mb_addr
        ):
            found = prior.value
    return found


def _syntax_state(nal: HS.NALParse, span: HS.SyntaxSpan) -> dict[str, Any]:
    state: dict[str, Any] = {
        "element": _normal_name(span.name),
        "category": span.category.value,
        "nal_type": nal.nal.nal_type,
        "slice_type": _prior_value(nal, span, "slice_type"),
    }
    mb_type = _prior_value(nal, span, "mb_type")
    if mb_type is not None and span.name != "mb_type":
        state["mb_type"] = mb_type
    if _element_group(span) == "coeff_token" and isinstance(span.value, dict):
        state["nC"] = int(span.value["nC"])
        state["max_coeff"] = _max_coeff(span.name)
    return state


def _transition_signature(nal: HS.NALParse, span: HS.SyntaxSpan, value: Any) -> dict[str, Any]:
    return {"state": _syntax_state(nal, span), "value": _json_value(value)}


def _signature_key(signature: dict[str, Any]) -> str:
    return json.dumps(signature, sort_keys=True, separators=(",", ":"))


def _slice_type(nal: HS.NALParse) -> int | None:
    for span in nal.spans:
        if span.name == "slice_type" and isinstance(span.value, int):
            return span.value % 5
    return None


def _alternative_codewords(
    nal: HS.NALParse, span: HS.SyntaxSpan
) -> list[tuple[Any, list[int]]]:
    group = _element_group(span)
    gt = _field_value(span)
    if group == "mb_type":
        # The current project uses inter-only P slices. Avoid introducing intra syntax,
        # which would test a different model/data configuration.
        if _slice_type(nal) != HE.SLICE_TYPE_P:
            return []
        values = list(range(5))
        return [(v, HE.encode_ue(v)) for v in values if v != gt]
    if group == "sub_mb_type":
        return [(v, HE.encode_ue(v)) for v in HE.legal_sub_mb_type() if v != gt]
    if group == "coded_block_pattern":
        return [(v, HE.encode_ue(v)) for v in range(48) if v != gt]
    if group == "coeff_token" and isinstance(span.value, dict):
        nc = int(span.value["nC"])
        max_coeff = _max_coeff(span.name)
        out = []
        for (total_coeff, trailing_ones), bits in HE.legal_coeff_token(nc):
            value = {
                "total_coeff": int(total_coeff),
                "trailing_ones": int(trailing_ones),
            }
            if total_coeff <= max_coeff and value != gt:
                out.append((value, bits))
        return out
    if group in ("mvd_l0", "mb_qp_delta") and isinstance(gt, int):
        values = [0, 1, -1, gt + 1, gt - 1]
        unique = []
        for value in values:
            if value != gt and value not in unique:
                unique.append(value)
        return [(v, HE.encode_se(v)) for v in unique]
    return []


def _codeword_for_gt(nal: HS.NALParse, span: HS.SyntaxSpan) -> list[int]:
    group = _element_group(span)
    value = _field_value(span)
    if group in ("mb_type", "sub_mb_type", "coded_block_pattern"):
        return HE.encode_ue(int(value))
    if group in ("mvd_l0", "mb_qp_delta"):
        return HE.encode_se(int(value))
    if group == "coeff_token" and isinstance(span.value, dict):
        wanted = (int(span.value["total_coeff"]), int(span.value["trailing_ones"]))
        for decoded, bits in HE.legal_coeff_token(int(span.value["nC"])):
            if decoded == wanted:
                return list(bits)
    raise ValueError(f"cannot encode GT value for {span.name}: {span.value!r}")


def _probe_public(probe: BranchProbe) -> dict[str, Any]:
    return {
        "probe_id": probe.probe_id,
        "clip_index": probe.clip_index,
        "h264_path": probe.h264_path,
        "element_group": probe.element_group,
        "element_name": probe.element_name,
        "category": probe.category,
        "mb_addr": probe.mb_addr,
        "branch_byte": probe.branch_byte,
        "branch_bit": probe.branch_bit,
        "bit_offset": probe.bit_offset,
        "gt_value": probe.gt_value,
        "forced_value": probe.forced_value,
        "gt_codeword": "".join(map(str, probe.gt_codeword)),
        "forced_codeword": "".join(map(str, probe.forced_codeword)),
        "forced_bit_count": len(probe.forced_bits),
        "syntax_state": probe.syntax_state,
        "transition_signature": probe.transition_signature,
        "coverage": probe.coverage,
    }


def build_probes(clips, nal_index, args) -> list[BranchProbe]:
    enabled = {x.strip() for x in args.elements.split(",") if x.strip()}
    unknown = enabled - set(_DEFAULT_ELEMENTS)
    if unknown:
        raise SystemExit(f"unsupported --elements: {sorted(unknown)}")
    rng = random.Random(args.seed)
    candidates: dict[str, list[dict[str, Any]]] = {name: [] for name in enabled}

    for clip_index, clip in enumerate(clips):
        data = clip.h264_path.read_bytes()
        source_nals = nal_index[str(clip.h264_path)]
        prefix = _concat_nals(data, source_nals, 0, clip.prefix_end_nal)
        stream = _concat_nals(data, source_nals, 0, clip.cont_end_nal)
        _, ps_len = _leading_param_bytes(source_nals, clip.cont_end_nal)
        parsed = HS.parse_stream(stream, parse_slice_data=True)
        for nal in parsed.nals:
            if nal.status != HS.ParseStatus.OK or nal.nal.nal_type not in VCL_NAL_TYPES:
                continue
            for span in nal.spans:
                group = _element_group(span)
                if group not in enabled or span.byte_start < len(prefix):
                    continue
                alternatives = _alternative_codewords(nal, span)
                if not alternatives:
                    continue
                forced_value, codeword = rng.choice(alternatives)
                gt_codeword = _codeword_for_gt(nal, span)
                branch_byte = span.byte_start
                if args.exclude_param_sets and branch_byte <= ps_len:
                    continue
                bit_offset = span.bit_start & 7
                fixed_high = HE.read_bits_int(stream, branch_byte * 8, bit_offset)
                forced_bits = HE.int_to_bits(fixed_high, bit_offset) + list(codeword)
                model_start = ps_len if args.exclude_param_sets else 0
                candidates[group].append(
                    {
                        "clip_index": clip_index,
                        "h264_path": str(clip.h264_path),
                        "element_group": group,
                        "element_name": span.name,
                        "category": span.category.value,
                        "mb_addr": span.mb_addr,
                        "branch_byte": branch_byte,
                        "branch_bit": span.bit_start,
                        "bit_offset": bit_offset,
                        "gt_value": _field_value(span),
                        "forced_value": _json_value(forced_value),
                        "gt_codeword": tuple(gt_codeword),
                        "forced_codeword": tuple(codeword),
                        "forced_bits": tuple(forced_bits),
                        "analysis_prefix": stream[:branch_byte],
                        "model_prompt": stream[model_start:branch_byte],
                        "syntax_state": _syntax_state(nal, span),
                        "transition_signature": _transition_signature(nal, span, forced_value),
                    }
                )

    selected: list[dict[str, Any]] = []
    for group in sorted(candidates):
        # Prevent coefficient-heavy clips from dominating a group: sample at most one
        # occurrence of each element group per clip, then sample clips uniformly.
        by_clip: dict[int, list[dict[str, Any]]] = {}
        for candidate in candidates[group]:
            by_clip.setdefault(candidate["clip_index"], []).append(candidate)
        clip_candidates = [rng.choice(items) for items in by_clip.values()]
        rng.shuffle(clip_candidates)
        selected.extend(clip_candidates[: args.probes_per_element])
    selected.sort(key=lambda x: (x["clip_index"], x["branch_byte"], x["element_name"]))
    return [
        BranchProbe(probe_id=i, coverage={}, **candidate)
        for i, candidate in enumerate(selected)
    ]


def _filter_train_rows(rows: list[dict[str, Any]], split_path: Path | None):
    if split_path is None:
        return rows
    split = json.loads(split_path.read_text(encoding="utf-8"))
    videos = {str(Path(v)) for v in split.get("videos", [])}
    return [row for row in rows if str(Path(row["h264_path"])) in videos]


def add_coverage(
    probes: list[BranchProbe], sources: dict[str, Path], splits: dict[str, Path], args
) -> list[BranchProbe]:
    if not sources or not probes:
        return probes
    target_keys = {_signature_key(p.transition_signature) for p in probes}
    coverage_by_label: dict[str, dict[str, dict[str, Any]]] = {}
    for label, manifest in sources.items():
        rows = load_manifest_rows(
            manifest, max_rows=args.coverage_max_rows or None, report_progress=True
        )
        rows = _filter_train_rows(rows, splits.get(label))
        counts: Counter = Counter()
        videos: dict[str, set[str]] = {key: set() for key in target_keys}
        seen_paths: set[str] = set()
        for row in rows:
            path = str(Path(row["h264_path"]))
            if path in seen_paths:
                continue
            seen_paths.add(path)
            parsed = HS.parse_stream(Path(path).read_bytes(), parse_slice_data=True)
            for nal in parsed.nals:
                if nal.status != HS.ParseStatus.OK or nal.nal.nal_type not in VCL_NAL_TYPES:
                    continue
                for span in nal.spans:
                    if _element_group(span) is None:
                        continue
                    key = _signature_key(_transition_signature(nal, span, _field_value(span)))
                    if key in target_keys:
                        counts[key] += 1
                        videos[key].add(path)
        coverage_by_label[label] = {
            key: {"occurrences": counts[key], "videos": len(videos[key])}
            for key in target_keys
        }

    updated = []
    for probe in probes:
        key = _signature_key(probe.transition_signature)
        cov = {label: table[key] for label, table in coverage_by_label.items()}
        updated.append(
            BranchProbe(**{**probe.__dict__, "coverage": cov})
        )
    return updated


def _find_branch_nal(parsed: HS.StreamParse, branch_byte: int) -> int | None:
    found = None
    for i, nal in enumerate(parsed.nals):
        if nal.nal.start_code_start <= branch_byte < nal.nal.payload_end:
            found = i
    return found


def _value_matches(actual: Any, expected: Any, group: str) -> bool:
    if group == "coeff_token" and isinstance(actual, dict):
        actual = {
            "total_coeff": actual.get("total_coeff"),
            "trailing_ones": actual.get("trailing_ones"),
        }
    return _json_value(actual) == _json_value(expected)


def _forced_value_verified(
    nal: HS.NALParse, probe: BranchProbe
) -> bool:
    for span in nal.spans:
        if span.bit_start != probe.branch_bit:
            continue
        if _normal_name(span.name) != _normal_name(probe.element_name):
            continue
        return _value_matches(span.value, probe.forced_value, probe.element_group)
    return False


def _gt_control_probe(probe: BranchProbe) -> BranchProbe:
    fixed_high = probe.forced_bits[: probe.bit_offset]
    return replace(
        probe,
        forced_value=probe.gt_value,
        forced_codeword=probe.gt_codeword,
        forced_bits=tuple(fixed_high) + probe.gt_codeword,
        transition_signature={"state": probe.syntax_state, "value": probe.gt_value},
        coverage={},
    )


def _closed_nal_count(parsed: HS.StreamParse, start: int) -> int:
    count = 0
    # A later NAL start proves that the current NAL was closed by a boundary.
    for i in range(start, max(start, len(parsed.nals) - 1)):
        if parsed.nals[i].status != HS.ParseStatus.OK:
            break
        count += 1
    return count


def _clean_future_frame_boundaries(
    stream: bytes, parsed: HS.StreamParse, start: int
) -> int:
    count = 0
    for nal in parsed.nals[start:]:
        if nal.status != HS.ParseStatus.OK:
            break
        if nal.nal.nal_type not in VCL_NAL_TYPES:
            continue
        raw = stream[nal.nal.payload_start : nal.nal.payload_end]
        if HS.slice_first_mb(raw) == 0:
            count += 1
    return count


def _first_failure(parsed: HS.StreamParse, branch_byte: int) -> dict[str, Any] | None:
    for nal_index, nal in enumerate(parsed.nals):
        if nal.status == HS.ParseStatus.OK:
            continue
        failure_byte = nal.desync_byte
        if failure_byte is None:
            failure_byte = nal.nal.start_code_start
        if failure_byte < branch_byte:
            continue
        return {
            "nal_index": nal_index,
            "byte": failure_byte,
            "element": nal.failure_element,
            "reason": nal.reason_kind,
            "context": _json_value(nal.failure_context),
        }
    return None


def _mask_audit(prefix: bytes, generated: bytes) -> dict[str, Any]:
    state = HM.MaskState()
    try:
        for byte in prefix:
            HM.advance(state, byte)
        for step, byte in enumerate(generated):
            allowed = HM.get_valid_byte_mask(state)
            if not any(allowed):
                return {
                    "checked_bytes": step,
                    "would_intervene": True,
                    "first_intervention_step": step,
                    "reason": "no_allowed_byte",
                }
            if not allowed[byte]:
                return {
                    "checked_bytes": step + 1,
                    "would_intervene": True,
                    "first_intervention_step": step,
                    "reason": "generated_byte_masked",
                }
            HM.advance(state, byte)
        return {
            "checked_bytes": len(generated),
            "would_intervene": False,
            "first_intervention_step": None,
            "reason": None,
        }
    except Exception as exc:
        return {
            "checked_bytes": 0,
            "would_intervene": None,
            "first_intervention_step": None,
            "reason": f"audit_error:{type(exc).__name__}:{exc}",
        }


def _ffmpeg_accepts(stream: bytes, args) -> tuple[bool | None, str | None]:
    if args.skip_ffmpeg:
        return None, None
    command = [
        args.ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ec",
        "0",
        "-err_detect",
        "explode+bitstream+buffer+compliant",
        "-f",
        "h264",
        "-i",
        "pipe:0",
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(
            command,
            input=stream,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=args.timeout_sec,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"{type(exc).__name__}:{exc}"
    error = result.stderr.decode("utf-8", errors="replace").strip()
    return result.returncode == 0, error or None


@torch.inference_mode()
def evaluate_probe(raw, probe: BranchProbe, device: torch.device, args) -> dict[str, Any]:
    max_gen = min(args.max_gen_bytes, model_max_gen(raw, probe.model_prompt))
    base = _probe_public(probe)
    if max_gen <= 0:
        return {**base, "status": "skipped_no_budget"}
    prompt_ids, region_ids, offset_ids = _prompt_from_stream(probe.model_prompt, device)
    trace: dict[str, Any] = {}
    generated, _ = free_run_rollout(
        raw,
        prompt_ids,
        region_ids,
        offset_ids,
        device,
        args.horizon_frames,
        max_gen,
        args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        forced_bits=list(probe.forced_bits),
        init_offset=_init_offset(probe.model_prompt),
        trace=trace,
    )
    stream = probe.analysis_prefix + generated
    parsed = HS.parse_stream(stream, parse_slice_data=True)
    branch_nal_index = _find_branch_nal(parsed, probe.branch_byte)
    if branch_nal_index is None:
        return {
            **base,
            "status": "branch_nal_missing",
            "generated_bytes": len(generated),
            "stop_reason": trace.get("stop_reason"),
        }
    branch_nal = parsed.nals[branch_nal_index]
    verified = _forced_value_verified(branch_nal, probe)
    closed_nals = _closed_nal_count(parsed, branch_nal_index)
    frame_boundaries = _clean_future_frame_boundaries(stream, parsed, branch_nal_index + 1)
    failure = _first_failure(parsed, probe.branch_byte)
    completed_bytes = (
        max(0, int(failure["byte"]) - probe.branch_byte)
        if failure is not None
        else len(stream) - probe.branch_byte
    )
    mask = _mask_audit(probe.analysis_prefix, generated)
    ffmpeg_ok, ffmpeg_error = _ffmpeg_accepts(stream, args)
    current_nal_success = bool(
        verified and branch_nal.status == HS.ParseStatus.OK and closed_nals >= 1
    )
    next_nal_success = bool(verified and closed_nals >= 2)
    horizon_success = bool(
        verified
        and failure is None
        and trace.get("stop_reason") == "frame_target"
        and frame_boundaries >= args.horizon_frames
    )
    return {
        **base,
        "status": "ok",
        "forced_value_verified": verified,
        "current_nal_success": current_nal_success,
        "next_nal_success": next_nal_success,
        "reached_next_frame": bool(verified and frame_boundaries >= 1),
        "horizon_success": horizon_success,
        "completed_nals_after_branch": closed_nals if verified else None,
        "completed_frame_boundaries_after_branch": frame_boundaries if verified else None,
        "completed_bytes_after_branch": completed_bytes if verified else None,
        "generated_bytes": len(generated),
        "stop_reason": trace.get("stop_reason"),
        "first_failure": failure,
        "mask_audit": mask,
        "ffmpeg_accepts": ffmpeg_ok,
        "ffmpeg_error": ffmpeg_error,
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    n = len(values)
    if n & 1:
        return float(values[n // 2])
    return 0.5 * (values[n // 2 - 1] + values[n // 2])


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [bool(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [row for row in rows if row.get("status") == "ok"]
    verified = [row for row in evaluated if row.get("forced_value_verified")]
    completed_bytes = [float(row["completed_bytes_after_branch"]) for row in verified]
    completed_nals = [float(row["completed_nals_after_branch"]) for row in verified]
    failures = [row.get("first_failure") for row in verified if row.get("first_failure")]
    mask_rows = [
        row for row in verified
        if (row.get("mask_audit") or {}).get("would_intervene") is not None
    ]
    return {
        "num_probes": len(rows),
        "num_evaluated": len(evaluated),
        "num_forced_value_verified": len(verified),
        "forced_value_verified_rate": (
            len(verified) / len(evaluated) if evaluated else None
        ),
        "current_nal_success_rate": _rate(verified, "current_nal_success"),
        "next_nal_success_rate": _rate(verified, "next_nal_success"),
        "reached_next_frame_rate": _rate(verified, "reached_next_frame"),
        "horizon_success_rate": _rate(verified, "horizon_success"),
        "completed_bytes_after_branch_mean": _mean(completed_bytes),
        "completed_bytes_after_branch_median": _median(completed_bytes),
        "completed_nals_after_branch_mean": _mean(completed_nals),
        "completed_nals_after_branch_median": _median(completed_nals),
        "syntax_mask_would_intervene_rate": (
            sum(bool(row["mask_audit"]["would_intervene"]) for row in mask_rows)
            / len(mask_rows)
            if mask_rows else None
        ),
        "ffmpeg_accept_rate": _rate(verified, "ffmpeg_accepts"),
        "failure_element_hist": dict(
            Counter((failure.get("element") or "unknown") for failure in failures).most_common()
        ),
        "failure_reason_hist": dict(
            Counter((failure.get("reason") or "unknown") for failure in failures).most_common()
        ),
    }


def summarize(checkpoint: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {"checkpoint": checkpoint, **_summarize_rows(rows)}
    groups = sorted({row["element_group"] for row in rows})
    summary["by_element"] = {
        group: _summarize_rows([row for row in rows if row["element_group"] == group])
        for group in groups
    }
    coverage_labels = sorted(
        {label for row in rows for label in row.get("coverage", {})}
    )
    summary["by_coverage"] = {}
    for label in coverage_labels:
        buckets: dict[str, list[dict[str, Any]]] = {
            "unseen": [],
            "rare_1_to_15": [],
            "common_16_plus": [],
        }
        for row in rows:
            occurrences = int((row.get("coverage", {}).get(label) or {}).get("occurrences", 0))
            bucket = (
                "unseen"
                if occurrences == 0
                else "rare_1_to_15"
                if occurrences < 16
                else "common_16_plus"
            )
            buckets[bucket].append(row)
        summary["by_coverage"][label] = {
            bucket: _summarize_rows(bucket_rows)
            for bucket, bucket_rows in buckets.items()
        }
    return summary


def paired_comparisons(
    details_by_checkpoint: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    names = list(details_by_checkpoint)
    if len(names) < 2:
        return []
    base_name = names[0]
    base = {row["probe_id"]: row for row in details_by_checkpoint[base_name]}
    out = []
    for name in names[1:]:
        other = {row["probe_id"]: row for row in details_by_checkpoint[name]}
        pairs = [
            (base[probe_id], other[probe_id])
            for probe_id in sorted(base.keys() & other.keys())
            if base[probe_id].get("forced_value_verified")
            and other[probe_id].get("forced_value_verified")
        ]
        byte_deltas = [
            float(b["completed_bytes_after_branch"] - a["completed_bytes_after_branch"])
            for a, b in pairs
        ]
        out.append(
            {
                "baseline": base_name,
                "comparison": name,
                "paired_probes": len(pairs),
                "current_nal_success_delta": (
                    _rate([b for _, b in pairs], "current_nal_success") or 0.0
                ) - (_rate([a for a, _ in pairs], "current_nal_success") or 0.0)
                if pairs else None,
                "horizon_success_delta": (
                    _rate([b for _, b in pairs], "horizon_success") or 0.0
                ) - (_rate([a for a, _ in pairs], "horizon_success") or 0.0)
                if pairs else None,
                "completed_bytes_delta_mean": _mean(byte_deltas),
                "comparison_survives_longer_rate": (
                    sum(delta > 0 for delta in byte_deltas) / len(byte_deltas)
                    if byte_deltas else None
                ),
            }
        )
    return out


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), sort_keys=True) + "\n")


def _write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    flat = []
    for summary in summaries:
        row = {
            k: v for k, v in summary.items()
            if k not in ("by_element", "by_coverage")
        }
        row["by_element"] = json.dumps(summary.get("by_element", {}), sort_keys=True)
        row["by_coverage"] = json.dumps(summary.get("by_coverage", {}), sort_keys=True)
        flat.append(row)
    fields = list(dict.fromkeys(key for row in flat for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)


def main() -> None:
    args = parse_args()
    if args.prefix_frames < 1 or args.cont_frames < 1 or args.horizon_frames < 1:
        raise SystemExit("frame counts must be positive")
    if args.probes_per_element < 1 or args.max_gen_bytes < 1:
        raise SystemExit("--probes-per-element and --max-gen-bytes must be positive")
    if args.checkpoint_labels is not None:
        if len(args.checkpoint_labels) != len(args.checkpoint_dirs):
            raise SystemExit("--checkpoint-labels must align one-for-one with --checkpoint-dirs")
        checkpoint_labels = list(args.checkpoint_labels)
    else:
        checkpoint_labels = [path.name for path in args.checkpoint_dirs]
    if len(set(checkpoint_labels)) != len(checkpoint_labels):
        raise SystemExit(
            "checkpoint labels are not unique; pass --checkpoint-labels, e.g. phase2 phase3"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )

    rows = load_manifest_rows(
        args.manifest, max_rows=args.max_manifest_rows or None, report_progress=True
    )
    rows = _filter_train_rows(rows, args.train_split_file)
    index_path = args.nal_index_path or default_nal_index_path(args.manifest)
    nal_index = load_nal_index(index_path, args.manifest, rows)
    clips = select_continuation_clips(rows, nal_index, args)
    if not clips:
        raise SystemExit("no continuation clips satisfy the requested frame/byte window")
    probes = build_probes(clips, nal_index, args)
    if not probes:
        raise SystemExit("no supported legal branch probes were found")

    coverage_sources = _parse_named_paths(args.coverage_source, "--coverage-source")
    coverage_splits = _parse_named_paths(
        args.coverage_train_split, "--coverage-train-split"
    )
    unknown_splits = coverage_splits.keys() - coverage_sources.keys()
    if unknown_splits:
        raise SystemExit(f"coverage split has no matching source: {sorted(unknown_splits)}")
    probes = add_coverage(probes, coverage_sources, coverage_splits, args)

    (args.out_dir / "config.json").write_text(
        json.dumps(jsonable(vars(args)), indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "probes.json").write_text(
        json.dumps([_probe_public(probe) for probe in probes], indent=2) + "\n",
        encoding="utf-8",
    )

    details_by_checkpoint: dict[str, list[dict[str, Any]]] = {}
    summaries = []
    for checkpoint_dir, name in zip(args.checkpoint_dirs, checkpoint_labels):
        print(f"Loading checkpoint: {checkpoint_dir}", flush=True)
        model = load_model(checkpoint_dir, device)
        raw = model.module if hasattr(model, "module") else model
        raw.eval()
        details = []
        for i, probe in enumerate(probes):
            print(f"  {name}: probe {i + 1}/{len(probes)}", flush=True)
            row = evaluate_probe(raw, probe, device, args)
            row["checkpoint"] = name
            details.append(row)
        del raw, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        details_by_checkpoint[name] = details
        summary = summarize(name, details)
        summaries.append(summary)
        _append_jsonl(args.out_dir / "branch_details.jsonl", details)
        _append_jsonl(args.out_dir / "metrics.jsonl", [summary])
        print(json.dumps(summary, indent=2), flush=True)

    comparisons = paired_comparisons(details_by_checkpoint)
    (args.out_dir / "paired_comparisons.json").write_text(
        json.dumps(comparisons, indent=2) + "\n", encoding="utf-8"
    )
    _write_summary_csv(args.out_dir / "summary.csv", summaries)
    print(f"Controlled legal-branch evaluation complete: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
