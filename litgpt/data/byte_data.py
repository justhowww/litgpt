# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""Byte-domain datasets for H.264 language-model pretraining.

This data module bypasses text tokenization. Raw byte values map directly to
token ids 0..255, and a small number of control ids mark generation starts.

The training unit is a VCL NAL unit from an Annex-B H.264 stream. Since the
preprocessing config pins one slice per frame, one VCL NAL is the frame/slice
unit used by the byte model.
"""

from __future__ import annotations

import json
import random
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split

from litgpt.data import DataModule
from litgpt.tokenizer import Tokenizer

BYTE_VOCAB_SIZE = 256  # Raw byte ids are exactly 0..255.
PAD_ID = 256  # Padding id for variable-length batches; never a real byte.
SLICE_BOS_ID = 257  # Starts AR generation of the full target slice B_t.
SPAN_BOS_ID = 258  # Starts FIM generation of the missing span B_miss.
VOCAB_SIZE = 259  # Model vocab size: 256 byte ids + 3 control ids.
IGNORE_INDEX = -100  # Cross-entropy ignore label for conditioning-only positions.

REGION_REF = 0  # Previous VCL slice bytes used as reference/context.
REGION_TARGET = 1  # Target-slice bytes in AR mode.
REGION_PREFIX = 2  # Received target bytes before the missing span in FIM mode.
REGION_ORPHAN = 3  # Received target bytes after the missing span in FIM mode.
REGION_BRIDGE = 4  # Generated missing-span bytes in FIM mode.
REGION_META = 5  # Parameter-set metadata bytes, currently latest SPS/PPS before target.
REGION_PAD = 6  # Region id for padded batch positions.

VCL_NAL_TYPES = {1, 5}  # H.264 VCL slices: 1 = non-IDR P slice, 5 = IDR/I slice.
PARAMETER_SET_NAL_TYPES = {7, 8}  # H.264 metadata NALs: 7 = SPS, 8 = PPS.
NAL_INDEX_VERSION = 1
DEFAULT_NAL_INDEX_NAME = "nal_index.sqlite"
TASKS = ("ar", "fim")
TaskName = Literal["ar", "fim"]
REFERENCE_MODES = ("normal", "no_ref", "zero_ref", "shuffled_ref")
ReferenceMode = Literal["normal", "no_ref", "zero_ref", "shuffled_ref"]


@dataclass
class ByteDataConfig:
    """Hyperparameters controlling byte-domain sample construction."""

    p_fim: float = 0.0  # Probability of FIM sample; otherwise sample is AR.
    num_ref_slices: int = 1  # Number of previous VCL slices included as B_ref.
    # Conditioning ablation. "normal" uses the target's true prior slices;
    # "no_ref" removes them; "zero_ref" preserves their lengths with zero
    # bytes; "shuffled_ref" uses deterministic, similarly sized slices from
    # another video.
    reference_mode: ReferenceMode = "normal"
    # Target VCL NAL types. Default is P slices only (type 1) because early
    # bridge recovery should test the common inter-frame packet-loss case with
    # a real reference frame. Add IDR/I slices (type 5) later for harder
    # header/intra-frame recovery experiments.
    target_nal_types: tuple[int, ...] = (1,)
    val_fraction: float = 0.01  # Fraction of slice samples held out for validation.
    seed: int = 42  # Seed for train/val split and deterministic span sampling.
    num_workers: int = 4  # DataLoader workers.
    fim_min_gap: int = 64  # Minimum FIM missing-span length in bytes.
    fim_max_gap: int = (
        1400  # Maximum FIM missing-span length in bytes; ~FU-A payload scale.
    )
    # Preliminary simplification: avoid sampling FIM gaps too close to the
    # NAL/slice header so early experiments focus on payload recovery. Future
    # evaluations should reduce this to 0 and explicitly test missing-header cases.
    slice_header_guard_bytes: int = 64
    condition_on_sps_pps: bool = True  # Feed latest SPS/PPS NAL bytes as B_meta conditioning.
    default_max_seq_length: int = (
        32768  # Used when LitGPT connect() does not provide max_seq_length.
    )


@dataclass(frozen=True)
class NALUnit:
    start: int
    end: int
    start_code_len: int
    nal_type: int

    @property
    def payload_start(self) -> int:
        return self.start + self.start_code_len + 1


@dataclass(frozen=True)
class SliceSample:
    h264_path: Path
    target_index: int
    ref_indices: tuple[int, ...]
    meta_indices: tuple[int, ...]
    nal_type: int


def load_manifest_rows(
    manifest_path: Path,
    max_rows: int | None = None,
    report_progress: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manifest_dir = manifest_path.parent
    started_at = time.perf_counter()
    with manifest_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "ok" and row.get("h264_path"):
                row = dict(row)
                row["h264_path"] = str(resolve_manifest_path(row["h264_path"], manifest_dir))
                rows.append(row)
                if max_rows is not None and len(rows) >= max_rows:
                    break
            if report_progress and line_number % 10_000 == 0:
                elapsed = time.perf_counter() - started_at
                print(
                    f"Reading manifest: {line_number:,} rows, "
                    f"{len(rows):,} usable, {elapsed:.1f}s",
                    flush=True,
                )
    if not rows:
        raise ValueError(f"No usable rows found in manifest: {manifest_path}")
    if report_progress:
        elapsed = time.perf_counter() - started_at
        print(
            f"Manifest loaded: {len(rows):,} usable rows in {elapsed:.1f}s",
            flush=True,
        )
    return rows


def resolve_manifest_path(path: str | Path, manifest_dir: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path

    direct = manifest_dir / path
    # Normal corpus paths begin with h264/. Check this syntactically before
    # touching the filesystem; calling exists() for every manifest row causes
    # a metadata-bound scan over large corpora before indexing even starts.
    if path.parts[:1] == ("h264",) or direct.exists():
        return direct

    # Backward compatibility for manifests produced from a relative output_dir,
    # e.g. "../corpus/h264/h264/part/clip.h264". The corpus invariant is that
    # manifest.jsonl and the h264/ directory live at the same level, so recover
    # the path suffix rooted at the last h264 component.
    h264_positions = [i for i, part in enumerate(path.parts) if part == "h264"]
    if h264_positions:
        h264_suffix = Path(*path.parts[h264_positions[-1] :])
        return manifest_dir / h264_suffix

    return manifest_dir / "h264" / path


def default_nal_index_path(manifest_path: Path) -> Path:
    return manifest_path.with_name(DEFAULT_NAL_INDEX_NAME)


def load_nal_index(
    index_path: Path, manifest_path: Path, rows: list[dict[str, Any]]
) -> dict[str, list[NALUnit]]:
    """Load precomputed NAL offsets after validating the source manifest."""
    manifest_stat = manifest_path.stat()
    wanted_paths = {str(Path(row["h264_path"])) for row in rows}
    nal_index: dict[str, list[NALUnit]] = {path: [] for path in wanted_paths}

    with sqlite3.connect(index_path) as connection:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        expected = {
            "version": str(NAL_INDEX_VERSION),
            "manifest_path": str(manifest_path.resolve()),
            "manifest_size": str(manifest_stat.st_size),
            "manifest_mtime_ns": str(manifest_stat.st_mtime_ns),
        }
        mismatches = [
            key for key, value in expected.items() if metadata.get(key) != value
        ]
        if mismatches:
            raise RuntimeError(
                f"Stale NAL index {index_path}; metadata differs for "
                f"{', '.join(mismatches)}. Rebuild the index."
            )

        indexed_file_count = connection.execute(
            "SELECT COUNT(*) FROM files"
        ).fetchone()[0]
        indexed_paths = {
            path for (path,) in connection.execute("SELECT path FROM files")
        }
        missing = wanted_paths - indexed_paths
        if missing:
            example = next(iter(missing))
            raise RuntimeError(
                f"NAL index {index_path} is missing {len(missing):,} manifest files, "
                f"for example {example}. Resume the index build."
            )

        base_query = """
            SELECT files.path, nals.start, nals.end, nals.start_code_len, nals.nal_type
            FROM nals
            JOIN files ON files.id = nals.file_id
        """
        if len(wanted_paths) == indexed_file_count:
            queries = [
                (base_query + " ORDER BY files.id, nals.nal_index", ())
            ]
        else:
            wanted_list = list(wanted_paths)
            queries = []
            for start in range(0, len(wanted_list), 900):
                path_chunk = wanted_list[start : start + 900]
                placeholders = ",".join("?" for _ in path_chunk)
                queries.append(
                    (
                        base_query
                        + f" WHERE files.path IN ({placeholders})"
                        + " ORDER BY files.id, nals.nal_index",
                        path_chunk,
                    )
                )

        for query, parameters in queries:
            for path, start, end, start_code_len, nal_type in connection.execute(
                query, parameters
            ):
                nal_index[path].append(
                    NALUnit(start, end, start_code_len, nal_type)
                )

    print(
        f"Loaded cached H.264 NAL index: {len(nal_index):,} files from {index_path}",
        flush=True,
    )
    return nal_index


def parse_annexb_nals(data: bytes) -> list[NALUnit]:
    starts = list(_iter_start_codes(data))
    nals: list[NALUnit] = []
    for i, (start, start_code_len) in enumerate(starts):
        header_pos = start + start_code_len
        if header_pos >= len(data):
            continue
        end = starts[i + 1][0] if i + 1 < len(starts) else len(data)
        nal_type = data[header_pos] & 0x1F
        nals.append(
            NALUnit(
                start=start, end=end, start_code_len=start_code_len, nal_type=nal_type
            )
        )
    return nals


def _iter_start_codes(data: bytes):
    # bytes.find performs the bulk scan in optimized C. The previous
    # byte-by-byte Python loop became the dominant cost when indexing tens of
    # gigabytes of H.264 files at every training startup.
    marker = b"\x00\x00\x01"
    search_from = 0
    while True:
        marker_start = data.find(marker, search_from)
        if marker_start < 0:
            return

        if marker_start > 0 and data[marker_start - 1] == 0:
            yield marker_start - 1, 4
        else:
            yield marker_start, 3
        search_from = marker_start + len(marker)


def bytes_to_ids(data: bytes) -> Tensor:
    return torch.tensor(list(data), dtype=torch.long)


def pad_and_truncate(tensors: list[Tensor], max_seq_length: int, value: int) -> Tensor:
    truncated = [tensor[:max_seq_length] for tensor in tensors]
    return torch.nn.utils.rnn.pad_sequence(
        truncated, batch_first=True, padding_value=value
    )


class ByteSliceDataset(Dataset):
    """Reference-conditioned AR/FIM dataset over H.264 frame/slice NALs.

    With probability ``p_fim`` the sample is FIM, otherwise it is AR.

    AR layout:
        input_ids = [B_meta, B_ref, SLICE_BOS, B_t[:-1]]
        labels    = [-100 ...,                    B_t]

    FIM layout:
        B_t       = [B_pre, B_miss, B_orph]
        input_ids = [B_meta, B_ref, B_pre, B_orph, SPAN_BOS, B_miss[:-1]]
        labels    = [-100 ...,                                      B_miss]
    """

    def __init__(
        self,
        rows: list[dict[str, Any]],
        max_seq_length: int,
        p_fim: float = 0.0,
        num_ref_slices: int = 1,
        target_nal_types: tuple[int, ...] = (1,),
        fim_min_gap: int = 64,
        fim_max_gap: int = 1400,
        slice_header_guard_bytes: int = 64,
        condition_on_sps_pps: bool = True,
        reference_mode: ReferenceMode = "normal",
        include_sps_pps_metadata: bool | None = None,
        include_parameter_sets: bool | None = None,
        nal_index: dict[str, list[NALUnit]] | None = None,
        seed: int = 42,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        if not 0.0 <= p_fim <= 1.0:
            raise ValueError("p_fim must be in [0, 1]")
        if max_seq_length < 4:
            raise ValueError("max_seq_length must be at least 4")
        if num_ref_slices < 0:
            raise ValueError("num_ref_slices must be non-negative")
        if reference_mode not in REFERENCE_MODES:
            raise ValueError(f"reference_mode must be one of {REFERENCE_MODES}")

        self.rows = rows
        self.max_seq_length = max_seq_length
        self.p_fim = p_fim
        self.num_ref_slices = num_ref_slices
        self.target_nal_types = set(target_nal_types)
        self.fim_min_gap = fim_min_gap
        self.fim_max_gap = fim_max_gap
        self.slice_header_guard_bytes = slice_header_guard_bytes
        if include_parameter_sets is not None:
            condition_on_sps_pps = include_parameter_sets
        if include_sps_pps_metadata is not None:
            condition_on_sps_pps = include_sps_pps_metadata
        self.condition_on_sps_pps = condition_on_sps_pps
        self.reference_mode = reference_mode
        self.seed = seed
        self.ignore_index = ignore_index
        self.samples, self.nal_index = self._build_index(nal_index)
        self.shuffled_ref_indices = (
            self._build_shuffled_ref_indices()
            if self.reference_mode == "shuffled_ref"
            else {}
        )

        if not self.samples:
            raise ValueError(
                "No usable VCL slice samples found. Check max_seq_length, target_nal_types, "
                "num_ref_slices, and preprocessing output."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Tensor | dict[str, int | str]]:
        rng = random.Random(self.seed + idx)
        sample = self.samples[idx]
        data = sample.h264_path.read_bytes()
        nals = self.nal_index[str(sample.h264_path)]

        meta = self._read_nal_bytes(data, nals, sample.meta_indices)
        ref_chunks, reference_source_path = self._get_reference_chunks(idx, data, nals, sample)
        target_nal = nals[sample.target_index]
        target = bytes_to_ids(data[target_nal.start : target_nal.end])

        if self.p_fim > 0 and rng.random() < self.p_fim:
            return self._build_fim_item(
                meta, ref_chunks, target, target_nal, rng, sample, reference_source_path
            )
        return self._build_ar_item(meta, ref_chunks, target, sample, reference_source_path)

    def _get_reference_chunks(
        self, idx: int, data: bytes, nals: list[NALUnit], sample: SliceSample
    ) -> tuple[list[Tensor], Path | None]:
        if self.reference_mode == "no_ref":
            return [], None

        true_chunks = self._read_nal_byte_chunks(data, nals, sample.ref_indices)
        if self.reference_mode == "zero_ref":
            return [torch.zeros_like(chunk) for chunk in true_chunks], sample.h264_path
        if self.reference_mode == "normal":
            return true_chunks, sample.h264_path

        source = self.samples[self.shuffled_ref_indices[idx]]
        source_data = source.h264_path.read_bytes()
        source_nals = self.nal_index[str(source.h264_path)]
        return (
            self._read_nal_byte_chunks(source_data, source_nals, source.ref_indices),
            source.h264_path,
        )

    def _build_shuffled_ref_indices(self) -> dict[int, int]:
        """Map each target to a similarly sized reference from another video."""
        buckets: dict[int, list[int]] = {}
        for idx, sample in enumerate(self.samples):
            ref_len = self._reference_length(sample)
            buckets.setdefault(ref_len // 1024, []).append(idx)

        all_indices = list(range(len(self.samples)))
        mapping: dict[int, int] = {}
        for idx, sample in enumerate(self.samples):
            candidates = buckets[self._reference_length(sample) // 1024]
            source_idx = self._next_other_video(idx, candidates)
            if source_idx is None:
                source_idx = self._next_other_video(idx, all_indices)
            if source_idx is None:
                raise ValueError("shuffled_ref requires samples from at least two videos")
            mapping[idx] = source_idx
        return mapping

    def _reference_length(self, sample: SliceSample) -> int:
        nals = self.nal_index[str(sample.h264_path)]
        return sum(nals[i].end - nals[i].start for i in sample.ref_indices)

    def _next_other_video(self, idx: int, candidates: list[int]) -> int | None:
        if not candidates:
            return None
        start = (self.seed + idx) % len(candidates)
        target_path = self.samples[idx].h264_path
        for offset in range(len(candidates)):
            candidate_idx = candidates[(start + offset) % len(candidates)]
            if self.samples[candidate_idx].h264_path != target_path:
                return candidate_idx
        return None

    def _build_index(
        self, cached_nal_index: dict[str, list[NALUnit]] | None
    ) -> tuple[list[SliceSample], dict[str, list[NALUnit]]]:
        samples: list[SliceSample] = []
        nal_index = cached_nal_index if cached_nal_index is not None else {}
        started_at = time.perf_counter()
        last_report_at = started_at
        indexed_bytes = 0
        total_rows = len(self.rows)
        for row_number, row in enumerate(self.rows, start=1):
            path = Path(row["h264_path"])
            path_key = str(path)
            if cached_nal_index is None:
                data = path.read_bytes()
                indexed_bytes += len(data)
                nals = parse_annexb_nals(data)
                nal_index[path_key] = nals
            else:
                try:
                    nals = nal_index[path_key]
                except KeyError as exc:
                    raise RuntimeError(
                        f"Cached NAL index has no entry for {path}"
                    ) from exc
            vcl_indices = [
                i for i, nal in enumerate(nals) if nal.nal_type in VCL_NAL_TYPES
            ]
            for pos, nal_idx in enumerate(vcl_indices):
                nal = nals[nal_idx]
                if nal.nal_type not in self.target_nal_types:
                    continue
                if pos < self.num_ref_slices:
                    continue
                if nal.end - nal.start > self.max_seq_length:
                    continue
                refs = tuple(vcl_indices[pos - self.num_ref_slices : pos])
                meta_indices = self._latest_parameter_set_indices(nals, nal_idx)
                samples.append(
                    SliceSample(path, nal_idx, refs, meta_indices, nal.nal_type)
                )

            now = time.perf_counter()
            if cached_nal_index is None and (
                now - last_report_at >= 10 or row_number == total_rows
            ):
                elapsed = now - started_at
                throughput = indexed_bytes / max(elapsed, 1e-9) / 1e6
                print(
                    "Indexing H.264 corpus: "
                    f"{row_number:,}/{total_rows:,} files, "
                    f"{indexed_bytes / 1e9:.2f} GB, "
                    f"{len(samples):,} samples, "
                    f"{throughput:.1f} MB/s",
                    flush=True,
                )
                last_report_at = now
        return samples, nal_index

    def _latest_parameter_set_indices(
        self, nals: list[NALUnit], target_index: int
    ) -> tuple[int, ...]:
        if not self.condition_on_sps_pps:
            return ()

        latest: dict[int, int] = {}
        for i, nal in enumerate(nals[:target_index]):
            if nal.nal_type in PARAMETER_SET_NAL_TYPES:
                latest[nal.nal_type] = i
        return tuple(sorted(latest.values()))

    def _read_nal_bytes(
        self, data: bytes, nals: list[NALUnit], indices: tuple[int, ...]
    ) -> Tensor:
        if not indices:
            return torch.empty(0, dtype=torch.long)

        chunks = [data[nals[i].start : nals[i].end] for i in indices]
        return bytes_to_ids(b"".join(chunks))

    def _read_nal_byte_chunks(
        self, data: bytes, nals: list[NALUnit], indices: tuple[int, ...]
    ) -> list[Tensor]:
        return [bytes_to_ids(data[nals[i].start : nals[i].end]) for i in indices]

    def _fit_conditioning_to_budget(
        self, meta: Tensor, ref_chunks: list[Tensor], target_len: int
    ) -> tuple[Tensor, Tensor, int]:
        budget = self.max_seq_length - target_len
        if budget <= 0:
            return meta[:0], torch.empty(0, dtype=torch.long), len(ref_chunks)
        if meta.numel() >= budget:
            return meta[-budget:], torch.empty(0, dtype=torch.long), len(ref_chunks)

        remaining = budget - meta.numel()
        kept: list[Tensor] = []
        used = 0
        for chunk in reversed(ref_chunks):
            if used + chunk.numel() > remaining:
                continue
            kept.append(chunk)
            used += chunk.numel()

        kept.reverse()
        ref = torch.cat(kept) if kept else torch.empty(0, dtype=torch.long)
        return meta, ref, len(ref_chunks) - len(kept)

    def _build_ar_item(
        self,
        meta: Tensor,
        ref_chunks: list[Tensor],
        target: Tensor,
        sample: SliceSample,
        reference_source_path: Path | None,
    ) -> dict[str, Any]:
        # Teacher forcing: [SLICE_BOS, B_t[:-1]] predicts B_t.
        meta, ref, dropped_ref_slices = self._fit_conditioning_to_budget(
            meta, ref_chunks, target.numel()
        )
        input_ids = torch.cat(
            (meta, ref, torch.tensor([SLICE_BOS_ID], dtype=torch.long), target[:-1])
        )
        labels = torch.full_like(input_ids, self.ignore_index)
        labels[-target.numel() :] = target
        region_ids = torch.cat(
            (
                torch.full((meta.numel(),), REGION_META, dtype=torch.long),
                torch.full((ref.numel(),), REGION_REF, dtype=torch.long),
                torch.full((target.numel(),), REGION_TARGET, dtype=torch.long),
            )
        )
        offset_ids = torch.arange(input_ids.numel(), dtype=torch.long)
        return self._pack_item(
            input_ids,
            labels,
            region_ids,
            offset_ids,
            sample,
            task="ar",
            dropped_ref_slices=dropped_ref_slices,
            reference_source_path=reference_source_path,
        )

    def _build_fim_item(
        self,
        meta: Tensor,
        ref_chunks: list[Tensor],
        target: Tensor,
        target_nal: NALUnit,
        rng: random.Random,
        sample: SliceSample,
        reference_source_path: Path | None,
    ) -> dict[str, Any]:
        prefix, missing, orphan, split_offset = self._sample_fim_parts(
            target, target_nal, rng
        )
        # prefix + orphan + bridge_in has the same length as target.
        meta, ref, dropped_ref_slices = self._fit_conditioning_to_budget(
            meta, ref_chunks, target.numel()
        )

        # Teacher forcing: [SPAN_BOS, B_miss[:-1]] predicts B_miss.
        bridge_in = torch.cat(
            (torch.tensor([SPAN_BOS_ID], dtype=torch.long), missing[:-1])
        )
        input_ids = torch.cat((meta, ref, prefix, orphan, bridge_in))
        labels = torch.full_like(input_ids, self.ignore_index)
        labels[-missing.numel() :] = missing
        region_ids = torch.cat(
            (
                torch.full((meta.numel(),), REGION_META, dtype=torch.long),
                torch.full((ref.numel(),), REGION_REF, dtype=torch.long),
                torch.full((prefix.numel(),), REGION_PREFIX, dtype=torch.long),
                torch.full((orphan.numel(),), REGION_ORPHAN, dtype=torch.long),
                torch.full((bridge_in.numel(),), REGION_BRIDGE, dtype=torch.long),
            )
        )
        offset_ids = torch.cat(
            (
                torch.arange(meta.numel(), dtype=torch.long),
                torch.arange(ref.numel(), dtype=torch.long),
                torch.arange(prefix.numel(), dtype=torch.long),
                torch.arange(
                    prefix.numel() + missing.numel(), target.numel(), dtype=torch.long
                ),
                torch.arange(
                    split_offset, split_offset + bridge_in.numel(), dtype=torch.long
                ),
            )
        )
        return self._pack_item(
            input_ids,
            labels,
            region_ids,
            offset_ids,
            sample,
            task="fim",
            dropped_ref_slices=dropped_ref_slices,
            reference_source_path=reference_source_path,
        )

    def _sample_fim_parts(
        self, target: Tensor, target_nal: NALUnit, rng: random.Random
    ) -> tuple[Tensor, Tensor, Tensor, int]:
        protected = min(
            target.numel() - 2,
            target_nal.start_code_len + 1 + self.slice_header_guard_bytes,
        )
        max_gap = min(self.fim_max_gap, target.numel() - protected - 1)
        if max_gap < 1:
            split = max(1, target.numel() // 2)
            gap = 1
        else:
            min_gap = min(self.fim_min_gap, max_gap)
            gap = rng.randint(min_gap, max_gap)
            split = rng.randint(protected, target.numel() - gap)

        prefix = target[:split]
        missing = target[split : split + gap]
        orphan = target[split + gap :]
        return prefix, missing, orphan, split

    def _pack_item(
        self,
        input_ids: Tensor,
        labels: Tensor,
        region_ids: Tensor,
        offset_ids: Tensor,
        sample: SliceSample,
        task: TaskName,
        dropped_ref_slices: int,
        reference_source_path: Path | None,
    ) -> dict[str, Any]:
        return {
            "input_ids": input_ids,
            "labels": labels,
            "region_ids": region_ids,
            "offset_ids": offset_ids,
            "token_counts": {
                "raw": int(input_ids.numel()),
                "raw_plus_prompt_template": int(input_ids.numel()),
            },
            "sample_meta": {
                "task": task,
                "h264_path": str(sample.h264_path),
                "target_index": sample.target_index,
                "num_ref_slices": len(sample.ref_indices),
                "reference_mode": self.reference_mode,
                "reference_source_path": (
                    str(reference_source_path) if reference_source_path is not None else None
                ),
                "dropped_ref_slices": dropped_ref_slices,
                "num_meta_nals": len(sample.meta_indices),
                "nal_type": sample.nal_type,
            },
        }


def collate_byte_samples(
    samples: list[dict[str, Any]], max_seq_length: int
) -> dict[str, Tensor | dict[str, Tensor]]:
    return {
        "input_ids": pad_and_truncate(
            [sample["input_ids"] for sample in samples], max_seq_length, PAD_ID
        ),
        "labels": pad_and_truncate(
            [sample["labels"] for sample in samples], max_seq_length, IGNORE_INDEX
        ),
        "region_ids": pad_and_truncate(
            [sample["region_ids"] for sample in samples], max_seq_length, REGION_PAD
        ),
        "offset_ids": pad_and_truncate(
            [sample["offset_ids"] for sample in samples], max_seq_length, 0
        ),
        "token_counts": {
            "raw": torch.tensor(
                [sample["token_counts"]["raw"] for sample in samples], dtype=torch.int64
            ).unsqueeze(1),
            "raw_plus_prompt_template": torch.tensor(
                [
                    sample["token_counts"]["raw_plus_prompt_template"]
                    for sample in samples
                ],
                dtype=torch.int64,
            ).unsqueeze(1),
        },
    }


@dataclass
class ByteDataModule(DataModule):
    """DataModule for mixed AR/FIM byte-domain training over H.264 slices."""

    manifest_path: Path
    config: ByteDataConfig = field(default_factory=ByteDataConfig)
    max_manifest_rows: int | None = None  # Optional corpus limit for smoke/debug runs.
    nal_index_path: Path | None = None  # Defaults to manifest_dir/nal_index.sqlite when present.

    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(
        default=ByteDataConfig.default_max_seq_length, init=False, repr=False
    )
    train_dataset: Dataset | None = field(default=None, init=False, repr=False)
    val_dataset: Dataset | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__init__()
        self.manifest_path = Path(self.manifest_path)
        if self.nal_index_path is not None:
            self.nal_index_path = Path(self.nal_index_path)

    def connect(
        self,
        tokenizer: Tokenizer | None = None,
        batch_size: int = 1,
        max_seq_length: int | None = None,
        **kwargs: Any,
    ) -> None:
        # Tokenizer is intentionally ignored: preprocessing already gives bytes,
        # and bytes map to ids directly.
        self.batch_size = batch_size
        self.max_seq_length = (
            self.config.default_max_seq_length
            if max_seq_length is None
            else max_seq_length
        )

    def setup(self, stage: str = "") -> None:
        rows = load_manifest_rows(self.manifest_path, max_rows=self.max_manifest_rows)
        index_path = self.nal_index_path or default_nal_index_path(self.manifest_path)
        nal_index = (
            load_nal_index(index_path, self.manifest_path, rows)
            if index_path.is_file()
            else None
        )
        if self.nal_index_path is not None and nal_index is None:
            raise FileNotFoundError(f"NAL index does not exist: {index_path}")
        dataset = ByteSliceDataset(
            rows,
            max_seq_length=self.max_seq_length,
            p_fim=self.config.p_fim,
            num_ref_slices=self.config.num_ref_slices,
            target_nal_types=self.config.target_nal_types,
            fim_min_gap=self.config.fim_min_gap,
            fim_max_gap=self.config.fim_max_gap,
            slice_header_guard_bytes=self.config.slice_header_guard_bytes,
            condition_on_sps_pps=self.config.condition_on_sps_pps,
            reference_mode=self.config.reference_mode,
            nal_index=nal_index,
            seed=self.config.seed,
        )
        val_size = max(1, int(len(dataset) * self.config.val_fraction))
        train_size = len(dataset) - val_size
        if train_size <= 0:
            raise ValueError(
                "Need at least two usable byte samples for train/val split"
            )
        generator = torch.Generator().manual_seed(self.config.seed)
        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size], generator=generator
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup()
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.setup()
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(
        self, samples: list[dict[str, Any]]
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        return collate_byte_samples(samples, max_seq_length=self.max_seq_length)
