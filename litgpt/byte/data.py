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
import os
import random
import sqlite3
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split

from litgpt.data.base import DataModule
from litgpt.tokenizer import Tokenizer

BYTE_VOCAB_SIZE = 256  # Raw byte ids are exactly 0..255.
PAD_ID = 256  # Padding id for variable-length batches; never a real byte.
SLICE_BOS_ID = 257  # Starts AR generation of the full target slice B_t.
SPAN_BOS_ID = 258  # Starts FIM generation of the missing span B_miss.
FIM_BEGIN_ID = 259  # Starts a PSM-formatted target after metadata/reference context.
FIM_HOLE_ID = 260  # Marks where the missing span was removed and the suffix begins.
FIM_END_ID = 261  # Ends the known suffix and starts missing-span generation.
SEQ_EOS_ID = 262  # Optional end-of-target marker; only emitted when use_eos=True.
VOCAB_SIZE = 259  # Bridge/AR vocab: bytes + PAD, SLICE_BOS, and SPAN_BOS.
PSM_VOCAB_SIZE = 262  # PSM vocab additionally includes BEGIN, HOLE, and END.
EOS_VOCAB_SIZE = 263  # Adds SEQ_EOS on top of the PSM markers; shared by both formats.
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
FIM_FORMATS = ("bridge", "psm")
FIMFormat = Literal["bridge", "psm"]
DATASET_MODES = ("slice", "window")
DatasetMode = Literal["slice", "window"]


def vocab_size_for_fim_format(fim_format: FIMFormat, use_eos: bool = False) -> int:
    if fim_format not in FIM_FORMATS:
        raise ValueError(f"fim_format must be one of {FIM_FORMATS}")
    # SEQ_EOS shares a single id across formats, so enabling it lifts both the
    # bridge and PSM vocabularies to the same size.
    if use_eos:
        return EOS_VOCAB_SIZE
    return PSM_VOCAB_SIZE if fim_format == "psm" else VOCAB_SIZE


@dataclass
class ByteDataConfig:
    """Hyperparameters controlling byte-domain sample construction."""

    p_fim: float = 0.0  # Probability of FIM sample; otherwise sample is AR.
    # "bridge" is the original layout with one SPAN_BOS marker. "psm" uses
    # explicit Prefix-Suffix-Middle markers following code-model FIM practice.
    fim_format: FIMFormat = "bridge"
    # When True, append SEQ_EOS after each AR target / FIM span so the model
    # learns to terminate. Default False keeps the oracle-length convention
    # where generation length is supplied externally.
    use_eos: bool = False
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
    # When True, the train/val split is performed over source videos (h264_path)
    # rather than over individual slice samples, eliminating within-video
    # leakage. val_fraction then denotes the fraction of *videos* held out.
    split_by_video: bool = False
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
    condition_on_sps_pps: bool = (
        True  # Feed latest SPS/PPS NAL bytes as B_meta conditioning.
    )
    default_max_seq_length: int = (
        32768  # Used when LitGPT connect() does not provide max_seq_length.
    )
    # "slice" = ByteSliceDataset (one ref slice + one target slice). "window" =
    # ByteStreamWindowDataset: the multi-frame contiguous-stream AR objective used
    # for H0 (verifying the AVC-LM/JPEG-LM generation legacy). See 0616.md.
    dataset_mode: DatasetMode = "slice"
    window_min_frames: int = 2  # Minimum VCL NALs per stream window ("window" mode).
    # NB: counts VCL NALs, which == frames only for one-slice-per-frame corpora. Under
    # AVC-LM's slice-max-mbs=1 (one slice per macroblock) this becomes a min-slices gate.


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
                row["h264_path"] = str(
                    resolve_manifest_path(row["h264_path"], manifest_dir)
                )
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


class LazyNalIndex:
    """Mapping ``path -> list[NALUnit]`` backed by ``nal_index.sqlite``.

    Materializes a file's NALs on demand (indexed by ``file_id``) with a bounded
    LRU cache, so host RAM does NOT scale with total corpus NAL count -- essential
    for the AVC-LM (slice-max-mbs=1) corpus, where the fully-resident dict was tens
    of GB and was multiplied per DataLoader worker. Only the compact per-file
    ``path -> file_id`` map stays resident; NALUnit objects come and go.

    Access matches the old dict (``nal_index[path]``, ``path in nal_index``,
    ``len(nal_index)``). A read-only sqlite connection is (re)opened per process so
    it is safe across forked DataLoader workers.
    """

    def __init__(self, index_path: Path, file_ids: dict[str, int], cache_size: int = 128) -> None:
        self._index_path = str(index_path)
        self._file_ids = file_ids  # path -> files.id (one small entry per file)
        self._cache: "OrderedDict[str, list[NALUnit]]" = OrderedDict()
        self._cache_size = cache_size
        self._conn: sqlite3.Connection | None = None
        self._pid: int | None = None

    def _connection(self) -> sqlite3.Connection:
        pid = os.getpid()
        if self._conn is None or self._pid != pid:
            # First use, or a forked worker inherited the parent's handle: a sqlite
            # connection must not be shared across processes, so open a fresh one.
            self._conn = sqlite3.connect(
                f"file:{self._index_path}?mode=ro", uri=True, check_same_thread=False
            )
            self._pid = pid
        return self._conn

    def __contains__(self, path: str) -> bool:
        return path in self._file_ids

    def __len__(self) -> int:
        return len(self._file_ids)

    def __getitem__(self, path: str) -> list[NALUnit]:
        cached = self._cache.get(path)
        if cached is not None:
            self._cache.move_to_end(path)
            return cached
        try:
            file_id = self._file_ids[path]
        except KeyError as exc:
            raise KeyError(path) from exc
        rows = self._connection().execute(
            "SELECT start, end, start_code_len, nal_type FROM nals "
            "WHERE file_id = ? ORDER BY nal_index",
            (file_id,),
        ).fetchall()
        nals = [NALUnit(s, e, scl, nt) for (s, e, scl, nt) in rows]
        self._cache[path] = nals
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return nals


def load_nal_index(
    index_path: Path, manifest_path: Path, rows: list[dict[str, Any]]
) -> LazyNalIndex:
    """Open the precomputed NAL index after validating the source manifest.

    Returns a LazyNalIndex (sqlite-backed, bounded RAM) rather than materializing
    every NALUnit -- the fully-resident dict did not scale to the per-MB corpus.
    """
    manifest_stat = manifest_path.stat()
    wanted_paths = {str(Path(row["h264_path"])) for row in rows}

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

        # One small row per file; keep only the path -> id map resident.
        all_file_ids = {
            path: file_id
            for (path, file_id) in connection.execute("SELECT path, id FROM files")
        }

    missing = wanted_paths - set(all_file_ids)
    if missing:
        example = next(iter(missing))
        raise RuntimeError(
            f"NAL index {index_path} is missing {len(missing):,} manifest files, "
            f"for example {example}. Resume the index build."
        )
    file_ids = {path: all_file_ids[path] for path in wanted_paths}

    print(
        f"Opened lazy H.264 NAL index: {len(file_ids):,} files from {index_path}",
        flush=True,
    )
    return LazyNalIndex(index_path, file_ids)


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
        bridge    = [B_meta, B_ref, B_pre, B_orph, SPAN_BOS, B_miss[:-1]]
        psm       = [B_meta, B_ref, FIM_BEGIN, B_pre, FIM_HOLE, B_orph,
                     FIM_END, B_miss[:-1]]
        labels    = [-100 ..., B_miss]
    """

    def __init__(
        self,
        rows: list[dict[str, Any]],
        max_seq_length: int,
        p_fim: float = 0.0,
        fim_format: FIMFormat = "bridge",
        use_eos: bool = False,
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
        if fim_format not in FIM_FORMATS:
            raise ValueError(f"fim_format must be one of {FIM_FORMATS}")
        if max_seq_length < 4:
            raise ValueError("max_seq_length must be at least 4")
        if num_ref_slices < 0:
            raise ValueError("num_ref_slices must be non-negative")
        if reference_mode not in REFERENCE_MODES:
            raise ValueError(f"reference_mode must be one of {REFERENCE_MODES}")
        if fim_min_gap < 1:
            raise ValueError("fim_min_gap must be positive")
        if fim_max_gap < fim_min_gap:
            raise ValueError("fim_max_gap must be greater than or equal to fim_min_gap")

        self.rows = rows
        self.max_seq_length = max_seq_length
        self.p_fim = p_fim
        self.fim_format = fim_format
        self.use_eos = use_eos
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
        ref_chunks, reference_source_path = self._get_reference_chunks(
            idx, data, nals, sample
        )
        target_nal = nals[sample.target_index]
        target = bytes_to_ids(data[target_nal.start : target_nal.end])

        fim_selected = self.p_fim > 0 and rng.random() < self.p_fim
        fim_eligible = self._max_fim_gap(target, target_nal) >= self.fim_min_gap
        if fim_selected and fim_eligible:
            return self._build_fim_item(
                meta, ref_chunks, target, target_nal, rng, sample, reference_source_path
            )
        return self._build_ar_item(
            meta, ref_chunks, target, sample, reference_source_path
        )

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
                raise ValueError(
                    "shuffled_ref requires samples from at least two videos"
                )
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
                format_overhead = (
                    2 if self.p_fim > 0 and self.fim_format == "psm" else 0
                )
                if nal.end - nal.start + format_overhead > self.max_seq_length:
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

    def _with_eos(self, content: Tensor) -> Tensor:
        """Append SEQ_EOS to a generated span when use_eos is enabled."""
        if not self.use_eos:
            return content
        return torch.cat((content, torch.tensor([SEQ_EOS_ID], dtype=torch.long)))

    def _build_ar_item(
        self,
        meta: Tensor,
        ref_chunks: list[Tensor],
        target: Tensor,
        sample: SliceSample,
        reference_source_path: Path | None,
    ) -> dict[str, Any]:
        # Teacher forcing: [SLICE_BOS, target_tail[:-1]] predicts target_tail,
        # where target_tail is B_t optionally followed by SEQ_EOS.
        eos_overhead = 1 if self.use_eos else 0
        meta, ref, dropped_ref_slices = self._fit_conditioning_to_budget(
            meta, ref_chunks, target.numel() + eos_overhead
        )
        target_tail = self._with_eos(target)
        gen_in = torch.cat(
            (torch.tensor([SLICE_BOS_ID], dtype=torch.long), target_tail[:-1])
        )
        input_ids = torch.cat((meta, ref, gen_in))
        labels = torch.full_like(input_ids, self.ignore_index)
        labels[-target_tail.numel() :] = target_tail
        region_ids = torch.cat(
            (
                torch.full((meta.numel(),), REGION_META, dtype=torch.long),
                torch.full((ref.numel(),), REGION_REF, dtype=torch.long),
                torch.full((gen_in.numel(),), REGION_TARGET, dtype=torch.long),
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
        format_overhead = 2 if self.fim_format == "psm" else 0
        eos_overhead = 1 if self.use_eos else 0
        meta, ref, dropped_ref_slices = self._fit_conditioning_to_budget(
            meta, ref_chunks, target.numel() + format_overhead + eos_overhead
        )
        # missing_tail is B_miss optionally followed by SEQ_EOS; teacher forcing
        # feeds [marker, missing_tail[:-1]] and supervises missing_tail.
        missing_tail = self._with_eos(missing)

        if self.fim_format == "psm":
            prefix_marker = torch.tensor([FIM_BEGIN_ID], dtype=torch.long)
            suffix_marker = torch.tensor([FIM_HOLE_ID], dtype=torch.long)
            middle_in = torch.cat(
                (torch.tensor([FIM_END_ID], dtype=torch.long), missing_tail[:-1])
            )
            input_ids = torch.cat(
                (
                    meta,
                    ref,
                    prefix_marker,
                    prefix,
                    suffix_marker,
                    orphan,
                    middle_in,
                )
            )
            region_ids = torch.cat(
                (
                    torch.full((meta.numel(),), REGION_META, dtype=torch.long),
                    torch.full((ref.numel(),), REGION_REF, dtype=torch.long),
                    torch.full(
                        (prefix_marker.numel() + prefix.numel(),),
                        REGION_PREFIX,
                        dtype=torch.long,
                    ),
                    torch.full(
                        (suffix_marker.numel() + orphan.numel(),),
                        REGION_ORPHAN,
                        dtype=torch.long,
                    ),
                    torch.full((middle_in.numel(),), REGION_BRIDGE, dtype=torch.long),
                )
            )
            offset_ids = torch.cat(
                (
                    torch.arange(meta.numel(), dtype=torch.long),
                    torch.arange(ref.numel(), dtype=torch.long),
                    torch.tensor([0], dtype=torch.long),
                    torch.arange(prefix.numel(), dtype=torch.long),
                    torch.tensor([prefix.numel() + missing.numel()], dtype=torch.long),
                    torch.arange(
                        prefix.numel() + missing.numel(),
                        target.numel(),
                        dtype=torch.long,
                    ),
                    torch.arange(
                        split_offset,
                        split_offset + middle_in.numel(),
                        dtype=torch.long,
                    ),
                )
            )
        else:
            # Teacher forcing: [SPAN_BOS, missing_tail[:-1]] predicts missing_tail.
            middle_in = torch.cat(
                (torch.tensor([SPAN_BOS_ID], dtype=torch.long), missing_tail[:-1])
            )
            input_ids = torch.cat((meta, ref, prefix, orphan, middle_in))
            region_ids = torch.cat(
                (
                    torch.full((meta.numel(),), REGION_META, dtype=torch.long),
                    torch.full((ref.numel(),), REGION_REF, dtype=torch.long),
                    torch.full((prefix.numel(),), REGION_PREFIX, dtype=torch.long),
                    torch.full((orphan.numel(),), REGION_ORPHAN, dtype=torch.long),
                    torch.full((middle_in.numel(),), REGION_BRIDGE, dtype=torch.long),
                )
            )
            offset_ids = torch.cat(
                (
                    torch.arange(meta.numel(), dtype=torch.long),
                    torch.arange(ref.numel(), dtype=torch.long),
                    torch.arange(prefix.numel(), dtype=torch.long),
                    torch.arange(
                        prefix.numel() + missing.numel(),
                        target.numel(),
                        dtype=torch.long,
                    ),
                    torch.arange(
                        split_offset,
                        split_offset + middle_in.numel(),
                        dtype=torch.long,
                    ),
                )
            )
        labels = torch.full_like(input_ids, self.ignore_index)
        labels[-missing_tail.numel() :] = missing_tail
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
        max_gap = self._max_fim_gap(target, target_nal)
        if max_gap < self.fim_min_gap:
            raise ValueError("Target slice cannot fit the configured minimum FIM gap")
        gap = rng.randint(self.fim_min_gap, max_gap)
        split = rng.randint(protected, target.numel() - gap)

        prefix = target[:split]
        missing = target[split : split + gap]
        orphan = target[split + gap :]
        return prefix, missing, orphan, split

    def _max_fim_gap(self, target: Tensor, target_nal: NALUnit) -> int:
        protected = min(
            target.numel() - 2,
            target_nal.start_code_len + 1 + self.slice_header_guard_bytes,
        )
        return min(self.fim_max_gap, target.numel() - protected - 1)

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
                "fim_format": self.fim_format,
                "h264_path": str(sample.h264_path),
                "target_index": sample.target_index,
                "num_ref_slices": len(sample.ref_indices),
                "reference_mode": self.reference_mode,
                "reference_source_path": (
                    str(reference_source_path)
                    if reference_source_path is not None
                    else None
                ),
                "dropped_ref_slices": dropped_ref_slices,
                "num_meta_nals": len(sample.meta_indices),
                "nal_type": sample.nal_type,
            },
        }


@dataclass(frozen=True)
class WindowSample:
    h264_path: Path
    start_nal: int  # inclusive NAL index where the window begins (a GOP boundary)
    end_nal: int  # exclusive NAL index where the window ends
    num_frames: int  # number of VCL NALs in the window (== frames iff one slice/frame;
    # under slice-max-mbs=1 this is a slice count, not a frame count)


class ByteStreamWindowDataset(Dataset):
    """Contiguous multi-frame stream-window dataset for AR pretraining (H0).

    Where ``ByteSliceDataset`` conditions on one reference slice and supervises one
    target slice, each sample here is a contiguous Annex-B byte window that begins
    at a GOP boundary (the parameter sets preceding an IDR, then the IDR) and packs
    complete NAL units until ``max_seq_length`` is reached -- never splitting a NAL
    or crossing a video. Next-byte loss is applied across the whole window: the
    AVC-LM full-stream objective. Previous frames are simply the causal context, so
    no reference slice is duplicated; offset ids reset at each NAL boundary.

    AR layout (``p_fim == 0``):
        window    = concat(bytes[NAL.start:NAL.end] for NAL in window)
        input_ids = [SLICE_BOS, window[:-1]]   # SLICE_BOS reused as stream-start
        labels    = window
        region    = REGION_META for parameter sets, REGION_TARGET for VCL bytes
        offset    = arange within each NAL, reset to 0 at every NAL boundary

    The masked-span FIM-over-windows mode (``p_fim > 0`` -- hole in the window's
    last VCL frame, prior frames as causal context) is the planned parallel mode;
    not implemented yet (see 0616.md design note).
    """

    def __init__(
        self,
        rows: list[dict[str, Any]],
        max_seq_length: int,
        *,
        min_frames: int = 2,
        p_fim: float = 0.0,
        nal_index: dict[str, list[NALUnit]] | None = None,
        seed: int = 42,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        if max_seq_length < 4:
            raise ValueError("max_seq_length must be at least 4")
        if min_frames < 1:
            raise ValueError("min_frames must be positive")
        if p_fim != 0.0:
            raise NotImplementedError(
                "ByteStreamWindowDataset FIM-over-windows mode is not implemented yet; "
                "see the 0616.md design note. Use p_fim=0 for the AR/H0 objective."
            )
        self.rows = rows
        self.max_seq_length = max_seq_length
        self.min_frames = min_frames
        self.p_fim = p_fim
        self.seed = seed
        self.ignore_index = ignore_index
        self.samples, self.nal_index = self._build_index(nal_index)
        if not self.samples:
            raise ValueError(
                "No usable stream windows found. Check max_seq_length, min_frames, "
                "and that the corpus contains IDR-anchored GOPs."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _build_index(
        self, cached_nal_index: dict[str, list[NALUnit]] | None
    ) -> tuple[list[WindowSample], dict[str, list[NALUnit]]]:
        samples: list[WindowSample] = []
        nal_index = cached_nal_index if cached_nal_index is not None else {}
        for row in self.rows:
            path = Path(row["h264_path"])
            path_key = str(path)
            if cached_nal_index is None:
                nals = parse_annexb_nals(path.read_bytes())
                nal_index[path_key] = nals
            else:
                try:
                    nals = nal_index[path_key]
                except KeyError as exc:
                    raise RuntimeError(
                        f"Cached NAL index has no entry for {path}"
                    ) from exc
            samples.extend(self._windows_for_video(path, nals))
        return samples, nal_index

    def _windows_for_video(self, path: Path, nals: list[NALUnit]) -> list[WindowSample]:
        """Non-overlapping windows, each beginning at an IDR boundary and packed
        forward by whole NALs until the byte budget is hit."""
        windows: list[WindowSample] = []
        n = len(nals)
        idr_positions = [k for k, nal in enumerate(nals) if nal.nal_type == 5]
        used_until = 0  # next window must start at or after this NAL index
        for k in idr_positions:
            # Back up to include the parameter sets immediately preceding the IDR
            # so the window is self-contained and decodable from its first byte.
            start = k
            while (
                start - 1 >= 0 and nals[start - 1].nal_type in PARAMETER_SET_NAL_TYPES
            ):
                start -= 1
            if start < used_until:
                continue  # this GOP is already inside a previously packed window
            total = 0
            vcl = 0
            end = start
            while end < n:
                nal_len = nals[end].end - nals[end].start
                if total + nal_len > self.max_seq_length:
                    break
                total += nal_len
                if nals[end].nal_type in VCL_NAL_TYPES:
                    vcl += 1
                end += 1
            if vcl >= self.min_frames:
                windows.append(WindowSample(path, start, end, vcl))
            used_until = max(used_until, end)
        return windows

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        data = sample.h264_path.read_bytes()
        nals = self.nal_index[str(sample.h264_path)]

        byte_chunks: list[Tensor] = []
        region_chunks: list[Tensor] = []
        offset_chunks: list[Tensor] = []
        for nal in nals[sample.start_nal : sample.end_nal]:
            length = nal.end - nal.start
            byte_chunks.append(bytes_to_ids(data[nal.start : nal.end]))
            region = (
                REGION_META
                if nal.nal_type in PARAMETER_SET_NAL_TYPES
                else REGION_TARGET
            )
            region_chunks.append(torch.full((length,), region, dtype=torch.long))
            offset_chunks.append(torch.arange(length, dtype=torch.long))

        window = torch.cat(byte_chunks)
        raw_region = torch.cat(region_chunks)
        raw_offset = torch.cat(offset_chunks)

        # Teacher forcing across the whole window. SLICE_BOS is reused as the
        # stream-start marker (this dataset never emits a slice-level BOS), so the
        # AR vocabulary is unchanged.
        bos = torch.tensor([SLICE_BOS_ID], dtype=torch.long)
        input_ids = torch.cat((bos, window[:-1]))
        labels = window.clone()
        region_ids = torch.cat(
            (torch.tensor([REGION_TARGET], dtype=torch.long), raw_region[:-1])
        )
        offset_ids = torch.cat((torch.tensor([0], dtype=torch.long), raw_offset[:-1]))
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
                "task": "ar",
                "fim_format": "stream",
                "h264_path": str(sample.h264_path),
                "start_nal": sample.start_nal,
                "end_nal": sample.end_nal,
                "num_frames": sample.num_frames,
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
    nal_index_path: Path | None = (
        None  # Defaults to manifest_dir/nal_index.sqlite when present.
    )

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

        def _build_dataset(rows_subset: list[dict[str, Any]]) -> Dataset:
            if self.config.dataset_mode == "window":
                return ByteStreamWindowDataset(
                    rows_subset,
                    max_seq_length=self.max_seq_length,
                    min_frames=self.config.window_min_frames,
                    p_fim=self.config.p_fim,
                    nal_index=nal_index,
                    seed=self.config.seed,
                )
            return ByteSliceDataset(
                rows_subset,
                max_seq_length=self.max_seq_length,
                p_fim=self.config.p_fim,
                fim_format=self.config.fim_format,
                use_eos=self.config.use_eos,
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

        if self.config.split_by_video:
            # Group rows by source video, then split videos into train/val so
            # no video contributes slices to both partitions. This produces a
            # genuinely held-out video evaluation set, eliminating within-video
            # leakage that slice-level random_split allows.
            video_to_rows: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                video_to_rows.setdefault(row["h264_path"], []).append(row)
            video_ids = sorted(video_to_rows.keys())
            generator = torch.Generator().manual_seed(self.config.seed)
            perm = torch.randperm(len(video_ids), generator=generator).tolist()
            n_val = max(1, int(len(video_ids) * self.config.val_fraction))
            if len(video_ids) - n_val <= 0:
                raise ValueError(
                    "Need at least two source videos for video-level split"
                )
            val_video_ids = {video_ids[i] for i in perm[:n_val]}
            train_rows = [r for r in rows if r["h264_path"] not in val_video_ids]
            val_rows = [r for r in rows if r["h264_path"] in val_video_ids]
            self.train_dataset = _build_dataset(train_rows)
            self.val_dataset = _build_dataset(val_rows)
        else:
            dataset = _build_dataset(rows)
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
