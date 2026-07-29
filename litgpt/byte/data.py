# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""Byte-domain datasets for H.264 language-model pretraining.

This data module bypasses text tokenization. Raw byte values map directly to
token ids 0..255, and a small number of control ids mark generation starts.

The training unit is a VCL NAL unit from an Annex-B H.264 stream. Since the
preprocessing config pins one slice per frame, one VCL NAL is the frame/slice
unit used by the byte model.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sqlite3
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Iterable
from typing import Any, Literal

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split

from litgpt.byte import h264_syntax as HS
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
FIM_LOSS_SCOPES = ("span", "full")
FIMLossScope = Literal["span", "full"]
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

    # Consecutive byte/control ids represented by one transformer position.
    # Dataset sample construction remains byte-native; collation converts the
    # causal prompt/target boundary into patches without leaking target bytes.
    byte_patch_size: int = 1
    p_fim: float = 0.0  # Probability of FIM sample; otherwise sample is AR.
    # Debug-only overfit mode. Normal training redraws a window-FIM hole on every
    # access. When true, seed+dataset-index selects one stable hole per train window
    # so the exact same examples can be replayed by the evaluator.
    fixed_fim_holes: bool = False
    # "bridge" is the original layout with one SPAN_BOS marker. "psm" uses
    # explicit Prefix-Suffix-Middle markers following code-model FIM practice.
    fim_format: FIMFormat = "bridge"
    # "span" supervises only the missing bytes and their optional EOS (the
    # original repair objective). "full" applies next-token loss across the
    # reordered FIM sequence, matching causal-FIM pretraining practice.
    fim_loss_scope: FIMLossScope = "span"
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
    # ByteStreamWindowDataset: the multi-frame contiguous-stream window, run as the
    # AR objective at p_fim=0 (H0, verifying the AVC-LM/JPEG-LM generation legacy)
    # and as masked-span infill at p_fim>0. AR and FIM share the window and differ
    # only in masking, so H0 is the p_fim=0 case of one pipeline. See 0616.md.
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
    """Each row contains a data entry. The attributes include:
    1. id 2. src_path 3. h264_path 4. source information 5. output setting 6.
    """
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

    def __init__(
        self, index_path: Path, file_ids: dict[str, int], cache_size: int = 128
    ) -> None:
        self._index_path = str(index_path)
        self._file_ids = file_ids  # path -> files.id (one small entry per file)
        self._cache: "OrderedDict[str, list[NALUnit]]" = OrderedDict()
        self._cache_size = cache_size
        self._conn: sqlite3.Connection | None = None
        self._pid: int | None = None

    def __getstate__(self) -> dict:
        # A sqlite3.Connection cannot be pickled (and must not cross processes), and
        # the cache is just a bounded materialization. Persist only what identifies the
        # index; the connection + cache are rebuilt lazily on first use. This keeps the
        # object picklable for checkpoint save (fabric pickles the training state) and
        # for spawn-based DataLoader workers.
        return {
            "_index_path": self._index_path,
            "_file_ids": self._file_ids,
            "_cache_size": self._cache_size,
        }

    def __setstate__(self, state: dict) -> None:
        self._index_path = state["_index_path"]
        self._file_ids = state["_file_ids"]
        self._cache_size = state["_cache_size"]
        self._cache = OrderedDict()
        self._conn = None
        self._pid = None

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
        rows = (
            self._connection()
            .execute(
                "SELECT start, end, start_code_len, nal_type FROM nals "
                "WHERE file_id = ? ORDER BY nal_index",
                (file_id,),
            )
            .fetchall()
        )
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


def _fim_training_labels(
    input_ids: Tensor,
    missing_tail: Tensor,
    *,
    loss_scope: FIMLossScope,
    ignore_index: int,
) -> Tensor:
    """Build aligned next-token targets for a reordered FIM sequence.

    ``input_ids`` is the complete serialized sequence with its final target token
    omitted. In span mode, only ``FIM_END -> missing -> EOS`` contributes loss. In
    full mode, every serialized transition contributes loss; the first serialized
    token remains context because there is no preceding BOS in the FIM prompt.
    """
    if loss_scope == "span":
        labels = torch.full_like(input_ids, ignore_index)
        labels[-missing_tail.numel() :] = missing_tail
        return labels
    if loss_scope != "full":
        raise ValueError(f"loss_scope must be one of {FIM_LOSS_SCOPES}")
    if input_ids.numel() == 0 or missing_tail.numel() == 0:
        raise ValueError("full-sequence FIM supervision requires a non-empty sequence")
    return torch.cat((input_ids[1:], missing_tail[-1:]))


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
        span labels = [-100 ..., B_miss]
        full labels = the next token at every position in the reordered sequence
    """

    def __init__(
        self,
        rows: list[dict[str, Any]],
        max_seq_length: int,
        p_fim: float = 0.0,
        fim_format: FIMFormat = "bridge",
        fim_loss_scope: FIMLossScope = "span",
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
        if fim_loss_scope not in FIM_LOSS_SCOPES:
            raise ValueError(f"fim_loss_scope must be one of {FIM_LOSS_SCOPES}")
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
        self.fim_loss_scope = fim_loss_scope
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
        labels = _fim_training_labels(
            input_ids,
            missing_tail,
            loss_scope=self.fim_loss_scope,
            ignore_index=self.ignore_index,
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
                "fim_loss_scope": self.fim_loss_scope,
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

    FIM layout (``p_fim > 0`` -- masked-span infill over the same window):
        A REAL frame f is chosen uniformly among the window's frames, the window is
        TRUNCATED at f's end (causality: real-time repair has no future frames), and
        a contiguous span is excised from f's interior:

        context   = window[:frame_lo]          # SPS/PPS + clean prior frames
        prefix    = window[frame_lo:split]     # f's received bytes before the hole
        missing   = window[split:split+gap]    # the excised span -- the only target
        orphan    = window[split+gap:frame_hi] # f's received bytes after the hole
        input_ids = context, FIM_BEGIN, prefix, FIM_HOLE, orphan, FIM_END,
                    missing_tail[:-1]
        span labels = ignore everywhere except the trailing missing_tail
        full labels = the next token at every position in the reordered sequence

    This mirrors ``ByteSliceDataset._build_fim_item`` (same markers, same
    prefix/missing/orphan decomposition and configurable loss scope); the
    differences are that the multi-frame window replaces the single reference slice
    as context, and the hole is placed within a real frame rather than within a
    single NAL. "Multi-frame" describes the CONTEXT, not the objective -- AR is the
    multi-frame objective, FIM is a single-span objective over a multi-frame context
    (0616.md).

    The hole shape follows BSCV's corruption operator (Tian et al., `corrupt_Gen.py`;
    see 04 - projects/.../corruption-vs-bscv.md, which designates it the training
    generator): a contiguous interior byte EXCISION at a uniformly random offset, one
    hole per damaged frame, with the frame chosen uniformly (BSCV's
    ``random.sample(frameIndexes_in_GOP, corr_prob)``) rather than pinned to the last
    frame -- pinning it would train every sample at maximal conditioning depth, while
    deployment loses packets at arbitrary GOP depth.

    CAVEAT -- this corpus cannot reproduce BSCV's artifact. BSCV excises 1024-2048 B
    from inside ONE slice payload with the NAL header intact, which presumes
    one-slice-per-frame. Under AVC-LM's ``slice-max-mbs=1`` a VCL NAL is a single
    ~10-byte macroblock, so a hole of that size necessarily spans ~100+ whole NALs
    and their start codes. The survivors still parse (``first_mb_in_slice`` simply
    jumps) and the decoder conceals the gap, so the "present but desynced slice" BSCV
    creates cannot occur -- per-MB slicing is an error-resilience mode, a desync
    firewall every ~10 bytes. Window FIM here is therefore a valid infill objective
    for measuring OBJECTIVE INTERFERENCE against the phase-1 AR baseline, but its
    reconstruction numbers are NOT a bitstream-repair task result. That requires a
    one-slice-per-frame corpus.
    """

    def __init__(
        self,
        rows: list[dict[str, Any]],
        max_seq_length: int,
        *,
        min_frames: int = 2,
        p_fim: float = 0.0,
        fim_format: FIMFormat = "psm",
        fim_loss_scope: FIMLossScope = "span",
        use_eos: bool = False,
        fim_min_gap: int = 64,
        fim_max_gap: int = 1400,
        frame_guard_bytes: int = 64,
        resample_fim: bool = False,
        nal_index: dict[str, list[NALUnit]] | None = None,
        seed: int = 42,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        if max_seq_length < 4:
            raise ValueError("max_seq_length must be at least 4")
        if min_frames < 1:
            raise ValueError("min_frames must be positive")
        if not 0.0 <= p_fim <= 1.0:
            raise ValueError("p_fim must be in [0, 1]")
        if fim_format not in FIM_FORMATS:
            raise ValueError(f"fim_format must be one of {FIM_FORMATS}")
        if fim_loss_scope not in FIM_LOSS_SCOPES:
            raise ValueError(f"fim_loss_scope must be one of {FIM_LOSS_SCOPES}")
        if fim_min_gap < 1 or fim_max_gap < fim_min_gap:
            raise ValueError("require 1 <= fim_min_gap <= fim_max_gap")
        if frame_guard_bytes < 0:
            raise ValueError("frame_guard_bytes must be non-negative")
        self.rows = rows
        self.max_seq_length = max_seq_length
        self.min_frames = min_frames
        self.p_fim = p_fim
        self.fim_format = fim_format
        self.fim_loss_scope = fim_loss_scope
        self.use_eos = use_eos
        self.fim_min_gap = fim_min_gap
        self.fim_max_gap = fim_max_gap
        self.frame_guard_bytes = frame_guard_bytes
        self.resample_fim = resample_fim
        # Indices pinned to a deterministic hole even when resample_fim is on. With a
        # window-level (within-video) split, train and val are random_split Subsets of
        # ONE dataset instance, so the val indices have to be named here -- there is no
        # second instance to configure. setup() fills this in.
        self.fixed_indices: frozenset[int] = frozenset()
        self.seed = seed
        self.ignore_index = ignore_index
        self.samples, self.nal_index = self._build_index(nal_index)
        if not self.samples:
            raise ValueError(
                "No usable stream windows found. Check max_seq_length, min_frames, "
                "and that the corpus contains IDR-anchored GOPs."
            )
        if self.p_fim > 0:
            self._assert_fim_reachable()

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
        # Window AR predicts one extra SEQ_EOS token when enabled. Reserve its
        # position here so collate never truncates that final supervised label.
        window_byte_budget = self.max_seq_length - (1 if self.use_eos else 0)
        n = len(nals)
        idr_positions = [k for k, nal in enumerate(nals) if nal.nal_type == 5]
        used_until = 0  # next window must start at or after this NAL index
        for k in idr_positions:
            # Back up to include the access unit's leading non-VCL NALs (SPS/PPS, plus
            # any SEI/AUD wedged between them and the IDR) so the window carries its
            # parameter sets and is self-contained/decodable from its first byte. Stop at
            # the previous VCL slice so we don't pull in an earlier frame. NOTE: an earlier
            # version stepped back only over PARAMETER_SET_NAL_TYPES; when an SEI sat
            # between the PPS and the IDR that stopped the backup at the SEI, so those
            # windows silently omitted SPS/PPS (baselines up to and including xl-avclm).
            start = k
            while start - 1 >= 0 and nals[start - 1].nal_type not in VCL_NAL_TYPES:
                start -= 1
            if start < used_until:
                continue  # this GOP is already inside a previously packed window
            total = 0
            vcl = 0
            end = start
            while end < n:
                nal_len = nals[end].end - nals[end].start
                if total + nal_len > window_byte_budget:
                    break
                total += nal_len
                if nals[end].nal_type in VCL_NAL_TYPES:
                    vcl += 1
                end += 1
            if vcl >= self.min_frames:
                windows.append(WindowSample(path, start, end, vcl))
            used_until = max(used_until, end)
        return windows

    def _frame_bounds(self, sample: WindowSample, data: bytes) -> list[tuple[int, int]]:
        """(lo, hi) byte offsets, within the window, of each REAL frame.

        A frame starts at a VCL NAL whose ``first_mb_in_slice == 0``. That check is
        HS.slice_first_mb -- the same primitive free_run_rollout and the standalone
        eval use to index frame boundaries, so the dataset can never disagree with
        them about what a frame is on a slice-max-mbs=1 corpus (one VCL NAL == one
        macroblock, so NAL boundaries are NOT frame boundaries). Do not reimplement.

        A NAL whose first_mb_in_slice cannot be read is treated as "not a frame
        start" rather than a desync: unlike free-run, these are ground-truth bytes,
        so an unreadable value means our parse is too weak, and the cost is only a
        missed hole candidate.
        """
        nals = self.nal_index[str(sample.h264_path)]
        starts: list[int] = []
        cursor = 0
        for nal in nals[sample.start_nal : sample.end_nal]:
            length = nal.end - nal.start
            if nal.nal_type in VCL_NAL_TYPES:
                payload = data[nal.start + nal.start_code_len : nal.end]
                if HS.slice_first_mb(payload) == 0:
                    starts.append(cursor)
            cursor += length
        if not starts:
            return []
        ends = starts[1:] + [cursor]
        return list(zip(starts, ends))

    def _fim_candidates(
        self, sample: WindowSample, data: bytes
    ) -> list[tuple[int, int]]:
        """Frames that can host a hole, as (frame_lo, frame_hi).

        Two filters. (1) The frame must hold ``frame_guard_bytes`` of untouchable
        head plus ``fim_min_gap`` of hole plus a byte -- the guard keeps the frame's
        first NAL (which carries the first_mb_in_slice == 0 that MAKES it a frame
        boundary) and the window's leading SPS/PPS out of the hole. (2) The truncated
        window plus marker/EOS overhead must fit max_seq_length: collate's
        pad_and_truncate cuts from the RIGHT, which would silently amputate the
        missing_tail and with it every supervised label in the sample.
        """
        overhead = self._fim_overhead()
        candidates = []
        for lo, hi in self._frame_bounds(sample, data):
            if hi + overhead > self.max_seq_length:
                continue
            if hi - (lo + self.frame_guard_bytes) - 1 >= self.fim_min_gap:
                candidates.append((lo, hi))
        return candidates

    def _fim_overhead(self) -> int:
        """Net length added by the FIM reordering, over a window cut at frame_hi.

        PSM inserts three markers (FIM_BEGIN, FIM_HOLE, FIM_END) but middle_in is
        [FIM_END, missing_tail[:-1]], so the teacher-forcing shift gives one byte
        back: net +2, not +3. Bridge inserts one marker and gives the same byte back:
        net 0. Matches ByteSliceDataset's `format_overhead = 2 if psm else 0`.
        """
        markers = 2 if self.fim_format == "psm" else 0
        return markers + (1 if self.use_eos else 0)

    def _assert_fim_reachable(self) -> None:
        """Fail loudly when p_fim > 0 but no window can host a hole.

        Without this the dataset degrades silently: __getitem__ falls through to the
        AR item, so the run trains pure AR while logging p_fim > 0 and reporting FIM
        metrics for an objective it never saw. That is not hypothetical -- it is what
        ByteSliceDataset does on this corpus, where a ~10-byte target NAL gives
        max_fim_gap == 1 against fim_min_gap == 64, so `fim_eligible` is False for
        every sample.
        """
        probe = self.samples[:: max(1, len(self.samples) // 32)][:32]
        windows_ok = 0
        frames_total = 0
        frames_ok = 0
        hole_fracs: list[float] = []
        for sample in probe:
            data = sample.h264_path.read_bytes()
            bounds = self._frame_bounds(sample, data)
            candidates = self._fim_candidates(sample, data)
            frames_total += len(bounds)
            frames_ok += len(candidates)
            windows_ok += bool(candidates)
            for lo, hi in candidates:
                usable = hi - (lo + self.frame_guard_bytes) - 1
                hole_fracs.append(min(self.fim_max_gap, usable) / max(hi - lo, 1))
        if windows_ok == 0:
            raise ValueError(
                f"p_fim={self.p_fim} but none of {len(probe)} probed windows can host "
                f"a hole (fim_min_gap={self.fim_min_gap}, "
                f"frame_guard_bytes={self.frame_guard_bytes}, "
                f"max_seq_length={self.max_seq_length}). Every sample would silently "
                "fall back to AR. Lower fim_min_gap, lower frame_guard_bytes, or "
                "check that the corpus has readable frame boundaries."
            )
        # A hole is placed in an ELIGIBLE frame, so an aggressive fim_min_gap does not
        # fail loudly -- it quietly narrows the hole to the corpus's largest frames
        # (the IDR and the fattest P-frames) and trains repair on those alone. Report
        # the eligible-frame fraction and the share of a frame the hole can reach so
        # that bias is visible at startup rather than inferred from results later.
        frac = frames_ok / max(frames_total, 1)
        mean_hole = sum(hole_fracs) / max(len(hole_fracs), 1)
        print(
            f"[window-fim] hole-eligible frames: {frames_ok}/{frames_total} "
            f"({frac:.0%}) over {len(probe)} probed windows | max hole covers "
            f"~{mean_hole:.0%} of its frame | gap=[{self.fim_min_gap},"
            f"{self.fim_max_gap}] guard={self.frame_guard_bytes}",
            flush=True,
        )
        if frac < 0.5:
            print(
                f"[window-fim] WARNING: {1 - frac:.0%} of frames are too small to host "
                f"fim_min_gap={self.fim_min_gap} (a frame needs > "
                f"{self.frame_guard_bytes + self.fim_min_gap + 1} bytes). Holes will "
                "concentrate in the largest frames, which biases what the FIM arm ever "
                "practises on.",
                flush=True,
            )

    def _window_tensors(
        self, sample: WindowSample, data: bytes
    ) -> tuple[Tensor, Tensor, Tensor]:
        """(bytes, region, offset) for the whole window, before AR/FIM shaping."""
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
        return (
            torch.cat(byte_chunks),
            torch.cat(region_chunks),
            torch.cat(offset_chunks),
        )

    def pin_fixed_indices(self, indices: Iterable[int]) -> None:
        """Pin these sample indices to a deterministic AR/FIM draw and hole (the val
        split), so val_loss_fim is measured on the same spans at every eval and is
        comparable across steps."""
        self.fixed_indices = frozenset(int(i) for i in indices)

    def _rng_for(self, idx: int) -> random.Random:
        """Per-sample RNG.

        Seeded by index (the default, and what ByteSliceDataset does) the hole is
        frozen for the whole run: `rng.random() < p_fim` returns the same answer for
        index idx every epoch, so p_fim stops being a per-visit coin flip and becomes
        a fixed PARTITION of windows -- half of them AR forever, half FIM forever with
        one hole each. Slice mode tolerates that because it has one sample per target
        NAL (thousands per video); a window dataset has ~one sample per video (one IDR
        per clip => one window), so the same pattern would (a) halve the AR arm
        relative to the phase-1 baseline it is supposed to be compared against, which
        confounds the interference measurement this run exists to make, and (b) turn
        FIM into a few hundred fixed (context, hole, answer) triples repeated ~1e5
        times, which a model that already memorises this corpus to tf_byte_acc 0.9998
        will simply look up.

        So training resamples per access: every window is AR on ~half its visits and
        gets a fresh hole on the others. The corpus split stays seeded (and is dumped
        to train_split.json), so what is lost is only the exact hole sequence.
        """
        if not self.resample_fim or idx in self.fixed_indices:
            return random.Random(self.seed + idx)
        return random.Random()  # fresh OS entropy per access, per worker

    def ar_item(self, idx: int) -> dict[str, Any]:
        """The AR view of a window, whatever p_fim is.

        For a caller that needs ``labels`` to BE the window's bytes. Under p_fim > 0
        ``__getitem__`` may hand back a FIM item, whose labels are IGNORE_INDEX
        everywhere outside the span and therefore not a byte string. The free-run
        probe is such a caller: it is an AR probe, and reading it through
        ``__getitem__`` made it fail with "bytes must be in range(0, 256)" as soon as
        FIM was switched on.

        Going through this method rather than skipping FIM samples also keeps the
        probe's clip pool independent of p_fim, so phase 2's free-run numbers are
        measured over the same windows as phase 1's and stay comparable.
        """
        sample = self.samples[idx]
        data = sample.h264_path.read_bytes()
        window, raw_region, raw_offset = self._window_tensors(sample, data)
        return self._build_ar_item(sample, window, raw_region, raw_offset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rng = self._rng_for(idx)
        sample = self.samples[idx]
        data = sample.h264_path.read_bytes()
        window, raw_region, raw_offset = self._window_tensors(sample, data)

        if self.p_fim > 0 and rng.random() < self.p_fim:
            candidates = self._fim_candidates(sample, data)
            if candidates:
                return self._build_fim_item(
                    sample, window, raw_region, raw_offset, candidates, rng
                )
        return self._build_ar_item(sample, window, raw_region, raw_offset)

    def _build_ar_item(
        self,
        sample: WindowSample,
        window: Tensor,
        raw_region: Tensor,
        raw_offset: Tensor,
    ) -> dict[str, Any]:
        # Teacher forcing across the whole window. SLICE_BOS is reused as the
        # stream-start marker (this dataset never emits a slice-level BOS), so the
        # AR vocabulary is unchanged. With EOS the final input is the last raw byte
        # and its target is SEQ_EOS; without EOS this reduces to the original
        # [BOS, window[:-1]] -> window construction.
        bos = torch.tensor([SLICE_BOS_ID], dtype=torch.long)
        target_tail = self._with_eos(window)
        input_ids = torch.cat((bos, target_tail[:-1]))
        labels = target_tail
        region_ids = torch.cat(
            (
                torch.tensor([REGION_TARGET], dtype=torch.long),
                raw_region if self.use_eos else raw_region[:-1],
            )
        )
        offset_ids = torch.cat(
            (
                torch.tensor([0], dtype=torch.long),
                raw_offset if self.use_eos else raw_offset[:-1],
            )
        )
        return self._pack_item(input_ids, labels, region_ids, offset_ids, sample, "ar")

    def _build_fim_item(
        self,
        sample: WindowSample,
        window: Tensor,
        raw_region: Tensor,
        raw_offset: Tensor,
        candidates: list[tuple[int, int]],
        rng: random.Random,
    ) -> dict[str, Any]:
        frame_lo, frame_hi = rng.choice(candidates)
        lo = frame_lo + self.frame_guard_bytes
        gap = rng.randint(self.fim_min_gap, min(self.fim_max_gap, frame_hi - lo - 1))
        split = rng.randint(lo, frame_hi - gap)

        # Everything after the hole's frame is dropped: real-time repair never has
        # future frames, so they must not reach the model as context (0616.md).
        context = window[:frame_lo]
        prefix = window[frame_lo:split]
        missing = window[split : split + gap]
        orphan = window[split + gap : frame_hi]
        missing_tail = self._with_eos(missing)

        if self.fim_format == "psm":
            head = torch.tensor([FIM_BEGIN_ID], dtype=torch.long)
            hole = torch.tensor([FIM_HOLE_ID], dtype=torch.long)
            middle_in = torch.cat(
                (torch.tensor([FIM_END_ID], dtype=torch.long), missing_tail[:-1])
            )
            pieces = [context, head, prefix, hole, orphan, middle_in]
            regions = [
                raw_region[:frame_lo],
                torch.full((1,), REGION_PREFIX, dtype=torch.long),
                torch.full((prefix.numel(),), REGION_PREFIX, dtype=torch.long),
                torch.full((1,), REGION_ORPHAN, dtype=torch.long),
                torch.full((orphan.numel(),), REGION_ORPHAN, dtype=torch.long),
                torch.full((middle_in.numel(),), REGION_BRIDGE, dtype=torch.long),
            ]
        else:
            middle_in = torch.cat(
                (torch.tensor([SPAN_BOS_ID], dtype=torch.long), missing_tail[:-1])
            )
            pieces = [context, prefix, orphan, middle_in]
            regions = [
                raw_region[:frame_lo],
                torch.full((prefix.numel(),), REGION_PREFIX, dtype=torch.long),
                torch.full((orphan.numel(),), REGION_ORPHAN, dtype=torch.long),
                torch.full((middle_in.numel(),), REGION_BRIDGE, dtype=torch.long),
            ]

        input_ids = torch.cat(pieces)
        region_ids = torch.cat(regions)
        # Offsets are the window-AR convention (arange within each NAL, reset at NAL
        # boundaries) for received bytes, and continue from the hole's window offset
        # across the generated span. A multi-NAL hole has no single within-NAL
        # position, so this is a placeholder that is only coherent while offset ids
        # are DISABLED -- which is why scripts/byte/train.py rejects window FIM
        # without --no-offset-id. Turning them back on needs a real design (260703
        # found encodings help, so someone will want to).
        offset_ids = torch.cat(
            (
                raw_offset[:frame_lo],
                torch.zeros(1, dtype=torch.long),
                raw_offset[frame_lo:split],
                torch.zeros(1, dtype=torch.long),
                raw_offset[split + gap : frame_hi],
                torch.arange(middle_in.numel(), dtype=torch.long),
            )
        )
        labels = _fim_training_labels(
            input_ids,
            missing_tail,
            loss_scope=self.fim_loss_scope,
            ignore_index=self.ignore_index,
        )
        return self._pack_item(
            input_ids,
            labels,
            region_ids,
            offset_ids,
            sample,
            "fim",
            fim_gap=gap,
            fim_split=split,
            frame_lo=frame_lo,
            frame_hi=frame_hi,
        )

    def _with_eos(self, content: Tensor) -> Tensor:
        if not self.use_eos:
            return content
        return torch.cat((content, torch.tensor([SEQ_EOS_ID], dtype=torch.long)))

    def _pack_item(
        self,
        input_ids: Tensor,
        labels: Tensor,
        region_ids: Tensor,
        offset_ids: Tensor,
        sample: WindowSample,
        task: TaskName,
        **fim_meta: int,
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
                "fim_format": self.fim_format if task == "fim" else "stream",
                "fim_loss_scope": self.fim_loss_scope,
                "h264_path": str(sample.h264_path),
                "start_nal": sample.start_nal,
                "end_nal": sample.end_nal,
                "num_frames": sample.num_frames,
                **fim_meta,
            },
        }


def collate_byte_samples(
    samples: list[dict[str, Any]],
    max_seq_length: int,
    byte_patch_size: int = 1,
) -> dict[str, Tensor | dict[str, Tensor]]:
    if byte_patch_size < 1:
        raise ValueError("byte_patch_size must be positive")
    if byte_patch_size > 1:
        samples = [
            patch_byte_sample(sample, byte_patch_size) for sample in samples
        ]
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
        **(
            {
                "target_region_ids": pad_and_truncate(
                    [sample["target_region_ids"] for sample in samples],
                    max_seq_length,
                    REGION_PAD,
                )
            }
            if byte_patch_size > 1
            else {}
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
            "transformer_positions": torch.tensor(
                [sample["input_ids"].size(0) for sample in samples],
                dtype=torch.int64,
            ).unsqueeze(1),
        },
    }


def _pad_patch_axis(
    tensor: Tensor, patch_size: int, value: int, *, left: bool
) -> Tensor:
    """Pad a one-dimensional tensor to K and return shape ``(positions, K)``."""
    pad = (-tensor.numel()) % patch_size
    if pad:
        fill = torch.full(
            (pad,), value, dtype=tensor.dtype, device=tensor.device
        )
        tensor = torch.cat((fill, tensor)) if left else torch.cat((tensor, fill))
    return tensor.view(-1, patch_size)


def patch_byte_sample(sample: dict[str, Any], patch_size: int) -> dict[str, Any]:
    """Convert a shifted byte-LM sample into causal next-patch supervision.

    The existing byte datasets use ``input[s] -> label[s]`` at the first
    supervised position and then teacher-force with
    ``input[s+1:] == label[s:-1]``. Merely reshaping those tensors would expose
    target bytes 0..K-2 while predicting them. Instead, this function:

      * left-pads and patches the complete known prompt ending at ``input[s]``;
      * right-pads the supervised target into patches;
      * lets the final prompt patch predict target patch 0;
      * uses each completed target patch as the next transformer input.

    Bytes within a target patch remain causal in ``GPT``'s shared MEGABYTE
    local Transformer. The construction supports window AR, slice AR, and
    trailing-target FIM.
    """
    if patch_size < 2:
        raise ValueError("patch_byte_sample requires patch_size >= 2")
    input_ids = sample["input_ids"]
    labels = sample["labels"]
    region_ids = sample["region_ids"]
    offset_ids = sample["offset_ids"]
    supervised = (labels != IGNORE_INDEX).nonzero(as_tuple=False).flatten()
    if supervised.numel() == 0:
        raise ValueError("byte sample has no supervised target")
    first = int(supervised[0])
    last = int(supervised[-1])
    expected = torch.arange(first, last + 1, device=supervised.device)
    if not torch.equal(supervised, expected):
        raise ValueError("patched-byte training requires one contiguous target tail")
    if last != labels.numel() - 1:
        raise ValueError("patched-byte training requires supervision through sample end")
    if last > first and not torch.equal(
        input_ids[first + 1 : last + 1], labels[first:last]
    ):
        raise ValueError("byte sample teacher-forcing shift is inconsistent")

    prompt_end = first + 1
    prompt_ids = _pad_patch_axis(
        input_ids[:prompt_end], patch_size, PAD_ID, left=True
    )
    prompt_regions = _pad_patch_axis(
        region_ids[:prompt_end], patch_size, REGION_PAD, left=True
    )
    prompt_offsets = _pad_patch_axis(
        offset_ids[:prompt_end], patch_size, 0, left=True
    )

    target_labels = _pad_patch_axis(
        labels[first:], patch_size, IGNORE_INDEX, left=False
    )
    target_regions = _pad_patch_axis(
        region_ids[first:], patch_size, REGION_PAD, left=False
    )
    target_inputs = target_labels.clamp_min(0)

    # A completed target byte later becomes an input byte. Its input-side
    # auxiliary ids therefore come from the *next* position in the original
    # shifted sample, not from the position whose label predicted it.
    num_reused_target_bytes = (target_labels.size(0) - 1) * patch_size
    reused_regions = region_ids[
        first + 1 : first + 1 + num_reused_target_bytes
    ].view(-1, patch_size)
    reused_offsets = offset_ids[
        first + 1 : first + 1 + num_reused_target_bytes
    ].view(-1, patch_size)

    input_patches = torch.cat((prompt_ids, target_inputs[:-1]), dim=0)
    input_regions = torch.cat((prompt_regions, reused_regions), dim=0)
    input_offsets = torch.cat((prompt_offsets, reused_offsets), dim=0)
    ignored_prompt = torch.full(
        (prompt_ids.size(0) - 1, patch_size),
        IGNORE_INDEX,
        dtype=labels.dtype,
        device=labels.device,
    )
    ignored_regions = torch.full(
        (prompt_ids.size(0) - 1, patch_size),
        REGION_PAD,
        dtype=region_ids.dtype,
        device=region_ids.device,
    )
    output_labels = torch.cat((ignored_prompt, target_labels), dim=0)
    output_regions = torch.cat((ignored_regions, target_regions), dim=0)

    if input_patches.shape != output_labels.shape:
        raise RuntimeError("patched input/target position counts disagree")
    return {
        **sample,
        "input_ids": input_patches,
        "labels": output_labels,
        "region_ids": input_regions,
        "offset_ids": input_offsets,
        "target_region_ids": output_regions,
        "token_counts": {
            **sample["token_counts"],
            "transformer_positions": int(input_patches.size(0)),
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
    train_split_dump_path: Path | None = (
        None  # If set, setup() writes the exact train split (videos + windows) here so a
        # later training-set eval can target precisely what was trained.
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
        if self.config.byte_patch_size < 1:
            raise ValueError("byte_patch_size must be positive")
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

        # ``self.max_seq_length`` is measured in transformer positions. Leave
        # one raw-byte slot of headroom because a conditioned AR/FIM sample has
        # a separately rounded prompt and target patch boundary.
        byte_max_seq_length = (
            self.max_seq_length
            if self.config.byte_patch_size == 1
            else self.max_seq_length * self.config.byte_patch_size - 1
        )

        def _build_dataset(rows_subset: list[dict[str, Any]]) -> Dataset:
            if self.config.dataset_mode == "window":
                return ByteStreamWindowDataset(
                    rows_subset,
                    max_seq_length=byte_max_seq_length,
                    min_frames=self.config.window_min_frames,
                    p_fim=self.config.p_fim,
                    fim_format=self.config.fim_format,
                    fim_loss_scope=self.config.fim_loss_scope,
                    use_eos=self.config.use_eos,
                    fim_min_gap=self.config.fim_min_gap,
                    fim_max_gap=self.config.fim_max_gap,
                    frame_guard_bytes=self.config.slice_header_guard_bytes,
                    nal_index=nal_index,
                    seed=self.config.seed,
                )
            return ByteSliceDataset(
                rows_subset,
                max_seq_length=byte_max_seq_length,
                p_fim=self.config.p_fim,
                fim_format=self.config.fim_format,
                fim_loss_scope=self.config.fim_loss_scope,
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
            # Separate instances here, so train resamples and val simply does not.
            if isinstance(self.train_dataset, ByteStreamWindowDataset):
                self.train_dataset.resample_fim = (
                    self.config.p_fim > 0 and not self.config.fixed_fim_holes
                )
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
            # One shared instance: turn resampling on, then pin the val indices back to
            # deterministic holes so val_loss_fim is the same spans at every eval.
            if isinstance(dataset, ByteStreamWindowDataset) and self.config.p_fim > 0:
                dataset.resample_fim = not self.config.fixed_fim_holes
                dataset.pin_fixed_indices(self.val_dataset.indices)

        if self.train_split_dump_path is not None:
            self._dump_train_split()

    def _dump_train_split(self) -> None:
        """Write the EXACT train split -- train videos and (window mode) their
        (h264_path, start_nal, end_nal) windows -- so a training-set eval can target
        precisely what was trained, regardless of the (seeded) train/val split. Atomic
        write; safe to call once per rank (identical content)."""
        ds = self.train_dataset
        if hasattr(ds, "indices") and hasattr(ds, "dataset"):  # torch Subset (window split)
            base = ds.dataset
            sample_indices = [int(i) for i in ds.indices]
            samples = [base.samples[i] for i in sample_indices]
        else:
            base = ds
            samples = list(getattr(ds, "samples", []))
            sample_indices = list(range(len(samples)))
        windows = [
            {"h264_path": str(s.h264_path), "start_nal": s.start_nal, "end_nal": s.end_nal}
            for s in samples
            if hasattr(s, "start_nal")
        ]
        videos = sorted({str(s.h264_path) for s in samples if hasattr(s, "h264_path")})
        fixed_holes: list[dict[str, Any]] = []
        if self.config.fixed_fim_holes:
            if not isinstance(base, ByteStreamWindowDataset):
                raise RuntimeError("fixed FIM holes require ByteStreamWindowDataset")
            for index in sample_indices:
                item = base[index]
                meta = item["sample_meta"]
                if meta.get("task") != "fim":
                    raise RuntimeError(
                        "fixed FIM hole recording expected every train item to be FIM; "
                        "use p_fim=1.0"
                    )
                labels = item["labels"]
                gap = int(meta["fim_gap"])
                eos_tokens = 1 if self.config.use_eos else 0
                missing_targets = labels[-(gap + eos_tokens) :]
                target = bytes(
                    int(token)
                    for token in missing_targets.tolist()
                    if 0 <= int(token) < BYTE_VOCAB_SIZE
                )
                fixed_holes.append(
                    {
                        "dataset_index": index,
                        "h264_path": str(meta["h264_path"]),
                        "start_nal": int(meta["start_nal"]),
                        "end_nal": int(meta["end_nal"]),
                        "frame_lo": int(meta["frame_lo"]),
                        "frame_hi": int(meta["frame_hi"]),
                        "fim_split": int(meta["fim_split"]),
                        "fim_gap": int(meta["fim_gap"]),
                        "target_length": len(target),
                        "target_sha256": hashlib.sha256(target).hexdigest(),
                    }
                )
        payload = {
            "manifest": str(self.manifest_path),
            "max_manifest_rows": self.max_manifest_rows,
            "byte_patch_size": self.config.byte_patch_size,
            "transformer_block_size": self.max_seq_length,
            "raw_byte_budget": (
                self.max_seq_length
                if self.config.byte_patch_size == 1
                else self.max_seq_length * self.config.byte_patch_size - 1
            ),
            "split_by_video": self.config.split_by_video,
            "val_fraction": self.config.val_fraction,
            "seed": self.config.seed,
            "dataset_mode": self.config.dataset_mode,
            "p_fim": self.config.p_fim,
            "fim_format": self.config.fim_format,
            "fim_loss_scope": self.config.fim_loss_scope,
            "use_eos": self.config.use_eos,
            "fixed_fim_holes_enabled": self.config.fixed_fim_holes,
            "n_train_videos": len(videos),
            "n_train_windows": len(windows),
            "videos": videos,
            "windows": windows,
            "fixed_fim_holes": fixed_holes,
        }
        path = Path(self.train_split_dump_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)  # atomic

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
        return collate_byte_samples(
            samples,
            max_seq_length=self.max_seq_length,
            byte_patch_size=self.config.byte_patch_size,
        )
