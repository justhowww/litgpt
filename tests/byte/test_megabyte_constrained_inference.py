from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from litgpt.byte.data import REGION_BRIDGE, SEQ_EOS_ID, SLICE_BOS_ID
from scripts.byte.eval import eval_fim_avclm as FIM


class _ScriptedMegabyte(nn.Module):
    """Minimal global/local model whose raw argmax is deliberately illegal."""

    def __init__(self, target: bytes, patch_size: int = 4) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.target = target
        self.max_seq_length = 16
        self.config = SimpleNamespace(
            byte_patch_size=patch_size,
            vocab_size=SEQ_EOS_ID + 1,
        )
        self._prompt_last_position: int | None = None

    def set_kv_cache(self, **kwargs) -> None:
        pass

    def clear_kv_cache(self) -> None:
        pass

    def megabyte_global_forward(
        self,
        idx,
        input_pos=None,
        input_pos_maxp1=None,
        region_ids=None,
        offset_ids=None,
    ):
        position = int(input_pos.reshape(-1)[-1])
        if self._prompt_last_position is None:
            self._prompt_last_position = position
        patch_index = position - self._prompt_last_position
        output = torch.zeros(
            idx.size(0), idx.size(1), 4, device=idx.device
        )
        output[:, -1, 0] = patch_index * self.config.byte_patch_size
        return output

    def megabyte_local_next_logits(self, global_output, previous_bytes):
        byte_index = int(global_output[0, 0]) + previous_bytes.size(1)
        logits = torch.full(
            (1, SEQ_EOS_ID + 1), -20.0, device=global_output.device
        )
        if byte_index < len(self.target):
            logits[0, 255] = 10.0  # raw argmax: forbidden by the fake codec
            logits[0, self.target[byte_index]] = 9.0
            logits[0, SEQ_EOS_ID] = 11.0
        else:
            logits[0, 0] = 1.0
            logits[0, SEQ_EOS_ID] = 12.0
        return logits


class _ParserState:
    def __init__(self, target: bytes) -> None:
        self.target = target
        self.committed: list[int] = []
        self.failure_reason = None
        self.mask_calls = 0
        self.strict_mask_calls = 0
        self.permissive_mask_calls = 0
        self.nal_index = 0
        self.automaton = None
        self.expect_nal_header = False


def _sample(target: bytes) -> FIM.WindowFimSample:
    prompt = torch.tensor([SLICE_BOS_ID, 7], dtype=torch.long)
    regions = torch.full_like(prompt, REGION_BRIDGE)
    offsets = torch.arange(prompt.numel())
    return FIM.WindowFimSample(
        sample_index=0,
        h264_path=Path("synthetic.h264"),
        start_nal=0,
        end_nal=0,
        frame_lo=0,
        frame_hi=len(target),
        split=0,
        gap=len(target),
        prompt_ids=prompt,
        prompt_region_ids=regions,
        prompt_offset_ids=offsets,
        teacher_input_ids=prompt,
        teacher_region_ids=regions,
        teacher_offset_ids=offsets,
        teacher_labels=torch.full_like(prompt, -100),
        target_bytes=target,
        window_bytes=target,
    )


def _install_fake_codec(monkeypatch, target: bytes) -> _ParserState:
    state = _ParserState(target)
    monkeypatch.setattr(
        FIM,
        "_seed_parser_state",
        lambda sample, slice_max_mbs: state,
    )

    def valid_mask(parser):
        parser.mask_calls += 1
        parser.strict_mask_calls += 1
        mask = [False] * 256
        if len(parser.committed) < len(parser.target):
            mask[parser.target[len(parser.committed)]] = True
        else:
            mask[0] = True
        return mask

    monkeypatch.setattr(FIM.HM, "get_valid_byte_mask", valid_mask)
    monkeypatch.setattr(
        FIM.HM, "advance", lambda parser, byte: parser.committed.append(byte)
    )
    monkeypatch.setattr(
        FIM.HM,
        "can_append_bytes",
        lambda parser, suffix, require_complete: (
            bytes(parser.committed) == parser.target
        ),
    )
    return state


def test_megabyte_mask_is_applied_before_every_local_byte(monkeypatch):
    target = b"\x01\x02\x03\x04\x05"
    state = _install_fake_codec(monkeypatch, target)

    result = FIM.generate_span(
        _ScriptedMegabyte(target),
        _sample(target),
        torch.device("cpu"),
        stop_mode="parser_reconnect",
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        max_gen_bytes=32,
        mask_illegal_bytes=True,
        slice_max_mbs=1,
    )

    assert result is not None
    assert result.data == target
    assert result.stop_reason == "parser_reconnect"
    assert result.parser_reconnect_found
    assert result.mask_argmax_rejected == len(target)
    assert state.mask_calls == len(target)


def test_megabyte_eos_waits_until_suffix_can_reconnect(monkeypatch):
    target = b"\x11\x12\x13\x14\x15"
    _install_fake_codec(monkeypatch, target)

    result = FIM.generate_span(
        _ScriptedMegabyte(target),
        _sample(target),
        torch.device("cpu"),
        stop_mode="learned_eos",
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        max_gen_bytes=32,
        mask_illegal_bytes=True,
        slice_max_mbs=1,
    )

    assert result is not None
    assert result.data == target
    assert result.stop_reason == "eos"
    assert result.eos_stopped
    assert result.mask_eos_blocked == len(target)
