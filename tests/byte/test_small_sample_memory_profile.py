import torch

from litgpt.byte.small_sample_memory_profile import SmallSampleMemoryProfiler


def test_sample_without_optional_pynvml(monkeypatch):
    profiler = SmallSampleMemoryProfiler.__new__(SmallSampleMemoryProfiler)
    profiler.microbatch = 1
    profiler.phases = []

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 2_000_000_000)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 3_000_000_000)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    def missing_pynvml(_device):
        raise ModuleNotFoundError("pynvml does not seem to be installed")

    monkeypatch.setattr(torch.cuda, "device_memory_used", missing_pynvml)

    profiler._sample("before_first_forward")

    assert profiler.phases == [
        {
            "phase": "before_first_forward",
            "microbatch": 1,
            "allocated_gb": 2.0,
            "reserved_gb": 3.0,
        }
    ]
