# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock
from unittest.mock import ANY, Mock

import pytest
import torch
from lightning.fabric.strategies import FSDPStrategy, SingleDeviceStrategy
from torch.utils.data import DataLoader

from litgpt import pretrain
from litgpt.args import EvalArgs, TrainArgs
from litgpt.byte.grpo import GRPOConfig
from litgpt.config import Config
from litgpt.pretrain import initialize_weights
from litgpt.utils import _RunIf


def test_optimizer_schedule_is_world_size_independent():
    block_size = 16384
    global_batch_size = 64
    requested_steps = 160000
    train = TrainArgs(
        global_batch_size=global_batch_size,
        micro_batch_size=1,
        max_tokens=requested_steps * global_batch_size * block_size,
        max_norm=1.0,
        lr_warmup_steps=3200,
    )
    dataloader = [None] * 38

    one_gpu = pretrain._optimizer_step_schedule(
        train,
        block_size,
        devices=1,
        num_nodes=1,
        max_iters=requested_steps * 64,
        train_dataloader=dataloader,
    )
    four_gpu = pretrain._optimizer_step_schedule(
        train,
        block_size,
        devices=4,
        num_nodes=1,
        max_iters=requested_steps * 16,
        train_dataloader=dataloader,
    )

    assert one_gpu == (64, requested_steps, 3200)
    assert four_gpu == (16, requested_steps, 3200)


def test_multi_gpu_grpo_selects_replicated_ddp():
    grpo = GRPOConfig(interval=10, group_size=2)

    assert pretrain._select_training_strategy(4, 1, grpo) == "ddp"
    assert pretrain._select_training_strategy(1, 1, grpo) == "auto"
    assert isinstance(
        pretrain._select_training_strategy(4, 1, None), FSDPStrategy
    )


def test_resume_rebases_microiterations_from_optimizer_step():
    state = {"iter_num": 1_280_000, "step_count": 20_000}

    resumed, rebased = pretrain._rebase_microiteration_counter(
        state, gradient_accumulation_iters=16
    )

    assert resumed == 1_280_000
    assert rebased == 320_000
    assert state == {"iter_num": 320_000, "step_count": 20_000}


@_RunIf(min_cuda_gpus=1, standalone=True)
@mock.patch("litgpt.pretrain.save_hyperparameters")
def test_optimizer_args(_, tmp_path):
    model_config = Config(block_size=2, n_layer=2, n_embd=4, n_head=2, padded_vocab_size=8)

    dataset = torch.tensor([[0, 1, 2], [3, 4, 5], [0, 1, 2]])
    dataloader = DataLoader(dataset)
    pretrain.get_dataloaders = Mock(return_value=(dataloader, dataloader))

    for i in ("AdamW", "SGD", "RMSprop"):
        pretrain.setup(
            "pythia-14m",
            devices=1,
            optimizer="RMSprop",
            model_config=model_config,
            out_dir=tmp_path,
            train=TrainArgs(global_batch_size=2, max_tokens=16, save_interval=1, micro_batch_size=1, max_norm=1.0),
            eval=EvalArgs(interval=1, max_iters=1, final_validation=False),
        )


@_RunIf(min_cuda_gpus=2, standalone=True)
# If we were to use `save_hyperparameters()`, we would have to patch `sys.argv` or otherwise
# the CLI would capture pytest args, but unfortunately patching would mess with subprocess
# launching, so we need to mock `save_hyperparameters()`
@mock.patch("litgpt.pretrain.save_hyperparameters")
# todo: it expects exactly 2 GPUs and has strange failing for validated 4 # GPUs, so we temporarily mark it as xfail
@pytest.mark.xfail(condition=torch.cuda.device_count() != 2, reason="This test is flaky, expects exactly 2 GPUs")
def test_pretrain(_, tmp_path):
    model_config = Config(block_size=2, n_layer=2, n_embd=8, n_head=4, padded_vocab_size=8)

    dataset = torch.tensor([[0, 1, 2], [3, 4, 5], [0, 1, 2]])
    dataloader = DataLoader(dataset)
    pretrain.get_dataloaders = Mock(return_value=(dataloader, dataloader))

    out_dir = tmp_path / "out"
    stdout = StringIO()
    with redirect_stdout(stdout):
        pretrain.setup(
            "pythia-14m",
            devices=2,
            model_config=model_config,
            out_dir=out_dir,
            train=TrainArgs(global_batch_size=2, max_tokens=16, save_interval=1, micro_batch_size=1, max_norm=1.0),
            eval=EvalArgs(interval=1, max_iters=1, final_validation=False),
        )

    if torch.distributed.get_rank() == 0:
        # tmp_path is not the same across all ranks, run assert only on rank 0
        out_dir_contents = set(os.listdir(out_dir))
        checkpoint_dirs = {"step-00000001", "step-00000002", "step-00000003", "step-00000004", "final"}
        assert checkpoint_dirs.issubset(out_dir_contents)
        assert all((out_dir / p).is_dir() for p in checkpoint_dirs)
        for checkpoint_dir in checkpoint_dirs:
            # the `tokenizer_dir` is None by default, so only 'lit_model.pth' shows here
            assert set(os.listdir(out_dir / checkpoint_dir)) == {"lit_model.pth", "model_config.yaml"}

        assert (out_dir / "logs" / "tensorboard" / "version_0").is_dir()

        # logs only appear on rank 0
        logs = stdout.getvalue()
        assert logs.count("(step)") == 4
        assert logs.count("val loss") == 4
        assert "Total parameters: 1,888" in logs

    torch.distributed.barrier()


@_RunIf(min_cuda_gpus=2, standalone=True)
@mock.patch("litgpt.pretrain.L.Fabric.load")
# See comment in `test_pretrain` why we need to mock `save_hyperparameters()`
@mock.patch("litgpt.pretrain.save_hyperparameters")
def test_initial_checkpoint_dir(_, load_mock, tmp_path):
    model_config = Config(block_size=2, n_layer=2, n_embd=8, n_head=4, padded_vocab_size=8)

    dataset = torch.tensor([[0, 1, 2], [3, 4, 5], [0, 1, 2]])
    dataloader = DataLoader(dataset)
    pretrain.get_dataloaders = Mock(return_value=(dataloader, dataloader))
    pretrain.fit = Mock()

    pretrain.setup(
        "pythia-14m",
        initial_checkpoint_dir=tmp_path,
        devices=torch.cuda.device_count(),
        model_config=model_config,
        out_dir=tmp_path,
    )

    load_mock.assert_called_once_with(
        tmp_path / "lit_model.pth", {"model": ANY}, weights_only=False
    )


@pytest.mark.parametrize(("strategy", "expected"), [(SingleDeviceStrategy, True), (FSDPStrategy, False)])
def test_initialize_weights(strategy, expected):
    fabric_mock = Mock()
    fabric_mock.strategy = Mock(spec=strategy)

    class Child(torch.nn.Module):
        pass

    class Parent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.child = Child()

    model = Parent()
    model.reset_parameters = Mock()
    model.child.reset_parameters = Mock()

    initialize_weights(fabric_mock, model, n_layer=2, n_embd=8)
    assert model.reset_parameters.call_count == int(expected)
    assert model.child.reset_parameters.call_count == int(expected)
