# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import math
import os
import pprint
import time
import warnings
from dataclasses import asdict
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Literal

import lightning as L
import torch
import torch.nn as nn
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities.throughput import ThroughputMonitor, measure_flops
from torch.utils.data import DataLoader
from torchmetrics.aggregation import RunningMean

from litgpt import Tokenizer
from litgpt.args import EvalArgs, LogArgs, TrainArgs
from litgpt.byte.free_run_eval import FreeRunEvalConfig
from litgpt.byte.grpo import GRPOConfig
from litgpt.byte.mrt import MRTConfig
from litgpt.config import name_to_config
from litgpt.constants import _TORCH_EQUAL_2_7, _TORCH_EQUAL_2_8
from litgpt.data import DataModule, TinyLlama
from litgpt.byte.reconstruction import (
    ReconstructionEvalConfig,
)
from litgpt.byte.training import (
    ByteTrainingRuntime,
    balanced_eos_auxiliary_loss,
    byte_training_loss,
    byte_weighted_cross_entropy,
    get_model_inputs_and_targets,
    namespace_reconstruction_metrics,
    validate,
)
from litgpt.model import GPT, Block, CausalSelfAttention, Config, LLaMAMLP
from litgpt.parser_config import save_hyperparameters
from litgpt.types import LoggerChoice
from litgpt.utils import (
    CycleIterator,
    capture_hparams,
    check_nvlink_connectivity,
    choose_logger,
    chunked_cross_entropy,
    copy_config_files,
    extend_checkpoint_dir,
    find_resume_path,
    get_default_supported_precision,
    init_out_dir,
    instantiate_torch_optimizer,
    num_parameters,
    parse_devices,
    reset_parameters,
    save_config,
)


def _select_training_strategy(
    devices: int,
    num_nodes: int,
    grpo: GRPOConfig | None,
):
    """Choose replicated DDP for GRPO and FSDP for ordinary pretraining.

    Byte GRPO unwraps the policy for autoregressive rollout generation.  That
    is safe for DDP because every rank owns a complete replica, but not for an
    FSDP model whose parameters are sharded.  The gradient-bearing candidate
    scoring pass still uses the DDP wrapper and therefore synchronizes updates.
    """
    if devices * num_nodes <= 1:
        return "auto"
    if grpo is not None and grpo.enabled:
        return "ddp"
    return FSDPStrategy(
        auto_wrap_policy={Block},
        state_dict_type="full",
        sharding_strategy="HYBRID_SHARD",
    )


def _freeze_inactive_megabyte_ddp_parameters(
    model: GPT,
    *,
    world_size: int,
    grpo: GRPOConfig | None,
) -> None:
    """Exclude permanently dormant MEGABYTE weights from GRPO's DDP reducer.

    Patched-byte models use ``megabyte_global_wte`` for global input and the
    tied ``megabyte_local.wte`` matrix for local input/output.  The ordinary
    byte ``transformer.wte`` and ``lm_head`` remain in checkpoints for format
    compatibility but never participate in a patch-8 forward.  DDP otherwise
    expects gradients for them and fails at the next forward.

    Scope this to multi-rank GRPO so ordinary FSDP pretraining keeps its
    existing parameter layout and checkpoint/resume behavior.
    """
    if (
        world_size <= 1
        or grpo is None
        or not grpo.enabled
        or model.config.byte_patch_size <= 1
    ):
        return
    model.lm_head.requires_grad_(False)
    model.transformer.wte.requires_grad_(False)


def setup(
    model_name: str,
    model_config: Config | None = None,
    out_dir: Path = Path("out/pretrain"),
    precision: Literal["bf16-true", "bf16-mixed", "32-true", None] = None,
    initial_checkpoint_dir: Path | None = None,
    resume: bool | Literal["auto"] | Path = False,
    data: DataModule | None = None,
    train: TrainArgs = TrainArgs(
        save_interval=1000,
        log_interval=1,
        global_batch_size=512,
        micro_batch_size=4,
        max_tokens=int(3e12),  # 3 trillion
        max_norm=1.0,
        min_lr=4e-5,
        lr_warmup_steps=2000,
        tie_embeddings=False,
    ),
    eval: EvalArgs = EvalArgs(interval=1000, max_iters=100),
    log: LogArgs = LogArgs(),
    optimizer: str | dict = "AdamW",
    devices: int | str = "auto",
    num_nodes: int = 1,
    tokenizer_dir: Path | None = None,
    logger_name: LoggerChoice = "tensorboard",
    seed: int = 42,
    compile_model: bool = True,
    reconstruction_eval: ReconstructionEvalConfig | None = None,
    ce_loss_weight: float = 1.0,
    ce_byte_only: bool = False,
    eos_loss_weight: float = 1.0,
    eos_aux_loss_weight: float = 0.0,
    fim_span_loss_weight: float = 0.0,
    mrt: MRTConfig | None = None,
    free_run_eval: FreeRunEvalConfig | None = None,
    grpo: GRPOConfig | None = None,
    grpo_reference_checkpoint_dir: Path | None = None,
):
    """Pretrain a model.

    Arguments:
        model_name: The name of the model to pretrain. Choose from names in ``litgpt.config``. Use "list" to list the supported models.
        model_config: A ``litgpt.Config`` object to define the model architecture. Mutually exclusive with
            ``model_config``. Overrides the `model_name` if specified.
        out_dir: Directory in which to save checkpoints and logs. If running in a Lightning Studio Job, look for it in
            /teamspace/jobs/<job-name>/share.
        precision: The precision to use for finetuning. Determines a compatible precision setting by default.
        initial_checkpoint_dir: Optional path to a checkpoint directory to initialize the model from.
            Useful for continued pretraining. Mutually exclusive with ``resume``.
        resume: Path to a checkpoint directory to resume from in case training was interrupted, or ``True`` to resume
            from the latest checkpoint in ``out_dir``. An error will be raised if no checkpoint is found. Passing
            ``'auto'`` will resume from the latest checkpoint but not error if no checkpoint exists.
        data: Data-related arguments. If not provided, the default is ``litgpt.data.TinyLlama``.
        train: Training-related arguments. See ``litgpt.args.TrainArgs`` for details.
        eval: Evaluation-related arguments. See ``litgpt.args.EvalArgs`` for details.
        optimizer: An optimizer name (such as "AdamW") or config.

        devices: How many devices/GPUs to use. Uses all GPUs by default.
        num_nodes: How many nodes the code is being run on.
        tokenizer_dir: Optional path to the tokenizer dir that was used for preprocessing the dataset. Only some data
            module require this.
        logger_name: The name of the logger to send metrics to.
        seed: The random seed to use for reproducibility.
    """
    if model_name == "list":
        available_models = "\n".join(sorted(name_to_config))
        print(f"Available values:\n{available_models}")
        quit()

    if initial_checkpoint_dir is not None:
        initial_checkpoint_dir = extend_checkpoint_dir(initial_checkpoint_dir)

    if grpo_reference_checkpoint_dir is not None:
        grpo_reference_checkpoint_dir = extend_checkpoint_dir(grpo_reference_checkpoint_dir)

    if tokenizer_dir is not None:
        tokenizer_dir = extend_checkpoint_dir(tokenizer_dir)

    if model_config is None:
        # Support both model_name options: meta-llama/Meta-Llama-3-8B & Meta-Llama-3-8B
        try:
            model_config = Config.from_name(model_name)
        except ValueError:
            print(f"Model name {model_name} is not supported.\n")
            available_models = "\n".join(sorted(name_to_config))
            print(f"Available values:\n{available_models}")
            quit()

    hparams = capture_hparams()
    data = TinyLlama() if data is None else data

    config = Config.from_name(model_name) if model_config is None else model_config
    precision = precision or get_default_supported_precision(training=True)
    devices = parse_devices(devices)
    out_dir = init_out_dir(out_dir)
    # in case the dataset requires the Tokenizer
    tokenizer = Tokenizer(tokenizer_dir) if tokenizer_dir is not None else None

    logger = choose_logger(
        logger_name,
        out_dir,
        name=f"pretrain-{config.name}",
        resume=bool(resume),
        log_interval=train.log_interval,
        log_args=asdict(log),
    )

    strategy = _select_training_strategy(devices, num_nodes, grpo)

    fabric = L.Fabric(devices=devices, num_nodes=num_nodes, strategy=strategy, precision=precision, loggers=[logger])

    if torch.cuda.is_available() and devices > 1:
        check_nvlink_connectivity(fabric)

    fabric.launch()

    fabric.print(pprint.pformat(hparams))
    if logger_name in ("tensorboard", "wandb", "mlflow"):
        fabric.logger.log_hyperparams(hparams)

    main(
        fabric=fabric,
        devices=devices,
        num_nodes=num_nodes,
        seed=seed,
        initial_checkpoint_dir=initial_checkpoint_dir,
        resume=resume,
        config=config,
        data=data,
        out_dir=out_dir,
        tokenizer_dir=tokenizer_dir,
        tokenizer=tokenizer,
        train=train,
        eval=eval,
        optimizer=optimizer,
        compile_model=compile_model,
        checkpoint_hparams=hparams,
        reconstruction_eval=reconstruction_eval,
        ce_loss_weight=ce_loss_weight,
        ce_byte_only=ce_byte_only,
        eos_loss_weight=eos_loss_weight,
        eos_aux_loss_weight=eos_aux_loss_weight,
        fim_span_loss_weight=fim_span_loss_weight,
        mrt=mrt,
        free_run_eval=free_run_eval,
        grpo=grpo,
        grpo_reference_checkpoint_dir=grpo_reference_checkpoint_dir,
    )


def main(
    fabric: L.Fabric,
    devices: int,
    seed: int,
    initial_checkpoint_dir: Path | None,
    resume: bool | Literal["auto"] | Path,
    config: Config,
    data: DataModule,
    out_dir: Path,
    tokenizer_dir: Path | None,
    tokenizer: Tokenizer | None,
    train: TrainArgs,
    eval: EvalArgs,
    optimizer: str | dict,
    compile_model: bool = True,
    num_nodes: int = 1,
    checkpoint_hparams: dict | None = None,
    reconstruction_eval: ReconstructionEvalConfig | None = None,
    ce_loss_weight: float = 1.0,
    ce_byte_only: bool = False,
    eos_loss_weight: float = 1.0,
    eos_aux_loss_weight: float = 0.0,
    fim_span_loss_weight: float = 0.0,
    mrt: MRTConfig | None = None,
    free_run_eval: FreeRunEvalConfig | None = None,
    grpo: GRPOConfig | None = None,
    grpo_reference_checkpoint_dir: Path | None = None,
) -> None:
    validate_args(train, eval, initial_checkpoint_dir, resume)

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    fabric.seed_everything(seed)  # same seed for every process to init model (FSDP)

    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True):
        model = GPT(config)

    initialize_weights(fabric, model, n_layer=config.n_layer, n_embd=config.n_embd)

    if train.tie_embeddings:
        model.transformer.wte.weight = model.lm_head.weight
    _freeze_inactive_megabyte_ddp_parameters(
        model,
        world_size=devices * num_nodes,
        grpo=grpo,
    )
    if train.max_seq_length:
        model.max_seq_length = train.max_seq_length

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters: {num_parameters(model):,}")

    # Byte-domain extension: compilation can be disabled for short smoke runs
    # where compile time and graph errors obscure data/model integration issues.
    if compile_model:
        model = torch.compile(model)
    model = fabric.setup(model)

    extra_kwargs = {"fused": fabric.device.type == "cuda"}
    optimizer = instantiate_torch_optimizer(optimizer, model.parameters(), **extra_kwargs)
    optimizer = fabric.setup_optimizers(optimizer)

    train_dataloader, val_dataloader = get_dataloaders(fabric, data, tokenizer, train, model.max_seq_length)
    byte_runtime = ByteTrainingRuntime.prepare(
        fabric,
        train_dataloader.dataset,
        val_dataloader.dataset,
        out_dir,
        ce_loss_weight=ce_loss_weight,
        ce_byte_only=ce_byte_only,
        eos_loss_weight=eos_loss_weight,
        eos_aux_loss_weight=eos_aux_loss_weight,
        fim_span_loss_weight=fim_span_loss_weight,
        reconstruction_config=reconstruction_eval,
        mrt_config=mrt,
        free_run_config=free_run_eval,
        grpo_config=grpo,
    )
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    state = {
        "model": model,
        "optimizer": optimizer,
        "train_dataloader": train_dataloader,
        "iter_num": 0,
        "step_count": 0,
    }

    if initial_checkpoint_dir:
        checkpoint_path = initial_checkpoint_dir / "lit_model.pth"
        fabric.print(f"Initializing model weights from {checkpoint_path}")
        # LitGPT training checkpoints contain optimizer and DataLoader objects,
        # not only a raw state_dict. Load just the model entry from this trusted
        # local checkpoint; PyTorch 2.6+ otherwise rejects those extra objects
        # under its default weights-only policy.
        fabric.load(checkpoint_path, {"model": model}, weights_only=False)

    if grpo is not None and grpo.enabled and grpo.kl_coeff > 0:
        if grpo_reference_checkpoint_dir is None:
            raise ValueError(
                "GRPO kl_coeff > 0 requires grpo_reference_checkpoint_dir "
                "(kept independent of initial_checkpoint_dir/resume so the KL "
                "anchor survives a resumed phase-2 run)."
            )
        fabric.print(f"Loading frozen GRPO reference policy from {grpo_reference_checkpoint_dir}")
        reference_model = GPT(config)
        reference_checkpoint = torch.load(
            grpo_reference_checkpoint_dir / "lit_model.pth",
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        reference_state_dict = reference_checkpoint.get("model", reference_checkpoint)
        reference_model.load_state_dict(reference_state_dict)
        reference_model = reference_model.to(fabric.device).eval()
        for parameter in reference_model.parameters():
            parameter.requires_grad_(False)
        byte_runtime.reference_model = reference_model

    resume = find_resume_path(resume, out_dir)
    if resume:
        fabric.print(f"Resuming training from {resume}")
        # Stage-1 checkpoints are trusted local artifacts and contain optimizer
        # and DataLoader state, which PyTorch 2.6's weights-only loader rejects.
        fabric.load(resume, state, weights_only=False)

    train_time = time.perf_counter()
    initial_step_count = state["step_count"]

    # work around PyTorch issue https://github.com/pytorch/pytorch/issues/152162
    # which does not like the lazy initialization to be called in dynamo.
    # TODO: Happens with PyTorch 2.7+
    if (
        (_TORCH_EQUAL_2_7 or _TORCH_EQUAL_2_8)
        and (model._forward_module.__class__.__name__ == "OptimizedModule")
        and (model._forward_module._orig_mod.__class__.__name__ == "FullyShardedDataParallel")
    ):
        from torch.distributed.fsdp._runtime_utils import _root_pre_forward

        _root_pre_forward(model._forward_module._orig_mod, model._forward_module._orig_mod, [], {})

    fit(
        fabric=fabric,
        devices=devices,
        num_nodes=num_nodes,
        state=state,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        out_dir=out_dir,
        tokenizer_dir=tokenizer_dir,
        train=train,
        eval=eval,
        checkpoint_hparams=checkpoint_hparams,
        byte_runtime=byte_runtime,
    )

    # Save final checkpoint
    save_checkpoint(
        fabric,
        state,
        tokenizer_dir,
        out_dir / "final" / "lit_model.pth",
        checkpoint_hparams=checkpoint_hparams,
    )

    total_tokens = state["step_count"] * train.global_batch_size * model.max_seq_length
    segment_tokens = (
        (state["step_count"] - initial_step_count)
        * train.global_batch_size
        * model.max_seq_length
    )
    patch_size = model.config.byte_patch_size

    elapsed_train_time = time.perf_counter() - train_time

    # Print formatted output
    separator = "-" * 40
    fabric.print(separator)
    fabric.print("| Performance")
    fabric.print(f"| - Transformer positions: {total_tokens:,}")
    if patch_size > 1:
        fabric.print(
            f"| - Represented byte slots: {total_tokens * patch_size:,} "
            f"(patch_size={patch_size})"
        )
    fabric.print(f"| - Training Time : {elapsed_train_time:.2f} s")
    fabric.print(f"| - Tok/sec       : {segment_tokens / elapsed_train_time:.2f} tok/s")
    fabric.print("| " + "-" * 40)

    if fabric.device.type == "cuda":
        memory_used = torch.cuda.max_memory_allocated() / 1e9
        fabric.print("| Memory Usage")
        fabric.print(f"| - Memory Used   : {memory_used:.2f} GB")
    fabric.print(separator)


def fit(
    fabric: L.Fabric,
    devices: int,
    state: dict,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    out_dir: Path,
    tokenizer_dir: Path | None,
    train: TrainArgs,
    eval: EvalArgs,
    num_nodes: int = 1,
    checkpoint_hparams: dict | None = None,
    byte_runtime: ByteTrainingRuntime | None = None,
) -> None:
    model = state["model"]
    optimizer = state["optimizer"]
    byte_runtime = byte_runtime or ByteTrainingRuntime()

    if eval.initial_validation:
        val_loss, _, _ = validate(fabric, model, val_dataloader, max_iters=eval.max_iters)
        val_loss = f"{val_loss:.3f}"
    else:
        fabric.print("Verifying settings ...")
        validate(fabric, model, val_dataloader, max_iters=2, verbose=False)  # sanity check
        val_loss = "n/a"

    throughput = ThroughputMonitor(fabric, window_size=5)

    with torch.device("meta"):
        meta_model = GPT(model.config)
        input_shape = (train.micro_batch_size, meta_model.max_seq_length)
        if meta_model.config.byte_patch_size > 1:
            input_shape += (meta_model.config.byte_patch_size,)
        x = torch.randint(0, 1, input_shape)
        # Byte-domain extension: FLOP measurement must follow the same optional
        # region/offset input signature as real training forwards.
        model_kwargs = {}
        if meta_model.config.byte_patch_size > 1:
            model_kwargs["patch_targets"] = x
        if meta_model.config.use_region_id:
            model_kwargs["region_ids"] = torch.zeros_like(x)
        if meta_model.config.use_offset_id:
            offsets = torch.arange(meta_model.max_seq_length)
            if meta_model.config.byte_patch_size > 1:
                offsets = offsets.view(1, -1, 1).expand_as(x)
            else:
                offsets = offsets.expand_as(x)
            model_kwargs["offset_ids"] = offsets
        model_fwd = lambda: meta_model(x, **model_kwargs)  # noqa: F821
        model_loss = lambda y: chunked_cross_entropy(y, x, chunk_size=0)  # noqa: F821
        measured_flops = measure_flops(meta_model, model_fwd, model_loss)
        fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    max_tokens_per_device = train.max_tokens // fabric.world_size
    tokens_per_iter = train.micro_batch_size * model.max_seq_length
    max_iters = max_tokens_per_device // tokens_per_iter
    gradient_accumulation_iters, max_steps, warmup_steps = _optimizer_step_schedule(
        train,
        model.max_seq_length,
        devices,
        num_nodes,
        max_iters,
        train_dataloader,
    )

    # ``iter_num`` counts per-rank microbatches, so its units change whenever the
    # number of devices, nodes, or the microbatch size changes. Checkpoints are only
    # written at optimizer-step boundaries, which lets us reconstruct the correct
    # current-launch counter from the device-independent ``step_count``. Without
    # this rebase, a 1-GPU -> 4-GPU resume appears four times farther through both
    # the LR schedule and token budget.
    resumed_iter_num, rebased_iter_num = _rebase_microiteration_counter(
        state, gradient_accumulation_iters
    )
    if resumed_iter_num != rebased_iter_num:
        fabric.print(
            "Rebasing device-dependent microiteration counter for this launch: "
            f"iter_num {resumed_iter_num} -> {rebased_iter_num} at optimizer step "
            f"{state['step_count']} (grad_accum={gradient_accumulation_iters})."
        )
    log_iter_interval = train.log_interval * gradient_accumulation_iters
    initial_iter = state["iter_num"]
    initial_step = state["step_count"]
    train_iterator = CycleIterator(train_dataloader)

    # Batch config -- read this alongside the per-step "peak mem" to size micro_batch_size.
    fabric.print(
        f"batch config | micro_batch={train.micro_batch_size} "
        f"grad_accum={gradient_accumulation_iters} "
        f"global_batch={train.global_batch_size} block/seq={model.max_seq_length} "
        f"devices={devices}x{num_nodes} max_steps={max_steps} warmup_steps={warmup_steps} "
        "-> watch 'peak mem' below vs the card size"
    )

    # Per-rank distributed topology (printed from EVERY rank, not just rank 0) -- the first
    # thing to check when a multi-GPU run hangs on a collective. Confirms each rank landed
    # on a DISTINCT physical GPU (same uuid on two ranks => device-binding bug, not NCCL),
    # the world size is right, and the NCCL transport env actually took effect.
    if torch.cuda.is_available():
        _dev = torch.cuda.current_device()
        try:
            _uuid = str(torch.cuda.get_device_properties(_dev).uuid)[:13]
        except Exception:
            _uuid = "n/a"
        print(
            f"[dist] rank={fabric.global_rank}/{fabric.world_size} "
            f"local_rank={fabric.local_rank} node={getattr(fabric, 'node_rank', 0)} "
            f"cuda:{_dev} uuid={_uuid} visible={os.environ.get('CUDA_VISIBLE_DEVICES', '<all>')} "
            f"| NCCL_P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE', '')} "
            f"NCCL_IB_DISABLE={os.environ.get('NCCL_IB_DISABLE', '')}",
            flush=True,
        )
    fabric.barrier()  # surfaces a broken communicator HERE (with the topology above) rather
    # than 30 min later inside validate()

    running_loss = RunningMean(window=gradient_accumulation_iters, sync_on_compute=False).to(
        fabric.device
    )
    running_full_ce = RunningMean(
        window=gradient_accumulation_iters, sync_on_compute=False
    ).to(fabric.device)
    running_fim_span_ce = RunningMean(
        window=gradient_accumulation_iters, sync_on_compute=False
    ).to(fabric.device)
    running_eos_aux = RunningMean(
        window=gradient_accumulation_iters, sync_on_compute=False
    ).to(fabric.device)
    # Track the CE represented by one optimizer step. MRT is applied only at
    # the accumulation boundary, so a single final microbatch is not a fair
    # scalar comparison with its decoder-risk update.
    step_ce_sum = torch.zeros((), device=fabric.device)
    step_ce_count = 0
    fabric.barrier()
    total_t0 = time.perf_counter()

    last_reconstruction_step = -1

    for train_data in train_iterator:
        if state["step_count"] >= max_steps:
            break

        # LR is constant across all microbatches contributing to one optimizer
        # update and depends only on the device-independent optimizer step.
        lr = get_lr(
            optimizer.defaults["lr"],
            state["step_count"],
            warmup_steps,
            max_steps,
            train.min_lr,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        state["iter_num"] += 1
        iter_t0 = time.perf_counter()

        # Byte-domain extension: dict batches already contain aligned labels and
        # auxiliary ids, while original text batches retain LitGPT's shift path.
        model_inputs, targets = get_model_inputs_and_targets(train_data, model.max_seq_length)

        is_accumulating = state["iter_num"] % gradient_accumulation_iters != 0
        run_mrt = byte_runtime.should_run_mrt(
            state["step_count"] + 1, is_accumulating
        )
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(**model_inputs)
            target_region_ids = None
            if isinstance(train_data, dict):
                target_region_ids = train_data.get("target_region_ids")
                if target_region_ids is None:
                    target_region_ids = train_data.get("region_ids")
                if isinstance(target_region_ids, torch.Tensor):
                    target_region_ids = target_region_ids[
                        :, : model.max_seq_length
                    ].contiguous().long()
            loss_terms = byte_runtime.loss_terms(
                logits, targets, target_region_ids
            )
            loss = loss_terms["objective"]
            # Byte-domain experiments may make decoder risk the primary
            # objective while retaining a small CE syntax regularizer.
            fabric.backward(
                byte_runtime.ce_loss_weight
                * loss
                / gradient_accumulation_iters
            )
        step_ce_sum += loss.detach()
        step_ce_count += 1

        mrt_metrics = None
        if run_mrt:
            weighted_ce_grad_norm = byte_runtime.gradient_l2_norm(model)
            ce_gradients = byte_runtime.capture_gradients(model)
            mrt_metrics = byte_runtime.run_mrt(
                fabric, model, state["step_count"] + 1
            )
            if mrt_metrics is not None and not mrt_metrics["mrt/skipped"]:
                raw_ce_step = float(step_ce_sum / step_ce_count)
                weighted_ce_step = byte_runtime.ce_loss_weight * raw_ce_step
                weighted_risk = mrt_metrics["mrt/weighted_expected_risk"]
                mrt_metrics.update(
                    {
                        "optimization/raw_ce_step": raw_ce_step,
                        "optimization/weighted_ce_step": weighted_ce_step,
                        "optimization/combined_objective_sampled": weighted_ce_step
                        + weighted_risk,
                        "optimization/weighted_ce_grad_norm": weighted_ce_grad_norm,
                        "optimization/mrt_grad_norm": byte_runtime.gradient_delta_l2_norm(
                            model, ce_gradients
                        ),
                        "optimization/combined_grad_norm_pre_clip": byte_runtime.gradient_l2_norm(
                            model
                        ),
                    }
                )

        run_grpo = byte_runtime.should_run_grpo(state["step_count"] + 1, is_accumulating)

        running_loss.update(loss.detach())
        running_full_ce.update(loss_terms["full_ce"].detach())
        running_fim_span_ce.update(loss_terms["fim_span_ce"].detach())
        running_eos_aux.update(loss_terms["eos_aux"].detach())

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=train.max_norm)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1
            if mrt_metrics is not None:
                byte_runtime.log_mrt(fabric, mrt_metrics, state["step_count"])
            step_ce_sum.zero_()
            step_ce_count = 0

            # Runs after the CE step commits and clears its gradient, since
            # GRPO (mu possibly > 1) owns its own zero_grad/backward/step
            # cycle on a clean slate rather than sharing this iteration's CE
            # gradient accumulation.
            if run_grpo:
                grpo_metrics = byte_runtime.run_grpo(
                    fabric, optimizer, model, train.max_norm, state["step_count"]
                )
                if grpo_metrics is not None:
                    byte_runtime.log_grpo(fabric, grpo_metrics, state["step_count"])

        if state["iter_num"] % log_iter_interval == 0:
            loss = running_loss.compute().item()  # expensive device-to-host synchronization
            t1 = time.perf_counter()
            throughput.update(
                time=(t1 - total_t0),
                flops=(measured_flops * log_iter_interval),
                batches=(state["iter_num"] - initial_iter),
                samples=((state["iter_num"] - initial_iter) * train.micro_batch_size),
                lengths=(
                    (state["iter_num"] - initial_iter)
                    * train.micro_batch_size
                    * model.max_seq_length
                ),
            )
            completed_steps = state["step_count"]
            completed_samples = completed_steps * train.global_batch_size
            completed_tokens = completed_samples * model.max_seq_length
            elapsed_steps = completed_steps - initial_step
            metrics = {
                "loss": loss,
                "training/full_ce": running_full_ce.compute().item(),
                "training/fim_span_ce": running_fim_span_ce.compute().item(),
                "training/eos_aux_loss": running_eos_aux.compute().item(),
                "training/objective": loss,
                "iter": state["iter_num"],
                "step": completed_steps,
                "epoch": train_iterator.epoch,
                "iter_time": t1 - iter_t0,
                "remaining_time": (
                    (t1 - total_t0) / elapsed_steps * (max_steps - completed_steps)
                ),
                # Canonical global counters remain continuous when world size changes.
                "samples_seen": completed_samples,
                "tokens": completed_tokens,
                "total_tokens": completed_tokens,
                "learning_rate": lr,
            }
            # Byte-domain extension: context and padding are masked, so record
            # how many tokens actually contribute to cross-entropy.
            if isinstance(train_data, dict):
                supervised_tokens = (targets != -100).sum()
                metrics["supervised_tokens"] = supervised_tokens
                metrics["raw_tokens"] = model_inputs["idx"].numel()
            # Peak GPU memory so far -- use it to size micro_batch_size. `reserved` is what
            # actually counts against the card (e.g. 48 GB a6000); it stabilizes within a
            # few steps, so the first log lines tell you the headroom. If reserved is well
            # under the card size you can raise MICRO_BATCH_SIZE.
            mem_reserved_gb = 0.0
            if fabric.device.type == "cuda":
                mem_reserved_gb = torch.cuda.max_memory_reserved() / 1e9
                metrics["gpu_mem_reserved_gb"] = mem_reserved_gb
                metrics["gpu_mem_alloc_gb"] = torch.cuda.max_memory_allocated() / 1e9
            if isinstance(val_loss, float):
                val_loss = f"{val_loss:.3f}"
            objective_detail = ""
            if (
                byte_runtime.fim_span_loss_weight > 0
                or byte_runtime.eos_aux_loss_weight > 0
            ):
                objective_detail = (
                    f" full: {metrics['training/full_ce']:.3f},"
                    f" span: {metrics['training/fim_span_ce']:.3f},"
                    f" eos_aux: {metrics['training/eos_aux_loss']:.3f},"
                )
            fabric.print(
                f"Epoch {metrics['epoch'] + 1} | iter {metrics['iter']} step {metrics['step']} |"
                f" loss train: {metrics['loss']:.3f},"
                f"{objective_detail}"
                f" val: {val_loss} |"
                f" iter time: {metrics['iter_time'] * 1000:.2f} ms"
                f"{' (step)' if not is_accumulating else ''}"
                f" | peak mem: {mem_reserved_gb:.1f} GB"
                f" remaining time: {timedelta(seconds=int(metrics['remaining_time']))!s}"
            )

            throughput_metrics = throughput.compute()
            metrics.update(throughput_metrics)
            fabric.log_dict(metrics, step=completed_steps)

        if val_dataloader is not None and not is_accumulating and state["step_count"] % eval.interval == 0:
            t0 = time.perf_counter()
            val_loss, task_losses, eos_metrics = validate(
                fabric, model, val_dataloader, max_iters=eval.max_iters
            )
            val_loss = val_loss.item()
            td = time.perf_counter() - t0

            task_summary = "".join(f", {name}: {tl:.4f}" for name, tl in task_losses.items())
            fabric.print(
                f"iter {state['iter_num']}: val loss {val_loss:.4f}{task_summary}, "
                f"val time: {td * 1000:.2f} ms"
            )
            metrics = {"val_loss": val_loss, "val_ppl": math.exp(val_loss)}
            for name, tl in task_losses.items():
                metrics[f"val_loss_{name}"] = tl
                metrics[f"val_ppl_{name}"] = math.exp(tl)
            metrics.update(eos_metrics)
            fabric.log_dict(metrics, step=state["step_count"])
            fabric.barrier()

        if train.save_interval is not None and not is_accumulating and state["step_count"] % train.save_interval == 0:
            save_checkpoint(
                fabric,
                state,
                tokenizer_dir,
                out_dir / f"step-{state['step_count']:08d}" / "lit_model.pth",
                checkpoint_hparams=checkpoint_hparams,
            )

        if (
            byte_runtime.reconstruction_due(state["step_count"])
            and not is_accumulating
        ):
            byte_runtime.evaluate_reconstruction(
                fabric, model, state["step_count"]
            )
            last_reconstruction_step = state["step_count"]

        if (
            byte_runtime.free_run_due(state["step_count"])
            and not is_accumulating
        ):
            byte_runtime.evaluate_free_run(
                fabric, model, state["step_count"]
            )

    # Final validation
    if eval.final_validation:
        val_loss, task_losses, eos_metrics = validate(
            fabric, model, val_dataloader, max_iters=eval.max_iters
        )
        metrics = {"val_loss": val_loss, "val_ppl": math.exp(val_loss)}
        for name, tl in task_losses.items():
            metrics[f"val_loss_{name}"] = tl
            metrics[f"val_ppl_{name}"] = math.exp(tl)
        metrics.update(eos_metrics)
        fabric.log_dict(metrics, step=state["step_count"])
        fabric.print(f"Final evaluation | val loss: {val_loss.item():.3f} | val ppl: {math.exp(val_loss):.3f}")

    if (
        byte_runtime.reconstruction_config is not None
        and byte_runtime.reconstruction_samples
        and last_reconstruction_step != state["step_count"]
    ):
        byte_runtime.evaluate_reconstruction(
            fabric, model, state["step_count"], final=True
        )


# Compatibility alias for callers that imported this private helper before the
# byte training helpers were moved into their own package.
_namespace_reconstruction_metrics = namespace_reconstruction_metrics


def get_dataloaders(
    fabric: L.Fabric, data: DataModule, tokenizer: Tokenizer, train: TrainArgs, block_size: int
) -> tuple[DataLoader, DataLoader]:
    data.connect(tokenizer=tokenizer, batch_size=train.micro_batch_size, max_seq_length=block_size)
    with fabric.rank_zero_first():
        data.prepare_data()
    data.setup()
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()
    return train_dataloader, val_dataloader


# learning rate decay scheduler (cosine with linear warmup)
def _optimizer_step_schedule(
    train: TrainArgs,
    max_seq_length: int,
    devices: int,
    num_nodes: int,
    max_iters: int,
    train_dataloader: DataLoader,
) -> tuple[int, int, int]:
    """Return accumulation, maximum, and warmup counts in optimizer-step units.

    ``TrainArgs.max_tokens`` is global, so dividing by the global tokens per
    optimizer update produces a limit that does not depend on launcher world size.
    ``TrainArgs.warmup_iters`` returns per-rank microiterations; convert it once at
    the boundary instead of scheduling directly in those device-dependent units.
    """
    gradient_accumulation_iters = train.gradient_accumulation_iters(devices, num_nodes)
    max_steps = train.max_tokens // (train.global_batch_size * max_seq_length)
    if train.max_steps is not None:
        max_steps = min(max_steps, train.max_steps)
    warmup_iters = train.warmup_iters(devices, num_nodes, max_iters, train_dataloader)
    warmup_steps = min(
        math.ceil(warmup_iters / gradient_accumulation_iters),
        max_steps,
    )
    return gradient_accumulation_iters, max_steps, warmup_steps


def _rebase_microiteration_counter(
    state: dict, gradient_accumulation_iters: int
) -> tuple[int, int]:
    """Re-express a boundary checkpoint's microiteration counter for this launch."""
    resumed_iter_num = int(state["iter_num"])
    rebased_iter_num = int(state["step_count"]) * gradient_accumulation_iters
    state["iter_num"] = rebased_iter_num
    return resumed_iter_num, rebased_iter_num


def get_lr(learning_rate: float, it: int, warmup_iters: int, max_iters: int, min_lr: float) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > max_iters, return min learning rate
    if it > max_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def initialize_weights(fabric: L.Fabric, model: GPT, n_layer: int, n_embd: int) -> None:
    """GPT-NeoX weight initialization (https://arxiv.org/abs/2204.06745)."""
    # Adapted from https://github.com/jzhang38/TinyLlama

    def init_weights(module, std):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)

    for mod in model.modules():
        if isinstance(mod, (nn.Embedding, nn.Linear)):
            module_width = getattr(mod, "_litgpt_init_n_embd", n_embd)
            mod.reset_parameters = partial(
                init_weights,
                mod,
                std=math.sqrt(2.0 / 5 / module_width),
            )

    # need a separate loop because `mod.proj` below is a `nn.Linear` too
    for mod in model.modules():
        if isinstance(mod, (LLaMAMLP, CausalSelfAttention)):
            mod.proj.reset_parameters = partial(
                init_weights,
                mod.proj,
                std=(
                    1
                    / math.sqrt(mod.config.n_embd)
                    / mod.config.n_layer
                ),
            )

    if not isinstance(fabric.strategy, FSDPStrategy):
        reset_parameters(model)


def save_checkpoint(fabric, state, tokenizer_dir, checkpoint_file, checkpoint_hparams=None):
    model = state["model"]
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    fabric.print(f"Saving checkpoint to {str(checkpoint_file)!r}")
    fabric.save(checkpoint_file, state)
    if fabric.global_rank == 0:
        save_hyperparameters(setup, checkpoint_file.parent, hparams=checkpoint_hparams)
        if tokenizer_dir is not None:
            copy_config_files(tokenizer_dir, checkpoint_file.parent)
        save_config(model.config, checkpoint_file.parent)


def validate_args(train: TrainArgs, eval: EvalArgs, initial_checkpoint_dir, resume) -> None:
    issues = []
    unsupported = [(train, ["epochs"]), (eval, ["max_new_tokens"])]
    for args, names in unsupported:
        for name in names:
            if getattr(args, name) is not None:
                issues.append(f"{__file__} doesn't support the {name!r} argument. This is set in {args}")
    if train.max_steps is not None:
        warnings.warn(
            "`train.max_steps` is intended for profiling or debug runs only. "
            "For full pretraining runs, prefer `train.max_tokens` or `train.max_time`.",
            UserWarning,
        )
    required = [(train, ["max_tokens", "max_norm"])]
    for args, names in required:
        for name in names:
            if getattr(args, name) is None:
                issues.append(f"{__file__} requires the {name!r} argument. This is set in {args}")
    if initial_checkpoint_dir and resume:
        issues.append("Can't provide both `--resume` and `--initial_checkpoint_dir`. Choose one.")
    if issues:
        raise ValueError("\n".join(issues))
