"""Minimal 2-GPU NCCL smoke test -- isolates the collective hang from litgpt.

Launch under srun with one task per GPU, e.g.:

  srun --nodes=1 --ntasks-per-node=2 --gres=gpu:rtxa6000:2 --cpus-per-task=4 \
       --partition=vulcan-ampere --qos=vulcan-medium --account=vulcan-metzler \
       python scripts/hpc/vulcan/nccl_test.py

Each rank prints its GPU and the all-reduced value. Expected: value == world_size (2),
and the two ranks must report DIFFERENT physical GPUs. If it HANGS, the problem is pure
NCCL transport (retry with NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1; add NCCL_DEBUG=INFO to
see why). If the ranks report the SAME GPU, it's a device-binding problem, not P2P.
"""
import os
from datetime import timedelta

import torch
import torch.distributed as dist

# Short timeout so a broken collective fails in ~2 min instead of the 30-min default.
dist.init_process_group("nccl", timeout=timedelta(seconds=120))
rank = dist.get_rank()
world = dist.get_world_size()
local = int(os.environ.get("SLURM_LOCALID", rank % max(1, torch.cuda.device_count())))
torch.cuda.set_device(local)

# report the physical GPU this rank landed on (uuid distinguishes them even if index 0)
name = torch.cuda.get_device_name(local)
print(f"[rank {rank}/{world}] local_id={local} cuda:{torch.cuda.current_device()} "
      f"visible={os.environ.get('CUDA_VISIBLE_DEVICES','<all>')} gpu='{name}'", flush=True)

t = torch.ones(1, device="cuda")
dist.all_reduce(t)  # <- this is the collective that hung in training
print(f"[rank {rank}] all_reduce -> {t.item():.0f} (expect {world})", flush=True)

dist.barrier()
if rank == 0:
    print("NCCL OK" if t.item() == world else "NCCL WRONG RESULT", flush=True)
dist.destroy_process_group()
