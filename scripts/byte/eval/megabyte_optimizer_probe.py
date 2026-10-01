#!/usr/bin/env python3
"""Is the MEGABYTE global model actually being trained? Read AdamW's own statistics.

Training checkpoints (FSDP state_dict_type="full") contain the complete AdamW
state, so this runs on CPU without any forward/backward pass. Per component
(global blocks, global embedding, global->local projection, local blocks, local
embeddings, output head) it reports, weighted by parameter count:

  grad_rms     RMS of sqrt(v): the long-run gradient magnitude per parameter
  snr          mean |m| / sqrt(v) (bias-corrected). For a pure-noise gradient
               with beta1=0.9 this is ~sqrt((1-b1)/(1+b1)) ~= 0.23; a consistent
               gradient pushes it toward 1. Near 0.23 = the component mostly
               random-walks rather than learning.
  rel_step     lr * RMS(m / sqrt(v)) / RMS(weight): the fraction of the weight
               scale Adam moves per step (weight decay excluded)
  weight_rms   RMS of the weights
  rel_change   (with --compare) ||W_a - W_b|| / ||W_b|| between two checkpoints

    python scripts/byte/eval/megabyte_optimizer_probe.py RUN/latest \
        --compare RUN/step-00200000 --out RUN/eval_global_ablation/optimizer_probe.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import torch

GROUPS = (
    ("global_blocks", r"^transformer\.h\."),
    ("global_embed", r"^megabyte_global_(wte|region_wte|offset_wte)\."),
    ("global_to_local", r"^megabyte_global_to_local\."),
    ("global_final_norm", r"^transformer\.ln_f\."),
    ("local_blocks", r"^megabyte_local\.h\."),
    ("local_embed", r"^megabyte_local\.(wte|pos_wte|pad_wte)\."),
    ("local_final_norm", r"^megabyte_local\.ln_f\."),
    ("lm_head", r"^lm_head\."),
    ("unused_token_wte", r"^transformer\.wte\."),
)
PREFIXES = ("_forward_module.", "_orig_mod.", "module.", "_fsdp_wrapped_module.")


def clean(name: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in PREFIXES:
            if name.startswith(prefix):
                name, changed = name[len(prefix):], True
        name = name.replace("._fsdp_wrapped_module", "").replace("._checkpoint_wrapped_module", "")
    return name


def group_of(name: str) -> str:
    for group, pattern in GROUPS:
        if re.search(pattern, name):
            return group
    return "other"


def load(path: Path) -> dict:
    file = path / "lit_model.pth" if path.is_dir() else path
    return torch.load(file, map_location="cpu", mmap=True, weights_only=False)


def model_state(ckpt: dict) -> dict:
    sd = ckpt["model"] if "model" in ckpt else ckpt
    return {clean(k): v for k, v in sd.items()}


def optimizer_state(ckpt: dict, names: list[str]):
    opt = ckpt.get("optimizer")
    if not isinstance(opt, dict) or "state" not in opt:
        raise SystemExit("checkpoint has no optimizer state (weights-only file?)")
    groups = opt.get("param_groups", [{}])
    hp = groups[0]
    state = opt["state"]
    keys = list(state)
    if keys and isinstance(keys[0], str):
        by_name = {clean(k): v for k, v in state.items()}
    else:  # index-keyed: params listed in param_groups order == model.parameters() order
        order = [i for g in groups for i in g.get("params", [])]
        if len(order) != len(names):
            raise SystemExit(f"cannot map {len(order)} optimizer params onto {len(names)} model params")
        by_name = {names[pos]: state[idx] for pos, idx in enumerate(order) if idx in state}
    return by_name, hp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--compare", type=Path, default=None, help="earlier checkpoint for relative weight change")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ckpt = load(args.checkpoint)
    weights = model_state(ckpt)
    names = list(weights)
    adam, hp = optimizer_state(ckpt, names)
    lr = float(hp.get("lr", float("nan")))
    beta1, beta2 = (float(b) for b in hp.get("betas", (0.9, 0.95)))
    eps = float(hp.get("eps", 1e-8))
    other = model_state(load(args.compare)) if args.compare else None

    acc = defaultdict(lambda: defaultdict(float))
    missing = []
    for name, w in weights.items():
        if not torch.is_floating_point(w):
            continue
        s = adam.get(name)
        g = group_of(name)
        n = w.numel()
        a = acc[g]
        a["params"] += n
        a["w_sq"] += float(w.float().pow(2).sum())
        if other is not None and name in other:
            a["delta_sq"] += float((w.float() - other[name].float()).pow(2).sum())
            a["ref_sq"] += float(other[name].float().pow(2).sum())
        if s is None or "exp_avg" not in s:
            missing.append(name)
            continue
        step = float(s.get("step", 1)) if not torch.is_tensor(s.get("step")) else float(s["step"])
        m = s["exp_avg"].float() / (1 - beta1 ** max(step, 1))
        v = s["exp_avg_sq"].float() / (1 - beta2 ** max(step, 1))
        denom = v.sqrt() + eps
        u = m / denom
        a["adam_params"] += n
        a["v_sum"] += float(v.sum())
        a["snr_sum"] += float(u.abs().sum())
        a["u_sq"] += float(u.pow(2).sum())
        a["step"] = step

    noise = math.sqrt((1 - beta1) / (1 + beta1))
    report = {}
    for g in [x for x, _ in GROUPS] + ["other"]:
        a = acc.get(g)
        if not a or not a["params"]:
            continue
        r = {"params": int(a["params"]), "weight_rms": math.sqrt(a["w_sq"] / a["params"])}
        if a["adam_params"]:
            k = a["adam_params"]
            r["grad_rms"] = math.sqrt(a["v_sum"] / k)
            r["snr"] = a["snr_sum"] / k
            r["snr_over_noise_floor"] = r["snr"] / noise
            r["update_rms"] = math.sqrt(a["u_sq"] / k)
            r["rel_step"] = lr * r["update_rms"] / max(r["weight_rms"], 1e-12)
        if a["ref_sq"]:
            r["rel_change"] = math.sqrt(a["delta_sq"] / a["ref_sq"])
        report[g] = r
    result = {
        "checkpoint": str(args.checkpoint),
        "compare": str(args.compare) if args.compare else None,
        "lr": lr, "betas": [beta1, beta2], "eps": eps,
        "noise_floor_snr": noise,
        "groups": report,
        "params_without_adam_state": missing[:20],
    }
    print(f"lr={lr:.3g}  betas=({beta1},{beta2})  pure-noise SNR floor={noise:.3f}")
    cols = ("params", "grad_rms", "snr", "snr_over_noise_floor", "rel_step", "weight_rms", "rel_change")
    heads = ("params", "grad_rms", "snr", "snr/noise", "rel_step", "weight_rms", "rel_change")
    print(f"{'group':18s}" + "".join(f"{c:>14s}" for c in heads))
    for g, r in report.items():
        cells = []
        for c in cols:
            x = r.get(c)
            cells.append(f"{'-':>14s}" if x is None else (f"{x:>14,d}" if c == "params" else f"{x:>14.3g}"))
        print(f"{g:18s}" + "".join(cells))
    if missing:
        print(f"warning: {len(missing)} params had no Adam state, e.g. {missing[:3]}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.out.exists():
            raise SystemExit(f"refusing to overwrite {args.out}")
        args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        print(f"written: {args.out}")


if __name__ == "__main__":
    main()
