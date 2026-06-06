import sys
from pathlib import Path
from typing import Any

import yaml

from litgpt.utils import CLI


def _yaml_safe(value: Any) -> Any:
    """Convert captured Python values to objects accepted by yaml.safe_dump."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _yaml_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def parser_commands() -> list[str]:
    return [
        "download",
        "chat",
        "finetune",
        "finetune_lora",
        "finetune_full",
        "finetune_adapter",
        "finetune_adapter_v2",
        "pretrain",
        "generate",
        "generate_full",
        "generate_adapter",
        "generate_adapter_v2",
        "generate_sequentially",
        "generate_speculatively",
        "generate_tp",
        "convert_to_litgpt",
        "convert_from_litgpt",
        "convert_pretrained_checkpoint",
        "merge_lora",
        "evaluate",
        "serve",
        "validate",
    ]


def save_hyperparameters(
    function: callable,
    checkpoint_dir: Path,
    known_commands: list[str] | None = None,
    hparams: dict[str, Any] | None = None,
) -> None:
    """Save explicit hyperparameters, or capture them from the standard LitGPT CLI."""
    if hparams is not None:
        # Custom launchers have their own CLI syntax, so reparsing sys.argv with
        # LitGPT's standard parser would fail while writing a checkpoint.
        with open(checkpoint_dir / "hyperparameters.yaml", "w", encoding="utf-8") as file:
            yaml.safe_dump(_yaml_safe(hparams), file, sort_keys=False)
        return

    from jsonargparse import capture_parser

    # TODO: Make this more robust
    # This hack strips away the subcommands from the top-level CLI
    # to parse the file as if it was called as a script
    if known_commands is None:
        known_commands = parser_commands()
    known_commands = [(c,) for c in known_commands]
    for known_command in known_commands:
        unwanted = slice(1, 1 + len(known_command))
        if tuple(sys.argv[unwanted]) == known_command:
            sys.argv[unwanted] = []

    parser = capture_parser(lambda: CLI(function))
    config = parser.parse_args()
    parser.save(config, checkpoint_dir / "hyperparameters.yaml", overwrite=True)
