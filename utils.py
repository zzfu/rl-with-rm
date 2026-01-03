"""
Utility functions for the rl-with-rm project.
"""

import argparse
import json
from dataclasses import fields, MISSING
from typing import get_origin, get_args


def add_dataclass_args(
    parser: argparse.ArgumentParser,
    dataclass_type,
    renames: dict = None,
) -> dict:
    """
    Auto-generate argparse arguments from a dataclass.

    Args:
        parser: ArgumentParser to add arguments to
        dataclass_type: The dataclass class (not instance)
        renames: Dict mapping field_name -> cli_name for overrides

    Returns:
        Dict mapping cli_name -> field_name (for building config from args)
    """
    renames = renames or {}
    reverse_renames = {}  # cli_name -> field_name

    # Add --config option for loading from config.json
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.json file to load as base config (CLI args override)",
    )

    for field in fields(dataclass_type):
        field_name = field.name
        cli_name = renames.get(field_name, field_name)
        reverse_renames[cli_name] = field_name

        # Get default value (for help text)
        if field.default is not MISSING:
            default = field.default
        elif field.default_factory is not MISSING:
            default = field.default_factory()
        else:
            default = None

        # Get field type
        field_type = field.type
        origin = get_origin(field_type)
        if origin is not None:
            # Handle Optional, List, etc.
            args = get_args(field_type)
            if type(None) in args:
                # Optional type - use the non-None type
                field_type = [a for a in args if a is not type(None)][0]

        # Build help text with default
        help_text = f"{field_name.replace('_', ' ').capitalize()}"
        if default is not None and default != "":
            help_text += f" (default: {default})"

        # Add argument based on type (default=None to detect explicit CLI args)
        if field_type == bool:
            # BooleanOptionalAction creates --flag and --no-flag
            parser.add_argument(
                f"--{cli_name}",
                action=argparse.BooleanOptionalAction,
                default=None,
                help=help_text,
            )
        else:
            parser.add_argument(
                f"--{cli_name}",
                type=field_type,
                default=None,
                help=help_text,
            )

    return reverse_renames


def build_config_from_args(dataclass_type, args: argparse.Namespace, reverse_renames: dict):
    """
    Build a dataclass config from parsed args.

    Merges in order (later overrides earlier):
    1. Dataclass defaults
    2. Config file (if --config provided)
    3. Explicit CLI args (non-None values)

    Args:
        dataclass_type: The dataclass class
        args: Parsed argparse namespace
        reverse_renames: Dict from cli_name -> field_name

    Returns:
        Instance of dataclass_type
    """
    # 1. Start with dataclass defaults
    kwargs = {}
    for field in fields(dataclass_type):
        if field.default is not MISSING:
            kwargs[field.name] = field.default
        elif field.default_factory is not MISSING:
            kwargs[field.name] = field.default_factory()

    # 2. Override with config file if provided
    if args.config:
        with open(args.config) as f:
            file_config = json.load(f)
        kwargs.update(file_config)

    # 3. Override with explicit CLI args (non-None values)
    for cli_name, field_name in reverse_renames.items():
        value = getattr(args, cli_name, None)
        if value is not None:
            kwargs[field_name] = value

    return dataclass_type(**kwargs)
