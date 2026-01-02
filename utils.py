"""
Utility functions for the rl-with-rm project.
"""

import argparse
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

    for field in fields(dataclass_type):
        field_name = field.name
        cli_name = renames.get(field_name, field_name)
        reverse_renames[cli_name] = field_name

        # Get default value
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

        # Add argument based on type
        if field_type == bool:
            # Bool fields get --flag and --no_flag
            parser.add_argument(
                f"--{cli_name}",
                action="store_true",
                default=default,
                help=f"Enable {field_name.replace('_', ' ')}",
            )
            parser.add_argument(
                f"--no_{cli_name}",
                action="store_false",
                dest=cli_name,
                help=f"Disable {field_name.replace('_', ' ')}",
            )
        else:
            parser.add_argument(
                f"--{cli_name}",
                type=field_type,
                default=default,
                help=f"{field_name.replace('_', ' ').capitalize()}",
            )

    return reverse_renames


def build_config_from_args(dataclass_type, args: argparse.Namespace, reverse_renames: dict):
    """
    Build a dataclass config from parsed args.

    Args:
        dataclass_type: The dataclass class
        args: Parsed argparse namespace
        reverse_renames: Dict from cli_name -> field_name

    Returns:
        Instance of dataclass_type
    """
    kwargs = {}
    for cli_name, field_name in reverse_renames.items():
        if hasattr(args, cli_name):
            kwargs[field_name] = getattr(args, cli_name)
    return dataclass_type(**kwargs)
