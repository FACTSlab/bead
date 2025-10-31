"""Configuration commands for bead CLI.

This module provides commands for viewing, validating, and managing configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import yaml
from pydantic import ValidationError

from bead.cli.utils import (
    format_output,
    get_nested_value,
    load_config_for_cli,
    print_error,
    print_info,
    print_success,
    redact_sensitive_values,
)
from bead.config import list_profiles, validate_config


@click.group()
def config() -> None:
    r"""Manage configuration commands.

    Provides commands for viewing, validating, and exporting configuration.

    \b
    Examples:
        $ bead config show
        $ bead config show --format json
        $ bead config show --key paths.data_dir
        $ bead config validate
        $ bead config export --output my-config.yaml
        $ bead config profiles
    """


@config.command()
@click.option(
    "--format",
    "-f",
    "format_type",
    type=click.Choice(["yaml", "json", "table"], case_sensitive=False),
    default="yaml",
    help="Output format (default: yaml)",
)
@click.option(
    "--key",
    "-k",
    type=str,
    default=None,
    help="Show specific config value (e.g., paths.data_dir)",
)
@click.option(
    "--no-redact",
    is_flag=True,
    default=False,
    help="Show sensitive values (API keys, etc.)",
)
@click.pass_context
def show(
    ctx: click.Context,
    format_type: str,
    key: str | None,
    no_redact: bool,
) -> None:
    r"""Display current configuration.

    Shows the merged configuration from profile, file, and environment variables.

    \b
    Examples:
        $ bead config show
        $ bead config show --format json
        $ bead config show --key paths.data_dir
        $ bead config show --no-redact  # Show API keys
    """
    config_file = ctx.obj.get("config_file")
    profile = ctx.obj.get("profile", "default")
    verbose = ctx.obj.get("verbose", False)

    try:
        cfg = load_config_for_cli(
            config_file=str(config_file) if config_file else None,
            profile=profile,
            verbose=verbose,
        )

        # Convert to dict
        config_dict = cfg.model_dump()

        # Redact sensitive values unless --no-redact
        if not no_redact:
            config_dict = redact_sensitive_values(config_dict)

        # Show specific key if requested
        if key:
            try:
                value = get_nested_value(config_dict, key)
                click.echo(value)
            except KeyError as e:
                print_error(f"Configuration key not found: {e}")
            return

        # Format and display
        try:
            output = format_output(config_dict, format_type)  # type: ignore[arg-type]
            click.echo(output)
        except ValueError as e:
            print_error(f"Failed to format output: {e}")

    except Exception as e:
        print_error(f"Failed to load configuration: {e}")


@config.command()
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Configuration file to validate",
)
@click.pass_context
def validate(ctx: click.Context, config_file: Path | None) -> None:
    r"""Validate configuration file.

    Checks YAML syntax and validates against bead configuration schema.

    \b
    Examples:
        $ bead config validate
        $ bead config validate --config-file my-config.yaml

    \b
    Exit codes:
        0 - Configuration is valid
        1 - Configuration is invalid
    """
    # Use CLI context config-file if not explicitly provided
    if config_file is None:
        config_file = ctx.obj.get("config_file")

    if config_file is None:
        print_error("No configuration file specified. Use --config-file or -c.")
        return

    profile = ctx.obj.get("profile", "default")
    verbose = ctx.obj.get("verbose", False)

    try:
        # Load and validate
        cfg = load_config_for_cli(
            config_file=str(config_file),
            profile=profile,
            verbose=verbose,
        )

        # Additional validation
        errors = validate_config(cfg)

        if errors:
            print_error("Configuration validation failed:")
            for error in errors:
                click.echo(f"  • {error}", err=True)
            click.get_current_context().exit(1)
        else:
            print_success(f"Configuration is valid: {config_file}")

    except ValidationError as e:
        print_error("Configuration validation failed:")
        for error in e.errors():
            location = " → ".join(str(loc) for loc in error["loc"])
            click.echo(f"  • {location}: {error['msg']}", err=True)
        click.get_current_context().exit(1)

    except Exception as e:
        print_error(f"Failed to validate configuration: {e}")
        click.get_current_context().exit(1)


@config.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output file (default: stdout)",
)
@click.option(
    "--comments",
    is_flag=True,
    default=False,
    help="Include comments explaining each field",
)
@click.option(
    "--no-redact",
    is_flag=True,
    default=False,
    help="Include sensitive values (API keys, etc.)",
)
@click.pass_context
def export(
    ctx: click.Context,
    output: Path | None,
    comments: bool,
    no_redact: bool,
) -> None:
    r"""Export current configuration to YAML.

    Exports the merged configuration (profile + file + env) to a YAML file.

    \b
    Examples:
        $ bead config export
        $ bead config export --output my-config.yaml
        $ bead config export --comments  # Include field explanations
        $ bead config export --no-redact --output full-config.yaml
    """
    config_file = ctx.obj.get("config_file")
    profile = ctx.obj.get("profile", "default")
    verbose = ctx.obj.get("verbose", False)

    try:
        cfg = load_config_for_cli(
            config_file=str(config_file) if config_file else None,
            profile=profile,
            verbose=verbose,
        )

        # Convert to dict
        config_dict = cfg.model_dump()

        # Redact sensitive values unless --no-redact
        if not no_redact:
            config_dict = redact_sensitive_values(config_dict)

        # Add comments if requested
        yaml_content = _generate_yaml_with_comments(config_dict) if comments else None

        # Save or print
        if output:
            if yaml_content:
                output.write_text(yaml_content)
            else:
                # Write config dict directly to YAML file
                with open(output, "w") as f:
                    yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            print_success(f"Configuration exported to: {output}")
        else:
            if yaml_content:
                click.echo(yaml_content)
            else:
                click.echo(
                    yaml.dump(config_dict, default_flow_style=False, sort_keys=False)
                )

    except Exception as e:
        print_error(f"Failed to export configuration: {e}")


@config.command()
def profiles() -> None:
    r"""List available configuration profiles.

    Shows all built-in profiles with descriptions.

    \b
    Examples:
        $ bead config profiles
    """
    available_profiles = list_profiles()

    print_info("Available configuration profiles:")
    click.echo()

    for profile_name in available_profiles:
        click.echo(f"  • {profile_name}")

    click.echo()
    print_info("Use --profile to select a profile:")
    click.echo("  $ bead --profile dev config show")


def _generate_yaml_with_comments(config_dict: dict[str, Any]) -> str:
    """Generate YAML with comments explaining fields.

    Parameters
    ----------
    config_dict : dict[str, Any]
        Configuration dictionary.

    Returns
    -------
    str
        YAML content with comments.
    """
    lines = ["# bead Configuration", "# Generated with comments", ""]

    # Add commented sections
    sections = {
        "profile": "Configuration profile (default, dev, prod, test)",
        "logging": "Logging configuration (level, format, file)",
        "paths": "Path configuration (directories for data, models, cache)",
        "resources": "Resource management (auto-download, caching, language)",
        "templates": "Template filling (strategy, constraints, MLM settings)",
        "models": "Model configuration (default models, GPU, API keys)",
        "items": "Item construction (validation, auto-save)",
        "lists": "List construction (partitioning, balancing)",
        "deployment": "Deployment configuration (platform, jsPsych, plugins)",
        "training": "Training configuration (framework, hyperparameters)",
    }

    for section, description in sections.items():
        if section in config_dict:
            lines.append(f"# {description}")
            section_yaml = yaml.dump(
                {section: config_dict[section]},
                default_flow_style=False,
                sort_keys=False,
            )
            lines.append(section_yaml.rstrip())
            lines.append("")

    return "\n".join(lines)
