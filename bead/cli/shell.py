"""Interactive CLI shell using Click and Prompt Toolkit.

This module provides an interactive REPL shell for the bead CLI with
autocomplete, command history, and rich formatting. Commands are executed
through Click's CLI system, providing full integration with all bead commands.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import TYPE_CHECKING

import click
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.styles import Style
from pygments.lexers.shell import BashLexer
from rich.console import Console
from rich.markdown import Markdown

if TYPE_CHECKING:
    from collections.abc import Iterable

console = Console()


# Create shell command group
@click.group()
def shell() -> None:
    r"""Interactive shell for bead CLI.

    Provides an interactive REPL with autocomplete, history, and rich formatting.
    """
    pass


class BeadCompleter(Completer):
    """Command completer for bead shell."""

    def __init__(self, commands: list[str]) -> None:
        """Initialize completer with available commands.

        Parameters
        ----------
        commands : list[str]
            List of available commands.
        """
        self.commands = commands
        # Initialize subcommands dict - will be populated lazily
        self.subcommands: dict[str, list[str]] = {}
        self._subcommands_loaded = False

    def get_completions(
        self, document: object, complete_event: object
    ) -> Iterable[Completion]:
        """Get completions for current input.

        Parameters
        ----------
        document : object
            Current document/input.
        complete_event : object
            Completion event.

        Yields
        ------
        Completion
            Completion suggestions.
        """
        # Access text_before_cursor attribute safely
        text = getattr(document, "text_before_cursor", "")
        words = text.split()

        # Lazy load subcommands to avoid circular import
        if not self._subcommands_loaded:
            try:
                from bead.cli.main import cli  # noqa: PLC0415

                for cmd_name, cmd_obj in cli.commands.items():
                    if hasattr(cmd_obj, "commands"):
                        # It's a group, get its subcommands
                        self.subcommands[cmd_name] = list(cmd_obj.commands.keys())
                self._subcommands_loaded = True
            except Exception:
                # Fallback to hardcoded list if import fails
                self.subcommands = {
                    "resources": ["create", "list", "validate", "import"],
                    "templates": ["fill", "list", "validate"],
                    "items": ["create", "list", "validate", "stats"],
                    "lists": ["partition", "list", "validate"],
                    "deployment": ["generate", "validate", "export"],
                    "training": ["train", "evaluate", "predict"],
                    "active-learning": ["select-items", "run", "monitor-convergence"],
                    "config": ["show", "validate", "edit"],
                    "workflow": ["run", "status", "resume", "rollback"],
                }
                self._subcommands_loaded = True

        if len(words) == 0:
            # Complete main commands
            for cmd in self.commands:
                if cmd.startswith(text):
                    yield Completion(cmd, start_position=-len(text))
        elif len(words) == 1:
            # Complete subcommands
            cmd = words[0]
            if cmd in self.subcommands:
                for subcmd in self.subcommands[cmd]:
                    yield Completion(subcmd, start_position=0)
        else:
            # File path completion
            last_word = words[-1]
            if "/" in last_word or last_word.startswith("."):
                try:
                    path = Path(last_word).parent if "/" in last_word else Path(".")
                    prefix = Path(last_word).name
                    if path.exists():
                        for item in path.iterdir():
                            if item.name.startswith(prefix):
                                suffix = "/" if item.is_dir() else ""
                                yield Completion(
                                    str(item.name) + suffix,
                                    start_position=-len(prefix),
                                )
                except Exception:
                    pass


# Shell style
shell_style = Style.from_dict(
    {
        "prompt": "bold cyan",
        "command": "bold green",
        "error": "bold red",
    }
)


_DEFAULT_HISTORY_FILE = Path.home() / ".bead_history"


@shell.command()
@click.option(
    "--history",
    type=click.Path(path_type=Path),
    default=_DEFAULT_HISTORY_FILE,
    help="Path to history file",
)
def repl(history_file: Path) -> None:
    """Start interactive REPL shell.

    Provides an interactive command-line interface with:
    - Command autocomplete
    - Command history
    - Rich formatting
    - Multi-line input support

    Parameters
    ----------
    history_file : Path
        Path to history file for command persistence.

    Examples
    --------
    Start shell:
        $ bead shell

    Start shell with custom history:
        $ bead shell --history ~/.my_bead_history
    """
    console.print("[bold cyan]bead Interactive Shell[/bold cyan]")
    console.print("[dim]Type 'help' for available commands, 'exit' to quit[/dim]\n")

    # Available commands - lazy import to avoid circular dependency
    commands: list[str]
    try:
        # Import here to avoid circular import at module level
        from bead.cli.main import cli as main_cli  # noqa: PLC0415

        commands = list(main_cli.commands.keys()) + ["help", "exit", "quit", "clear"]
    except Exception:
        # Fallback to hardcoded list if import fails
        commands = [
            "resources",
            "templates",
            "items",
            "lists",
            "deployment",
            "training",
            "active-learning",
            "config",
            "workflow",
            "help",
            "exit",
            "quit",
        ]

    # Create completer
    completer = BeadCompleter(commands)

    # Create history
    history = FileHistory(str(history_file)) if history_file else None

    # Create session
    session = PromptSession(
        history=history,
        completer=completer,
        auto_suggest=AutoSuggestFromHistory(),
        lexer=PygmentsLexer(BashLexer),
        style=shell_style,
        complete_while_typing=True,
    )

    # Main loop
    while True:
        try:
            # Get user input
            text = session.prompt("bead> ")

            if not text.strip():
                continue

            # Handle built-in commands
            if text.strip() in ("exit", "quit"):
                console.print("[green]Goodbye![/green]")
                break
            elif text.strip() == "help":
                _show_help()
                continue
            elif text.strip().startswith("clear"):
                console.clear()
                continue

            # Execute command
            _execute_command(text)

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
        except EOFError:
            console.print("\n[green]Goodbye![/green]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def _show_help() -> None:
    """Show help message."""
    help_text = """
# bead Interactive Shell

## Available Commands

- **resources** - Manage lexicons and templates
- **templates** - Fill and manage templates
- **items** - Create and manage experimental items
- **lists** - Partition items into experiment lists
- **deployment** - Generate and deploy experiments
- **training** - Train models with active learning
- **active-learning** - Active learning commands
- **config** - Configuration management
- **workflow** - End-to-end pipeline workflows

## Built-in Commands

- **help** - Show this help message
- **exit** / **quit** - Exit the shell
- **clear** - Clear the screen

## Usage

Type commands as you would in the regular CLI, but without the 'bead' prefix:

    bead> resources list
    bead> templates fill --strategy exhaustive
    bead> items create --template template.jsonl

## Features

- **Autocomplete**: Press TAB to autocomplete commands
- **History**: Use arrow keys to navigate command history
- **Multi-line**: Use backslash for line continuation
    """
    console.print(Markdown(help_text))


def _execute_command(text: str) -> None:
    """Execute a command using Click's CLI system.

    Parameters
    ----------
    text : str
        Command text to execute.
    """
    if not text.strip():
        return

    # Parse command using shlex to handle quoted arguments properly
    try:
        parts = shlex.split(text)
    except ValueError as e:
        console.print(f"[red]Error parsing command: {e}[/red]")
        return

    if not parts:
        return

    # Import main CLI group
    from bead.cli.main import cli  # noqa: PLC0415

    # Create a context with default options
    ctx_obj: dict[str, object] = {
        "config_file": None,
        "profile": "default",
        "verbose": False,
        "quiet": False,
    }

    # Build command line arguments
    # Click expects sys.argv format, so we need to prepend the program name
    # and handle the command parts
    try:
        # Create a context for the main CLI group
        with cli.make_context("bead", list(parts), obj=ctx_obj) as ctx:
            # Invoke the command
            cli.invoke(ctx)
    except click.exceptions.Exit as e:
        # Click commands use ctx.exit() which raises Exit
        # Exit code 0 is success, non-zero is error
        if e.exit_code != 0:
            console.print(f"[red]Command failed with exit code {e.exit_code}[/red]")
    except click.exceptions.ClickException as e:
        # Click-specific exceptions (e.g., BadParameter, UsageError)
        console.print(f"[red]{e.format_message()}[/red]")
    except SystemExit as e:
        # Some commands may call sys.exit()
        if e.code and e.code != 0:
            console.print(f"[red]Command exited with code {e.code}[/red]")
    except Exception as e:
        # Catch-all for other exceptions
        console.print(f"[red]Error executing command: {e}[/red]")
        if ctx_obj.get("verbose", False):
            import traceback  # noqa: PLC0415

            console.print(f"[dim]{traceback.format_exc()}[/dim]")


# Register repl command
shell.add_command(repl)

# Entry point for direct execution
if __name__ == "__main__":
    shell()
