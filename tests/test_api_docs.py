"""Test Python code blocks in API documentation.

Uses pytest-examples to extract and test code blocks from markdown files.
"""

import os
import sys
from pathlib import Path

import pytest
from pytest_examples import CodeExample, EvalExample, find_examples

# Path to API documentation
DOCS_DIR = Path(__file__).parent.parent / "docs" / "user-guide" / "api"

# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "api_docs"

# Path to gallery (for importing gallery utils)
GALLERY_DIR = Path(__file__).parent.parent / "gallery" / "eng" / "argument_structure"


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up environment for executing code examples."""
    # Add gallery to sys.path so we can import utils
    if str(GALLERY_DIR) not in sys.path:
        sys.path.insert(0, str(GALLERY_DIR))

    # Change to fixtures directory so relative paths work
    original_dir = os.getcwd()
    os.chdir(FIXTURES_DIR)

    yield

    # Restore original directory
    os.chdir(original_dir)


@pytest.mark.parametrize("example", find_examples(DOCS_DIR), ids=str)
def test_api_docs_code_blocks(example: CodeExample, eval_example: EvalExample) -> None:
    """Test that code blocks in API docs are syntactically valid and executable.

    This uses pytest-examples to:
    1. Extract Python code blocks from markdown
    2. Check syntax validity via linting (black + ruff)
    3. Execute code blocks to verify they actually work
    4. When --update-examples is used, format code blocks in place

    Parameters
    ----------
    example : CodeExample
        The code example extracted from markdown
    eval_example : EvalExample
        The evaluator fixture provided by pytest-examples
    """
    # Ignore D100 (module docstrings), D102 (method docstrings), F821 (undefined names),
    # F401 (unused imports), E402 (imports not at top), I001 (import sorting) - these are
    # isolated documentation snippets showing specific concepts, not complete scripts
    eval_example.set_config(ruff_ignore=["D100", "D102", "F821", "F401", "E402", "I001"])

    # When --update-examples is passed, format and update print statements
    # Otherwise, lint and execute to verify the code actually runs
    if eval_example.update_examples:
        eval_example.format(example)
        eval_example.run_print_update(example)
    else:
        eval_example.lint(example)
        # Execute the code to verify it works
        eval_example.run(example)
