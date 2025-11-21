# Installation

## Requirements

- Python 3.13 or higher
- Operating Systems: macOS, Linux, Windows (WSL recommended)

## Install from PyPI

Install the latest stable release:

```bash
pip install bead
```

## Install from Source

For the latest development version:

```bash
git clone https://github.com/aaronstevenwhite/bead.git
cd bead
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e "."
```

## Development Installation

For contributing to bead, install with development dependencies:

```bash
pip install -e ".[dev,api,training]"
```

This installs:
- `dev`: testing and linting tools (pytest, ruff, pyright)
- `api`: model adapters (OpenAI, Anthropic, Google, HuggingFace)
- `training`: active learning dependencies (PyTorch, transformers)

## Verify Installation

Check that bead is installed correctly:

```bash
python -c "import bead; print(bead.__version__)"
```

Or use the CLI:

```bash
bead --version
```

## Optional Dependencies

Install specific adapters as needed:

```bash
# HuggingFace models for template filling and active learning
pip install bead[api]

# Active learning with PyTorch
pip install bead[training]

# All dependencies
pip install bead[dev,api,training]
```

## Troubleshooting

### Python Version

Verify you have Python 3.13+:

```bash
python --version
```

If not, install from [python.org](https://www.python.org/downloads/) or use pyenv:

```bash
pyenv install 3.13.0
pyenv local 3.13.0
```

### Virtual Environment

Always use a virtual environment to avoid conflicts:

```bash
python3.13 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'bead'`
**Solution**: Ensure you activated the virtual environment where bead is installed.

**Issue**: `ImportError` for optional dependencies
**Solution**: Install the required extra, e.g., `pip install bead[api]`

## Next Steps

Continue to the [Quick Start](quickstart.md) guide for a complete tutorial.
