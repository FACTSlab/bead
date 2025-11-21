# bead

**A Python framework for constructing, deploying, and analyzing large-scale linguistic judgment experiments with active learning.**

## Overview

bead implements a 6-stage pipeline for linguistic experiment design:

1. **Resources**: lexical items and templates with constraints
2. **Templates**: template filling strategies (exhaustive, random, stratified, MLM, mixed)
3. **Items**: experimental item construction (8 task types)
4. **Lists**: list partitioning with constraint satisfaction
5. **Deployment**: jsPsych 8.x batch experiment generation for JATOS
6. **Training**: active learning with GLMM support and convergence detection

## Key Features

- **Stand-off annotation** with UUID-based references for provenance tracking
- **8 task types**: forced-choice, ordinal scale, binary, categorical, multi-select, magnitude, free text, cloze
- **GLMM support**: Generalized Linear Mixed Models with random effects
- **Batch deployment**: server-side list distribution via JATOS batch sessions
- **Language-agnostic**: works with any language supported by UniMorph
- **Configuration-first**: single YAML file orchestrates entire pipeline
- **Type-safe**: full Python 3.13 type hints with Pydantic v2 validation

## Quick Links

- [Installation](installation.md): get started in 5 minutes
- [Quick Start](quickstart.md): complete tutorial in 15 minutes
- [User Guide](user-guide/concepts.md): in-depth documentation
- [API Reference](api/resources.md): complete API documentation
- [Examples](examples/gallery.md): gallery of example projects

## Installation

```bash
pip install bead
```

For development installation:

```bash
git clone https://github.com/aaronstevenwhite/bead.git
cd bead
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,api,training]"
```

## Requirements

- Python 3.13+
- Operating Systems: macOS, Linux, Windows (WSL recommended)

## Citation

If you use bead in your research, please cite:

```
[Citation information to be added]
```

## License

[License information to be added]
