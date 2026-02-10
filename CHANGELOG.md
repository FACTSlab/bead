# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-02-09

### Added

#### Span Labeling Data Model (`bead.items`)

- **Span**, **SpanLabel**, **SpanSegment** models for stand-off token-level annotation
- **SpanSpec** for defining label vocabularies and relation types
- **SpanRelation** for directed labeled relations between spans
- `add_spans_to_item()` composability function for attaching spans to any item type
- Prompt span references: `[[label]]` and `[[label:text]]` template syntax
  - Auto-fills span token text or uses explicit display text
  - Colors match between stimulus highlighting and prompt highlighting
  - Resolved Python-side at trial generation; plugins receive pre-rendered HTML
  - Early validation warning in `add_spans_to_item()`, hard validation at trial generation

#### Tokenization (`bead.tokenization`)

- **Token** model with `text`, `whitespace`, `index`, `token_space_after` fields
- **TokenizedText** container with token-level access and reconstruction
- Tokenizer backends: whitespace (default), spaCy, Stanza
- Lazy imports for optional NLP dependencies

#### jsPsych Plugins (`bead.deployment.jspsych`)

- 8 new TypeScript plugins following the `JsPsychPlugin` pattern:
  - **bead-binary-choice**: two-alternative forced choice with keyboard support
  - **bead-categorical**: labeled category selection (radio buttons)
  - **bead-free-text**: open-ended text input with optional word count
  - **bead-magnitude**: numeric magnitude estimation with reference stimulus
  - **bead-multi-select**: checkbox-based multi-selection with min/max constraints
  - **bead-slider-rating**: continuous slider with labeled endpoints
  - **bead-rating**: Likert-scale ordinal rating with keyboard shortcuts
  - **bead-span-label**: interactive span highlighting with label assignment, relations, and search
- **span-renderer** library for token-level span highlighting with overlap support
- **gallery-bundle** IIFE build aggregating all plugins for standalone HTML demos
- Keyboard navigation support in forced-choice, rating, and binary-choice plugins
- Material Design styling with responsive layout

#### Deployment Pipeline

- `SpanDisplayConfig` with `color_palette` and `dark_color_palette` for consistent span coloring
- `SpanColorMap` dataclass for deterministic color assignment (same label = same color pair)
- `_assign_span_colors()` shared between stimulus and prompt renderers
- `_generate_span_stimulus_html()` for token-level highlighting in deployed experiments
- Prompt span reference resolution integrated into all 5 composite trial creators (likert, slider, binary, forced-choice, span-labeling)
- Deployment CSS for `.bead-q-highlight`, `.bead-q-chip`, `.bead-span-subscript` in experiment template

#### Interactive Gallery

- 17 demo pages using stimuli from MegaAcceptability, MegaVeridicality, and Semantic Proto-Roles
- Demos cover all plugin types and composite span+task combinations
- Gallery documentation with tabbed Demo / Python / Trial JSON views
- Standalone HTML demos with gallery-bundle.js (no build step required)

#### Tests

- 79 Python span-related tests (items, tokenization, deployment)
- 42 TypeScript tests (20 plugin + 22 span-renderer)
- Prompt span reference tests: parser, color assignment, resolver, integration

### Changed

- Trial generation now supports span-aware stimulus rendering for all task types
- Forced-choice and rating plugins updated with keyboard shortcut support
- Span-label plugin enhanced with searchable fixed labels, interactive relation creation, and relation cleanup on span deletion

## [0.1.0] - 2026-02-04

### Added

#### Core Pipeline (6 Stages)

- **Resources** (`bead.resources`): Lexical items and templates with linguistic features
  - LexicalItem, MultiWordExpression, Lexicon models
  - Template, TemplateSequence, TemplateTree, Slot models
  - Constraint DSL for slot, template, and cross-template constraints
  - Adapters: UniMorph (morphology), Glazing (VerbNet, PropBank, FrameNet)
  - AdapterCache for caching resource adapter results

- **Templates** (`bead.templates`): Template filling and stimulus generation
  - CSPFiller with backtracking and forward checking
  - Strategies: exhaustive, random, stratified
  - ConstraintResolver for DSL-based constraint evaluation
  - HuggingFace MLM adapter for model-based slot ranking
  - Streaming iterator-based generation for large datasets

- **Items** (`bead.items`): Experimental item construction
  - 8 task types: binary, forced-choice, categorical, cloze, free-text, ordinal-scale, magnitude, multi-select
  - ItemTemplate with chunking, timing, and parsing modes
  - Model adapters: HuggingFace (LM, MLM, NLI, SentenceTransformer), OpenAI, Anthropic, Google, TogetherAI
  - ModelOutputCache for efficient caching
  - Rate limiting and retry-with-backoff for API calls

- **Lists** (`bead.lists`): List partitioning with constraint satisfaction
  - ExperimentList and ListCollection models
  - Constraints: uniqueness, balance, quantile, grouped-quantile, diversity, size, ordering, conditional-uniqueness
  - Partitioner and Balancer for balanced assignment
  - JSONL serialization via `to_jsonl()` and `from_jsonl()`

- **Deployment** (`bead.deployment`): Web experiment generation
  - jsPsych 8.x experiment generator with Material Design UI
  - JATOS batch exporter with server-side list distribution
  - 8 distribution strategies: random, sequential, balanced, latin-square, stratified, weighted-random, quota-based, metadata-based
  - Demographics, instructions, and rating scale configuration

- **Training** (`bead.active_learning`): Active learning with convergence detection
  - ActiveLearningLoop orchestrator
  - UncertaintySampler (entropy-based) and RandomSelector
  - 8 task-specific models matching item types
  - Random effects support with participant-level intercepts and slopes
  - HuggingFace and PyTorch Lightning trainers
  - Mixed effects training support

#### Supporting Modules

- **Simulation** (`bead.simulation`): Synthetic judgment generation
  - Annotators: LM-based, random, oracle, distance-based
  - Noise models: temperature, random, systematic
  - Task-specific strategies for all 8 item types
  - SimulationRunner for multi-annotator simulation

- **Evaluation** (`bead.evaluation`): Performance assessment
  - ConvergenceDetector with statistical significance testing
  - InterAnnotatorMetrics: percentage agreement, Cohen's kappa, Fleiss' kappa, Krippendorff's alpha

- **Behavioral** (`bead.behavioral`): Behavioral analytics via slopit
  - JudgmentAnalytics and ParticipantBehavioralSummary models
  - Keystroke, focus, timing, and paste detection analysis
  - Quality control filtering and exclusion list generation

- **Participants** (`bead.participants`): Participant metadata
  - UUID-based participant identification
  - Privacy-preserving external ID mapping
  - Configurable metadata schema validation
  - Merge utilities for pandas and polars

- **Data Collection** (`bead.data_collection`): Platform integration
  - JATOSDataCollector with authentication
  - ProlificDataCollector with webhook support
  - DataMerger for multi-source data

- **DSL** (`bead.dsl`): Constraint domain-specific language
  - Lark-based parser with AST construction
  - Cached evaluation with variable scoping
  - Standard library: string, math, collection, logic functions

- **Config** (`bead.config`): Configuration system
  - YAML-based configuration with environment variable support
  - Profiles: default, dev, prod, test
  - Validation and merging utilities

#### CLI

- `bead init`: Project scaffolding
- `bead config`: Configuration management (show, validate, export, profiles)
- `bead resources`: Resource loading and inspection
- `bead templates`: Template filling
- `bead items`: Item construction
- `bead lists`: List partitioning
- `bead deploy`: jsPsych/JATOS export
- `bead simulate`: Annotation simulation
- `bead training`: Active learning loop
- `bead workflow`: Pipeline orchestration
- `bead shell`: Interactive REPL

#### Infrastructure

- Python 3.13+ with full type annotations
- Pydantic v2 validation
- TypeScript plugins for jsPsych with Biome linting
- MkDocs documentation with mkdocstrings
- CI/CD: GitHub Actions for testing, docs, PyPI publishing
- Read the Docs integration

[Unreleased]: https://github.com/FACTSlab/bead/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/FACTSlab/bead/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/FACTSlab/bead/releases/tag/v0.1.0
