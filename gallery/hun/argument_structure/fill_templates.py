#!/usr/bin/env python3
"""Fill Hungarian Stage-1 argument-structure templates safely and systematically.

This filler processes one matrix verb lemma at a time instead of giving the
entire Hungarian verb lexicon to MixedFillingStrategy at once.

The intended structure is:

    VERB LEMMAS × STAGE-1 FRAME STRUCTURES

All UniMorph forms belonging to the current verb lemma are preserved.
Controlled support verbs such as történik can also be made available to
templates that require them.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from bead.resources.lexical_item import LexicalItem
from bead.resources.lexicon import Lexicon
from bead.resources.template_collection import TemplateCollection
from bead.templates.adapters.cache import ModelOutputCache
from bead.templates.adapters.huggingface import HuggingFaceMLMAdapter
from bead.templates.filler import FilledTemplate
from bead.templates.resolver import ConstraintResolver
from bead.templates.strategies import MixedFillingStrategy


logger = logging.getLogger(__name__)


def normalize_surface(text: str) -> str:
    """Normalize whitespace without changing Hungarian word boundaries."""
    text = " ".join(text.split())
    for punctuation in (".", ",", "?", "!"):
        text = text.replace(f" {punctuation}", punctuation)
    return text


def _surface_is_standalone(text: str, surface: str) -> bool:
    """Return True when a lexical surface form occurs as its own word/phrase."""
    import re
    return re.search(rf"(?<!\w){re.escape(surface)}(?!\w)", text, flags=re.UNICODE) is not None


def validate_rendered_surface(template_string: str, slot_fillers: Dict[str, LexicalItem], rendered: str) -> None:
    """Fail fast on rendering errors that would contaminate acceptability data.

    This does *not* judge whether a verb licenses the requested frame. It only
    checks that the requested lexical fillers were rendered as separate words,
    with no unresolved placeholders or broken punctuation boundaries.
    """
    if "{" in rendered or "}" in rendered:
        raise ValueError(f"Unresolved placeholder in rendered sentence: {rendered!r}")
    if "," in rendered and ", " not in rendered:
        raise ValueError(f"Missing space after comma: {rendered!r}")

    for slot_name, item in slot_fillers.items():
        surface = item.form or item.lemma
        if not _surface_is_standalone(rendered, surface):
            raise ValueError(
                f"Broken word boundary for slot {slot_name!r} ({surface!r}) in {rendered!r}; "
                f"template={template_string!r}"
            )


def render_template_hun(template_string: str, slot_fillers: Dict[str, LexicalItem]) -> str:
    """Render Hungarian placeholders with strict, deterministic word boundaries."""
    surfaces = {name: (item.form or item.lemma) for name, item in slot_fillers.items()}
    try:
        result = template_string.format_map(surfaces)
    except KeyError as exc:
        raise ValueError(f"No filler supplied for template slot {exc.args[0]!r}: {template_string!r}") from exc

    result = normalize_surface(result)
    validate_rendered_surface(template_string, slot_fillers, result)
    return result


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate the YAML configuration."""
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML mapping in {config_path}, got {type(config).__name__}")

    return config


def find_repo_root(start: Path) -> Path:
    """Locate the repository root containing `bead`."""
    start = start.resolve()

    for candidate in (start, *start.parents):
        if (candidate / "bead").is_dir():
            return candidate

    return start


def resolve_path(raw_path: str | Path, *, config_dir: Path, repo_root: Path) -> Path:
    """Resolve config paths relative to the config, repository, or current directory."""
    path = Path(raw_path).expanduser()

    if path.is_absolute():
        return path

    candidates = [config_dir / path, repo_root / path, Path.cwd() / path]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return (config_dir / path).resolve()


def select_dry_run_templates(templates: List[Any]) -> List[Any]:
    """Select five representative Hungarian Stage-1 templates."""
    preferred_names = ["subj_nom-verb.", "tr-indef.", "tr-def.", "tr_dat-indef.", "subj_nom-verb-hogy_ind."]
    by_name = {template.name: template for template in templates}
    selected = [by_name[name] for name in preferred_names if name in by_name]
    return selected or templates[:5]


def ordered_verb_lemmas(lexicon: Lexicon) -> List[str]:
    """Return unique verb lemmas in their original lexicon order."""
    lemmas: List[str] = []
    seen = set()

    for item in lexicon.items.values():
        if item.lemma not in seen:
            seen.add(item.lemma)
            lemmas.append(item.lemma)

    return lemmas


def make_verb_subset(full_verb_lexicon: Lexicon, lemmas: set[str]) -> Lexicon:
    """Create a temporary verb lexicon containing all forms of selected lemmas."""
    items = {item_id: item for item_id, item in full_verb_lexicon.items.items() if item.lemma in lemmas}

    return Lexicon(
        name="verbs",
        description=f"Temporary Hungarian verb subset for: {', '.join(sorted(lemmas))}",
        language_code="hun",
        items=items,
    )


def template_support_lemmas(template: Any) -> set[str]:
    """Return controlled support verb lemmas required by a template."""
    support = set()

    if "comp_verb" in template.slots:
        support.add("történik")

    return support


def combination_has_matrix_verb(combination: Dict[str, LexicalItem], matrix_lemma: str) -> bool:
    """Require the matrix verb slot to contain the matrix lemma currently being tested."""
    matrix_verb = combination.get("verb")

    if matrix_verb is None:
        return True

    return matrix_verb.lemma == matrix_lemma


def take_valid_combinations(generated: Iterable[Dict[str, LexicalItem]], matrix_lemma: str, limit: int) -> List[Dict[str, LexicalItem]]:
    """Take at most limit combinations containing the intended matrix verb."""
    results = []

    for combination in generated:
        if not combination_has_matrix_verb(combination, matrix_lemma):
            continue

        results.append(combination)

        if len(results) >= limit:
            break

    return results


def build_slot_strategies(config: Dict[str, Any], *, config_dir: Path, repo_root: Path) -> Dict[str, tuple[str, Dict[str, Any]]]:
    """Create mixed-strategy configuration and initialize MLM support only when needed."""
    template_config = config["template"]
    configured_slots = template_config.get("slot_strategies", {})

    uses_mlm = any(slot_config.get("strategy") == "mlm" for slot_config in configured_slots.values())

    resolver = ConstraintResolver()
    model_adapter = None
    cache = None
    mlm_config = template_config.get("mlm", {})

    if uses_mlm:
        model_name = mlm_config["model_name"]
        logger.info("Loading Hungarian MLM model: %s", model_name)

        model_adapter = HuggingFaceMLMAdapter(model_name=model_name, device=mlm_config.get("device", "cpu"))
        model_adapter.load_model()

        cache_raw = config.get("paths", {}).get("cache_dir", ".cache")
        cache_dir = resolve_path(cache_raw, config_dir=config_dir, repo_root=repo_root)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = ModelOutputCache(cache_dir=cache_dir)

        logger.info("Hungarian MLM model loaded")
    else:
        logger.info("No MLM slot strategy configured; skipping model loading")

    result: Dict[str, tuple[str, Dict[str, Any]]] = {}

    for slot_name, slot_config in configured_slots.items():
        strategy_name = slot_config["strategy"]

        if strategy_name == "mlm":
            strategy_config: Dict[str, Any] = {
                "resolver": resolver,
                "model_adapter": model_adapter,
                "cache": cache,
                "beam_size": mlm_config.get("beam_size", 5),
                "top_k": mlm_config.get("top_k", 10),
            }

            for optional_key in ("max_fills", "enforce_unique"):
                if optional_key in slot_config:
                    strategy_config[optional_key] = slot_config[optional_key]

            result[slot_name] = ("mlm", strategy_config)
        else:
            result[slot_name] = (strategy_name, {})

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill Hungarian Stage-1 argument-structure templates")

    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"), help="Path to config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Use five representative templates and five verb lemmas")
    parser.add_argument("--template-limit", type=int, default=None, help="Process only the first N templates")
    parser.add_argument("--verb-start", type=int, default=0, help="Zero-based index of the first verb lemma to process")
    parser.add_argument("--verb-count", type=int, default=None, help="Number of verb lemmas to process starting at --verb-start")
    parser.add_argument("--verb-limit", type=int, default=None, help="Backward-compatible alias for --verb-count starting at lemma 0")
    parser.add_argument("--max-per-template", "--max-per-verb-template", dest="max_per_verb_template", type=int, default=3, help="Maximum outputs retained for each verb-template combination")
    parser.add_argument("--append", action="store_true", help="Append to the existing output JSONL instead of overwriting it")
    parser.add_argument("--output", type=Path, default=None, help="Override configured output path")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Log a generation error and continue. Default is fail-fast, which is safer before data collection.",
    )

    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    config_dir = config_path.parent
    repo_root = find_repo_root(config_dir)
    config = load_config(config_path)

    logging_config = config.get("logging", {})
    level_name = str(logging_config.get("level", "INFO")).upper()

    logging.basicConfig(level=getattr(logging, level_name, logging.INFO), format=logging_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger("bead.templates.strategies").setLevel(logging.WARNING)

    templates_raw = config["resources"]["templates"][0]["path"]
    templates_path = resolve_path(templates_raw, config_dir=config_dir, repo_root=repo_root)

    if not templates_path.exists():
        raise FileNotFoundError(f"Template file not found: {templates_path}")

    logger.info("Loading templates from %s", templates_path)

    template_collection = TemplateCollection.from_jsonl(templates_path, "generic_frames")
    templates = list(template_collection.templates.values())

    logger.info("Loaded %d templates", len(templates))

    if args.dry_run:
        templates = select_dry_run_templates(templates)
        logger.info("DRY RUN: selected %d templates", len(templates))

        for template in templates:
            logger.info("  %s", template.name)

    if args.template_limit is not None:
        templates = templates[:args.template_limit]
        logger.info("Template limit applied: %d", len(templates))

    full_verb_lexicon: Lexicon | None = None
    nonverb_lexicons: List[Lexicon] = []

    for lex_config in config["resources"]["lexicons"]:
        lex_path = resolve_path(lex_config["path"], config_dir=config_dir, repo_root=repo_root)

        if not lex_path.exists():
            raise FileNotFoundError(f"Lexicon file not found: {lex_path}")

        lexicon = Lexicon.from_jsonl(lex_path, lex_config["name"])

        logger.info("Loaded %d items from %s", len(lexicon.items), lex_config["name"])

        if lex_config["name"] == "verbs":
            full_verb_lexicon = lexicon
        else:
            nonverb_lexicons.append(lexicon)

    if full_verb_lexicon is None:
        raise RuntimeError("The config does not define a 'verbs' lexicon.")

    all_lemmas = ordered_verb_lemmas(full_verb_lexicon)

    logger.info("Full verb lexicon contains %d lemmas and %d forms", len(all_lemmas), len(full_verb_lexicon.items))

    if args.verb_limit is not None:
        if args.verb_count is not None:
            raise ValueError("Use either --verb-limit or --verb-count, not both.")

        args.verb_start = 0
        args.verb_count = args.verb_limit

    if args.dry_run:
        args.verb_start = 0
        args.verb_count = 5
        args.max_per_verb_template = min(args.max_per_verb_template, 3)

    start = max(args.verb_start, 0)

    if args.verb_count is None:
        end = len(all_lemmas)
    else:
        end = min(start + args.verb_count, len(all_lemmas))

    selected_lemmas = all_lemmas[start:end]

    if not selected_lemmas:
        raise ValueError(f"No verb lemmas selected: start={start}, count={args.verb_count}")

    logger.info("Selected verb lemmas %d:%d (%d lemmas)", start, end, len(selected_lemmas))

    slot_strategies = build_slot_strategies(config, config_dir=config_dir, repo_root=repo_root)
    strategy = MixedFillingStrategy(slot_strategies=slot_strategies)

    output_raw = args.output or config["template"]["output_path"]
    output_path = resolve_path(output_raw, config_dir=config_dir, repo_root=repo_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if args.append else "w"

    total_written = 0
    successful_verb_template_pairs = 0
    empty_verb_template_pairs = 0

    with output_path.open(mode, encoding="utf-8") as output_file:
        for verb_index, matrix_lemma in enumerate(selected_lemmas, start=start):
            logger.info("VERB %d/%d: %s", verb_index + 1, len(all_lemmas), matrix_lemma)

            for template_index, template in enumerate(templates, start=1):
                support_lemmas = template_support_lemmas(template)
                required_lemmas = {matrix_lemma, *support_lemmas}
                temporary_verbs = make_verb_subset(full_verb_lexicon, required_lemmas)
                lexicons = [temporary_verbs, *nonverb_lexicons]

                try:
                    generated: Iterable[Dict[str, LexicalItem]] = strategy.generate_from_template(template=template, lexicons=lexicons, language_code="hun")
                    combinations = take_valid_combinations(generated=generated, matrix_lemma=matrix_lemma, limit=args.max_per_verb_template)

                    if not combinations:
                        empty_verb_template_pairs += 1
                        logger.debug("No fills: verb=%s template=%s", matrix_lemma, template.name)
                        continue

                    successful_verb_template_pairs += 1

                    for combination in combinations:
                        rendered = render_template_hun(template.template_string, combination)

                        filled = FilledTemplate(
                            template_id=str(template.id),
                            template_name=template.name,
                            slot_fillers=combination,
                            rendered_text=rendered,
                            strategy_name="mixed",
                            template_slots={name: slot.required for name, slot in template.slots.items()},
                        )

                        output_file.write(filled.model_dump_json() + "\n")
                        total_written += 1

                except Exception:
                    logger.exception("Failed: verb=%s template=%s", matrix_lemma, template.name)
                    if not args.continue_on_error:
                        raise

            output_file.flush()

    logger.info("=" * 80)
    logger.info("FILLING COMPLETE")
    logger.info("=" * 80)
    logger.info("Verb lemmas processed: %d", len(selected_lemmas))
    logger.info("Templates tested per verb: %d", len(templates))
    logger.info("Verb × template pairs tested: %d", len(selected_lemmas) * len(templates))
    logger.info("Pairs producing at least one sentence: %d", successful_verb_template_pairs)
    logger.info("Pairs producing zero sentences: %d", empty_verb_template_pairs)
    logger.info("Filled sentences written: %d", total_written)
    logger.info("Saved to: %s", output_path)


if __name__ == "__main__":
    main()