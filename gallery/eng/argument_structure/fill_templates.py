#!/usr/bin/env python3
"""Fill templates using configuration-driven MLM strategy.

This script loads templates and lexicons, creates a TemplateFiller with
MixedFillingStrategy using slot_strategies from config.yaml, and outputs
filled templates.

All parameters are configurable via config.yaml, with optional CLI overrides.
"""

import argparse
import logging
from pathlib import Path

import yaml

from bead.data.serialization import write_jsonlines
from bead.resources.lexicon import Lexicon
from bead.resources.template_collection import TemplateCollection
from bead.templates.adapters.cache import ModelOutputCache
from bead.templates.adapters.huggingface import HuggingFaceMLMAdapter
from bead.templates.filler import FilledTemplate
from bead.templates.resolver import ConstraintResolver
from bead.templates.strategies import MixedFillingStrategy

from utils.renderers import OtherNounRenderer

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    """Fill templates using config-driven MLM strategy."""
    parser = argparse.ArgumentParser(description="Fill templates with MLM strategy")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode: use 10 verbs with one simple and one progressive template (both with 3+ nouns)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output path from config",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config["logging"]["level"]),
        format=config["logging"]["format"],
    )

    logger.info("Loading templates and lexicons...")

    # Resolve paths
    templates_path = Path(config["resources"]["templates"][0]["path"])
    output_path = args.output or Path(config["template"]["output_path"])

    # Load templates
    template_collection = TemplateCollection.from_jsonl(
        templates_path, "generic_frames"
    )
    logger.info(
        f"Loaded {len(template_collection.templates)} templates from {templates_path}"
    )

    # Apply dry-run mode: select specific templates
    templates = list(template_collection.templates.values())
    if args.dry_run:
        logger.info("DRY RUN MODE: Selecting 1 simple + 1 progressive template with 3 noun slots")

        # Find templates with 3 noun slots
        def count_noun_slots(template):
            return sum(1 for slot_name in template.slots if slot_name.startswith('noun_'))

        templates_with_3_nouns = [t for t in templates if count_noun_slots(t) == 3]

        # Separate simple and progressive
        simple_templates = [t for t in templates_with_3_nouns if 'progressive' not in t.name]
        progressive_templates = [t for t in templates_with_3_nouns if 'progressive' in t.name]

        # Select one of each
        selected = []
        if simple_templates:
            selected.append(simple_templates[0])
            logger.info(f"  Simple template: {simple_templates[0].name}")
        if progressive_templates:
            selected.append(progressive_templates[0])
            logger.info(f"  Progressive template: {progressive_templates[0].name}")

        templates = selected if selected else templates[:2]
        logger.info(f"Selected {len(templates)} templates for dry run")

    # Load lexicons
    lexicons: list[Lexicon] = []
    for lex_config in config["resources"]["lexicons"]:
        lex_path = Path(lex_config["path"])
        lexicon = Lexicon.from_jsonl(lex_path, lex_config["name"])

        # In dry-run mode, limit verb lexicon to 10 verbs
        if args.dry_run and lex_config["name"] == "verbnet_verbs":
            # Get first 10 unique verb lemmas
            verb_lemmas = []
            limited_items = {}
            for item_id, item in lexicon.items.items():
                if item.lemma not in verb_lemmas:
                    verb_lemmas.append(item.lemma)
                    if len(verb_lemmas) >= 10:
                        break
                # Keep all forms of verbs we're including
                if item.lemma in verb_lemmas[:10]:
                    limited_items[item_id] = item

            lexicon.items = limited_items
            logger.info(
                f"DRY RUN: Limited verbnet_verbs to 10 lemmas ({len(lexicon.items)} forms)"
            )

        lexicons.append(lexicon)
        logger.info(f"Loaded {len(lexicon.items)} items from {lex_config['name']}")

    # Initialize constraint resolver
    resolver = ConstraintResolver()

    # Initialize model adapter for MLM
    mlm_config = config["template"]["mlm"]
    logger.info(f"Loading MLM model: {mlm_config['model_name']}...")
    model_adapter = HuggingFaceMLMAdapter(
        model_name=mlm_config["model_name"],
        device=mlm_config.get("device", "cpu"),
    )
    model_adapter.load_model()
    logger.info("MLM model loaded successfully")

    # Initialize cache
    cache_dir = Path(config["paths"]["cache_dir"])
    cache = ModelOutputCache(cache_dir=cache_dir)

    # Build slot_strategies dict for MixedFillingStrategy
    # Format: {slot_name: (strategy_name, config_dict)}
    slot_strategies: dict[str, tuple[str, dict]] = {}

    for slot_name, slot_config in config["template"]["slot_strategies"].items():
        strategy_name = slot_config["strategy"]

        if strategy_name == "mlm":
            # MLM strategy needs special config with resolver, model_adapter, etc.
            mlm_slot_config = {
                "resolver": resolver,
                "model_adapter": model_adapter,
                "cache": cache,
                "beam_size": mlm_config.get("beam_size", 5),
                "top_k": mlm_config.get("top_k", 10),
            }
            # Add per-slot max_fills and enforce_unique if specified
            if "max_fills" in slot_config:
                mlm_slot_config["max_fills"] = slot_config["max_fills"]
            if "enforce_unique" in slot_config:
                mlm_slot_config["enforce_unique"] = slot_config["enforce_unique"]

            slot_strategies[slot_name] = ("mlm", mlm_slot_config)
        else:
            # For other strategies (exhaustive, random, etc.)
            slot_strategies[slot_name] = (strategy_name, {})

    # Create renderer for English-specific noun handling
    # Uses OtherNounRenderer for "another"/"the other" patterns with repeated nouns
    renderer = OtherNounRenderer()
    logger.info("Using OtherNounRenderer for English-specific noun handling")

    # Create filler with MixedFillingStrategy
    logger.info("Creating template filler with mixed strategy...")
    strategy = MixedFillingStrategy(
        slot_strategies=slot_strategies,
    )

    # Fill templates
    logger.info("Filling templates...")
    filled_templates = []
    for i, template in enumerate(templates, 1):
        logger.info(f"Filling template {i}/{len(templates)}: {template.name}")
        try:
            combos = list(
                strategy.generate_from_template(
                    template=template, lexicons=lexicons, language_code="en"
                )
            )

            # Convert combinations to FilledTemplate objects
            for combo in combos:
                # Render text with English-specific noun handling
                rendered = renderer.render(
                    template.template_string, combo, template.slots
                )

                filled = FilledTemplate(
                    template_id=str(template.id),
                    template_name=template.name,
                    slot_fillers=combo,
                    rendered_text=rendered,
                    strategy_name="mixed",
                    template_slots={
                        name: slot.required for name, slot in template.slots.items()
                    },
                )
                filled_templates.append(filled)

            logger.info(f"  Generated {len(combos)} filled templates")
        except Exception as e:
            logger.error(f"  Failed to fill template {template.name}: {e}")
            continue

    logger.info(f"Total filled templates: {len(filled_templates)}")

    # Save filled templates
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonlines(filled_templates, output_path)
    logger.info(f"Saved filled templates to {output_path}")


if __name__ == "__main__":
    main()
