"""jsPsych batch experiment generator.

Generates complete jsPsych 8.x experiments using JATOS batch sessions for
server-side list distribution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import UUID

from jinja2 import Environment, FileSystemLoader

from bead.data.serialization import SerializationError, write_jsonlines
from bead.deployment.jspsych.config import (
    ChoiceConfig,
    ExperimentConfig,
    RatingScaleConfig,
)
from bead.deployment.jspsych.randomizer import generate_randomizer_function
from bead.deployment.jspsych.trials import create_trial
from bead.items.item import Item
from bead.items.item_template import ItemTemplate
from bead.lists import ExperimentList
from bead.lists.constraints import OrderingConstraint


class JsPsychExperimentGenerator:
    """Generator for jsPsych 8.x experiments.

    This class orchestrates the generation of complete jsPsych experiments,
    including HTML, CSS, JavaScript, and data files. It converts bead's
    ExperimentList and Item models into a deployable jsPsych experiment.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration.
    output_dir : Path
        Output directory for generated files.
    rating_config : RatingScaleConfig | None
        Configuration for rating scale trials (required for rating experiments).
        Defaults to RatingScaleConfig() if not provided.
    choice_config : ChoiceConfig | None
        Configuration for choice trials (required for choice experiments).
        Defaults to ChoiceConfig() if not provided.

    Attributes
    ----------
    config : ExperimentConfig
        Experiment configuration.
    output_dir : Path
        Output directory for generated files.
    rating_config : RatingScaleConfig
        Configuration for rating scale trials.
    choice_config : ChoiceConfig
        Configuration for choice trials.
    jinja_env : Environment
        Jinja2 environment for template rendering.

    Examples
    --------
    >>> from pathlib import Path
    >>> config = ExperimentConfig(
    ...     experiment_type="likert_rating",
    ...     title="Acceptability Study",
    ...     description="Rate sentences",
    ...     instructions="Rate each sentence from 1 to 7"
    ... )
    >>> generator = JsPsychExperimentGenerator(
    ...     config=config,
    ...     output_dir=Path("/tmp/experiment")
    ... )
    >>> # generator.generate(lists, items)
    """

    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: Path,
        rating_config: RatingScaleConfig | None = None,
        choice_config: ChoiceConfig | None = None,
    ) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.rating_config = rating_config or RatingScaleConfig()
        self.choice_config = choice_config or ChoiceConfig()

        # Setup Jinja2 environment
        template_dir = Path(__file__).parent / "templates"
        self.jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))

    def generate(
        self,
        lists: list[ExperimentList],
        items: dict[UUID, Item],
        templates: dict[UUID, ItemTemplate],
    ) -> Path:
        """Generate complete jsPsych batch experiment.

        Creates a unified batch experiment that uses JATOS batch sessions for
        server-side list distribution. All participants are automatically assigned
        to lists according to the distribution strategy specified in the experiment
        configuration.

        Parameters
        ----------
        lists : list[ExperimentList]
            Experiment lists for batch distribution (required, must be non-empty).
            All lists will be serialized to lists.jsonl and made available for
            participant assignment.
        items : dict[UUID, Item]
            Dictionary of items keyed by UUID (required, must be non-empty).
            All items referenced by lists must be present in this dictionary.
        templates : dict[UUID, ItemTemplate]
            Dictionary of item templates keyed by UUID (required, must be non-empty).
            All templates referenced by items must be present in this dictionary.

        Returns
        -------
        Path
            Path to the generated experiment directory containing:
            - index.html
            - js/experiment.js, js/list_distributor.js
            - css/experiment.css
            - data/config.json, data/lists.jsonl, data/items.jsonl, data/distribution.json

        Raises
        ------
        ValueError
            If lists is empty, items is empty, templates is empty, or if any
            referenced UUIDs are not found in the provided dictionaries.
        SerializationError
            If writing JSONL files fails.

        Examples
        --------
        >>> from pathlib import Path
        >>> from bead.deployment.distribution import ListDistributionStrategy, DistributionStrategyType
        >>> strategy = ListDistributionStrategy(strategy_type=DistributionStrategyType.BALANCED)
        >>> config = ExperimentConfig(
        ...     experiment_type="forced_choice",
        ...     title="Test",
        ...     description="Test",
        ...     instructions="Test",
        ...     distribution_strategy=strategy
        ... )
        >>> generator = JsPsychExperimentGenerator(config=config, output_dir=Path("/tmp/exp"))
        >>> # output_dir = generator.generate(lists, items, templates)
        """
        # Validate inputs (no fallbacks)
        if not lists:
            raise ValueError(
                "generate() requires at least one ExperimentList. Got empty list. "
                "Create lists using ListPartitioner before calling generate(). "
                "Example: partitioner.partition_with_batch_constraints(items, metadata, ...)"
            )

        if not items:
            raise ValueError(
                "generate() requires items dictionary. Got empty dict. "
                "Ensure items are constructed before calling generate(). "
                "Items must be created using item construction utilities from bead.items."
            )

        if not templates:
            raise ValueError(
                "generate() requires templates dictionary. Got empty dict. "
                "Ensure item templates are included. If items don't use templates, "
                "provide an empty template: {item.item_template_id: ItemTemplate(...)}."
            )

        # Validate all item references can be resolved
        self._validate_item_references(lists, items)

        # Validate all template references can be resolved
        self._validate_template_references(items, templates)

        # Create directory structure
        self._create_directory_structure()

        # Write batch data files (lists, items, distribution config)
        self._write_lists_jsonl(lists)
        self._write_items_jsonl(items)
        self._write_distribution_config()

        # Generate HTML/CSS/JS files
        self._generate_html()
        self._generate_css()
        self._generate_experiment_script()
        self._generate_config_file()
        self._copy_list_distributor_script()

        return self.output_dir

    def _validate_item_references(
        self,
        lists: list[ExperimentList],
        items: dict[UUID, Item],
    ) -> None:
        """Validate all item UUIDs in lists can be resolved.

        Parameters
        ----------
        lists : list[ExperimentList]
            Lists to validate.
        items : dict[UUID, Item]
            Items dictionary.

        Raises
        ------
        ValueError
            If any item UUID in lists is not found in items dict.
        """
        for exp_list in lists:
            for item_id in exp_list.item_refs:
                if item_id not in items:
                    available_sample = list(items.keys())[:5]
                    raise ValueError(
                        f"Item {item_id} referenced in list '{exp_list.name}' "
                        f"(list_number={exp_list.list_number}) not found in items dictionary. "
                        f"Available item UUIDs (first 5): {available_sample}{'...' if len(items) > 5 else ''}. "
                        f"Ensure all items referenced in lists are included in the items dict passed to generate()."
                    )

    def _validate_template_references(
        self,
        items: dict[UUID, Item],
        templates: dict[UUID, ItemTemplate],
    ) -> None:
        """Validate all template UUIDs in items can be resolved.

        Parameters
        ----------
        items : dict[UUID, Item]
            Items dictionary.
        templates : dict[UUID, ItemTemplate]
            Templates dictionary.

        Raises
        ------
        ValueError
            If any template UUID in items is not found in templates dict.
        """
        for item_id, item in items.items():
            if item.item_template_id not in templates:
                available_sample = list(templates.keys())[:5]
                raise ValueError(
                    f"Template {item.item_template_id} referenced by item {item_id} "
                    f"not found in templates dictionary. "
                    f"Available template UUIDs (first 5): {available_sample}{'...' if len(templates) > 5 else ''}. "
                    f"Ensure all templates referenced by items are included in the templates dict passed to generate()."
                )

    def _write_lists_jsonl(self, lists: list[ExperimentList]) -> None:
        """Write experiment lists to data/lists.jsonl.

        Parameters
        ----------
        lists : list[ExperimentList]
            Lists to serialize.

        Raises
        ------
        SerializationError
            If writing JSONL fails.
        """
        output_path = self.output_dir / "data" / "lists.jsonl"
        try:
            write_jsonlines(lists, output_path)
        except SerializationError as e:
            raise SerializationError(
                f"Failed to write lists.jsonl to {output_path}: {e}. "
                f"Check write permissions and disk space. "
                f"Attempted to serialize {len(lists)} lists."
            ) from e

    def _write_items_jsonl(self, items: dict[UUID, Item]) -> None:
        """Write items to data/items.jsonl.

        Parameters
        ----------
        items : dict[UUID, Item]
            Items dictionary to serialize.

        Raises
        ------
        SerializationError
            If writing JSONL fails.
        """
        output_path = self.output_dir / "data" / "items.jsonl"
        try:
            # Convert dict values to list for serialization
            items_list = list(items.values())
            write_jsonlines(items_list, output_path)
        except SerializationError as e:
            raise SerializationError(
                f"Failed to write items.jsonl to {output_path}: {e}. "
                f"Check write permissions and disk space. "
                f"Attempted to serialize {len(items)} items."
            ) from e

    def _write_distribution_config(self) -> None:
        """Write distribution strategy config to data/distribution.json.

        Raises
        ------
        SerializationError
            If writing JSON fails.
        """
        output_path = self.output_dir / "data" / "distribution.json"
        try:
            # Use model_dump_json() to handle UUID serialization
            json_str = self.config.distribution_strategy.model_dump_json(indent=2)
            output_path.write_text(json_str)
        except (OSError, TypeError) as e:
            raise SerializationError(
                f"Failed to write distribution.json to {output_path}: {e}. "
                f"Check write permissions and disk space. "
                f"Strategy type: {self.config.distribution_strategy.strategy_type}"
            ) from e

    def _copy_list_distributor_script(self) -> None:
        """Copy list_distributor.js template to js/ directory.

        Raises
        ------
        FileNotFoundError
            If list_distributor.js template is not found.
        OSError
            If copying fails.
        """
        template_path = Path(__file__).parent / "templates" / "list_distributor.js"
        output_path = self.output_dir / "js" / "list_distributor.js"

        if not template_path.exists():
            raise FileNotFoundError(
                f"list_distributor.js template not found at {template_path}. "
                f"This file is required for batch mode. "
                f"Ensure bead is installed correctly."
            )

        try:
            output_path.write_text(template_path.read_text())
        except OSError as e:
            raise OSError(
                f"Failed to copy list_distributor.js to {output_path}: {e}. "
                f"Check write permissions."
            ) from e

    def _create_directory_structure(self) -> None:
        """Create output directory structure.

        Creates:
        - output_dir/
        - output_dir/css/
        - output_dir/js/
        - output_dir/data/
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "css").mkdir(exist_ok=True)
        (self.output_dir / "js").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)

    def _generate_html(self) -> None:
        """Generate index.html file."""
        template = self.jinja_env.get_template("index.html")

        html_content = template.render(
            title=self.config.title,
            ui_theme=self.config.ui_theme,
            use_jatos=self.config.use_jatos,
        )

        output_file = self.output_dir / "index.html"
        output_file.write_text(html_content)

    def _generate_css(self) -> None:
        """Generate experiment.css file by copying template."""
        template_file = Path(__file__).parent / "templates" / "experiment.css"
        output_file = self.output_dir / "css" / "experiment.css"

        # Copy CSS template directly (no rendering needed)
        output_file.write_text(template_file.read_text())

    def _generate_experiment_script(self) -> None:
        """Generate experiment.js file."""
        template = self.jinja_env.get_template("experiment.js.template")

        # Auto-generate Prolific redirect URL if completion code is provided
        on_finish_url = self.config.on_finish_url
        if self.config.prolific_completion_code:
            on_finish_url = (
                f"https://app.prolific.co/submissions/complete?"
                f"cc={self.config.prolific_completion_code}"
            )

        js_content = template.render(
            title=self.config.title,
            description=self.config.description,
            instructions=self.config.instructions,
            show_progress_bar=self.config.show_progress_bar,
            use_jatos=self.config.use_jatos,
            on_finish_url=on_finish_url,
        )

        output_file = self.output_dir / "js" / "experiment.js"
        output_file.write_text(js_content)

    def _generate_config_file(self) -> None:
        """Generate config.json file with experiment configuration."""
        output_file = self.output_dir / "data" / "config.json"
        json_str = self.config.model_dump_json(indent=2)
        output_file.write_text(json_str)
