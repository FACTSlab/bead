"""Main jsPsych experiment generator.

This module provides the JsPsychExperimentGenerator class, which orchestrates
the generation of complete jsPsych 8.x experiments from ExperimentLists and Items.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import UUID

from jinja2 import Environment, FileSystemLoader

from bead.deployment.jspsych.config import (
    ChoiceConfig,
    ExperimentConfig,
    RatingScaleConfig,
)
from bead.deployment.jspsych.randomizer import generate_randomizer_function
from bead.deployment.jspsych.trials import create_trial
from bead.items.models import Item, ItemTemplate
from bead.lists.constraints import OrderingConstraint
from bead.lists.models import ExperimentList


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
        """Generate complete jsPsych experiment.

        Parameters
        ----------
        lists : list[ExperimentList]
            Experiment lists to generate trials from. For Phase 19, we typically
            use a single list, but the API supports multiple lists for future phases.
        items : dict[UUID, Item]
            Dictionary of items keyed by UUID.
        templates : dict[UUID, ItemTemplate]
            Dictionary of item templates keyed by UUID.

        Returns
        -------
        Path
            Path to the generated experiment directory.

        Raises
        ------
        ValueError
            If no lists provided or if items/templates are missing.
        """
        if not lists:
            raise ValueError("At least one ExperimentList is required")

        # Create directory structure
        self._create_directory_structure()

        # For Phase 19, we use the first list
        # (multi-list support will be added in future phases)
        experiment_list = lists[0]

        # Generate timeline data (trials)
        timeline_data = self._generate_timeline_data(experiment_list, items, templates)

        # Extract ordering constraints and item metadata
        ordering_constraints = self._extract_ordering_constraints(experiment_list)
        item_metadata = self._extract_item_metadata(experiment_list, items)

        # Generate randomizer code if randomization is enabled
        randomizer_code = ""
        if self.config.randomize_trial_order and ordering_constraints:
            randomizer_code = generate_randomizer_function(
                item_ids=experiment_list.item_refs,
                constraints=ordering_constraints,
                metadata=item_metadata,
            )

        # Generate all files
        self._generate_html(timeline_data, randomizer_code)
        self._generate_css()
        self._generate_experiment_script(timeline_data, randomizer_code)
        self._generate_config_file()

        return self.output_dir

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

    def _generate_timeline_data(
        self,
        experiment_list: ExperimentList,
        items: dict[UUID, Item],
        templates: dict[UUID, ItemTemplate],
    ) -> dict[str, Any]:
        """Generate timeline data from experiment list.

        Parameters
        ----------
        experiment_list : ExperimentList
            Experiment list with item IDs.
        items : dict[UUID, Item]
            Dictionary of items keyed by UUID.
        templates : dict[UUID, ItemTemplate]
            Dictionary of item templates keyed by UUID.

        Returns
        -------
        dict[str, Any]
            Timeline data structure with trials.

        Raises
        ------
        ValueError
            If an item ID in the list is not found in items dict,
            or if a template ID is not found in templates dict.
        """
        trials: list[dict[str, Any]] = []

        for trial_number, item_id in enumerate(experiment_list.item_refs):
            if item_id not in items:
                raise ValueError(f"Item {item_id} not found in items dictionary")

            item = items[item_id]

            # Look up the template for this item
            if item.item_template_id not in templates:
                raise ValueError(
                    f"Template {item.item_template_id} not found in "
                    f"templates dictionary"
                )

            template = templates[item.item_template_id]

            trial = create_trial(
                item=item,
                template=template,
                experiment_config=self.config,
                trial_number=trial_number,
                rating_config=self.rating_config,
                choice_config=self.choice_config,
            )
            trials.append(trial)

        return {"trials": trials}

    def _extract_ordering_constraints(
        self,
        experiment_list: ExperimentList,
    ) -> list[OrderingConstraint]:
        """Extract ordering constraints from experiment list.

        Parameters
        ----------
        experiment_list : ExperimentList
            Experiment list with constraints.

        Returns
        -------
        list[OrderingConstraint]
            List of ordering constraints.
        """
        ordering_constraints: list[OrderingConstraint] = []

        for constraint in experiment_list.list_constraints:
            if isinstance(constraint, OrderingConstraint):
                ordering_constraints.append(constraint)

        return ordering_constraints

    def _extract_item_metadata(
        self,
        experiment_list: ExperimentList,
        items: dict[UUID, Item],
    ) -> dict[UUID, dict[str, Any]]:
        """Extract item metadata needed for constraint checking.

        Parameters
        ----------
        experiment_list : ExperimentList
            Experiment list with item IDs.
        items : dict[UUID, Item]
            Dictionary of items.

        Returns
        -------
        dict[UUID, dict[str, Any]]
            Item metadata keyed by item UUID.
        """
        metadata: dict[UUID, dict[str, Any]] = {}

        for item_id in experiment_list.item_refs:
            if item_id in items:
                item = items[item_id]
                # Extract relevant metadata for constraint checking
                metadata[item_id] = dict(item.item_metadata)

        return metadata

    def _generate_html(
        self,
        timeline_data: dict[str, Any],
        randomizer_code: str,
    ) -> None:
        """Generate index.html file.

        Parameters
        ----------
        timeline_data : dict[str, Any]
            Timeline data structure.
        randomizer_code : str
            JavaScript randomizer code.
        """
        template = self.jinja_env.get_template("index.html")

        html_content = template.render(
            title=self.config.title,
            ui_theme=self.config.ui_theme,
            config_json=json.dumps(self.config.model_dump()),
            timeline_json=json.dumps(timeline_data),
        )

        output_file = self.output_dir / "index.html"
        output_file.write_text(html_content)

    def _generate_css(self) -> None:
        """Generate experiment.css file by copying template."""
        template_file = Path(__file__).parent / "templates" / "experiment.css"
        output_file = self.output_dir / "css" / "experiment.css"

        # Copy CSS template directly (no rendering needed)
        output_file.write_text(template_file.read_text())

    def _generate_experiment_script(
        self,
        timeline_data: dict[str, Any],
        randomizer_code: str,
    ) -> None:
        """Generate experiment.js file.

        Parameters
        ----------
        timeline_data : dict[str, Any]
            Timeline data structure.
        randomizer_code : str
            JavaScript randomizer code.
        """
        template = self.jinja_env.get_template("experiment.js.template")

        js_content = template.render(
            title=self.config.title,
            description=self.config.description,
            instructions=self.config.instructions,
            show_progress_bar=self.config.show_progress_bar,
            randomize_trial_order=self.config.randomize_trial_order,
            on_finish_url=self.config.on_finish_url,
            randomizer_code=randomizer_code,
        )

        output_file = self.output_dir / "js" / "experiment.js"
        output_file.write_text(js_content)

    def _generate_config_file(self) -> None:
        """Generate config.json file with experiment configuration."""
        config_data = self.config.model_dump()

        output_file = self.output_dir / "data" / "config.json"
        output_file.write_text(json.dumps(config_data, indent=2))
