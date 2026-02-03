"""jsPsych batch experiment generator.

Generates complete jsPsych 8.x experiments using JATOS batch sessions for
server-side list distribution.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

from jinja2 import Environment, FileSystemLoader

from bead.data.base import JsonValue
from bead.data.serialization import SerializationError, write_jsonlines
from bead.deployment.jspsych.config import (
    ChoiceConfig,
    ExperimentConfig,
    InstructionsConfig,
    RatingScaleConfig,
)
from bead.deployment.jspsych.trials import create_trial
from bead.items.item import Item
from bead.items.item_template import ItemTemplate
from bead.lists import ExperimentList


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
            - data/config.json, data/lists.jsonl, data/items.jsonl,
              data/distribution.json

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
        >>> from bead.deployment.distribution import (
        ...     ListDistributionStrategy, DistributionStrategyType
        ... )
        >>> strategy = ListDistributionStrategy(
        ...     strategy_type=DistributionStrategyType.BALANCED
        ... )
        >>> config = ExperimentConfig(
        ...     experiment_type="forced_choice",
        ...     title="Test",
        ...     description="Test",
        ...     instructions="Test",
        ...     distribution_strategy=strategy
        ... )
        >>> generator = JsPsychExperimentGenerator(
        ...     config=config, output_dir=Path("/tmp/exp")
        ... )
        >>> # output_dir = generator.generate(lists, items, templates)
        """
        # Validate inputs (no fallbacks)
        if not lists:
            raise ValueError(
                "generate() requires at least one ExperimentList. Got empty list."
                " Create lists using ListPartitioner before calling generate()."
                " Example: partitioner.partition_with_batch_constraints(...)"
            )

        if not items:
            raise ValueError(
                "generate() requires items dictionary. Got empty dict."
                " Ensure items are constructed before calling generate()."
                " Items must be created using bead.items utilities."
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

        # Write batch data files (lists, items, distribution config, trials)
        self._write_lists_jsonl(lists)
        self._write_items_jsonl(items)
        self._write_distribution_config()
        self._write_trials_json(lists, items, templates)

        # Generate HTML/CSS/JS files
        self._generate_html()
        self._generate_css()
        self._generate_experiment_script()
        self._generate_config_file()
        self._copy_list_distributor_script()

        # Copy slopit bundle if enabled
        if self.config.slopit.enabled:
            self._copy_slopit_bundle()

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
                    ellipsis = "..." if len(items) > 5 else ""
                    raise ValueError(
                        f"Item {item_id} referenced in list '{exp_list.name}' "
                        f"(list_number={exp_list.list_number}) not found in items. "
                        f"Available UUIDs (first 5): {available_sample}{ellipsis}. "
                        f"Include all referenced items in items dict."
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
                ellipsis = "..." if len(templates) > 5 else ""
                raise ValueError(
                    f"Template {item.item_template_id} for item {item_id} "
                    f"not found in templates. "
                    f"Available UUIDs (first 5): {available_sample}{ellipsis}. "
                    f"Include all referenced templates in templates dict."
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

    def _write_trials_json(
        self,
        lists: list[ExperimentList],
        items: dict[UUID, Item],
        templates: dict[UUID, ItemTemplate],
    ) -> None:
        """Write pre-generated trials to data/trials.json.

        Creates trials for each list and stores them in a JSON file
        keyed by list ID for efficient loading in the experiment.

        Parameters
        ----------
        lists : list[ExperimentList]
            Experiment lists.
        items : dict[UUID, Item]
            Items dictionary.
        templates : dict[UUID, ItemTemplate]
            Templates dictionary.

        Raises
        ------
        SerializationError
            If writing JSON fails.
        """
        output_path = self.output_dir / "data" / "trials.json"
        trials_by_list: dict[str, list[dict[str, JsonValue]]] = {}

        for exp_list in lists:
            list_trials: list[dict[str, JsonValue]] = []
            for trial_num, item_id in enumerate(exp_list.item_refs):
                item = items[item_id]
                template = templates[item.item_template_id]
                trial = create_trial(
                    item=item,
                    template=template,
                    experiment_config=self.config,
                    trial_number=trial_num,
                    rating_config=self.rating_config,
                    choice_config=self.choice_config,
                )
                list_trials.append(trial)
            trials_by_list[str(exp_list.id)] = list_trials

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(trials_by_list, f, indent=2)
        except Exception as e:
            raise SerializationError(
                f"Failed to write trials.json to {output_path}: {e}"
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
        """Copy list_distributor.js from compiled dist/ to js/ directory.

        Raises
        ------
        FileNotFoundError
            If list_distributor.js is not found in dist/.
        OSError
            If copying fails.
        """
        dist_path = Path(__file__).parent / "dist" / "lib" / "list-distributor.js"
        output_path = self.output_dir / "js" / "list_distributor.js"

        if not dist_path.exists():
            raise FileNotFoundError(
                f"list-distributor.js not found at {dist_path}. "
                f"Ensure TypeScript is compiled. "
                f"Run 'npm run build' in the jspsych directory."
            )

        try:
            output_path.write_text(dist_path.read_text())
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
            slopit_enabled=self.config.slopit.enabled,
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

        # Prepare slopit config for template
        slopit_config = None
        if self.config.slopit.enabled:
            slopit_config = {
                "keystroke": self.config.slopit.keystroke.model_dump(),
                "focus": self.config.slopit.focus.model_dump(),
                "paste": self.config.slopit.paste.model_dump(),
                "target_selectors": self.config.slopit.target_selectors,
            }

        # Prepare demographics config for template
        demographics_enabled = False
        demographics_title = "Participant Information"
        demographics_fields: list[dict[str, JsonValue]] = []
        demographics_submit_text = "Continue"

        if self.config.demographics is not None and self.config.demographics.enabled:
            demographics_enabled = True
            demographics_title = self.config.demographics.title
            demographics_submit_text = self.config.demographics.submit_button_text
            for field in self.config.demographics.fields:
                field_data: dict[str, JsonValue] = {
                    "name": field.name,
                    "label": field.label,
                    "field_type": field.field_type,
                    "required": field.required,
                }
                if field.placeholder:
                    field_data["placeholder"] = field.placeholder
                if field.options:
                    field_data["options"] = field.options
                if field.range is not None:
                    field_data["range_min"] = field.range.min
                    field_data["range_max"] = field.range.max
                demographics_fields.append(field_data)

        # Prepare instructions config for template
        instructions_is_multi_page = isinstance(
            self.config.instructions, InstructionsConfig
        )
        instructions_pages: list[dict[str, str | None]] = []
        instructions_show_page_numbers = True
        instructions_allow_backwards = True
        instructions_button_next = "Next"
        instructions_button_finish = "Begin Experiment"
        simple_instructions: str | None = None

        if instructions_is_multi_page:
            assert isinstance(self.config.instructions, InstructionsConfig)
            instructions_show_page_numbers = self.config.instructions.show_page_numbers
            instructions_allow_backwards = self.config.instructions.allow_backwards
            instructions_button_next = self.config.instructions.button_label_next
            instructions_button_finish = self.config.instructions.button_label_finish
            for page in self.config.instructions.pages:
                instructions_pages.append(
                    {
                        "title": page.title,
                        "content": page.content,
                    }
                )
        else:
            # Simple string instructions
            simple_instructions = (
                self.config.instructions
                if isinstance(self.config.instructions, str)
                else None
            )

        js_content = template.render(
            title=self.config.title,
            description=self.config.description,
            instructions=simple_instructions,
            show_progress_bar=self.config.show_progress_bar,
            use_jatos=self.config.use_jatos,
            on_finish_url=on_finish_url,
            slopit_enabled=self.config.slopit.enabled,
            slopit_config=slopit_config,
            # Demographics variables
            demographics_enabled=demographics_enabled,
            demographics_title=demographics_title,
            demographics_fields=demographics_fields,
            demographics_submit_text=demographics_submit_text,
            # Instructions variables
            instructions_is_multi_page=instructions_is_multi_page,
            instructions_pages=instructions_pages,
            instructions_show_page_numbers=instructions_show_page_numbers,
            instructions_allow_backwards=instructions_allow_backwards,
            instructions_button_next=instructions_button_next,
            instructions_button_finish=instructions_button_finish,
        )

        output_file = self.output_dir / "js" / "experiment.js"
        output_file.write_text(js_content)

    def _generate_config_file(self) -> None:
        """Generate config.json file with experiment configuration."""
        output_file = self.output_dir / "data" / "config.json"
        json_str = self.config.model_dump_json(indent=2)
        output_file.write_text(json_str)

    def _copy_slopit_bundle(self) -> None:
        """Copy slopit bundle to js/ directory.

        Copies the pre-built slopit bundle from the bead deployment dist
        directory to the experiment output directory.

        Raises
        ------
        FileNotFoundError
            If slopit bundle is not found.
        OSError
            If copying fails.
        """
        # Look for slopit bundle in dist directory
        dist_dir = Path(__file__).parent / "dist"
        bundle_path = dist_dir / "slopit-bundle.js"

        if not bundle_path.exists():
            raise FileNotFoundError(
                f"Slopit bundle not found at {bundle_path}. "
                f"Ensure the slopit packages are built. "
                f"Run 'npm run build' in the jspsych directory, or install "
                f"bead with: pip install bead[behavioral-analysis]"
            )

        output_path = self.output_dir / "js" / "slopit-bundle.js"
        try:
            output_path.write_text(bundle_path.read_text())
        except OSError as e:
            raise OSError(
                f"Failed to copy slopit bundle to {output_path}: {e}. "
                f"Check write permissions."
            ) from e
