# Interactive Task Gallery

Try each bead task interface below. Every demo is a live jsPsych experiment running in your browser. Examples use stimuli from psycholinguistics research on acceptability, veridicality, semantic proto-roles, event typicality, and telicity.

<script>
window.addEventListener('message', e => {
  if (e.data?.type === 'bead-resize') {
    for (const f of document.querySelectorAll('iframe')) {
      if (f.contentWindow === e.source) { f.style.height = e.data.height + 'px'; break; }
    }
  }
});
</script>

---

## Judgment Tasks

### Likert Rating Scale

Rate a sentence on a discrete scale with labeled endpoints. This example asks about the naturalness of the verb *hope* in an NP-to-VP raising frame.

=== "Demo"

    <iframe src="../../gallery/demos/rating-likert.html" width="100%" height="150" style="border:1px solid #e0e0e0;border-radius:6px;transition:height 0.15s ease"></iframe>

=== "Python"

    ```python
    from bead.items.ordinal_scale import create_ordinal_scale_item

    item = create_ordinal_scale_item(
        text="Someone hoped someone to leave.",
        prompt="How natural is this sentence?",
        scale_bounds=(1, 7),
        scale_labels={
            1: "Extremely unnatural",
            7: "Totally natural",
        },
    )
    ```

=== "Trial JSON"

    ```json
    {
      "type": "bead-rating",
      "prompt": "How natural is this sentence?",
      "stimulus": "Someone hoped someone to leave.",
      "scale_min": 1,
      "scale_max": 7,
      "scale_labels": {
        "1": "Extremely unnatural",
        "7": "Totally natural"
      },
      "metadata": {"verb": "hope", "frame": "NP_to_VP"}
    }
    ```

### Slider Rating

Continuous rating on a slider scale. This example asks how prototypical an event is.

=== "Demo"

    <iframe src="../../gallery/demos/rating-slider.html" width="100%" height="150" style="border:1px solid #e0e0e0;border-radius:6px;transition:height 0.15s ease"></iframe>

=== "Python"

    ```python
    from bead.items.ordinal_scale import create_ordinal_scale_item

    item = create_ordinal_scale_item(
        text="The chef cooked the meal.",
        prompt="How prototypical is this event?",
        scale_bounds=(0, 100),
        scale_labels={
            0: "Very atypical",
            100: "Very prototypical",
        },
    )
    ```

=== "Trial JSON"

    ```json
    {
      "type": "bead-slider-rating",
      "prompt": "How prototypical is this event?",
      "stimulus": "The chef cooked the meal.",
      "slider_min": 0,
      "slider_max": 100,
      "slider_start": 50,
      "labels": ["Very atypical", "Very prototypical"],
      "metadata": {"verb": "cook"}
    }
    ```

### Forced Choice

Choose between two alternatives. This example uses a classic syntactic ambiguity to demonstrate comparative judgment.

=== "Demo"

    <iframe src="../../gallery/demos/forced-choice.html" width="100%" height="150" style="border:1px solid #e0e0e0;border-radius:6px;transition:height 0.15s ease"></iframe>

=== "Python"

    ```python
    from bead.items.forced_choice import create_forced_choice_item

    item = create_forced_choice_item(
        alternatives=[
            "The turkey",
            "Something else",
        ],
        prompt="The turkey was ready to eat. What planned to eat?",
    )
    ```

=== "Trial JSON"

    ```json
    {
      "type": "bead-forced-choice",
      "prompt": "The turkey was ready to eat. What planned to eat?",
      "alternatives": [
        "The turkey",
        "Something else"
      ],
      "layout": "vertical",
      "metadata": {"sentence": "The turkey was ready to eat."}
    }
    ```

### Binary Judgment

Yes/No acceptability judgment. This example tests the verb *persuade* in an NP-to-VP object-control frame.

=== "Demo"

    <iframe src="../../gallery/demos/binary-choice.html" width="100%" height="150" style="border:1px solid #e0e0e0;border-radius:6px;transition:height 0.15s ease"></iframe>

=== "Python"

    ```python
    from bead.items.binary import create_binary_item

    item = create_binary_item(
        text="Someone persuaded someone to leave.",
        prompt="Is this sentence acceptable?",
        options=["Acceptable", "Unacceptable"],
    )
    ```

=== "Trial JSON"

    ```json
    {
      "type": "bead-binary-choice",
      "prompt": "Is this sentence acceptable?",
      "stimulus": "Someone persuaded someone to leave.",
      "choices": ["Acceptable", "Unacceptable"],
      "metadata": {"verb": "persuade", "frame": "NP_to_VP"}
    }
    ```

### Categorical Classification

Select one category from an unordered set. This example tests factivity using a natural inference task.

=== "Demo"

    <iframe src="../../gallery/demos/categorical.html" width="100%" height="150" style="border:1px solid #e0e0e0;border-radius:6px;transition:height 0.15s ease"></iframe>

=== "Python"

    ```python
    from bead.items.categorical import create_categorical_item

    item = create_categorical_item(
        text=(
            "Sentence 1: The doctor managed to treat the patient.\n"
            "Sentence 2: The patient was treated."
        ),
        prompt="If the first sentence is true, is the second sentence true?",
        categories=["Definitely not", "Maybe", "Definitely"],
    )
    ```

=== "Trial JSON"

    ```json
    {
      "type": "bead-categorical",
      "prompt": "If the first sentence is true, is the second sentence true?",
      "categories": ["Definitely not", "Maybe", "Definitely"],
      "metadata": {"recast_type": "factivity"}
    }
    ```

### Magnitude Estimation

Rate a target stimulus relative to a reference using an exponential slider. The slider maps linear position to exponential values via `exp(x/100) - 1`, placing the reference value at 1/3 from the left. Arrow keys give proportional ~3% changes at any scale; the right end approaches infinity.

=== "Demo"

    <iframe src="../../gallery/demos/magnitude.html" width="100%" height="250" style="border:1px solid #e0e0e0;border-radius:6px;transition:height 0.15s ease"></iframe>

=== "Python"

    ```python
    from bead.items.magnitude import create_magnitude_item

    item = create_magnitude_item(
        text="The alien cooked the pencil.",
        prompt="How typical is the target relative to the reference?",
        reference_text="The chef cooked the meal.",
        reference_value=100,
    )
    ```

=== "Trial JSON"

    ```json
    {
      "type": "bead-magnitude",
      "prompt": "How typical is the target relative to the reference?",
      "prompt_position": "below",
      "reference_stimulus": "The chef cooked the meal.",
      "reference_value": 100,
      "stimulus": "The alien cooked the pencil.",
      "input_mode": "exp-slider",
      "metadata": {"verb": "cook"}
    }
    ```

### Free Text Response

Open-ended text response, single-line or multiline. This example asks for an event summarization of a historical passage.

=== "Demo"

    <iframe src="../../gallery/demos/free-text.html" width="100%" height="150" style="border:1px solid #e0e0e0;border-radius:6px;transition:height 0.15s ease"></iframe>

=== "Python"

    ```python
    from bead.items.free_text import create_free_text_item

    item = create_free_text_item(
        text="The 1846 US occupation of Monterey put an end to any Mexican "
             "military presence at the Presidio. The fort was abandoned in 1866.",
        prompt="Summarize the key event described in this passage.",
        multiline=True,
        min_length=5,
        max_length=200,
    )
    ```

=== "Trial JSON"

    ```json
    {
      "type": "bead-free-text",
      "prompt": "Summarize the key event described in this passage.",
      "multiline": true,
      "rows": 3,
      "min_length": 5,
      "max_length": 200,
      "metadata": {"event_type": "Abandoning"}
    }
    ```

---

## Selection Tasks

### Cloze (Fill-in-the-Blank)

Dropdown selection for fill-in-the-blank gaps. This example tests temporal interpretation using a telicity cloze task with preposition, numeral, and duration unit fields.

=== "Demo"

    <iframe src="../../gallery/demos/cloze-dropdown.html" width="100%" height="150" style="border:1px solid #e0e0e0;border-radius:6px;transition:height 0.15s ease"></iframe>

=== "Python"

    ```python
    from bead.items.cloze import create_cloze_item

    item = create_cloze_item(
        text="The hurricane hit the coastline {{prep}} {{numeral}} {{unit}}.",
        constraints={
            "prep": ["in", "for"],
            "numeral": None,  # free text
            "unit": ["seconds", "minutes", "hours", "days", "weeks"],
        },
    )
    ```

=== "Trial JSON"

    ```json
    {
      "type": "bead-cloze-multi",
      "text": "The hurricane hit the coastline %% %% %%.",
      "fields": [
        {"type": "dropdown", "options": ["in", "for"]},
        {"type": "text", "placeholder": "#"},
        {"type": "dropdown", "options": ["seconds", "minutes", "hours", "days", "weeks"]}
      ],
      "require_all": true
    }
    ```

### Multi-Select

Select one or more options from a set using checkboxes. This example tests pronoun resolution in a discourse with multiple potential referents.

=== "Demo"

    <iframe src="../../gallery/demos/multi-select.html" width="100%" height="200" style="border:1px solid #e0e0e0;border-radius:6px;transition:height 0.15s ease"></iframe>

=== "Python"

    ```python
    from bead.items.multi_select import create_multi_select_item

    item = create_multi_select_item(
        text="Whenever anyone laughed, the magician scowled and their "
             "assistant smirked. They were secretly pleased.",
        prompt="Who was secretly pleased?",
        options=[
            "The magician",
            "The assistant",
            "Neither",
        ],
        min_selections=1,
    )
    ```

=== "Trial JSON"

    ```json
    {
      "type": "bead-multi-select",
      "prompt": "Who was secretly pleased?",
      "stimulus": "Whenever anyone laughed, the magician scowled and their assistant smirked. They were secretly pleased.",
      "options": [
        "The magician",
        "The assistant",
        "Neither"
      ],
      "metadata": {"phenomenon": "pronoun_resolution"}
    }
    ```

---

## Span Annotation

### Interactive Span Labeling (Fixed Labels)

Select token ranges and assign labels from a searchable fixed set. Type to filter labels or use keyboard shortcuts 1-9.

=== "Demo"

    <iframe src="../../gallery/demos/span-interactive.html" width="100%" height="200" style="border:1px solid #e0e0e0;border-radius:6px;transition:height 0.15s ease"></iframe>

=== "Python"

    ```python
    from bead.items.span_labeling import create_interactive_span_item

    item = create_interactive_span_item(
        text="The committee unanimously approved the new budget proposal "
             "after reviewing the evidence.",
        prompt="Select and label semantic roles.",
        label_set=[
            "Agent", "Patient", "Theme", "Experiencer",
            "Instrument", "Beneficiary", "Location", "Time",
            "Manner", "Cause", "Purpose", "Source",
            "Goal", "Stimulus", "Result", "Predicate",
        ],
    )
    ```

=== "Trial JSON"

    ```json
    {
      "type": "bead-span-label",
      "tokens": {
        "text": ["The", "committee", "unanimously", "approved", "the",
                 "new", "budget", "proposal", "after", "reviewing",
                 "the", "evidence", "."]
      },
      "span_spec": {
        "interaction_mode": "interactive",
        "label_source": "fixed",
        "labels": ["Agent", "Patient", "Theme", "Experiencer",
                   "Instrument", "Beneficiary", "Location", "Time",
                   "Manner", "Cause", "Purpose", "Source",
                   "Goal", "Stimulus", "Result", "Predicate"]
      }
    }
    ```

### Wikidata Entity Labeling

Interactive span labeling with Wikidata autocomplete search for labels. Select entities and search Wikidata to link them.

=== "Demo"

    <iframe src="../../gallery/demos/span-wikidata.html" width="100%" height="200" style="border:1px solid #e0e0e0;border-radius:6px;transition:height 0.15s ease"></iframe>

=== "Python"

    ```python
    from bead.items.span_labeling import create_interactive_span_item

    item = create_interactive_span_item(
        text="Albert Einstein developed the theory of relativity "
             "at the Institute for Advanced Study in Princeton.",
        prompt="Select entities and search Wikidata to assign labels.",
        label_source="wikidata",
    )
    ```

=== "Trial JSON"

    ```json
    {
      "type": "bead-span-label",
      "tokens": {
        "text": ["Albert", "Einstein", "developed", "the", "theory",
                 "of", "relativity", "at", "the", "Institute",
                 "for", "Advanced", "Study", "in", "Princeton", "."]
      },
      "span_spec": {
        "interaction_mode": "interactive",
        "label_source": "wikidata",
        "wikidata_language": "en"
      }
    }
    ```

---

## Composite Tasks

Span highlights work as an orthogonal overlay on any existing task type. The same item can have both span annotations and a rating scale, forced choice, or binary judgment.

**Prompt span references**: prompts use `[[label]]` syntax to reference span labels. `[[label]]` auto-fills with the span's token text; `[[label:custom text]]` uses a custom surface form. At deployment, references are replaced with color-highlighted HTML matching the span colors in the stimulus. See the [Items API guide](../user-guide/api/items.md#prompt-span-references) for details.

### Span + Likert Rating

Proto-role property rating with highlighted arguments using thematic role labels. Question text uses colored highlighting that matches the span colors.

=== "Demo"

    <iframe src="../../gallery/demos/span-with-rating.html" width="100%" height="200" style="border:1px solid #e0e0e0;border-radius:6px;transition:height 0.15s ease"></iframe>

=== "Python"

    ```python
    from bead.items.ordinal_scale import create_ordinal_scale_item
    from bead.items.span_labeling import add_spans_to_item
    from bead.items.spans import Span, SpanLabel, SpanSegment

    item = create_ordinal_scale_item(
        text="The boy broke the vase.",
        prompt="How likely is it that [[breaker]] existed after [[event:the breaking]]?",
        scale_bounds=(1, 5),
        scale_labels={1: "Very unlikely", 5: "Very likely"},
    )

    item = add_spans_to_item(item, spans=[
        Span(span_id="span_0",
             segments=[SpanSegment(element_name="text", indices=[0, 1])],
             label=SpanLabel(label="breaker")),
        Span(span_id="span_1",
             segments=[SpanSegment(element_name="text", indices=[2])],
             label=SpanLabel(label="event")),
        Span(span_id="span_2",
             segments=[SpanSegment(element_name="text", indices=[3, 4])],
             label=SpanLabel(label="breakee")),
    ])
    ```

=== "Trial JSON"

    ```json
    {
      "type": "bead-rating",
      "prompt": "<div class=\"stimulus-container\">...(span-highlighted tokens)...</div><p class=\"bead-task-prompt\">How likely is it that <span class=\"bead-q-highlight\" style=\"background:#BBDEFB\">The boy<span class=\"bead-q-chip\" style=\"background:#1565C0\">breaker</span></span> existed after <span class=\"bead-q-highlight\" style=\"background:#C8E6C9\">the breaking<span class=\"bead-q-chip\" style=\"background:#2E7D32\">event</span></span>?</p>",
      "scale_min": 1,
      "scale_max": 5,
      "scale_labels": {"1": "Very unlikely", "5": "Very likely"},
      "metadata": {"trial_type": "likert_rating"}
    }
    ```

### Span + Slider Rating

Veridicality inference with highlighted spans but no labels (null labels). The highlighted regions draw attention to the predicate and embedded clause without adding subscript badges.

=== "Demo"

    <iframe src="../../gallery/demos/span-with-slider.html" width="100%" height="200" style="border:1px solid #e0e0e0;border-radius:6px;transition:height 0.15s ease"></iframe>

=== "Python"

    ```python
    from bead.items.ordinal_scale import create_ordinal_scale_item
    from bead.items.span_labeling import add_spans_to_item
    from bead.items.spans import Span, SpanLabel, SpanSegment

    item = create_ordinal_scale_item(
        text="Jo confirmed that Bo left.",
        prompt="How likely is it that someone left?",
        scale_bounds=(0, 100),
    )

    item = add_spans_to_item(item, spans=[
        Span(span_id="span_0",
             segments=[SpanSegment(element_name="text", indices=[1])],
             label=None),
        Span(span_id="span_1",
             segments=[SpanSegment(element_name="text", indices=[3, 4])],
             label=None),
    ])
    ```

=== "Trial JSON"

    ```json
    {
      "type": "bead-slider-rating",
      "prompt": "<div class=\"stimulus-container\">...(span-highlighted tokens, no subscript badges)...</div><p class=\"bead-task-prompt\">How likely is it that someone left?</p>",
      "labels": ["Not at all", "Very much"],
      "slider_min": 0,
      "slider_max": 100,
      "slider_start": 50,
      "metadata": {"trial_type": "slider_rating"}
    }
    ```

### Span + Forced Choice

Compare change-of-state across predicates with thematic role labels. Question text uses colored highlighting matching the span annotation colors.

=== "Demo"

    <iframe src="../../gallery/demos/span-with-choice.html" width="100%" height="200" style="border:1px solid #e0e0e0;border-radius:6px;transition:height 0.15s ease"></iframe>

=== "Python"

    ```python
    from bead.items.forced_choice import create_forced_choice_item

    item = create_forced_choice_item(
        alternatives=[
            "The boy tapped the vase.",
            "The boy hit the vase.",
        ],
        prompt="In which event is it more likely that the vase broke?",
    )
    ```

=== "Trial JSON"

    ```json
    {
      "type": "bead-forced-choice",
      "prompt": "In which event is it more likely that the vase broke?",
      "alternatives": [
        "The boy tapped the vase.",
        "The boy hit the vase."
      ],
      "layout": "horizontal",
      "metadata": {"trial_type": "forced_choice"}
    }
    ```

### Span + Binary Judgment

Change-of-location property with four thematic role arguments. Question text uses colored highlighting matching the span colors.

=== "Demo"

    <iframe src="../../gallery/demos/span-with-binary.html" width="100%" height="200" style="border:1px solid #e0e0e0;border-radius:6px;transition:height 0.15s ease"></iframe>

=== "Python"

    ```python
    from bead.items.binary import create_binary_item
    from bead.items.span_labeling import add_spans_to_item
    from bead.items.spans import Span, SpanLabel, SpanSegment

    item = create_binary_item(
        text="The merchant traded the silk for the spices.",
        prompt="Did [[traded-away:the silk]] change location as a result of [[event:the trading]]?",
        options=["Yes", "No"],
    )

    item = add_spans_to_item(item, spans=[
        Span(span_id="span_0",
             segments=[SpanSegment(element_name="text", indices=[0, 1])],
             label=SpanLabel(label="trader")),
        Span(span_id="span_1",
             segments=[SpanSegment(element_name="text", indices=[2])],
             label=SpanLabel(label="event")),
        Span(span_id="span_2",
             segments=[SpanSegment(element_name="text", indices=[3, 4])],
             label=SpanLabel(label="traded-away")),
        Span(span_id="span_3",
             segments=[SpanSegment(element_name="text", indices=[6, 7])],
             label=SpanLabel(label="traded-for")),
    ])
    ```

=== "Trial JSON"

    ```json
    {
      "type": "bead-binary-choice",
      "prompt": "Did <span class=\"bead-q-highlight\" style=\"background:#FFE0B2\">the silk<span class=\"bead-q-chip\" style=\"background:#E65100\">traded-away</span></span> change location as a result of <span class=\"bead-q-highlight\" style=\"background:#C8E6C9\">the trading<span class=\"bead-q-chip\" style=\"background:#2E7D32\">event</span></span>?",
      "stimulus": "<div class=\"stimulus-container\">...(span-highlighted tokens with subscript badges)...</div>",
      "choices": ["Yes", "No"],
      "metadata": {"trial_type": "binary_choice"}
    }
    ```

### Span + Free Text

Event summarization with a highlighted event span. The annotated span draws attention to the target event in a longer passage.

=== "Demo"

    <iframe src="../../gallery/demos/span-with-freetext.html" width="100%" height="250" style="border:1px solid #e0e0e0;border-radius:6px;transition:height 0.15s ease"></iframe>

=== "Python"

    ```python
    from bead.items.free_text import create_free_text_item
    from bead.items.span_labeling import add_spans_to_item
    from bead.items.spans import Span, SpanLabel, SpanSegment

    item = create_free_text_item(
        text="The 1846 US occupation of Monterey put an end to any Mexican "
             "military presence at the Presidio. The fort was abandoned in 1866.",
        prompt="Summarize [[event:the highlighted event]] in one sentence.",
        multiline=True,
        min_length=5,
        max_length=200,
    )

    item = add_spans_to_item(item, spans=[
        Span(span_id="span_0",
             segments=[SpanSegment(element_name="text", indices=[21])],
             label=SpanLabel(label="event")),
    ])
    ```

=== "Trial JSON"

    ```json
    {
      "type": "bead-free-text",
      "prompt": "Summarize <span class=\"bead-q-highlight\" style=\"background:#BBDEFB\">the highlighted event<span class=\"bead-q-chip\" style=\"background:#1565C0\">event</span></span> in one sentence.",
      "stimulus": "<div class=\"stimulus-container\">...(span-highlighted tokens)...</div>",
      "multiline": true,
      "rows": 3,
      "min_length": 5,
      "max_length": 200,
      "metadata": {"trial_type": "free_text"}
    }
    ```

---

## Relation Annotation

### Span Relations (Fixed Labels)

Interactive span and relation annotation with searchable fixed label sets. Create spans, then use "Add Relation" to draw directed relations between them using thematic role labels.

=== "Demo"

    <iframe src="../../gallery/demos/span-relations-fixed.html" width="100%" height="250" style="border:1px solid #e0e0e0;border-radius:6px;transition:height 0.15s ease"></iframe>

=== "Python"

    ```python
    from bead.items.span_labeling import create_interactive_span_item
    from bead.items.spans import SpanSpec

    item = create_interactive_span_item(
        text="The scientist presented the findings to the committee "
             "at the annual conference.",
        prompt="Create spans and relations for semantic role labeling.",
        label_set=[
            "Agent", "Patient", "Theme", "Recipient",
            "Instrument", "Location", "Time", "Predicate",
            "Stimulus", "Goal",
        ],
        span_spec=SpanSpec(
            interaction_mode="interactive",
            label_source="fixed",
            enable_relations=True,
            relation_label_source="fixed",
            relation_labels=[
                "agent-of", "patient-of", "theme-of",
                "recipient-of", "location-of", "time-of",
                "predicate-of",
            ],
            relation_directed=True,
        ),
    )
    ```

=== "Trial JSON"

    ```json
    {
      "type": "bead-span-label",
      "tokens": {
        "text": ["The", "scientist", "presented", "the", "findings",
                 "to", "the", "committee", "at", "the", "annual",
                 "conference", "."]
      },
      "span_spec": {
        "interaction_mode": "interactive",
        "label_source": "fixed",
        "labels": ["Agent", "Patient", "Theme", "Recipient",
                   "Instrument", "Location", "Time", "Predicate",
                   "Stimulus", "Goal"],
        "enable_relations": true,
        "relation_label_source": "fixed",
        "relation_labels": ["agent-of", "patient-of", "theme-of",
                            "recipient-of", "location-of", "time-of",
                            "predicate-of"],
        "relation_directed": true
      }
    }
    ```

### Span Relations (Wikidata)

Interactive entity linking and relation annotation with Wikidata search for both entity and relation labels. Useful for knowledge graph construction.

=== "Demo"

    <iframe src="../../gallery/demos/span-relations-wikidata.html" width="100%" height="250" style="border:1px solid #e0e0e0;border-radius:6px;transition:height 0.15s ease"></iframe>

=== "Python"

    ```python
    from bead.items.span_labeling import create_interactive_span_item
    from bead.items.spans import SpanSpec

    item = create_interactive_span_item(
        text="Marie Curie was born in Warsaw and later became "
             "a professor at the University of Paris.",
        prompt="Link entities via Wikidata and draw relations between them.",
        label_source="wikidata",
        span_spec=SpanSpec(
            interaction_mode="interactive",
            label_source="wikidata",
            enable_relations=True,
            relation_label_source="wikidata",
            relation_directed=True,
        ),
    )
    ```

=== "Trial JSON"

    ```json
    {
      "type": "bead-span-label",
      "tokens": {
        "text": ["Marie", "Curie", "was", "born", "in", "Warsaw",
                 "and", "later", "became", "a", "professor", "at",
                 "the", "University", "of", "Paris", "."]
      },
      "span_spec": {
        "interaction_mode": "interactive",
        "label_source": "wikidata",
        "enable_relations": true,
        "relation_label_source": "wikidata",
        "relation_directed": true,
        "wikidata_language": "en"
      }
    }
    ```
