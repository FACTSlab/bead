/**
 * bead-categorical plugin
 *
 * jsPsych plugin for categorical classification (unordered category selection).
 *
 * Follows the same metadata-spreading pattern as all other bead plugins.
 *
 * @author Bead Project
 * @version 0.2.0
 */

import type { JsPsych, JsPsychPlugin, PluginInfo } from "../types/jspsych.js";

/** Position of the prompt relative to the stimulus */
type PromptPosition = "above" | "below";

/** Bead item/template metadata */
interface BeadMetadata {
  [key: string]: unknown;
}

/** Categorical trial parameters */
export interface CategoricalTrialParams {
  /** The prompt/question to display */
  prompt: string;
  /** HTML stimulus to display */
  stimulus: string;
  /** Category labels (unordered) */
  categories: string[];
  /** Position of the prompt relative to the stimulus */
  prompt_position: PromptPosition;
  /** Whether to require a response before continuing */
  require_response: boolean;
  /** Text for the continue button */
  button_label: string;
  /** Complete item and template metadata */
  metadata: BeadMetadata;
}

/** Plugin info constant */
const info: PluginInfo = {
  name: "bead-categorical",
  parameters: {
    prompt: {
      type: 8, // ParameterType.HTML_STRING
      default: "Select a category:",
    },
    stimulus: {
      type: 8, // ParameterType.HTML_STRING
      default: "",
    },
    categories: {
      type: 1, // ParameterType.STRING
      default: [],
      array: true,
    },
    prompt_position: {
      type: 1, // ParameterType.STRING
      default: "above",
    },
    require_response: {
      type: 0, // ParameterType.BOOL
      default: true,
    },
    button_label: {
      type: 1, // ParameterType.STRING
      default: "Continue",
    },
    metadata: {
      type: 12, // ParameterType.OBJECT
      default: {},
    },
  },
};

/**
 * BeadCategoricalPlugin - jsPsych plugin for unordered category selection
 */
class BeadCategoricalPlugin implements JsPsychPlugin<typeof info, CategoricalTrialParams> {
  static info = info;

  private jsPsych: JsPsych;

  constructor(jsPsych: JsPsych) {
    this.jsPsych = jsPsych;
  }

  trial(display_element: HTMLElement, trial: CategoricalTrialParams): void {
    let selected_index: number | null = null;

    const start_time = performance.now();

    // Build HTML
    let html = '<div class="bead-categorical-container">';

    if (trial.prompt && trial.prompt_position === "above") {
      html += `<div class="bead-categorical-prompt">${trial.prompt}</div>`;
    }

    if (trial.stimulus) {
      html += `<div class="bead-categorical-stimulus">${trial.stimulus}</div>`;
    }

    if (trial.prompt && trial.prompt_position === "below") {
      html += `<div class="bead-categorical-prompt">${trial.prompt}</div>`;
    }

    html += '<div class="bead-categorical-options">';
    for (let i = 0; i < trial.categories.length; i++) {
      html += `<button class="bead-button bead-categorical-button" data-index="${i}">${trial.categories[i]}</button>`;
    }
    html += "</div>";

    // Continue button
    const disabled = trial.require_response ? "disabled" : "";
    html += `
      <div class="bead-categorical-button-container">
        <button class="bead-button bead-continue-button" id="bead-categorical-continue" ${disabled}>
          ${trial.button_label}
        </button>
      </div>
    `;

    html += "</div>";

    display_element.innerHTML = html;

    // Category button listeners
    const buttons = display_element.querySelectorAll<HTMLButtonElement>(".bead-categorical-button");
    const continueBtn = display_element.querySelector<HTMLButtonElement>(
      "#bead-categorical-continue",
    );

    for (const button of buttons) {
      button.addEventListener("click", (e) => {
        const target = e.currentTarget as HTMLButtonElement;
        const indexAttr = target.getAttribute("data-index");
        if (indexAttr !== null) {
          selected_index = Number.parseInt(indexAttr, 10);

          // Visual feedback
          for (const btn of buttons) {
            btn.classList.remove("selected");
          }
          target.classList.add("selected");

          if (continueBtn) {
            continueBtn.disabled = false;
          }
        }
      });
    }

    if (continueBtn) {
      continueBtn.addEventListener("click", () => {
        if (!trial.require_response || selected_index !== null) {
          end_trial();
        }
      });
    }

    const end_trial = (): void => {
      const rt = performance.now() - start_time;

      const trial_data: Record<string, unknown> = {
        ...trial.metadata,
        response: selected_index !== null ? trial.categories[selected_index] : null,
        response_index: selected_index,
        rt: rt,
      };

      display_element.innerHTML = "";
      this.jsPsych.finishTrial(trial_data);
    };
  }
}

export { BeadCategoricalPlugin };
