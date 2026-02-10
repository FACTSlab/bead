/**
 * bead-slider-rating plugin
 *
 * jsPsych plugin for continuous/slider rating judgments.
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

/** Slider rating trial parameters */
export interface SliderRatingTrialParams {
  /** The prompt/question to display */
  prompt: string | null;
  /** HTML stimulus to display */
  stimulus: string;
  /** Position of the prompt relative to the stimulus */
  prompt_position: PromptPosition;
  /** Minimum slider value */
  slider_min: number;
  /** Maximum slider value */
  slider_max: number;
  /** Slider step size */
  step: number;
  /** Initial slider position */
  slider_start: number;
  /** Labels for slider endpoints */
  labels: string[];
  /** Whether the slider must be moved before continuing */
  require_movement: boolean;
  /** Text for the continue button */
  button_label: string;
  /** Complete item and template metadata */
  metadata: BeadMetadata;
}

/** Plugin info constant */
const info: PluginInfo = {
  name: "bead-slider-rating",
  parameters: {
    prompt: {
      type: 8, // ParameterType.HTML_STRING
      default: null,
    },
    stimulus: {
      type: 8, // ParameterType.HTML_STRING
      default: "",
    },
    prompt_position: {
      type: 1, // ParameterType.STRING
      default: "above",
    },
    slider_min: {
      type: 2, // ParameterType.INT
      default: 0,
    },
    slider_max: {
      type: 2, // ParameterType.INT
      default: 100,
    },
    step: {
      type: 2, // ParameterType.INT
      default: 1,
    },
    slider_start: {
      type: 2, // ParameterType.INT
      default: 50,
    },
    labels: {
      type: 1, // ParameterType.STRING
      default: [],
      array: true,
    },
    require_movement: {
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
 * BeadSliderRatingPlugin - jsPsych plugin for slider-based judgments
 */
class BeadSliderRatingPlugin implements JsPsychPlugin<typeof info, SliderRatingTrialParams> {
  static info = info;

  private jsPsych: JsPsych;

  constructor(jsPsych: JsPsych) {
    this.jsPsych = jsPsych;
  }

  trial(display_element: HTMLElement, trial: SliderRatingTrialParams): void {
    let slider_value = trial.slider_start;
    let has_moved = false;

    const start_time = performance.now();

    // Build HTML
    let html = '<div class="bead-slider-container">';

    if (trial.prompt !== null && trial.prompt_position === "above") {
      html += `<div class="bead-slider-prompt">${trial.prompt}</div>`;
    }

    if (trial.stimulus) {
      html += `<div class="bead-slider-stimulus">${trial.stimulus}</div>`;
    }

    if (trial.prompt !== null && trial.prompt_position === "below") {
      html += `<div class="bead-slider-prompt">${trial.prompt}</div>`;
    }

    html += '<div class="bead-slider-wrapper">';

    // Labels row
    if (trial.labels.length > 0) {
      html += '<div class="bead-slider-labels">';
      for (const label of trial.labels) {
        html += `<span class="bead-slider-label">${label}</span>`;
      }
      html += "</div>";
    }

    // Slider input
    html += `<input type="range" class="bead-slider-input" min="${trial.slider_min}" max="${trial.slider_max}" step="${trial.step}" value="${trial.slider_start}">`;

    // Value display
    html += `<div class="bead-slider-value">${trial.slider_start}</div>`;

    html += "</div>"; // Close wrapper

    // Continue button
    const disabled = trial.require_movement ? "disabled" : "";
    html += `
      <div class="bead-slider-button-container">
        <button class="bead-button bead-continue-button" id="bead-slider-continue" ${disabled}>
          ${trial.button_label}
        </button>
      </div>
    `;

    html += "</div>"; // Close container

    display_element.innerHTML = html;

    // Slider listener
    const slider = display_element.querySelector<HTMLInputElement>(".bead-slider-input");
    const value_display = display_element.querySelector<HTMLDivElement>(".bead-slider-value");
    const continue_button =
      display_element.querySelector<HTMLButtonElement>("#bead-slider-continue");

    if (slider) {
      slider.addEventListener("input", () => {
        slider_value = Number.parseFloat(slider.value);
        has_moved = true;
        if (value_display) {
          value_display.textContent = String(slider_value);
        }
        if (continue_button && trial.require_movement) {
          continue_button.disabled = false;
        }
      });
    }

    // Continue button listener
    if (continue_button) {
      continue_button.addEventListener("click", () => {
        if (!trial.require_movement || has_moved) {
          end_trial();
        }
      });
    }

    const end_trial = (): void => {
      const rt = performance.now() - start_time;

      const trial_data: Record<string, unknown> = {
        ...trial.metadata,
        response: slider_value,
        rt: rt,
      };

      display_element.innerHTML = "";
      this.jsPsych.finishTrial(trial_data);
    };
  }
}

export { BeadSliderRatingPlugin };
