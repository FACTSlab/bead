/**
 * bead-rating plugin
 *
 * jsPsych plugin for ordinal scale judgments (Likert scales, rating scales).
 *
 * Features:
 * - Discrete scales (e.g., 1-7, 1-9)
 * - Custom labels for scale points
 * - Keyboard shortcuts (number keys 1-9)
 * - Material Design styling
 * - Required response validation
 * - Preserves all item and template metadata
 *
 * @author Bead Project
 * @version 0.2.0
 */

import type { JsPsych, JsPsychPlugin, KeyboardResponseInfo, PluginInfo } from "../types/jspsych.js";

/** Position of the prompt relative to the stimulus */
type PromptPosition = "above" | "below";

/** Bead item/template metadata */
interface BeadMetadata {
  [key: string]: unknown;
}

/** Rating trial parameters */
export interface RatingTrialParams {
  /** The prompt to display above the rating scale */
  prompt: string | null;
  /** HTML stimulus to display */
  stimulus: string;
  /** Position of the prompt relative to the stimulus */
  prompt_position: PromptPosition;
  /** Minimum value of the scale */
  scale_min: number;
  /** Maximum value of the scale */
  scale_max: number;
  /** Labels for specific scale points (e.g., {1: "Strongly Disagree", 7: "Strongly Agree"}) */
  scale_labels: Record<number, string>;
  /** Whether to require a response before continuing */
  require_response: boolean;
  /** Text for the continue button */
  button_label: string;
  /** Complete item and template metadata (automatically populated from trial.data) */
  metadata: BeadMetadata;
}

/** Rating response data */
interface RatingResponse {
  rating: number | null;
  rt: number | null;
}

/** Plugin info constant */
const info: PluginInfo = {
  name: "bead-rating",
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
    scale_min: {
      type: 2, // ParameterType.INT
      default: 1,
    },
    scale_max: {
      type: 2, // ParameterType.INT
      default: 7,
    },
    scale_labels: {
      type: 12, // ParameterType.OBJECT
      default: {},
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
 * BeadRatingPlugin - jsPsych plugin for ordinal scale judgments
 */
class BeadRatingPlugin implements JsPsychPlugin<typeof info, RatingTrialParams> {
  static info = info;

  private jsPsych: JsPsych;

  constructor(jsPsych: JsPsych) {
    this.jsPsych = jsPsych;
  }

  trial(display_element: HTMLElement, trial: RatingTrialParams): void {
    const response: RatingResponse = {
      rating: null,
      rt: null,
    };

    const start_time = performance.now();

    // Create HTML
    let html = '<div class="bead-rating-container">';

    if (trial.prompt !== null && trial.prompt_position === "above") {
      html += `<div class="bead-rating-prompt">${trial.prompt}</div>`;
    }

    if (trial.stimulus) {
      html += `<div class="bead-rating-stimulus">${trial.stimulus}</div>`;
    }

    if (trial.prompt !== null && trial.prompt_position === "below") {
      html += `<div class="bead-rating-prompt">${trial.prompt}</div>`;
    }

    html += '<div class="bead-rating-scale">';

    // Create rating buttons
    for (let i = trial.scale_min; i <= trial.scale_max; i++) {
      const label = trial.scale_labels[i] ?? i;
      html += `
        <div class="bead-rating-option">
          <button class="bead-rating-button" data-value="${i}">${i}</button>
          <div class="bead-rating-label">${label}</div>
        </div>
      `;
    }

    html += "</div>"; // Close scale

    // Continue button
    html += `
      <div class="bead-rating-button-container">
        <button class="bead-button bead-continue-button" id="bead-rating-continue" disabled>
          ${trial.button_label}
        </button>
      </div>
    `;

    html += "</div>"; // Close container

    display_element.innerHTML = html;

    // Add event listeners for rating buttons
    const rating_buttons =
      display_element.querySelectorAll<HTMLButtonElement>(".bead-rating-button");
    for (const button of rating_buttons) {
      button.addEventListener("click", (e) => {
        const target = e.target as HTMLButtonElement;
        const valueAttr = target.getAttribute("data-value");
        if (valueAttr !== null) {
          const value = Number.parseInt(valueAttr, 10);
          select_rating(value);
        }
      });
    }

    // Keyboard listener
    const keyboard_listener = this.jsPsych.pluginAPI.getKeyboardResponse({
      callback_function: (info: KeyboardResponseInfo) => {
        // Check if key is 1-9
        const key = info.key;
        const num = Number.parseInt(key, 10);
        if (!Number.isNaN(num) && num >= trial.scale_min && num <= trial.scale_max) {
          select_rating(num);
        }
      },
      valid_responses: "ALL_KEYS",
      rt_method: "performance",
      persist: true,
      allow_held_key: false,
    });

    // Continue button listener
    const continue_button =
      display_element.querySelector<HTMLButtonElement>("#bead-rating-continue");
    if (continue_button) {
      continue_button.addEventListener("click", () => {
        if (response.rating !== null || !trial.require_response) {
          end_trial();
        }
      });
    }

    const select_rating = (value: number): void => {
      // Update response
      response.rating = value;
      response.rt = performance.now() - start_time;

      // Update UI
      for (const btn of rating_buttons) {
        btn.classList.remove("selected");
      }
      const selected_button = display_element.querySelector<HTMLButtonElement>(
        `[data-value="${value}"]`,
      );
      if (selected_button) {
        selected_button.classList.add("selected");
      }

      // Enable continue button
      if (continue_button) {
        continue_button.disabled = false;
      }
    };

    const end_trial = (): void => {
      // Kill keyboard listener
      if (keyboard_listener) {
        this.jsPsych.pluginAPI.cancelKeyboardResponse(keyboard_listener);
      }

      // Preserve all metadata from trial.metadata and add response data
      const trial_data: Record<string, unknown> = {
        ...trial.metadata, // Spread all metadata
        rating: response.rating,
        rt: response.rt,
      };

      // Clear display
      display_element.innerHTML = "";

      // End trial
      this.jsPsych.finishTrial(trial_data);
    };
  }
}

export { BeadRatingPlugin };
