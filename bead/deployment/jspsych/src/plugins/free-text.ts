/**
 * bead-free-text plugin
 *
 * jsPsych plugin for open-ended text responses (single-line or multiline).
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

/** Free text trial parameters */
export interface FreeTextTrialParams {
  /** The prompt/question to display */
  prompt: string;
  /** HTML stimulus to display */
  stimulus: string;
  /** Whether to use a textarea (true) or single-line input (false) */
  multiline: boolean;
  /** Minimum character length (0 = no minimum) */
  min_length: number;
  /** Maximum character length (0 = unlimited) */
  max_length: number;
  /** Placeholder text */
  placeholder: string;
  /** Number of rows for textarea */
  rows: number;
  /** Position of the prompt relative to the stimulus */
  prompt_position: PromptPosition;
  /** Whether to require a response */
  require_response: boolean;
  /** Text for the continue button */
  button_label: string;
  /** Complete item and template metadata */
  metadata: BeadMetadata;
}

/** Plugin info constant */
const info: PluginInfo = {
  name: "bead-free-text",
  parameters: {
    prompt: {
      type: 8, // ParameterType.HTML_STRING
      default: "Enter your response:",
    },
    stimulus: {
      type: 8, // ParameterType.HTML_STRING
      default: "",
    },
    prompt_position: {
      type: 1, // ParameterType.STRING
      default: "above",
    },
    multiline: {
      type: 0, // ParameterType.BOOL
      default: false,
    },
    min_length: {
      type: 2, // ParameterType.INT
      default: 0,
    },
    max_length: {
      type: 2, // ParameterType.INT
      default: 0,
    },
    placeholder: {
      type: 1, // ParameterType.STRING
      default: "",
    },
    rows: {
      type: 2, // ParameterType.INT
      default: 4,
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
 * BeadFreeTextPlugin - jsPsych plugin for open-ended text responses
 */
class BeadFreeTextPlugin implements JsPsychPlugin<typeof info, FreeTextTrialParams> {
  static info = info;

  private jsPsych: JsPsych;

  constructor(jsPsych: JsPsych) {
    this.jsPsych = jsPsych;
  }

  trial(display_element: HTMLElement, trial: FreeTextTrialParams): void {
    const start_time = performance.now();

    // Build HTML
    let html = '<div class="bead-free-text-container">';

    if (trial.prompt && trial.prompt_position === "above") {
      html += `<div class="bead-free-text-prompt">${trial.prompt}</div>`;
    }

    if (trial.stimulus) {
      html += `<div class="bead-free-text-stimulus">${trial.stimulus}</div>`;
    }

    if (trial.prompt && trial.prompt_position === "below") {
      html += `<div class="bead-free-text-prompt">${trial.prompt}</div>`;
    }

    const maxAttr = trial.max_length > 0 ? ` maxlength="${trial.max_length}"` : "";
    const placeholderAttr = trial.placeholder ? ` placeholder="${trial.placeholder}"` : "";

    if (trial.multiline) {
      html += `<textarea class="bead-free-text-input" id="bead-free-text-input" rows="${trial.rows}"${maxAttr}${placeholderAttr}></textarea>`;
    } else {
      html += `<input type="text" class="bead-free-text-input" id="bead-free-text-input"${maxAttr}${placeholderAttr}>`;
    }

    if (trial.max_length > 0) {
      html += `<div class="bead-free-text-counter"><span id="bead-char-count">0</span> / ${trial.max_length}</div>`;
    }

    // Continue button
    const disabled = trial.require_response ? "disabled" : "";
    html += `
      <div class="bead-free-text-button-container">
        <button class="bead-button bead-continue-button" id="bead-free-text-continue" ${disabled}>
          ${trial.button_label}
        </button>
      </div>
    `;

    html += "</div>";

    display_element.innerHTML = html;

    // Input listener
    const input = display_element.querySelector<HTMLInputElement | HTMLTextAreaElement>(
      "#bead-free-text-input",
    );
    const continueBtn = display_element.querySelector<HTMLButtonElement>(
      "#bead-free-text-continue",
    );
    const charCount = display_element.querySelector<HTMLSpanElement>("#bead-char-count");

    if (input) {
      input.addEventListener("input", () => {
        const len = input.value.length;
        if (charCount) charCount.textContent = String(len);
        if (continueBtn) {
          const meetsMin = len >= trial.min_length;
          const hasContent = input.value.trim().length > 0;
          continueBtn.disabled = trial.require_response && (!hasContent || !meetsMin);
        }
      });
      input.focus();
    }

    if (continueBtn) {
      continueBtn.addEventListener("click", () => {
        end_trial();
      });
    }

    const end_trial = (): void => {
      const rt = performance.now() - start_time;

      const trial_data: Record<string, unknown> = {
        ...trial.metadata,
        response: input ? input.value : "",
        rt: rt,
      };

      display_element.innerHTML = "";
      this.jsPsych.finishTrial(trial_data);
    };
  }
}

export { BeadFreeTextPlugin };
