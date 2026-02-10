/**
 * bead-multi-select plugin
 *
 * jsPsych plugin for selecting one or more options from a set (checkboxes).
 *
 * Follows the same metadata-spreading pattern as all other bead plugins.
 *
 * @author Bead Project
 * @version 0.2.0
 */

import type { JsPsych, JsPsychPlugin, PluginInfo } from "../types/jspsych.js";

/** Bead item/template metadata */
interface BeadMetadata {
  [key: string]: unknown;
}

/** Prompt position relative to stimulus */
type PromptPosition = "above" | "below";

/** Multi-select trial parameters */
export interface MultiSelectTrialParams {
  /** The prompt/question to display */
  prompt: string;
  /** HTML stimulus to display */
  stimulus: string;
  /** Where to place the prompt relative to the stimulus */
  prompt_position: PromptPosition;
  /** Options to select from */
  options: string[];
  /** Minimum number of selections required */
  min_selections: number;
  /** Maximum number of selections allowed (0 = unlimited) */
  max_selections: number;
  /** Whether to require at least min_selections */
  require_response: boolean;
  /** Text for the continue button */
  button_label: string;
  /** Complete item and template metadata */
  metadata: BeadMetadata;
}

/** Plugin info constant */
const info: PluginInfo = {
  name: "bead-multi-select",
  parameters: {
    prompt: {
      type: 8, // ParameterType.HTML_STRING
      default: "Select all that apply:",
    },
    stimulus: {
      type: 8, // ParameterType.HTML_STRING
      default: "",
    },
    prompt_position: {
      type: 1, // ParameterType.STRING
      default: "above",
    },
    options: {
      type: 1, // ParameterType.STRING
      default: [],
      array: true,
    },
    min_selections: {
      type: 2, // ParameterType.INT
      default: 1,
    },
    max_selections: {
      type: 2, // ParameterType.INT
      default: 0,
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
 * BeadMultiSelectPlugin - jsPsych plugin for checkbox-based multi-selection
 */
class BeadMultiSelectPlugin implements JsPsychPlugin<typeof info, MultiSelectTrialParams> {
  static info = info;

  private jsPsych: JsPsych;

  constructor(jsPsych: JsPsych) {
    this.jsPsych = jsPsych;
  }

  trial(display_element: HTMLElement, trial: MultiSelectTrialParams): void {
    const start_time = performance.now();

    // Determine if compact layout is appropriate
    const maxLen = Math.max(...trial.options.map((o) => o.length));
    const useCompact = maxLen < 25 && trial.options.length <= 6;

    // Build HTML
    let html = '<div class="bead-multi-select-container">';

    // Prompt (above)
    if (trial.prompt && trial.prompt_position === "above") {
      html += `<div class="bead-multi-select-prompt">${trial.prompt}</div>`;
    }

    if (trial.stimulus) {
      html += `<div class="bead-multi-select-stimulus">${trial.stimulus}</div>`;
    }

    // Prompt (below)
    if (trial.prompt && trial.prompt_position === "below") {
      html += `<div class="bead-multi-select-prompt">${trial.prompt}</div>`;
    }

    const compactClass = useCompact ? " bead-multi-select-compact" : "";
    html += `<div class="bead-multi-select-options${compactClass}">`;
    for (let i = 0; i < trial.options.length; i++) {
      const opt = trial.options[i] ?? "";
      html += `
        <label class="bead-multi-select-option">
          <input type="checkbox" class="bead-multi-select-checkbox" data-index="${i}" value="${opt}">
          <span class="bead-multi-select-label">${opt}</span>
        </label>
      `;
    }
    html += "</div>";

    // Continue button
    const disabled = trial.require_response ? "disabled" : "";
    html += `
      <div class="bead-multi-select-button-container">
        <button class="bead-button bead-continue-button" id="bead-multi-select-continue" ${disabled}>
          ${trial.button_label}
        </button>
      </div>
    `;

    html += "</div>";

    display_element.innerHTML = html;

    // Checkbox listeners
    const checkboxes = display_element.querySelectorAll<HTMLInputElement>(
      ".bead-multi-select-checkbox",
    );
    const continueBtn = display_element.querySelector<HTMLButtonElement>(
      "#bead-multi-select-continue",
    );

    const updateButton = (): void => {
      const checked = display_element.querySelectorAll<HTMLInputElement>(
        ".bead-multi-select-checkbox:checked",
      );
      const count = checked.length;

      // Enforce max_selections
      if (trial.max_selections > 0 && count >= trial.max_selections) {
        for (const cb of checkboxes) {
          if (!cb.checked) cb.disabled = true;
        }
      } else {
        for (const cb of checkboxes) {
          cb.disabled = false;
        }
      }

      if (continueBtn) {
        continueBtn.disabled = trial.require_response && count < trial.min_selections;
      }
    };

    for (const cb of checkboxes) {
      cb.addEventListener("change", updateButton);
    }

    if (continueBtn) {
      continueBtn.addEventListener("click", () => {
        end_trial();
      });
    }

    const end_trial = (): void => {
      const rt = performance.now() - start_time;
      const checked = display_element.querySelectorAll<HTMLInputElement>(
        ".bead-multi-select-checkbox:checked",
      );

      const selected: string[] = [];
      const selected_indices: number[] = [];
      for (const cb of checked) {
        selected.push(cb.value);
        const idx = cb.getAttribute("data-index");
        if (idx !== null) selected_indices.push(Number.parseInt(idx, 10));
      }

      const trial_data: Record<string, unknown> = {
        ...trial.metadata,
        selected: selected,
        selected_indices: selected_indices,
        rt: rt,
      };

      display_element.innerHTML = "";
      this.jsPsych.finishTrial(trial_data);
    };
  }
}

export { BeadMultiSelectPlugin };
