/**
 * bead-binary-choice plugin
 *
 * jsPsych plugin for binary acceptability judgments (Yes/No, Acceptable/Unacceptable).
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

/** Binary choice trial parameters */
export interface BinaryChoiceTrialParams {
  /** The prompt/question to display */
  prompt: string;
  /** HTML stimulus to display (sentence, passage, etc.) */
  stimulus: string;
  /** Labels for the two choices */
  choices: [string, string];
  /** Position of the prompt relative to the stimulus */
  prompt_position: PromptPosition;
  /** Whether to require a response */
  require_response: boolean;
  /** Complete item and template metadata */
  metadata: BeadMetadata;
}

/** Plugin info constant */
const info: PluginInfo = {
  name: "bead-binary-choice",
  parameters: {
    prompt: {
      type: 8, // ParameterType.HTML_STRING
      default: "Is this sentence acceptable?",
    },
    stimulus: {
      type: 8, // ParameterType.HTML_STRING
      default: "",
    },
    choices: {
      type: 1, // ParameterType.STRING
      default: ["Yes", "No"],
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
    metadata: {
      type: 12, // ParameterType.OBJECT
      default: {},
    },
  },
};

/**
 * BeadBinaryChoicePlugin - jsPsych plugin for binary judgments
 */
class BeadBinaryChoicePlugin implements JsPsychPlugin<typeof info, BinaryChoiceTrialParams> {
  static info = info;

  private jsPsych: JsPsych;

  constructor(jsPsych: JsPsych) {
    this.jsPsych = jsPsych;
  }

  trial(display_element: HTMLElement, trial: BinaryChoiceTrialParams): void {
    let response_index: number | null = null;
    let rt: number | null = null;

    const start_time = performance.now();

    // Build HTML
    let html = '<div class="bead-binary-choice-container">';

    if (trial.prompt && trial.prompt_position === "above") {
      html += `<div class="bead-binary-choice-prompt">${trial.prompt}</div>`;
    }

    if (trial.stimulus) {
      html += `<div class="bead-binary-choice-stimulus">${trial.stimulus}</div>`;
    }

    if (trial.prompt && trial.prompt_position === "below") {
      html += `<div class="bead-binary-choice-prompt">${trial.prompt}</div>`;
    }

    html += '<div class="bead-binary-choice-buttons">';
    for (let i = 0; i < trial.choices.length; i++) {
      html += `<button class="bead-button bead-binary-button" data-index="${i}">${trial.choices[i]}</button>`;
    }
    html += "</div>";

    html += "</div>";

    display_element.innerHTML = html;

    // Button listeners
    const buttons = display_element.querySelectorAll<HTMLButtonElement>(".bead-binary-button");
    for (const button of buttons) {
      button.addEventListener("click", (e) => {
        const target = e.target as HTMLButtonElement;
        const indexAttr = target.getAttribute("data-index");
        if (indexAttr !== null) {
          select_choice(Number.parseInt(indexAttr, 10));
        }
      });
    }

    const select_choice = (index: number): void => {
      response_index = index;
      rt = performance.now() - start_time;

      // Visual feedback
      for (const btn of buttons) {
        btn.classList.remove("selected");
      }
      const selected = display_element.querySelector<HTMLButtonElement>(
        `.bead-binary-button[data-index="${index}"]`,
      );
      if (selected) {
        selected.classList.add("selected");
      }

      setTimeout(() => {
        end_trial();
      }, 200);
    };

    const end_trial = (): void => {
      const trial_data: Record<string, unknown> = {
        ...trial.metadata,
        response: response_index,
        response_label: response_index !== null ? trial.choices[response_index] : null,
        rt: rt,
      };

      display_element.innerHTML = "";
      this.jsPsych.finishTrial(trial_data);
    };
  }
}

export { BeadBinaryChoicePlugin };
