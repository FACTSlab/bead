/**
 * bead-forced-choice plugin
 *
 * jsPsych plugin for comparative judgments and forced choice tasks.
 *
 * Features:
 * - Side-by-side stimulus display
 * - Button or keyboard selection
 * - Optional similarity rating after choice
 * - Material Design card layout
 * - Preserves all item and template metadata
 *
 * @author Bead Project
 * @version 0.2.0
 */

import type { JsPsych, JsPsychPlugin, KeyboardResponseInfo, PluginInfo } from "../types/jspsych.js";

/** Bead item/template metadata */
interface BeadMetadata {
  [key: string]: unknown;
}

/** Position type for left/right alternatives */
type Position = "left" | "right";

/** Layout direction for alternatives */
type Layout = "horizontal" | "vertical";

/** Forced choice trial parameters */
export interface ForcedChoiceTrialParams {
  /** The prompt/question to display */
  prompt: string;
  /** Array of alternatives to choose from */
  alternatives: string[];
  /** Layout direction for alternatives */
  layout: Layout;
  /** Whether to randomize left/right position */
  randomize_position: boolean;
  /** Enable keyboard responses (1/2 or left/right arrow) */
  enable_keyboard: boolean;
  /** Whether to require a response */
  require_response: boolean;
  /** Text for the continue button (if applicable) */
  button_label: string;
  /** Complete item and template metadata */
  metadata: BeadMetadata;
}

/** Forced choice response data */
interface ForcedChoiceResponse {
  choice: string | null;
  choice_index: number | null;
  position: Position | null;
  rt: number | null;
}

/** Plugin info constant */
const info: PluginInfo = {
  name: "bead-forced-choice",
  parameters: {
    prompt: {
      type: 8, // ParameterType.HTML_STRING
      default: "Which do you prefer?",
    },
    alternatives: {
      type: 1, // ParameterType.STRING
      default: [],
      array: true,
    },
    layout: {
      type: 1, // ParameterType.STRING
      default: "horizontal",
    },
    randomize_position: {
      type: 0, // ParameterType.BOOL
      default: true,
    },
    enable_keyboard: {
      type: 0, // ParameterType.BOOL
      default: true,
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
 * BeadForcedChoicePlugin - jsPsych plugin for comparative judgments
 */
class BeadForcedChoicePlugin implements JsPsychPlugin<typeof info, ForcedChoiceTrialParams> {
  static info = info;

  private jsPsych: JsPsych;

  constructor(jsPsych: JsPsych) {
    this.jsPsych = jsPsych;
  }

  trial(display_element: HTMLElement, trial: ForcedChoiceTrialParams): void {
    const response: ForcedChoiceResponse = {
      choice: null,
      choice_index: null,
      position: null,
      rt: null,
    };

    const start_time = performance.now();

    // Randomize position if requested
    let left_index = 0;
    let right_index = 1;
    if (trial.randomize_position && Math.random() < 0.5) {
      left_index = 1;
      right_index = 0;
    }

    // Create HTML
    let html = '<div class="bead-forced-choice-container">';

    if (trial.prompt) {
      html += `<div class="bead-forced-choice-prompt">${trial.prompt}</div>`;
    }

    html += `<div class="bead-forced-choice-alternatives bead-layout-${trial.layout}">`;

    // Left alternative
    html += `
      <div class="bead-card bead-alternative" data-index="${left_index}" data-position="left">
        <div class="bead-alternative-label">Option 1</div>
        <div class="bead-alternative-content">${trial.alternatives[left_index] ?? "Alternative A"}</div>
        <button class="bead-button bead-choice-button" data-index="${left_index}" data-position="left">
          Select
        </button>
      </div>
    `;

    // Right alternative
    html += `
      <div class="bead-card bead-alternative" data-index="${right_index}" data-position="right">
        <div class="bead-alternative-label">Option 2</div>
        <div class="bead-alternative-content">${trial.alternatives[right_index] ?? "Alternative B"}</div>
        <button class="bead-button bead-choice-button" data-index="${right_index}" data-position="right">
          Select
        </button>
      </div>
    `;

    html += "</div>"; // Close alternatives

    html += "</div>"; // Close container

    display_element.innerHTML = html;

    // Add event listeners for card clicks (primary interaction)
    const alternative_cards = display_element.querySelectorAll<HTMLDivElement>(".bead-alternative");
    for (const card of alternative_cards) {
      card.addEventListener("click", () => {
        const indexAttr = card.getAttribute("data-index");
        const positionAttr = card.getAttribute("data-position") as Position | null;
        if (indexAttr !== null && positionAttr !== null) {
          const index = Number.parseInt(indexAttr, 10);
          select_choice(index, positionAttr);
        }
      });
    }

    // Keyboard listener
    let keyboard_listener: unknown = null;
    if (trial.enable_keyboard) {
      keyboard_listener = this.jsPsych.pluginAPI.getKeyboardResponse({
        callback_function: (info: KeyboardResponseInfo) => {
          const key = info.key;
          if (key === "1" || key === "ArrowLeft") {
            select_choice(left_index, "left");
          } else if (key === "2" || key === "ArrowRight") {
            select_choice(right_index, "right");
          }
        },
        valid_responses: ["1", "2", "ArrowLeft", "ArrowRight"],
        rt_method: "performance",
        persist: false,
        allow_held_key: false,
      });
    }

    const select_choice = (index: number, position: Position): void => {
      // Update response
      response.choice = trial.alternatives[index] ?? null;
      response.choice_index = index;
      response.position = position;
      response.rt = performance.now() - start_time;

      // Visual feedback
      const alternative_cards =
        display_element.querySelectorAll<HTMLDivElement>(".bead-alternative");
      for (const card of alternative_cards) {
        card.classList.remove("selected");
      }
      const selected_card = display_element.querySelector<HTMLDivElement>(
        `.bead-alternative[data-position="${position}"]`,
      );
      if (selected_card) {
        selected_card.classList.add("selected");
      }

      // End trial immediately or after delay
      setTimeout(() => {
        end_trial();
      }, 300); // Small delay for visual feedback
    };

    const end_trial = (): void => {
      // Kill keyboard listener
      if (keyboard_listener) {
        this.jsPsych.pluginAPI.cancelKeyboardResponse(keyboard_listener);
      }

      // Preserve all metadata and add response data
      const trial_data: Record<string, unknown> = {
        ...trial.metadata, // Spread all metadata
        choice: response.choice,
        choice_index: response.choice_index,
        position_chosen: response.position,
        left_index: left_index,
        right_index: right_index,
        rt: response.rt,
      };

      // Clear display
      display_element.innerHTML = "";

      // End trial
      this.jsPsych.finishTrial(trial_data);
    };
  }
}

export { BeadForcedChoicePlugin };
