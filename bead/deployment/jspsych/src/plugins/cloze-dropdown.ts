/**
 * bead-cloze-multi plugin
 *
 * jsPsych plugin for fill-in-the-blank tasks with multiple gaps.
 *
 * Features:
 * - Multiple gaps in text
 * - Dropdown (extensional constraints)
 * - Text input (intensional constraints or free text)
 * - Mixed field types
 * - Per-gap response time tracking
 * - Material Design form controls
 * - Preserves all item and template metadata
 *
 * @author Bead Project
 * @version 0.2.0
 */

import type { JsPsych, JsPsychPlugin, PluginInfo } from "../types/jspsych.js";

/** Field type for cloze fields */
type ClozeFieldType = "dropdown" | "text";

/** Unfilled slot from bead metadata */
interface UnfilledSlot {
  slot_name: string;
  position: number;
  constraint_ids: string[];
}

/** Field configuration for cloze task */
export interface ClozeFieldConfig {
  slot_name?: string;
  position?: number;
  type: ClozeFieldType;
  options?: (string | { value: string; label: string })[];
  placeholder?: string;
}

/** Bead item/template metadata */
interface BeadMetadata {
  unfilled_slots?: UnfilledSlot[];
  [key: string]: unknown;
}

/** Cloze trial parameters */
export interface ClozeTrialParams {
  /** The text with gaps (use %% or {{slot_name}} for gaps) */
  text: string | null;
  prompt?: string;
  canary?: string;
  /** Field configurations (auto-generated from unfilled_slots) */
  fields: ClozeFieldConfig[];
  /** Whether to require all fields to be filled */
  require_all: boolean;
  /** Text for the continue button */
  button_label: string;
  /** Complete item and template metadata */
  metadata: BeadMetadata;
}

/** Per-field responses */
type ClozeResponses = Record<string, string>;

/** Per-field response times */
type ClozeResponseTimes = Record<string, number>;

/** Per-field start times */
type FieldStartTimes = Record<string, number>;

/** Plugin info constant */
const info: PluginInfo = {
  name: "bead-cloze-multi",
  parameters: {
    text: {
      type: 8, // ParameterType.HTML_STRING
      default: null,
    },
    prompt: {
      type: 8, // ParameterType.HTML_STRING
      default: "",
    },
    canary: {
      type: 8, // ParameterType.HTML_STRING
      default: "",
    },
    fields: {
      type: 13, // ParameterType.COMPLEX
      default: [],
      array: true,
    },
    require_all: {
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
  data: {},
};

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

/**
 * BeadClozeMultiPlugin - jsPsych plugin for fill-in-the-blank tasks
 */
class BeadClozeMultiPlugin implements JsPsychPlugin<typeof info, ClozeTrialParams> {
  static info = info;

  private jsPsych: JsPsych;

  constructor(jsPsych: JsPsych) {
    this.jsPsych = jsPsych;
  }

  trial(display_element: HTMLElement, trial: ClozeTrialParams): void {
    const responses: ClozeResponses = Object.create(null) as ClozeResponses;
    const response_times: ClozeResponseTimes = Object.create(null) as ClozeResponseTimes;
    const field_start_times: FieldStartTimes = Object.create(null) as FieldStartTimes;

    const start_time = performance.now();

    const fields =
      trial.fields.length === 0 && trial.metadata.unfilled_slots
        ? trial.metadata.unfilled_slots.map((slot) => ({
            slot_name: slot.slot_name,
            position: slot.position,
            type: "text" as const,
            placeholder: slot.slot_name,
          }))
        : trial.fields;

    // Generate HTML
    let html = '<div class="bead-cloze-container">';
    if (trial.prompt) html += `<div class="bead-cloze-prompt">${trial.prompt}</div>`;

    if (trial.text) {
      // Replace gaps with input fields
      let processed_text = trial.text;
      fields.forEach((field, index) => {
        const field_id = `bead-cloze-field-${index}`;
        let field_html: string;

        if (field.type === "dropdown") {
          if (!field.options?.length)
            throw new Error(`Cloze field ${field.slot_name ?? index} has no options`);
          // Dropdown field
          const optionsHtml = field.options
            .map((opt) => {
              const { value, label } = typeof opt === "string" ? { value: opt, label: opt } : opt;
              return `<option value="${escapeHtml(value)}">${escapeHtml(label)}</option>`;
            })
            .join("");
          field_html = `
            <select aria-label="${escapeHtml(field.slot_name ?? `Gap ${index + 1}`)}" class="bead-dropdown bead-cloze-field" id="${field_id}" data-field="${index}">
              <option value="" selected disabled>${escapeHtml(field.placeholder || "Select...")}</option>
              ${optionsHtml}
            </select>
          `;
        } else {
          // Text input field
          field_html = `
            <input type="text"
                   aria-label="${escapeHtml(field.slot_name ?? `Gap ${index + 1}`)}"
                   class="bead-text-field bead-cloze-field"
                   id="${field_id}"
                   data-field="${index}"
                   placeholder="${escapeHtml(field.placeholder ?? "")}" />
          `;
        }

        // Replace placeholder in text
        const placeholder = field.slot_name ? `{{${field.slot_name}}}` : "%%";
        processed_text = processed_text.replace(placeholder, () => field_html);
      });

      html += `<div class="bead-cloze-text">${processed_text}</div>`;
    }

    if (trial.canary) {
      html += `<span class="canary" hidden aria-hidden="true">${escapeHtml(trial.canary)}</span>`;
    }

    // Continue button
    html += `
      <div class="bead-cloze-button-container">
        <button class="bead-button bead-continue-button" id="bead-cloze-continue" ${trial.require_all ? "disabled" : ""}>
          ${trial.button_label}
        </button>
      </div>
    `;

    html += "</div>"; // Close container

    display_element.innerHTML = html;

    // Add event listeners for input fields
    const input_fields = display_element.querySelectorAll<HTMLInputElement | HTMLSelectElement>(
      ".bead-cloze-field",
    );
    for (const field of input_fields) {
      const index = field.dataset["field"];
      if (index === undefined) continue;
      const field_index = fields[Number(index)]?.slot_name ?? index;

      // Track when user starts interacting with this field
      field.addEventListener("focus", () => {
        if (field_start_times[field_index] === undefined) {
          field_start_times[field_index] = performance.now();
        }
      });

      // Track responses
      field.addEventListener("change", () => {
        responses[field_index] = field.value;

        // Record response time for this field
        const startTime = field_start_times[field_index];
        if (startTime !== undefined) {
          response_times[field_index] = performance.now() - startTime;
        }

        check_completion();
      });

      field.addEventListener("input", () => {
        responses[field_index] = field.value;
        check_completion();
      });
    }

    // Continue button listener
    const continue_button =
      display_element.querySelector<HTMLButtonElement>("#bead-cloze-continue");
    if (continue_button) {
      continue_button.addEventListener("click", () => {
        end_trial();
      });
    }

    const check_completion = (): void => {
      if (trial.require_all && continue_button) {
        // Check if all fields are filled
        const all_filled = fields.every((field, index) => {
          const response = responses[field.slot_name ?? index.toString()];
          return response !== undefined && response.trim() !== "";
        });
        continue_button.disabled = !all_filled;
      }
    };

    const end_trial = (): void => {
      // Gather all responses
      const rt = performance.now() - start_time;
      const trial_data: Record<string, unknown> = {
        ...trial.metadata, // Preserve all metadata
        responses: responses,
        response_times: response_times,
        rt,
        total_rt: rt,
      };

      // Clear display
      display_element.innerHTML = "";

      // End trial
      this.jsPsych.finishTrial(trial_data);
    };
  }
}

export { BeadClozeMultiPlugin };
