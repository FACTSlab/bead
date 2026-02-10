/**
 * bead-magnitude plugin
 *
 * jsPsych plugin for numeric magnitude input (bounded or unbounded).
 * Supports true magnitude estimation with a reference stimulus and an
 * exponential slider that maps linear position to exponential values.
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

/** Input mode for magnitude estimation */
type InputMode = "number" | "exp-slider";

// ── Exponential slider math ─────────────────────────────────────────

/** Compute the maximum internal x position from the reference value. */
function computeXMax(referenceValue: number): number {
  return 3 * 100 * Math.log(referenceValue + 1);
}

/** Map internal slider position x to display value. */
function xToValue(x: number): number {
  if (x <= 0) return 0;
  return Math.exp(x / 100) - 1;
}

/** Map display value back to internal slider position. */
function valueToX(value: number): number {
  if (value <= 0) return 0;
  return 100 * Math.log(value + 1);
}

/** Format a magnitude value for display. */
function formatValue(value: number): string {
  if (value >= 1_000_000) return "\u221E";
  if (value >= 10_000) return Math.round(value).toLocaleString();
  if (value >= 100) return Math.round(value).toString();
  if (value >= 10) return value.toFixed(1);
  if (value >= 1) return value.toFixed(2);
  if (value > 0) return value.toFixed(3);
  return "0";
}

/** Magnitude trial parameters */
export interface MagnitudeTrialParams {
  /** The prompt/question to display */
  prompt: string;
  /** HTML stimulus to display (the target) */
  stimulus: string;
  /** Where to place the prompt relative to the stimulus */
  prompt_position: PromptPosition;
  /** HTML for a reference stimulus (shown above target for magnitude estimation) */
  reference_stimulus: string;
  /** Numeric value assigned to the reference stimulus */
  reference_value: number;
  /** Unit label displayed next to the input or value display */
  unit: string;
  /** Input mode: "number" for numeric input, "exp-slider" for exponential slider */
  input_mode: InputMode;
  /** Arrow key step in internal x-units (exp-slider mode only). Default: 3. */
  arrow_step: number;
  /** Starting internal x position (null = handle hidden until first interaction) */
  slider_start: number | null;
  /** Minimum allowed value (number mode only, null = unbounded) */
  input_min: number | null;
  /** Maximum allowed value (number mode only, null = unbounded) */
  input_max: number | null;
  /** Step size for the input (number mode only) */
  step: number | null;
  /** Placeholder text for the input (number mode only) */
  placeholder: string;
  /** Whether to require a response */
  require_response: boolean;
  /** Text for the continue button */
  button_label: string;
  /** Complete item and template metadata */
  metadata: BeadMetadata;
}

/** Plugin info constant */
const info: PluginInfo = {
  name: "bead-magnitude",
  parameters: {
    prompt: {
      type: 8, // ParameterType.HTML_STRING
      default: "Enter a value:",
    },
    stimulus: {
      type: 8, // ParameterType.HTML_STRING
      default: "",
    },
    prompt_position: {
      type: 1, // ParameterType.STRING
      default: "above",
    },
    reference_stimulus: {
      type: 8, // ParameterType.HTML_STRING
      default: "",
    },
    reference_value: {
      type: 2, // ParameterType.INT
      default: 100,
    },
    unit: {
      type: 1, // ParameterType.STRING
      default: "",
    },
    input_mode: {
      type: 1, // ParameterType.STRING
      default: "number",
    },
    arrow_step: {
      type: 3, // ParameterType.FLOAT
      default: 3,
    },
    slider_start: {
      type: 3, // ParameterType.FLOAT
      default: null,
    },
    input_min: {
      type: 3, // ParameterType.FLOAT
      default: null,
    },
    input_max: {
      type: 3, // ParameterType.FLOAT
      default: null,
    },
    step: {
      type: 3, // ParameterType.FLOAT
      default: null,
    },
    placeholder: {
      type: 1, // ParameterType.STRING
      default: "",
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
 * BeadMagnitudePlugin - jsPsych plugin for numeric magnitude input
 */
class BeadMagnitudePlugin implements JsPsychPlugin<typeof info, MagnitudeTrialParams> {
  static info = info;

  private jsPsych: JsPsych;

  constructor(jsPsych: JsPsych) {
    this.jsPsych = jsPsych;
  }

  trial(display_element: HTMLElement, trial: MagnitudeTrialParams): void {
    const start_time = performance.now();
    const hasReference = trial.reference_stimulus !== "";

    // Build HTML
    let html = '<div class="bead-magnitude-container">';

    // Prompt (above position)
    if (trial.prompt && trial.prompt_position === "above") {
      html += `<div class="bead-magnitude-prompt">${trial.prompt}</div>`;
    }

    // Reference stimulus (for true magnitude estimation)
    if (hasReference) {
      html += '<div class="bead-magnitude-reference-header">';
      html += '<div class="bead-magnitude-section-label">Reference</div>';
      html += `<div class="bead-magnitude-reference-chip">${trial.reference_value}</div>`;
      html += "</div>";
      html += '<div class="bead-magnitude-reference">';
      html += `<div class="bead-magnitude-reference-text">${trial.reference_stimulus}</div>`;
      html += "</div>";
    }

    // Target stimulus
    if (trial.stimulus) {
      if (hasReference) {
        html += '<div class="bead-magnitude-section-label">Target</div>';
      }
      html += `<div class="bead-magnitude-stimulus">${trial.stimulus}</div>`;
    }

    // Prompt (below position)
    if (trial.prompt && trial.prompt_position === "below") {
      html += `<div class="bead-magnitude-prompt">${trial.prompt}</div>`;
    }

    // Input section
    if (trial.input_mode === "exp-slider") {
      html += this.buildExpSliderHTML(trial);
    } else {
      html += this.buildNumberInputHTML(trial);
    }

    // Continue button
    const disabled = trial.require_response ? "disabled" : "";
    html += `
      <div class="bead-magnitude-button-container">
        <button class="bead-button bead-continue-button" id="bead-magnitude-continue" ${disabled}>
          ${trial.button_label}
        </button>
      </div>
    `;

    html += "</div>";

    display_element.innerHTML = html;

    // Set up interaction
    if (trial.input_mode === "exp-slider") {
      this.setupExpSlider(display_element, trial, start_time, hasReference);
    } else {
      this.setupNumberInput(display_element, trial, start_time, hasReference);
    }
  }

  // ── Number input (existing behavior) ────────────────────────────

  private buildNumberInputHTML(trial: MagnitudeTrialParams): string {
    let html = '<div class="bead-magnitude-input-wrapper">';
    html += '<input type="number" class="bead-magnitude-input" id="bead-magnitude-input"';
    if (trial.input_min !== null) html += ` min="${trial.input_min}"`;
    if (trial.input_max !== null) html += ` max="${trial.input_max}"`;
    if (trial.step !== null) html += ` step="${trial.step}"`;
    if (trial.placeholder) html += ` placeholder="${trial.placeholder}"`;
    html += ">";
    if (trial.unit) {
      html += `<span class="bead-magnitude-unit">${trial.unit}</span>`;
    }
    html += "</div>";
    return html;
  }

  private setupNumberInput(
    display_element: HTMLElement,
    trial: MagnitudeTrialParams,
    start_time: number,
    hasReference: boolean,
  ): void {
    const input = display_element.querySelector<HTMLInputElement>("#bead-magnitude-input");
    const continueBtn = display_element.querySelector<HTMLButtonElement>(
      "#bead-magnitude-continue",
    );

    if (input) {
      input.addEventListener("input", () => {
        if (continueBtn) {
          continueBtn.disabled = trial.require_response && input.value.trim() === "";
        }
      });
      input.focus();
    }

    if (continueBtn) {
      continueBtn.addEventListener("click", () => {
        if (!trial.require_response || (input && input.value.trim() !== "")) {
          end_trial();
        }
      });
    }

    const end_trial = (): void => {
      const rt = performance.now() - start_time;
      const value = input ? Number.parseFloat(input.value) : null;

      const trial_data: Record<string, unknown> = {
        ...trial.metadata,
        response: Number.isNaN(value ?? Number.NaN) ? null : value,
        rt: rt,
      };

      if (hasReference) {
        trial_data["reference_value"] = trial.reference_value;
      }

      display_element.innerHTML = "";
      this.jsPsych.finishTrial(trial_data);
    };
  }

  // ── Exponential slider ──────────────────────────────────────────

  private buildExpSliderHTML(trial: MagnitudeTrialParams): string {
    let html = '<div class="bead-magnitude-slider-wrapper">';

    // Value display
    html += '<div class="bead-magnitude-slider-value" id="bead-magnitude-slider-value">';
    html += trial.slider_start !== null ? formatValue(xToValue(trial.slider_start)) : "--";
    html += "</div>";

    // Track area with endpoints
    html += '<div class="bead-magnitude-slider-track-area">';
    html += '<span class="bead-magnitude-slider-endpoint bead-magnitude-slider-left">0</span>';

    // The track
    html += '<div class="bead-magnitude-slider-track" id="bead-magnitude-slider-track"';
    html += ' tabindex="0" role="slider" aria-valuemin="0"';
    html += ` aria-valuemax="${Math.round(xToValue(computeXMax(trial.reference_value)))}"`;
    html += ' aria-valuenow="0" aria-label="Magnitude estimation slider">';

    // Filled portion
    const startPct =
      trial.slider_start !== null
        ? (trial.slider_start / computeXMax(trial.reference_value)) * 100
        : 0;
    html += `<div class="bead-magnitude-slider-fill" id="bead-magnitude-slider-fill" style="width:${startPct}%"></div>`;

    // Reference tick at 1/3
    html += '<div class="bead-magnitude-slider-ref-tick" style="left:33.33%">';
    html += `<span class="bead-magnitude-slider-ref-label">${trial.reference_value}</span>`;
    html += "</div>";

    // Handle
    const handleClass =
      trial.slider_start !== null
        ? "bead-magnitude-slider-handle"
        : "bead-magnitude-slider-handle hidden";
    html += `<div class="${handleClass}" id="bead-magnitude-slider-handle" style="left:${startPct}%"></div>`;

    html += "</div>"; // track
    html +=
      '<span class="bead-magnitude-slider-endpoint bead-magnitude-slider-right">\u221E</span>';
    html += "</div>"; // track-area
    html += "</div>"; // wrapper
    return html;
  }

  private setupExpSlider(
    display_element: HTMLElement,
    trial: MagnitudeTrialParams,
    start_time: number,
    hasReference: boolean,
  ): void {
    const xMax = computeXMax(trial.reference_value);
    let currentX = trial.slider_start ?? -1;
    let hasInteracted = currentX >= 0;

    const track = display_element.querySelector<HTMLDivElement>("#bead-magnitude-slider-track");
    const handle = display_element.querySelector<HTMLDivElement>("#bead-magnitude-slider-handle");
    const fill = display_element.querySelector<HTMLDivElement>("#bead-magnitude-slider-fill");
    const valueDisplay = display_element.querySelector<HTMLDivElement>(
      "#bead-magnitude-slider-value",
    );
    const continueBtn = display_element.querySelector<HTMLButtonElement>(
      "#bead-magnitude-continue",
    );

    if (!track || !handle || !fill || !valueDisplay) return;

    const updateUI = (): void => {
      if (currentX < 0) return;
      const pct = (currentX / xMax) * 100;

      handle.style.left = `${pct}%`;
      fill.style.width = `${pct}%`;

      const value = xToValue(currentX);
      let displayText = formatValue(value);
      if (trial.unit) {
        displayText += ` ${trial.unit}`;
      }
      valueDisplay.textContent = displayText;

      track.setAttribute("aria-valuenow", String(Math.round(value)));

      if (continueBtn && trial.require_response) {
        continueBtn.disabled = false;
      }
    };

    const setPosition = (x: number): void => {
      currentX = Math.max(0, Math.min(xMax, x));
      if (!hasInteracted) {
        hasInteracted = true;
        handle.classList.remove("hidden");
      }
      updateUI();
    };

    // Render initial position if slider_start was set
    if (hasInteracted) {
      updateUI();
    }

    // ── Mouse events ──

    const onMouseDown = (e: MouseEvent): void => {
      e.preventDefault();
      const rect = track.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const x = (px / rect.width) * xMax;
      setPosition(x);
      track.focus();

      const onMouseMove = (ev: MouseEvent): void => {
        const movePx = ev.clientX - rect.left;
        setPosition((movePx / rect.width) * xMax);
      };

      const onMouseUp = (): void => {
        document.removeEventListener("mousemove", onMouseMove);
        document.removeEventListener("mouseup", onMouseUp);
      };

      document.addEventListener("mousemove", onMouseMove);
      document.addEventListener("mouseup", onMouseUp);
    };

    track.addEventListener("mousedown", onMouseDown);

    // ── Touch events ──

    const onTouchStart = (e: TouchEvent): void => {
      e.preventDefault();
      const rect = track.getBoundingClientRect();
      const touch = e.touches[0];
      if (!touch) return;
      const px = touch.clientX - rect.left;
      setPosition((px / rect.width) * xMax);
      track.focus();

      const onTouchMove = (ev: TouchEvent): void => {
        const t = ev.touches[0];
        if (!t) return;
        const movePx = t.clientX - rect.left;
        setPosition((movePx / rect.width) * xMax);
      };

      const onTouchEnd = (): void => {
        document.removeEventListener("touchmove", onTouchMove);
        document.removeEventListener("touchend", onTouchEnd);
      };

      document.addEventListener("touchmove", onTouchMove, { passive: false });
      document.addEventListener("touchend", onTouchEnd);
    };

    track.addEventListener("touchstart", onTouchStart, { passive: false });

    // ── Keyboard events ──

    track.addEventListener("keydown", (e: KeyboardEvent) => {
      if (e.key === "ArrowRight" || e.key === "ArrowUp") {
        e.preventDefault();
        if (!hasInteracted) {
          setPosition(xMax / 3);
        } else {
          setPosition(currentX + trial.arrow_step);
        }
      } else if (e.key === "ArrowLeft" || e.key === "ArrowDown") {
        e.preventDefault();
        if (!hasInteracted) {
          setPosition(xMax / 3);
        } else {
          setPosition(currentX - trial.arrow_step);
        }
      } else if (e.key === "Home") {
        e.preventDefault();
        setPosition(0);
      } else if (e.key === "End") {
        e.preventDefault();
        setPosition(xMax);
      }
    });

    // Focus the track for keyboard interaction
    track.focus();

    // ── Continue button ──

    if (continueBtn) {
      continueBtn.addEventListener("click", () => {
        if (!trial.require_response || hasInteracted) {
          end_trial();
        }
      });
    }

    const end_trial = (): void => {
      const rt = performance.now() - start_time;
      const value = hasInteracted ? xToValue(currentX) : null;

      const trial_data: Record<string, unknown> = {
        ...trial.metadata,
        response: value !== null && Number.isFinite(value) ? Math.round(value * 1000) / 1000 : null,
        response_x: hasInteracted ? Math.round(currentX * 100) / 100 : null,
        rt: rt,
      };

      if (hasReference) {
        trial_data["reference_value"] = trial.reference_value;
      }

      display_element.innerHTML = "";
      this.jsPsych.finishTrial(trial_data);
    };
  }
}

export { BeadMagnitudePlugin };
export { computeXMax, xToValue, valueToX, formatValue };
