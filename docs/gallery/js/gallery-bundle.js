(function () {
  'use strict';

  /* @bead/jspsych-gallery - Interactive demo bundle */
  var __defProp = Object.defineProperty;
  var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
  var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);

  // src/plugins/binary-choice.ts
  var info = {
    name: "bead-binary-choice",
    parameters: {
      prompt: {
        type: 8,
        // ParameterType.HTML_STRING
        default: "Is this sentence acceptable?"
      },
      stimulus: {
        type: 8,
        // ParameterType.HTML_STRING
        default: ""
      },
      choices: {
        type: 1,
        // ParameterType.STRING
        default: ["Yes", "No"],
        array: true
      },
      prompt_position: {
        type: 1,
        // ParameterType.STRING
        default: "above"
      },
      require_response: {
        type: 0,
        // ParameterType.BOOL
        default: true
      },
      metadata: {
        type: 12,
        // ParameterType.OBJECT
        default: {}
      }
    }
  };
  var BeadBinaryChoicePlugin = class {
    constructor(jsPsych) {
      __publicField(this, "jsPsych");
      this.jsPsych = jsPsych;
    }
    trial(display_element, trial) {
      let response_index = null;
      let rt = null;
      const start_time = performance.now();
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
      const buttons = display_element.querySelectorAll(".bead-binary-button");
      for (const button of buttons) {
        button.addEventListener("click", (e) => {
          const target = e.target;
          const indexAttr = target.getAttribute("data-index");
          if (indexAttr !== null) {
            select_choice(Number.parseInt(indexAttr, 10));
          }
        });
      }
      const select_choice = (index) => {
        response_index = index;
        rt = performance.now() - start_time;
        for (const btn of buttons) {
          btn.classList.remove("selected");
        }
        const selected = display_element.querySelector(
          `.bead-binary-button[data-index="${index}"]`
        );
        if (selected) {
          selected.classList.add("selected");
        }
        setTimeout(() => {
          end_trial();
        }, 200);
      };
      const end_trial = () => {
        const trial_data = {
          ...trial.metadata,
          response: response_index,
          response_label: response_index !== null ? trial.choices[response_index] : null,
          rt
        };
        display_element.innerHTML = "";
        this.jsPsych.finishTrial(trial_data);
      };
    }
  };
  __publicField(BeadBinaryChoicePlugin, "info", info);

  // src/plugins/categorical.ts
  var info2 = {
    name: "bead-categorical",
    parameters: {
      prompt: {
        type: 8,
        // ParameterType.HTML_STRING
        default: "Select a category:"
      },
      stimulus: {
        type: 8,
        // ParameterType.HTML_STRING
        default: ""
      },
      categories: {
        type: 1,
        // ParameterType.STRING
        default: [],
        array: true
      },
      prompt_position: {
        type: 1,
        // ParameterType.STRING
        default: "above"
      },
      require_response: {
        type: 0,
        // ParameterType.BOOL
        default: true
      },
      button_label: {
        type: 1,
        // ParameterType.STRING
        default: "Continue"
      },
      metadata: {
        type: 12,
        // ParameterType.OBJECT
        default: {}
      }
    }
  };
  var BeadCategoricalPlugin = class {
    constructor(jsPsych) {
      __publicField(this, "jsPsych");
      this.jsPsych = jsPsych;
    }
    trial(display_element, trial) {
      let selected_index = null;
      const start_time = performance.now();
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
      const buttons = display_element.querySelectorAll(".bead-categorical-button");
      const continueBtn = display_element.querySelector(
        "#bead-categorical-continue"
      );
      for (const button of buttons) {
        button.addEventListener("click", (e) => {
          const target = e.currentTarget;
          const indexAttr = target.getAttribute("data-index");
          if (indexAttr !== null) {
            selected_index = Number.parseInt(indexAttr, 10);
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
      const end_trial = () => {
        const rt = performance.now() - start_time;
        const trial_data = {
          ...trial.metadata,
          response: selected_index !== null ? trial.categories[selected_index] : null,
          response_index: selected_index,
          rt
        };
        display_element.innerHTML = "";
        this.jsPsych.finishTrial(trial_data);
      };
    }
  };
  __publicField(BeadCategoricalPlugin, "info", info2);

  // src/plugins/cloze-dropdown.ts
  var info3 = {
    name: "bead-cloze-multi",
    parameters: {
      text: {
        type: 8,
        // ParameterType.HTML_STRING
        default: null
      },
      fields: {
        type: 13,
        // ParameterType.COMPLEX
        default: [],
        array: true
      },
      require_all: {
        type: 0,
        // ParameterType.BOOL
        default: true
      },
      button_label: {
        type: 1,
        // ParameterType.STRING
        default: "Continue"
      },
      metadata: {
        type: 12,
        // ParameterType.OBJECT
        default: {}
      }
    }
  };
  var BeadClozeMultiPlugin = class {
    constructor(jsPsych) {
      __publicField(this, "jsPsych");
      this.jsPsych = jsPsych;
    }
    trial(display_element, trial) {
      const responses = {};
      const response_times = {};
      const field_start_times = {};
      const start_time = performance.now();
      if (trial.fields.length === 0 && trial.metadata.unfilled_slots) {
        trial.fields = trial.metadata.unfilled_slots.map((slot) => ({
          slot_name: slot.slot_name,
          position: slot.position,
          type: slot.constraint_ids.length > 0 ? "dropdown" : "text",
          options: [],
          // Would be populated from constraints in real implementation
          placeholder: slot.slot_name
        }));
      }
      let html = '<div class="bead-cloze-container">';
      if (trial.text) {
        let processed_text = trial.text;
        trial.fields.forEach((field, index) => {
          const field_id = `bead-cloze-field-${index}`;
          let field_html;
          if (field.type === "dropdown" && field.options && field.options.length > 0) {
            const optionsHtml = field.options.map((opt) => `<option value="${opt}">${opt}</option>`).join("");
            field_html = `
            <select class="bead-dropdown bead-cloze-field" id="${field_id}" data-field="${index}">
              <option value="">Select...</option>
              ${optionsHtml}
            </select>
          `;
          } else {
            field_html = `
            <input type="text"
                   class="bead-text-field bead-cloze-field"
                   id="${field_id}"
                   data-field="${index}"
                   placeholder="${field.placeholder ?? ""}" />
          `;
          }
          const placeholder = field.slot_name ? `{{${field.slot_name}}}` : "%%";
          processed_text = processed_text.replace(placeholder, field_html);
        });
        html += `<div class="bead-cloze-text">${processed_text}</div>`;
      }
      html += `
      <div class="bead-cloze-button-container">
        <button class="bead-button bead-continue-button" id="bead-cloze-continue" ${trial.require_all ? "disabled" : ""}>
          ${trial.button_label}
        </button>
      </div>
    `;
      html += "</div>";
      display_element.innerHTML = html;
      const input_fields = display_element.querySelectorAll(
        ".bead-cloze-field"
      );
      for (const field of input_fields) {
        const field_index = field.getAttribute("data-field");
        if (field_index === null) continue;
        field.addEventListener("focus", () => {
          if (field_start_times[field_index] === void 0) {
            field_start_times[field_index] = performance.now();
          }
        });
        field.addEventListener("change", () => {
          responses[field_index] = field.value;
          const startTime = field_start_times[field_index];
          if (startTime !== void 0) {
            response_times[field_index] = performance.now() - startTime;
          }
          check_completion();
        });
        field.addEventListener("input", () => {
          responses[field_index] = field.value;
          check_completion();
        });
      }
      const continue_button = display_element.querySelector("#bead-cloze-continue");
      if (continue_button) {
        continue_button.addEventListener("click", () => {
          end_trial();
        });
      }
      const check_completion = () => {
        if (trial.require_all && continue_button) {
          const all_filled = trial.fields.every((_field, index) => {
            const response = responses[index.toString()];
            return response !== void 0 && response.trim() !== "";
          });
          continue_button.disabled = !all_filled;
        }
      };
      const end_trial = () => {
        const trial_data = {
          ...trial.metadata,
          // Preserve all metadata
          responses,
          response_times,
          total_rt: performance.now() - start_time
        };
        display_element.innerHTML = "";
        this.jsPsych.finishTrial(trial_data);
      };
    }
  };
  __publicField(BeadClozeMultiPlugin, "info", info3);

  // src/plugins/forced-choice.ts
  var info4 = {
    name: "bead-forced-choice",
    parameters: {
      prompt: {
        type: 8,
        // ParameterType.HTML_STRING
        default: "Which do you prefer?"
      },
      alternatives: {
        type: 1,
        // ParameterType.STRING
        default: [],
        array: true
      },
      layout: {
        type: 1,
        // ParameterType.STRING
        default: "horizontal"
      },
      randomize_position: {
        type: 0,
        // ParameterType.BOOL
        default: true
      },
      enable_keyboard: {
        type: 0,
        // ParameterType.BOOL
        default: true
      },
      require_response: {
        type: 0,
        // ParameterType.BOOL
        default: true
      },
      button_label: {
        type: 1,
        // ParameterType.STRING
        default: "Continue"
      },
      metadata: {
        type: 12,
        // ParameterType.OBJECT
        default: {}
      }
    }
  };
  var BeadForcedChoicePlugin = class {
    constructor(jsPsych) {
      __publicField(this, "jsPsych");
      this.jsPsych = jsPsych;
    }
    trial(display_element, trial) {
      const response = {
        choice: null,
        choice_index: null,
        position: null,
        rt: null
      };
      const start_time = performance.now();
      let left_index = 0;
      let right_index = 1;
      if (trial.randomize_position && Math.random() < 0.5) {
        left_index = 1;
        right_index = 0;
      }
      let html = '<div class="bead-forced-choice-container">';
      if (trial.prompt) {
        html += `<div class="bead-forced-choice-prompt">${trial.prompt}</div>`;
      }
      html += `<div class="bead-forced-choice-alternatives bead-layout-${trial.layout}">`;
      html += `
      <div class="bead-card bead-alternative" data-index="${left_index}" data-position="left">
        <div class="bead-alternative-label">Option 1</div>
        <div class="bead-alternative-content">${trial.alternatives[left_index] ?? "Alternative A"}</div>
        <button class="bead-button bead-choice-button" data-index="${left_index}" data-position="left">
          Select
        </button>
      </div>
    `;
      html += `
      <div class="bead-card bead-alternative" data-index="${right_index}" data-position="right">
        <div class="bead-alternative-label">Option 2</div>
        <div class="bead-alternative-content">${trial.alternatives[right_index] ?? "Alternative B"}</div>
        <button class="bead-button bead-choice-button" data-index="${right_index}" data-position="right">
          Select
        </button>
      </div>
    `;
      html += "</div>";
      html += "</div>";
      display_element.innerHTML = html;
      const alternative_cards = display_element.querySelectorAll(".bead-alternative");
      for (const card of alternative_cards) {
        card.addEventListener("click", () => {
          const indexAttr = card.getAttribute("data-index");
          const positionAttr = card.getAttribute("data-position");
          if (indexAttr !== null && positionAttr !== null) {
            const index = Number.parseInt(indexAttr, 10);
            select_choice(index, positionAttr);
          }
        });
      }
      let keyboard_listener = null;
      if (trial.enable_keyboard) {
        keyboard_listener = this.jsPsych.pluginAPI.getKeyboardResponse({
          callback_function: (info11) => {
            const key = info11.key;
            if (key === "1" || key === "ArrowLeft") {
              select_choice(left_index, "left");
            } else if (key === "2" || key === "ArrowRight") {
              select_choice(right_index, "right");
            }
          },
          valid_responses: ["1", "2", "ArrowLeft", "ArrowRight"],
          rt_method: "performance",
          persist: false,
          allow_held_key: false
        });
      }
      const select_choice = (index, position) => {
        response.choice = trial.alternatives[index] ?? null;
        response.choice_index = index;
        response.position = position;
        response.rt = performance.now() - start_time;
        const alternative_cards2 = display_element.querySelectorAll(".bead-alternative");
        for (const card of alternative_cards2) {
          card.classList.remove("selected");
        }
        const selected_card = display_element.querySelector(
          `.bead-alternative[data-position="${position}"]`
        );
        if (selected_card) {
          selected_card.classList.add("selected");
        }
        setTimeout(() => {
          end_trial();
        }, 300);
      };
      const end_trial = () => {
        if (keyboard_listener) {
          this.jsPsych.pluginAPI.cancelKeyboardResponse(keyboard_listener);
        }
        const trial_data = {
          ...trial.metadata,
          // Spread all metadata
          choice: response.choice,
          choice_index: response.choice_index,
          position_chosen: response.position,
          left_index,
          right_index,
          rt: response.rt
        };
        display_element.innerHTML = "";
        this.jsPsych.finishTrial(trial_data);
      };
    }
  };
  __publicField(BeadForcedChoicePlugin, "info", info4);

  // src/plugins/free-text.ts
  var info5 = {
    name: "bead-free-text",
    parameters: {
      prompt: {
        type: 8,
        // ParameterType.HTML_STRING
        default: "Enter your response:"
      },
      stimulus: {
        type: 8,
        // ParameterType.HTML_STRING
        default: ""
      },
      prompt_position: {
        type: 1,
        // ParameterType.STRING
        default: "above"
      },
      multiline: {
        type: 0,
        // ParameterType.BOOL
        default: false
      },
      min_length: {
        type: 2,
        // ParameterType.INT
        default: 0
      },
      max_length: {
        type: 2,
        // ParameterType.INT
        default: 0
      },
      placeholder: {
        type: 1,
        // ParameterType.STRING
        default: ""
      },
      rows: {
        type: 2,
        // ParameterType.INT
        default: 4
      },
      require_response: {
        type: 0,
        // ParameterType.BOOL
        default: true
      },
      button_label: {
        type: 1,
        // ParameterType.STRING
        default: "Continue"
      },
      metadata: {
        type: 12,
        // ParameterType.OBJECT
        default: {}
      }
    }
  };
  var BeadFreeTextPlugin = class {
    constructor(jsPsych) {
      __publicField(this, "jsPsych");
      this.jsPsych = jsPsych;
    }
    trial(display_element, trial) {
      const start_time = performance.now();
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
      const input = display_element.querySelector(
        "#bead-free-text-input"
      );
      const continueBtn = display_element.querySelector(
        "#bead-free-text-continue"
      );
      const charCount = display_element.querySelector("#bead-char-count");
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
      const end_trial = () => {
        const rt = performance.now() - start_time;
        const trial_data = {
          ...trial.metadata,
          response: input ? input.value : "",
          rt
        };
        display_element.innerHTML = "";
        this.jsPsych.finishTrial(trial_data);
      };
    }
  };
  __publicField(BeadFreeTextPlugin, "info", info5);

  // src/plugins/magnitude.ts
  function computeXMax(referenceValue) {
    return 3 * 100 * Math.log(referenceValue + 1);
  }
  function xToValue(x) {
    if (x <= 0) return 0;
    return Math.exp(x / 100) - 1;
  }
  function formatValue(value) {
    if (value >= 1e6) return "\u221E";
    if (value >= 1e4) return Math.round(value).toLocaleString();
    if (value >= 100) return Math.round(value).toString();
    if (value >= 10) return value.toFixed(1);
    if (value >= 1) return value.toFixed(2);
    if (value > 0) return value.toFixed(3);
    return "0";
  }
  var info6 = {
    name: "bead-magnitude",
    parameters: {
      prompt: {
        type: 8,
        // ParameterType.HTML_STRING
        default: "Enter a value:"
      },
      stimulus: {
        type: 8,
        // ParameterType.HTML_STRING
        default: ""
      },
      prompt_position: {
        type: 1,
        // ParameterType.STRING
        default: "above"
      },
      reference_stimulus: {
        type: 8,
        // ParameterType.HTML_STRING
        default: ""
      },
      reference_value: {
        type: 2,
        // ParameterType.INT
        default: 100
      },
      unit: {
        type: 1,
        // ParameterType.STRING
        default: ""
      },
      input_mode: {
        type: 1,
        // ParameterType.STRING
        default: "number"
      },
      arrow_step: {
        type: 3,
        // ParameterType.FLOAT
        default: 3
      },
      slider_start: {
        type: 3,
        // ParameterType.FLOAT
        default: null
      },
      input_min: {
        type: 3,
        // ParameterType.FLOAT
        default: null
      },
      input_max: {
        type: 3,
        // ParameterType.FLOAT
        default: null
      },
      step: {
        type: 3,
        // ParameterType.FLOAT
        default: null
      },
      placeholder: {
        type: 1,
        // ParameterType.STRING
        default: ""
      },
      require_response: {
        type: 0,
        // ParameterType.BOOL
        default: true
      },
      button_label: {
        type: 1,
        // ParameterType.STRING
        default: "Continue"
      },
      metadata: {
        type: 12,
        // ParameterType.OBJECT
        default: {}
      }
    }
  };
  var BeadMagnitudePlugin = class {
    constructor(jsPsych) {
      __publicField(this, "jsPsych");
      this.jsPsych = jsPsych;
    }
    trial(display_element, trial) {
      const start_time = performance.now();
      const hasReference = trial.reference_stimulus !== "";
      let html = '<div class="bead-magnitude-container">';
      if (trial.prompt && trial.prompt_position === "above") {
        html += `<div class="bead-magnitude-prompt">${trial.prompt}</div>`;
      }
      if (hasReference) {
        html += '<div class="bead-magnitude-reference-header">';
        html += '<div class="bead-magnitude-section-label">Reference</div>';
        html += `<div class="bead-magnitude-reference-chip">${trial.reference_value}</div>`;
        html += "</div>";
        html += '<div class="bead-magnitude-reference">';
        html += `<div class="bead-magnitude-reference-text">${trial.reference_stimulus}</div>`;
        html += "</div>";
      }
      if (trial.stimulus) {
        if (hasReference) {
          html += '<div class="bead-magnitude-section-label">Target</div>';
        }
        html += `<div class="bead-magnitude-stimulus">${trial.stimulus}</div>`;
      }
      if (trial.prompt && trial.prompt_position === "below") {
        html += `<div class="bead-magnitude-prompt">${trial.prompt}</div>`;
      }
      if (trial.input_mode === "exp-slider") {
        html += this.buildExpSliderHTML(trial);
      } else {
        html += this.buildNumberInputHTML(trial);
      }
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
      if (trial.input_mode === "exp-slider") {
        this.setupExpSlider(display_element, trial, start_time, hasReference);
      } else {
        this.setupNumberInput(display_element, trial, start_time, hasReference);
      }
    }
    // ── Number input (existing behavior) ────────────────────────────
    buildNumberInputHTML(trial) {
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
    setupNumberInput(display_element, trial, start_time, hasReference) {
      const input = display_element.querySelector("#bead-magnitude-input");
      const continueBtn = display_element.querySelector(
        "#bead-magnitude-continue"
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
          if (!trial.require_response || input && input.value.trim() !== "") {
            end_trial();
          }
        });
      }
      const end_trial = () => {
        const rt = performance.now() - start_time;
        const value = input ? Number.parseFloat(input.value) : null;
        const trial_data = {
          ...trial.metadata,
          response: Number.isNaN(value ?? Number.NaN) ? null : value,
          rt
        };
        if (hasReference) {
          trial_data["reference_value"] = trial.reference_value;
        }
        display_element.innerHTML = "";
        this.jsPsych.finishTrial(trial_data);
      };
    }
    // ── Exponential slider ──────────────────────────────────────────
    buildExpSliderHTML(trial) {
      let html = '<div class="bead-magnitude-slider-wrapper">';
      html += '<div class="bead-magnitude-slider-value" id="bead-magnitude-slider-value">';
      html += trial.slider_start !== null ? formatValue(xToValue(trial.slider_start)) : "--";
      html += "</div>";
      html += '<div class="bead-magnitude-slider-track-area">';
      html += '<span class="bead-magnitude-slider-endpoint bead-magnitude-slider-left">0</span>';
      html += '<div class="bead-magnitude-slider-track" id="bead-magnitude-slider-track"';
      html += ' tabindex="0" role="slider" aria-valuemin="0"';
      html += ` aria-valuemax="${Math.round(xToValue(computeXMax(trial.reference_value)))}"`;
      html += ' aria-valuenow="0" aria-label="Magnitude estimation slider">';
      const startPct = trial.slider_start !== null ? trial.slider_start / computeXMax(trial.reference_value) * 100 : 0;
      html += `<div class="bead-magnitude-slider-fill" id="bead-magnitude-slider-fill" style="width:${startPct}%"></div>`;
      html += '<div class="bead-magnitude-slider-ref-tick" style="left:33.33%">';
      html += `<span class="bead-magnitude-slider-ref-label">${trial.reference_value}</span>`;
      html += "</div>";
      const handleClass = trial.slider_start !== null ? "bead-magnitude-slider-handle" : "bead-magnitude-slider-handle hidden";
      html += `<div class="${handleClass}" id="bead-magnitude-slider-handle" style="left:${startPct}%"></div>`;
      html += "</div>";
      html += '<span class="bead-magnitude-slider-endpoint bead-magnitude-slider-right">\u221E</span>';
      html += "</div>";
      html += "</div>";
      return html;
    }
    setupExpSlider(display_element, trial, start_time, hasReference) {
      const xMax = computeXMax(trial.reference_value);
      let currentX = trial.slider_start ?? -1;
      let hasInteracted = currentX >= 0;
      const track = display_element.querySelector("#bead-magnitude-slider-track");
      const handle = display_element.querySelector("#bead-magnitude-slider-handle");
      const fill = display_element.querySelector("#bead-magnitude-slider-fill");
      const valueDisplay = display_element.querySelector(
        "#bead-magnitude-slider-value"
      );
      const continueBtn = display_element.querySelector(
        "#bead-magnitude-continue"
      );
      if (!track || !handle || !fill || !valueDisplay) return;
      const updateUI = () => {
        if (currentX < 0) return;
        const pct = currentX / xMax * 100;
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
      const setPosition = (x) => {
        currentX = Math.max(0, Math.min(xMax, x));
        if (!hasInteracted) {
          hasInteracted = true;
          handle.classList.remove("hidden");
        }
        updateUI();
      };
      if (hasInteracted) {
        updateUI();
      }
      const onMouseDown = (e) => {
        e.preventDefault();
        const rect = track.getBoundingClientRect();
        const px = e.clientX - rect.left;
        const x = px / rect.width * xMax;
        setPosition(x);
        track.focus();
        const onMouseMove = (ev) => {
          const movePx = ev.clientX - rect.left;
          setPosition(movePx / rect.width * xMax);
        };
        const onMouseUp = () => {
          document.removeEventListener("mousemove", onMouseMove);
          document.removeEventListener("mouseup", onMouseUp);
        };
        document.addEventListener("mousemove", onMouseMove);
        document.addEventListener("mouseup", onMouseUp);
      };
      track.addEventListener("mousedown", onMouseDown);
      const onTouchStart = (e) => {
        e.preventDefault();
        const rect = track.getBoundingClientRect();
        const touch = e.touches[0];
        if (!touch) return;
        const px = touch.clientX - rect.left;
        setPosition(px / rect.width * xMax);
        track.focus();
        const onTouchMove = (ev) => {
          const t = ev.touches[0];
          if (!t) return;
          const movePx = t.clientX - rect.left;
          setPosition(movePx / rect.width * xMax);
        };
        const onTouchEnd = () => {
          document.removeEventListener("touchmove", onTouchMove);
          document.removeEventListener("touchend", onTouchEnd);
        };
        document.addEventListener("touchmove", onTouchMove, { passive: false });
        document.addEventListener("touchend", onTouchEnd);
      };
      track.addEventListener("touchstart", onTouchStart, { passive: false });
      track.addEventListener("keydown", (e) => {
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
      track.focus();
      if (continueBtn) {
        continueBtn.addEventListener("click", () => {
          if (!trial.require_response || hasInteracted) {
            end_trial();
          }
        });
      }
      const end_trial = () => {
        const rt = performance.now() - start_time;
        const value = hasInteracted ? xToValue(currentX) : null;
        const trial_data = {
          ...trial.metadata,
          response: value !== null && Number.isFinite(value) ? Math.round(value * 1e3) / 1e3 : null,
          response_x: hasInteracted ? Math.round(currentX * 100) / 100 : null,
          rt
        };
        if (hasReference) {
          trial_data["reference_value"] = trial.reference_value;
        }
        display_element.innerHTML = "";
        this.jsPsych.finishTrial(trial_data);
      };
    }
  };
  __publicField(BeadMagnitudePlugin, "info", info6);

  // src/plugins/multi-select.ts
  var info7 = {
    name: "bead-multi-select",
    parameters: {
      prompt: {
        type: 8,
        // ParameterType.HTML_STRING
        default: "Select all that apply:"
      },
      stimulus: {
        type: 8,
        // ParameterType.HTML_STRING
        default: ""
      },
      prompt_position: {
        type: 1,
        // ParameterType.STRING
        default: "above"
      },
      options: {
        type: 1,
        // ParameterType.STRING
        default: [],
        array: true
      },
      min_selections: {
        type: 2,
        // ParameterType.INT
        default: 1
      },
      max_selections: {
        type: 2,
        // ParameterType.INT
        default: 0
      },
      require_response: {
        type: 0,
        // ParameterType.BOOL
        default: true
      },
      button_label: {
        type: 1,
        // ParameterType.STRING
        default: "Continue"
      },
      metadata: {
        type: 12,
        // ParameterType.OBJECT
        default: {}
      }
    }
  };
  var BeadMultiSelectPlugin = class {
    constructor(jsPsych) {
      __publicField(this, "jsPsych");
      this.jsPsych = jsPsych;
    }
    trial(display_element, trial) {
      const start_time = performance.now();
      const maxLen = Math.max(...trial.options.map((o) => o.length));
      const useCompact = maxLen < 25 && trial.options.length <= 6;
      let html = '<div class="bead-multi-select-container">';
      if (trial.prompt && trial.prompt_position === "above") {
        html += `<div class="bead-multi-select-prompt">${trial.prompt}</div>`;
      }
      if (trial.stimulus) {
        html += `<div class="bead-multi-select-stimulus">${trial.stimulus}</div>`;
      }
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
      const checkboxes = display_element.querySelectorAll(
        ".bead-multi-select-checkbox"
      );
      const continueBtn = display_element.querySelector(
        "#bead-multi-select-continue"
      );
      const updateButton = () => {
        const checked = display_element.querySelectorAll(
          ".bead-multi-select-checkbox:checked"
        );
        const count = checked.length;
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
      const end_trial = () => {
        const rt = performance.now() - start_time;
        const checked = display_element.querySelectorAll(
          ".bead-multi-select-checkbox:checked"
        );
        const selected = [];
        const selected_indices = [];
        for (const cb of checked) {
          selected.push(cb.value);
          const idx = cb.getAttribute("data-index");
          if (idx !== null) selected_indices.push(Number.parseInt(idx, 10));
        }
        const trial_data = {
          ...trial.metadata,
          selected,
          selected_indices,
          rt
        };
        display_element.innerHTML = "";
        this.jsPsych.finishTrial(trial_data);
      };
    }
  };
  __publicField(BeadMultiSelectPlugin, "info", info7);

  // src/plugins/rating.ts
  var info8 = {
    name: "bead-rating",
    parameters: {
      prompt: {
        type: 8,
        // ParameterType.HTML_STRING
        default: null
      },
      stimulus: {
        type: 8,
        // ParameterType.HTML_STRING
        default: ""
      },
      prompt_position: {
        type: 1,
        // ParameterType.STRING
        default: "above"
      },
      scale_min: {
        type: 2,
        // ParameterType.INT
        default: 1
      },
      scale_max: {
        type: 2,
        // ParameterType.INT
        default: 7
      },
      scale_labels: {
        type: 12,
        // ParameterType.OBJECT
        default: {}
      },
      require_response: {
        type: 0,
        // ParameterType.BOOL
        default: true
      },
      button_label: {
        type: 1,
        // ParameterType.STRING
        default: "Continue"
      },
      metadata: {
        type: 12,
        // ParameterType.OBJECT
        default: {}
      }
    }
  };
  var BeadRatingPlugin = class {
    constructor(jsPsych) {
      __publicField(this, "jsPsych");
      this.jsPsych = jsPsych;
    }
    trial(display_element, trial) {
      const response = {
        rating: null,
        rt: null
      };
      const start_time = performance.now();
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
      for (let i = trial.scale_min; i <= trial.scale_max; i++) {
        const label = trial.scale_labels[i] ?? i;
        html += `
        <div class="bead-rating-option">
          <button class="bead-rating-button" data-value="${i}">${i}</button>
          <div class="bead-rating-label">${label}</div>
        </div>
      `;
      }
      html += "</div>";
      html += `
      <div class="bead-rating-button-container">
        <button class="bead-button bead-continue-button" id="bead-rating-continue" disabled>
          ${trial.button_label}
        </button>
      </div>
    `;
      html += "</div>";
      display_element.innerHTML = html;
      const rating_buttons = display_element.querySelectorAll(".bead-rating-button");
      for (const button of rating_buttons) {
        button.addEventListener("click", (e) => {
          const target = e.target;
          const valueAttr = target.getAttribute("data-value");
          if (valueAttr !== null) {
            const value = Number.parseInt(valueAttr, 10);
            select_rating(value);
          }
        });
      }
      const keyboard_listener = this.jsPsych.pluginAPI.getKeyboardResponse({
        callback_function: (info11) => {
          const key = info11.key;
          const num = Number.parseInt(key, 10);
          if (!Number.isNaN(num) && num >= trial.scale_min && num <= trial.scale_max) {
            select_rating(num);
          }
        },
        valid_responses: "ALL_KEYS",
        rt_method: "performance",
        persist: true,
        allow_held_key: false
      });
      const continue_button = display_element.querySelector("#bead-rating-continue");
      if (continue_button) {
        continue_button.addEventListener("click", () => {
          if (response.rating !== null || !trial.require_response) {
            end_trial();
          }
        });
      }
      const select_rating = (value) => {
        response.rating = value;
        response.rt = performance.now() - start_time;
        for (const btn of rating_buttons) {
          btn.classList.remove("selected");
        }
        const selected_button = display_element.querySelector(
          `[data-value="${value}"]`
        );
        if (selected_button) {
          selected_button.classList.add("selected");
        }
        if (continue_button) {
          continue_button.disabled = false;
        }
      };
      const end_trial = () => {
        if (keyboard_listener) {
          this.jsPsych.pluginAPI.cancelKeyboardResponse(keyboard_listener);
        }
        const trial_data = {
          ...trial.metadata,
          // Spread all metadata
          rating: response.rating,
          rt: response.rt
        };
        display_element.innerHTML = "";
        this.jsPsych.finishTrial(trial_data);
      };
    }
  };
  __publicField(BeadRatingPlugin, "info", info8);

  // src/plugins/slider-rating.ts
  var info9 = {
    name: "bead-slider-rating",
    parameters: {
      prompt: {
        type: 8,
        // ParameterType.HTML_STRING
        default: null
      },
      stimulus: {
        type: 8,
        // ParameterType.HTML_STRING
        default: ""
      },
      prompt_position: {
        type: 1,
        // ParameterType.STRING
        default: "above"
      },
      slider_min: {
        type: 2,
        // ParameterType.INT
        default: 0
      },
      slider_max: {
        type: 2,
        // ParameterType.INT
        default: 100
      },
      step: {
        type: 2,
        // ParameterType.INT
        default: 1
      },
      slider_start: {
        type: 2,
        // ParameterType.INT
        default: 50
      },
      labels: {
        type: 1,
        // ParameterType.STRING
        default: [],
        array: true
      },
      require_movement: {
        type: 0,
        // ParameterType.BOOL
        default: true
      },
      button_label: {
        type: 1,
        // ParameterType.STRING
        default: "Continue"
      },
      metadata: {
        type: 12,
        // ParameterType.OBJECT
        default: {}
      }
    }
  };
  var BeadSliderRatingPlugin = class {
    constructor(jsPsych) {
      __publicField(this, "jsPsych");
      this.jsPsych = jsPsych;
    }
    trial(display_element, trial) {
      let slider_value = trial.slider_start;
      let has_moved = false;
      const start_time = performance.now();
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
      if (trial.labels.length > 0) {
        html += '<div class="bead-slider-labels">';
        for (const label of trial.labels) {
          html += `<span class="bead-slider-label">${label}</span>`;
        }
        html += "</div>";
      }
      html += `<input type="range" class="bead-slider-input" min="${trial.slider_min}" max="${trial.slider_max}" step="${trial.step}" value="${trial.slider_start}">`;
      html += `<div class="bead-slider-value">${trial.slider_start}</div>`;
      html += "</div>";
      const disabled = trial.require_movement ? "disabled" : "";
      html += `
      <div class="bead-slider-button-container">
        <button class="bead-button bead-continue-button" id="bead-slider-continue" ${disabled}>
          ${trial.button_label}
        </button>
      </div>
    `;
      html += "</div>";
      display_element.innerHTML = html;
      const slider = display_element.querySelector(".bead-slider-input");
      const value_display = display_element.querySelector(".bead-slider-value");
      const continue_button = display_element.querySelector("#bead-slider-continue");
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
      if (continue_button) {
        continue_button.addEventListener("click", () => {
          if (!trial.require_movement || has_moved) {
            end_trial();
          }
        });
      }
      const end_trial = () => {
        const rt = performance.now() - start_time;
        const trial_data = {
          ...trial.metadata,
          response: slider_value,
          rt
        };
        display_element.innerHTML = "";
        this.jsPsych.finishTrial(trial_data);
      };
    }
  };
  __publicField(BeadSliderRatingPlugin, "info", info9);

  // src/lib/wikidata-search.ts
  var WIKIDATA_API = "https://www.wikidata.org/w/api.php";
  var CACHE_SIZE = 100;
  var DEBOUNCE_MS = 300;
  var cache = /* @__PURE__ */ new Map();
  function cacheKey(query, opts) {
    return `${opts.language}:${query}:${opts.limit}:${(opts.entityTypes ?? []).join(",")}`;
  }
  function putCache(key, value) {
    if (cache.size >= CACHE_SIZE) {
      const firstKey = cache.keys().next().value;
      if (firstKey !== void 0) {
        cache.delete(firstKey);
      }
    }
    cache.set(key, value);
  }
  async function searchWikidata(query, options) {
    if (!query || query.trim().length === 0) {
      return [];
    }
    const key = cacheKey(query, options);
    const cached = cache.get(key);
    if (cached) {
      return cached;
    }
    const params = new URLSearchParams({
      action: "wbsearchentities",
      search: query.trim(),
      language: options.language,
      limit: String(options.limit),
      format: "json",
      origin: "*"
    });
    if (options.entityTypes && options.entityTypes.length > 0) {
      params.set("type", options.entityTypes[0] ?? "item");
    }
    const url = `${WIKIDATA_API}?${params.toString()}`;
    try {
      const response = await fetch(url);
      if (!response.ok) {
        return [];
      }
      const data = await response.json();
      const results = (data.search ?? []).map((item) => ({
        id: String(item["id"] ?? ""),
        label: String(item["label"] ?? ""),
        description: String(item["description"] ?? ""),
        aliases: Array.isArray(item["aliases"]) ? item["aliases"].map(String) : []
      }));
      putCache(key, results);
      return results;
    } catch {
      return [];
    }
  }
  var debounceTimer = null;
  function debouncedSearchWikidata(query, options, callback) {
    if (debounceTimer !== null) {
      clearTimeout(debounceTimer);
    }
    debounceTimer = setTimeout(async () => {
      const results = await searchWikidata(query, options);
      callback(results);
    }, DEBOUNCE_MS);
  }

  // src/plugins/span-label.ts
  var info10 = {
    name: "bead-span-label",
    parameters: {
      tokens: {
        type: 12,
        // OBJECT
        default: {}
      },
      space_after: {
        type: 12,
        // OBJECT
        default: {}
      },
      spans: {
        type: 12,
        // OBJECT
        default: [],
        array: true
      },
      relations: {
        type: 12,
        // OBJECT
        default: [],
        array: true
      },
      span_spec: {
        type: 12,
        // OBJECT
        default: null
      },
      display_config: {
        type: 12,
        // OBJECT
        default: null
      },
      prompt: {
        type: 8,
        // HTML_STRING
        default: "Select and label spans"
      },
      button_label: {
        type: 1,
        // STRING
        default: "Continue"
      },
      require_response: {
        type: 0,
        // BOOL
        default: true
      },
      metadata: {
        type: 12,
        // OBJECT
        default: {}
      }
    }
  };
  var DEFAULT_PALETTE = [
    "#BBDEFB",
    "#C8E6C9",
    "#FFE0B2",
    "#F8BBD0",
    "#D1C4E9",
    "#B2EBF2",
    "#DCEDC8",
    "#FFD54F"
  ];
  var DARK_PALETTE = [
    "#1565C0",
    "#2E7D32",
    "#E65100",
    "#AD1457",
    "#4527A0",
    "#00838F",
    "#558B2F",
    "#F9A825"
  ];
  var BeadSpanLabelPlugin = class {
    constructor(jsPsych) {
      __publicField(this, "jsPsych");
      this.jsPsych = jsPsych;
    }
    trial(display_element, trial) {
      const start_time = performance.now();
      const events = [];
      const tokens = Object.keys(trial.tokens).length > 0 ? trial.tokens : trial.metadata.tokenized_elements ?? {};
      const spaceAfter = Object.keys(trial.space_after).length > 0 ? trial.space_after : trial.metadata.token_space_after ?? {};
      const spanSpec = trial.span_spec ?? trial.metadata.span_spec ?? null;
      const preSpans = trial.spans.length > 0 ? trial.spans : trial.metadata.spans ?? [];
      const preRelations = trial.relations.length > 0 ? trial.relations : trial.metadata.span_relations ?? [];
      const palette = trial.display_config?.color_palette ?? DEFAULT_PALETTE;
      const isInteractive = spanSpec?.interaction_mode === "interactive";
      const activeSpans = [...preSpans];
      const activeRelations = [...preRelations];
      let selectionStart = null;
      let selectedIndices = [];
      let nextSpanId = activeSpans.length;
      let nextRelationId = activeRelations.length;
      let relationState = "IDLE";
      let relationSource = null;
      let relationTarget = null;
      let html = '<div class="bead-span-label-container">';
      if (trial.prompt) {
        html += `<div class="bead-rating-prompt">${trial.prompt}</div>`;
      }
      const elementNames = Object.keys(tokens).sort();
      for (const elemName of elementNames) {
        const elemTokens = tokens[elemName] ?? [];
        const elemSpaceAfter = spaceAfter[elemName] ?? [];
        html += `<div class="bead-span-container" data-element="${elemName}">`;
        for (let i = 0; i < elemTokens.length; i++) {
          const tokenText = elemTokens[i];
          const interactive = isInteractive ? " interactive" : "";
          html += `<span class="bead-token${interactive}" data-index="${i}" data-element="${elemName}">${tokenText}</span>`;
          if (i < elemSpaceAfter.length && elemSpaceAfter[i]) {
            html += `<span class="bead-space" data-after="${i}" data-element="${elemName}"> </span>`;
          }
        }
        html += "</div>";
      }
      if (isInteractive && spanSpec?.label_source === "wikidata") {
        html += '<div class="bead-label-selector bead-wikidata-panel bead-search-disabled" id="bead-label-panel">';
        html += '<div class="bead-wikidata-search">';
        html += '<input type="text" id="bead-wikidata-input" placeholder="Select tokens to annotate..." autocomplete="off" disabled>';
        html += '<button type="button" class="bead-search-cancel" id="bead-search-cancel" style="display:none;" title="Cancel annotation">&times;</button>';
        html += '<div class="bead-wikidata-results" id="bead-wikidata-results" style="display:none;"></div>';
        html += "</div></div>";
      } else if (isInteractive && spanSpec?.labels && spanSpec.labels.length > 0) {
        html += '<div class="bead-label-selector bead-label-search-panel bead-search-disabled" id="bead-label-panel">';
        html += '<div class="bead-label-search-wrapper">';
        html += '<input type="text" id="bead-label-search-input" placeholder="Select tokens to annotate..." autocomplete="off" disabled>';
        html += '<button type="button" class="bead-search-cancel" id="bead-search-cancel" style="display:none;" title="Cancel annotation">&times;</button>';
        html += '<div class="bead-label-search-results" id="bead-label-search-results" style="display:none;"></div>';
        html += "</div></div>";
      }
      if (spanSpec?.enable_relations) {
        if (isInteractive) {
          html += '<div class="bead-relation-controls" id="bead-relation-controls">';
          html += '<button class="bead-add-relation-button" id="bead-add-relation" disabled>Add Relation</button>';
          html += '<span class="bead-relation-status" id="bead-relation-status"></span>';
          html += '<button class="bead-relation-cancel" id="bead-relation-cancel" style="display:none;">Cancel</button>';
          html += "</div>";
          if (spanSpec.relation_label_source === "wikidata") {
            html += '<div class="bead-label-selector bead-wikidata-panel" id="bead-relation-label-panel" style="display:none;">';
            html += '<div class="bead-wikidata-search">';
            html += '<input type="text" id="bead-relation-wikidata-input" placeholder="Search Wikidata for relation label..." autocomplete="off">';
            html += '<div class="bead-wikidata-results" id="bead-relation-wikidata-results" style="display:none;"></div>';
            html += "</div></div>";
          } else if (spanSpec.relation_labels && spanSpec.relation_labels.length > 0) {
            html += '<div class="bead-label-selector bead-label-search-panel" id="bead-relation-label-panel" style="display:none;">';
            html += '<div class="bead-label-search-wrapper">';
            html += '<input type="text" id="bead-relation-label-input" placeholder="Search relation labels..." autocomplete="off">';
            html += '<div class="bead-label-search-results" id="bead-relation-label-results" style="display:none;"></div>';
            html += "</div></div>";
          }
        }
        html += '<div class="bead-relation-list" id="bead-relation-list"></div>';
      }
      html += '<div class="bead-span-bottom-bar">';
      html += `<div class="bead-span-bottom-spacer"></div>`;
      html += `<button class="bead-button bead-continue-button" id="bead-span-continue" ${isInteractive && trial.require_response ? "disabled" : ""}>`;
      html += `${trial.button_label}`;
      html += "</button>";
      html += "</div>";
      html += "</div>";
      display_element.innerHTML = html;
      applySpanHighlights();
      renderSpanList();
      if (isInteractive) {
        setupInteractiveHandlers();
        if (spanSpec?.label_source === "wikidata") {
          setupWikidataSearch();
        } else if (spanSpec?.labels && spanSpec.labels.length > 0) {
          setupFixedLabelSearch();
        }
        const searchCancelBtn = display_element.querySelector("#bead-search-cancel");
        if (searchCancelBtn) {
          searchCancelBtn.addEventListener("click", () => {
            cancelCurrentSelection();
          });
        }
        if (spanSpec?.enable_relations) {
          setupRelationHandlers();
        }
      }
      renderRelationArcsOverlay();
      renderRelationList();
      const continueBtn = display_element.querySelector("#bead-span-continue");
      if (continueBtn) {
        continueBtn.addEventListener("click", () => {
          endTrial();
        });
      }
      function applySpanHighlights() {
        const allTokens = display_element.querySelectorAll(".bead-token");
        for (const t of allTokens) {
          t.classList.remove("highlighted", "span-first", "span-middle", "span-last", "span-single");
          t.removeAttribute("data-span-ids");
          t.removeAttribute("data-span-count");
          t.style.removeProperty("background-color");
          t.style.removeProperty("background");
        }
        const allSpaces = display_element.querySelectorAll(".bead-space");
        for (const s of allSpaces) {
          s.classList.remove("highlighted");
          s.style.removeProperty("background-color");
          s.style.removeProperty("background");
        }
        const tokenSpanMap = /* @__PURE__ */ new Map();
        for (const span of activeSpans) {
          for (const seg of span.segments) {
            for (const idx of seg.indices) {
              const key = `${seg.element_name}:${idx}`;
              if (!tokenSpanMap.has(key)) {
                tokenSpanMap.set(key, []);
              }
              tokenSpanMap.get(key)?.push(span.span_id);
            }
          }
        }
        const spanColorMap = assignColors();
        for (const t of allTokens) {
          const elemName = t.getAttribute("data-element") ?? "";
          const idx = t.getAttribute("data-index") ?? "";
          const key = `${elemName}:${idx}`;
          const spanIds = tokenSpanMap.get(key) ?? [];
          if (spanIds.length > 0) {
            t.classList.add("highlighted");
            t.setAttribute("data-span-ids", spanIds.join(","));
            t.setAttribute("data-span-count", String(spanIds.length));
            applySpanColor(t, spanIds, spanColorMap);
          }
        }
        for (const elemName of elementNames) {
          const elemTokens = tokens[elemName] ?? [];
          for (let i = 0; i < elemTokens.length; i++) {
            const key = `${elemName}:${i}`;
            const spanIds = tokenSpanMap.get(key) ?? [];
            if (spanIds.length === 0) continue;
            const t = display_element.querySelector(
              `.bead-token[data-element="${elemName}"][data-index="${i}"]`
            );
            if (!t) continue;
            const leftKey = `${elemName}:${i - 1}`;
            const leftSpanIds = tokenSpanMap.get(leftKey) ?? [];
            const hasLeftNeighbor = spanIds.some((id) => leftSpanIds.includes(id));
            const rightKey = `${elemName}:${i + 1}`;
            const rightSpanIds = tokenSpanMap.get(rightKey) ?? [];
            const hasRightNeighbor = spanIds.some((id) => rightSpanIds.includes(id));
            if (hasLeftNeighbor && hasRightNeighbor) {
              t.classList.add("span-middle");
            } else if (hasLeftNeighbor) {
              t.classList.add("span-last");
            } else if (hasRightNeighbor) {
              t.classList.add("span-first");
            } else {
              t.classList.add("span-single");
            }
            if (hasRightNeighbor) {
              const spaceEl = display_element.querySelector(
                `.bead-space[data-element="${elemName}"][data-after="${i}"]`
              );
              if (spaceEl) {
                spaceEl.classList.add("highlighted");
                const sharedIds = spanIds.filter((id) => rightSpanIds.includes(id));
                applySpanColor(spaceEl, sharedIds.length > 0 ? sharedIds : spanIds, spanColorMap);
              }
            }
          }
        }
      }
      function applySpanColor(el, spanIds, colorMap) {
        if (spanIds.length === 1) {
          el.style.backgroundColor = colorMap.get(spanIds[0] ?? "") ?? palette[0] ?? "#BBDEFB";
        } else if (spanIds.length > 1) {
          const colors = spanIds.map((id) => colorMap.get(id) ?? palette[0] ?? "#BBDEFB");
          const stripeWidth = 100 / colors.length;
          const stops = colors.map((c, ci) => `${c} ${ci * stripeWidth}%, ${c} ${(ci + 1) * stripeWidth}%`).join(", ");
          el.style.background = `linear-gradient(135deg, ${stops})`;
        }
      }
      function assignColors() {
        const colorMap = /* @__PURE__ */ new Map();
        const labelColors = spanSpec?.label_colors ?? {};
        const labelToColor = /* @__PURE__ */ new Map();
        let colorIdx = 0;
        for (const span of activeSpans) {
          const label = span.label?.label;
          if (label && labelColors[label]) {
            colorMap.set(span.span_id, labelColors[label] ?? "#BBDEFB");
          } else if (label && labelToColor.has(label)) {
            colorMap.set(span.span_id, labelToColor.get(label) ?? "#BBDEFB");
          } else {
            const color = palette[colorIdx % palette.length] ?? "#BBDEFB";
            colorMap.set(span.span_id, color);
            if (label) labelToColor.set(label, color);
            colorIdx++;
          }
        }
        return colorMap;
      }
      function renderSpanList() {
        const existing = display_element.querySelectorAll(".bead-span-subscript");
        for (const el of existing) el.remove();
        const darkColorMap = assignDarkColors();
        for (const span of activeSpans) {
          if (!span.label?.label) continue;
          const allIndices = [];
          for (const seg of span.segments) {
            for (const idx of seg.indices) {
              allIndices.push({ elem: seg.element_name, idx });
            }
          }
          if (allIndices.length === 0) continue;
          const lastToken = allIndices[allIndices.length - 1];
          if (!lastToken) continue;
          const tokenEl = display_element.querySelector(
            `.bead-token[data-element="${lastToken.elem}"][data-index="${lastToken.idx}"]`
          );
          if (!tokenEl) continue;
          tokenEl.style.position = "relative";
          const badge = document.createElement("span");
          badge.className = "bead-span-subscript";
          const darkColor = darkColorMap.get(span.span_id) ?? DARK_PALETTE[0] ?? "#1565C0";
          badge.style.backgroundColor = darkColor;
          badge.setAttribute("data-span-id", span.span_id);
          const labelSpan = document.createElement("span");
          labelSpan.textContent = span.label.label;
          badge.appendChild(labelSpan);
          if (isInteractive) {
            const deleteBtn = document.createElement("button");
            deleteBtn.className = "bead-subscript-delete";
            deleteBtn.textContent = "\xD7";
            deleteBtn.addEventListener("click", (e) => {
              e.stopPropagation();
              deleteSpan(span.span_id);
            });
            badge.appendChild(deleteBtn);
          }
          tokenEl.appendChild(badge);
        }
        adjustSubscriptPositions();
      }
      function adjustSubscriptPositions() {
        const badges = Array.from(
          display_element.querySelectorAll(".bead-span-subscript")
        );
        if (badges.length < 2) return;
        for (const b of badges) b.style.transform = "";
        badges.sort((a, b) => a.getBoundingClientRect().left - b.getBoundingClientRect().left);
        const placed = [];
        for (const badge of badges) {
          let rect = badge.getBoundingClientRect();
          let shift = 0;
          let hasOverlap = true;
          let iterations = 0;
          while (hasOverlap && iterations < 10) {
            hasOverlap = false;
            for (const p of placed) {
              const hOverlap = rect.left < p.rect.right + 3 && rect.right > p.rect.left - 3;
              const vOverlap = rect.top < p.rect.bottom + 1 && rect.bottom > p.rect.top - 1;
              if (hOverlap && vOverlap) {
                shift += p.rect.bottom - rect.top + 2;
                badge.style.transform = `translateY(${shift}px)`;
                rect = badge.getBoundingClientRect();
                hasOverlap = true;
                break;
              }
            }
            iterations++;
          }
          placed.push({ el: badge, rect: badge.getBoundingClientRect() });
        }
      }
      function assignDarkColors() {
        const colorMap = /* @__PURE__ */ new Map();
        let colorIdx = 0;
        const labelToColor = /* @__PURE__ */ new Map();
        for (const span of activeSpans) {
          const label = span.label?.label;
          if (label && labelToColor.has(label)) {
            colorMap.set(span.span_id, labelToColor.get(label) ?? DARK_PALETTE[0] ?? "#1565C0");
          } else {
            const color = DARK_PALETTE[colorIdx % DARK_PALETTE.length] ?? "#1565C0";
            colorMap.set(span.span_id, color);
            if (label) labelToColor.set(label, color);
            colorIdx++;
          }
        }
        return colorMap;
      }
      function getSpanText(span) {
        const parts = [];
        for (const seg of span.segments) {
          const elemTokens = tokens[seg.element_name] ?? [];
          for (const idx of seg.indices) {
            if (idx < elemTokens.length) {
              parts.push(elemTokens[idx] ?? "");
            }
          }
        }
        return parts.join(" ");
      }
      function setupInteractiveHandlers() {
        const tokenEls = display_element.querySelectorAll(".bead-token.interactive");
        let isDragging = false;
        let dragStartIdx = null;
        let dragElemName = null;
        for (const tokenEl of tokenEls) {
          tokenEl.addEventListener("mousedown", (e) => {
            e.preventDefault();
            const idx = Number.parseInt(tokenEl.getAttribute("data-index") ?? "0", 10);
            const elemName = tokenEl.getAttribute("data-element") ?? "";
            isDragging = true;
            dragStartIdx = idx;
            dragElemName = elemName;
            if (e.shiftKey && selectionStart !== null) {
              const start = Math.min(selectionStart, idx);
              const end = Math.max(selectionStart, idx);
              selectedIndices = [];
              for (let i = start; i <= end; i++) {
                selectedIndices.push(i);
              }
            } else {
              selectedIndices = [idx];
              selectionStart = idx;
            }
            updateSelectionUI(elemName);
            showLabelPanel();
          });
          tokenEl.addEventListener("mouseover", () => {
            if (!isDragging || dragStartIdx === null || dragElemName === null) return;
            const idx = Number.parseInt(tokenEl.getAttribute("data-index") ?? "0", 10);
            const elemName = tokenEl.getAttribute("data-element") ?? "";
            if (elemName !== dragElemName) return;
            const start = Math.min(dragStartIdx, idx);
            const end = Math.max(dragStartIdx, idx);
            selectedIndices = [];
            for (let i = start; i <= end; i++) {
              selectedIndices.push(i);
            }
            updateSelectionUI(elemName);
          });
        }
        document.addEventListener("mouseup", () => {
          if (isDragging) {
            isDragging = false;
            showLabelPanel();
          }
        });
        const labelButtons = display_element.querySelectorAll(".bead-label-button");
        for (const btn of labelButtons) {
          btn.addEventListener("click", () => {
            const label = btn.getAttribute("data-label") ?? "";
            if (selectedIndices.length > 0 && label) {
              createSpanFromSelection(label);
            }
          });
        }
        document.addEventListener("keydown", handleKeyDown);
      }
      function cancelCurrentSelection() {
        selectedIndices = [];
        selectionStart = null;
        const allTokens = display_element.querySelectorAll(".bead-token");
        for (const t of allTokens) {
          t.classList.remove("selecting");
        }
        const labelPanel = display_element.querySelector("#bead-label-panel");
        if (labelPanel) {
          labelPanel.classList.add("bead-search-disabled");
          const searchInput = labelPanel.querySelector("input");
          if (searchInput) {
            searchInput.disabled = true;
            searchInput.value = "";
            searchInput.placeholder = "Select tokens to annotate...";
          }
          const resultsDiv = labelPanel.querySelector(
            ".bead-label-search-results, .bead-wikidata-results"
          );
          if (resultsDiv) resultsDiv.style.display = "none";
          const cancelBtn = labelPanel.querySelector(".bead-search-cancel");
          if (cancelBtn) cancelBtn.style.display = "none";
        }
      }
      function showLabelPanel() {
        const labelPanel = display_element.querySelector("#bead-label-panel");
        if (!labelPanel) return;
        const hasSelection = selectedIndices.length > 0;
        if (hasSelection) {
          labelPanel.classList.remove("bead-search-disabled");
          const searchInput = labelPanel.querySelector("input");
          if (searchInput) {
            searchInput.disabled = false;
            searchInput.placeholder = "Search labels...";
            setTimeout(() => searchInput.focus(), 0);
          }
          const cancelBtn = labelPanel.querySelector(".bead-search-cancel");
          if (cancelBtn) cancelBtn.style.display = "";
        } else {
          cancelCurrentSelection();
        }
      }
      function handleKeyDown(e) {
        if (e.key === "Escape") {
          if (selectedIndices.length > 0) {
            cancelCurrentSelection();
            return;
          }
        }
        const num = Number.parseInt(e.key, 10);
        if (!Number.isNaN(num) && num >= 1 && num <= 9) {
          const labels = spanSpec?.labels ?? [];
          if (num <= labels.length && selectedIndices.length > 0) {
            createSpanFromSelection(labels[num - 1] ?? "");
          }
        }
      }
      function updateSelectionUI(elementName) {
        const tokenEls = display_element.querySelectorAll(
          `.bead-token[data-element="${elementName}"]`
        );
        for (const t of tokenEls) {
          const idx = Number.parseInt(t.getAttribute("data-index") ?? "0", 10);
          if (selectedIndices.includes(idx)) {
            t.classList.add("selecting");
          } else {
            t.classList.remove("selecting");
          }
        }
      }
      function createSpanFromSelection(label, labelId) {
        const elemName = elementNames[0] ?? "text";
        const spanId = `span_${nextSpanId++}`;
        const spanLabel = labelId ? { label, label_id: labelId } : { label };
        const newSpan = {
          span_id: spanId,
          segments: [
            {
              element_name: elemName,
              indices: [...selectedIndices].sort((a, b) => a - b)
            }
          ],
          label: spanLabel
        };
        activeSpans.push(newSpan);
        events.push({
          type: "select",
          timestamp: performance.now() - start_time,
          span_id: spanId,
          indices: [...selectedIndices],
          label
        });
        selectedIndices = [];
        selectionStart = null;
        applySpanHighlights();
        renderSpanList();
        renderRelationList();
        updateContinueButton();
        cancelCurrentSelection();
      }
      function deleteSpan(spanId) {
        const idx = activeSpans.findIndex((s) => s.span_id === spanId);
        if (idx >= 0) {
          activeSpans.splice(idx, 1);
          for (let ri = activeRelations.length - 1; ri >= 0; ri--) {
            const rel = activeRelations[ri];
            if (rel && (rel.source_span_id === spanId || rel.target_span_id === spanId)) {
              activeRelations.splice(ri, 1);
            }
          }
          events.push({
            type: "delete",
            timestamp: performance.now() - start_time,
            span_id: spanId
          });
          applySpanHighlights();
          renderSpanList();
          renderRelationList();
          updateContinueButton();
        }
      }
      function setupWikidataSearch() {
        const input = display_element.querySelector("#bead-wikidata-input");
        const resultsDiv = display_element.querySelector("#bead-wikidata-results");
        if (!input || !resultsDiv) return;
        const searchOptions = {
          language: spanSpec?.wikidata_language ?? "en",
          limit: spanSpec?.wikidata_result_limit ?? 10,
          ...spanSpec?.wikidata_entity_types ? { entityTypes: spanSpec.wikidata_entity_types } : {}
        };
        input.addEventListener("input", () => {
          const query = input.value.trim();
          if (query.length === 0) {
            resultsDiv.style.display = "none";
            resultsDiv.innerHTML = "";
            return;
          }
          debouncedSearchWikidata(query, searchOptions, (results) => {
            resultsDiv.innerHTML = "";
            if (results.length === 0) {
              resultsDiv.style.display = "none";
              return;
            }
            resultsDiv.style.display = "block";
            for (const entity of results) {
              const item = document.createElement("div");
              item.className = "bead-wikidata-result";
              item.innerHTML = `<div><strong>${entity.label}</strong> <span class="qid">${entity.id}</span></div>${entity.description ? `<div class="description">${entity.description}</div>` : ""}`;
              item.addEventListener("click", () => {
                createSpanFromSelection(entity.label, entity.id);
                input.value = "";
                resultsDiv.style.display = "none";
                resultsDiv.innerHTML = "";
              });
              resultsDiv.appendChild(item);
            }
          });
        });
      }
      function setupFixedLabelSearch() {
        const input = display_element.querySelector("#bead-label-search-input");
        const resultsDiv = display_element.querySelector(
          "#bead-label-search-results"
        );
        if (!input || !resultsDiv) return;
        const allLabels = spanSpec?.labels ?? [];
        let highlightedIdx = -1;
        function fuzzyMatch(query, target) {
          const q = query.toLowerCase();
          const t = target.toLowerCase();
          let qi = 0;
          for (let ti = 0; ti < t.length && qi < q.length; ti++) {
            if (t[ti] === q[qi]) qi++;
          }
          return qi === q.length;
        }
        const renderResults = (query) => {
          resultsDiv.innerHTML = "";
          const lower = query.toLowerCase();
          const filtered = lower === "" ? allLabels : allLabels.filter((l) => fuzzyMatch(lower, l)).sort((a, b) => {
            const aPrefix = a.toLowerCase().startsWith(lower);
            const bPrefix = b.toLowerCase().startsWith(lower);
            if (aPrefix && !bPrefix) return -1;
            if (!aPrefix && bPrefix) return 1;
            return 0;
          });
          if (filtered.length === 0) {
            resultsDiv.style.display = "none";
            return;
          }
          resultsDiv.style.display = "block";
          highlightedIdx = -1;
          for (let fi = 0; fi < filtered.length; fi++) {
            const label = filtered[fi] ?? "";
            const globalIdx = allLabels.indexOf(label);
            palette[globalIdx % palette.length] ?? "#BBDEFB";
            const darkColor = DARK_PALETTE[globalIdx % DARK_PALETTE.length] ?? "#1565C0";
            const shortcut = globalIdx < 9 ? `${globalIdx + 1}` : "";
            const item = document.createElement("div");
            item.className = "bead-label-search-result";
            item.setAttribute("data-label", label);
            item.setAttribute("data-fi", String(fi));
            item.innerHTML = `<span class="label-color" style="background:${darkColor}"></span><span class="label-name">${label}</span>${shortcut ? `<span class="label-shortcut">${shortcut}</span>` : ""}`;
            item.addEventListener("click", () => {
              if (selectedIndices.length > 0) {
                createSpanFromSelection(label);
                input.value = "";
                resultsDiv.style.display = "none";
              }
            });
            resultsDiv.appendChild(item);
          }
        };
        input.addEventListener("focus", () => {
          if (selectedIndices.length > 0) {
            renderResults(input.value);
          }
        });
        input.addEventListener("input", () => {
          renderResults(input.value);
        });
        input.addEventListener("keydown", (e) => {
          const items = resultsDiv.querySelectorAll(".bead-label-search-result");
          if (items.length === 0) return;
          if (e.key === "ArrowDown") {
            e.preventDefault();
            highlightedIdx = Math.min(highlightedIdx + 1, items.length - 1);
            updateHighlight(items);
          } else if (e.key === "ArrowUp") {
            e.preventDefault();
            highlightedIdx = Math.max(highlightedIdx - 1, 0);
            updateHighlight(items);
          } else if (e.key === "Enter") {
            e.preventDefault();
            if (highlightedIdx >= 0 && highlightedIdx < items.length) {
              const label = items[highlightedIdx]?.getAttribute("data-label") ?? "";
              if (label && selectedIndices.length > 0) {
                createSpanFromSelection(label);
                input.value = "";
                resultsDiv.style.display = "none";
              }
            }
          } else if (e.key === "Escape") {
            resultsDiv.style.display = "none";
          }
        });
        function updateHighlight(items) {
          for (let i = 0; i < items.length; i++) {
            items[i]?.classList.toggle("highlighted", i === highlightedIdx);
          }
          items[highlightedIdx]?.scrollIntoView({ block: "nearest" });
        }
        document.addEventListener("click", (e) => {
          if (!input.contains(e.target) && !resultsDiv.contains(e.target)) {
            resultsDiv.style.display = "none";
          }
        });
      }
      function setupRelationHandlers() {
        const addBtn = display_element.querySelector("#bead-add-relation");
        const cancelBtn = display_element.querySelector("#bead-relation-cancel");
        const statusEl = display_element.querySelector("#bead-relation-status");
        if (addBtn) {
          addBtn.addEventListener("click", () => {
            relationState = "WAITING_SOURCE";
            relationSource = null;
            relationTarget = null;
            updateRelationUI();
          });
        }
        if (cancelBtn) {
          cancelBtn.addEventListener("click", () => {
            cancelRelationCreation();
          });
        }
        function updateRelationUI() {
          if (!addBtn || !cancelBtn || !statusEl) return;
          addBtn.disabled = relationState !== "IDLE" || activeSpans.length < 2;
          addBtn.style.display = relationState === "IDLE" ? "" : "none";
          cancelBtn.style.display = relationState !== "IDLE" ? "" : "none";
          if (relationState === "WAITING_SOURCE") {
            statusEl.textContent = "Click a span label to select the source.";
          } else if (relationState === "WAITING_TARGET") {
            statusEl.textContent = "Click a span label to select the target.";
          } else if (relationState === "WAITING_LABEL") {
            statusEl.textContent = "Choose a relation label.";
          } else {
            statusEl.textContent = "";
          }
          const badges = display_element.querySelectorAll(".bead-span-subscript");
          for (const badge of badges) {
            badge.classList.remove("relation-source", "relation-target-candidate");
            const spanId = badge.getAttribute("data-span-id");
            if (relationState === "WAITING_SOURCE" || relationState === "WAITING_TARGET") {
              badge.style.cursor = "pointer";
              if (spanId === relationSource) {
                badge.classList.add("relation-source");
              } else if (relationState === "WAITING_TARGET") {
                badge.classList.add("relation-target-candidate");
              }
            } else {
              badge.style.cursor = "default";
            }
          }
          const labelPanel = display_element.querySelector("#bead-relation-label-panel");
          if (labelPanel) {
            labelPanel.style.display = relationState === "WAITING_LABEL" ? "flex" : "none";
            if (relationState === "WAITING_LABEL") {
              const searchInput = labelPanel.querySelector("input");
              if (searchInput) setTimeout(() => searchInput.focus(), 0);
            }
          }
        }
        display_element["_updateRelationUI"] = updateRelationUI;
        display_element.addEventListener("click", (e) => {
          const badge = e.target.closest(".bead-span-subscript");
          if (!badge) return;
          const spanId = badge.getAttribute("data-span-id");
          if (!spanId) return;
          if (relationState === "WAITING_SOURCE") {
            relationSource = spanId;
            relationState = "WAITING_TARGET";
            updateRelationUI();
          } else if (relationState === "WAITING_TARGET") {
            if (spanId === relationSource) return;
            relationTarget = spanId;
            relationState = "WAITING_LABEL";
            updateRelationUI();
            if (!spanSpec?.relation_labels?.length && spanSpec?.relation_label_source !== "wikidata") {
              createRelation(void 0);
            }
          }
        });
        if (spanSpec?.relation_labels && spanSpec.relation_labels.length > 0 && spanSpec.relation_label_source !== "wikidata") {
          setupRelationLabelSearch();
        }
        if (spanSpec?.relation_label_source === "wikidata") {
          setupRelationWikidataSearch();
        }
        function setupRelationLabelSearch() {
          const input = display_element.querySelector("#bead-relation-label-input");
          const resultsDiv = display_element.querySelector(
            "#bead-relation-label-results"
          );
          if (!input || !resultsDiv) return;
          const allLabels = spanSpec?.relation_labels ?? [];
          let highlightedIdx = -1;
          const renderResults = (query) => {
            resultsDiv.innerHTML = "";
            const lower = query.toLowerCase();
            const filtered = lower === "" ? allLabels : allLabels.filter((l) => l.toLowerCase().includes(lower));
            if (filtered.length === 0) {
              resultsDiv.style.display = "none";
              return;
            }
            resultsDiv.style.display = "block";
            highlightedIdx = -1;
            for (const label of filtered) {
              const item = document.createElement("div");
              item.className = "bead-label-search-result";
              item.setAttribute("data-label", label);
              item.innerHTML = `<span class="label-name">${label}</span>`;
              item.addEventListener("click", () => {
                createRelation({ label });
                input.value = "";
                resultsDiv.style.display = "none";
              });
              resultsDiv.appendChild(item);
            }
          };
          input.addEventListener("focus", () => renderResults(input.value));
          input.addEventListener("input", () => renderResults(input.value));
          input.addEventListener("keydown", (e) => {
            const items = resultsDiv.querySelectorAll(".bead-label-search-result");
            if (items.length === 0) return;
            if (e.key === "ArrowDown") {
              e.preventDefault();
              highlightedIdx = Math.min(highlightedIdx + 1, items.length - 1);
              for (let i = 0; i < items.length; i++)
                items[i]?.classList.toggle("highlighted", i === highlightedIdx);
              items[highlightedIdx]?.scrollIntoView({ block: "nearest" });
            } else if (e.key === "ArrowUp") {
              e.preventDefault();
              highlightedIdx = Math.max(highlightedIdx - 1, 0);
              for (let i = 0; i < items.length; i++)
                items[i]?.classList.toggle("highlighted", i === highlightedIdx);
              items[highlightedIdx]?.scrollIntoView({ block: "nearest" });
            } else if (e.key === "Enter") {
              e.preventDefault();
              if (highlightedIdx >= 0 && highlightedIdx < items.length) {
                const label = items[highlightedIdx]?.getAttribute("data-label") ?? "";
                if (label) {
                  createRelation({ label });
                  input.value = "";
                  resultsDiv.style.display = "none";
                }
              }
            } else if (e.key === "Escape") {
              cancelRelationCreation();
            }
          });
        }
        function setupRelationWikidataSearch() {
          const input = display_element.querySelector(
            "#bead-relation-wikidata-input"
          );
          const resultsDiv = display_element.querySelector(
            "#bead-relation-wikidata-results"
          );
          if (!input || !resultsDiv) return;
          const searchOptions = {
            language: spanSpec?.wikidata_language ?? "en",
            limit: spanSpec?.wikidata_result_limit ?? 10,
            entityTypes: ["property"]
          };
          input.addEventListener("input", () => {
            const query = input.value.trim();
            if (query.length === 0) {
              resultsDiv.style.display = "none";
              resultsDiv.innerHTML = "";
              return;
            }
            debouncedSearchWikidata(query, searchOptions, (results) => {
              resultsDiv.innerHTML = "";
              if (results.length === 0) {
                resultsDiv.style.display = "none";
                return;
              }
              resultsDiv.style.display = "block";
              for (const entity of results) {
                const item = document.createElement("div");
                item.className = "bead-wikidata-result";
                item.innerHTML = `<div><strong>${entity.label}</strong> <span class="qid">${entity.id}</span></div>${entity.description ? `<div class="description">${entity.description}</div>` : ""}`;
                item.addEventListener("click", () => {
                  createRelation({ label: entity.label, label_id: entity.id });
                  input.value = "";
                  resultsDiv.style.display = "none";
                  resultsDiv.innerHTML = "";
                });
                resultsDiv.appendChild(item);
              }
            });
          });
        }
        function createRelation(label) {
          if (!relationSource || !relationTarget) return;
          const relId = `rel_${nextRelationId++}`;
          const newRelation = {
            relation_id: relId,
            source_span_id: relationSource,
            target_span_id: relationTarget,
            ...label !== void 0 ? { label } : {},
            directed: spanSpec?.relation_directed ?? true
          };
          activeRelations.push(newRelation);
          events.push({
            type: "relation_create",
            timestamp: performance.now() - start_time,
            relation_id: relId,
            ...label?.label !== void 0 ? { label: label.label } : {}
          });
          relationState = "IDLE";
          relationSource = null;
          relationTarget = null;
          renderRelationArcsOverlay();
          renderRelationList();
          updateRelationUI();
          updateContinueButton();
        }
        function cancelRelationCreation() {
          relationState = "IDLE";
          relationSource = null;
          relationTarget = null;
          updateRelationUI();
        }
      }
      function deleteRelation(relId) {
        const idx = activeRelations.findIndex((r) => r.relation_id === relId);
        if (idx >= 0) {
          activeRelations.splice(idx, 1);
          events.push({
            type: "relation_delete",
            timestamp: performance.now() - start_time,
            relation_id: relId
          });
          renderRelationArcsOverlay();
          renderRelationList();
          updateContinueButton();
        }
      }
      function renderRelationList() {
        const listEl = display_element.querySelector("#bead-relation-list");
        if (!listEl) return;
        listEl.innerHTML = "";
        for (const rel of activeRelations) {
          const sourceSpan = activeSpans.find((s) => s.span_id === rel.source_span_id);
          const targetSpan = activeSpans.find((s) => s.span_id === rel.target_span_id);
          if (!sourceSpan || !targetSpan) continue;
          const entry = document.createElement("div");
          entry.className = "bead-relation-entry";
          const sourceText = getSpanText(sourceSpan);
          const targetText = getSpanText(targetSpan);
          const labelText = rel.label?.label ?? "(no label)";
          const arrow = rel.directed ? " \u2192 " : " \u2014 ";
          entry.innerHTML = `<span>${sourceText}${arrow}<em>${labelText}</em>${arrow}${targetText}</span>`;
          if (isInteractive) {
            const delBtn = document.createElement("button");
            delBtn.className = "bead-relation-delete";
            delBtn.textContent = "\xD7";
            delBtn.addEventListener("click", () => deleteRelation(rel.relation_id));
            entry.appendChild(delBtn);
          }
          listEl.appendChild(entry);
        }
        const updateUI = display_element["_updateRelationUI"];
        if (typeof updateUI === "function") {
          updateUI();
        }
      }
      function computeSpanPositions() {
        const positions = /* @__PURE__ */ new Map();
        const container = display_element.querySelector(".bead-span-container");
        if (!container) return positions;
        const containerRect = container.getBoundingClientRect();
        for (const span of activeSpans) {
          let minLeft = Number.POSITIVE_INFINITY;
          let minTop = Number.POSITIVE_INFINITY;
          let maxRight = Number.NEGATIVE_INFINITY;
          let maxBottom = Number.NEGATIVE_INFINITY;
          for (const seg of span.segments) {
            for (const idx of seg.indices) {
              const tokenEl = display_element.querySelector(
                `.bead-token[data-element="${seg.element_name}"][data-index="${idx}"]`
              );
              if (tokenEl) {
                const rect = tokenEl.getBoundingClientRect();
                minLeft = Math.min(minLeft, rect.left - containerRect.left);
                minTop = Math.min(minTop, rect.top - containerRect.top);
                maxRight = Math.max(maxRight, rect.right - containerRect.left);
                maxBottom = Math.max(maxBottom, rect.bottom - containerRect.top);
              }
            }
          }
          if (minLeft !== Number.POSITIVE_INFINITY) {
            positions.set(
              span.span_id,
              new DOMRect(minLeft, minTop, maxRight - minLeft, maxBottom - minTop)
            );
          }
        }
        return positions;
      }
      function renderRelationArcsOverlay() {
        if (activeRelations.length === 0) return;
        const container = display_element.querySelector(".bead-span-container");
        if (!container) return;
        const existingArcDiv = display_element.querySelector(".bead-relation-arc-area");
        if (existingArcDiv) existingArcDiv.remove();
        const spanPositions = computeSpanPositions();
        if (spanPositions.size === 0) return;
        const arcArea = document.createElement("div");
        arcArea.className = "bead-relation-arc-area";
        arcArea.style.position = "relative";
        arcArea.style.width = "100%";
        const baseHeight = 28;
        const levelSpacing = 28;
        const totalHeight = baseHeight + (activeRelations.length - 1) * levelSpacing + 12;
        arcArea.style.height = `${totalHeight}px`;
        arcArea.style.marginBottom = "4px";
        const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        svg.classList.add("bead-relation-layer");
        svg.setAttribute("width", "100%");
        svg.setAttribute("height", String(totalHeight));
        svg.style.overflow = "visible";
        const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
        const marker = document.createElementNS("http://www.w3.org/2000/svg", "marker");
        marker.setAttribute("id", "rel-arrow");
        marker.setAttribute("markerWidth", "8");
        marker.setAttribute("markerHeight", "6");
        marker.setAttribute("refX", "8");
        marker.setAttribute("refY", "3");
        marker.setAttribute("orient", "auto");
        const polygon = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
        polygon.setAttribute("points", "0 0, 8 3, 0 6");
        polygon.setAttribute("fill", "#546e7a");
        marker.appendChild(polygon);
        defs.appendChild(marker);
        svg.appendChild(defs);
        container.getBoundingClientRect();
        arcArea.getBoundingClientRect();
        for (let i = 0; i < activeRelations.length; i++) {
          const rel = activeRelations[i];
          if (!rel) continue;
          const sourceRect = spanPositions.get(rel.source_span_id);
          const targetRect = spanPositions.get(rel.target_span_id);
          if (!sourceRect || !targetRect) continue;
          const x1 = sourceRect.x + sourceRect.width / 2;
          const x2 = targetRect.x + targetRect.width / 2;
          const bottomY = totalHeight;
          const railY = totalHeight - baseHeight - i * levelSpacing;
          const r = 5;
          const strokeColor = "#546e7a";
          const dir = x2 > x1 ? 1 : -1;
          const d = [
            `M ${x1} ${bottomY}`,
            `L ${x1} ${railY + r}`,
            `Q ${x1} ${railY} ${x1 + r * dir} ${railY}`,
            `L ${x2 - r * dir} ${railY}`,
            `Q ${x2} ${railY} ${x2} ${railY + r}`,
            `L ${x2} ${bottomY}`
          ].join(" ");
          const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
          path.setAttribute("d", d);
          path.setAttribute("stroke", strokeColor);
          path.setAttribute("fill", "none");
          path.setAttribute("stroke-width", "1.5");
          if (rel.directed) {
            path.setAttribute("marker-end", "url(#rel-arrow)");
          }
          svg.appendChild(path);
          if (rel.label?.label) {
            const midX = (x1 + x2) / 2;
            const labelText = rel.label.label;
            const fo = document.createElementNS("http://www.w3.org/2000/svg", "foreignObject");
            const labelWidth = labelText.length * 7 + 16;
            fo.setAttribute("x", String(midX - labelWidth / 2));
            fo.setAttribute("y", String(railY - 10));
            fo.setAttribute("width", String(labelWidth));
            fo.setAttribute("height", "20");
            const labelDiv = document.createElement("div");
            labelDiv.style.cssText = `
            font-size: 11px;
            font-family: inherit;
            color: #455a64;
            background: #fafafa;
            padding: 1px 6px;
            border-radius: 3px;
            text-align: center;
            line-height: 18px;
            white-space: nowrap;
          `;
            labelDiv.textContent = labelText;
            fo.appendChild(labelDiv);
            svg.appendChild(fo);
          }
        }
        arcArea.appendChild(svg);
        container.parentNode?.insertBefore(arcArea, container);
      }
      function updateContinueButton() {
        if (!continueBtn || !isInteractive) return;
        const minSpans = spanSpec?.min_spans ?? 0;
        continueBtn.disabled = activeSpans.length < minSpans;
      }
      const endTrial = () => {
        document.removeEventListener("keydown", handleKeyDown);
        const trial_data = {
          ...trial.metadata,
          spans: activeSpans,
          relations: activeRelations,
          span_events: events,
          rt: performance.now() - start_time
        };
        display_element.innerHTML = "";
        this.jsPsych.finishTrial(trial_data);
      };
    }
  };
  __publicField(BeadSpanLabelPlugin, "info", info10);

  // src/gallery/gallery-bundle.ts
  window.BeadRatingPlugin = BeadRatingPlugin;
  window.BeadForcedChoicePlugin = BeadForcedChoicePlugin;
  window.BeadBinaryChoicePlugin = BeadBinaryChoicePlugin;
  window.BeadSliderRatingPlugin = BeadSliderRatingPlugin;
  window.BeadClozeMultiPlugin = BeadClozeMultiPlugin;
  window.BeadSpanLabelPlugin = BeadSpanLabelPlugin;
  window.BeadCategoricalPlugin = BeadCategoricalPlugin;
  window.BeadMagnitudePlugin = BeadMagnitudePlugin;
  window.BeadFreeTextPlugin = BeadFreeTextPlugin;
  window.BeadMultiSelectPlugin = BeadMultiSelectPlugin;

})();
