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
 * @version 0.1.0
 */

var jsPsychBeadClozeMulti = (function (jspsych) {
  'use strict';

  const info = {
    name: 'bead-cloze-multi',
    parameters: {
      /** The text with gaps (use %% or {{slot_name}} for gaps) */
      text: {
        type: jspsych.ParameterType.HTML_STRING,
        pretty_name: 'Text',
        default: null,
      },
      /** Field configurations (auto-generated from unfilled_slots) */
      fields: {
        type: jspsych.ParameterType.COMPLEX,
        pretty_name: 'Field configurations',
        default: [],
        array: true,
      },
      /** Whether to require all fields to be filled */
      require_all: {
        type: jspsych.ParameterType.BOOL,
        pretty_name: 'Require all fields',
        default: true,
      },
      /** Text for the continue button */
      button_label: {
        type: jspsych.ParameterType.STRING,
        pretty_name: 'Button label',
        default: 'Continue',
      },
      /** Complete item and template metadata */
      metadata: {
        type: jspsych.ParameterType.OBJECT,
        pretty_name: 'Item and template metadata',
        default: {},
      },
    },
  };

  /**
   * bead-cloze-multi plugin
   */
  class BeadClozeMultiPlugin {
    constructor(jsPsych) {
      this.jsPsych = jsPsych;
    }

    trial(display_element, trial) {
      const responses = {};
      const response_times = {};
      const field_start_times = {};

      const start_time = performance.now();

      // Auto-generate fields from metadata if not provided
      if (trial.fields.length === 0 && trial.metadata && trial.metadata.unfilled_slots) {
        trial.fields = trial.metadata.unfilled_slots.map(slot => ({
          slot_name: slot.slot_name,
          position: slot.position,
          type: slot.constraint_ids.length > 0 ? 'dropdown' : 'text',
          options: [],  // Would be populated from constraints in real implementation
          placeholder: slot.slot_name,
        }));
      }

      // Generate HTML
      let html = '<div class="bead-cloze-container">';

      if (trial.text) {
        // Replace gaps with input fields
        let processed_text = trial.text;
        trial.fields.forEach((field, index) => {
          const field_id = `bead-cloze-field-${index}`;
          let field_html;

          if (field.type === 'dropdown' && field.options && field.options.length > 0) {
            // Dropdown field
            field_html = `
              <select class="bead-dropdown bead-cloze-field" id="${field_id}" data-field="${index}">
                <option value="">Select...</option>
                ${field.options.map(opt => `<option value="${opt}">${opt}</option>`).join('')}
              </select>
            `;
          } else {
            // Text input field
            field_html = `
              <input type="text"
                     class="bead-text-field bead-cloze-field"
                     id="${field_id}"
                     data-field="${index}"
                     placeholder="${field.placeholder || ''}" />
            `;
          }

          // Replace placeholder in text
          const placeholder = field.slot_name ? `{{${field.slot_name}}}` : '%%';
          processed_text = processed_text.replace(placeholder, field_html);
        });

        html += `<div class="bead-cloze-text">${processed_text}</div>`;
      }

      // Continue button
      html += `
        <div class="bead-cloze-button-container">
          <button class="bead-button bead-continue-button" id="bead-cloze-continue" ${trial.require_all ? 'disabled' : ''}>
            ${trial.button_label}
          </button>
        </div>
      `;

      html += '</div>'; // Close container

      display_element.innerHTML = html;

      // Add event listeners for input fields
      const input_fields = display_element.querySelectorAll('.bead-cloze-field');
      input_fields.forEach((field) => {
        const field_index = field.getAttribute('data-field');

        // Track when user starts interacting with this field
        field.addEventListener('focus', () => {
          if (!field_start_times[field_index]) {
            field_start_times[field_index] = performance.now();
          }
        });

        // Track responses
        field.addEventListener('change', () => {
          responses[field_index] = field.value;

          // Record response time for this field
          if (field_start_times[field_index]) {
            response_times[field_index] = performance.now() - field_start_times[field_index];
          }

          check_completion();
        });

        field.addEventListener('input', () => {
          responses[field_index] = field.value;
          check_completion();
        });
      });

      // Continue button listener
      const continue_button = display_element.querySelector('#bead-cloze-continue');
      continue_button.addEventListener('click', () => {
        end_trial();
      });

      const check_completion = () => {
        if (trial.require_all) {
          // Check if all fields are filled
          const all_filled = trial.fields.every((field, index) => {
            return responses[index] && responses[index].trim() !== '';
          });
          continue_button.disabled = !all_filled;
        }
      };

      const end_trial = () => {
        // Gather all responses
        const trial_data = {
          ...trial.metadata,  // Preserve all metadata
          responses: responses,
          response_times: response_times,
          total_rt: performance.now() - start_time,
        };

        // Clear display
        display_element.innerHTML = '';

        // End trial
        this.jsPsych.finishTrial(trial_data);
      };
    }
  }

  BeadClozeMultiPlugin.info = info;

  return BeadClozeMultiPlugin;
})(jsPsychModule);
