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
 * @version 0.1.0
 */

var jsPsychBeadRating = (function (jspsych) {
  'use strict';

  const info = {
    name: 'bead-rating',
    parameters: {
      /** The prompt to display above the rating scale */
      prompt: {
        type: jspsych.ParameterType.HTML_STRING,
        pretty_name: 'Prompt',
        default: null,
      },
      /** Minimum value of the scale */
      scale_min: {
        type: jspsych.ParameterType.INT,
        pretty_name: 'Scale minimum',
        default: 1,
      },
      /** Maximum value of the scale */
      scale_max: {
        type: jspsych.ParameterType.INT,
        pretty_name: 'Scale maximum',
        default: 7,
      },
      /** Labels for specific scale points (e.g., {1: "Strongly Disagree", 7: "Strongly Agree"}) */
      scale_labels: {
        type: jspsych.ParameterType.OBJECT,
        pretty_name: 'Scale labels',
        default: {},
      },
      /** Whether to require a response before continuing */
      require_response: {
        type: jspsych.ParameterType.BOOL,
        pretty_name: 'Require response',
        default: true,
      },
      /** Text for the continue button */
      button_label: {
        type: jspsych.ParameterType.STRING,
        pretty_name: 'Button label',
        default: 'Continue',
      },
      /** Complete item and template metadata (automatically populated from trial.data) */
      metadata: {
        type: jspsych.ParameterType.OBJECT,
        pretty_name: 'Item and template metadata',
        default: {},
      },
    },
  };

  /**
   * bead-rating plugin
   */
  class BeadRatingPlugin {
    constructor(jsPsych) {
      this.jsPsych = jsPsych;
    }

    trial(display_element, trial) {
      let response = {
        rating: null,
        rt: null,
      };

      const start_time = performance.now();

      // Override scale bounds from metadata if available
      if (trial.metadata && trial.metadata.task_spec && trial.metadata.task_spec.scale_bounds) {
        trial.scale_min = trial.metadata.task_spec.scale_bounds[0];
        trial.scale_max = trial.metadata.task_spec.scale_bounds[1];
      }

      // Override scale labels from metadata if available
      if (trial.metadata && trial.metadata.task_spec && trial.metadata.task_spec.scale_labels) {
        trial.scale_labels = trial.metadata.task_spec.scale_labels;
      }

      // Override prompt from metadata if available
      if (trial.metadata && trial.metadata.task_spec && trial.metadata.task_spec.prompt && !trial.prompt) {
        trial.prompt = trial.metadata.task_spec.prompt;
      }

      // Create HTML
      let html = '<div class="bead-rating-container">';

      if (trial.prompt !== null) {
        html += `<div class="bead-rating-prompt">${trial.prompt}</div>`;
      }

      html += '<div class="bead-rating-scale">';

      // Create rating buttons
      for (let i = trial.scale_min; i <= trial.scale_max; i++) {
        const label = trial.scale_labels[i] || i;
        html += `
          <div class="bead-rating-option">
            <button class="bead-rating-button" data-value="${i}">${i}</button>
            <div class="bead-rating-label">${label}</div>
          </div>
        `;
      }

      html += '</div>'; // Close scale

      // Continue button
      html += `
        <div class="bead-rating-button-container">
          <button class="bead-button bead-continue-button" id="bead-rating-continue" disabled>
            ${trial.button_label}
          </button>
        </div>
      `;

      html += '</div>'; // Close container

      display_element.innerHTML = html;

      // Add event listeners for rating buttons
      const rating_buttons = display_element.querySelectorAll('.bead-rating-button');
      rating_buttons.forEach((button) => {
        button.addEventListener('click', (e) => {
          const value = parseInt(e.target.getAttribute('data-value'));
          select_rating(value);
        });
      });

      // Keyboard listener
      const keyboard_listener = this.jsPsych.pluginAPI.getKeyboardResponse({
        callback_function: (info) => {
          // Check if key is 1-9
          const key = info.key;
          const num = parseInt(key);
          if (!isNaN(num) && num >= trial.scale_min && num <= trial.scale_max) {
            select_rating(num);
          }
        },
        valid_responses: 'ALL_KEYS',
        rt_method: 'performance',
        persist: true,
        allow_held_key: false,
      });

      // Continue button listener
      const continue_button = display_element.querySelector('#bead-rating-continue');
      continue_button.addEventListener('click', () => {
        if (response.rating !== null || !trial.require_response) {
          end_trial();
        }
      });

      const select_rating = (value) => {
        // Update response
        response.rating = value;
        response.rt = performance.now() - start_time;

        // Update UI
        rating_buttons.forEach((btn) => {
          btn.classList.remove('selected');
        });
        const selected_button = display_element.querySelector(`[data-value="${value}"]`);
        selected_button.classList.add('selected');

        // Enable continue button
        continue_button.disabled = false;
      };

      const end_trial = () => {
        // Kill keyboard listener
        if (keyboard_listener) {
          this.jsPsych.pluginAPI.cancelKeyboardResponse(keyboard_listener);
        }

        // Preserve all metadata from trial.metadata and add response data
        const trial_data = {
          ...trial.metadata,  // Spread all metadata
          rating: response.rating,
          rt: response.rt,
        };

        // Clear display
        display_element.innerHTML = '';

        // End trial
        this.jsPsych.finishTrial(trial_data);
      };
    }
  }

  BeadRatingPlugin.info = info;

  return BeadRatingPlugin;
})(jsPsychModule);
