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
 * @version 0.1.0
 */

var jsPsychBeadForcedChoice = (function (jspsych) {
  'use strict';

  const info = {
    name: 'bead-forced-choice',
    parameters: {
      /** The prompt/question to display */
      prompt: {
        type: jspsych.ParameterType.HTML_STRING,
        pretty_name: 'Prompt',
        default: 'Which do you prefer?',
      },
      /** Array of alternatives to choose from */
      alternatives: {
        type: jspsych.ParameterType.STRING,
        pretty_name: 'Alternatives',
        default: [],
        array: true,
      },
      /** Whether to randomize left/right position */
      randomize_position: {
        type: jspsych.ParameterType.BOOL,
        pretty_name: 'Randomize position',
        default: true,
      },
      /** Enable keyboard responses (1/2 or left/right arrow) */
      enable_keyboard: {
        type: jspsych.ParameterType.BOOL,
        pretty_name: 'Enable keyboard',
        default: true,
      },
      /** Whether to require a response */
      require_response: {
        type: jspsych.ParameterType.BOOL,
        pretty_name: 'Require response',
        default: true,
      },
      /** Text for the continue button (if applicable) */
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
   * bead-forced-choice plugin
   */
  class BeadForcedChoicePlugin {
    constructor(jsPsych) {
      this.jsPsych = jsPsych;
    }

    trial(display_element, trial) {
      let response = {
        choice: null,
        choice_index: null,
        position: null,  // 'left' or 'right'
        rt: null,
      };

      const start_time = performance.now();

      // Extract alternatives from metadata if not provided
      if (trial.alternatives.length === 0 && trial.metadata && trial.metadata.rendered_elements) {
        const elements = trial.metadata.rendered_elements;
        const choice_keys = Object.keys(elements).filter(k =>
          k.startsWith('choice_') || k.startswith('option_')
        ).sort();

        if (choice_keys.length >= 2) {
          trial.alternatives = choice_keys.map(k => elements[k]);
        } else {
          // Fallback: use all rendered elements
          trial.alternatives = Object.values(elements);
        }
      }

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

      html += '<div class="bead-forced-choice-alternatives">';

      // Left alternative
      html += `
        <div class="bead-card bead-alternative" data-index="${left_index}" data-position="left">
          <div class="bead-alternative-label">Option 1</div>
          <div class="bead-alternative-content">${trial.alternatives[left_index] || 'Alternative A'}</div>
          <button class="bead-button bead-choice-button" data-index="${left_index}" data-position="left">
            Select
          </button>
        </div>
      `;

      // Right alternative
      html += `
        <div class="bead-card bead-alternative" data-index="${right_index}" data-position="right">
          <div class="bead-alternative-label">Option 2</div>
          <div class="bead-alternative-content">${trial.alternatives[right_index] || 'Alternative B'}</div>
          <button class="bead-button bead-choice-button" data-index="${right_index}" data-position="right">
            Select
          </button>
        </div>
      `;

      html += '</div>'; // Close alternatives

      html += '</div>'; // Close container

      display_element.innerHTML = html;

      // Add event listeners for choice buttons
      const choice_buttons = display_element.querySelectorAll('.bead-choice-button');
      choice_buttons.forEach((button) => {
        button.addEventListener('click', (e) => {
          const index = parseInt(e.target.getAttribute('data-index'));
          const position = e.target.getAttribute('data-position');
          select_choice(index, position);
        });
      });

      // Keyboard listener
      let keyboard_listener = null;
      if (trial.enable_keyboard) {
        keyboard_listener = this.jsPsych.pluginAPI.getKeyboardResponse({
          callback_function: (info) => {
            const key = info.key;
            if (key === '1' || key === 'ArrowLeft') {
              select_choice(left_index, 'left');
            } else if (key === '2' || key === 'ArrowRight') {
              select_choice(right_index, 'right');
            }
          },
          valid_responses: ['1', '2', 'ArrowLeft', 'ArrowRight'],
          rt_method: 'performance',
          persist: false,
          allow_held_key: false,
        });
      }

      const select_choice = (index, position) => {
        // Update response
        response.choice = trial.alternatives[index];
        response.choice_index = index;
        response.position = position;
        response.rt = performance.now() - start_time;

        // Visual feedback
        const alternative_cards = display_element.querySelectorAll('.bead-alternative');
        alternative_cards.forEach((card) => {
          card.classList.remove('selected');
        });
        const selected_card = display_element.querySelector(`[data-position="${position}"]`);
        if (selected_card && selected_card.classList.contains('bead-alternative')) {
          selected_card.classList.add('selected');
        }

        // End trial immediately or after delay
        setTimeout(() => {
          end_trial();
        }, 300);  // Small delay for visual feedback
      };

      const end_trial = () => {
        // Kill keyboard listener
        if (keyboard_listener) {
          this.jsPsych.pluginAPI.cancelKeyboardResponse(keyboard_listener);
        }

        // Preserve all metadata and add response data
        const trial_data = {
          ...trial.metadata,  // Spread all metadata
          choice: response.choice,
          choice_index: response.choice_index,
          position_chosen: response.position,
          left_index: left_index,
          right_index: right_index,
          rt: response.rt,
        };

        // Clear display
        display_element.innerHTML = '';

        // End trial
        this.jsPsych.finishTrial(trial_data);
      };
    }
  }

  BeadForcedChoicePlugin.info = info;

  return BeadForcedChoicePlugin;
})(jsPsychModule);
