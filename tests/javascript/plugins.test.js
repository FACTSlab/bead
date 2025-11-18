/**
 * Unit tests for bead jsPsych plugins
 *
 * Tests plugin structure, info validation, and core functionality.
 */

// Mock jsPsych ParameterType constants
const mockJsPsych = {
    ParameterType: {
        HTML_STRING: 'HTML_STRING',
        INT: 'INT',
        STRING: 'STRING',
        BOOL: 'BOOL',
        OBJECT: 'OBJECT',
        KEYS: 'KEYS',
        FUNCTION: 'FUNCTION',
    },
    pluginAPI: {
        getKeyboardResponse: jest.fn(),
        cancelKeyboardResponse: jest.fn(),
    },
    finishTrial: jest.fn(),
};

// Mock global jsPsychModule
global.jsPsychModule = mockJsPsych;

// Mock performance.now()
global.performance = {
    now: jest.fn(() => 1000),
};

// Load the plugins
let BeadRatingPlugin, BeadForcedChoicePlugin, BeadClozeDropdownPlugin;

describe('Plugin Loading', () => {
    test('bead-rating plugin loads successfully', () => {
        const fs = require('fs');
        const pluginCode = fs.readFileSync('bead/deployment/jspsych/plugins/rating.js', 'utf8');
        eval(pluginCode);
        BeadRatingPlugin = jsPsychBeadRating;

        expect(BeadRatingPlugin).toBeDefined();
        expect(BeadRatingPlugin.info).toBeDefined();
    });

    test('bead-forced-choice plugin loads successfully', () => {
        const fs = require('fs');
        const pluginCode = fs.readFileSync('bead/deployment/jspsych/plugins/forced-choice.js', 'utf8');
        eval(pluginCode);
        BeadForcedChoicePlugin = jsPsychBeadForcedChoice;

        expect(BeadForcedChoicePlugin).toBeDefined();
        expect(BeadForcedChoicePlugin.info).toBeDefined();
    });

    test('bead-cloze-dropdown plugin loads successfully', () => {
        const fs = require('fs');
        const pluginCode = fs.readFileSync('bead/deployment/jspsych/plugins/cloze-dropdown.js', 'utf8');
        eval(pluginCode);
        BeadClozeDropdownPlugin = jsPsychBeadClozeMulti;

        expect(BeadClozeDropdownPlugin).toBeDefined();
        expect(BeadClozeDropdownPlugin.info).toBeDefined();
    });
});

describe('bead-rating plugin', () => {
    let BeadRating;

    beforeAll(() => {
        const fs = require('fs');
        const pluginCode = fs.readFileSync('bead/deployment/jspsych/plugins/rating.js', 'utf8');
        eval(pluginCode);
        BeadRating = jsPsychBeadRating;
    });

    describe('info structure', () => {
        test('has correct plugin name', () => {
            expect(BeadRating.info.name).toBe('bead-rating');
        });

        test('has required parameters', () => {
            const params = BeadRating.info.parameters;
            expect(params.prompt).toBeDefined();
            expect(params.scale_min).toBeDefined();
            expect(params.scale_max).toBeDefined();
            expect(params.scale_labels).toBeDefined();
            expect(params.require_response).toBeDefined();
            expect(params.button_label).toBeDefined();
            expect(params.metadata).toBeDefined();
        });

        test('has correct parameter defaults', () => {
            const params = BeadRating.info.parameters;
            expect(params.scale_min.default).toBe(1);
            expect(params.scale_max.default).toBe(7);
            expect(params.require_response.default).toBe(true);
            expect(params.button_label.default).toBe('Continue');
        });
    });

    describe('plugin instantiation', () => {
        test('can be instantiated', () => {
            const plugin = new BeadRating(mockJsPsych);
            expect(plugin).toBeDefined();
            expect(plugin.jsPsych).toBe(mockJsPsych);
        });

        test('has trial method', () => {
            const plugin = new BeadRating(mockJsPsych);
            expect(typeof plugin.trial).toBe('function');
        });
    });
});

describe('bead-forced-choice plugin', () => {
    let BeadForcedChoice;

    beforeAll(() => {
        const fs = require('fs');
        const pluginCode = fs.readFileSync('bead/deployment/jspsych/plugins/forced-choice.js', 'utf8');
        eval(pluginCode);
        BeadForcedChoice = jsPsychBeadForcedChoice;
    });

    describe('info structure', () => {
        test('has correct plugin name', () => {
            expect(BeadForcedChoice.info.name).toBe('bead-forced-choice');
        });

        test('has required parameters', () => {
            const params = BeadForcedChoice.info.parameters;
            expect(params.alternatives).toBeDefined();
            expect(params.prompt).toBeDefined();
            expect(params.button_label).toBeDefined();
            expect(params.require_response).toBeDefined();
            expect(params.randomize_position).toBeDefined();
            expect(params.enable_keyboard).toBeDefined();
            expect(params.metadata).toBeDefined();
        });

        test('has correct parameter defaults', () => {
            const params = BeadForcedChoice.info.parameters;
            expect(params.require_response.default).toBe(true);
            expect(params.randomize_position.default).toBe(true);
            expect(params.enable_keyboard.default).toBe(true);
            expect(params.button_label.default).toBe('Continue');
        });
    });

    describe('plugin instantiation', () => {
        test('can be instantiated', () => {
            const plugin = new BeadForcedChoice(mockJsPsych);
            expect(plugin).toBeDefined();
            expect(plugin.jsPsych).toBe(mockJsPsych);
        });

        test('has trial method', () => {
            const plugin = new BeadForcedChoice(mockJsPsych);
            expect(typeof plugin.trial).toBe('function');
        });
    });
});

describe('bead-cloze-dropdown plugin', () => {
    let BeadClozeDropdown;

    beforeAll(() => {
        const fs = require('fs');
        const pluginCode = fs.readFileSync('bead/deployment/jspsych/plugins/cloze-dropdown.js', 'utf8');
        eval(pluginCode);
        BeadClozeDropdown = jsPsychBeadClozeMulti;
    });

    describe('info structure', () => {
        test('has correct plugin name', () => {
            expect(BeadClozeDropdown.info.name).toBe('bead-cloze-multi');
        });

        test('has required parameters', () => {
            const params = BeadClozeDropdown.info.parameters;
            expect(params.text).toBeDefined();
            expect(params.fields).toBeDefined();
            expect(params.require_all).toBeDefined();
            expect(params.button_label).toBeDefined();
            expect(params.metadata).toBeDefined();
        });

        test('has correct parameter defaults', () => {
            const params = BeadClozeDropdown.info.parameters;
            expect(params.require_all.default).toBe(true);
            expect(params.button_label.default).toBe('Continue');
        });
    });

    describe('plugin instantiation', () => {
        test('can be instantiated', () => {
            const plugin = new BeadClozeDropdown(mockJsPsych);
            expect(plugin).toBeDefined();
            expect(plugin.jsPsych).toBe(mockJsPsych);
        });

        test('has trial method', () => {
            const plugin = new BeadClozeDropdown(mockJsPsych);
            expect(typeof plugin.trial).toBe('function');
        });
    });
});
