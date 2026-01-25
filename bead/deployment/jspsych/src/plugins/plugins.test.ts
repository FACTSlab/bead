/**
 * Unit tests for bead jsPsych plugins.
 *
 * Tests plugin structure, info validation, and instantiation.
 */

import { describe, expect, test, vi } from "vitest";
import type { JsPsych } from "../types/jspsych.js";
import { BeadClozeMultiPlugin } from "./cloze-dropdown.js";
import { BeadForcedChoicePlugin } from "./forced-choice.js";
import { BeadRatingPlugin } from "./rating.js";

// Mock jsPsych instance
function createMockJsPsych(): JsPsych {
  return {
    pluginAPI: {
      getKeyboardResponse: vi.fn(),
      cancelKeyboardResponse: vi.fn(),
    },
    finishTrial: vi.fn(),
    // Add required methods that may be accessed
  } as unknown as JsPsych;
}

describe("bead-rating plugin", () => {
  describe("info structure", () => {
    test("has correct plugin name", () => {
      expect(BeadRatingPlugin.info.name).toBe("bead-rating");
    });

    test("has required parameters", () => {
      const params = BeadRatingPlugin.info.parameters;
      expect(params["prompt"]).toBeDefined();
      expect(params["scale_min"]).toBeDefined();
      expect(params["scale_max"]).toBeDefined();
      expect(params["scale_labels"]).toBeDefined();
      expect(params["require_response"]).toBeDefined();
      expect(params["button_label"]).toBeDefined();
      expect(params["metadata"]).toBeDefined();
    });

    test("has correct parameter defaults", () => {
      const params = BeadRatingPlugin.info.parameters;
      expect(params["scale_min"]?.default).toBe(1);
      expect(params["scale_max"]?.default).toBe(7);
      expect(params["require_response"]?.default).toBe(true);
      expect(params["button_label"]?.default).toBe("Continue");
    });
  });

  describe("plugin instantiation", () => {
    test("can be instantiated", () => {
      const mockJsPsych = createMockJsPsych();
      const plugin = new BeadRatingPlugin(mockJsPsych);
      expect(plugin).toBeDefined();
    });

    test("has trial method", () => {
      const mockJsPsych = createMockJsPsych();
      const plugin = new BeadRatingPlugin(mockJsPsych);
      expect(typeof plugin.trial).toBe("function");
    });
  });
});

describe("bead-forced-choice plugin", () => {
  describe("info structure", () => {
    test("has correct plugin name", () => {
      expect(BeadForcedChoicePlugin.info.name).toBe("bead-forced-choice");
    });

    test("has required parameters", () => {
      const params = BeadForcedChoicePlugin.info.parameters;
      expect(params["alternatives"]).toBeDefined();
      expect(params["prompt"]).toBeDefined();
      expect(params["button_label"]).toBeDefined();
      expect(params["require_response"]).toBeDefined();
      expect(params["randomize_position"]).toBeDefined();
      expect(params["enable_keyboard"]).toBeDefined();
      expect(params["metadata"]).toBeDefined();
    });

    test("has correct parameter defaults", () => {
      const params = BeadForcedChoicePlugin.info.parameters;
      expect(params["require_response"]?.default).toBe(true);
      expect(params["randomize_position"]?.default).toBe(true);
      expect(params["enable_keyboard"]?.default).toBe(true);
      expect(params["button_label"]?.default).toBe("Continue");
    });
  });

  describe("plugin instantiation", () => {
    test("can be instantiated", () => {
      const mockJsPsych = createMockJsPsych();
      const plugin = new BeadForcedChoicePlugin(mockJsPsych);
      expect(plugin).toBeDefined();
    });

    test("has trial method", () => {
      const mockJsPsych = createMockJsPsych();
      const plugin = new BeadForcedChoicePlugin(mockJsPsych);
      expect(typeof plugin.trial).toBe("function");
    });
  });
});

describe("bead-cloze-multi plugin", () => {
  describe("info structure", () => {
    test("has correct plugin name", () => {
      expect(BeadClozeMultiPlugin.info.name).toBe("bead-cloze-multi");
    });

    test("has required parameters", () => {
      const params = BeadClozeMultiPlugin.info.parameters;
      expect(params["text"]).toBeDefined();
      expect(params["fields"]).toBeDefined();
      expect(params["require_all"]).toBeDefined();
      expect(params["button_label"]).toBeDefined();
      expect(params["metadata"]).toBeDefined();
    });

    test("has correct parameter defaults", () => {
      const params = BeadClozeMultiPlugin.info.parameters;
      expect(params["require_all"]?.default).toBe(true);
      expect(params["button_label"]?.default).toBe("Continue");
    });
  });

  describe("plugin instantiation", () => {
    test("can be instantiated", () => {
      const mockJsPsych = createMockJsPsych();
      const plugin = new BeadClozeMultiPlugin(mockJsPsych);
      expect(plugin).toBeDefined();
    });

    test("has trial method", () => {
      const mockJsPsych = createMockJsPsych();
      const plugin = new BeadClozeMultiPlugin(mockJsPsych);
      expect(typeof plugin.trial).toBe("function");
    });
  });
});
