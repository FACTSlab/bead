/**
 * Unit tests for bead jsPsych plugins.
 *
 * Tests plugin structure, info validation, and instantiation.
 */

import { describe, expect, test, vi } from "vitest";
import type { JsPsych } from "../types/jspsych.js";
import { BeadClozeMultiPlugin } from "./cloze-dropdown.js";
import { BeadForcedChoicePlugin } from "./forced-choice.js";
import { BeadMagnitudePlugin, computeXMax, formatValue, valueToX, xToValue } from "./magnitude.js";
import { BeadRatingPlugin } from "./rating.js";
import { BeadSpanLabelPlugin } from "./span-label.js";

// Mock jsPsych instance
function createMockJsPsych(): JsPsych {
  return {
    pluginAPI: {
      getKeyboardResponse: vi.fn(),
      cancelKeyboardResponse: vi.fn(),
    },
    finishTrial: vi.fn(),
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

describe("bead-span-label plugin", () => {
  describe("info structure", () => {
    test("has correct plugin name", () => {
      expect(BeadSpanLabelPlugin.info.name).toBe("bead-span-label");
    });

    test("has required parameters", () => {
      const params = BeadSpanLabelPlugin.info.parameters;
      expect(params["tokens"]).toBeDefined();
      expect(params["space_after"]).toBeDefined();
      expect(params["spans"]).toBeDefined();
      expect(params["relations"]).toBeDefined();
      expect(params["span_spec"]).toBeDefined();
      expect(params["display_config"]).toBeDefined();
      expect(params["prompt"]).toBeDefined();
      expect(params["button_label"]).toBeDefined();
      expect(params["require_response"]).toBeDefined();
      expect(params["metadata"]).toBeDefined();
    });

    test("has correct parameter defaults", () => {
      const params = BeadSpanLabelPlugin.info.parameters;
      expect(params["require_response"]?.default).toBe(true);
      expect(params["button_label"]?.default).toBe("Continue");
      expect(params["prompt"]?.default).toBe("Select and label spans");
    });
  });

  describe("plugin instantiation", () => {
    test("can be instantiated", () => {
      const mockJsPsych = createMockJsPsych();
      const plugin = new BeadSpanLabelPlugin(mockJsPsych);
      expect(plugin).toBeDefined();
    });

    test("has trial method", () => {
      const mockJsPsych = createMockJsPsych();
      const plugin = new BeadSpanLabelPlugin(mockJsPsych);
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

describe("bead-magnitude plugin", () => {
  describe("info structure", () => {
    test("has correct plugin name", () => {
      expect(BeadMagnitudePlugin.info.name).toBe("bead-magnitude");
    });

    test("has required parameters", () => {
      const params = BeadMagnitudePlugin.info.parameters;
      expect(params["prompt"]).toBeDefined();
      expect(params["stimulus"]).toBeDefined();
      expect(params["reference_stimulus"]).toBeDefined();
      expect(params["reference_value"]).toBeDefined();
      expect(params["input_mode"]).toBeDefined();
      expect(params["arrow_step"]).toBeDefined();
      expect(params["slider_start"]).toBeDefined();
      expect(params["input_min"]).toBeDefined();
      expect(params["input_max"]).toBeDefined();
      expect(params["require_response"]).toBeDefined();
      expect(params["button_label"]).toBeDefined();
      expect(params["metadata"]).toBeDefined();
    });

    test("has correct parameter defaults", () => {
      const params = BeadMagnitudePlugin.info.parameters;
      expect(params["reference_value"]?.default).toBe(100);
      expect(params["input_mode"]?.default).toBe("number");
      expect(params["arrow_step"]?.default).toBe(3);
      expect(params["slider_start"]?.default).toBeNull();
      expect(params["require_response"]?.default).toBe(true);
      expect(params["button_label"]?.default).toBe("Continue");
    });
  });

  describe("plugin instantiation", () => {
    test("can be instantiated", () => {
      const mockJsPsych = createMockJsPsych();
      const plugin = new BeadMagnitudePlugin(mockJsPsych);
      expect(plugin).toBeDefined();
    });

    test("has trial method", () => {
      const mockJsPsych = createMockJsPsych();
      const plugin = new BeadMagnitudePlugin(mockJsPsych);
      expect(typeof plugin.trial).toBe("function");
    });
  });
});

describe("exponential slider math", () => {
  test("computeXMax for reference_value=100", () => {
    const xMax = computeXMax(100);
    expect(xMax).toBeCloseTo(3 * 100 * Math.log(101), 5);
  });

  test("xToValue at x=0 returns 0", () => {
    expect(xToValue(0)).toBe(0);
  });

  test("xToValue at reference position returns reference_value", () => {
    const xRef = computeXMax(100) / 3;
    expect(xToValue(xRef)).toBeCloseTo(100, 5);
  });

  test("xToValue at xMax returns very large number", () => {
    const xMax = computeXMax(100);
    const maxVal = xToValue(xMax);
    expect(maxVal).toBeGreaterThan(1_000_000);
  });

  test("valueToX inverts xToValue", () => {
    const testValues = [0, 1, 10, 100, 1000, 50000];
    for (const v of testValues) {
      expect(xToValue(valueToX(v))).toBeCloseTo(v, 5);
    }
  });

  test("valueToX at 0 returns 0", () => {
    expect(valueToX(0)).toBe(0);
  });

  test("reference_value maps to 1/3 of xMax", () => {
    const xMax = computeXMax(100);
    const xRef = valueToX(100);
    expect(xRef / xMax).toBeCloseTo(1 / 3, 5);
  });

  test("formatValue handles all ranges", () => {
    expect(formatValue(0)).toBe("0");
    expect(formatValue(0.005)).toBe("0.005");
    expect(formatValue(5.123)).toBe("5.12");
    expect(formatValue(42.567)).toBe("42.6");
    expect(formatValue(100)).toBe("100");
    expect(formatValue(2_000_000)).toBe("\u221E");
  });
});
