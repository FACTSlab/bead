/**
 * Unit tests for randomizer.ts
 *
 * Tests the trial randomization and constraint checking functions.
 */

import { describe, expect, test } from "vitest";
import {
  checkMinDistance,
  checkNoAdjacent,
  checkPrecedence,
  getPropertyValue,
  shuffle,
  type Trial,
  type TrialMetadata,
} from "./randomizer.js";

describe("shuffle (Fisher-Yates)", () => {
  test("shuffles array in place", () => {
    const array = [1, 2, 3, 4, 5];
    const rng = (): number => 0.5; // Deterministic RNG

    shuffle(array, rng);

    expect(array).toHaveLength(5);
    expect(array).toContain(1);
    expect(array).toContain(2);
    expect(array).toContain(3);
    expect(array).toContain(4);
    expect(array).toContain(5);
  });

  test("produces different orders with different seeds", () => {
    const array1 = [1, 2, 3, 4, 5];
    const array2 = [1, 2, 3, 4, 5];

    // Use simple counter-based RNGs with different starting points
    let counter1 = 0;
    const rng1 = (): number => {
      counter1++;
      return (counter1 * 0.1) % 1.0;
    };

    let counter2 = 0;
    const rng2 = (): number => {
      counter2++;
      return (counter2 * 0.7) % 1.0;
    };

    shuffle(array1, rng1);
    shuffle(array2, rng2);

    // Arrays should be different with different seeds
    const isDifferent = array1.some((val, idx) => val !== array2[idx]);
    expect(isDifferent).toBe(true);
  });

  test("handles empty array", () => {
    const array: number[] = [];
    const rng = (): number => 0.5;

    expect(() => shuffle(array, rng)).not.toThrow();
    expect(array).toEqual([]);
  });

  test("handles single element array", () => {
    const array = [42];
    const rng = (): number => 0.5;

    shuffle(array, rng);

    expect(array).toEqual([42]);
  });
});

describe("getPropertyValue", () => {
  test("retrieves top-level property", () => {
    const obj = { name: "test", value: 42 };
    expect(getPropertyValue(obj, "name")).toBe("test");
    expect(getPropertyValue(obj, "value")).toBe(42);
  });

  test("retrieves nested property", () => {
    const obj = {
      item_metadata: {
        condition: "A",
        nested: {
          deep: "value",
        },
      },
    };

    expect(getPropertyValue(obj, "item_metadata.condition")).toBe("A");
    expect(getPropertyValue(obj, "item_metadata.nested.deep")).toBe("value");
  });

  test("returns undefined for missing property", () => {
    const obj = { name: "test" };
    expect(getPropertyValue(obj, "missing")).toBeUndefined();
    expect(getPropertyValue(obj, "nested.missing")).toBeUndefined();
  });

  test("handles null and undefined safely", () => {
    expect(getPropertyValue(undefined, "path")).toBeUndefined();
  });
});

describe("checkPrecedence", () => {
  test("returns true when precedence satisfied", () => {
    const trials: Trial[] = [{ item_id: "item1" }, { item_id: "item2" }, { item_id: "item3" }];
    const pairs: [string, string][] = [
      ["item1", "item3"],
      ["item2", "item3"],
    ];

    expect(checkPrecedence(trials, pairs)).toBe(true);
  });

  test("returns false when precedence violated", () => {
    const trials: Trial[] = [{ item_id: "item3" }, { item_id: "item1" }, { item_id: "item2" }];
    const pairs: [string, string][] = [["item1", "item3"]]; // item1 should come before item3

    expect(checkPrecedence(trials, pairs)).toBe(false);
  });

  test("handles missing items gracefully", () => {
    const trials: Trial[] = [{ item_id: "item1" }, { item_id: "item2" }];
    const pairs: [string, string][] = [["item1", "item99"]]; // item99 not in trials

    expect(checkPrecedence(trials, pairs)).toBe(true); // Ignored
  });

  test("handles empty pairs array", () => {
    const trials: Trial[] = [{ item_id: "item1" }];
    const pairs: [string, string][] = [];

    expect(checkPrecedence(trials, pairs)).toBe(true);
  });
});

describe("checkNoAdjacent", () => {
  test("returns true when no adjacent items have same value", () => {
    const trials: Trial[] = [{ item_id: "item1" }, { item_id: "item2" }, { item_id: "item3" }];
    const metadata: TrialMetadata = {
      item1: { condition: "A" },
      item2: { condition: "B" },
      item3: { condition: "A" },
    };

    expect(checkNoAdjacent(trials, "condition", metadata)).toBe(true);
  });

  test("returns false when adjacent items have same value", () => {
    const trials: Trial[] = [{ item_id: "item1" }, { item_id: "item2" }, { item_id: "item3" }];
    const metadata: TrialMetadata = {
      item1: { condition: "A" },
      item2: { condition: "A" }, // Same as item1
      item3: { condition: "B" },
    };

    expect(checkNoAdjacent(trials, "condition", metadata)).toBe(false);
  });

  test("handles nested property paths", () => {
    const trials: Trial[] = [{ item_id: "item1" }, { item_id: "item2" }];
    const metadata: TrialMetadata = {
      item1: { item_metadata: { condition: "A" } },
      item2: { item_metadata: { condition: "B" } },
    };

    expect(checkNoAdjacent(trials, "item_metadata.condition", metadata)).toBe(true);
  });

  test("ignores undefined values", () => {
    const trials: Trial[] = [{ item_id: "item1" }, { item_id: "item2" }];
    const metadata: TrialMetadata = {
      item1: { condition: "A" },
      item2: {}, // Missing condition
    };

    expect(checkNoAdjacent(trials, "condition", metadata)).toBe(true);
  });
});

describe("checkMinDistance", () => {
  test("returns true when minimum distance satisfied", () => {
    const trials: Trial[] = [
      { item_id: "item1" }, // pos 0
      { item_id: "item2" }, // pos 1
      { item_id: "item3" }, // pos 2
      { item_id: "item4" }, // pos 3
      { item_id: "item5" }, // pos 4
    ];
    const metadata: TrialMetadata = {
      item1: { condition: "A" },
      item2: { condition: "B" },
      item3: { condition: "C" },
      item4: { condition: "B" },
      item5: { condition: "A" },
    };

    // Distance between A's: 4 - 0 - 1 = 3
    // Distance between B's: 3 - 1 - 1 = 1
    expect(checkMinDistance(trials, "condition", 1, metadata)).toBe(true);
  });

  test("returns false when minimum distance violated", () => {
    const trials: Trial[] = [
      { item_id: "item1" }, // pos 0, A
      { item_id: "item2" }, // pos 1, B
      { item_id: "item3" }, // pos 2, A
    ];
    const metadata: TrialMetadata = {
      item1: { condition: "A" },
      item2: { condition: "B" },
      item3: { condition: "A" },
    };

    // Distance between A's: 2 - 0 - 1 = 1 (need min 2)
    expect(checkMinDistance(trials, "condition", 2, metadata)).toBe(false);
  });

  test("handles single occurrence (no distance constraint)", () => {
    const trials: Trial[] = [{ item_id: "item1" }, { item_id: "item2" }];
    const metadata: TrialMetadata = {
      item1: { condition: "A" },
      item2: { condition: "B" },
    };

    expect(checkMinDistance(trials, "condition", 10, metadata)).toBe(true);
  });
});
