/**
 * Unit tests for span-renderer.ts
 *
 * Tests token-span mapping, color assignment, and rendering utilities.
 *
 * @vitest-environment jsdom
 */

import { describe, expect, test } from "vitest";
import {
  type SpanData,
  type SpanDisplayConfig,
  assignSpanColors,
  computeTokenSpanMap,
  renderRelationArcs,
  renderTokenizedText,
} from "./span-renderer.js";

const DEFAULT_CONFIG: SpanDisplayConfig = {
  highlight_style: "background",
  color_palette: ["#BBDEFB", "#C8E6C9", "#FFE0B2", "#F8BBD0"],
  show_labels: true,
  show_tooltips: true,
  token_delimiter: " ",
  label_position: "inline",
};

describe("computeTokenSpanMap", () => {
  test("returns empty lists for tokens with no spans", () => {
    const tokens = ["The", "cat", "sat"];
    const map = computeTokenSpanMap(tokens, []);

    expect(map.get(0)).toEqual([]);
    expect(map.get(1)).toEqual([]);
    expect(map.get(2)).toEqual([]);
  });

  test("maps single span to covered tokens", () => {
    const tokens = ["The", "cat", "sat"];
    const spans: SpanData[] = [
      {
        span_id: "span_0",
        segments: [{ element_name: "text", indices: [0, 1] }],
      },
    ];

    const map = computeTokenSpanMap(tokens, spans);

    expect(map.get(0)).toEqual(["span_0"]);
    expect(map.get(1)).toEqual(["span_0"]);
    expect(map.get(2)).toEqual([]);
  });

  test("handles overlapping spans", () => {
    const tokens = ["The", "big", "cat"];
    const spans: SpanData[] = [
      {
        span_id: "span_0",
        segments: [{ element_name: "text", indices: [0, 1] }],
      },
      {
        span_id: "span_1",
        segments: [{ element_name: "text", indices: [1, 2] }],
      },
    ];

    const map = computeTokenSpanMap(tokens, spans);

    expect(map.get(0)).toEqual(["span_0"]);
    expect(map.get(1)).toEqual(["span_0", "span_1"]);
    expect(map.get(2)).toEqual(["span_1"]);
  });

  test("filters by element name", () => {
    const tokens = ["Hello"];
    const spans: SpanData[] = [
      {
        span_id: "span_0",
        segments: [{ element_name: "context", indices: [0] }],
      },
    ];

    const map = computeTokenSpanMap(tokens, spans, "text");
    expect(map.get(0)).toEqual([]);
  });

  test("ignores out-of-bounds indices", () => {
    const tokens = ["Hello"];
    const spans: SpanData[] = [
      {
        span_id: "span_0",
        segments: [{ element_name: "text", indices: [0, 99] }],
      },
    ];

    const map = computeTokenSpanMap(tokens, spans);
    expect(map.get(0)).toEqual(["span_0"]);
    expect(map.has(99)).toBe(false);
  });
});

describe("assignSpanColors", () => {
  test("assigns colors from palette", () => {
    const spans: SpanData[] = [
      { span_id: "span_0", segments: [], label: { label: "Person" } },
      { span_id: "span_1", segments: [], label: { label: "Location" } },
    ];

    const colors = assignSpanColors(spans, ["#FF0000", "#00FF00"]);

    expect(colors.get("span_0")).toBe("#FF0000");
    expect(colors.get("span_1")).toBe("#00FF00");
  });

  test("reuses color for same label", () => {
    const spans: SpanData[] = [
      { span_id: "span_0", segments: [], label: { label: "Person" } },
      { span_id: "span_1", segments: [], label: { label: "Person" } },
    ];

    const colors = assignSpanColors(spans, ["#FF0000", "#00FF00"]);

    expect(colors.get("span_0")).toBe("#FF0000");
    expect(colors.get("span_1")).toBe("#FF0000");
  });

  test("cycles palette for many labels", () => {
    const spans: SpanData[] = [
      { span_id: "s0", segments: [], label: { label: "A" } },
      { span_id: "s1", segments: [], label: { label: "B" } },
      { span_id: "s2", segments: [], label: { label: "C" } },
    ];

    const colors = assignSpanColors(spans, ["#FF0000", "#00FF00"]);

    expect(colors.get("s0")).toBe("#FF0000");
    expect(colors.get("s1")).toBe("#00FF00");
    expect(colors.get("s2")).toBe("#FF0000"); // cycles
  });

  test("respects explicit label color overrides", () => {
    const spans: SpanData[] = [
      { span_id: "span_0", segments: [], label: { label: "Person" } },
      { span_id: "span_1", segments: [], label: { label: "Location" } },
    ];

    const colors = assignSpanColors(spans, ["#000000"], { Person: "#CUSTOM1" });

    expect(colors.get("span_0")).toBe("#CUSTOM1");
    expect(colors.get("span_1")).toBe("#000000"); // from palette
  });

  test("handles spans without labels", () => {
    const spans: SpanData[] = [
      { span_id: "span_0", segments: [] },
      { span_id: "span_1", segments: [] },
    ];

    const colors = assignSpanColors(spans, ["#FF0000", "#00FF00"]);

    expect(colors.get("span_0")).toBe("#FF0000");
    expect(colors.get("span_1")).toBe("#00FF00");
  });
});

describe("renderTokenizedText", () => {
  test("renders tokens as span elements", () => {
    const el = renderTokenizedText(["Hello", "world"], [true, false], [], DEFAULT_CONFIG);

    const tokens = el.querySelectorAll(".bead-token");
    expect(tokens).toHaveLength(2);
    expect(tokens[0]?.textContent).toBe("Hello");
    expect(tokens[1]?.textContent).toBe("world");
  });

  test("adds space after tokens with space_after=true", () => {
    const el = renderTokenizedText(["Hello", "world"], [true, false], [], DEFAULT_CONFIG);

    // Container should have: <span>Hello</span> " " <span>world</span>
    const textContent = el.textContent;
    expect(textContent).toContain("Hello");
    expect(textContent).toContain("world");
  });

  test("marks highlighted tokens with span data", () => {
    const spans: SpanData[] = [
      {
        span_id: "span_0",
        segments: [{ element_name: "text", indices: [0] }],
        label: { label: "Person" },
      },
    ];

    const el = renderTokenizedText(["John", "sat"], [true, false], spans, DEFAULT_CONFIG);

    const highlighted = el.querySelectorAll(".highlighted");
    expect(highlighted).toHaveLength(1);
    expect(highlighted[0]?.getAttribute("data-span-ids")).toBe("span_0");
    expect(highlighted[0]?.getAttribute("data-span-count")).toBe("1");
  });

  test("sets tooltip on highlighted tokens", () => {
    const spans: SpanData[] = [
      {
        span_id: "span_0",
        segments: [{ element_name: "text", indices: [0] }],
        label: { label: "Person" },
      },
    ];

    const el = renderTokenizedText(["John"], [false], spans, DEFAULT_CONFIG);

    const token = el.querySelector(".bead-token");
    expect(token?.getAttribute("title")).toBe("Person");
  });

  test("sets data-index on each token", () => {
    const el = renderTokenizedText(["a", "b", "c"], [true, true, false], [], DEFAULT_CONFIG);

    const tokens = el.querySelectorAll(".bead-token");
    expect(tokens[0]?.getAttribute("data-index")).toBe("0");
    expect(tokens[1]?.getAttribute("data-index")).toBe("1");
    expect(tokens[2]?.getAttribute("data-index")).toBe("2");
  });

  test("does not add space between tokens with space_after=false", () => {
    const el = renderTokenizedText(["don", "'t"], [false, false], [], DEFAULT_CONFIG);

    // Should be "don't" with no space
    const spans = el.querySelectorAll(".bead-token");
    expect(spans).toHaveLength(2);
    // No text node between them
    const firstToken = spans[0];
    const nextSibling = firstToken?.nextSibling;
    expect(nextSibling?.nodeName).toBe("SPAN"); // directly adjacent
  });
});

describe("renderRelationArcs", () => {
  test("creates SVG element", () => {
    const svg = renderRelationArcs([], new Map(), DEFAULT_CONFIG);

    expect(svg.tagName).toBe("svg");
    expect(svg.classList.contains("bead-relation-layer")).toBe(true);
  });

  test("includes arrowhead marker definition", () => {
    const svg = renderRelationArcs([], new Map(), DEFAULT_CONFIG);

    const marker = svg.querySelector("#arrowhead");
    expect(marker).not.toBeNull();
  });

  test("renders directed relation with marker-end", () => {
    const positions = new Map<string, DOMRect>();
    positions.set("span_0", new DOMRect(10, 50, 40, 20));
    positions.set("span_1", new DOMRect(100, 50, 40, 20));

    const relations = [
      {
        relation_id: "rel_0",
        source_span_id: "span_0",
        target_span_id: "span_1",
        label: { label: "agent-of" },
        directed: true,
      },
    ];

    const svg = renderRelationArcs(relations, positions, DEFAULT_CONFIG);

    const path = svg.querySelector("path");
    expect(path).not.toBeNull();
    expect(path?.classList.contains("directed")).toBe(true);
    expect(path?.getAttribute("marker-end")).toBe("url(#arrowhead)");
  });

  test("renders undirected relation without marker", () => {
    const positions = new Map<string, DOMRect>();
    positions.set("span_0", new DOMRect(10, 50, 40, 20));
    positions.set("span_1", new DOMRect(100, 50, 40, 20));

    const relations = [
      {
        relation_id: "rel_0",
        source_span_id: "span_0",
        target_span_id: "span_1",
        directed: false,
      },
    ];

    const svg = renderRelationArcs(relations, positions, DEFAULT_CONFIG);

    const path = svg.querySelector("path");
    expect(path).not.toBeNull();
    expect(path?.classList.contains("directed")).toBe(false);
    expect(path?.getAttribute("marker-end")).toBeNull();
  });

  test("renders relation label text", () => {
    const positions = new Map<string, DOMRect>();
    positions.set("span_0", new DOMRect(10, 50, 40, 20));
    positions.set("span_1", new DOMRect(100, 50, 40, 20));

    const relations = [
      {
        relation_id: "rel_0",
        source_span_id: "span_0",
        target_span_id: "span_1",
        label: { label: "agent-of" },
        directed: true,
      },
    ];

    const svg = renderRelationArcs(relations, positions, DEFAULT_CONFIG);

    const text = svg.querySelector("text");
    expect(text?.textContent).toBe("agent-of");
  });

  test("skips relations with missing span positions", () => {
    const positions = new Map<string, DOMRect>();
    positions.set("span_0", new DOMRect(10, 50, 40, 20));
    // span_1 missing

    const relations = [
      {
        relation_id: "rel_0",
        source_span_id: "span_0",
        target_span_id: "span_1",
        directed: true,
      },
    ];

    const svg = renderRelationArcs(relations, positions, DEFAULT_CONFIG);

    const paths = svg.querySelectorAll("path");
    expect(paths).toHaveLength(0);
  });
});
