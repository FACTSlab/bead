/**
 * Shared span rendering utilities.
 *
 * Provides functions for rendering tokenized text with span highlights,
 * assigning colors, computing token-span maps, and rendering relation arcs.
 *
 * @author Bead Project
 * @version 0.2.0
 */

/** Span data structure (matches Python Span model serialization) */
export interface SpanData {
  span_id: string;
  segments: Array<{
    element_name: string;
    indices: number[];
  }>;
  head_index?: number;
  label?: {
    label: string;
    label_id?: string;
  };
  span_type?: string;
}

/** Relation data structure */
export interface RelationData {
  relation_id: string;
  source_span_id: string;
  target_span_id: string;
  label?: {
    label: string;
    label_id?: string;
  };
  directed: boolean;
}

/** Display configuration */
export interface SpanDisplayConfig {
  highlight_style: "background" | "underline" | "border";
  color_palette: string[];
  show_labels: boolean;
  show_tooltips: boolean;
  token_delimiter: string;
  label_position: "inline" | "below" | "tooltip";
}

const DEFAULT_PALETTE = [
  "#BBDEFB",
  "#C8E6C9",
  "#FFE0B2",
  "#F8BBD0",
  "#D1C4E9",
  "#B2EBF2",
  "#DCEDC8",
  "#FFD54F",
];

/**
 * Compute which spans cover each token index.
 *
 * @param tokens Token array for a single element
 * @param spans All span data
 * @param elementName Name of the element to compute for
 * @returns Map from token index to list of span_ids covering that token
 */
export function computeTokenSpanMap(
  tokens: string[],
  spans: SpanData[],
  elementName = "text",
): Map<number, string[]> {
  const map: Map<number, string[]> = new Map();

  for (let i = 0; i < tokens.length; i++) {
    map.set(i, []);
  }

  for (const span of spans) {
    for (const segment of span.segments) {
      if (segment.element_name === elementName) {
        for (const idx of segment.indices) {
          if (idx < tokens.length) {
            const list = map.get(idx);
            if (list) {
              list.push(span.span_id);
            }
          }
        }
      }
    }
  }

  return map;
}

/**
 * Assign colors to spans from palette, respecting per-label overrides.
 *
 * @param spans Span data array
 * @param palette Color palette
 * @param labelColors Optional per-label color overrides
 * @returns Map from span_id to CSS color string
 */
export function assignSpanColors(
  spans: SpanData[],
  palette: string[] = DEFAULT_PALETTE,
  labelColors?: Record<string, string>,
): Map<string, string> {
  const colorMap: Map<string, string> = new Map();
  const labelToColor: Map<string, string> = new Map();
  let colorIdx = 0;

  for (const span of spans) {
    const label = span.label?.label;

    // Check for explicit label color
    if (label && labelColors?.[label]) {
      colorMap.set(span.span_id, labelColors[label] ?? palette[0] ?? "#BBDEFB");
      continue;
    }

    // Reuse color for same label
    if (label && labelToColor.has(label)) {
      colorMap.set(span.span_id, labelToColor.get(label) ?? palette[0] ?? "#BBDEFB");
      continue;
    }

    // Assign next color from palette
    const color = palette[colorIdx % palette.length] ?? "#BBDEFB";
    colorMap.set(span.span_id, color);
    if (label) {
      labelToColor.set(label, color);
    }
    colorIdx++;
  }

  return colorMap;
}

/**
 * Render tokenized text into a DOM element with correct spacing and span highlights.
 *
 * @param tokens Token strings for a single element
 * @param spaceAfter Per-token space_after flags
 * @param spans Span data
 * @param config Display configuration
 * @param elementName Element name for span matching
 * @returns Container HTMLElement with highlighted tokens
 */
export function renderTokenizedText(
  tokens: string[],
  spaceAfter: boolean[],
  spans: SpanData[],
  config: SpanDisplayConfig,
  elementName = "text",
): HTMLElement {
  const container = document.createElement("div");
  container.className = "bead-span-container";
  container.setAttribute("data-element", elementName);

  const tokenSpanMap = computeTokenSpanMap(tokens, spans, elementName);
  const colorMap = assignSpanColors(spans, config.color_palette);

  for (let i = 0; i < tokens.length; i++) {
    const tokenEl = document.createElement("span");
    tokenEl.className = "bead-token";
    tokenEl.textContent = tokens[i] ?? "";
    tokenEl.setAttribute("data-index", String(i));
    tokenEl.setAttribute("data-element", elementName);

    const spanIds = tokenSpanMap.get(i) ?? [];
    if (spanIds.length > 0) {
      tokenEl.classList.add("highlighted");
      tokenEl.setAttribute("data-span-count", String(spanIds.length));
      tokenEl.setAttribute("data-span-ids", spanIds.join(","));

      // Apply color
      if (config.highlight_style === "background") {
        if (spanIds.length === 1) {
          tokenEl.style.backgroundColor = colorMap.get(spanIds[0] ?? "") ?? "#BBDEFB";
        } else {
          const colors = spanIds.map((id) => colorMap.get(id) ?? "#BBDEFB");
          tokenEl.style.background = `linear-gradient(${colors.join(", ")})`;
        }
      } else if (config.highlight_style === "underline") {
        const color = colorMap.get(spanIds[0] ?? "") ?? "#BBDEFB";
        tokenEl.style.textDecoration = "underline";
        tokenEl.style.textDecorationColor = color;
      } else if (config.highlight_style === "border") {
        const color = colorMap.get(spanIds[0] ?? "") ?? "#BBDEFB";
        tokenEl.style.border = `1px solid ${color}`;
      }

      // Tooltip
      if (config.show_tooltips && spanIds.length > 0) {
        const labels = spanIds
          .map((id) => {
            const span = spans.find((s) => s.span_id === id);
            return span?.label?.label ?? id;
          })
          .join(", ");
        tokenEl.title = labels;
      }
    }

    container.appendChild(tokenEl);

    // Add spacing
    if (i < spaceAfter.length && spaceAfter[i]) {
      container.appendChild(document.createTextNode(" "));
    }
  }

  return container;
}

/**
 * Render relation arcs as an SVG overlay.
 *
 * @param relations Relation data
 * @param spanPositions Map from span_id to bounding rect
 * @param config Display configuration
 * @returns SVG element with relation arcs
 */
export function renderRelationArcs(
  relations: RelationData[],
  spanPositions: Map<string, DOMRect>,
  config: SpanDisplayConfig,
): SVGSVGElement {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.classList.add("bead-relation-layer");
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", "100%");
  svg.style.position = "absolute";
  svg.style.top = "0";
  svg.style.left = "0";
  svg.style.pointerEvents = "none";

  // Arrowhead marker for directed relations
  const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
  const marker = document.createElementNS("http://www.w3.org/2000/svg", "marker");
  marker.setAttribute("id", "arrowhead");
  marker.setAttribute("markerWidth", "10");
  marker.setAttribute("markerHeight", "7");
  marker.setAttribute("refX", "10");
  marker.setAttribute("refY", "3.5");
  marker.setAttribute("orient", "auto");
  const polygon = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
  polygon.setAttribute("points", "0 0, 10 3.5, 0 7");
  polygon.setAttribute("fill", "#424242");
  marker.appendChild(polygon);
  defs.appendChild(marker);
  svg.appendChild(defs);

  const palette = config.color_palette.length > 0 ? config.color_palette : DEFAULT_PALETTE;

  for (let i = 0; i < relations.length; i++) {
    const rel = relations[i];
    if (!rel) continue;

    const sourceRect = spanPositions.get(rel.source_span_id);
    const targetRect = spanPositions.get(rel.target_span_id);
    if (!sourceRect || !targetRect) continue;

    const x1 = sourceRect.left + sourceRect.width / 2;
    const x2 = targetRect.left + targetRect.width / 2;
    const y1 = sourceRect.top;
    const y2 = targetRect.top;

    // Draw arc
    const midX = (x1 + x2) / 2;
    const arcHeight = Math.abs(x2 - x1) * 0.3 + 20;
    const midY = Math.min(y1, y2) - arcHeight;

    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", `M ${x1} ${y1} Q ${midX} ${midY} ${x2} ${y2}`);
    path.classList.add("bead-relation-arc");
    path.setAttribute("stroke", palette[i % palette.length] ?? "#424242");
    path.setAttribute("fill", "none");
    path.setAttribute("stroke-width", "1.5");

    if (rel.directed) {
      path.classList.add("directed");
      path.setAttribute("marker-end", "url(#arrowhead)");
    }

    svg.appendChild(path);

    // Label text
    if (rel.label?.label) {
      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("x", String(midX));
      text.setAttribute("y", String(midY - 4));
      text.setAttribute("text-anchor", "middle");
      text.classList.add("bead-relation-label-text");
      text.textContent = rel.label.label;
      svg.appendChild(text);
    }
  }

  return svg;
}
