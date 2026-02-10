/**
 * bead-span-label plugin
 *
 * jsPsych plugin for span selection and labeling. Supports:
 * - Token-level span selection (click, drag, shift+click)
 * - Static span display with overlapping highlights
 * - Fixed label sets and Wikidata entity search
 * - Relation annotation between spans
 * - Keyboard shortcuts for labels (1-9)
 *
 * @author Bead Project
 * @version 0.2.0
 */

import { debouncedSearchWikidata } from "../lib/wikidata-search.js";
import type { WikidataEntity } from "../lib/wikidata-search.js";
import type { JsPsych, JsPsychPlugin, PluginInfo } from "../types/jspsych.js";

/** Span segment data */
interface SpanSegmentData {
  element_name: string;
  indices: number[];
}

/** Span label data */
interface SpanLabelData {
  label: string;
  label_id?: string;
}

/** Span data */
interface SpanData {
  span_id: string;
  segments: SpanSegmentData[];
  head_index?: number;
  label?: SpanLabelData;
  span_type?: string;
}

/** Relation data */
interface RelationData {
  relation_id: string;
  source_span_id: string;
  target_span_id: string;
  label?: SpanLabelData;
  directed: boolean;
}

/** Span specification from Python */
interface SpanSpecData {
  index_mode: "token" | "character";
  interaction_mode: "static" | "interactive";
  label_source: "fixed" | "wikidata";
  labels?: string[];
  label_colors?: Record<string, string>;
  allow_overlapping: boolean;
  min_spans?: number;
  max_spans?: number;
  enable_relations: boolean;
  relation_label_source: "fixed" | "wikidata";
  relation_labels?: string[];
  relation_directed: boolean;
  min_relations?: number;
  max_relations?: number;
  wikidata_language: string;
  wikidata_entity_types?: string[];
  wikidata_result_limit: number;
}

/** Display configuration */
interface SpanDisplayConfigData {
  highlight_style: "background" | "underline" | "border";
  color_palette: string[];
  show_labels: boolean;
  show_tooltips: boolean;
  token_delimiter: string;
  label_position: "inline" | "below" | "tooltip";
}

/** Bead metadata */
interface BeadMetadata {
  spans?: SpanData[];
  span_relations?: RelationData[];
  tokenized_elements?: Record<string, string[]>;
  token_space_after?: Record<string, boolean[]>;
  span_spec?: SpanSpecData;
  [key: string]: unknown;
}

/** Span event for logging */
interface SpanEvent {
  type: "select" | "deselect" | "label" | "delete" | "relation_create" | "relation_delete";
  timestamp: number;
  span_id?: string;
  relation_id?: string;
  indices?: number[];
  label?: string;
}

/** Trial parameters */
export interface SpanLabelTrialParams {
  tokens: Record<string, string[]>;
  space_after: Record<string, boolean[]>;
  spans: SpanData[];
  relations: RelationData[];
  span_spec: SpanSpecData | null;
  display_config: SpanDisplayConfigData | null;
  prompt: string;
  button_label: string;
  require_response: boolean;
  metadata: BeadMetadata;
}

/** Plugin info constant */
const info: PluginInfo = {
  name: "bead-span-label",
  parameters: {
    tokens: {
      type: 12, // OBJECT
      default: {},
    },
    space_after: {
      type: 12, // OBJECT
      default: {},
    },
    spans: {
      type: 12, // OBJECT
      default: [],
      array: true,
    },
    relations: {
      type: 12, // OBJECT
      default: [],
      array: true,
    },
    span_spec: {
      type: 12, // OBJECT
      default: null,
    },
    display_config: {
      type: 12, // OBJECT
      default: null,
    },
    prompt: {
      type: 8, // HTML_STRING
      default: "Select and label spans",
    },
    button_label: {
      type: 1, // STRING
      default: "Continue",
    },
    require_response: {
      type: 0, // BOOL
      default: true,
    },
    metadata: {
      type: 12, // OBJECT
      default: {},
    },
  },
};

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

// Dark versions of palette colors for badge backgrounds (white text)
const DARK_PALETTE = [
  "#1565C0",
  "#2E7D32",
  "#E65100",
  "#AD1457",
  "#4527A0",
  "#00838F",
  "#558B2F",
  "#F9A825",
];

/**
 * BeadSpanLabelPlugin - jsPsych plugin for span annotation
 */
class BeadSpanLabelPlugin implements JsPsychPlugin<typeof info, SpanLabelTrialParams> {
  static info = info;

  private jsPsych: JsPsych;

  constructor(jsPsych: JsPsych) {
    this.jsPsych = jsPsych;
  }

  trial(display_element: HTMLElement, trial: SpanLabelTrialParams): void {
    const start_time = performance.now();
    const events: SpanEvent[] = [];

    // Resolve config from metadata or parameters
    const tokens =
      Object.keys(trial.tokens).length > 0
        ? trial.tokens
        : (trial.metadata.tokenized_elements ?? {});
    const spaceAfter =
      Object.keys(trial.space_after).length > 0
        ? trial.space_after
        : (trial.metadata.token_space_after ?? {});
    const spanSpec = trial.span_spec ?? trial.metadata.span_spec ?? null;
    const preSpans = trial.spans.length > 0 ? trial.spans : (trial.metadata.spans ?? []);
    const preRelations =
      trial.relations.length > 0 ? trial.relations : (trial.metadata.span_relations ?? []);

    const palette = trial.display_config?.color_palette ?? DEFAULT_PALETTE;
    const isInteractive = spanSpec?.interaction_mode === "interactive";

    // Working state
    const activeSpans: SpanData[] = [...preSpans];
    const activeRelations: RelationData[] = [...preRelations];
    let selectionStart: number | null = null;
    let selectedIndices: number[] = [];
    let nextSpanId = activeSpans.length;
    let nextRelationId = activeRelations.length;

    // Relation creation state machine
    type RelationState = "IDLE" | "WAITING_SOURCE" | "WAITING_TARGET" | "WAITING_LABEL";
    let relationState: RelationState = "IDLE";
    let relationSource: string | null = null;
    let relationTarget: string | null = null;

    // Build HTML
    let html = '<div class="bead-span-label-container">';

    if (trial.prompt) {
      html += `<div class="bead-rating-prompt">${trial.prompt}</div>`;
    }

    // Render tokens for each element
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

    // Label selector (for interactive mode)
    if (isInteractive && spanSpec?.label_source === "wikidata") {
      // Wikidata search panel
      html +=
        '<div class="bead-label-selector bead-wikidata-panel bead-search-disabled" id="bead-label-panel">';
      html += '<div class="bead-wikidata-search">';
      html +=
        '<input type="text" id="bead-wikidata-input" placeholder="Select tokens to annotate..." autocomplete="off" disabled>';
      html +=
        '<button type="button" class="bead-search-cancel" id="bead-search-cancel" style="display:none;" title="Cancel annotation">&times;</button>';
      html +=
        '<div class="bead-wikidata-results" id="bead-wikidata-results" style="display:none;"></div>';
      html += "</div></div>";
    } else if (isInteractive && spanSpec?.labels && spanSpec.labels.length > 0) {
      // Searchable fixed label panel (mirrors the Wikidata UX)
      html +=
        '<div class="bead-label-selector bead-label-search-panel bead-search-disabled" id="bead-label-panel">';
      html += '<div class="bead-label-search-wrapper">';
      html +=
        '<input type="text" id="bead-label-search-input" placeholder="Select tokens to annotate..." autocomplete="off" disabled>';
      html +=
        '<button type="button" class="bead-search-cancel" id="bead-search-cancel" style="display:none;" title="Cancel annotation">&times;</button>';
      html +=
        '<div class="bead-label-search-results" id="bead-label-search-results" style="display:none;"></div>';
      html += "</div></div>";
    }

    // Relation controls and list
    if (spanSpec?.enable_relations) {
      if (isInteractive) {
        html += '<div class="bead-relation-controls" id="bead-relation-controls">';
        html +=
          '<button class="bead-add-relation-button" id="bead-add-relation" disabled>Add Relation</button>';
        html += '<span class="bead-relation-status" id="bead-relation-status"></span>';
        html +=
          '<button class="bead-relation-cancel" id="bead-relation-cancel" style="display:none;">Cancel</button>';
        html += "</div>";
        // Relation label search (for choosing the label after source+target)
        if (spanSpec.relation_label_source === "wikidata") {
          html +=
            '<div class="bead-label-selector bead-wikidata-panel" id="bead-relation-label-panel" style="display:none;">';
          html += '<div class="bead-wikidata-search">';
          html +=
            '<input type="text" id="bead-relation-wikidata-input" placeholder="Search Wikidata for relation label..." autocomplete="off">';
          html +=
            '<div class="bead-wikidata-results" id="bead-relation-wikidata-results" style="display:none;"></div>';
          html += "</div></div>";
        } else if (spanSpec.relation_labels && spanSpec.relation_labels.length > 0) {
          html +=
            '<div class="bead-label-selector bead-label-search-panel" id="bead-relation-label-panel" style="display:none;">';
          html += '<div class="bead-label-search-wrapper">';
          html +=
            '<input type="text" id="bead-relation-label-input" placeholder="Search relation labels..." autocomplete="off">';
          html +=
            '<div class="bead-label-search-results" id="bead-relation-label-results" style="display:none;"></div>';
          html += "</div></div>";
        }
      }
      html += '<div class="bead-relation-list" id="bead-relation-list"></div>';
    }

    // Bottom bar with continue button
    html += '<div class="bead-span-bottom-bar">';
    html += `<div class="bead-span-bottom-spacer"></div>`;
    html += `<button class="bead-button bead-continue-button" id="bead-span-continue" ${isInteractive && trial.require_response ? "disabled" : ""}>`;
    html += `${trial.button_label}`;
    html += "</button>";
    html += "</div>";

    html += "</div>";
    display_element.innerHTML = html;

    // Apply static span highlights
    applySpanHighlights();

    // Render span list
    renderSpanList();

    if (isInteractive) {
      setupInteractiveHandlers();
      if (spanSpec?.label_source === "wikidata") {
        setupWikidataSearch();
      } else if (spanSpec?.labels && spanSpec.labels.length > 0) {
        setupFixedLabelSearch();
      }
      // Cancel button handler
      const searchCancelBtn =
        display_element.querySelector<HTMLButtonElement>("#bead-search-cancel");
      if (searchCancelBtn) {
        searchCancelBtn.addEventListener("click", () => {
          cancelCurrentSelection();
        });
      }

      if (spanSpec?.enable_relations) {
        setupRelationHandlers();
      }
    }

    // Render relation arcs for pre-existing relations
    renderRelationArcsOverlay();
    renderRelationList();

    // Continue button
    const continueBtn = display_element.querySelector<HTMLButtonElement>("#bead-span-continue");
    if (continueBtn) {
      continueBtn.addEventListener("click", () => {
        endTrial();
      });
    }

    function applySpanHighlights(): void {
      // Clear existing highlights on tokens
      const allTokens = display_element.querySelectorAll<HTMLSpanElement>(".bead-token");
      for (const t of allTokens) {
        t.classList.remove("highlighted", "span-first", "span-middle", "span-last", "span-single");
        t.removeAttribute("data-span-ids");
        t.removeAttribute("data-span-count");
        t.style.removeProperty("background-color");
        t.style.removeProperty("background");
      }

      // Clear space highlights
      const allSpaces = display_element.querySelectorAll<HTMLSpanElement>(".bead-space");
      for (const s of allSpaces) {
        s.classList.remove("highlighted");
        s.style.removeProperty("background-color");
        s.style.removeProperty("background");
      }

      // Build token -> span mapping
      const tokenSpanMap: Map<string, string[]> = new Map();
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

      // Apply highlights to tokens
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

      // Determine contiguous position classes and highlight spaces globally.
      // For each highlighted token, check if ANY span bridges it to its neighbor.
      for (const elemName of elementNames) {
        const elemTokens = tokens[elemName] ?? [];
        for (let i = 0; i < elemTokens.length; i++) {
          const key = `${elemName}:${i}`;
          const spanIds = tokenSpanMap.get(key) ?? [];
          if (spanIds.length === 0) continue;

          const t = display_element.querySelector<HTMLSpanElement>(
            `.bead-token[data-element="${elemName}"][data-index="${i}"]`,
          );
          if (!t) continue;

          // Check if any span covers both this token and its left neighbor
          const leftKey = `${elemName}:${i - 1}`;
          const leftSpanIds = tokenSpanMap.get(leftKey) ?? [];
          const hasLeftNeighbor = spanIds.some((id) => leftSpanIds.includes(id));

          // Check if any span covers both this token and its right neighbor
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

          // Highlight the space between this token and right neighbor if bridged
          if (hasRightNeighbor) {
            const spaceEl = display_element.querySelector<HTMLSpanElement>(
              `.bead-space[data-element="${elemName}"][data-after="${i}"]`,
            );
            if (spaceEl) {
              spaceEl.classList.add("highlighted");
              // Use the shared span IDs for the space color
              const sharedIds = spanIds.filter((id) => rightSpanIds.includes(id));
              applySpanColor(spaceEl, sharedIds.length > 0 ? sharedIds : spanIds, spanColorMap);
            }
          }
        }
      }
    }

    function applySpanColor(
      el: HTMLElement,
      spanIds: string[],
      colorMap: Map<string, string>,
    ): void {
      if (spanIds.length === 1) {
        el.style.backgroundColor = colorMap.get(spanIds[0] ?? "") ?? palette[0] ?? "#BBDEFB";
      } else if (spanIds.length > 1) {
        // For overlapping spans, use striped gradient
        const colors = spanIds.map((id) => colorMap.get(id) ?? palette[0] ?? "#BBDEFB");
        const stripeWidth = 100 / colors.length;
        const stops = colors
          .map((c, ci) => `${c} ${ci * stripeWidth}%, ${c} ${(ci + 1) * stripeWidth}%`)
          .join(", ");
        el.style.background = `linear-gradient(135deg, ${stops})`;
      }
    }

    function assignColors(): Map<string, string> {
      const colorMap: Map<string, string> = new Map();
      const labelColors = spanSpec?.label_colors ?? {};
      const labelToColor: Map<string, string> = new Map();
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

    function renderSpanList(): void {
      // Remove existing subscript labels
      const existing = display_element.querySelectorAll(".bead-span-subscript");
      for (const el of existing) el.remove();

      const darkColorMap = assignDarkColors();

      for (const span of activeSpans) {
        if (!span.label?.label) continue;

        // Find the last token of this span to position the label
        const allIndices: Array<{ elem: string; idx: number }> = [];
        for (const seg of span.segments) {
          for (const idx of seg.indices) {
            allIndices.push({ elem: seg.element_name, idx });
          }
        }
        if (allIndices.length === 0) continue;

        // Use the last token in reading order
        const lastToken = allIndices[allIndices.length - 1];
        if (!lastToken) continue;
        const tokenEl = display_element.querySelector<HTMLElement>(
          `.bead-token[data-element="${lastToken.elem}"][data-index="${lastToken.idx}"]`,
        );
        if (!tokenEl) continue;

        // Make the token position: relative so we can position the badge
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
          deleteBtn.textContent = "\u00d7";
          deleteBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            deleteSpan(span.span_id);
          });
          badge.appendChild(deleteBtn);
        }

        tokenEl.appendChild(badge);
      }

      // Resolve overlapping subscript badges
      adjustSubscriptPositions();
    }

    function adjustSubscriptPositions(): void {
      const badges = Array.from(
        display_element.querySelectorAll<HTMLElement>(".bead-span-subscript"),
      );
      if (badges.length < 2) return;

      // Reset previous adjustments
      for (const b of badges) b.style.transform = "";

      // Sort left-to-right by position
      badges.sort((a, b) => a.getBoundingClientRect().left - b.getBoundingClientRect().left);

      // Place badges one by one, shifting down if overlapping any already-placed badge
      const placed: Array<{ el: HTMLElement; rect: DOMRect }> = [];

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

    function assignDarkColors(): Map<string, string> {
      const colorMap: Map<string, string> = new Map();
      let colorIdx = 0;
      const labelToColor: Map<string, string> = new Map();

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

    function getSpanText(span: SpanData): string {
      const parts: string[] = [];
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

    function setupInteractiveHandlers(): void {
      const tokenEls = display_element.querySelectorAll<HTMLSpanElement>(".bead-token.interactive");
      let isDragging = false;
      let dragStartIdx: number | null = null;
      let dragElemName: string | null = null;

      for (const tokenEl of tokenEls) {
        // Mousedown starts a drag
        tokenEl.addEventListener("mousedown", (e) => {
          e.preventDefault(); // Prevent text selection
          const idx = Number.parseInt(tokenEl.getAttribute("data-index") ?? "0", 10);
          const elemName = tokenEl.getAttribute("data-element") ?? "";

          isDragging = true;
          dragStartIdx = idx;
          dragElemName = elemName;

          if (e.shiftKey && selectionStart !== null) {
            // Range selection from previous anchor
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

        // Mouseover extends drag
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

      // Mouseup ends drag
      document.addEventListener("mouseup", () => {
        if (isDragging) {
          isDragging = false;
          showLabelPanel();
        }
      });

      // Label button handlers
      const labelButtons =
        display_element.querySelectorAll<HTMLButtonElement>(".bead-label-button");
      for (const btn of labelButtons) {
        btn.addEventListener("click", () => {
          const label = btn.getAttribute("data-label") ?? "";
          if (selectedIndices.length > 0 && label) {
            createSpanFromSelection(label);
          }
        });
      }

      // Keyboard shortcuts for labels
      document.addEventListener("keydown", handleKeyDown);
    }

    function cancelCurrentSelection(): void {
      selectedIndices = [];
      selectionStart = null;
      // Remove selecting class from all tokens
      const allTokens = display_element.querySelectorAll<HTMLSpanElement>(".bead-token");
      for (const t of allTokens) {
        t.classList.remove("selecting");
      }
      // Disable search panel
      const labelPanel = display_element.querySelector<HTMLElement>("#bead-label-panel");
      if (labelPanel) {
        labelPanel.classList.add("bead-search-disabled");
        const searchInput = labelPanel.querySelector<HTMLInputElement>("input");
        if (searchInput) {
          searchInput.disabled = true;
          searchInput.value = "";
          searchInput.placeholder = "Select tokens to annotate...";
        }
        const resultsDiv = labelPanel.querySelector<HTMLElement>(
          ".bead-label-search-results, .bead-wikidata-results",
        );
        if (resultsDiv) resultsDiv.style.display = "none";
        const cancelBtn = labelPanel.querySelector<HTMLElement>(".bead-search-cancel");
        if (cancelBtn) cancelBtn.style.display = "none";
      }
    }

    function showLabelPanel(): void {
      const labelPanel = display_element.querySelector<HTMLElement>("#bead-label-panel");
      if (!labelPanel) return;

      const hasSelection = selectedIndices.length > 0;
      if (hasSelection) {
        labelPanel.classList.remove("bead-search-disabled");
        const searchInput = labelPanel.querySelector<HTMLInputElement>("input");
        if (searchInput) {
          searchInput.disabled = false;
          searchInput.placeholder = "Search labels...";
          setTimeout(() => searchInput.focus(), 0);
        }
        const cancelBtn = labelPanel.querySelector<HTMLElement>(".bead-search-cancel");
        if (cancelBtn) cancelBtn.style.display = "";
      } else {
        cancelCurrentSelection();
      }
    }

    function handleKeyDown(e: KeyboardEvent): void {
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

    function updateSelectionUI(elementName: string): void {
      const tokenEls = display_element.querySelectorAll<HTMLSpanElement>(
        `.bead-token[data-element="${elementName}"]`,
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

    function createSpanFromSelection(label: string, labelId?: string): void {
      const elemName = elementNames[0] ?? "text";
      const spanId = `span_${nextSpanId++}`;

      const spanLabel: SpanLabelData = labelId ? { label, label_id: labelId } : { label };

      const newSpan: SpanData = {
        span_id: spanId,
        segments: [
          {
            element_name: elemName,
            indices: [...selectedIndices].sort((a, b) => a - b),
          },
        ],
        label: spanLabel,
      };

      activeSpans.push(newSpan);
      events.push({
        type: "select",
        timestamp: performance.now() - start_time,
        span_id: spanId,
        indices: [...selectedIndices],
        label,
      });

      // Clear selection
      selectedIndices = [];
      selectionStart = null;

      // Update UI
      applySpanHighlights();
      renderSpanList();
      renderRelationList();
      updateContinueButton();

      // Reset search panel to disabled state
      cancelCurrentSelection();
    }

    function deleteSpan(spanId: string): void {
      const idx = activeSpans.findIndex((s) => s.span_id === spanId);
      if (idx >= 0) {
        activeSpans.splice(idx, 1);
        // Also remove any relations involving this span
        for (let ri = activeRelations.length - 1; ri >= 0; ri--) {
          const rel = activeRelations[ri];
          if (rel && (rel.source_span_id === spanId || rel.target_span_id === spanId)) {
            activeRelations.splice(ri, 1);
          }
        }
        events.push({
          type: "delete",
          timestamp: performance.now() - start_time,
          span_id: spanId,
        });
        applySpanHighlights();
        renderSpanList();
        renderRelationList();
        updateContinueButton();
      }
    }

    function setupWikidataSearch(): void {
      const input = display_element.querySelector<HTMLInputElement>("#bead-wikidata-input");
      const resultsDiv = display_element.querySelector<HTMLDivElement>("#bead-wikidata-results");
      if (!input || !resultsDiv) return;

      const searchOptions = {
        language: spanSpec?.wikidata_language ?? "en",
        limit: spanSpec?.wikidata_result_limit ?? 10,
        ...(spanSpec?.wikidata_entity_types ? { entityTypes: spanSpec.wikidata_entity_types } : {}),
      };

      input.addEventListener("input", () => {
        const query = input.value.trim();
        if (query.length === 0) {
          resultsDiv.style.display = "none";
          resultsDiv.innerHTML = "";
          return;
        }

        debouncedSearchWikidata(query, searchOptions, (results: WikidataEntity[]) => {
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

    function setupFixedLabelSearch(): void {
      const input = display_element.querySelector<HTMLInputElement>("#bead-label-search-input");
      const resultsDiv = display_element.querySelector<HTMLDivElement>(
        "#bead-label-search-results",
      );
      if (!input || !resultsDiv) return;

      const allLabels = spanSpec?.labels ?? [];
      let highlightedIdx = -1;

      function fuzzyMatch(query: string, target: string): boolean {
        const q = query.toLowerCase();
        const t = target.toLowerCase();
        let qi = 0;
        for (let ti = 0; ti < t.length && qi < q.length; ti++) {
          if (t[ti] === q[qi]) qi++;
        }
        return qi === q.length;
      }

      const renderResults = (query: string): void => {
        resultsDiv.innerHTML = "";
        const lower = query.toLowerCase();
        const filtered =
          lower === ""
            ? allLabels
            : allLabels
                .filter((l) => fuzzyMatch(lower, l))
                .sort((a, b) => {
                  // Prefix matches first, then fuzzy matches
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
          const color = palette[globalIdx % palette.length] ?? "#BBDEFB";
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

      // Show all labels on focus
      input.addEventListener("focus", () => {
        if (selectedIndices.length > 0) {
          renderResults(input.value);
        }
      });

      input.addEventListener("input", () => {
        renderResults(input.value);
      });

      // Keyboard navigation within the search results
      input.addEventListener("keydown", (e: KeyboardEvent) => {
        const items = resultsDiv.querySelectorAll<HTMLElement>(".bead-label-search-result");
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

      function updateHighlight(items: NodeListOf<HTMLElement>): void {
        for (let i = 0; i < items.length; i++) {
          items[i]?.classList.toggle("highlighted", i === highlightedIdx);
        }
        items[highlightedIdx]?.scrollIntoView({ block: "nearest" });
      }

      // Close results when clicking outside
      document.addEventListener("click", (e) => {
        if (!input.contains(e.target as Node) && !resultsDiv.contains(e.target as Node)) {
          resultsDiv.style.display = "none";
        }
      });
    }

    function setupRelationHandlers(): void {
      const addBtn = display_element.querySelector<HTMLButtonElement>("#bead-add-relation");
      const cancelBtn = display_element.querySelector<HTMLButtonElement>("#bead-relation-cancel");
      const statusEl = display_element.querySelector<HTMLSpanElement>("#bead-relation-status");

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

      function updateRelationUI(): void {
        if (!addBtn || !cancelBtn || !statusEl) return;

        // Update Add Relation button
        addBtn.disabled = relationState !== "IDLE" || activeSpans.length < 2;
        addBtn.style.display = relationState === "IDLE" ? "" : "none";
        cancelBtn.style.display = relationState !== "IDLE" ? "" : "none";

        // Status text
        if (relationState === "WAITING_SOURCE") {
          statusEl.textContent = "Click a span label to select the source.";
        } else if (relationState === "WAITING_TARGET") {
          statusEl.textContent = "Click a span label to select the target.";
        } else if (relationState === "WAITING_LABEL") {
          statusEl.textContent = "Choose a relation label.";
        } else {
          statusEl.textContent = "";
        }

        // Visual feedback on span badges
        const badges = display_element.querySelectorAll<HTMLElement>(".bead-span-subscript");
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

        // Show/hide relation label panel
        const labelPanel = display_element.querySelector<HTMLElement>("#bead-relation-label-panel");
        if (labelPanel) {
          labelPanel.style.display = relationState === "WAITING_LABEL" ? "flex" : "none";
          if (relationState === "WAITING_LABEL") {
            const searchInput = labelPanel.querySelector<HTMLInputElement>("input");
            if (searchInput) setTimeout(() => searchInput.focus(), 0);
          }
        }
      }

      // Expose updateRelationUI so it's called after span changes
      (display_element as unknown as Record<string, unknown>)["_updateRelationUI"] =
        updateRelationUI;

      // Click handler for span badges (delegated)
      display_element.addEventListener("click", (e) => {
        const badge = (e.target as HTMLElement).closest<HTMLElement>(".bead-span-subscript");
        if (!badge) return;
        const spanId = badge.getAttribute("data-span-id");
        if (!spanId) return;

        if (relationState === "WAITING_SOURCE") {
          relationSource = spanId;
          relationState = "WAITING_TARGET";
          updateRelationUI();
        } else if (relationState === "WAITING_TARGET") {
          if (spanId === relationSource) return; // Can't self-relate
          relationTarget = spanId;
          relationState = "WAITING_LABEL";
          updateRelationUI();
          // If no labels configured, create with no label
          if (
            !spanSpec?.relation_labels?.length &&
            spanSpec?.relation_label_source !== "wikidata"
          ) {
            createRelation(undefined);
          }
        }
      });

      // Setup relation label search (fixed labels)
      if (
        spanSpec?.relation_labels &&
        spanSpec.relation_labels.length > 0 &&
        spanSpec.relation_label_source !== "wikidata"
      ) {
        setupRelationLabelSearch();
      }

      // Setup relation label search (wikidata)
      if (spanSpec?.relation_label_source === "wikidata") {
        setupRelationWikidataSearch();
      }

      function setupRelationLabelSearch(): void {
        const input = display_element.querySelector<HTMLInputElement>("#bead-relation-label-input");
        const resultsDiv = display_element.querySelector<HTMLDivElement>(
          "#bead-relation-label-results",
        );
        if (!input || !resultsDiv) return;

        const allLabels = spanSpec?.relation_labels ?? [];
        let highlightedIdx = -1;

        const renderResults = (query: string): void => {
          resultsDiv.innerHTML = "";
          const lower = query.toLowerCase();
          const filtered =
            lower === "" ? allLabels : allLabels.filter((l) => l.toLowerCase().includes(lower));

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

        input.addEventListener("keydown", (e: KeyboardEvent) => {
          const items = resultsDiv.querySelectorAll<HTMLElement>(".bead-label-search-result");
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

      function setupRelationWikidataSearch(): void {
        const input = display_element.querySelector<HTMLInputElement>(
          "#bead-relation-wikidata-input",
        );
        const resultsDiv = display_element.querySelector<HTMLDivElement>(
          "#bead-relation-wikidata-results",
        );
        if (!input || !resultsDiv) return;

        const searchOptions = {
          language: spanSpec?.wikidata_language ?? "en",
          limit: spanSpec?.wikidata_result_limit ?? 10,
          entityTypes: ["property"] as string[],
        };

        input.addEventListener("input", () => {
          const query = input.value.trim();
          if (query.length === 0) {
            resultsDiv.style.display = "none";
            resultsDiv.innerHTML = "";
            return;
          }

          debouncedSearchWikidata(query, searchOptions, (results: WikidataEntity[]) => {
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

      function createRelation(label: SpanLabelData | undefined): void {
        if (!relationSource || !relationTarget) return;

        const relId = `rel_${nextRelationId++}`;
        const newRelation: RelationData = {
          relation_id: relId,
          source_span_id: relationSource,
          target_span_id: relationTarget,
          ...(label !== undefined ? { label } : {}),
          directed: spanSpec?.relation_directed ?? true,
        };

        activeRelations.push(newRelation);
        events.push({
          type: "relation_create",
          timestamp: performance.now() - start_time,
          relation_id: relId,
          ...(label?.label !== undefined ? { label: label.label } : {}),
        });

        relationState = "IDLE";
        relationSource = null;
        relationTarget = null;

        renderRelationArcsOverlay();
        renderRelationList();
        updateRelationUI();
        updateContinueButton();
      }

      function cancelRelationCreation(): void {
        relationState = "IDLE";
        relationSource = null;
        relationTarget = null;
        updateRelationUI();
      }
    }

    function deleteRelation(relId: string): void {
      const idx = activeRelations.findIndex((r) => r.relation_id === relId);
      if (idx >= 0) {
        activeRelations.splice(idx, 1);
        events.push({
          type: "relation_delete",
          timestamp: performance.now() - start_time,
          relation_id: relId,
        });
        renderRelationArcsOverlay();
        renderRelationList();
        updateContinueButton();
      }
    }

    function renderRelationList(): void {
      const listEl = display_element.querySelector<HTMLElement>("#bead-relation-list");
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
          delBtn.textContent = "\u00d7";
          delBtn.addEventListener("click", () => deleteRelation(rel.relation_id));
          entry.appendChild(delBtn);
        }

        listEl.appendChild(entry);
      }

      // Update Add Relation button state
      const updateUI = (display_element as unknown as Record<string, unknown>)["_updateRelationUI"];
      if (typeof updateUI === "function") {
        (updateUI as () => void)();
      }
    }

    function computeSpanPositions(): Map<string, DOMRect> {
      const positions: Map<string, DOMRect> = new Map();
      const container = display_element.querySelector<HTMLElement>(".bead-span-container");
      if (!container) return positions;
      const containerRect = container.getBoundingClientRect();

      for (const span of activeSpans) {
        let minLeft = Number.POSITIVE_INFINITY;
        let minTop = Number.POSITIVE_INFINITY;
        let maxRight = Number.NEGATIVE_INFINITY;
        let maxBottom = Number.NEGATIVE_INFINITY;

        for (const seg of span.segments) {
          for (const idx of seg.indices) {
            const tokenEl = display_element.querySelector<HTMLElement>(
              `.bead-token[data-element="${seg.element_name}"][data-index="${idx}"]`,
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
            new DOMRect(minLeft, minTop, maxRight - minLeft, maxBottom - minTop),
          );
        }
      }
      return positions;
    }

    function renderRelationArcsOverlay(): void {
      if (activeRelations.length === 0) return;

      const container = display_element.querySelector<HTMLElement>(".bead-span-container");
      if (!container) return;

      // Remove existing arc container
      const existingArcDiv = display_element.querySelector(".bead-relation-arc-area");
      if (existingArcDiv) existingArcDiv.remove();

      const spanPositions = computeSpanPositions();
      if (spanPositions.size === 0) return;

      // Create a dedicated div above the token container for arcs
      const arcArea = document.createElement("div");
      arcArea.className = "bead-relation-arc-area";
      arcArea.style.position = "relative";
      arcArea.style.width = "100%";

      // Stagger heights for multiple relations
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

      // Arrowhead marker
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

      // Get container rect for coordinate mapping
      const containerRect = container.getBoundingClientRect();
      const arcAreaRect = arcArea.getBoundingClientRect();

      for (let i = 0; i < activeRelations.length; i++) {
        const rel = activeRelations[i];
        if (!rel) continue;

        const sourceRect = spanPositions.get(rel.source_span_id);
        const targetRect = spanPositions.get(rel.target_span_id);
        if (!sourceRect || !targetRect) continue;

        // X coordinates: center of each span, relative to container left edge
        const x1 = sourceRect.x + sourceRect.width / 2;
        const x2 = targetRect.x + targetRect.width / 2;

        // Y coordinates: bottom of arc area connects down to tokens
        const bottomY = totalHeight;
        // Horizontal rail height: stagger from top
        const railY = totalHeight - baseHeight - i * levelSpacing;

        const r = 5; // Corner radius
        const strokeColor = "#546e7a";

        // Manhattan path: up from bottom, across horizontally, down to target
        const dir = x2 > x1 ? 1 : -1;
        const d = [
          `M ${x1} ${bottomY}`,
          `L ${x1} ${railY + r}`,
          `Q ${x1} ${railY} ${x1 + r * dir} ${railY}`,
          `L ${x2 - r * dir} ${railY}`,
          `Q ${x2} ${railY} ${x2} ${railY + r}`,
          `L ${x2} ${bottomY}`,
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

        // Label on the horizontal rail
        if (rel.label?.label) {
          const midX = (x1 + x2) / 2;
          // Background rect for label
          const labelText = rel.label.label;

          // Use a foreignObject for better text rendering
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

      // Insert arc area before the token container
      container.parentNode?.insertBefore(arcArea, container);
    }

    function updateContinueButton(): void {
      if (!continueBtn || !isInteractive) return;
      const minSpans = spanSpec?.min_spans ?? 0;
      continueBtn.disabled = activeSpans.length < minSpans;
    }

    const endTrial = (): void => {
      // Remove keyboard listener
      document.removeEventListener("keydown", handleKeyDown);

      const trial_data: Record<string, unknown> = {
        ...trial.metadata,
        spans: activeSpans,
        relations: activeRelations,
        span_events: events,
        rt: performance.now() - start_time,
      };

      display_element.innerHTML = "";
      this.jsPsych.finishTrial(trial_data);
    };
  }
}

export { BeadSpanLabelPlugin };
