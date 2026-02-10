/**
 * Client-side Wikidata entity search.
 *
 * Uses the Wikidata API (wbsearchentities) for autocomplete typeahead
 * on span labels and relation labels when label_source is "wikidata".
 *
 * Features: 300ms debouncing, LRU result caching, origin=* for CORS,
 * configurable language/limit/entity types.
 *
 * @author Bead Project
 * @version 0.2.0
 */

/** Wikidata entity result */
export interface WikidataEntity {
  id: string;
  label: string;
  description: string;
  aliases: string[];
}

/** Search options */
export interface WikidataSearchOptions {
  language: string;
  limit: number;
  entityTypes?: string[];
}

const WIKIDATA_API = "https://www.wikidata.org/w/api.php";
const CACHE_SIZE = 100;
const DEBOUNCE_MS = 300;

// Simple LRU cache
const cache: Map<string, WikidataEntity[]> = new Map();

function cacheKey(query: string, opts: WikidataSearchOptions): string {
  return `${opts.language}:${query}:${opts.limit}:${(opts.entityTypes ?? []).join(",")}`;
}

function putCache(key: string, value: WikidataEntity[]): void {
  if (cache.size >= CACHE_SIZE) {
    const firstKey = cache.keys().next().value;
    if (firstKey !== undefined) {
      cache.delete(firstKey);
    }
  }
  cache.set(key, value);
}

/**
 * Search Wikidata entities.
 */
export async function searchWikidata(
  query: string,
  options: WikidataSearchOptions,
): Promise<WikidataEntity[]> {
  if (!query || query.trim().length === 0) {
    return [];
  }

  const key = cacheKey(query, options);
  const cached = cache.get(key);
  if (cached) {
    return cached;
  }

  const params = new URLSearchParams({
    action: "wbsearchentities",
    search: query.trim(),
    language: options.language,
    limit: String(options.limit),
    format: "json",
    origin: "*",
  });

  if (options.entityTypes && options.entityTypes.length > 0) {
    params.set("type", options.entityTypes[0] ?? "item");
  }

  const url = `${WIKIDATA_API}?${params.toString()}`;

  try {
    const response = await fetch(url);
    if (!response.ok) {
      return [];
    }

    const data = await response.json();
    const results: WikidataEntity[] = (data.search ?? []).map((item: Record<string, unknown>) => ({
      id: String(item["id"] ?? ""),
      label: String(item["label"] ?? ""),
      description: String(item["description"] ?? ""),
      aliases: Array.isArray(item["aliases"]) ? item["aliases"].map(String) : [],
    }));

    putCache(key, results);
    return results;
  } catch {
    return [];
  }
}

// Debounce utility
let debounceTimer: ReturnType<typeof setTimeout> | null = null;

/**
 * Debounced Wikidata search.
 */
export function debouncedSearchWikidata(
  query: string,
  options: WikidataSearchOptions,
  callback: (results: WikidataEntity[]) => void,
): void {
  if (debounceTimer !== null) {
    clearTimeout(debounceTimer);
  }

  debounceTimer = setTimeout(async () => {
    const results = await searchWikidata(query, options);
    callback(results);
  }, DEBOUNCE_MS);
}
