/**
 * Type declarations for jsPsych 8.x plugin API.
 *
 * These types define the interfaces needed for creating jsPsych plugins
 * and interacting with the jsPsych runtime.
 */

/** Parameter types available in jsPsych */
export const ParameterType: {
  readonly BOOL: 0;
  readonly STRING: 1;
  readonly INT: 2;
  readonly FLOAT: 3;
  readonly FUNCTION: 4;
  readonly KEY: 5;
  readonly KEYS: 6;
  readonly SELECT: 7;
  readonly HTML_STRING: 8;
  readonly IMAGE: 9;
  readonly AUDIO: 10;
  readonly VIDEO: 11;
  readonly OBJECT: 12;
  readonly COMPLEX: 13;
  readonly TIMELINE: 14;
};

export type ParameterTypeValue = (typeof ParameterType)[keyof typeof ParameterType];

/** Definition of a single parameter in plugin info */
export interface ParameterInfo {
  type: ParameterTypeValue;
  pretty_name?: string;
  default?: unknown;
  description?: string;
  array?: boolean;
  options?: readonly string[];
  nested?: Record<string, ParameterInfo>;
}

/** Plugin info object that defines the plugin's parameters */
export interface PluginInfo {
  name: string;
  parameters: Record<string, ParameterInfo>;
  data?: Record<string, ParameterInfo>;
}

/** Keyboard response info returned by getKeyboardResponse callback */
export interface KeyboardResponseInfo {
  key: string;
  rt: number;
}

/** Plugin API methods available to plugins */
export interface PluginAPI {
  getKeyboardResponse(options: {
    callback_function: (info: KeyboardResponseInfo) => void;
    valid_responses?: string[] | "ALL_KEYS" | "NO_KEYS";
    rt_method?: "performance" | "date";
    persist?: boolean;
    allow_held_key?: boolean;
    minimum_valid_rt?: number;
  }): unknown;

  cancelKeyboardResponse(listener: unknown): void;

  cancelAllKeyboardResponses(): void;

  compareKeys(key1: string | null, key2: string | null): boolean;

  setTimeout(callback: () => void, delay: number): number;

  clearAllTimeouts(): void;
}

/** Data API for accessing and modifying experiment data */
export interface DataAPI {
  get(): {
    json(): string;
    csv(): string;
    values(): TrialData[];
    filter(filters: Record<string, unknown> | ((trial: TrialData) => boolean)): DataAPI;
    filterCustom(fn: (trial: TrialData) => boolean): DataAPI;
    select(column: string): { values: unknown[] };
    count(): number;
  };

  addProperties(properties: Record<string, unknown>): void;

  write(data: TrialData): void;
}

/** Trial data object */
export interface TrialData {
  trial_type?: string;
  trial_index?: number;
  time_elapsed?: number;
  [key: string]: unknown;
}

/** Extension interface for jsPsych extensions */
export interface JsPsychExtension {
  initialize?(params: Record<string, unknown>): Promise<void> | void;
  on_start?(params: Record<string, unknown>): void;
  on_load?(params: Record<string, unknown>): void;
  on_finish?(params: Record<string, unknown>): Record<string, unknown> | undefined;
}

/** Extension info */
export interface ExtensionInfo {
  name: string;
}

/** Main jsPsych instance interface */
export interface JsPsych {
  version(): string;

  pluginAPI: PluginAPI;

  data: DataAPI;

  run(timeline: TimelineNode[]): Promise<void>;

  finishTrial(data?: Record<string, unknown>): void;

  endExperiment(end_message?: string, data?: Record<string, unknown>): void;

  getCurrentTrial(): TimelineNode | null;

  getDisplayElement(): HTMLElement;

  getProgressBarCompleted(): number;

  setProgressBar(value: number): void;

  timelineVariable(variable_name: string): unknown;

  pauseExperiment(): void;

  resumeExperiment(): void;

  abortCurrentTimeline(): void;

  abortExperiment(end_message?: string): void;

  getStartTime(): number;

  getTotalTime(): number;

  getDisplayContainerElement(): HTMLElement;
}

/** Timeline node (trial) structure */
export interface TimelineNode {
  type?: unknown;
  timeline?: TimelineNode[];
  timeline_variables?: Record<string, unknown>[];
  conditional_function?: () => boolean;
  loop_function?: (data: DataAPI) => boolean;
  on_start?: (trial: TimelineNode) => void;
  on_load?: () => void;
  on_finish?: (data: TrialData) => void;
  data?: Record<string, unknown>;
  extensions?: Array<{
    type: unknown;
    params?: Record<string, unknown>;
  }>;
  [key: string]: unknown;
}

/** jsPsych plugin class interface */
export interface JsPsychPlugin<Info extends PluginInfo, TrialParams = Record<string, unknown>> {
  trial(
    display_element: HTMLElement,
    trial: TrialParams,
    on_load?: () => void,
  ): void | Promise<void>;
}

/** Extract trial parameter types from plugin info */
export type TrialType<Info extends PluginInfo> = {
  [K in keyof Info["parameters"]]: Info["parameters"][K] extends {
    default: infer D;
  }
    ? Info["parameters"][K]["type"] extends typeof ParameterType.STRING
      ? string
      : Info["parameters"][K]["type"] extends typeof ParameterType.INT
        ? number
        : Info["parameters"][K]["type"] extends typeof ParameterType.FLOAT
          ? number
          : Info["parameters"][K]["type"] extends typeof ParameterType.BOOL
            ? boolean
            : Info["parameters"][K]["type"] extends typeof ParameterType.OBJECT
              ? Record<string, unknown>
              : Info["parameters"][K]["type"] extends typeof ParameterType.HTML_STRING
                ? string
                : D
    : unknown;
};

/** jsPsych module declaration for global access in browser */
declare global {
  const jsPsychModule: {
    ParameterType: typeof ParameterType;
  };

  function initJsPsych(options?: {
    display_element?: HTMLElement | string;
    on_trial_start?: (trial: TimelineNode) => void;
    on_trial_finish?: (data: TrialData) => void;
    on_data_update?: (data: TrialData) => void;
    on_interaction_data_update?: (data: Record<string, unknown>) => void;
    on_close?: () => void;
    on_finish?: (data: DataAPI) => void;
    show_progress_bar?: boolean;
    auto_update_progress_bar?: boolean;
    message_progress_bar?: string;
    extensions?: Array<{
      type: unknown;
      params?: Record<string, unknown>;
    }>;
    override_safe_mode?: boolean;
    case_sensitive_responses?: boolean;
    minimum_valid_rt?: number;
    experiment_width?: number;
  }): JsPsych;

  /** jsPsych HTML keyboard response plugin */
  const jsPsychHtmlKeyboardResponse: unknown;
}
