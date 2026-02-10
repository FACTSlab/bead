/**
 * Gallery bundle entry point
 *
 * Registers all bead jsPsych plugins as window globals so they can be
 * loaded via a single <script> tag alongside the jsPsych CDN. Built as
 * IIFE format by tsup.gallery.config.ts.
 *
 * @author Bead Project
 * @version 0.2.0
 */

import { BeadBinaryChoicePlugin } from "../plugins/binary-choice.js";
import { BeadCategoricalPlugin } from "../plugins/categorical.js";
import { BeadClozeMultiPlugin } from "../plugins/cloze-dropdown.js";
import { BeadForcedChoicePlugin } from "../plugins/forced-choice.js";
import { BeadFreeTextPlugin } from "../plugins/free-text.js";
import { BeadMagnitudePlugin } from "../plugins/magnitude.js";
import { BeadMultiSelectPlugin } from "../plugins/multi-select.js";
import { BeadRatingPlugin } from "../plugins/rating.js";
import { BeadSliderRatingPlugin } from "../plugins/slider-rating.js";
import { BeadSpanLabelPlugin } from "../plugins/span-label.js";

declare global {
  interface Window {
    BeadRatingPlugin: typeof BeadRatingPlugin;
    BeadForcedChoicePlugin: typeof BeadForcedChoicePlugin;
    BeadBinaryChoicePlugin: typeof BeadBinaryChoicePlugin;
    BeadSliderRatingPlugin: typeof BeadSliderRatingPlugin;
    BeadClozeMultiPlugin: typeof BeadClozeMultiPlugin;
    BeadSpanLabelPlugin: typeof BeadSpanLabelPlugin;
    BeadCategoricalPlugin: typeof BeadCategoricalPlugin;
    BeadMagnitudePlugin: typeof BeadMagnitudePlugin;
    BeadFreeTextPlugin: typeof BeadFreeTextPlugin;
    BeadMultiSelectPlugin: typeof BeadMultiSelectPlugin;
  }
}

window.BeadRatingPlugin = BeadRatingPlugin;
window.BeadForcedChoicePlugin = BeadForcedChoicePlugin;
window.BeadBinaryChoicePlugin = BeadBinaryChoicePlugin;
window.BeadSliderRatingPlugin = BeadSliderRatingPlugin;
window.BeadClozeMultiPlugin = BeadClozeMultiPlugin;
window.BeadSpanLabelPlugin = BeadSpanLabelPlugin;
window.BeadCategoricalPlugin = BeadCategoricalPlugin;
window.BeadMagnitudePlugin = BeadMagnitudePlugin;
window.BeadFreeTextPlugin = BeadFreeTextPlugin;
window.BeadMultiSelectPlugin = BeadMultiSelectPlugin;
