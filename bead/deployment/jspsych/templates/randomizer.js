// This file is generated from Python OrderingConstraint specifications
// Do not edit manually - regenerate using bead.deployment.jspsych.randomizer

// Embedded metadata for constraint checking
const trialMetadata = {{ metadata | tojson }};

// Constraint specifications
const constraints = {{ constraints | tojson }};

/**
 * Shuffle array in place using Fisher-Yates algorithm
 * @param {Array} array - Array to shuffle
 * @param {function} rng - Random number generator
 */
function shuffle(array, rng) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}

{% if has_precedence %}
/**
 * Check if trial order satisfies precedence constraints
 * @param {Array} trials - Array of trial objects with item_id property
 * @param {Array} pairs - Array of [itemA_id, itemB_id] precedence pairs
 * @returns {boolean} True if all precedence constraints satisfied
 */
function checkPrecedence(trials, pairs) {
    const positions = {};
    trials.forEach((trial, idx) => {
        positions[trial.item_id] = idx;
    });

    for (const [itemA, itemB] of pairs) {
        if (positions[itemA] !== undefined && positions[itemB] !== undefined) {
            if (positions[itemA] >= positions[itemB]) {
                return false;
            }
        }
    }
    return true;
}
{% endif %}

{% if has_no_adjacent %}
/**
 * Check if no adjacent trials have same property value
 * @param {Array} trials - Array of trial objects
 * @param {string} property - Property path to check
 * @param {Object} metadata - Trial metadata
 * @returns {boolean} True if no adjacent items have same value
 */
function checkNoAdjacent(trials, property, metadata) {
    for (let i = 0; i < trials.length - 1; i++) {
        const valueA = getPropertyValue(metadata[trials[i].item_id], property);
        const valueB = getPropertyValue(metadata[trials[i + 1].item_id], property);

        if (valueA !== undefined && valueB !== undefined && valueA === valueB) {
            return false;
        }
    }
    return true;
}
{% endif %}

{% if has_min_distance %}
/**
 * Check if minimum distance constraint is satisfied
 * @param {Array} trials - Array of trial objects
 * @param {string} property - Property path to check
 * @param {number} minDist - Minimum distance required
 * @param {Object} metadata - Trial metadata
 * @returns {boolean} True if minimum distance satisfied
 */
function checkMinDistance(trials, property, minDist, metadata) {
    const valuePositions = {};

    trials.forEach((trial, idx) => {
        const value = getPropertyValue(metadata[trial.item_id], property);
        if (value !== undefined) {
            if (!valuePositions[value]) {
                valuePositions[value] = [];
            }
            valuePositions[value].push(idx);
        }
    });

    for (const positions of Object.values(valuePositions)) {
        for (let i = 0; i < positions.length - 1; i++) {
            const distance = positions[i + 1] - positions[i] - 1;
            if (distance < minDist) {
                return false;
            }
        }
    }
    return true;
}
{% endif %}

/**
 * Get property value from nested object using dot notation
 * @param {Object} obj - Object to query
 * @param {string} path - Dot-notation path (e.g., "item_metadata.condition")
 * @returns {*} Property value or undefined
 */
function getPropertyValue(obj, path) {
    const parts = path.split('.');
    let current = obj;
    for (const part of parts) {
        if (current === undefined || current === null) {
            return undefined;
        }
        current = current[part];
    }
    return current;
}

/**
 * Check if trial order satisfies all constraints
 * @param {Array} trials - Array of trial objects
 * @param {Object} metadata - Trial metadata
 * @returns {boolean} True if all constraints satisfied
 */
function checkAllConstraints(trials, metadata) {
    {% if has_precedence %}
    // Check precedence constraints
    for (const constraint of constraints) {
        if (constraint.precedence_pairs && constraint.precedence_pairs.length > 0) {
            if (!checkPrecedence(trials, constraint.precedence_pairs)) {
                return false;
            }
        }
    }
    {% endif %}

    {% if has_no_adjacent %}
    // Check no-adjacent constraints
    for (const constraint of constraints) {
        if (constraint.no_adjacent_property) {
            if (!checkNoAdjacent(trials, constraint.no_adjacent_property, metadata)) {
                return false;
            }
        }
    }
    {% endif %}

    {% if has_min_distance %}
    // Check minimum distance constraints
    for (const constraint of constraints) {
        if (constraint.min_distance && constraint.no_adjacent_property) {
            if (!checkMinDistance(trials, constraint.no_adjacent_property, constraint.min_distance, metadata)) {
                return false;
            }
        }
    }
    {% endif %}

    return true;
}

/**
 * Main entry point: randomize trials with constraint enforcement
 * @param {Array} trials - Array of jsPsych trial objects with item_id property
 * @param {string|number} seed - Random seed (usually participant ID)
 * @returns {Array} Randomized trials satisfying all constraints
 */
function randomizeTrials(trials, seed) {
    const rng = new Math.seedrandom(seed);

    {% if has_practice %}
    // Separate practice items (must come first)
    const practiceTrials = trials.filter(t => {
        const meta = trialMetadata[t.item_id];
        const value = getPropertyValue(meta, '{{ practice_property }}');
        return value === true;
    });
    const mainTrials = trials.filter(t => {
        const meta = trialMetadata[t.item_id];
        const value = getPropertyValue(meta, '{{ practice_property }}');
        return value !== true;
    });
    {% else %}
    const practiceTrials = [];
    const mainTrials = trials.slice();
    {% endif %}

    {% if has_blocking %}
    // Group main trials by block property
    const blocks = {};
    mainTrials.forEach(t => {
        const blockValue = getPropertyValue(trialMetadata[t.item_id], '{{ block_property }}');
        const blockKey = blockValue !== undefined ? String(blockValue) : '__undefined__';
        if (!blocks[blockKey]) {
            blocks[blockKey] = [];
        }
        blocks[blockKey].push(t);
    });

    // Randomize block order
    const blockKeys = Object.keys(blocks);
    shuffle(blockKeys, rng);

    let randomizedMain = [];
    blockKeys.forEach(key => {
        const blockTrials = blocks[key];
        {% if randomize_within_blocks %}
        // Randomize within blocks
        shuffle(blockTrials, rng);
        {% endif %}
        randomizedMain = randomizedMain.concat(blockTrials);
    });
    {% else %}
    // Rejection sampling: try to find valid order
    const maxAttempts = 1000;
    let randomizedMain = mainTrials.slice();
    let lastAttempt = randomizedMain.slice();

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
        shuffle(randomizedMain, rng);
        lastAttempt = randomizedMain.slice();

        if (checkAllConstraints(randomizedMain, trialMetadata)) {
            break;
        }

        if (attempt === maxAttempts - 1) {
            console.warn('Could not find constraint-satisfying order after ' +
                        maxAttempts + ' attempts. Using last attempt.');
            randomizedMain = lastAttempt;
        }
    }
    {% endif %}

    return practiceTrials.concat(randomizedMain);
}

// Export for use in jsPsych experiments
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { randomizeTrials };
}
