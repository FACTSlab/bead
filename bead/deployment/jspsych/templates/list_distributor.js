/**
 * List Distribution System for JATOS Batch Sessions
 *
 * Manages server-side list assignment using JATOS batch sessions.
 * Supports 8 distribution strategies with strict error handling (no fallbacks).
 *
 * @module list_distributor
 */

/**
 * Sleep for specified milliseconds.
 *
 * @param {number} ms - Milliseconds to sleep
 * @returns {Promise<void>}
 */
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Load lists from lists.jsonl file.
 *
 * @param {string} jsonlPath - Path to lists.jsonl file
 * @returns {Promise<Array<Object>>} Array of ExperimentList objects
 * @throws {Error} If file not found or parsing fails
 */
async function loadLists(jsonlPath) {
    try {
        const response = await fetch(jsonlPath);
        if (!response.ok) {
            throw new Error(
                `Failed to fetch lists.jsonl (HTTP ${response.status}). ` +
                `Expected file at: ${jsonlPath}. ` +
                `Verify the experiment was generated correctly using JsPsychExperimentGenerator.generate().`
            );
        }

        const text = await response.text();
        const lists = [];
        const lines = text.trim().split('\n');

        for (const line of lines) {
            if (line.trim()) {
                try {
                    const list = JSON.parse(line);
                    lists.push(list);
                } catch (error) {
                    throw new Error(
                        `Failed to parse list from lists.jsonl: ${error.message}. ` +
                        `Line content: ${line.substring(0, 100)}...`
                    );
                }
            }
        }

        if (lists.length === 0) {
            throw new Error(
                `Loaded lists.jsonl but got empty array. ` +
                `Verify your ExperimentLists were created and passed to generate(). ` +
                `File path: ${jsonlPath}`
            );
        }

        return lists;
    } catch (error) {
        if (error.message.includes('Failed to fetch')) {
            throw error;
        }
        throw new Error(`Error loading lists: ${error.message}`);
    }
}

/**
 * Load items from items.jsonl file.
 *
 * @param {string} jsonlPath - Path to items.jsonl file
 * @returns {Promise<Object>} Dictionary of items keyed by UUID
 * @throws {Error} If file not found or parsing fails
 */
async function loadItems(jsonlPath) {
    try {
        const response = await fetch(jsonlPath);
        if (!response.ok) {
            throw new Error(
                `Failed to fetch items.jsonl (HTTP ${response.status}). ` +
                `Expected file at: ${jsonlPath}. ` +
                `Verify the experiment was generated correctly.`
            );
        }

        const text = await response.text();
        const items = {};
        const lines = text.trim().split('\n');

        for (const line of lines) {
            if (line.trim()) {
                try {
                    const item = JSON.parse(line);
                    items[item.id] = item;
                } catch (error) {
                    throw new Error(
                        `Failed to parse item from items.jsonl: ${error.message}. ` +
                        `Line content: ${line.substring(0, 100)}...`
                    );
                }
            }
        }

        if (Object.keys(items).length === 0) {
            throw new Error(
                `Loaded items.jsonl but got empty dictionary. ` +
                `Verify your Items were created and passed to generate(). ` +
                `File path: ${jsonlPath}`
            );
        }

        return items;
    } catch (error) {
        if (error.message.includes('Failed to fetch')) {
            throw error;
        }
        throw new Error(`Error loading items: ${error.message}`);
    }
}

/**
 * Initialize batch session state for list distribution.
 *
 * @param {Object} config - Distribution strategy configuration
 * @param {Array<Object>} lists - Array of experiment lists
 * @returns {Promise<void>}
 * @throws {Error} If initialization fails
 */
async function initializeBatchSession(config, lists) {
    // Initialize distribution config
    await jatos.batchSession.set('distribution', {
        strategy_type: config.strategy_type,
        strategy_config: config.strategy_config || {},
        initialized: true,
        created_at: new Date().toISOString()
    });

    // Initialize statistics
    const assignment_counts = {};
    const completion_counts = {};
    for (let i = 0; i < lists.length; i++) {
        assignment_counts[i] = 0;
        completion_counts[i] = 0;
    }

    await jatos.batchSession.set('statistics', {
        assignment_counts: assignment_counts,
        completion_counts: completion_counts,
        total_assignments: 0,
        total_completions: 0
    });

    // Initialize assignments
    await jatos.batchSession.set('assignments', {});

    // Initialize pool
    const availableIndices = lists.map((_, idx) => idx);
    await jatos.batchSession.set('pool', {
        available_list_indices: availableIndices,
        exhausted: false
    });

    // Strategy-specific initialization
    if (config.strategy_type === 'sequential') {
        await jatos.batchSession.set('strategy_state', { next_index: 0 });
    } else if (config.strategy_type === 'quota_based') {
        if (!config.strategy_config.participants_per_list) {
            throw new Error(
                `QuotaConfig requires 'participants_per_list' in strategy_config. ` +
                `Got: ${JSON.stringify(config.strategy_config)}. ` +
                `Add 'participants_per_list: <int>' to your distribution_strategy config.`
            );
        }
        const quotas = {};
        for (let i = 0; i < lists.length; i++) {
            quotas[i] = config.strategy_config.participants_per_list;
        }
        await jatos.batchSession.set('strategy_state', { remaining_quotas: quotas });
    } else if (config.strategy_type === 'latin_square') {
        const matrix = generateBalancedLatinSquare(lists.length);
        await jatos.batchSession.set('strategy_state', {
            latin_square_matrix: matrix,
            latin_square_position: 0
        });
    } else {
        await jatos.batchSession.set('strategy_state', {});
    }
}

/**
 * Generate balanced Latin square using Bradley's (1958) algorithm.
 *
 * @param {number} n - Number of lists
 * @returns {Array<Array<number>>} Latin square matrix
 */
function generateBalancedLatinSquare(n) {
    const square = [];
    for (let i = 0; i < n; i++) {
        const row = [];
        for (let j = 0; j < n; j++) {
            if (i % 2 === 0) {
                row.push((Math.floor(i / 2) + j) % n);
            } else {
                row.push((Math.floor(i / 2) + n - j) % n);
            }
        }
        square.push(row);
    }
    return square;
}

/**
 * Select list index based on distribution strategy.
 *
 * @param {Object} config - Distribution strategy configuration
 * @param {Array<Object>} lists - Array of experiment lists
 * @param {Object} state - Current batch session state
 * @returns {number} Selected list index
 * @throws {Error} If strategy unknown or selection fails
 */
function selectListForAssignment(config, lists, state) {
    switch(config.strategy_type) {
        case 'random':
            return assignRandom(lists);
        case 'sequential':
            return assignSequential(lists, state);
        case 'balanced':
            return assignBalanced(lists, state);
        case 'latin_square':
            return assignLatinSquare(lists, state);
        case 'stratified':
            return assignStratified(lists, config, state);
        case 'weighted_random':
            return assignWeightedRandom(lists, config, state);
        case 'quota_based':
            return assignQuota(lists, config, state);
        case 'metadata_based':
            return assignMetadataBased(lists, config, state);
        default:
            throw new Error(
                `Unknown strategy type: '${config.strategy_type}'. ` +
                `Valid types: random, sequential, balanced, latin_square, ` +
                `stratified, weighted_random, quota_based, metadata_based. ` +
                `Check your distribution.json config.`
            );
    }
}

/**
 * Random assignment strategy.
 *
 * @param {Array<Object>} lists - Available lists
 * @returns {number} Random list index
 * @throws {Error} If no lists available
 */
function assignRandom(lists) {
    if (lists.length === 0) {
        throw new Error(
            `No lists available for random assignment. ` +
            `Verify lists.jsonl was generated and is readable.`
        );
    }
    return Math.floor(Math.random() * lists.length);
}

/**
 * Sequential (round-robin) assignment strategy.
 *
 * @param {Array<Object>} lists - Available lists
 * @param {Object} state - Batch session state
 * @returns {number} Next list index in sequence
 */
function assignSequential(lists, state) {
    const currentIndex = state.strategy_state.next_index || 0;
    const listIndex = currentIndex % lists.length;
    return listIndex;
}

/**
 * Balanced assignment strategy (assign to least-used list).
 *
 * @param {Array<Object>} lists - Available lists
 * @param {Object} state - Batch session state
 * @returns {number} Least-used list index
 */
function assignBalanced(lists, state) {
    const counts = state.statistics.assignment_counts;

    // Find minimum count
    let minCount = Infinity;
    const minIndices = [];

    for (let i = 0; i < lists.length; i++) {
        const count = counts[i] || 0;
        if (count < minCount) {
            minCount = count;
            minIndices.length = 0;
            minIndices.push(i);
        } else if (count === minCount) {
            minIndices.push(i);
        }
    }

    // Random selection among minimum
    return minIndices[Math.floor(Math.random() * minIndices.length)];
}

/**
 * Latin square counterbalancing strategy.
 *
 * @param {Array<Object>} lists - Available lists
 * @param {Object} state - Batch session state
 * @returns {number} List index from Latin square
 */
function assignLatinSquare(lists, state) {
    const matrix = state.strategy_state.latin_square_matrix;
    const position = state.strategy_state.latin_square_position || 0;

    const row = position % matrix.length;
    const col = Math.floor(position / matrix.length) % matrix[0].length;
    const listIndex = matrix[row][col];

    return listIndex;
}

/**
 * Stratified assignment strategy (balance across factors).
 *
 * @param {Array<Object>} lists - Available lists
 * @param {Object} config - Strategy configuration
 * @param {Object} state - Batch session state
 * @returns {number} List index for stratified assignment
 * @throws {Error} If factors not specified
 */
function assignStratified(lists, config, state) {
    if (!config.strategy_config.factors || config.strategy_config.factors.length === 0) {
        throw new Error(
            `StratifiedConfig requires 'factors' in strategy_config. ` +
            `Got: ${JSON.stringify(config.strategy_config)}. ` +
            `Provide a list like ['condition', 'verb_type'].`
        );
    }

    const factors = config.strategy_config.factors;
    const counts = state.statistics.assignment_counts;

    // Group lists by factor combinations
    const strata = {};
    for (let i = 0; i < lists.length; i++) {
        const key = factors.map(f => lists[i].list_metadata[f] || 'null').join('|');
        if (!strata[key]) {
            strata[key] = [];
        }
        strata[key].push(i);
    }

    // Find stratum with minimum assignments
    let minCount = Infinity;
    let minStratumIndices = [];

    for (const [key, indices] of Object.entries(strata)) {
        const stratumCount = indices.reduce((sum, idx) => sum + (counts[idx] || 0), 0);
        if (stratumCount < minCount) {
            minCount = stratumCount;
            minStratumIndices = indices;
        }
    }

    // Random from minimum stratum
    return minStratumIndices[Math.floor(Math.random() * minStratumIndices.length)];
}

/**
 * Weighted random assignment strategy.
 *
 * @param {Array<Object>} lists - Available lists
 * @param {Object} config - Strategy configuration
 * @param {Object} state - Batch session state
 * @returns {number} List index selected by weighted random
 * @throws {Error} If weight_expression not specified
 */
function assignWeightedRandom(lists, config, state) {
    if (!config.strategy_config.weight_expression) {
        throw new Error(
            `WeightedRandomConfig requires 'weight_expression' in strategy_config. ` +
            `Got: ${JSON.stringify(config.strategy_config)}. ` +
            `Provide a JavaScript expression like 'list_metadata.priority || 1.0'.`
        );
    }

    const expr = config.strategy_config.weight_expression;
    const normalize = config.strategy_config.normalize_weights !== false;

    // Compute weights
    const weights = lists.map(list => {
        const list_metadata = list.list_metadata || {};
        try {
            return eval(expr);
        } catch (error) {
            throw new Error(
                `Failed to evaluate weight_expression '${expr}' for list ${list.name}: ${error.message}. ` +
                `Check your expression syntax.`
            );
        }
    });

    // Normalize if requested
    let w = weights;
    if (normalize) {
        const sum = weights.reduce((a, b) => a + b, 0);
        if (sum === 0) {
            throw new Error(
                `Sum of weights is 0. Cannot normalize. ` +
                `Weight expression: '${expr}'. ` +
                `Check that your expression produces positive values.`
            );
        }
        w = weights.map(weight => weight / sum);
    }

    // Cumulative distribution and sample
    const cumulative = [];
    let sum = 0;
    for (const weight of w) {
        sum += weight;
        cumulative.push(sum);
    }

    const random = Math.random() * cumulative[cumulative.length - 1];
    for (let i = 0; i < cumulative.length; i++) {
        if (random <= cumulative[i]) {
            return i;
        }
    }

    return lists.length - 1;
}

/**
 * Quota-based assignment strategy.
 *
 * @param {Array<Object>} lists - Available lists
 * @param {Object} config - Strategy configuration
 * @param {Object} state - Batch session state
 * @returns {number} List index with remaining quota
 * @throws {Error} If quotas exhausted and overflow not allowed
 */
function assignQuota(lists, config, state) {
    if (!config.strategy_config.participants_per_list) {
        throw new Error(
            `QuotaConfig requires 'participants_per_list' in strategy_config. ` +
            `Got: ${JSON.stringify(config.strategy_config)}. ` +
            `Add 'participants_per_list: <int>' to your distribution_strategy config.`
        );
    }

    const quotas = state.strategy_state.remaining_quotas;
    const available = [];

    for (let i = 0; i < lists.length; i++) {
        if (quotas[i] > 0) {
            available.push(i);
        }
    }

    if (available.length === 0) {
        if (config.strategy_config.allow_overflow === true) {
            // Use balanced assignment
            return assignBalanced(lists, state);
        } else {
            throw new Error(
                `All lists have reached quota and allow_overflow=false. ` +
                `Current quotas: ${JSON.stringify(quotas)}. ` +
                `Options: (1) Set allow_overflow: true, ` +
                `(2) Increase participants_per_list, or (3) Add more lists.`
            );
        }
    }

    // Random from available
    return available[Math.floor(Math.random() * available.length)];
}

/**
 * Metadata-based assignment strategy.
 *
 * @param {Array<Object>} lists - Available lists
 * @param {Object} config - Strategy configuration
 * @param {Object} state - Batch session state
 * @returns {number} List index based on metadata filtering/ranking
 * @throws {Error} If no lists match filter or expressions missing
 */
function assignMetadataBased(lists, config, state) {
    const hasFilter = config.strategy_config.filter_expression;
    const hasRank = config.strategy_config.rank_expression;

    if (!hasFilter && !hasRank) {
        throw new Error(
            `MetadataBasedConfig requires at least one of 'filter_expression' or 'rank_expression'. ` +
            `Got: ${JSON.stringify(config.strategy_config)}. ` +
            `Add 'filter_expression' (e.g., "list_metadata.difficulty === 'easy'") ` +
            `or 'rank_expression' (e.g., "list_metadata.priority || 0").`
        );
    }

    // Filter lists
    let available = lists.map((list, idx) => ({list, idx}));

    if (hasFilter) {
        const filterExpr = config.strategy_config.filter_expression;
        available = available.filter(item => {
            const list_metadata = item.list.list_metadata || {};
            try {
                return eval(filterExpr);
            } catch (error) {
                throw new Error(
                    `Failed to evaluate filter_expression '${filterExpr}' for list ${item.list.name}: ${error.message}. ` +
                    `Check your expression syntax.`
                );
            }
        });

        if (available.length === 0) {
            throw new Error(
                `No lists match filter_expression: '${filterExpr}'. ` +
                `All ${lists.length} lists were filtered out. ` +
                `Check your filter expression or list metadata.`
            );
        }
    }

    // Rank lists
    if (hasRank) {
        const rankExpr = config.strategy_config.rank_expression;
        const ascending = config.strategy_config.rank_ascending !== false;

        available = available.map(item => {
            const list_metadata = item.list.list_metadata || {};
            let score;
            try {
                score = eval(rankExpr);
            } catch (error) {
                throw new Error(
                    `Failed to evaluate rank_expression '${rankExpr}' for list ${item.list.name}: ${error.message}. ` +
                    `Check your expression syntax.`
                );
            }
            return {...item, score};
        });

        available.sort((a, b) => ascending ? a.score - b.score : b.score - a.score);
    }

    // Return top-ranked
    return available[0].idx;
}

/**
 * Update batch session state after assignment.
 *
 * @param {Object} config - Distribution strategy configuration
 * @param {number} listIndex - Assigned list index
 * @param {Object} state - Current batch session state
 * @returns {Promise<void>}
 */
async function updateStrategyState(config, listIndex, state) {
    // Update sequential counter
    if (config.strategy_type === 'sequential') {
        const currentIndex = state.strategy_state.next_index || 0;
        await jatos.batchSession.replace('strategy_state/next_index', currentIndex + 1);
    }

    // Update quota
    if (config.strategy_type === 'quota_based') {
        const quotas = state.strategy_state.remaining_quotas;
        await jatos.batchSession.replace(
            `strategy_state/remaining_quotas/${listIndex}`,
            quotas[listIndex] - 1
        );
    }

    // Update Latin square position
    if (config.strategy_type === 'latin_square') {
        const position = state.strategy_state.latin_square_position || 0;
        await jatos.batchSession.replace('strategy_state/latin_square_position', position + 1);
    }
}

/**
 * Assign list to participant.
 *
 * @param {string} workerId - JATOS worker ID
 * @param {Object} config - Distribution strategy configuration
 * @param {Array<Object>} lists - Array of experiment lists
 * @returns {Promise<number>} Assigned list index
 * @throws {Error} If assignment fails after retries
 */
async function assignList(workerId, config, lists) {
    // Check for existing assignment
    const allAssignments = jatos.batchSession.get('assignments') || {};
    const existingAssignment = allAssignments[workerId];

    if (existingAssignment) {
        console.log('Worker already assigned:', existingAssignment);
        return existingAssignment.list_index;
    }

    // Attempt assignment with retries
    const maxRetries = 5;
    for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
            // Get current state
            const state = {
                statistics: jatos.batchSession.get('statistics'),
                strategy_state: jatos.batchSession.get('strategy_state'),
                pool: jatos.batchSession.get('pool')
            };

            // Select list
            const listIndex = selectListForAssignment(config, lists, state);

            // Create assignment record
            const assignment = {
                list_index: listIndex,
                list_id: lists[listIndex].id,
                assigned_at: new Date().toISOString(),
                completed: false
            };

            // Update batch session
            allAssignments[workerId] = assignment;
            await jatos.batchSession.set('assignments', allAssignments);

            // Update statistics
            const stats = jatos.batchSession.get('statistics');
            stats.assignment_counts[listIndex] = (stats.assignment_counts[listIndex] || 0) + 1;
            stats.total_assignments += 1;
            await jatos.batchSession.set('statistics', stats);

            // Update strategy state
            await updateStrategyState(config, listIndex, state);

            console.log(`Assigned worker ${workerId} to list ${listIndex}`);
            return listIndex;

        } catch (error) {
            console.warn(`Assignment attempt ${attempt + 1} failed:`, error);
            if (attempt === maxRetries - 1) {
                throw new Error(
                    `Failed to assign list after ${maxRetries} retries. ` +
                    `Last error: ${error.message}. ` +
                    `This may indicate: (1) Concurrent modification conflicts, ` +
                    `(2) Network issues, or (3) JATOS batch session not available.`
                );
            }
            // Exponential backoff
            await sleep(Math.pow(2, attempt) * 100);
        }
    }
}

/**
 * Mark participant as completed.
 *
 * @param {string} workerId - JATOS worker ID
 * @returns {Promise<void>}
 */
async function markCompleted(workerId) {
    const allAssignments = jatos.batchSession.get('assignments') || {};
    const assignment = allAssignments[workerId];

    if (!assignment) {
        console.warn('No assignment found for worker:', workerId);
        return;
    }

    // Update assignment
    assignment.completed = true;
    allAssignments[workerId] = assignment;
    await jatos.batchSession.set('assignments', allAssignments);

    // Update statistics
    const stats = jatos.batchSession.get('statistics');
    stats.completion_counts[assignment.list_index] =
        (stats.completion_counts[assignment.list_index] || 0) + 1;
    stats.total_completions += 1;
    await jatos.batchSession.set('statistics', stats);
}

/**
 * ListDistributor class for managing list distribution.
 */
class ListDistributor {
    /**
     * Create a ListDistributor.
     *
     * @param {Object} config - Distribution strategy configuration
     * @param {Array<Object>} lists - Array of experiment lists
     */
    constructor(config, lists) {
        if (!config) {
            throw new Error(
                `ListDistributor requires config parameter. Got: ${config}. ` +
                `Pass the distribution_strategy from your config.`
            );
        }

        if (!lists || lists.length === 0) {
            throw new Error(
                `ListDistributor requires non-empty lists array. Got: ${lists}. ` +
                `Verify lists.jsonl was loaded correctly.`
            );
        }

        this.config = config;
        this.lists = lists;
        this.workerId = null;
        this.assignedListIndex = null;
    }

    /**
     * Initialize distributor and assign list to current worker.
     *
     * @returns {Promise<number>} Assigned list index
     * @throws {Error} If initialization or assignment fails
     */
    async initialize() {
        // Get worker ID
        this.workerId = jatos.workerId;

        if (!this.workerId) {
            throw new Error(
                `JATOS workerId not available. ` +
                `This experiment requires JATOS. ` +
                `Ensure you are running this through JATOS, not as a standalone file.`
            );
        }

        // Initialize batch session if needed
        await this._initializeBatchSession();

        // Debug mode: always return same list
        if (this.config.debug_mode) {
            console.log('Debug mode: assigning list', this.config.debug_list_index);
            this.assignedListIndex = this.config.debug_list_index || 0;
            return this.assignedListIndex;
        }

        // Assign list
        this.assignedListIndex = await assignList(this.workerId, this.config, this.lists);

        return this.assignedListIndex;
    }

    /**
     * Get the assigned list object.
     *
     * @returns {Object} ExperimentList object
     * @throws {Error} If list not yet assigned
     */
    getAssignedList() {
        if (this.assignedListIndex === null) {
            throw new Error(
                `List not yet assigned. Call initialize() first before getAssignedList().`
            );
        }

        if (this.assignedListIndex >= this.lists.length) {
            throw new Error(
                `Assigned list index ${this.assignedListIndex} out of bounds. ` +
                `Only ${this.lists.length} lists available. ` +
                `Check debug_list_index configuration.`
            );
        }

        return this.lists[this.assignedListIndex];
    }

    /**
     * Mark current participant as completed.
     *
     * @returns {Promise<void>}
     */
    async markCompleted() {
        if (this.workerId === null || this.assignedListIndex === null) {
            console.warn('Cannot mark completed: not initialized');
            return;
        }

        await markCompleted(this.workerId);
    }

    /**
     * Get current distribution statistics.
     *
     * @returns {Object} Statistics object
     */
    getStatistics() {
        return jatos.batchSession.get('statistics');
    }

    /**
     * Initialize batch session (with lock mechanism).
     *
     * @private
     * @returns {Promise<void>}
     */
    async _initializeBatchSession() {
        if (jatos.batchSession.defined('distribution/initialized')) {
            console.log('Batch session already initialized');
            return;
        }

        console.log('Initializing batch session...');

        // Acquire lock
        const lockAcquired = await this._acquireLock('init_lock');

        if (!lockAcquired) {
            // Another worker is initializing, wait
            await this._waitForInitialization();
            return;
        }

        try {
            // Double-check (may have been initialized while waiting for lock)
            if (jatos.batchSession.defined('distribution/initialized')) {
                return;
            }

            await initializeBatchSession(this.config, this.lists);
            console.log('Batch session initialized');

        } finally {
            await this._releaseLock('init_lock');
        }
    }

    /**
     * Acquire initialization lock.
     *
     * @private
     * @param {string} lockName - Name of lock
     * @param {number} timeout - Timeout in milliseconds
     * @returns {Promise<boolean>} True if lock acquired
     */
    async _acquireLock(lockName, timeout = 5000) {
        const startTime = Date.now();

        while (Date.now() - startTime < timeout) {
            try {
                const lockValue = jatos.batchSession.get(lockName);

                if (!lockValue) {
                    // Lock available, try to acquire
                    await jatos.batchSession.set(lockName, {
                        holder: this.workerId,
                        acquired_at: new Date().toISOString()
                    });
                    return true;
                }

                // Lock held, wait and retry
                await sleep(100);

            } catch (error) {
                console.warn('Error acquiring lock:', error);
                await sleep(100);
            }
        }

        console.warn(`Failed to acquire lock '${lockName}' within ${timeout}ms`);
        return false;
    }

    /**
     * Release initialization lock.
     *
     * @private
     * @param {string} lockName - Name of lock
     * @returns {Promise<void>}
     */
    async _releaseLock(lockName) {
        try {
            await jatos.batchSession.remove(lockName);
        } catch (error) {
            console.error('Error releasing lock:', error);
        }
    }

    /**
     * Wait for initialization to complete.
     *
     * @private
     * @param {number} timeout - Timeout in milliseconds
     * @returns {Promise<void>}
     * @throws {Error} If initialization timeout
     */
    async _waitForInitialization(timeout = 10000) {
        const startTime = Date.now();

        while (Date.now() - startTime < timeout) {
            if (jatos.batchSession.defined('distribution/initialized')) {
                return;
            }
            await sleep(200);
        }

        throw new Error(
            `Batch session initialization timeout (${timeout}ms). ` +
            `This may indicate: (1) Network issues, ` +
            `(2) JATOS server problems, or (3) Another worker is stuck. ` +
            `Check JATOS server logs.`
        );
    }
}
