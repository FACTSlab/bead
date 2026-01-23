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
    // Set distribution config
    await jatos.batchSession.set('distribution', {
        strategy_type: config.strategy_type,
        strategy_config: config.strategy_config || {},
        initialized: true,
        created_at: new Date().toISOString()
    });
    
    // Initialize statistics (all strategies use this)
    const assignment_counts = {};
    const completion_counts = {};
    for (let i = 0; i < lists.length; i++) {
        assignment_counts[i] = 0;
        completion_counts[i] = 0;
    }
    
    await jatos.batchSession.set('statistics', {
        assignment_counts,
        completion_counts,
        total_assignments: 0,
        total_completions: 0
    });
    
    // Initialize assignments
    await jatos.batchSession.set('assignments', {});
    
    // Strategy-specific initialization
    const maxParticipants = config.max_participants || 1000;
    
    switch(config.strategy_type) {
        case 'random':
            const randomQueue = initializeRandom(config, lists, maxParticipants);
            await jatos.batchSession.set('lists_queue', randomQueue);
            await jatos.batchSession.set('strategy_state', {});
            break;
            
        case 'sequential':
            const seqQueue = initializeSequential(config, lists, maxParticipants);
            await jatos.batchSession.set('lists_queue', seqQueue);
            await jatos.batchSession.set('strategy_state', {next_index: 0});
            break;
            
        case 'balanced':
            // No queue needed, uses statistics
            await jatos.batchSession.set('strategy_state', {});
            break;
            
        case 'latin_square':
            const {queue, matrix} = initializeLatinSquare(config, lists);
            await jatos.batchSession.set('lists_queue', queue);
            await jatos.batchSession.set('strategy_state', {
                latin_square_matrix: matrix,
                latin_square_position: 0
            });
            break;
            
        case 'stratified':
            // Validate factors
            if (!config.strategy_config.factors || config.strategy_config.factors.length === 0) {
                throw new Error(
                    `StratifiedConfig requires 'factors' in strategy_config. ` +
                    `Got: ${JSON.stringify(config.strategy_config)}. ` +
                    `Provide a list like ['condition', 'verb_type'].`
                );
            }
            await jatos.batchSession.set('strategy_state', {});
            break;
            
        case 'weighted_random':
            // Validate weight_expression
            if (!config.strategy_config.weight_expression) {
                throw new Error(
                    `WeightedRandomConfig requires 'weight_expression' in strategy_config. ` +
                    `Got: ${JSON.stringify(config.strategy_config)}. ` +
                    `Provide a JavaScript expression like 'list_metadata.priority || 1.0'.`
                );
            }
            await jatos.batchSession.set('strategy_state', {});
            break;
            
        case 'quota_based':
            if (!config.strategy_config.participants_per_list) {
                throw new Error(
                    `QuotaConfig requires 'participants_per_list' in strategy_config. ` +
                    `Got: ${JSON.stringify(config.strategy_config)}. ` +
                    `Add 'participants_per_list: <int>' to your distribution_strategy config.`
                );
            }
            const {queue: quotaQueue, quotas} = initializeQuotaBased(config, lists);
            await jatos.batchSession.set('lists_queue', quotaQueue);
            await jatos.batchSession.set('strategy_state', {remaining_quotas: quotas});
            break;
            
        case 'metadata_based':
            // Validate expressions
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
            await jatos.batchSession.set('strategy_state', {});
            break;
            
        default:
            throw new Error(
                `Unknown strategy type: '${config.strategy_type}'. ` +
                `Valid types: random, sequential, balanced, latin_square, ` +
                `stratified, weighted_random, quota_based, metadata_based.`
            );
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
 * Fisher-Yates shuffle algorithm for array randomization.
 *
 * @param {Array} array - Array to shuffle in place
 */
function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}

/**
 * Initialize queue for random strategy.
 *
 * @param {Object} config - Distribution strategy configuration
 * @param {Array<Object>} lists - Array of experiment lists
 * @param {number} maxParticipants - Maximum number of participants
 * @returns {Array<Object>} Queue of list assignments
 */
function initializeRandom(config, lists, maxParticipants) {
    const queue = [];
    const perList = Math.ceil((maxParticipants || 1000) / lists.length);
    for (let i = 0; i < lists.length; i++) {
        for (let j = 0; j < perList; j++) {
            queue.push({list_index: i, list_id: lists[i].id});
        }
    }
    shuffleArray(queue);  // Fisher-Yates shuffle
    return queue;
}

/**
 * Initialize queue for sequential strategy.
 *
 * @param {Object} config - Distribution strategy configuration
 * @param {Array<Object>} lists - Array of experiment lists
 * @param {number} maxParticipants - Maximum number of participants
 * @returns {Array<Object>} Queue of list assignments
 */
function initializeSequential(config, lists, maxParticipants) {
    const queue = [];
    for (let i = 0; i < (maxParticipants || 1000); i++) {
        const listIndex = i % lists.length;
        queue.push({list_index: listIndex, list_id: lists[listIndex].id});
    }
    return queue;
}

/**
 * Initialize queue and matrix for Latin square strategy.
 *
 * @param {Object} config - Distribution strategy configuration
 * @param {Array<Object>} lists - Array of experiment lists
 * @returns {Object} Object with queue and matrix
 */
function initializeLatinSquare(config, lists) {
    const matrix = generateBalancedLatinSquare(lists.length);
    const queue = [];
    
    // Generate queue from matrix
    for (let row = 0; row < matrix.length; row++) {
        for (let col = 0; col < matrix[row].length; col++) {
            const listIndex = matrix[row][col];
            queue.push({list_index: listIndex, list_id: lists[listIndex].id});
        }
    }
    
    return {queue, matrix};
}

/**
 * Initialize queue and quotas for quota-based strategy.
 *
 * @param {Object} config - Distribution strategy configuration
 * @param {Array<Object>} lists - Array of experiment lists
 * @returns {Object} Object with queue and quotas
 */
function initializeQuotaBased(config, lists) {
    const quota = config.strategy_config.participants_per_list;
    const quotas = {};
    const queue = [];
    
    for (let i = 0; i < lists.length; i++) {
        quotas[i] = quota;
        for (let j = 0; j < quota; j++) {
            queue.push({list_index: i, list_id: lists[i].id});
        }
    }
    
    shuffleArray(queue);  // Randomize order
    return {queue, quotas};
}

/**
 * Atomic queue update with retry using .fail() callbacks.
 *
 * @param {string} workerId - JATOS worker ID
 * @param {Object} selected - Selected list assignment {list_index, list_id}
 * @param {Array<Object>} updatedQueue - Updated queue array
 * @param {Array<Object>} lists - Array of experiment lists
 * @returns {Promise<number>} Assigned list index
 */
function updateQueueAtomically(workerId, selected, updatedQueue, lists) {
    return new Promise((resolve, reject) => {
        function attemptUpdate(retries = 5) {
            // Verify queue hasn't changed (optimistic locking)
            const currentQueue = jatos.batchSession.get('lists_queue') || [];
            
            // If queue changed significantly, retry
            if (Math.abs(currentQueue.length - updatedQueue.length) > 1) {
                if (retries > 0) {
                    setTimeout(() => attemptUpdate(retries - 1), 100 * (6 - retries));
                    return;
                }
                reject(new Error('Queue modified concurrently'));
                return;
            }
            
            // Update queue
            jatos.batchSession.set('lists_queue', updatedQueue)
                .then(() => {
                    // Record assignment
                    const assignments = jatos.batchSession.get('assignments') || {};
                    assignments[workerId] = {
                        list_index: selected.list_index,
                        list_id: selected.list_id,
                        assigned_at: new Date().toISOString(),
                        completed: false
                    };
                    return jatos.batchSession.set('assignments', assignments);
                })
                .then(() => {
                    // Update statistics
                    const stats = jatos.batchSession.get('statistics') || {
                        assignment_counts: {},
                        completion_counts: {},
                        total_assignments: 0,
                        total_completions: 0
                    };
                    stats.assignment_counts[selected.list_index] = 
                        (stats.assignment_counts[selected.list_index] || 0) + 1;
                    stats.total_assignments += 1;
                    return jatos.batchSession.set('statistics', stats);
                })
                .then(() => resolve(selected.list_index))
                .fail((error) => {
                    if (retries > 0) {
                        setTimeout(() => attemptUpdate(retries - 1), 100 * (6 - retries));
                    } else {
                        reject(new Error(`Failed to update queue: ${error.message}`));
                    }
                });
        }
        
        attemptUpdate();
    });
}

/**
 * Atomic statistics update with version checking.
 *
 * @param {string} workerId - JATOS worker ID
 * @param {number} listIndex - Assigned list index
 * @param {Object} oldCounts - Previous assignment counts
 * @param {Object} oldStats - Previous statistics object
 * @param {Array<Object>} lists - Array of experiment lists
 * @returns {Promise<number>} Assigned list index
 */
function updateStatisticsAtomically(workerId, listIndex, oldCounts, oldStats, lists) {
    return new Promise((resolve, reject) => {
        function attemptUpdate(retries = 5) {
            const currentStats = jatos.batchSession.get('statistics') || {};
            const currentCounts = currentStats.assignment_counts || {};
            
            // Optimistic locking: check if counts changed
            const expectedCount = oldCounts[listIndex] || 0;
            const actualCount = currentCounts[listIndex] || 0;
            
            if (actualCount !== expectedCount && retries > 0) {
                // Count changed, retry
                setTimeout(() => {
                    updateStatisticsAtomically(workerId, listIndex,
                        currentCounts, currentStats, lists).then(resolve).catch(reject);
                }, 100 * (6 - retries));
                return;
            }
            
            // Update
            currentStats.assignment_counts = currentCounts;
            currentStats.assignment_counts[listIndex] = 
                (currentStats.assignment_counts[listIndex] || 0) + 1;
            currentStats.total_assignments = (currentStats.total_assignments || 0) + 1;
            
            jatos.batchSession.set('statistics', currentStats)
                .then(() => {
                    // Record assignment
                    const assignments = jatos.batchSession.get('assignments') || {};
                    assignments[workerId] = {
                        list_index: listIndex,
                        list_id: lists[listIndex].id,
                        assigned_at: new Date().toISOString(),
                        completed: false
                    };
                    return jatos.batchSession.set('assignments', assignments);
                })
                .then(() => resolve(listIndex))
                .fail((error) => {
                    if (retries > 0) {
                        setTimeout(() => attemptUpdate(retries - 1), 100 * (6 - retries));
                    } else {
                        reject(new Error(`Failed to update statistics: ${error.message}`));
                    }
                });
        }
        
        attemptUpdate();
    });
}

/**
 * Atomic sequential update with retry.
 *
 * @param {string} workerId - JATOS worker ID
 * @param {Object} selected - Selected list assignment {list_index, list_id}
 * @param {number} nextIndex - Next index value
 * @param {Array<Object>} lists - Array of experiment lists
 * @returns {Promise<number>} Assigned list index
 */
function updateSequentialAtomically(workerId, selected, nextIndex, lists) {
    return new Promise((resolve, reject) => {
        function attemptUpdate(retries = 5) {
            const currentIndex = jatos.batchSession.get('strategy_state/next_index') || 0;
            
            // Optimistic locking: check if index changed
            if (currentIndex !== (nextIndex - 1) && retries > 0) {
                setTimeout(() => {
                    const newIndex = currentIndex + 1;
                    updateSequentialAtomically(workerId, selected, newIndex, lists)
                        .then(resolve).catch(reject);
                }, 100 * (6 - retries));
                return;
            }
            
            // Update index
            jatos.batchSession.replace('strategy_state/next_index', nextIndex)
                .then(() => {
                    // Record assignment
                    const assignments = jatos.batchSession.get('assignments') || {};
                    assignments[workerId] = {
                        list_index: selected.list_index,
                        list_id: selected.list_id,
                        assigned_at: new Date().toISOString(),
                        completed: false
                    };
                    return jatos.batchSession.set('assignments', assignments);
                })
                .then(() => {
                    // Update statistics
                    const stats = jatos.batchSession.get('statistics') || {
                        assignment_counts: {},
                        completion_counts: {},
                        total_assignments: 0,
                        total_completions: 0
                    };
                    stats.assignment_counts[selected.list_index] = 
                        (stats.assignment_counts[selected.list_index] || 0) + 1;
                    stats.total_assignments += 1;
                    return jatos.batchSession.set('statistics', stats);
                })
                .then(() => resolve(selected.list_index))
                .fail((error) => {
                    if (retries > 0) {
                        setTimeout(() => attemptUpdate(retries - 1), 100 * (6 - retries));
                    } else {
                        reject(new Error(`Failed to update sequential: ${error.message}`));
                    }
                });
        }
        
        attemptUpdate();
    });
}

/**
 * Atomic Latin square update with retry.
 *
 * @param {string} workerId - JATOS worker ID
 * @param {number} listIndex - Assigned list index
 * @param {number} nextPosition - Next position value
 * @param {Array<Object>} lists - Array of experiment lists
 * @returns {Promise<number>} Assigned list index
 */
function updateLatinSquareAtomically(workerId, listIndex, nextPosition, lists) {
    return new Promise((resolve, reject) => {
        function attemptUpdate(retries = 5) {
            const currentPosition = jatos.batchSession.get('strategy_state/latin_square_position') || 0;
            
            // Optimistic locking: check if position changed
            if (currentPosition !== (nextPosition - 1) && retries > 0) {
                setTimeout(() => {
                    const newPosition = currentPosition + 1;
                    updateLatinSquareAtomically(workerId, listIndex, newPosition, lists)
                        .then(resolve).catch(reject);
                }, 100 * (6 - retries));
                return;
            }
            
            // Update position
            jatos.batchSession.replace('strategy_state/latin_square_position', nextPosition)
                .then(() => {
                    // Record assignment
                    const assignments = jatos.batchSession.get('assignments') || {};
                    assignments[workerId] = {
                        list_index: listIndex,
                        list_id: lists[listIndex].id,
                        assigned_at: new Date().toISOString(),
                        completed: false
                    };
                    return jatos.batchSession.set('assignments', assignments);
                })
                .then(() => {
                    // Update statistics
                    const stats = jatos.batchSession.get('statistics') || {
                        assignment_counts: {},
                        completion_counts: {},
                        total_assignments: 0,
                        total_completions: 0
                    };
                    stats.assignment_counts[listIndex] = 
                        (stats.assignment_counts[listIndex] || 0) + 1;
                    stats.total_assignments += 1;
                    return jatos.batchSession.set('statistics', stats);
                })
                .then(() => resolve(listIndex))
                .fail((error) => {
                    if (retries > 0) {
                        setTimeout(() => attemptUpdate(retries - 1), 100 * (6 - retries));
                    } else {
                        reject(new Error(`Failed to update Latin square: ${error.message}`));
                    }
                });
        }
        
        attemptUpdate();
    });
}

/**
 * Atomic quota update with retry.
 *
 * @param {string} workerId - JATOS worker ID
 * @param {number} listIndex - Assigned list index
 * @param {Object} oldQuotas - Previous quotas object
 * @param {Array<Object>} lists - Array of experiment lists
 * @returns {Promise<number>} Assigned list index
 */
function updateQuotaAtomically(workerId, listIndex, oldQuotas, lists) {
    return new Promise((resolve, reject) => {
        function attemptUpdate(retries = 5) {
            const currentQuotas = jatos.batchSession.get('strategy_state/remaining_quotas') || {};
            
            // Optimistic locking: check if quota changed
            const expectedQuota = oldQuotas[listIndex] || 0;
            const actualQuota = currentQuotas[listIndex] || 0;
            
            if (actualQuota !== expectedQuota && retries > 0) {
                // Quota changed, retry
                setTimeout(() => {
                    updateQuotaAtomically(workerId, listIndex, currentQuotas, lists)
                        .then(resolve).catch(reject);
                }, 100 * (6 - retries));
                return;
            }
            
            if (actualQuota <= 0) {
                reject(new Error(`Quota exhausted for list ${listIndex}`));
                return;
            }
            
            // Update quota
            const newQuotas = {...currentQuotas};
            newQuotas[listIndex] = actualQuota - 1;
            
            jatos.batchSession.set('strategy_state/remaining_quotas', newQuotas)
                .then(() => {
                    // Record assignment
                    const assignments = jatos.batchSession.get('assignments') || {};
                    assignments[workerId] = {
                        list_index: listIndex,
                        list_id: lists[listIndex].id,
                        assigned_at: new Date().toISOString(),
                        completed: false
                    };
                    return jatos.batchSession.set('assignments', assignments);
                })
                .then(() => {
                    // Update statistics
                    const stats = jatos.batchSession.get('statistics') || {
                        assignment_counts: {},
                        completion_counts: {},
                        total_assignments: 0,
                        total_completions: 0
                    };
                    stats.assignment_counts[listIndex] = 
                        (stats.assignment_counts[listIndex] || 0) + 1;
                    stats.total_assignments += 1;
                    return jatos.batchSession.set('statistics', stats);
                })
                .then(() => resolve(listIndex))
                .fail((error) => {
                    if (retries > 0) {
                        setTimeout(() => attemptUpdate(retries - 1), 100 * (6 - retries));
                    } else {
                        reject(new Error(`Failed to update quota: ${error.message}`));
                    }
                });
        }
        
        attemptUpdate();
    });
}

/**
 * Unified assignment function routing to strategy-specific implementations.
 *
 * @param {string} workerId - JATOS worker ID
 * @param {Object} config - Distribution strategy configuration
 * @param {Array<Object>} lists - Array of experiment lists
 * @returns {Promise<number>} Assigned list index
 * @throws {Error} If strategy unknown or assignment fails
 */
async function assignList(workerId, config, lists) {
    // Check existing assignment (idempotency)
    const assignments = jatos.batchSession.get('assignments') || {};
    if (assignments[workerId]) {
        console.log('Worker already assigned:', assignments[workerId]);
        return assignments[workerId].list_index;
    }
    
    // Route to strategy-specific assignment
    switch(config.strategy_type) {
        case 'random':
            return assignRandom(workerId, config, lists);
        case 'sequential':
            return assignSequential(workerId, config, lists);
        case 'balanced':
            return assignBalanced(workerId, config, lists);
        case 'latin_square':
            return assignLatinSquare(workerId, config, lists);
        case 'stratified':
            return assignStratified(workerId, config, lists);
        case 'weighted_random':
            return assignWeightedRandom(workerId, config, lists);
        case 'quota_based':
            return assignQuotaBased(workerId, config, lists);
        case 'metadata_based':
            return assignMetadataBased(workerId, config, lists);
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
 * Random assignment strategy (queue-based).
 *
 * @param {string} workerId - JATOS worker ID
 * @param {Object} config - Distribution strategy configuration
 * @param {Array<Object>} lists - Available lists
 * @returns {Promise<number>} Assigned list index
 * @throws {Error} If no lists available
 */
async function assignRandom(workerId, config, lists) {
    const queue = jatos.batchSession.get('lists_queue') || [];
    if (queue.length === 0) {
        throw new Error(
            `No lists available in queue for random assignment. ` +
            `Verify lists.jsonl was generated and batch session initialized.`
        );
    }
    
    const randomIndex = Math.floor(Math.random() * queue.length);
    const selected = queue[randomIndex];
    const updatedQueue = queue.filter((_, idx) => idx !== randomIndex);
    
    // Atomic update with retry
    return updateQueueAtomically(workerId, selected, updatedQueue, lists);
}

/**
 * Sequential (round-robin) assignment strategy (queue-based).
 *
 * @param {string} workerId - JATOS worker ID
 * @param {Object} config - Distribution strategy configuration
 * @param {Array<Object>} lists - Available lists
 * @returns {Promise<number>} Assigned list index
 * @throws {Error} If queue exhausted
 */
async function assignSequential(workerId, config, lists) {
    const queue = jatos.batchSession.get('lists_queue') || [];
    const nextIndex = jatos.batchSession.get('strategy_state/next_index') || 0;
    
    if (nextIndex >= queue.length) {
        throw new Error(
            `Sequential queue exhausted (position ${nextIndex} >= queue length ${queue.length}). ` +
            `Increase max_participants or add more lists.`
        );
    }
    
    const selected = queue[nextIndex];
    
    // Atomic increment with retry
    return updateSequentialAtomically(workerId, selected, nextIndex + 1, lists);
}

/**
 * Balanced assignment strategy (assign to least-used list, state-based).
 *
 * @param {string} workerId - JATOS worker ID
 * @param {Object} config - Distribution strategy configuration
 * @param {Array<Object>} lists - Available lists
 * @returns {Promise<number>} Assigned list index
 * @throws {Error} If assignment fails after retries
 */
async function assignBalanced(workerId, config, lists) {
    // Retry loop with optimistic locking
    for (let attempt = 0; attempt < 5; attempt++) {
        const stats = jatos.batchSession.get('statistics') || {};
        const counts = stats.assignment_counts || {};
        
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
        
        const listIndex = minIndices[Math.floor(Math.random() * minIndices.length)];
        
        try {
            // Atomic update with version check
            const result = await updateStatisticsAtomically(workerId, listIndex, counts, stats, lists);
            return result;
        } catch (error) {
            if (attempt === 4) {
                throw new Error(
                    `Failed to assign balanced list after 5 retries. ` +
                    `Last error: ${error.message}. ` +
                    `This may indicate concurrent modification conflicts.`
                );
            }
            // Retry with backoff
            await sleep(100 * Math.pow(2, attempt));
        }
    }
    
    throw new Error('Failed to assign balanced list after retries');
}

/**
 * Latin square counterbalancing strategy (queue-based).
 *
 * @param {string} workerId - JATOS worker ID
 * @param {Object} config - Distribution strategy configuration
 * @param {Array<Object>} lists - Available lists
 * @returns {Promise<number>} Assigned list index
 * @throws {Error} If matrix not initialized
 */
async function assignLatinSquare(workerId, config, lists) {
    const matrix = jatos.batchSession.get('strategy_state/latin_square_matrix');
    const position = jatos.batchSession.get('strategy_state/latin_square_position') || 0;
    
    if (!matrix || !Array.isArray(matrix) || matrix.length === 0) {
        throw new Error(
            `Latin square matrix not initialized. ` +
            `Verify batch session was initialized correctly.`
        );
    }
    
    const row = position % matrix.length;
    const col = Math.floor(position / matrix.length) % matrix[0].length;
    const listIndex = matrix[row][col];
    
    // Atomic increment
    return updateLatinSquareAtomically(workerId, listIndex, position + 1, lists);
}

/**
 * Stratified assignment strategy (balance across factors, state-based).
 *
 * @param {string} workerId - JATOS worker ID
 * @param {Object} config - Distribution strategy configuration
 * @param {Array<Object>} lists - Available lists
 * @returns {Promise<number>} Assigned list index
 * @throws {Error} If factors not specified
 */
async function assignStratified(workerId, config, lists) {
    if (!config.strategy_config.factors || config.strategy_config.factors.length === 0) {
        throw new Error(
            `StratifiedConfig requires 'factors' in strategy_config. ` +
            `Got: ${JSON.stringify(config.strategy_config)}. ` +
            `Provide a list like ['condition', 'verb_type'].`
        );
    }
    
    // Retry loop with optimistic locking
    for (let attempt = 0; attempt < 5; attempt++) {
        const factors = config.strategy_config.factors;
        const stats = jatos.batchSession.get('statistics') || {};
        const counts = stats.assignment_counts || {};
        
        // Group lists by factor combinations
        const strata = {};
        for (let i = 0; i < lists.length; i++) {
            const key = factors.map(f => lists[i].list_metadata?.[f] || 'null').join('|');
            if (!strata[key]) strata[key] = [];
            strata[key].push(i);
        }
        
        // Find stratum with minimum total assignments
        let minCount = Infinity;
        let minStratumIndices = [];
        
        for (const [key, indices] of Object.entries(strata)) {
            const stratumCount = indices.reduce((sum, idx) => sum + (counts[idx] || 0), 0);
            if (stratumCount < minCount) {
                minCount = stratumCount;
                minStratumIndices = indices;
            }
        }
        
        const listIndex = minStratumIndices[Math.floor(Math.random() * minStratumIndices.length)];
        
        try {
            // Atomic update with retry
            const result = await updateStatisticsAtomically(workerId, listIndex, counts, stats, lists);
            return result;
        } catch (error) {
            if (attempt === 4) {
                throw new Error(
                    `Failed to assign stratified list after 5 retries. ` +
                    `Last error: ${error.message}.`
                );
            }
            await sleep(100 * Math.pow(2, attempt));
        }
    }
    
    throw new Error('Failed to assign stratified list after retries');
}

/**
 * Weighted random assignment strategy (state-based, weights from metadata).
 *
 * @param {string} workerId - JATOS worker ID
 * @param {Object} config - Distribution strategy configuration
 * @param {Array<Object>} lists - Available lists
 * @returns {Promise<number>} Assigned list index
 * @throws {Error} If weight_expression not specified
 */
async function assignWeightedRandom(workerId, config, lists) {
    if (!config.strategy_config.weight_expression) {
        throw new Error(
            `WeightedRandomConfig requires 'weight_expression' in strategy_config. ` +
            `Got: ${JSON.stringify(config.strategy_config)}. ` +
            `Provide a JavaScript expression like 'list_metadata.priority || 1.0'.`
        );
    }
    
    const expr = config.strategy_config.weight_expression;
    const normalize = config.strategy_config.normalize_weights !== false;
    
    // Compute weights from metadata
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
    
    // Sample from cumulative distribution
    const cumulative = [];
    let sum = 0;
    for (const weight of w) {
        sum += weight;
        cumulative.push(sum);
    }
    
    const random = Math.random() * cumulative[cumulative.length - 1];
    let listIndex = lists.length - 1;
    for (let i = 0; i < cumulative.length; i++) {
        if (random <= cumulative[i]) {
            listIndex = i;
            break;
        }
    }
    
    // Update statistics atomically
    const stats = jatos.batchSession.get('statistics') || {};
    const counts = stats.assignment_counts || {};
    return updateStatisticsAtomically(workerId, listIndex, counts, stats, lists);
}

/**
 * Quota-based assignment strategy (queue-based with quota tracking).
 *
 * @param {string} workerId - JATOS worker ID
 * @param {Object} config - Distribution strategy configuration
 * @param {Array<Object>} lists - Available lists
 * @returns {Promise<number>} Assigned list index
 * @throws {Error} If quotas exhausted and overflow not allowed
 */
async function assignQuotaBased(workerId, config, lists) {
    if (!config.strategy_config.participants_per_list) {
        throw new Error(
            `QuotaConfig requires 'participants_per_list' in strategy_config. ` +
            `Got: ${JSON.stringify(config.strategy_config)}. ` +
            `Add 'participants_per_list: <int>' to your distribution_strategy config.`
        );
    }
    
    const quotas = jatos.batchSession.get('strategy_state/remaining_quotas') || {};
    
    // Find available lists
    const available = [];
    for (let i = 0; i < lists.length; i++) {
        if (quotas[i] > 0) {
            available.push(i);
        }
    }
    
    if (available.length === 0) {
        if (config.strategy_config.allow_overflow === true) {
            // Fall back to balanced
            return assignBalanced(workerId, config, lists);
        } else {
            throw new Error(
                `All lists have reached quota and allow_overflow=false. ` +
                `Current quotas: ${JSON.stringify(quotas)}. ` +
                `Options: (1) Set allow_overflow: true, ` +
                `(2) Increase participants_per_list, or (3) Add more lists.`
            );
        }
    }
    
    const listIndex = available[Math.floor(Math.random() * available.length)];
    
    // Atomic decrement with retry
    return updateQuotaAtomically(workerId, listIndex, quotas, lists);
}

/**
 * Metadata-based assignment strategy (state-based, filters/ranks from metadata).
 *
 * @param {string} workerId - JATOS worker ID
 * @param {Object} config - Distribution strategy configuration
 * @param {Array<Object>} lists - Available lists
 * @returns {Promise<number>} Assigned list index
 * @throws {Error} If no lists match filter or expressions missing
 */
async function assignMetadataBased(workerId, config, lists) {
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
    
    const listIndex = available[0].idx;
    
    // Update statistics atomically
    const stats = jatos.batchSession.get('statistics') || {};
    const counts = stats.assignment_counts || {};
    return updateStatisticsAtomically(workerId, listIndex, counts, stats, lists);
}


/**
 * Mark participant as completed.
 *
 * @param {string} workerId - JATOS worker ID
 * @returns {Promise<void>}
 * @throws {Error} If update fails after retries
 */
async function markCompleted(workerId) {
    // Retry loop for atomic completion update
    for (let attempt = 0; attempt < 5; attempt++) {
        try {
            const allAssignments = jatos.batchSession.get('assignments') || {};
            const assignment = allAssignments[workerId];

            if (!assignment) {
                console.warn('No assignment found for worker:', workerId);
                return;
            }

            // Check if already completed (idempotency)
            if (assignment.completed) {
                console.log('Worker already marked as completed:', workerId);
                return;
            }

            // Update assignment
            assignment.completed = true;
            allAssignments[workerId] = assignment;
            await jatos.batchSession.set('assignments', allAssignments);

            // Update statistics atomically
            const stats = jatos.batchSession.get('statistics') || {
                assignment_counts: {},
                completion_counts: {},
                total_assignments: 0,
                total_completions: 0
            };
            stats.completion_counts[assignment.list_index] =
                (stats.completion_counts[assignment.list_index] || 0) + 1;
            stats.total_completions = (stats.total_completions || 0) + 1;
            await jatos.batchSession.set('statistics', stats);
            
            return;
        } catch (error) {
            if (attempt === 4) {
                throw new Error(
                    `Failed to mark worker ${workerId} as completed after 5 retries. ` +
                    `Last error: ${error.message}. ` +
                    `This may indicate network issues or JATOS server problems.`
                );
            }
            await sleep(100 * Math.pow(2, attempt));
        }
    }
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
        // Validate lists
        if (!this.lists || this.lists.length === 0) {
            throw new Error(
                `Cannot initialize: no lists available. ` +
                `Verify lists.jsonl was loaded correctly and contains at least one list.`
            );
        }

        // Get worker ID
        this.workerId = jatos.workerId;

        if (!this.workerId) {
            throw new Error(
                `JATOS workerId not available. ` +
                `This experiment requires JATOS. ` +
                `Ensure you are running this through JATOS, not as a standalone file.`
            );
        }

        // Validate config
        if (!this.config || !this.config.strategy_type) {
            throw new Error(
                `Invalid distribution config: missing strategy_type. ` +
                `Verify distribution.json was loaded correctly.`
            );
        }

        // Initialize batch session if needed
        try {
            await this._initializeBatchSession();
        } catch (error) {
            throw new Error(
                `Failed to initialize batch session: ${error.message}. ` +
                `This may indicate: (1) Network issues, ` +
                `(2) JATOS server problems, or (3) Invalid configuration.`
            );
        }

        // Debug mode: always return same list
        if (this.config.debug_mode) {
            const debugIndex = this.config.debug_list_index || 0;
            if (debugIndex < 0 || debugIndex >= this.lists.length) {
                throw new Error(
                    `Invalid debug_list_index: ${debugIndex}. ` +
                    `Must be between 0 and ${this.lists.length - 1}.`
                );
            }
            console.log('Debug mode: assigning list', debugIndex);
            this.assignedListIndex = debugIndex;
            return this.assignedListIndex;
        }

        // Assign list with error handling
        try {
            this.assignedListIndex = await assignList(this.workerId, this.config, this.lists);
            console.log(`Assigned worker ${this.workerId} to list ${this.assignedListIndex}`);
            return this.assignedListIndex;
        } catch (error) {
            throw new Error(
                `Failed to assign list: ${error.message}. ` +
                `Worker ID: ${this.workerId}, Strategy: ${this.config.strategy_type}. ` +
                `This may indicate: (1) All lists exhausted, ` +
                `(2) Concurrent modification conflicts, or (3) Network issues.`
            );
        }
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
     * @throws {Error} If marking completed fails
     */
    async markCompleted() {
        if (this.workerId === null || this.assignedListIndex === null) {
            console.warn('Cannot mark completed: not initialized');
            return;
        }

        try {
            await markCompleted(this.workerId);
            console.log(`Marked worker ${this.workerId} as completed`);
        } catch (error) {
            throw new Error(
                `Failed to mark worker ${this.workerId} as completed: ${error.message}. ` +
                `This may indicate network issues or JATOS server problems.`
            );
        }
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
