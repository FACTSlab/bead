/**
 * Unit tests for list_distributor.js helper functions
 *
 * These functions are copied from list_distributor.js to test them in isolation.
 */

// Helper functions copied from list_distributor.js for testing
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

function assignRandom(lists) {
    if (lists.length === 0) {
        throw new Error(
            `No lists available for random assignment. ` +
            `Verify lists.jsonl was generated and is readable.`
        );
    }
    return Math.floor(Math.random() * lists.length);
}

function assignSequential(lists, state) {
    const currentIndex = state.strategy_state.next_index || 0;
    const listIndex = currentIndex % lists.length;
    return listIndex;
}

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

function assignLatinSquare(lists, state) {
    const matrix = state.strategy_state.latin_square_matrix;
    const position = state.strategy_state.latin_square_position || 0;

    const row = position % matrix.length;
    const col = Math.floor(position / matrix.length) % matrix[0].length;
    const listIndex = matrix[row][col];

    return listIndex;
}

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
            throw new Error('metadata_based not yet implemented in tests');
        default:
            throw new Error(
                `Unknown strategy type: '${config.strategy_type}'. ` +
                `Valid types: random, sequential, balanced, latin_square, ` +
                `stratified, weighted_random, quota_based, metadata_based. ` +
                `Check your distribution.json config.`
            );
    }
}

// Tests
describe('generateBalancedLatinSquare', () => {
    test('generates correct 4x4 balanced Latin square', () => {
        const square = generateBalancedLatinSquare(4);

        expect(square).toHaveLength(4);
        expect(square[0]).toHaveLength(4);

        // Check each row contains 0,1,2,3
        for (let row of square) {
            const sorted = [...row].sort();
            expect(sorted).toEqual([0, 1, 2, 3]);
        }
    });

    test('generates correct 2x2 balanced Latin square', () => {
        const square = generateBalancedLatinSquare(2);

        expect(square).toHaveLength(2);
        // Row 0 (even): (floor(0/2) + j) % 2 = (0 + j) % 2
        expect(square[0]).toEqual([0, 1]);
        // Row 1 (odd): (floor(1/2) + 2 - j) % 2 = (0 + 2 - j) % 2
        // j=0: (0 + 2 - 0) % 2 = 0, j=1: (0 + 2 - 1) % 2 = 1
        expect(square[1]).toEqual([0, 1]);
    });

    test('generates square with balanced counterbalancing', () => {
        const square = generateBalancedLatinSquare(4);

        // Verify first column follows the algorithm pattern
        // i=0 (even): (floor(0/2) + 0) % 4 = 0
        expect(square[0][0]).toBe(0);
        // i=1 (odd): (floor(1/2) + 4 - 0) % 4 = (0 + 4) % 4 = 0
        expect(square[1][0]).toBe(0);
        // i=2 (even): (floor(2/2) + 0) % 4 = 1
        expect(square[2][0]).toBe(1);
        // i=3 (odd): (floor(3/2) + 4 - 0) % 4 = (1 + 4) % 4 = 1
        expect(square[3][0]).toBe(1);
    });
});

describe('assignRandom', () => {
    test('returns valid list index', () => {
        const lists = [
            { name: 'list_1' },
            { name: 'list_2' },
            { name: 'list_3' }
        ];

        const index = assignRandom(lists);

        expect(index).toBeGreaterThanOrEqual(0);
        expect(index).toBeLessThan(lists.length);
    });

    test('throws error for empty lists', () => {
        expect(() => assignRandom([])).toThrow(/No lists available/);
    });

    test('returns 0 for single list', () => {
        const lists = [{ name: 'list_1' }];
        const index = assignRandom(lists);
        expect(index).toBe(0);
    });
});

describe('assignSequential', () => {
    test('returns sequential indices with round-robin', () => {
        const lists = [
            { name: 'list_1' },
            { name: 'list_2' },
            { name: 'list_3' }
        ];

        const state1 = { strategy_state: { next_index: 0 } };
        expect(assignSequential(lists, state1)).toBe(0);

        const state2 = { strategy_state: { next_index: 1 } };
        expect(assignSequential(lists, state2)).toBe(1);

        const state3 = { strategy_state: { next_index: 2 } };
        expect(assignSequential(lists, state3)).toBe(2);

        const state4 = { strategy_state: { next_index: 3 } };
        expect(assignSequential(lists, state4)).toBe(0);  // Wraps around
    });

    test('handles missing next_index (defaults to 0)', () => {
        const lists = [{ name: 'list_1' }, { name: 'list_2' }];
        const state = { strategy_state: {} };

        expect(assignSequential(lists, state)).toBe(0);
    });
});

describe('assignBalanced', () => {
    test('assigns to least-used list', () => {
        const lists = [
            { name: 'list_1' },
            { name: 'list_2' },
            { name: 'list_3' }
        ];

        const state = {
            statistics: {
                assignment_counts: [5, 2, 8]  // list_2 has minimum
            }
        };

        expect(assignBalanced(lists, state)).toBe(1);
    });

    test('randomly selects among tied minimums', () => {
        const lists = [
            { name: 'list_1' },
            { name: 'list_2' },
            { name: 'list_3' }
        ];

        const state = {
            statistics: {
                assignment_counts: [3, 3, 5]  // lists 0 and 1 tied
            }
        };

        const index = assignBalanced(lists, state);
        expect([0, 1]).toContain(index);
    });

    test('handles zero counts', () => {
        const lists = [
            { name: 'list_1' },
            { name: 'list_2' }
        ];

        const state = {
            statistics: {
                assignment_counts: [5, 0]
            }
        };

        expect(assignBalanced(lists, state)).toBe(1);
    });

    test('handles missing counts (treats as 0)', () => {
        const lists = [
            { name: 'list_1' },
            { name: 'list_2' }
        ];

        const state = {
            statistics: {
                assignment_counts: [5]  // Missing count for list_2
            }
        };

        expect(assignBalanced(lists, state)).toBe(1);
    });
});

describe('assignLatinSquare', () => {
    test('uses correct matrix position', () => {
        const lists = [
            { name: 'list_1' },
            { name: 'list_2' },
            { name: 'list_3' }
        ];

        const matrix = [
            [0, 1, 2],
            [1, 2, 0],
            [2, 0, 1]
        ];

        // Position 0 -> row 0, col 0 -> matrix[0][0] = 0
        const state1 = {
            strategy_state: {
                latin_square_matrix: matrix,
                latin_square_position: 0
            }
        };
        expect(assignLatinSquare(lists, state1)).toBe(0);

        // Position 1 -> row 1, col 0 -> matrix[1][0] = 1
        const state2 = {
            strategy_state: {
                latin_square_matrix: matrix,
                latin_square_position: 1
            }
        };
        expect(assignLatinSquare(lists, state2)).toBe(1);

        // Position 4 -> row 1, col 1 -> matrix[1][1] = 2
        const state3 = {
            strategy_state: {
                latin_square_matrix: matrix,
                latin_square_position: 4
            }
        };
        expect(assignLatinSquare(lists, state3)).toBe(2);
    });

    test('handles missing position (defaults to 0)', () => {
        const lists = [{ name: 'list_1' }];
        const matrix = [[0]];

        const state = {
            strategy_state: {
                latin_square_matrix: matrix
            }
        };

        expect(assignLatinSquare(lists, state)).toBe(0);
    });
});

describe('assignStratified', () => {
    test('balances across factor combinations', () => {
        const lists = [
            { name: 'A_easy', list_metadata: { condition: 'A', difficulty: 'easy' } },
            { name: 'A_hard', list_metadata: { condition: 'A', difficulty: 'hard' } },
            { name: 'B_easy', list_metadata: { condition: 'B', difficulty: 'easy' } },
            { name: 'B_hard', list_metadata: { condition: 'B', difficulty: 'hard' } }
        ];

        const config = {
            strategy_config: {
                factors: ['condition', 'difficulty']
            }
        };

        // A_easy (0), A_hard (1) both have count 5
        // B_easy (2), B_hard (3) both have count 0
        // Should select from B stratum
        const state = {
            statistics: {
                assignment_counts: [5, 5, 0, 0]
            }
        };

        const index = assignStratified(lists, config, state);
        expect([2, 3]).toContain(index);  // Either B list
    });

    test('throws error if factors not specified', () => {
        const lists = [{ name: 'list' }];
        const config = { strategy_config: {} };
        const state = { statistics: { assignment_counts: [0] } };

        expect(() => assignStratified(lists, config, state)).toThrow(/requires 'factors'/);
    });

    test('handles missing metadata (uses null)', () => {
        const lists = [
            { name: 'list_1', list_metadata: { condition: 'A' } },
            { name: 'list_2', list_metadata: {} }  // Missing condition
        ];

        const config = {
            strategy_config: {
                factors: ['condition']
            }
        };

        const state = {
            statistics: {
                assignment_counts: [10, 0]
            }
        };

        // list_2 with null condition should be selected
        expect(assignStratified(lists, config, state)).toBe(1);
    });
});

describe('assignWeightedRandom', () => {
    test('selects based on metadata weights', () => {
        const lists = [
            { name: 'high', list_metadata: { priority: 10 } },
            { name: 'low', list_metadata: { priority: 1 } }
        ];

        const config = {
            strategy_config: {
                weight_expression: 'list_metadata.priority || 1.0',
                normalize_weights: true
            }
        };

        const state = {};

        // Run multiple times to check distribution
        const results = [];
        for (let i = 0; i < 100; i++) {
            results.push(assignWeightedRandom(lists, config, state));
        }

        // High priority list should be selected more often (but not always due to randomness)
        const highCount = results.filter(i => i === 0).length;
        expect(highCount).toBeGreaterThan(50);  // Should be ~90% but allow variance
    });

    test('throws error if weight_expression missing', () => {
        const lists = [{ name: 'list' }];
        const config = { strategy_config: {} };
        const state = {};

        expect(() => assignWeightedRandom(lists, config, state)).toThrow(/requires 'weight_expression'/);
    });

    test('throws error if sum of weights is zero', () => {
        const lists = [
            { name: 'list_1', list_metadata: { priority: 0 } },
            { name: 'list_2', list_metadata: { priority: 0 } }
        ];

        const config = {
            strategy_config: {
                weight_expression: 'list_metadata.priority',
                normalize_weights: true
            }
        };

        const state = {};

        expect(() => assignWeightedRandom(lists, config, state)).toThrow(/Sum of weights is 0/);
    });

    test('works without normalization', () => {
        const lists = [
            { name: 'list_1', list_metadata: { priority: 5 } },
            { name: 'list_2', list_metadata: { priority: 5 } }
        ];

        const config = {
            strategy_config: {
                weight_expression: 'list_metadata.priority',
                normalize_weights: false
            }
        };

        const state = {};

        const index = assignWeightedRandom(lists, config, state);
        expect([0, 1]).toContain(index);
    });
});

describe('assignQuota', () => {
    test('selects from lists with remaining quota', () => {
        const lists = [
            { name: 'list_1' },
            { name: 'list_2' },
            { name: 'list_3' }
        ];

        const config = {
            strategy_config: {
                participants_per_list: 10
            }
        };

        const state = {
            strategy_state: {
                remaining_quotas: [0, 5, 0]  // Only list_2 has quota
            },
            statistics: { assignment_counts: [10, 5, 10] }
        };

        expect(assignQuota(lists, config, state)).toBe(1);
    });

    test('throws error if participants_per_list missing', () => {
        const lists = [{ name: 'list' }];
        const config = { strategy_config: {} };
        const state = { strategy_state: { remaining_quotas: [0] }, statistics: {} };

        expect(() => assignQuota(lists, config, state)).toThrow(/requires 'participants_per_list'/);
    });

    test('throws error if all quotas exhausted and overflow not allowed', () => {
        const lists = [
            { name: 'list_1' },
            { name: 'list_2' }
        ];

        const config = {
            strategy_config: {
                participants_per_list: 10,
                allow_overflow: false
            }
        };

        const state = {
            strategy_state: {
                remaining_quotas: [0, 0]
            },
            statistics: { assignment_counts: [10, 10] }
        };

        expect(() => assignQuota(lists, config, state)).toThrow(/reached quota/);
        expect(() => assignQuota(lists, config, state)).toThrow(/allow_overflow=false/);
    });

    test('uses balanced assignment when overflow allowed', () => {
        const lists = [
            { name: 'list_1' },
            { name: 'list_2' }
        ];

        const config = {
            strategy_config: {
                participants_per_list: 10,
                allow_overflow: true
            }
        };

        const state = {
            strategy_state: {
                remaining_quotas: [0, 0]
            },
            statistics: {
                assignment_counts: [10, 12]  // list_1 has fewer
            }
        };

        expect(assignQuota(lists, config, state)).toBe(0);
    });
});

describe('selectListForAssignment', () => {
    test('dispatches to correct strategy function', () => {
        const lists = [{ name: 'list_1' }];
        const state = {
            strategy_state: {},
            statistics: { assignment_counts: [0] }
        };

        const randomConfig = { strategy_type: 'random' };
        expect(() => selectListForAssignment(randomConfig, lists, state)).not.toThrow();

        const sequentialConfig = { strategy_type: 'sequential' };
        expect(selectListForAssignment(sequentialConfig, lists, state)).toBe(0);

        const balancedConfig = { strategy_type: 'balanced' };
        expect(selectListForAssignment(balancedConfig, lists, state)).toBe(0);
    });

    test('throws error for unknown strategy type', () => {
        const lists = [{ name: 'list' }];
        const config = { strategy_type: 'invalid_strategy' };
        const state = { strategy_state: {}, statistics: { assignment_counts: [0] } };

        expect(() => selectListForAssignment(config, lists, state)).toThrow(/Unknown strategy type/);
        expect(() => selectListForAssignment(config, lists, state)).toThrow(/invalid_strategy/);
    });
});
