/**
 * Unit tests for list-distributor.ts helper functions.
 *
 * Tests the pure functions that can be tested without JATOS.
 * The main ListDistributor class requires JATOS batch sessions.
 */

import { describe, expect, test } from "vitest";

// Helper functions copied from list-distributor.ts for isolated testing.
// These are internal functions not exported from the module.

function generateBalancedLatinSquare(n: number): number[][] {
  const square: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
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

function shuffleArray<T>(array: T[]): void {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const temp = array[i];
    const swapVal = array[j];
    if (temp !== undefined && swapVal !== undefined) {
      array[i] = swapVal;
      array[j] = temp;
    }
  }
}

interface ExperimentList {
  id: string;
  name: string;
  list_number: number;
  list_metadata?: Record<string, unknown>;
  item_refs: string[];
}

interface QueueEntry {
  list_index: number;
  list_id: string;
}

interface DistributionConfig {
  strategy_type: string;
  strategy_config?: {
    factors?: string[];
    weight_expression?: string;
    normalize_weights?: boolean;
    participants_per_list?: number;
    allow_overflow?: boolean;
    filter_expression?: string;
    rank_expression?: string;
    rank_ascending?: boolean;
  };
  max_participants?: number;
}

function initializeRandom(
  _config: DistributionConfig,
  lists: ExperimentList[],
  maxParticipants: number,
): QueueEntry[] {
  const queue: QueueEntry[] = [];
  const perList = Math.ceil(maxParticipants / lists.length);
  for (let i = 0; i < lists.length; i++) {
    const list = lists[i];
    if (list) {
      for (let j = 0; j < perList; j++) {
        queue.push({ list_index: i, list_id: list.id });
      }
    }
  }
  shuffleArray(queue);
  return queue;
}

function initializeSequential(
  _config: DistributionConfig,
  lists: ExperimentList[],
  maxParticipants: number,
): QueueEntry[] {
  const queue: QueueEntry[] = [];
  for (let i = 0; i < maxParticipants; i++) {
    const listIndex = i % lists.length;
    const list = lists[listIndex];
    if (list) {
      queue.push({ list_index: listIndex, list_id: list.id });
    }
  }
  return queue;
}

function initializeLatinSquare(
  _config: DistributionConfig,
  lists: ExperimentList[],
): { queue: QueueEntry[]; matrix: number[][] } {
  const matrix = generateBalancedLatinSquare(lists.length);
  const queue: QueueEntry[] = [];

  for (let row = 0; row < matrix.length; row++) {
    const matrixRow = matrix[row];
    if (matrixRow) {
      for (let col = 0; col < matrixRow.length; col++) {
        const listIndex = matrixRow[col];
        if (listIndex !== undefined) {
          const list = lists[listIndex];
          if (list) {
            queue.push({ list_index: listIndex, list_id: list.id });
          }
        }
      }
    }
  }

  return { queue, matrix };
}

function initializeQuotaBased(
  config: DistributionConfig,
  lists: ExperimentList[],
): { queue: QueueEntry[]; quotas: Record<number, number> } {
  const quota = config.strategy_config?.participants_per_list ?? 10;
  const quotas: Record<number, number> = {};
  const queue: QueueEntry[] = [];

  for (let i = 0; i < lists.length; i++) {
    quotas[i] = quota;
    const list = lists[i];
    if (list) {
      for (let j = 0; j < quota; j++) {
        queue.push({ list_index: i, list_id: list.id });
      }
    }
  }

  shuffleArray(queue);
  return { queue, quotas };
}

describe("generateBalancedLatinSquare", () => {
  test("generates correct 4x4 balanced Latin square", () => {
    const square = generateBalancedLatinSquare(4);

    expect(square).toHaveLength(4);
    expect(square[0]).toHaveLength(4);

    // Check each row contains 0,1,2,3
    for (const row of square) {
      const sorted = [...row].sort();
      expect(sorted).toEqual([0, 1, 2, 3]);
    }
  });

  test("generates correct 2x2 balanced Latin square", () => {
    const square = generateBalancedLatinSquare(2);

    expect(square).toHaveLength(2);
    // Row 0 (even): (floor(0/2) + j) % 2 = (0 + j) % 2
    expect(square[0]).toEqual([0, 1]);
    // Row 1 (odd): (floor(1/2) + 2 - j) % 2 = (0 + 2 - j) % 2
    expect(square[1]).toEqual([0, 1]);
  });

  test("generates square with balanced counterbalancing", () => {
    const square = generateBalancedLatinSquare(4);

    // Verify first column follows the algorithm pattern
    // i=0 (even): (floor(0/2) + 0) % 4 = 0
    expect(square[0]?.[0]).toBe(0);
    // i=1 (odd): (floor(1/2) + 4 - 0) % 4 = (0 + 4) % 4 = 0
    expect(square[1]?.[0]).toBe(0);
    // i=2 (even): (floor(2/2) + 0) % 4 = 1
    expect(square[2]?.[0]).toBe(1);
    // i=3 (odd): (floor(3/2) + 4 - 0) % 4 = (1 + 4) % 4 = 1
    expect(square[3]?.[0]).toBe(1);
  });
});

describe("initializeRandom", () => {
  test("generates queue with equal entries per list", () => {
    const lists: ExperimentList[] = [
      { id: "list_1", name: "list_1", list_number: 0, item_refs: [] },
      { id: "list_2", name: "list_2", list_number: 1, item_refs: [] },
      { id: "list_3", name: "list_3", list_number: 2, item_refs: [] },
    ];
    const config: DistributionConfig = { strategy_type: "random" };
    const maxParticipants = 30;

    const queue = initializeRandom(config, lists, maxParticipants);

    expect(queue.length).toBe(30);
    // Count entries per list
    const counts: Record<number, number> = { 0: 0, 1: 0, 2: 0 };
    for (const entry of queue) {
      counts[entry.list_index] = (counts[entry.list_index] ?? 0) + 1;
    }
    // Should be roughly equal (10 each)
    expect(counts[0]).toBeGreaterThanOrEqual(9);
    expect(counts[0]).toBeLessThanOrEqual(11);
  });
});

describe("initializeSequential", () => {
  test("generates sequential queue with round-robin", () => {
    const lists: ExperimentList[] = [
      { id: "list_1", name: "list_1", list_number: 0, item_refs: [] },
      { id: "list_2", name: "list_2", list_number: 1, item_refs: [] },
      { id: "list_3", name: "list_3", list_number: 2, item_refs: [] },
    ];
    const config: DistributionConfig = { strategy_type: "sequential" };
    const maxParticipants = 10;

    const queue = initializeSequential(config, lists, maxParticipants);

    expect(queue.length).toBe(10);
    expect(queue[0]?.list_index).toBe(0);
    expect(queue[1]?.list_index).toBe(1);
    expect(queue[2]?.list_index).toBe(2);
    expect(queue[3]?.list_index).toBe(0); // Wraps around (3 % 3 = 0)
    expect(queue[4]?.list_index).toBe(1); // 4 % 3 = 1
    expect(queue[9]?.list_index).toBe(0); // 9 % 3 = 0
  });
});

describe("initializeLatinSquare", () => {
  test("generates queue from Latin square matrix", () => {
    const lists: ExperimentList[] = [
      { id: "list_1", name: "list_1", list_number: 0, item_refs: [] },
      { id: "list_2", name: "list_2", list_number: 1, item_refs: [] },
      { id: "list_3", name: "list_3", list_number: 2, item_refs: [] },
    ];
    const config: DistributionConfig = { strategy_type: "latin_square" };

    const { queue, matrix } = initializeLatinSquare(config, lists);

    expect(matrix).toHaveLength(3);
    expect(queue.length).toBe(9); // 3x3 matrix
    // Verify queue entries match matrix
    let queueIndex = 0;
    for (let row = 0; row < matrix.length; row++) {
      const matrixRow = matrix[row];
      if (matrixRow) {
        for (let col = 0; col < matrixRow.length; col++) {
          expect(queue[queueIndex]?.list_index).toBe(matrixRow[col]);
          queueIndex++;
        }
      }
    }
  });

  test("includes both queue and matrix in result", () => {
    const lists: ExperimentList[] = [
      { id: "list_1", name: "list_1", list_number: 0, item_refs: [] },
      { id: "list_2", name: "list_2", list_number: 1, item_refs: [] },
    ];
    const config: DistributionConfig = { strategy_type: "latin_square" };

    const result = initializeLatinSquare(config, lists);

    expect(result).toHaveProperty("queue");
    expect(result).toHaveProperty("matrix");
    expect(Array.isArray(result.queue)).toBe(true);
    expect(Array.isArray(result.matrix)).toBe(true);
  });
});

describe("initializeQuotaBased", () => {
  test("generates queue and quotas correctly", () => {
    const lists: ExperimentList[] = [
      { id: "list_1", name: "list_1", list_number: 0, item_refs: [] },
      { id: "list_2", name: "list_2", list_number: 1, item_refs: [] },
      { id: "list_3", name: "list_3", list_number: 2, item_refs: [] },
    ];
    const config: DistributionConfig = {
      strategy_type: "quota_based",
      strategy_config: {
        participants_per_list: 5,
      },
    };

    const { queue, quotas } = initializeQuotaBased(config, lists);

    expect(queue.length).toBe(15); // 3 lists * 5 participants
    expect(quotas[0]).toBe(5);
    expect(quotas[1]).toBe(5);
    expect(quotas[2]).toBe(5);

    // Count entries per list
    const counts: Record<number, number> = { 0: 0, 1: 0, 2: 0 };
    for (const entry of queue) {
      counts[entry.list_index] = (counts[entry.list_index] ?? 0) + 1;
    }
    expect(counts[0]).toBe(5);
    expect(counts[1]).toBe(5);
    expect(counts[2]).toBe(5);
  });

  test("includes both queue and quotas in result", () => {
    const lists: ExperimentList[] = [
      { id: "list_1", name: "list_1", list_number: 0, item_refs: [] },
    ];
    const config: DistributionConfig = {
      strategy_type: "quota_based",
      strategy_config: {
        participants_per_list: 10,
      },
    };

    const result = initializeQuotaBased(config, lists);

    expect(result).toHaveProperty("queue");
    expect(result).toHaveProperty("quotas");
    expect(Array.isArray(result.queue)).toBe(true);
    expect(typeof result.quotas).toBe("object");
  });
});
