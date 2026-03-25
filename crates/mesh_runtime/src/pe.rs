use std::collections::{HashMap, VecDeque};

use crate::coords::{Coord, Direction};
use crate::message::{Message, SlotId};
use crate::profiling::PeCounters;

/// Activation function applied after fragment collection.
#[derive(Debug, Clone)]
pub enum Activation {
    ReLU,
}

impl Activation {
    /// Apply the activation function in-place.
    pub fn apply(&self, data: &mut [f32]) {
        match self {
            Activation::ReLU => {
                for v in data.iter_mut() {
                    *v = v.max(0.0);
                }
            }
        }
    }
}

/// A processing element on the 2D mesh.
///
/// Each PE has local SRAM (blob store), an input queue for arriving messages,
/// a set of configured tasks, and profiling counters.
#[derive(Debug, Clone)]
pub struct PE {
    pub coord: Coord,
    /// Local SRAM: named slots holding f32 vectors.
    sram: HashMap<SlotId, Vec<f32>>,
    /// Incoming messages waiting to be processed.
    pub input_queue: VecDeque<Message>,
    /// Task configurations assigned to this PE.
    pub tasks: Vec<TaskConfig>,
    /// Optional capacity limit in bytes. Panics on write if exceeded.
    pub sram_capacity_bytes: Option<usize>,
    /// Profiling counters.
    pub counters: PeCounters,
}

/// A task configuration assigned to a PE.
///
/// The `trigger_slot` determines when this task fires: whenever a message
/// payload is delivered to that SRAM slot, the task is scheduled for execution.
#[derive(Debug, Clone)]
pub struct TaskConfig {
    pub kind: TaskKind,
    pub trigger_slot: SlotId,
}

/// The kind of task and its parameters.
#[derive(Debug, Clone)]
pub enum TaskKind {
    /// Read payload from input_slot and emit a message to route_dest
    /// using the pre-computed hop list.
    ForwardActivation {
        input_slot: SlotId,
        route_dest: Coord,
        hops: Vec<Direction>,
    },
    /// Consume payload from input_slot and mark it as simulation output.
    CollectOutput { input_slot: SlotId },
    /// Compute y = W @ x + b using weight/bias from SRAM, route output
    /// fragment to the collect PE.
    Linear {
        input_slot: SlotId,
        weight_slot: SlotId,
        bias_slot: SlotId,
        tile_rows: u32,
        tile_cols: u32,
        route_dest: Coord,
        hops: Vec<Direction>,
        fragment_slot: SlotId,
        fragment_offset: u32,
    },
    /// Accumulate output fragments into a pre-allocated buffer.
    /// Each trigger writes fragment data at fragment_offset.
    /// When all fragments have arrived, stores the completed buffer as output.
    ConcatCollect {
        num_fragments: u32,
        total_rows: u32,
        fragment_offset: u32,
        /// Number of sequence positions (0 = infer as 1 for backward compat).
        num_positions: u32,
    },
    /// Accumulate fragments, apply activation, and broadcast to next layer's tile PEs.
    /// Used for intermediate layers in multi-layer MLPs.
    ConcatCollectForward {
        num_fragments: u32,
        total_rows: u32,
        fragment_offset: u32,
        activation: Option<Activation>,
        route_dests: Vec<(Coord, Vec<Direction>)>,
        /// Per-destination payload slots (which SRAM slot to deliver into).
        payload_slots: Vec<SlotId>,
        /// Number of sequence positions (0 = infer as 1 for backward compat).
        num_positions: u32,
        /// When true, scatter row i to destination i instead of broadcasting.
        scatter: bool,
    },
    /// Element-wise addition of two SRAM slots.
    Add {
        input_slot_a: SlotId,
        input_slot_b: SlotId,
        output_slot: SlotId,
        output_dests: Vec<(Coord, Vec<Direction>)>,
        payload_slots: Vec<SlotId>,
    },
    /// Numerically stable row-wise softmax (in-place on a single PE).
    Softmax {
        input_slot: SlotId,
        output_slot: SlotId,
    },
    /// Matrix-vector multiply: M @ v or M^T @ v.
    MatMul {
        matrix_slot: SlotId,
        vector_slot: SlotId,
        rows: u32,
        cols: u32,
        transpose: bool,
        output_slot: SlotId,
        output_dests: Vec<(Coord, Vec<Direction>)>,
        payload_slots: Vec<SlotId>,
    },
    /// RMSNorm phase 1: compute sum(x^2) for local slice, send to reduce PE.
    RmsNormPartialSum {
        input_slot: SlotId,
        reduce_dest: Coord,
        reduce_hops: Vec<Direction>,
        partial_sum_slot: SlotId,
        slice_offset: u32,
        slice_size: u32,
        /// Total features per position (0 = legacy single-position mode).
        /// When > 0 and data.len() > feature_count, the input is position-major
        /// and per-position partial sums are computed.
        feature_count: u32,
    },
    /// RMSNorm phase 2: apply x * scale * gamma using scale from reduce PE.
    RmsNormNormalize {
        input_slot: SlotId,
        scale_slot: SlotId,
        gamma_slot: SlotId,
        output_dests: Vec<(Coord, Vec<Direction>)>,
        payload_slots: Vec<SlotId>,
        slice_offset: u32,
        slice_size: u32,
    },
    /// RMSNorm reduce: accumulate partial sums, compute scale, broadcast.
    RmsNormReduce {
        num_tiles: u32,
        feature_count: u32,
        eps: f32,
        tile_dests: Vec<(Coord, Vec<Direction>)>,
        scale_slot: SlotId,
    },
}

impl std::fmt::Display for TaskKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskKind::ForwardActivation { .. } => write!(f, "forward_activation"),
            TaskKind::CollectOutput { .. } => write!(f, "collect_output"),
            TaskKind::Linear { .. } => write!(f, "linear"),
            TaskKind::ConcatCollect { .. } => write!(f, "concat_collect"),
            TaskKind::ConcatCollectForward { .. } => write!(f, "concat_collect_forward"),
            TaskKind::Add { .. } => write!(f, "add"),
            TaskKind::Softmax { .. } => write!(f, "softmax"),
            TaskKind::MatMul { .. } => write!(f, "mat_mul"),
            TaskKind::RmsNormPartialSum { .. } => write!(f, "rms_norm_partial_sum"),
            TaskKind::RmsNormNormalize { .. } => write!(f, "rms_norm_normalize"),
            TaskKind::RmsNormReduce { .. } => write!(f, "rms_norm_reduce"),
        }
    }
}

impl PE {
    pub fn new(coord: Coord) -> Self {
        Self {
            coord,
            sram: HashMap::new(),
            input_queue: VecDeque::new(),
            tasks: Vec::new(),
            sram_capacity_bytes: None,
            counters: PeCounters::default(),
        }
    }

    /// Write data to an SRAM slot (insert or overwrite).
    ///
    /// Panics if the write would cause total SRAM usage to exceed the
    /// configured `sram_capacity_bytes` limit.
    pub fn write_slot(&mut self, slot: SlotId, data: Vec<f32>) {
        self.sram.insert(slot, data);

        if let Some(limit) = self.sram_capacity_bytes {
            let used = self.sram_used_bytes();
            if used > limit {
                panic!(
                    "PE {}: SRAM capacity exceeded ({} bytes used, {} byte limit)",
                    self.coord, used, limit
                );
            }
        }
    }

    /// Total SRAM usage in bytes across all slots.
    pub fn sram_used_bytes(&self) -> usize {
        self.sram
            .values()
            .map(|v| v.len() * std::mem::size_of::<f32>())
            .sum()
    }

    /// Read data from an SRAM slot. Panics if the slot does not exist
    /// (missing slot indicates a compiler/setup bug).
    pub fn read_slot(&self, slot: SlotId) -> &Vec<f32> {
        self.sram
            .get(&slot)
            .unwrap_or_else(|| panic!("PE {}: read from empty SRAM slot {}", self.coord, slot))
    }

    /// Remove an SRAM slot, returning its data.
    pub fn remove_slot(&mut self, slot: SlotId) -> Option<Vec<f32>> {
        self.sram.remove(&slot)
    }

    /// Check if a slot exists in SRAM.
    pub fn has_slot(&self, slot: SlotId) -> bool {
        self.sram.contains_key(&slot)
    }

    /// Returns indices of tasks triggered by a write to the given slot.
    pub fn triggered_tasks(&self, slot: SlotId) -> Vec<usize> {
        self.tasks
            .iter()
            .enumerate()
            .filter(|(_, t)| t.trigger_slot == slot)
            .map(|(i, _)| i)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coords::Direction;

    fn make_pe() -> PE {
        PE::new(Coord::new(0, 0))
    }

    #[test]
    fn write_and_read_slot() {
        let mut pe = make_pe();
        pe.write_slot(0, vec![1.0, 2.0, 3.0]);
        assert_eq!(pe.read_slot(0), &vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn overwrite_slot() {
        let mut pe = make_pe();
        pe.write_slot(0, vec![1.0]);
        pe.write_slot(0, vec![9.0, 8.0]);
        assert_eq!(pe.read_slot(0), &vec![9.0, 8.0]);
    }

    #[test]
    #[should_panic(expected = "read from empty SRAM slot")]
    fn read_missing_slot_panics() {
        let pe = make_pe();
        pe.read_slot(42);
    }

    #[test]
    fn remove_slot() {
        let mut pe = make_pe();
        pe.write_slot(0, vec![1.0]);
        assert!(pe.has_slot(0));
        let data = pe.remove_slot(0);
        assert_eq!(data, Some(vec![1.0]));
        assert!(!pe.has_slot(0));
    }

    #[test]
    fn has_slot() {
        let mut pe = make_pe();
        assert!(!pe.has_slot(0));
        pe.write_slot(0, vec![]);
        assert!(pe.has_slot(0));
    }

    #[test]
    fn sram_capacity_enforced() {
        let mut pe = make_pe();
        // 3 floats * 4 bytes = 12 bytes; set limit to 16
        pe.sram_capacity_bytes = Some(16);
        pe.write_slot(0, vec![1.0, 2.0, 3.0]); // 12 bytes — fits
        pe.write_slot(1, vec![4.0]); // +4 = 16 bytes — fits exactly
    }

    #[test]
    #[should_panic(expected = "SRAM capacity exceeded")]
    fn sram_capacity_exceeded_panics() {
        let mut pe = make_pe();
        pe.sram_capacity_bytes = Some(16);
        pe.write_slot(0, vec![1.0, 2.0, 3.0]); // 12 bytes
        pe.write_slot(1, vec![4.0, 5.0]); // +8 = 20 bytes — exceeds 16
    }

    #[test]
    fn sram_overwrite_reclaims_space() {
        let mut pe = make_pe();
        pe.sram_capacity_bytes = Some(16);
        pe.write_slot(0, vec![1.0, 2.0, 3.0]); // 12 bytes
        pe.write_slot(0, vec![1.0]); // overwrite: now 4 bytes
        pe.write_slot(1, vec![2.0, 3.0, 4.0]); // +12 = 16 bytes — fits
    }

    #[test]
    fn sram_no_limit_allows_any_size() {
        let mut pe = make_pe();
        // sram_capacity_bytes is None by default
        pe.write_slot(0, vec![0.0; 100_000]); // should not panic
    }

    #[test]
    fn sram_used_bytes() {
        let mut pe = make_pe();
        assert_eq!(pe.sram_used_bytes(), 0);
        pe.write_slot(0, vec![1.0, 2.0]); // 8 bytes
        assert_eq!(pe.sram_used_bytes(), 8);
        pe.write_slot(1, vec![3.0]); // +4 = 12 bytes
        assert_eq!(pe.sram_used_bytes(), 12);
        pe.remove_slot(0);
        assert_eq!(pe.sram_used_bytes(), 4);
    }

    #[test]
    fn input_queue_fifo() {
        let mut pe = make_pe();
        let msg1 = crate::message::Message {
            id: 1,
            source: Coord::new(0, 0),
            dest: Coord::new(1, 0),
            hops: vec![],
            current_hop: 0,
            payload: vec![1.0],
            payload_slot: 0,
            timestamp: 0,
        };
        let msg2 = crate::message::Message {
            id: 2,
            ..msg1.clone()
        };

        pe.input_queue.push_back(msg1);
        pe.input_queue.push_back(msg2);

        assert_eq!(pe.input_queue.pop_front().unwrap().id, 1);
        assert_eq!(pe.input_queue.pop_front().unwrap().id, 2);
    }

    #[test]
    fn triggered_tasks_match() {
        let mut pe = make_pe();
        pe.tasks.push(TaskConfig {
            kind: TaskKind::ForwardActivation {
                input_slot: 0,
                route_dest: Coord::new(1, 0),
                hops: vec![Direction::East],
            },
            trigger_slot: 0,
        });
        pe.tasks.push(TaskConfig {
            kind: TaskKind::CollectOutput { input_slot: 5 },
            trigger_slot: 5,
        });

        assert_eq!(pe.triggered_tasks(0), vec![0]);
        assert_eq!(pe.triggered_tasks(5), vec![1]);
        assert!(pe.triggered_tasks(99).is_empty());
    }
}
