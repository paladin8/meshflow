use std::collections::{HashMap, VecDeque};

use crate::coords::Coord;
use crate::message::{Message, SlotId};
use crate::profiling::PeCounters;

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
    /// Optional capacity limit in bytes (not enforced in M1).
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
        hops: Vec<crate::coords::Direction>,
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
        hops: Vec<crate::coords::Direction>,
        fragment_slot: SlotId,
    },
    /// Accumulate output fragments into a pre-allocated buffer.
    /// Each trigger writes fragment data at offset trigger_slot * rows_per_fragment.
    /// When all fragments have arrived, stores the completed buffer as output.
    ConcatCollect {
        num_fragments: u32,
        rows_per_fragment: u32,
    },
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
    pub fn write_slot(&mut self, slot: SlotId, data: Vec<f32>) {
        self.sram.insert(slot, data);
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
