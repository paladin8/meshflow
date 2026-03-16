use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::coords::Coord;
use crate::message::Message;

/// A simulation event to be processed by the runtime.
#[derive(Debug)]
pub struct Event {
    /// Logical timestamp when this event should be processed.
    pub timestamp: u64,
    /// What happens when this event is processed.
    pub kind: EventKind,
    /// Which PE this event targets.
    pub coord: Coord,
    /// Global monotonic sequence number for deterministic tie-breaking.
    pub sequence: u64,
}

/// The kind of event.
#[derive(Debug)]
pub enum EventKind {
    /// Deliver a message to a PE (may be intermediate hop or final delivery).
    DeliverMessage { message: Message },
    /// Execute a configured task on a PE.
    ExecuteTask { task_index: usize },
}

// Ordering for min-heap: (timestamp, y, x, sequence) ascending.
// BinaryHeap is a max-heap, so we reverse the ordering.

// PartialEq compares only ordering fields (timestamp, coord, sequence).
// Since sequence is globally unique, two distinct events are never equal.
impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        self.timestamp == other.timestamp
            && self.coord == other.coord
            && self.sequence == other.sequence
    }
}

impl Eq for Event {}

impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Event {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior in BinaryHeap (which is a max-heap).
        // Primary: timestamp (lower first)
        // Secondary: coord.y (lower first)
        // Tertiary: coord.x (lower first)
        // Quaternary: sequence (lower first)
        let forward = (self.timestamp, self.coord.y, self.coord.x, self.sequence).cmp(&(
            other.timestamp,
            other.coord.y,
            other.coord.x,
            other.sequence,
        ));
        forward.reverse()
    }
}

/// A priority queue of events, ordered by earliest timestamp first.
///
/// Uses reversed Ord on Event so BinaryHeap (max-heap) acts as a min-heap.
#[derive(Debug)]
pub struct EventQueue {
    heap: BinaryHeap<Event>,
    next_sequence: u64,
}

impl EventQueue {
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
            next_sequence: 0,
        }
    }

    /// Push an event, automatically assigning a sequence number for
    /// deterministic tie-breaking.
    pub fn push(&mut self, timestamp: u64, coord: Coord, kind: EventKind) {
        let seq = self.next_sequence;
        self.next_sequence += 1;
        self.heap.push(Event {
            timestamp,
            kind,
            coord,
            sequence: seq,
        });
    }

    /// Pop the earliest event (lowest timestamp, then lowest coord, then lowest sequence).
    pub fn pop(&mut self) -> Option<Event> {
        self.heap.pop()
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }
}

impl Default for EventQueue {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deliver_event(queue: &mut EventQueue, timestamp: u64, x: u32, y: u32) {
        let msg = Message {
            id: 0,
            source: Coord::new(0, 0),
            dest: Coord::new(x, y),
            hops: vec![],
            current_hop: 0,
            payload: vec![],
            payload_slot: 0,
            timestamp,
        };
        queue.push(
            timestamp,
            Coord::new(x, y),
            EventKind::DeliverMessage { message: msg },
        );
    }

    #[test]
    fn empty_queue() {
        let mut q = EventQueue::new();
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
        assert!(q.pop().is_none());
    }

    #[test]
    fn ordering_by_timestamp() {
        let mut q = EventQueue::new();
        deliver_event(&mut q, 10, 0, 0);
        deliver_event(&mut q, 5, 0, 0);
        deliver_event(&mut q, 20, 0, 0);

        assert_eq!(q.pop().unwrap().timestamp, 5);
        assert_eq!(q.pop().unwrap().timestamp, 10);
        assert_eq!(q.pop().unwrap().timestamp, 20);
    }

    #[test]
    fn tie_break_by_y_then_x() {
        let mut q = EventQueue::new();
        deliver_event(&mut q, 1, 2, 1); // (2,1)
        deliver_event(&mut q, 1, 0, 2); // (0,2)
        deliver_event(&mut q, 1, 1, 0); // (1,0)
        deliver_event(&mut q, 1, 0, 0); // (0,0)

        let e1 = q.pop().unwrap();
        assert_eq!((e1.coord.x, e1.coord.y), (0, 0));

        let e2 = q.pop().unwrap();
        assert_eq!((e2.coord.x, e2.coord.y), (1, 0));

        let e3 = q.pop().unwrap();
        assert_eq!((e3.coord.x, e3.coord.y), (2, 1));

        let e4 = q.pop().unwrap();
        assert_eq!((e4.coord.x, e4.coord.y), (0, 2));
    }

    #[test]
    fn tie_break_by_sequence_at_same_coord() {
        let mut q = EventQueue::new();
        // Same timestamp, same coord — should pop in insertion order
        deliver_event(&mut q, 1, 0, 0);
        deliver_event(&mut q, 1, 0, 0);
        deliver_event(&mut q, 1, 0, 0);

        let e1 = q.pop().unwrap();
        let e2 = q.pop().unwrap();
        let e3 = q.pop().unwrap();
        assert!(e1.sequence < e2.sequence);
        assert!(e2.sequence < e3.sequence);
    }

    #[test]
    fn execute_task_event() {
        let mut q = EventQueue::new();
        q.push(
            5,
            Coord::new(1, 1),
            EventKind::ExecuteTask { task_index: 3 },
        );

        let event = q.pop().unwrap();
        assert_eq!(event.timestamp, 5);
        assert_eq!(event.coord, Coord::new(1, 1));
        match event.kind {
            EventKind::ExecuteTask { task_index } => assert_eq!(task_index, 3),
            _ => panic!("expected ExecuteTask"),
        }
    }

    #[test]
    fn mixed_event_types_ordered_by_timestamp() {
        let mut q = EventQueue::new();
        q.push(
            10,
            Coord::new(0, 0),
            EventKind::ExecuteTask { task_index: 0 },
        );
        deliver_event(&mut q, 5, 0, 0);

        assert_eq!(q.pop().unwrap().timestamp, 5);
        assert_eq!(q.pop().unwrap().timestamp, 10);
    }
}
