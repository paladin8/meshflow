use crate::coords::{Coord, Direction};

/// Identifies an SRAM slot in a PE's local memory.
pub type SlotId = u32;

/// A message carrying an activation payload through the mesh.
#[derive(Debug, Clone)]
pub struct Message {
    pub id: u64,
    /// Origin PE (retained for debugging/profiling, not used by event loop).
    pub source: Coord,
    /// Destination PE (retained for debugging/profiling, not used by event loop).
    pub dest: Coord,
    /// Pre-computed hop list from route generator.
    pub hops: Vec<Direction>,
    /// Progress through the hop list. When current_hop == hops.len(), message has arrived.
    pub current_hop: usize,
    /// Tensor data traveling inline with the message.
    pub payload: Vec<f32>,
    /// SRAM slot the payload is written to upon final delivery.
    pub payload_slot: SlotId,
    /// Logical timestamp when this message was created.
    pub timestamp: u64,
    /// Hop indices for intermediate broadcast delivery.
    /// Empty = point-to-point (deliver only at final destination).
    /// When non-empty, payload is delivered to intermediate PEs at these hop indices
    /// in addition to the final destination.
    pub deliver_at: Vec<usize>,
}

impl Message {
    /// Returns true if this message has reached its destination.
    pub fn is_arrived(&self) -> bool {
        self.current_hop >= self.hops.len()
    }

    /// Returns the next hop direction, or None if already arrived.
    pub fn next_hop(&self) -> Option<Direction> {
        self.hops.get(self.current_hop).copied()
    }

    /// Advance the hop counter by one.
    pub fn advance_hop(&mut self) {
        self.current_hop += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_message(hops: Vec<Direction>) -> Message {
        Message {
            id: 0,
            source: Coord::new(0, 0),
            dest: Coord::new(3, 2),
            hops,
            current_hop: 0,
            payload: vec![1.0, 2.0, 3.0],
            payload_slot: 0,
            timestamp: 0,
            deliver_at: vec![],
        }
    }

    #[test]
    fn empty_hops_is_arrived() {
        let msg = make_message(vec![]);
        assert!(msg.is_arrived());
        assert_eq!(msg.next_hop(), None);
    }

    #[test]
    fn single_hop_progression() {
        let mut msg = make_message(vec![Direction::East]);
        assert!(!msg.is_arrived());
        assert_eq!(msg.next_hop(), Some(Direction::East));

        msg.advance_hop();
        assert!(msg.is_arrived());
        assert_eq!(msg.next_hop(), None);
    }

    #[test]
    fn multi_hop_progression() {
        let mut msg = make_message(vec![Direction::East, Direction::East, Direction::North]);
        assert_eq!(msg.next_hop(), Some(Direction::East));
        msg.advance_hop();
        assert_eq!(msg.next_hop(), Some(Direction::East));
        msg.advance_hop();
        assert_eq!(msg.next_hop(), Some(Direction::North));
        msg.advance_hop();
        assert!(msg.is_arrived());
    }
}
