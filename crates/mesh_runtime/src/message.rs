use crate::coords::Coord;

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
    /// Tensor data traveling inline with the message.
    pub payload: Vec<f32>,
    /// SRAM slot the payload is written to upon final delivery.
    pub payload_slot: SlotId,
    /// Logical timestamp when this message was created.
    pub timestamp: u64,
    /// Route color ID for link multiplexing (0 = uncolored).
    pub color: u32,
    /// Number of hops traversed so far (incremented on each forward).
    /// Used for total_hops profiling at final delivery.
    pub hop_count: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_message() -> Message {
        Message {
            id: 0,
            source: Coord::new(0, 0),
            dest: Coord::new(3, 2),
            payload: vec![1.0, 2.0, 3.0],
            payload_slot: 0,
            timestamp: 0,
            color: 0,
            hop_count: 0,
        }
    }

    #[test]
    fn message_fields() {
        let msg = make_message();
        assert_eq!(msg.hop_count, 0);
        assert_eq!(msg.color, 0);
        assert_eq!(msg.payload.len(), 3);
    }

    #[test]
    fn message_hop_count_increments() {
        let mut msg = make_message();
        msg.hop_count += 1;
        assert_eq!(msg.hop_count, 1);
        msg.hop_count += 1;
        assert_eq!(msg.hop_count, 2);
    }
}
