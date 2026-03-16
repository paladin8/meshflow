use std::collections::HashMap;

use crate::coords::Coord;

/// Per-PE counters accumulated during simulation.
#[derive(Debug, Clone, Default)]
pub struct PeCounters {
    pub messages_received: u64,
    pub messages_sent: u64,
    pub tasks_executed: u64,
    pub slots_written: u64,
}

/// Global profiling summary returned after simulation.
#[derive(Debug, Clone)]
pub struct ProfileSummary {
    pub total_messages: u64,
    pub total_hops: u64,
    pub total_events_processed: u64,
    pub total_tasks_executed: u64,
    pub final_timestamp: u64,
    pub per_pe: HashMap<Coord, PeCounters>,
}

impl ProfileSummary {
    pub fn new() -> Self {
        Self {
            total_messages: 0,
            total_hops: 0,
            total_events_processed: 0,
            total_tasks_executed: 0,
            final_timestamp: 0,
            per_pe: HashMap::new(),
        }
    }
}

impl Default for ProfileSummary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pe_counters_default_to_zero() {
        let c = PeCounters::default();
        assert_eq!(c.messages_received, 0);
        assert_eq!(c.messages_sent, 0);
        assert_eq!(c.tasks_executed, 0);
        assert_eq!(c.slots_written, 0);
    }

    #[test]
    fn profile_summary_starts_empty() {
        let p = ProfileSummary::new();
        assert_eq!(p.total_messages, 0);
        assert_eq!(p.total_hops, 0);
        assert!(p.per_pe.is_empty());
    }
}
