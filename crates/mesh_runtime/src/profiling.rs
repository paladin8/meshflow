use std::collections::HashMap;
use std::fmt;

use crate::coords::Coord;

/// Per-PE counters accumulated during simulation.
#[derive(Debug, Clone, Default)]
pub struct PeCounters {
    pub messages_received: u64,
    pub messages_sent: u64,
    pub tasks_executed: u64,
    pub slots_written: u64,
    pub max_queue_depth: u64,
}

/// The kind of trace event recorded during simulation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceEventKind {
    MessageDeliver,
    TaskExecute,
    MessageSend,
}

impl fmt::Display for TraceEventKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TraceEventKind::MessageDeliver => write!(f, "message_deliver"),
            TraceEventKind::TaskExecute => write!(f, "task_execute"),
            TraceEventKind::MessageSend => write!(f, "message_send"),
        }
    }
}

/// A single trace event recorded during simulation.
#[derive(Debug, Clone)]
pub struct TraceEvent {
    pub timestamp: u64,
    pub coord: Coord,
    pub kind: TraceEventKind,
    pub detail: String,
}

/// Timing record for a single task execution.
#[derive(Debug, Clone)]
pub struct OperatorTiming {
    pub task_kind: String,
    pub coord: Coord,
    pub start_ts: u64,
    pub end_ts: u64,
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
    pub trace_events: Vec<TraceEvent>,
    pub operator_timings: Vec<OperatorTiming>,
    pub link_counts: HashMap<(Coord, Coord), u64>,
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
            trace_events: Vec::new(),
            operator_timings: Vec::new(),
            link_counts: HashMap::new(),
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
        assert_eq!(c.max_queue_depth, 0);
    }

    #[test]
    fn profile_summary_starts_empty() {
        let p = ProfileSummary::new();
        assert_eq!(p.total_messages, 0);
        assert_eq!(p.total_hops, 0);
        assert!(p.per_pe.is_empty());
        assert!(p.trace_events.is_empty());
        assert!(p.operator_timings.is_empty());
        assert!(p.link_counts.is_empty());
    }
}
