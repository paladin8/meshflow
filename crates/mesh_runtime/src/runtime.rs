use std::collections::HashMap;

use crate::coords::Coord;
use crate::event::{EventKind, EventQueue};
use crate::mesh::Mesh;
use crate::message::{Message, SlotId};
use crate::pe::{TaskConfig, TaskKind};
use crate::profiling::{OperatorTiming, ProfileSummary, TraceEvent, TraceEventKind};
use crate::route::generate_route_xy;

/// Simulator configuration.
#[derive(Debug, Clone)]
pub struct SimConfig {
    pub width: u32,
    pub height: u32,
    pub hop_latency: u64,
    pub task_base_latency: u64,
    pub max_events: u64,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            width: 4,
            height: 4,
            hop_latency: 1,
            task_base_latency: 1,
            max_events: 100_000,
        }
    }
}

/// A message to inject into the simulation.
pub struct InjectMessage {
    pub source: Coord,
    pub dest: Coord,
    pub payload: Vec<f32>,
    pub payload_slot: SlotId,
}

/// A task to configure on a PE before simulation.
pub struct InjectTask {
    pub coord: Coord,
    pub kind: InjectTaskKind,
    pub trigger_slot: SlotId,
}

/// Task kind for injection (route hops are computed automatically).
#[derive(Debug, Clone)]
pub enum InjectTaskKind {
    ForwardActivation {
        input_slot: SlotId,
        route_dest: Coord,
    },
    CollectOutput {
        input_slot: SlotId,
    },
}

/// Simulation result: collected outputs and profiling data.
#[derive(Debug)]
pub struct SimResult {
    pub outputs: HashMap<Coord, Vec<f32>>,
    pub profile: ProfileSummary,
}

/// The main simulator. Configures the mesh, injects messages and tasks,
/// runs the event loop, and collects results.
pub struct Simulator {
    config: SimConfig,
    mesh: Mesh,
    queue: EventQueue,
    outputs: HashMap<Coord, Vec<f32>>,
    profile: ProfileSummary,
    next_message_id: u64,
    /// Current pending DeliverMessage event count per PE.
    pending_counts: HashMap<Coord, u64>,
    /// Peak pending count per PE (written to PeCounters at end).
    max_pending: HashMap<Coord, u64>,
}

impl Simulator {
    pub fn new(config: SimConfig) -> Self {
        Self {
            mesh: Mesh::new(config.width, config.height),
            queue: EventQueue::new(),
            outputs: HashMap::new(),
            profile: ProfileSummary::new(),
            next_message_id: 0,
            pending_counts: HashMap::new(),
            max_pending: HashMap::new(),
            config,
        }
    }

    /// Configure a task on a PE. Hop lists for ForwardActivation tasks
    /// are pre-computed here, not during event processing.
    pub fn add_task(&mut self, task: InjectTask) {
        let kind = match task.kind {
            InjectTaskKind::ForwardActivation {
                input_slot,
                route_dest,
            } => {
                let hops = generate_route_xy(task.coord, route_dest);
                TaskKind::ForwardActivation {
                    input_slot,
                    route_dest,
                    hops,
                }
            }
            InjectTaskKind::CollectOutput { input_slot } => TaskKind::CollectOutput { input_slot },
        };

        self.mesh.pe_mut(task.coord).tasks.push(TaskConfig {
            kind,
            trigger_slot: task.trigger_slot,
        });
    }

    /// Configure a task on a PE directly from a pre-built TaskConfig.
    /// Used by artifact loading where hops are already in the artifact.
    pub fn add_task_direct(&mut self, coord: Coord, task: TaskConfig) {
        self.mesh.pe_mut(coord).tasks.push(task);
    }

    /// Set SRAM capacity limit on a PE.
    pub fn set_sram_capacity(&mut self, coord: Coord, capacity_bytes: usize) {
        self.mesh.pe_mut(coord).sram_capacity_bytes = Some(capacity_bytes);
    }

    /// Write data to a PE's SRAM slot. Used for pre-loading weights
    /// from a compiled artifact.
    pub fn write_sram(&mut self, coord: Coord, slot: SlotId, data: Vec<f32>) {
        self.mesh.pe_mut(coord).write_slot(slot, data);
    }

    /// Inject a message into the simulation. Generates the hop list and
    /// enqueues the initial DeliverMessage event at timestamp 0.
    pub fn inject_message(&mut self, msg: InjectMessage) {
        let hops = generate_route_xy(msg.source, msg.dest);
        let id = self.next_message_id;
        self.next_message_id += 1;

        let message = Message {
            id,
            source: msg.source,
            dest: msg.dest,
            hops,
            current_hop: 0,
            payload: msg.payload,
            payload_slot: msg.payload_slot,
            timestamp: 0,
        };

        // Enqueue at the source PE at timestamp 0.
        // If same-PE (empty hops), the event loop will detect is_arrived()
        // and do final delivery. Otherwise it will forward hop by hop.
        // Note: no trace event for injection — it's setup, not a runtime send.
        self.track_enqueue(msg.source);
        self.queue
            .push(0, msg.source, EventKind::DeliverMessage { message });
    }

    /// Track a DeliverMessage enqueue for pending depth counting.
    fn track_enqueue(&mut self, coord: Coord) {
        let count = self.pending_counts.entry(coord).or_insert(0);
        *count += 1;
        let max = self.max_pending.entry(coord).or_insert(0);
        if *count > *max {
            *max = *count;
        }
    }

    /// Enqueue a DeliverMessage event with full profiling:
    /// pending depth tracking + MessageSend trace event.
    ///
    /// Use this for all runtime message sends. `inject_message` is the
    /// only caller that bypasses this (setup, not a runtime event).
    fn enqueue_deliver(
        &mut self,
        timestamp: u64,
        target: Coord,
        source_coord: Coord,
        message: Message,
    ) {
        self.track_enqueue(target);

        // Trace event
        self.profile.trace_events.push(TraceEvent {
            timestamp,
            coord: source_coord,
            kind: TraceEventKind::MessageSend,
            detail: format!("({},{})", message.dest.x, message.dest.y),
        });

        self.queue
            .push(timestamp, target, EventKind::DeliverMessage { message });
    }

    /// Track a DeliverMessage dequeue for pending depth counting.
    fn track_dequeue(&mut self, coord: Coord) {
        if let Some(count) = self.pending_counts.get_mut(&coord) {
            *count = count.saturating_sub(1);
        }
    }

    /// Run the simulation to completion and return results.
    pub fn run(mut self) -> SimResult {
        while let Some(event) = self.queue.pop() {
            if self.profile.total_events_processed >= self.config.max_events {
                break;
            }

            match event.kind {
                EventKind::DeliverMessage { mut message } => {
                    self.track_dequeue(event.coord);
                    self.profile.trace_events.push(TraceEvent {
                        timestamp: event.timestamp,
                        coord: event.coord,
                        kind: TraceEventKind::MessageDeliver,
                        detail: format!("({},{})", message.dest.x, message.dest.y),
                    });
                    self.process_deliver(event.timestamp, event.coord, &mut message);
                }
                EventKind::ExecuteTask { task_index } => {
                    self.process_execute(event.timestamp, event.coord, task_index);
                }
            }

            self.profile.total_events_processed += 1;
            self.profile.final_timestamp = event.timestamp;
        }

        // Merge max_pending into PeCounters before collecting
        for (coord, max_depth) in &self.max_pending {
            let pe = self.mesh.pe_mut(*coord);
            pe.counters.max_queue_depth = *max_depth;
        }

        // Collect per-PE counters into profile summary
        for pe in self.mesh.iter_pes() {
            self.profile.per_pe.insert(pe.coord, pe.counters.clone());
        }

        SimResult {
            outputs: self.outputs,
            profile: self.profile,
        }
    }

    fn process_deliver(&mut self, timestamp: u64, coord: Coord, message: &mut Message) {
        let pe = self.mesh.pe_mut(coord);
        pe.counters.messages_received += 1;

        if !message.is_arrived() {
            // INTERMEDIATE HOP: forward to next PE without touching SRAM
            let dir = message
                .next_hop()
                .expect("message not arrived but no next hop");
            message.advance_hop();

            let neighbor = coord
                .step(dir, self.config.width, self.config.height)
                .unwrap_or_else(|| {
                    panic!(
                        "Route goes out of bounds: PE {} stepping {:?} on {}x{} mesh",
                        coord, dir, self.config.width, self.config.height
                    )
                });

            pe.counters.messages_sent += 1;

            // Track link usage
            *self
                .profile
                .link_counts
                .entry((coord, neighbor))
                .or_insert(0) += 1;

            // Move message into the next event
            let forwarded = std::mem::take(&mut message.payload);
            let mut new_message = message.clone();
            new_message.payload = forwarded;

            self.enqueue_deliver(
                timestamp + self.config.hop_latency,
                neighbor,
                coord,
                new_message,
            );
        } else {
            // FINAL DELIVERY: write payload to SRAM, trigger tasks
            let payload = std::mem::take(&mut message.payload);
            let payload_slot = message.payload_slot;
            let hop_count = message.hops.len() as u64;

            pe.write_slot(payload_slot, payload);
            pe.counters.slots_written += 1;
            self.profile.total_messages += 1;
            self.profile.total_hops += hop_count;

            // Check triggered tasks
            let triggered = pe.triggered_tasks(payload_slot);
            for task_index in triggered {
                self.queue.push(
                    timestamp + self.config.task_base_latency,
                    coord,
                    EventKind::ExecuteTask { task_index },
                );
            }
        }
    }

    fn process_execute(&mut self, timestamp: u64, coord: Coord, task_index: usize) {
        // Clone the task to avoid borrow conflicts with the PE
        let task = self.mesh.pe(coord).tasks[task_index].kind.clone();

        // Record operator timing and trace event
        let task_kind_str = task.to_string();
        self.profile.operator_timings.push(OperatorTiming {
            task_kind: task_kind_str.clone(),
            coord,
            start_ts: timestamp,
            end_ts: timestamp + self.config.task_base_latency,
        });
        self.profile.trace_events.push(TraceEvent {
            timestamp,
            coord,
            kind: TraceEventKind::TaskExecute,
            detail: task_kind_str,
        });

        match task {
            TaskKind::ForwardActivation {
                input_slot,
                route_dest,
                hops,
            } => {
                let pe = self.mesh.pe_mut(coord);
                let data = pe.read_slot(input_slot).clone();
                pe.counters.tasks_executed += 1;
                pe.counters.messages_sent += 1;
                self.profile.total_tasks_executed += 1;

                let message = Message {
                    id: self.next_message_id,
                    source: coord,
                    dest: route_dest,
                    hops,
                    current_hop: 0,
                    payload: data,
                    payload_slot: 0, // convention: deliver to slot 0
                    timestamp,
                };
                self.next_message_id += 1;

                // Enqueue at current PE — the event loop will forward or
                // deliver locally depending on whether hops is empty.
                self.enqueue_deliver(timestamp, coord, coord, message);
            }
            TaskKind::CollectOutput { input_slot } => {
                let pe = self.mesh.pe_mut(coord);
                let data = pe.read_slot(input_slot).clone();
                pe.counters.tasks_executed += 1;
                self.profile.total_tasks_executed += 1;

                self.outputs.insert(coord, data);
            }
            TaskKind::Linear {
                input_slot,
                weight_slot,
                bias_slot,
                tile_rows,
                tile_cols,
                route_dest,
                hops,
                fragment_slot,
                fragment_offset: _,
            } => {
                let pe = self.mesh.pe_mut(coord);
                let x = pe.read_slot(input_slot).clone();
                let w = pe.read_slot(weight_slot).clone();
                let b = pe.read_slot(bias_slot).clone();
                pe.counters.tasks_executed += 1;
                pe.counters.messages_sent += 1;
                self.profile.total_tasks_executed += 1;

                // Compute y = W @ x + b
                let rows = tile_rows as usize;
                let cols = tile_cols as usize;
                let mut y = Vec::with_capacity(rows);
                for i in 0..rows {
                    let mut sum = b[i];
                    for j in 0..cols {
                        sum += w[i * cols + j] * x[j];
                    }
                    y.push(sum);
                }

                let message = Message {
                    id: self.next_message_id,
                    source: coord,
                    dest: route_dest,
                    hops,
                    current_hop: 0,
                    payload: y,
                    payload_slot: fragment_slot,
                    timestamp,
                };
                self.next_message_id += 1;

                self.enqueue_deliver(timestamp, coord, coord, message);
            }
            TaskKind::ConcatCollect {
                num_fragments,
                total_rows,
                fragment_offset,
            } => {
                let trigger_slot = self.mesh.pe(coord).tasks[task_index].trigger_slot;
                let completed = self.process_concat_fragment(
                    coord,
                    trigger_slot,
                    num_fragments,
                    total_rows,
                    fragment_offset,
                );
                if let Some(result) = completed {
                    self.outputs.insert(coord, result);
                }
            }
            TaskKind::ConcatCollectForward {
                num_fragments,
                total_rows,
                fragment_offset,
                ref activation,
                ref route_dests,
            } => {
                let trigger_slot = self.mesh.pe(coord).tasks[task_index].trigger_slot;
                let activation = activation.clone();
                let route_dests = route_dests.clone();
                let completed = self.process_concat_fragment(
                    coord,
                    trigger_slot,
                    num_fragments,
                    total_rows,
                    fragment_offset,
                );
                if let Some(mut result) = completed {
                    // Apply activation function if specified
                    if let Some(act) = &activation {
                        act.apply(&mut result);
                    }

                    // Broadcast to next layer's tile PEs with send serialization:
                    // each send costs 1 time unit.
                    // Update PE counters first to avoid borrow conflicts.
                    let num_sends = route_dests.len() as u64;
                    self.mesh.pe_mut(coord).counters.messages_sent += num_sends;

                    for (i, (dest, hops)) in route_dests.iter().enumerate() {
                        let send_time = timestamp + self.config.task_base_latency + i as u64;
                        let message = Message {
                            id: self.next_message_id,
                            source: coord,
                            dest: *dest,
                            hops: hops.clone(),
                            current_hop: 0,
                            payload: result.clone(),
                            payload_slot: 0, // deliver to input slot 0 on tile PEs
                            timestamp: send_time,
                        };
                        self.next_message_id += 1;
                        self.enqueue_deliver(send_time, coord, coord, message);
                    }
                }
            }
        }
    }

    /// Shared accumulator logic for ConcatCollect and ConcatCollectForward.
    /// Returns Some(completed_buffer) when all fragments have arrived, None otherwise.
    fn process_concat_fragment(
        &mut self,
        coord: Coord,
        trigger_slot: SlotId,
        num_fragments: u32,
        total_rows: u32,
        fragment_offset: u32,
    ) -> Option<Vec<f32>> {
        let pe = self.mesh.pe_mut(coord);
        let fragment = pe.read_slot(trigger_slot).clone();
        pe.counters.tasks_executed += 1;
        self.profile.total_tasks_executed += 1;

        let total_size = total_rows as usize;
        let offset = fragment_offset as usize;
        let frag_len = fragment.len();

        // Use reserved SRAM slots for accumulator buffer and counter.
        let accum_slot: SlotId = u32::MAX;
        let counter_slot: SlotId = u32::MAX - 1;

        // Initialize accumulator on first fragment
        if !pe.has_slot(accum_slot) {
            pe.write_slot(accum_slot, vec![0.0; total_size]);
            pe.write_slot(counter_slot, vec![0.0]);
        }

        // Write fragment into accumulator at correct offset, then
        // remove the fragment slot to keep SRAM usage at O(1).
        {
            let mut buf = pe.read_slot(accum_slot).clone();
            buf[offset..offset + frag_len].copy_from_slice(&fragment);
            pe.write_slot(accum_slot, buf);
            pe.remove_slot(trigger_slot);
        }

        // Increment counter
        let count = {
            let c = pe.read_slot(counter_slot)[0] as u32 + 1;
            pe.write_slot(counter_slot, vec![c as f32]);
            c
        };

        // All fragments collected — return completed buffer
        if count == num_fragments {
            let result = pe.remove_slot(accum_slot).unwrap();
            pe.remove_slot(counter_slot);
            Some(result)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sim(width: u32, height: u32) -> Simulator {
        Simulator::new(SimConfig {
            width,
            height,
            ..Default::default()
        })
    }

    #[test]
    fn empty_simulation() {
        let s = sim(2, 2);
        let result = s.run();
        assert!(result.outputs.is_empty());
        assert_eq!(result.profile.total_events_processed, 0);
        assert_eq!(result.profile.total_messages, 0);
    }

    #[test]
    fn single_hop_delivery() {
        let mut s = sim(4, 4);
        s.add_task(InjectTask {
            coord: Coord::new(1, 0),
            kind: InjectTaskKind::CollectOutput { input_slot: 0 },
            trigger_slot: 0,
        });
        s.inject_message(InjectMessage {
            source: Coord::new(0, 0),
            dest: Coord::new(1, 0),
            payload: vec![1.0, 2.0, 3.0],
            payload_slot: 0,
        });

        let result = s.run();
        assert_eq!(
            result.outputs.get(&Coord::new(1, 0)),
            Some(&vec![1.0, 2.0, 3.0])
        );
        assert_eq!(result.profile.total_hops, 1);
        assert_eq!(result.profile.total_messages, 1);
        // Events: deliver at (0,0) [intermediate], deliver at (1,0) [final], execute collect
        assert_eq!(result.profile.total_events_processed, 3);
    }

    #[test]
    fn multi_hop_delivery() {
        let mut s = sim(4, 4);
        s.add_task(InjectTask {
            coord: Coord::new(3, 2),
            kind: InjectTaskKind::CollectOutput { input_slot: 0 },
            trigger_slot: 0,
        });
        s.inject_message(InjectMessage {
            source: Coord::new(0, 0),
            dest: Coord::new(3, 2),
            payload: vec![42.0],
            payload_slot: 0,
        });

        let result = s.run();
        assert_eq!(result.outputs.get(&Coord::new(3, 2)), Some(&vec![42.0]));
        assert_eq!(result.profile.total_hops, 5);
        // 5 intermediate hops + 1 final delivery + 1 collect task
        assert_eq!(result.profile.total_events_processed, 7);
        // final_timestamp = 5 hops * 1 hop_latency + 1 task_base_latency
        assert_eq!(result.profile.final_timestamp, 6);
    }

    #[test]
    fn same_pe_delivery() {
        let mut s = sim(2, 2);
        s.add_task(InjectTask {
            coord: Coord::new(0, 0),
            kind: InjectTaskKind::CollectOutput { input_slot: 0 },
            trigger_slot: 0,
        });
        s.inject_message(InjectMessage {
            source: Coord::new(0, 0),
            dest: Coord::new(0, 0),
            payload: vec![7.0, 8.0],
            payload_slot: 0,
        });

        let result = s.run();
        assert_eq!(result.outputs.get(&Coord::new(0, 0)), Some(&vec![7.0, 8.0]));
        assert_eq!(result.profile.total_hops, 0);
        assert_eq!(result.profile.total_messages, 1);
    }

    #[test]
    fn forward_activation_chain() {
        // (0,0) -> forward -> (2,0) -> forward -> (4,0) -> collect
        let mut s = sim(5, 1);

        s.add_task(InjectTask {
            coord: Coord::new(0, 0),
            kind: InjectTaskKind::ForwardActivation {
                input_slot: 0,
                route_dest: Coord::new(2, 0),
            },
            trigger_slot: 0,
        });
        s.add_task(InjectTask {
            coord: Coord::new(2, 0),
            kind: InjectTaskKind::ForwardActivation {
                input_slot: 0,
                route_dest: Coord::new(4, 0),
            },
            trigger_slot: 0,
        });
        s.add_task(InjectTask {
            coord: Coord::new(4, 0),
            kind: InjectTaskKind::CollectOutput { input_slot: 0 },
            trigger_slot: 0,
        });

        s.inject_message(InjectMessage {
            source: Coord::new(0, 0),
            dest: Coord::new(0, 0),
            payload: vec![1.0, 2.0, 3.0],
            payload_slot: 0,
        });

        let result = s.run();
        assert_eq!(
            result.outputs.get(&Coord::new(4, 0)),
            Some(&vec![1.0, 2.0, 3.0]),
            "Payload should arrive intact at end of chain"
        );
    }

    #[test]
    fn multiple_concurrent_messages() {
        let mut s = sim(4, 4);

        // Two independent message/collect pairs
        s.add_task(InjectTask {
            coord: Coord::new(3, 0),
            kind: InjectTaskKind::CollectOutput { input_slot: 0 },
            trigger_slot: 0,
        });
        s.add_task(InjectTask {
            coord: Coord::new(0, 3),
            kind: InjectTaskKind::CollectOutput { input_slot: 0 },
            trigger_slot: 0,
        });

        s.inject_message(InjectMessage {
            source: Coord::new(0, 0),
            dest: Coord::new(3, 0),
            payload: vec![10.0],
            payload_slot: 0,
        });
        s.inject_message(InjectMessage {
            source: Coord::new(0, 0),
            dest: Coord::new(0, 3),
            payload: vec![20.0],
            payload_slot: 0,
        });

        let result = s.run();
        assert_eq!(result.outputs.get(&Coord::new(3, 0)), Some(&vec![10.0]));
        assert_eq!(result.outputs.get(&Coord::new(0, 3)), Some(&vec![20.0]));
        assert_eq!(result.profile.total_messages, 2);
    }

    #[test]
    fn profiling_accuracy() {
        // Single message: (0,0) -> (2,0), 2 hops
        let mut s = sim(3, 1);
        s.add_task(InjectTask {
            coord: Coord::new(2, 0),
            kind: InjectTaskKind::CollectOutput { input_slot: 0 },
            trigger_slot: 0,
        });
        s.inject_message(InjectMessage {
            source: Coord::new(0, 0),
            dest: Coord::new(2, 0),
            payload: vec![1.0],
            payload_slot: 0,
        });

        let result = s.run();

        // Global counters
        assert_eq!(result.profile.total_hops, 2);
        assert_eq!(result.profile.total_messages, 1);
        // collect task
        assert_eq!(result.profile.total_tasks_executed, 1);
        // Events: deliver@(0,0) [fwd], deliver@(1,0) [fwd], deliver@(2,0) [final], execute collect
        assert_eq!(result.profile.total_events_processed, 4);
        // 2 hops * 1 latency + 1 task latency = 3
        assert_eq!(result.profile.final_timestamp, 3);

        // Per-PE counters
        let pe00 = result.profile.per_pe.get(&Coord::new(0, 0)).unwrap();
        assert_eq!(pe00.messages_received, 1); // initial delivery
        assert_eq!(pe00.messages_sent, 1); // forwarded

        let pe10 = result.profile.per_pe.get(&Coord::new(1, 0)).unwrap();
        assert_eq!(pe10.messages_received, 1);
        assert_eq!(pe10.messages_sent, 1);

        let pe20 = result.profile.per_pe.get(&Coord::new(2, 0)).unwrap();
        assert_eq!(pe20.messages_received, 1); // final delivery
        assert_eq!(pe20.messages_sent, 0);
        assert_eq!(pe20.slots_written, 1);
        assert_eq!(pe20.tasks_executed, 1);
    }

    #[test]
    fn max_events_safety_limit() {
        // Create a scenario that would run many events, but cap it
        let mut s = Simulator::new(SimConfig {
            width: 10,
            height: 10,
            max_events: 5,
            ..Default::default()
        });

        s.inject_message(InjectMessage {
            source: Coord::new(0, 0),
            dest: Coord::new(9, 9),
            payload: vec![1.0],
            payload_slot: 0,
        });

        let result = s.run();
        assert_eq!(result.profile.total_events_processed, 5);
        // Message should NOT have been fully delivered
        assert!(result.outputs.is_empty());
    }

    #[test]
    fn link_counts_tracked() {
        // Message: (0,0) -> (2,0), hops through (0,0)->(1,0)->(2,0)
        let mut s = sim(3, 1);
        s.add_task(InjectTask {
            coord: Coord::new(2, 0),
            kind: InjectTaskKind::CollectOutput { input_slot: 0 },
            trigger_slot: 0,
        });
        s.inject_message(InjectMessage {
            source: Coord::new(0, 0),
            dest: Coord::new(2, 0),
            payload: vec![1.0],
            payload_slot: 0,
        });

        let result = s.run();

        // Two links traversed: (0,0)->(1,0) and (1,0)->(2,0)
        assert_eq!(
            result
                .profile
                .link_counts
                .get(&(Coord::new(0, 0), Coord::new(1, 0))),
            Some(&1)
        );
        assert_eq!(
            result
                .profile
                .link_counts
                .get(&(Coord::new(1, 0), Coord::new(2, 0))),
            Some(&1)
        );
        assert_eq!(result.profile.link_counts.len(), 2);
    }

    #[test]
    fn operator_timings_recorded() {
        // Forward -> Collect chain: should record 2 operator timings
        let mut s = sim(3, 1);
        s.add_task(InjectTask {
            coord: Coord::new(0, 0),
            kind: InjectTaskKind::ForwardActivation {
                input_slot: 0,
                route_dest: Coord::new(2, 0),
            },
            trigger_slot: 0,
        });
        s.add_task(InjectTask {
            coord: Coord::new(2, 0),
            kind: InjectTaskKind::CollectOutput { input_slot: 0 },
            trigger_slot: 0,
        });
        s.inject_message(InjectMessage {
            source: Coord::new(0, 0),
            dest: Coord::new(0, 0),
            payload: vec![1.0],
            payload_slot: 0,
        });

        let result = s.run();

        assert_eq!(result.profile.operator_timings.len(), 2);

        let fwd = &result.profile.operator_timings[0];
        assert_eq!(fwd.task_kind, "forward_activation");
        assert_eq!(fwd.coord, Coord::new(0, 0));
        assert!(fwd.end_ts > fwd.start_ts);

        let collect = &result.profile.operator_timings[1];
        assert_eq!(collect.task_kind, "collect_output");
        assert_eq!(collect.coord, Coord::new(2, 0));
    }

    #[test]
    fn trace_events_recorded() {
        // Single hop: (0,0) -> (1,0) with collect
        let mut s = sim(2, 1);
        s.add_task(InjectTask {
            coord: Coord::new(1, 0),
            kind: InjectTaskKind::CollectOutput { input_slot: 0 },
            trigger_slot: 0,
        });
        s.inject_message(InjectMessage {
            source: Coord::new(0, 0),
            dest: Coord::new(1, 0),
            payload: vec![1.0],
            payload_slot: 0,
        });

        let result = s.run();

        // Should have trace events for: deliver@(0,0), send@(0,0), deliver@(1,0), execute@(1,0)
        use crate::profiling::TraceEventKind;
        let delivers: Vec<_> = result
            .profile
            .trace_events
            .iter()
            .filter(|e| e.kind == TraceEventKind::MessageDeliver)
            .collect();
        let sends: Vec<_> = result
            .profile
            .trace_events
            .iter()
            .filter(|e| e.kind == TraceEventKind::MessageSend)
            .collect();
        let executes: Vec<_> = result
            .profile
            .trace_events
            .iter()
            .filter(|e| e.kind == TraceEventKind::TaskExecute)
            .collect();

        assert_eq!(delivers.len(), 2); // intermediate + final
        assert_eq!(sends.len(), 1); // forwarded from (0,0) to (1,0)
        assert_eq!(executes.len(), 1); // collect task
        assert_eq!(executes[0].detail, "collect_output");
    }

    #[test]
    fn max_queue_depth_tracked() {
        // Two messages targeting the same PE injected at t=0
        let mut s = sim(2, 1);
        s.add_task(InjectTask {
            coord: Coord::new(1, 0),
            kind: InjectTaskKind::CollectOutput { input_slot: 0 },
            trigger_slot: 0,
        });
        // Both messages go to (1,0) via 1 hop from (0,0)
        s.inject_message(InjectMessage {
            source: Coord::new(0, 0),
            dest: Coord::new(1, 0),
            payload: vec![1.0],
            payload_slot: 0,
        });
        s.inject_message(InjectMessage {
            source: Coord::new(0, 0),
            dest: Coord::new(1, 0),
            payload: vec![2.0],
            payload_slot: 0,
        });

        let result = s.run();

        // (0,0) had 2 pending DeliverMessage events at injection time
        let pe00 = result.profile.per_pe.get(&Coord::new(0, 0)).unwrap();
        assert_eq!(pe00.max_queue_depth, 2);
    }

    #[test]
    fn deterministic_event_ordering() {
        // Run the same simulation twice, verify identical results
        let run = || {
            let mut s = sim(4, 4);
            s.add_task(InjectTask {
                coord: Coord::new(3, 0),
                kind: InjectTaskKind::CollectOutput { input_slot: 0 },
                trigger_slot: 0,
            });
            s.add_task(InjectTask {
                coord: Coord::new(0, 3),
                kind: InjectTaskKind::CollectOutput { input_slot: 0 },
                trigger_slot: 0,
            });
            s.inject_message(InjectMessage {
                source: Coord::new(0, 0),
                dest: Coord::new(3, 0),
                payload: vec![10.0],
                payload_slot: 0,
            });
            s.inject_message(InjectMessage {
                source: Coord::new(0, 0),
                dest: Coord::new(0, 3),
                payload: vec![20.0],
                payload_slot: 0,
            });
            s.run()
        };

        let r1 = run();
        let r2 = run();

        assert_eq!(
            r1.profile.total_events_processed,
            r2.profile.total_events_processed
        );
        assert_eq!(r1.profile.final_timestamp, r2.profile.final_timestamp);
        assert_eq!(r1.profile.total_hops, r2.profile.total_hops);
        assert_eq!(r1.outputs, r2.outputs);
    }
}
