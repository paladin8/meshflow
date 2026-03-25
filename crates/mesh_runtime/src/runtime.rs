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

    // Reserved SRAM slot IDs for internal task state.
    // These must not collide with each other or with user-facing slots.
    // - ConcatCollect/ConcatCollectForward: MAX (accumulator), MAX-1 (counter)
    // - RmsNormReduce: MAX (accumulator), MAX-1 (counter)
    // - MatMul: MAX-2 (counter)
    // No PE currently hosts tasks that would collide (e.g., ConcatCollect + MatMul).
    const ACCUM_SLOT: SlotId = u32::MAX;
    const COUNTER_SLOT: SlotId = u32::MAX - 1;
    const MATMUL_COUNTER_SLOT: SlotId = u32::MAX - 2;

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

                // Compute y = W @ x + b, batched over positions.
                // If input is longer than tile_cols, it contains multiple
                // positions concatenated. Each position is processed with
                // the same weights. Output is row-major: for each output
                // row i, emit all positions' values before row i+1.
                let rows = tile_rows as usize;
                let cols = tile_cols as usize;
                let num_positions = x.len() / cols;
                let mut y = Vec::with_capacity(rows * num_positions);
                for i in 0..rows {
                    for p in 0..num_positions {
                        let x_pos = &x[p * cols..(p + 1) * cols];
                        let mut sum = b[i];
                        for j in 0..cols {
                            sum += w[i * cols + j] * x_pos[j];
                        }
                        y.push(sum);
                    }
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
                num_positions,
            } => {
                let trigger_slot = self.mesh.pe(coord).tasks[task_index].trigger_slot;
                let completed = self.process_concat_fragment(
                    coord,
                    trigger_slot,
                    num_fragments,
                    total_rows,
                    fragment_offset,
                    num_positions,
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
                num_positions,
                scatter,
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
                    num_positions,
                );
                if let Some(mut result) = completed {
                    // Apply activation function if specified
                    if let Some(act) = &activation {
                        act.apply(&mut result);
                    }

                    // Send to next layer's tile PEs with send serialization.
                    let num_sends = route_dests.len() as u64;
                    self.mesh.pe_mut(coord).counters.messages_sent += num_sends;

                    if scatter && !route_dests.is_empty() {
                        // Scatter: send row i to destination i
                        let row_size = result.len() / route_dests.len();
                        for (i, (dest, hops)) in route_dests.iter().enumerate() {
                            let send_time = timestamp + self.config.task_base_latency + i as u64;
                            let row = result[i * row_size..(i + 1) * row_size].to_vec();
                            let message = Message {
                                id: self.next_message_id,
                                source: coord,
                                dest: *dest,
                                hops: hops.clone(),
                                current_hop: 0,
                                payload: row,
                                payload_slot: 0,
                                timestamp: send_time,
                            };
                            self.next_message_id += 1;
                            self.enqueue_deliver(send_time, coord, coord, message);
                        }
                    } else {
                        // Broadcast: send full result to all destinations
                        for (i, (dest, hops)) in route_dests.iter().enumerate() {
                            let send_time = timestamp + self.config.task_base_latency + i as u64;
                            let message = Message {
                                id: self.next_message_id,
                                source: coord,
                                dest: *dest,
                                hops: hops.clone(),
                                current_hop: 0,
                                payload: result.clone(),
                                payload_slot: 0,
                                timestamp: send_time,
                            };
                            self.next_message_id += 1;
                            self.enqueue_deliver(send_time, coord, coord, message);
                        }
                    }
                }
            }
            TaskKind::Add {
                input_slot_a,
                input_slot_b,
                output_slot,
                ref output_dests,
                ref payload_slots,
            } => {
                let pe = self.mesh.pe_mut(coord);
                pe.counters.tasks_executed += 1;
                self.profile.total_tasks_executed += 1;

                // Guard: both inputs must be present before computing.
                // The task triggers on each input arrival; only the second
                // trigger finds both slots populated.
                if !pe.has_slot(input_slot_a) || !pe.has_slot(input_slot_b) {
                    return;
                }

                let a = pe.read_slot(input_slot_a).clone();
                let b = pe.read_slot(input_slot_b).clone();

                // Element-wise addition
                debug_assert_eq!(
                    a.len(),
                    b.len(),
                    "PE {}: Add inputs have mismatched lengths ({} vs {})",
                    coord,
                    a.len(),
                    b.len()
                );
                let result: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
                self.task_write_slot(timestamp, coord, output_slot, result.clone());

                // Route to destinations
                let output_dests = output_dests.clone();
                let payload_slots = payload_slots.clone();
                let num_sends = output_dests.len() as u64;
                self.mesh.pe_mut(coord).counters.messages_sent += num_sends;

                for (i, (dest, hops)) in output_dests.iter().enumerate() {
                    let send_time = timestamp + self.config.task_base_latency + i as u64;
                    let slot = payload_slots.get(i).copied().unwrap_or(0);
                    let message = Message {
                        id: self.next_message_id,
                        source: coord,
                        dest: *dest,
                        hops: hops.clone(),
                        current_hop: 0,
                        payload: result.clone(),
                        payload_slot: slot,
                        timestamp: send_time,
                    };
                    self.next_message_id += 1;
                    self.enqueue_deliver(send_time, coord, coord, message);
                }
            }
            TaskKind::Softmax {
                input_slot,
                output_slot,
            } => {
                let pe = self.mesh.pe_mut(coord);
                let data = pe.read_slot(input_slot).clone();
                pe.counters.tasks_executed += 1;
                self.profile.total_tasks_executed += 1;

                // Numerically stable softmax: exp(x - max) / sum(exp(x - max))
                let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_vals: Vec<f32> = data.iter().map(|x| (x - max_val).exp()).collect();
                let sum: f32 = exp_vals.iter().sum();
                let result: Vec<f32> = exp_vals.iter().map(|x| x / sum).collect();

                self.task_write_slot(timestamp, coord, output_slot, result);
            }
            TaskKind::MatMul {
                ref operand_slots,
                num_dynamic_operands,
                output_slot,
                ref output_dests,
                ref payload_slots,
            } => {
                // Counter-based trigger: fires on each dynamic operand arrival,
                // only computes when all dynamic operands have arrived.
                // Note: tasks_executed counts triggers (including early returns),
                // not completions — consistent with ConcatCollect/RmsNormReduce.
                let pe = self.mesh.pe_mut(coord);
                pe.counters.tasks_executed += 1;
                self.profile.total_tasks_executed += 1;

                // Use reserved counter slot for dynamic operand counting
                let counter_slot = Self::MATMUL_COUNTER_SLOT;
                let count = if num_dynamic_operands > 1 {
                    if !pe.has_slot(counter_slot) {
                        pe.write_slot(counter_slot, vec![0.0]);
                    }
                    let c = pe.read_slot(counter_slot)[0] as u32 + 1;
                    pe.write_slot(counter_slot, vec![c as f32]);
                    c
                } else {
                    // Single dynamic operand (or all pre-loaded): compute immediately
                    num_dynamic_operands
                };

                if count < num_dynamic_operands {
                    return; // Still waiting for more operands
                }

                // Clean up counter
                if num_dynamic_operands > 1 {
                    pe.remove_slot(counter_slot);
                }

                // All operands ready. Slot 0 = left operand (vector),
                // remaining slots = right operand columns (one vector per slot).
                // Compute dot products: result[i] = dot(left, right_col[i])
                let left = pe.read_slot(operand_slots[0]).clone();
                let right_slots = &operand_slots[1..];
                let mut result = Vec::with_capacity(right_slots.len());
                for &slot in right_slots {
                    let col = pe.read_slot(slot);
                    debug_assert_eq!(
                        left.len(),
                        col.len(),
                        "PE {}: MatMul operand length mismatch ({} vs {})",
                        coord,
                        left.len(),
                        col.len()
                    );
                    let dot: f32 = left.iter().zip(col.iter()).map(|(a, b)| a * b).sum();
                    result.push(dot);
                }

                // Note: dynamic operand slots (K/V vectors) are NOT cleaned up here.
                // At current dimensions this is negligible; a future optimization
                // could remove broadcast slots after computation.

                let output_dests = output_dests.clone();
                let payload_slots = payload_slots.clone();
                self.task_write_slot(timestamp, coord, output_slot, result.clone());

                // Route to destinations
                let num_sends = output_dests.len() as u64;
                self.mesh.pe_mut(coord).counters.messages_sent += num_sends;

                for (i, (dest, hops)) in output_dests.iter().enumerate() {
                    let send_time = timestamp + self.config.task_base_latency + i as u64;
                    let slot = payload_slots.get(i).copied().unwrap_or(0);
                    let message = Message {
                        id: self.next_message_id,
                        source: coord,
                        dest: *dest,
                        hops: hops.clone(),
                        current_hop: 0,
                        payload: result.clone(),
                        payload_slot: slot,
                        timestamp: send_time,
                    };
                    self.next_message_id += 1;
                    self.enqueue_deliver(send_time, coord, coord, message);
                }
            }
            TaskKind::RmsNormPartialSum {
                input_slot,
                reduce_dest,
                ref reduce_hops,
                partial_sum_slot,
                slice_offset: _,
                slice_size: _,
            } => {
                let pe = self.mesh.pe_mut(coord);
                let data = pe.read_slot(input_slot).clone();
                pe.counters.tasks_executed += 1;
                pe.counters.messages_sent += 1;
                self.profile.total_tasks_executed += 1;

                // Compute sum(x^2) for local feature slice
                let partial_sum: f32 = data.iter().map(|x| x * x).sum();

                // Send partial sum to reduce PE
                let reduce_hops = reduce_hops.clone();
                let message = Message {
                    id: self.next_message_id,
                    source: coord,
                    dest: reduce_dest,
                    hops: reduce_hops,
                    current_hop: 0,
                    payload: vec![partial_sum],
                    payload_slot: partial_sum_slot,
                    timestamp,
                };
                self.next_message_id += 1;
                self.enqueue_deliver(timestamp, coord, coord, message);
            }
            TaskKind::RmsNormNormalize {
                input_slot,
                scale_slot,
                gamma_slot,
                ref output_dests,
                ref payload_slots,
                slice_offset: _,
                slice_size: _,
            } => {
                let pe = self.mesh.pe_mut(coord);
                let data = pe.read_slot(input_slot).clone();
                let scale = pe.read_slot(scale_slot)[0];
                let gamma = pe.read_slot(gamma_slot).clone();
                pe.counters.tasks_executed += 1;
                self.profile.total_tasks_executed += 1;

                // Apply x * scale * gamma
                debug_assert_eq!(
                    data.len(),
                    gamma.len(),
                    "PE {}: RmsNormNormalize data/gamma length mismatch ({} vs {})",
                    coord,
                    data.len(),
                    gamma.len()
                );
                let result: Vec<f32> = data
                    .iter()
                    .zip(gamma.iter())
                    .map(|(x, g)| x * scale * g)
                    .collect();

                // Route to destinations
                let output_dests = output_dests.clone();
                let payload_slots = payload_slots.clone();
                let num_sends = output_dests.len() as u64;
                self.mesh.pe_mut(coord).counters.messages_sent += num_sends;

                for (i, (dest, hops)) in output_dests.iter().enumerate() {
                    let send_time = timestamp + self.config.task_base_latency + i as u64;
                    let slot = payload_slots.get(i).copied().unwrap_or(0);
                    let message = Message {
                        id: self.next_message_id,
                        source: coord,
                        dest: *dest,
                        hops: hops.clone(),
                        current_hop: 0,
                        payload: result.clone(),
                        payload_slot: slot,
                        timestamp: send_time,
                    };
                    self.next_message_id += 1;
                    self.enqueue_deliver(send_time, coord, coord, message);
                }
            }
            TaskKind::RmsNormReduce {
                num_tiles,
                feature_count,
                eps,
                ref tile_dests,
                scale_slot,
            } => {
                // Counter-based accumulation (like ConcatCollect):
                // fires on each partial sum arrival, computes when all arrive.
                // Note: tasks_executed counts triggers (including early returns),
                // not completions — consistent with ConcatCollect/MatMul.
                let trigger_slot = self.mesh.pe(coord).tasks[task_index].trigger_slot;
                let pe = self.mesh.pe_mut(coord);
                pe.counters.tasks_executed += 1;
                self.profile.total_tasks_executed += 1;

                let partial_sum = pe.read_slot(trigger_slot)[0];

                // Use reserved SRAM slots for accumulator and counter
                let accum_slot = Self::ACCUM_SLOT;
                let counter_slot = Self::COUNTER_SLOT;

                if !pe.has_slot(accum_slot) {
                    pe.write_slot(accum_slot, vec![0.0]);
                    pe.write_slot(counter_slot, vec![0.0]);
                }

                // Accumulate partial sum
                let running_sum = pe.read_slot(accum_slot)[0] + partial_sum;
                pe.write_slot(accum_slot, vec![running_sum]);
                pe.remove_slot(trigger_slot);

                // Increment counter
                let count = pe.read_slot(counter_slot)[0] as u32 + 1;
                pe.write_slot(counter_slot, vec![count as f32]);

                if count < num_tiles {
                    return; // Still waiting for more partial sums
                }

                // All partial sums received — compute scale factor
                let mean_sq = running_sum / feature_count as f32;
                let scale = 1.0 / (mean_sq + eps).sqrt();

                // Clean up accumulator
                pe.remove_slot(accum_slot);
                pe.remove_slot(counter_slot);

                // Write scale to local SRAM (for debugging/inspection)
                pe.write_slot(scale_slot, vec![scale]);

                // Broadcast scale factor to all tile PEs
                let tile_dests = tile_dests.clone();
                let num_sends = tile_dests.len() as u64;
                self.mesh.pe_mut(coord).counters.messages_sent += num_sends;

                for (i, (dest, hops)) in tile_dests.iter().enumerate() {
                    let send_time = timestamp + self.config.task_base_latency + i as u64;
                    let message = Message {
                        id: self.next_message_id,
                        source: coord,
                        dest: *dest,
                        hops: hops.clone(),
                        current_hop: 0,
                        payload: vec![scale],
                        payload_slot: scale_slot,
                        timestamp: send_time,
                    };
                    self.next_message_id += 1;
                    self.enqueue_deliver(send_time, coord, coord, message);
                }
            }
        }
    }

    /// Write data to a PE's SRAM slot during task execution, then schedule
    /// any co-located tasks triggered by that slot.
    ///
    /// Use this in `process_execute` whenever a task writes a result to local
    /// SRAM that another task on the same PE should react to (e.g., Softmax
    /// writing scores that trigger MatMul).
    ///
    /// Do NOT use this for:
    /// - `process_deliver` (already handles its own trigger check)
    /// - `write_sram` (setup-time weight loading, no triggers)
    /// - `process_concat_fragment` (internal accumulator writes)
    fn task_write_slot(&mut self, timestamp: u64, coord: Coord, slot: SlotId, data: Vec<f32>) {
        self.mesh.pe_mut(coord).write_slot(slot, data);
        let triggered = self.mesh.pe(coord).triggered_tasks(slot);
        for task_index in triggered {
            self.queue.push(
                timestamp + self.config.task_base_latency,
                coord,
                EventKind::ExecuteTask { task_index },
            );
        }
    }

    /// Shared accumulator logic for ConcatCollect and ConcatCollectForward.
    /// Returns Some(completed_buffer) when all fragments have arrived, None otherwise.
    ///
    /// When `num_positions > 0`, fragments are in row-major layout (rows outer,
    /// positions inner). The accumulator scales to `total_rows * num_positions`,
    /// and the completed buffer is transposed to position-major on return.
    fn process_concat_fragment(
        &mut self,
        coord: Coord,
        trigger_slot: SlotId,
        num_fragments: u32,
        total_rows: u32,
        fragment_offset: u32,
        num_positions: u32,
    ) -> Option<Vec<f32>> {
        let pe = self.mesh.pe_mut(coord);
        let fragment = pe.read_slot(trigger_slot).clone();
        pe.counters.tasks_executed += 1;
        self.profile.total_tasks_executed += 1;

        let num_pos = if num_positions > 0 {
            num_positions as usize
        } else {
            1
        };
        let total_size = total_rows as usize * num_pos;
        let offset = fragment_offset as usize * num_pos;
        let frag_len = fragment.len();

        // Use reserved SRAM slots for accumulator buffer and counter.
        let accum_slot = Self::ACCUM_SLOT;
        let counter_slot = Self::COUNTER_SLOT;

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
            let mut result = pe.remove_slot(accum_slot).unwrap();
            pe.remove_slot(counter_slot);

            // Transpose from row-major (total_rows, num_pos) to
            // position-major (num_pos, total_rows) when multi-position.
            if num_pos > 1 {
                let tr = total_rows as usize;
                let mut transposed = vec![0.0; total_size];
                for r in 0..tr {
                    for p in 0..num_pos {
                        transposed[p * tr + r] = result[r * num_pos + p];
                    }
                }
                result = transposed;
            }

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

    #[test]
    fn add_element_wise() {
        // Two inputs arrive at (0,0), Add produces sum, collected at (1,0)
        let mut s = sim(2, 1);

        // Add task: slot 0 + slot 1 -> slot 2, route to (1,0)
        let hops = crate::route::generate_route_xy(Coord::new(0, 0), Coord::new(1, 0));
        s.add_task_direct(
            Coord::new(0, 0),
            TaskConfig {
                kind: TaskKind::Add {
                    input_slot_a: 0,
                    input_slot_b: 1,
                    output_slot: 2,
                    output_dests: vec![(Coord::new(1, 0), hops)],
                    payload_slots: vec![0],
                },
                trigger_slot: 1, // trigger when second input arrives
            },
        );
        s.add_task_direct(
            Coord::new(1, 0),
            TaskConfig {
                kind: TaskKind::CollectOutput { input_slot: 0 },
                trigger_slot: 0,
            },
        );

        // Pre-load first input (skip connection)
        s.write_sram(Coord::new(0, 0), 0, vec![1.0, 2.0, 3.0]);
        // Inject second input (main path)
        s.inject_message(InjectMessage {
            source: Coord::new(0, 0),
            dest: Coord::new(0, 0),
            payload: vec![10.0, 20.0, 30.0],
            payload_slot: 1,
        });

        let result = s.run();
        let output = result.outputs.get(&Coord::new(1, 0)).unwrap();
        assert_eq!(output, &vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn add_fan_out_to_multiple_dests() {
        // Add at (0,0) sends result to (1,0) and (2,0)
        let mut s = sim(3, 1);

        let hops1 = crate::route::generate_route_xy(Coord::new(0, 0), Coord::new(1, 0));
        let hops2 = crate::route::generate_route_xy(Coord::new(0, 0), Coord::new(2, 0));
        s.add_task_direct(
            Coord::new(0, 0),
            TaskConfig {
                kind: TaskKind::Add {
                    input_slot_a: 0,
                    input_slot_b: 1,
                    output_slot: 2,
                    output_dests: vec![(Coord::new(1, 0), hops1), (Coord::new(2, 0), hops2)],
                    payload_slots: vec![0, 0],
                },
                trigger_slot: 1,
            },
        );
        s.add_task_direct(
            Coord::new(1, 0),
            TaskConfig {
                kind: TaskKind::CollectOutput { input_slot: 0 },
                trigger_slot: 0,
            },
        );
        s.add_task_direct(
            Coord::new(2, 0),
            TaskConfig {
                kind: TaskKind::CollectOutput { input_slot: 0 },
                trigger_slot: 0,
            },
        );

        s.write_sram(Coord::new(0, 0), 0, vec![1.0, 2.0]);
        s.inject_message(InjectMessage {
            source: Coord::new(0, 0),
            dest: Coord::new(0, 0),
            payload: vec![3.0, 4.0],
            payload_slot: 1,
        });

        let result = s.run();
        assert_eq!(result.outputs.get(&Coord::new(1, 0)), Some(&vec![4.0, 6.0]));
        assert_eq!(result.outputs.get(&Coord::new(2, 0)), Some(&vec![4.0, 6.0]));
    }

    #[test]
    fn softmax_basic() {
        // Softmax at (0,0): reads slot 0, writes slot 1, triggers collect at slot 1
        let mut s = sim(1, 1);

        s.add_task_direct(
            Coord::new(0, 0),
            TaskConfig {
                kind: TaskKind::Softmax {
                    input_slot: 0,
                    output_slot: 1,
                },
                trigger_slot: 0,
            },
        );
        s.add_task_direct(
            Coord::new(0, 0),
            TaskConfig {
                kind: TaskKind::CollectOutput { input_slot: 1 },
                trigger_slot: 1,
            },
        );

        s.inject_message(InjectMessage {
            source: Coord::new(0, 0),
            dest: Coord::new(0, 0),
            payload: vec![1.0, 2.0, 3.0, 4.0],
            payload_slot: 0,
        });

        let result = s.run();
        let output = result.outputs.get(&Coord::new(0, 0)).unwrap();

        // Verify softmax properties
        assert_eq!(output.len(), 4);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax should sum to 1.0");

        // Values should be monotonically increasing
        for i in 1..output.len() {
            assert!(output[i] > output[i - 1]);
        }
    }

    #[test]
    fn softmax_numerical_stability() {
        // Large values that would overflow without max subtraction
        let mut s = sim(1, 1);

        s.add_task_direct(
            Coord::new(0, 0),
            TaskConfig {
                kind: TaskKind::Softmax {
                    input_slot: 0,
                    output_slot: 1,
                },
                trigger_slot: 0,
            },
        );
        s.add_task_direct(
            Coord::new(0, 0),
            TaskConfig {
                kind: TaskKind::CollectOutput { input_slot: 1 },
                trigger_slot: 1,
            },
        );

        s.inject_message(InjectMessage {
            source: Coord::new(0, 0),
            dest: Coord::new(0, 0),
            payload: vec![1000.0, 1001.0, 1002.0],
            payload_slot: 0,
        });

        let result = s.run();
        let output = result.outputs.get(&Coord::new(0, 0)).unwrap();

        let sum: f32 = output.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "softmax should sum to 1.0 even with large inputs"
        );
        // No NaN or Inf
        for v in output {
            assert!(v.is_finite(), "softmax output should be finite");
        }
    }

    #[test]
    fn softmax_uniform_input() {
        // Equal inputs → uniform distribution
        let mut s = sim(1, 1);

        s.add_task_direct(
            Coord::new(0, 0),
            TaskConfig {
                kind: TaskKind::Softmax {
                    input_slot: 0,
                    output_slot: 1,
                },
                trigger_slot: 0,
            },
        );
        s.add_task_direct(
            Coord::new(0, 0),
            TaskConfig {
                kind: TaskKind::CollectOutput { input_slot: 1 },
                trigger_slot: 1,
            },
        );

        s.inject_message(InjectMessage {
            source: Coord::new(0, 0),
            dest: Coord::new(0, 0),
            payload: vec![5.0, 5.0, 5.0, 5.0],
            payload_slot: 0,
        });

        let result = s.run();
        let output = result.outputs.get(&Coord::new(0, 0)).unwrap();

        for v in output {
            assert!(
                (v - 0.25).abs() < 1e-6,
                "equal inputs should give uniform distribution"
            );
        }
    }

    #[test]
    fn matmul_dot_products() {
        // MatMul at (0,0): Q row in slot 0 (pre-loaded), K vectors in slots 1-3 (broadcast).
        // Computes dot(Q, K_i) for each K_i, collects result at (1,0).
        let mut s = sim(2, 1);

        // operand_slots: [0, 1, 2, 3] — slot 0 = Q, slots 1-3 = K vectors
        // num_dynamic_operands = 3 (K vectors arrive via messages)
        // Need one TaskConfig per dynamic operand slot (1, 2, 3)
        let hops = crate::route::generate_route_xy(Coord::new(0, 0), Coord::new(1, 0));
        for trigger in [1, 2, 3] {
            s.add_task_direct(
                Coord::new(0, 0),
                TaskConfig {
                    kind: TaskKind::MatMul {
                        operand_slots: vec![0, 1, 2, 3],
                        num_dynamic_operands: 3,
                        output_slot: 10,
                        output_dests: vec![(Coord::new(1, 0), hops.clone())],
                        payload_slots: vec![0],
                    },
                    trigger_slot: trigger,
                },
            );
        }
        s.add_task_direct(
            Coord::new(1, 0),
            TaskConfig {
                kind: TaskKind::CollectOutput { input_slot: 0 },
                trigger_slot: 0,
            },
        );

        // Pre-load Q row
        s.write_sram(Coord::new(0, 0), 0, vec![1.0, 2.0]);

        // Inject K vectors (3 separate messages)
        // K0 = [1, 0] → dot = 1*1 + 2*0 = 1
        // K1 = [0, 1] → dot = 1*0 + 2*1 = 2
        // K2 = [1, 1] → dot = 1*1 + 2*1 = 3
        for (slot, payload) in [
            (1, vec![1.0, 0.0]),
            (2, vec![0.0, 1.0]),
            (3, vec![1.0, 1.0]),
        ] {
            s.inject_message(InjectMessage {
                source: Coord::new(0, 0),
                dest: Coord::new(0, 0),
                payload,
                payload_slot: slot,
            });
        }

        let result = s.run();
        let output = result.outputs.get(&Coord::new(1, 0)).unwrap();
        assert_eq!(output, &vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn matmul_single_dynamic_operand() {
        // MatMul with 1 dynamic operand — no counter needed
        let mut s = sim(2, 1);

        let hops = crate::route::generate_route_xy(Coord::new(0, 0), Coord::new(1, 0));
        s.add_task_direct(
            Coord::new(0, 0),
            TaskConfig {
                kind: TaskKind::MatMul {
                    operand_slots: vec![0, 1],
                    num_dynamic_operands: 1,
                    output_slot: 10,
                    output_dests: vec![(Coord::new(1, 0), hops)],
                    payload_slots: vec![0],
                },
                trigger_slot: 1,
            },
        );
        s.add_task_direct(
            Coord::new(1, 0),
            TaskConfig {
                kind: TaskKind::CollectOutput { input_slot: 0 },
                trigger_slot: 0,
            },
        );

        // Pre-load left operand and single right operand
        s.write_sram(Coord::new(0, 0), 0, vec![3.0, 4.0]);
        s.inject_message(InjectMessage {
            source: Coord::new(0, 0),
            dest: Coord::new(0, 0),
            payload: vec![2.0, 1.0],
            payload_slot: 1,
        });

        let result = s.run();
        let output = result.outputs.get(&Coord::new(1, 0)).unwrap();
        // dot([3,4], [2,1]) = 6 + 4 = 10
        assert_eq!(output, &vec![10.0]);
    }

    #[test]
    fn matmul_triggers_softmax_chain() {
        // MatMul(QK^T) -> Softmax -> Collect, all on same PE
        let mut s = sim(1, 1);

        // MatMul: slot 0 = Q, slot 1 = K0, slot 2 = K1
        // Writes score row to slot 10
        // One TaskConfig per dynamic operand slot
        for trigger in [1, 2] {
            s.add_task_direct(
                Coord::new(0, 0),
                TaskConfig {
                    kind: TaskKind::MatMul {
                        operand_slots: vec![0, 1, 2],
                        num_dynamic_operands: 2,
                        output_slot: 10,
                        output_dests: vec![],
                        payload_slots: vec![],
                    },
                    trigger_slot: trigger,
                },
            );
        }
        // Softmax: reads slot 10, writes slot 11
        s.add_task_direct(
            Coord::new(0, 0),
            TaskConfig {
                kind: TaskKind::Softmax {
                    input_slot: 10,
                    output_slot: 11,
                },
                trigger_slot: 10,
            },
        );
        // Collect from slot 11
        s.add_task_direct(
            Coord::new(0, 0),
            TaskConfig {
                kind: TaskKind::CollectOutput { input_slot: 11 },
                trigger_slot: 11,
            },
        );

        s.write_sram(Coord::new(0, 0), 0, vec![1.0, 0.0]); // Q
        s.inject_message(InjectMessage {
            source: Coord::new(0, 0),
            dest: Coord::new(0, 0),
            payload: vec![1.0, 0.0], // K0: dot = 1
            payload_slot: 1,
        });
        s.inject_message(InjectMessage {
            source: Coord::new(0, 0),
            dest: Coord::new(0, 0),
            payload: vec![0.0, 1.0], // K1: dot = 0
            payload_slot: 2,
        });

        let result = s.run();
        let output = result.outputs.get(&Coord::new(0, 0)).unwrap();

        // Scores = [1, 0], softmax([1, 0]) ≈ [0.731, 0.269]
        assert_eq!(output.len(), 2);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(output[0] > output[1]); // score 1 > score 0
    }

    #[test]
    fn rmsnorm_full_pipeline() {
        // RMSNorm with 2 tile PEs and 1 reduce PE.
        // Input: tile0 has [3.0, 4.0], tile1 has [0.0, 0.0]
        // feature_count = 4, gamma = [1.0, 1.0, 1.0, 1.0]
        //
        // sum(x^2) = 9 + 16 + 0 + 0 = 25
        // mean(x^2) = 25 / 4 = 6.25
        // scale = 1 / sqrt(6.25 + 1e-6) = 1 / 2.5 = 0.4
        // normalized: [3*0.4, 4*0.4, 0*0.4, 0*0.4] = [1.2, 1.6, 0.0, 0.0]
        let mut s = sim(3, 2);

        let tile0 = Coord::new(0, 0);
        let tile1 = Coord::new(0, 1);
        let reduce = Coord::new(1, 0);
        let collect0 = Coord::new(2, 0);
        let collect1 = Coord::new(2, 1);

        // Tile 0: RmsNormPartialSum (phase 1) + RmsNormNormalize (phase 2)
        let hops_to_reduce = crate::route::generate_route_xy(tile0, reduce);
        let hops_to_collect0 = crate::route::generate_route_xy(tile0, collect0);
        s.add_task_direct(
            tile0,
            TaskConfig {
                kind: TaskKind::RmsNormPartialSum {
                    input_slot: 0,
                    reduce_dest: reduce,
                    reduce_hops: hops_to_reduce,
                    partial_sum_slot: 0, // slot on reduce PE
                    slice_offset: 0,
                    slice_size: 0,
                },
                trigger_slot: 0,
            },
        );
        s.add_task_direct(
            tile0,
            TaskConfig {
                kind: TaskKind::RmsNormNormalize {
                    input_slot: 0,
                    scale_slot: 1,
                    gamma_slot: 2,
                    output_dests: vec![(collect0, hops_to_collect0)],
                    payload_slots: vec![0],
                    slice_offset: 0,
                    slice_size: 0,
                },
                trigger_slot: 1, // triggered by scale factor arrival
            },
        );
        s.write_sram(tile0, 2, vec![1.0, 1.0]); // gamma

        // Tile 1: same structure
        let hops_to_reduce1 = crate::route::generate_route_xy(tile1, reduce);
        let hops_to_collect1 = crate::route::generate_route_xy(tile1, collect1);
        s.add_task_direct(
            tile1,
            TaskConfig {
                kind: TaskKind::RmsNormPartialSum {
                    input_slot: 0,
                    reduce_dest: reduce,
                    reduce_hops: hops_to_reduce1,
                    partial_sum_slot: 1, // different slot on reduce PE
                    slice_offset: 0,
                    slice_size: 0,
                },
                trigger_slot: 0,
            },
        );
        s.add_task_direct(
            tile1,
            TaskConfig {
                kind: TaskKind::RmsNormNormalize {
                    input_slot: 0,
                    scale_slot: 1,
                    gamma_slot: 2,
                    output_dests: vec![(collect1, hops_to_collect1)],
                    payload_slots: vec![0],
                    slice_offset: 0,
                    slice_size: 0,
                },
                trigger_slot: 1,
            },
        );
        s.write_sram(tile1, 2, vec![1.0, 1.0]); // gamma

        // Reduce PE: accumulates partial sums, broadcasts scale
        // One TaskConfig per partial sum slot (0 and 1)
        let hops_to_tile0 = crate::route::generate_route_xy(reduce, tile0);
        let hops_to_tile1 = crate::route::generate_route_xy(reduce, tile1);
        for trigger in [0, 1] {
            s.add_task_direct(
                reduce,
                TaskConfig {
                    kind: TaskKind::RmsNormReduce {
                        num_tiles: 2,
                        feature_count: 4,
                        eps: 1e-6,
                        tile_dests: vec![
                            (tile0, hops_to_tile0.clone()),
                            (tile1, hops_to_tile1.clone()),
                        ],
                        scale_slot: 1, // writes scale to slot 1 on tile PEs
                    },
                    trigger_slot: trigger,
                },
            );
        }

        // Collect PEs
        s.add_task_direct(
            collect0,
            TaskConfig {
                kind: TaskKind::CollectOutput { input_slot: 0 },
                trigger_slot: 0,
            },
        );
        s.add_task_direct(
            collect1,
            TaskConfig {
                kind: TaskKind::CollectOutput { input_slot: 0 },
                trigger_slot: 0,
            },
        );

        // Inject activations
        s.inject_message(InjectMessage {
            source: tile0,
            dest: tile0,
            payload: vec![3.0, 4.0],
            payload_slot: 0,
        });
        s.inject_message(InjectMessage {
            source: tile1,
            dest: tile1,
            payload: vec![0.0, 0.0],
            payload_slot: 0,
        });

        let result = s.run();

        let out0 = result.outputs.get(&collect0).unwrap();
        let out1 = result.outputs.get(&collect1).unwrap();

        // scale = 1/sqrt(25/4 + 1e-6) ≈ 0.4
        let expected_scale = 1.0 / (25.0_f32 / 4.0 + 1e-6).sqrt();
        assert!((out0[0] - 3.0 * expected_scale).abs() < 1e-5);
        assert!((out0[1] - 4.0 * expected_scale).abs() < 1e-5);
        assert!((out1[0] - 0.0).abs() < 1e-5);
        assert!((out1[1] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn rmsnorm_with_gamma_scaling() {
        // Same as above but with non-trivial gamma
        let mut s = sim(2, 1);

        let tile = Coord::new(0, 0);
        let reduce = Coord::new(1, 0);

        // Single tile PE with the full feature vector
        let hops_to_reduce = crate::route::generate_route_xy(tile, reduce);
        s.add_task_direct(
            tile,
            TaskConfig {
                kind: TaskKind::RmsNormPartialSum {
                    input_slot: 0,
                    reduce_dest: reduce,
                    reduce_hops: hops_to_reduce,
                    partial_sum_slot: 0,
                    slice_offset: 0,
                    slice_size: 0,
                },
                trigger_slot: 0,
            },
        );
        s.add_task_direct(
            tile,
            TaskConfig {
                kind: TaskKind::RmsNormNormalize {
                    input_slot: 0,
                    scale_slot: 1,
                    gamma_slot: 2,
                    output_dests: vec![],
                    payload_slots: vec![],
                    slice_offset: 0,
                    slice_size: 0,
                },
                trigger_slot: 1,
            },
        );
        // gamma = [2.0, 0.5]
        s.write_sram(tile, 2, vec![2.0, 0.5]);

        let hops_to_tile = crate::route::generate_route_xy(reduce, tile);
        s.add_task_direct(
            reduce,
            TaskConfig {
                kind: TaskKind::RmsNormReduce {
                    num_tiles: 1,
                    feature_count: 2,
                    eps: 1e-6,
                    tile_dests: vec![(tile, hops_to_tile)],
                    scale_slot: 1,
                },
                trigger_slot: 0,
            },
        );

        // Input: [3.0, 4.0]
        // sum(x^2) = 9 + 16 = 25, mean = 25/2 = 12.5
        // scale = 1/sqrt(12.5 + 1e-6) ≈ 0.2828
        // result = [3 * scale * 2.0, 4 * scale * 0.5]
        s.inject_message(InjectMessage {
            source: tile,
            dest: tile,
            payload: vec![3.0, 4.0],
            payload_slot: 0,
        });

        let result = s.run();

        // Verify all 3 tasks ran: partial_sum + reduce + normalize
        assert!(result.profile.total_tasks_executed >= 3);

        // Verify operator timings include our task kinds
        let kinds: Vec<&str> = result
            .profile
            .operator_timings
            .iter()
            .map(|t| t.task_kind.as_str())
            .collect();
        assert!(kinds.contains(&"rms_norm_partial_sum"));
        assert!(kinds.contains(&"rms_norm_reduce"));
        assert!(kinds.contains(&"rms_norm_normalize"));
    }

    #[test]
    fn rmsnorm_reduce_counter_based() {
        // Verify that RmsNormReduce only computes after all partial sums arrive
        // 3 tile PEs → reduce PE → collect
        let mut s = sim(4, 1);

        let reduce = Coord::new(3, 0);
        let tiles = [Coord::new(0, 0), Coord::new(1, 0), Coord::new(2, 0)];

        // Each tile sends a partial sum to the reduce PE
        for (i, &tile) in tiles.iter().enumerate() {
            let hops = crate::route::generate_route_xy(tile, reduce);
            s.add_task_direct(
                tile,
                TaskConfig {
                    kind: TaskKind::RmsNormPartialSum {
                        input_slot: 0,
                        reduce_dest: reduce,
                        reduce_hops: hops,
                        partial_sum_slot: i as u32, // each tile writes to a different slot
                        slice_offset: 0,
                        slice_size: 0,
                    },
                    trigger_slot: 0,
                },
            );
        }

        // Reduce PE — trigger on slot 0 (all partial sums trigger here via different slots,
        // but the reduce task fires on each arrival and uses counter)
        s.add_task_direct(
            reduce,
            TaskConfig {
                kind: TaskKind::RmsNormReduce {
                    num_tiles: 3,
                    feature_count: 6, // 3 tiles * 2 features each
                    eps: 0.0,
                    tile_dests: vec![], // no broadcast needed for this test
                    scale_slot: 10,
                },
                trigger_slot: 0,
            },
        );
        // Also trigger on slots 1 and 2
        s.add_task_direct(
            reduce,
            TaskConfig {
                kind: TaskKind::RmsNormReduce {
                    num_tiles: 3,
                    feature_count: 6,
                    eps: 0.0,
                    tile_dests: vec![],
                    scale_slot: 10,
                },
                trigger_slot: 1,
            },
        );
        s.add_task_direct(
            reduce,
            TaskConfig {
                kind: TaskKind::RmsNormReduce {
                    num_tiles: 3,
                    feature_count: 6,
                    eps: 0.0,
                    tile_dests: vec![],
                    scale_slot: 10,
                },
                trigger_slot: 2,
            },
        );

        // tile0: [1, 0] → sum_sq = 1
        // tile1: [0, 2] → sum_sq = 4
        // tile2: [3, 0] → sum_sq = 9
        // total = 14, mean = 14/6, scale = 1/sqrt(14/6)
        s.inject_message(InjectMessage {
            source: tiles[0],
            dest: tiles[0],
            payload: vec![1.0, 0.0],
            payload_slot: 0,
        });
        s.inject_message(InjectMessage {
            source: tiles[1],
            dest: tiles[1],
            payload: vec![0.0, 2.0],
            payload_slot: 0,
        });
        s.inject_message(InjectMessage {
            source: tiles[2],
            dest: tiles[2],
            payload: vec![3.0, 0.0],
            payload_slot: 0,
        });

        let result = s.run();

        // Verify all tasks executed: 3 partial_sum + 3 reduce triggers (but only last computes)
        // The reduce task fires 3 times but the first two return early.
        // total_tasks_executed should include all 3 reduce firings + 3 partial_sum
        assert!(result.profile.total_tasks_executed >= 6);
    }

    #[test]
    fn linear_batched_two_positions() {
        // W = [[1,0],[0,1]] (identity), b = [0,0]
        // Input: 2 positions of 2 features = [1, 2, 3, 4]
        // Position 0: [1, 2] → identity → [1, 2]
        // Position 1: [3, 4] → identity → [3, 4]
        // Tile output (row-major): [r0p0, r0p1, r1p0, r1p1] = [1, 3, 2, 4]
        // Collect should gather and eventually produce position-major output
        // For now, just verify the tile output is correct
        let mut s = sim(2, 2);

        // Single tile PE at (0,0), collect at (0,1)
        let hops = crate::route::generate_route_xy(Coord::new(0, 0), Coord::new(0, 1));
        s.add_task_direct(
            Coord::new(0, 0),
            TaskConfig {
                kind: TaskKind::Linear {
                    input_slot: 0,
                    weight_slot: 1,
                    bias_slot: 2,
                    tile_rows: 2,
                    tile_cols: 2,
                    route_dest: Coord::new(0, 1),
                    hops,
                    fragment_slot: 0,
                    fragment_offset: 0,
                },
                trigger_slot: 0,
            },
        );
        s.add_task_direct(
            Coord::new(0, 1),
            TaskConfig {
                kind: TaskKind::CollectOutput { input_slot: 0 },
                trigger_slot: 0,
            },
        );

        // Identity weights, zero bias
        s.write_sram(Coord::new(0, 0), 1, vec![1.0, 0.0, 0.0, 1.0]);
        s.write_sram(Coord::new(0, 0), 2, vec![0.0, 0.0]);

        // 2 positions: [1, 2] and [3, 4]
        s.inject_message(InjectMessage {
            source: Coord::new(0, 0),
            dest: Coord::new(0, 0),
            payload: vec![1.0, 2.0, 3.0, 4.0],
            payload_slot: 0,
        });

        let result = s.run();
        let output = result.outputs.get(&Coord::new(0, 1)).unwrap();

        // Row-major output: [r0p0, r0p1, r1p0, r1p1] = [1, 3, 2, 4]
        assert_eq!(output, &vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn linear_batched_with_weights() {
        // W = [[1, 2], [3, 4]], b = [0.5, 0.5]
        // Position 0: x=[1, 0] → y=[1*1+2*0+0.5, 3*1+4*0+0.5] = [1.5, 3.5]
        // Position 1: x=[0, 1] → y=[1*0+2*1+0.5, 3*0+4*1+0.5] = [2.5, 4.5]
        // Row-major output: [r0p0, r0p1, r1p0, r1p1] = [1.5, 2.5, 3.5, 4.5]
        let mut s = sim(2, 2);

        let hops = crate::route::generate_route_xy(Coord::new(0, 0), Coord::new(0, 1));
        s.add_task_direct(
            Coord::new(0, 0),
            TaskConfig {
                kind: TaskKind::Linear {
                    input_slot: 0,
                    weight_slot: 1,
                    bias_slot: 2,
                    tile_rows: 2,
                    tile_cols: 2,
                    route_dest: Coord::new(0, 1),
                    hops,
                    fragment_slot: 0,
                    fragment_offset: 0,
                },
                trigger_slot: 0,
            },
        );
        s.add_task_direct(
            Coord::new(0, 1),
            TaskConfig {
                kind: TaskKind::CollectOutput { input_slot: 0 },
                trigger_slot: 0,
            },
        );

        s.write_sram(Coord::new(0, 0), 1, vec![1.0, 2.0, 3.0, 4.0]);
        s.write_sram(Coord::new(0, 0), 2, vec![0.5, 0.5]);

        s.inject_message(InjectMessage {
            source: Coord::new(0, 0),
            dest: Coord::new(0, 0),
            payload: vec![1.0, 0.0, 0.0, 1.0],
            payload_slot: 0,
        });

        let result = s.run();
        let output = result.outputs.get(&Coord::new(0, 1)).unwrap();
        assert_eq!(output.len(), 4);
        assert!((output[0] - 1.5).abs() < 1e-6); // r0p0
        assert!((output[1] - 2.5).abs() < 1e-6); // r0p1
        assert!((output[2] - 3.5).abs() < 1e-6); // r1p0
        assert!((output[3] - 4.5).abs() < 1e-6); // r1p1
    }

    #[test]
    fn concat_collect_batched_transpose() {
        // 2 tile PEs → collect PE, with num_positions=2
        // Tile 0 (rows 0-1, 2 positions): row-major [r0p0, r0p1, r1p0, r1p1]
        // Tile 1 (rows 2-3, 2 positions): row-major [r2p0, r2p1, r3p0, r3p1]
        // After collect + transpose: [p0r0, p0r1, p0r2, p0r3, p1r0, p1r1, p1r2, p1r3]
        let mut s = sim(1, 3);

        let collect_coord = Coord::new(0, 2);

        // Tile 0 → collect (fragment_offset=0)
        for i in 0..2 {
            s.add_task_direct(
                collect_coord,
                TaskConfig {
                    kind: TaskKind::ConcatCollect {
                        num_fragments: 2,
                        total_rows: 4,
                        fragment_offset: i * 2,
                        num_positions: 2,
                    },
                    trigger_slot: i,
                },
            );
        }
        s.add_task_direct(
            collect_coord,
            TaskConfig {
                kind: TaskKind::CollectOutput {
                    input_slot: u32::MAX,
                },
                trigger_slot: u32::MAX,
            },
        );

        // Fragment 0: rows 0-1, 2 positions → [1, 10, 2, 20]
        // (r0p0=1, r0p1=10, r1p0=2, r1p1=20)
        s.inject_message(InjectMessage {
            source: Coord::new(0, 0),
            dest: collect_coord,
            payload: vec![1.0, 10.0, 2.0, 20.0],
            payload_slot: 0,
        });
        // Fragment 1: rows 2-3, 2 positions → [3, 30, 4, 40]
        s.inject_message(InjectMessage {
            source: Coord::new(0, 1),
            dest: collect_coord,
            payload: vec![3.0, 30.0, 4.0, 40.0],
            payload_slot: 1,
        });

        let result = s.run();

        // The ConcatCollect stores the result in ACCUM_SLOT, then CollectOutput
        // reads it. But we need CollectOutput to trigger on the accum slot.
        // Actually, ConcatCollect stores output in self.outputs directly.
        let output = result.outputs.get(&collect_coord).unwrap();

        // Position-major: [p0r0, p0r1, p0r2, p0r3, p1r0, p1r1, p1r2, p1r3]
        assert_eq!(output, &vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn concat_collect_forward_scatter() {
        // Collect 2 fragments (2 positions), scatter rows to 2 destinations
        // Fragment 0: [r0p0, r0p1] = [1, 10]
        // Fragment 1: [r1p0, r1p1] = [2, 20]
        // After transpose: [p0r0, p0r1, p1r0, p1r1] = [1, 2, 10, 20]
        // Scatter: dest 0 gets [1, 2] (position 0), dest 1 gets [10, 20] (position 1)
        let mut s = sim(3, 2);

        let collect_coord = Coord::new(0, 0);
        let dest0 = Coord::new(1, 0);
        let dest1 = Coord::new(2, 0);

        let hops0 = crate::route::generate_route_xy(collect_coord, dest0);
        let hops1 = crate::route::generate_route_xy(collect_coord, dest1);

        for i in 0..2 {
            s.add_task_direct(
                collect_coord,
                TaskConfig {
                    kind: TaskKind::ConcatCollectForward {
                        num_fragments: 2,
                        total_rows: 2,
                        fragment_offset: i,
                        activation: None,
                        route_dests: vec![(dest0, hops0.clone()), (dest1, hops1.clone())],
                        num_positions: 2,
                        scatter: true,
                    },
                    trigger_slot: i,
                },
            );
        }

        // Collect outputs at destinations
        s.add_task_direct(
            dest0,
            TaskConfig {
                kind: TaskKind::CollectOutput { input_slot: 0 },
                trigger_slot: 0,
            },
        );
        s.add_task_direct(
            dest1,
            TaskConfig {
                kind: TaskKind::CollectOutput { input_slot: 0 },
                trigger_slot: 0,
            },
        );

        // Fragment 0: row 0, 2 positions → [1, 10]
        s.inject_message(InjectMessage {
            source: collect_coord,
            dest: collect_coord,
            payload: vec![1.0, 10.0],
            payload_slot: 0,
        });
        // Fragment 1: row 1, 2 positions → [2, 20]
        s.inject_message(InjectMessage {
            source: collect_coord,
            dest: collect_coord,
            payload: vec![2.0, 20.0],
            payload_slot: 1,
        });

        let result = s.run();

        // dest0 gets position 0: [p0r0, p0r1] = [1, 2]
        assert_eq!(result.outputs.get(&dest0), Some(&vec![1.0, 2.0]));
        // dest1 gets position 1: [p1r0, p1r1] = [10, 20]
        assert_eq!(result.outputs.get(&dest1), Some(&vec![10.0, 20.0]));
    }
}
