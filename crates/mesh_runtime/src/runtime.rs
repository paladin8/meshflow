use std::collections::HashMap;

use crate::coords::{Coord, Direction};
use crate::event::{EventKind, EventQueue};
use crate::mesh::Mesh;
use crate::message::{Message, SlotId};
use crate::pe::{BroadcastRouteRuntime, RouteAction, TaskConfig, TaskKind};
use crate::profiling::{OperatorTiming, ProfileSummary, TraceEvent, TraceEventKind};
use crate::route::generate_route_xy;

/// Simulator configuration.
#[derive(Debug, Clone)]
pub struct SimConfig {
    pub width: u32,
    pub height: u32,
    pub hop_latency: u64,
    pub task_base_latency: u64,
    pub cost_per_element: u64,
    pub max_events: u64,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            width: 4,
            height: 4,
            hop_latency: 1,
            task_base_latency: 1,
            cost_per_element: 1,
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

/// Task kind for injection (routing table entries are computed automatically).
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
    /// Per-(directed_link, color) free-at time for contention tracking.
    /// Key is (from_coord, to_coord, color), value is the earliest time
    /// this (link, color) pair is available for a new message.
    link_color_free_at: HashMap<(Coord, Coord, u32), u64>,
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
            link_color_free_at: HashMap::new(),
            config,
        }
    }

    /// Compute task execution cost: fixed overhead + work-proportional cost.
    fn task_cost(&self, elements: u64) -> u64 {
        self.config.task_base_latency + elements * self.config.cost_per_element
    }

    /// Configure a task on a PE. For ForwardActivation tasks, routing table
    /// entries are computed and installed on intermediate PEs.
    pub fn add_task(&mut self, task: InjectTask) {
        let kind = match task.kind {
            InjectTaskKind::ForwardActivation {
                input_slot,
                route_dest,
            } => {
                // Compute route and install routing table entries on source + intermediate PEs.
                let hops = generate_route_xy(task.coord, route_dest);
                let color = 0u32; // test injection always uses color 0

                // Source PE: first hop direction
                if !hops.is_empty() {
                    self.mesh
                        .pe_mut(task.coord)
                        .routing_table
                        .insert(color, RouteAction::Forward(hops[0]));
                }

                // Walk the hop list and install routing table entries at intermediate PEs
                let mut current = task.coord;
                for (hop_idx, &dir) in hops.iter().enumerate() {
                    current = current
                        .step(dir, self.config.width, self.config.height)
                        .unwrap();
                    // Skip the final destination (last hop)
                    if hop_idx < hops.len() - 1 {
                        let next_dir = hops[hop_idx + 1];
                        self.mesh
                            .pe_mut(current)
                            .routing_table
                            .insert(color, RouteAction::Forward(next_dir));
                    }
                }

                TaskKind::ForwardActivation {
                    input_slot,
                    routes: vec![BroadcastRouteRuntime {
                        dest: route_dest,
                        payload_slot: 0,
                        color,
                    }],
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
    /// Used by artifact loading where routing tables are already in the artifact.
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

    /// Set a routing table entry on a PE.
    pub fn set_routing_entry(&mut self, coord: Coord, color: u32, action: RouteAction) {
        self.mesh.pe_mut(coord).routing_table.insert(color, action);
    }

    /// Inject a message into the simulation.
    ///
    /// For self-delivery (source == dest), enqueues at source for local delivery.
    /// For non-self messages, computes the first hop via XY routing and
    /// enqueues at the first neighbor (same as emit_message).
    pub fn inject_message(&mut self, msg: InjectMessage) {
        let id = self.next_message_id;
        self.next_message_id += 1;

        let message = Message {
            id,
            source: msg.source,
            dest: msg.dest,
            payload: msg.payload,
            payload_slot: msg.payload_slot,
            timestamp: 0,
            color: 0,
            hop_count: 0,
        };

        if msg.source == msg.dest {
            // Self-delivery: enqueue at source for local delivery
            self.track_enqueue(msg.source);
            self.queue
                .push(0, msg.source, EventKind::DeliverMessage { message });
        } else {
            // Non-self: compute route, install routing table entries on
            // intermediate PEs, then forward to first neighbor.
            let hops = generate_route_xy(msg.source, msg.dest);
            if hops.is_empty() {
                self.track_enqueue(msg.source);
                self.queue
                    .push(0, msg.source, EventKind::DeliverMessage { message });
                return;
            }

            // Install routing table entries along the full path
            // (source PE included for inject messages)
            let color = message.color;

            // Source PE: first hop direction
            self.mesh
                .pe_mut(msg.source)
                .routing_table
                .insert(color, RouteAction::Forward(hops[0]));

            // Intermediate PEs
            let mut current = msg.source;
            for (hop_idx, &dir) in hops.iter().enumerate() {
                current = current
                    .step(dir, self.config.width, self.config.height)
                    .unwrap();
                if hop_idx < hops.len() - 1 {
                    let next_dir = hops[hop_idx + 1];
                    self.mesh
                        .pe_mut(current)
                        .routing_table
                        .insert(color, RouteAction::Forward(next_dir));
                }
            }

            // Enqueue at source PE (process_deliver will use routing table)
            self.track_enqueue(msg.source);
            self.queue
                .push(0, msg.source, EventKind::DeliverMessage { message });
        }
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

        // Compute color summary stats from link_color_sets
        if !self.profile.link_color_sets.is_empty() {
            self.profile.max_colors_per_link = self
                .profile
                .link_color_sets
                .values()
                .map(|s| s.len() as u32)
                .max()
                .unwrap_or(0);

            let all_colors: std::collections::HashSet<u32> = self
                .profile
                .link_color_sets
                .values()
                .flat_map(|s| s.iter().copied())
                .collect();
            self.profile.total_colors_used = all_colors.len() as u32;
        }

        SimResult {
            outputs: self.outputs,
            profile: self.profile,
        }
    }

    fn process_deliver(&mut self, timestamp: u64, coord: Coord, message: &mut Message) {
        let pe = self.mesh.pe_mut(coord);
        pe.counters.messages_received += 1;

        // Check routing table, but only if we're not at the final destination.
        // Self-delivery messages (dest == coord) always deliver locally.
        let route_action = if message.dest != coord {
            pe.routing_table.get(&message.color).cloned()
        } else {
            None
        };

        if let Some(action) = route_action {
            match action {
                RouteAction::Forward(dir) => {
                    // Forward only — no local delivery
                    self.forward_message(timestamp, coord, message, dir);
                }
                RouteAction::DeliverAndForward {
                    direction,
                    deliver_slot,
                } => {
                    // Deliver a copy locally, then forward
                    pe.write_slot(deliver_slot, message.payload.clone());
                    pe.counters.slots_written += 1;
                    self.profile.total_messages += 1;

                    // Trigger tasks waiting on this slot
                    let triggered = pe.triggered_tasks(deliver_slot);
                    for task_index in triggered {
                        self.queue.push(
                            timestamp + self.config.task_base_latency,
                            coord,
                            EventKind::ExecuteTask { task_index },
                        );
                    }

                    self.forward_message(timestamp, coord, message, direction);
                }
            }
        } else {
            // No routing entry or at final destination — deliver payload locally
            let payload = std::mem::take(&mut message.payload);
            let payload_slot = message.payload_slot;
            let hop_count = message.hop_count as u64;

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

    /// Forward a message to the next PE in the given direction.
    fn forward_message(
        &mut self,
        timestamp: u64,
        coord: Coord,
        message: &mut Message,
        dir: Direction,
    ) {
        let neighbor = coord
            .step(dir, self.config.width, self.config.height)
            .unwrap_or_else(|| {
                panic!(
                    "Route goes out of bounds: PE {} stepping {:?} on {}x{} mesh",
                    coord, dir, self.config.width, self.config.height
                )
            });

        self.mesh.pe_mut(coord).counters.messages_sent += 1;

        // Track link usage
        *self
            .profile
            .link_counts
            .entry((coord, neighbor))
            .or_insert(0) += 1;

        // Track per-link color sets for profiling
        self.profile
            .link_color_sets
            .entry((coord, neighbor))
            .or_default()
            .insert(message.color);

        // Per-(link, color) contention tracking
        let key = (coord, neighbor, message.color);
        let free_at = self.link_color_free_at.get(&key).copied().unwrap_or(0);
        let entry_time = std::cmp::max(timestamp, free_at);

        if entry_time > timestamp {
            // Same-color contention detected
            self.profile.color_contentions += 1;
        }

        self.link_color_free_at
            .insert(key, entry_time + self.config.hop_latency);

        // Build forwarded message
        let forwarded = std::mem::take(&mut message.payload);
        let mut new_message = message.clone();
        new_message.payload = forwarded;
        new_message.hop_count += 1;

        self.enqueue_deliver(
            entry_time + self.config.hop_latency,
            neighbor,
            coord,
            new_message,
        );
    }

    // Reserved SRAM slot IDs for internal task state.
    const ACCUM_SLOT: SlotId = u32::MAX;
    const COUNTER_SLOT: SlotId = u32::MAX - 1;

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
                ref routes,
            } => {
                let pe = self.mesh.pe_mut(coord);
                let data = pe.read_slot(input_slot).clone();
                pe.counters.tasks_executed += 1;
                pe.counters.messages_sent += 1;
                self.profile.total_tasks_executed += 1;

                let send_time = timestamp + self.task_cost(0);
                let routes = routes.clone();
                self.broadcast_to_dests(send_time, coord, &routes, data);
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
                ref routes,
                fragment_offset: _,
            } => {
                let pe = self.mesh.pe_mut(coord);
                let x = pe.read_slot(input_slot).clone();
                let w = pe.read_slot(weight_slot).clone();
                let b = pe.read_slot(bias_slot).clone();
                pe.counters.tasks_executed += 1;
                pe.counters.messages_sent += 1;
                self.profile.total_tasks_executed += 1;

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

                let routes = routes.clone();
                let elements = (tile_rows as u64) * (tile_cols as u64) * (num_positions as u64);
                let send_time = timestamp + self.task_cost(elements);
                self.broadcast_to_dests(send_time, coord, &routes, y);
            }
            TaskKind::ConcatCollect {
                num_fragments,
                total_rows,
                fragment_offset,
                fragment_rows,
                num_positions,
            } => {
                let trigger_slot = self.mesh.pe(coord).tasks[task_index].trigger_slot;
                let completed = self.process_concat_fragment(
                    coord,
                    trigger_slot,
                    num_fragments,
                    total_rows,
                    fragment_offset,
                    fragment_rows,
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
                ref routes,
                fragment_rows,
                num_positions,
                scatter,
            } => {
                let trigger_slot = self.mesh.pe(coord).tasks[task_index].trigger_slot;
                let activation = activation.clone();
                let routes = routes.clone();
                let completed = self.process_concat_fragment(
                    coord,
                    trigger_slot,
                    num_fragments,
                    total_rows,
                    fragment_offset,
                    fragment_rows,
                    num_positions,
                );
                if let Some(mut result) = completed {
                    if let Some(act) = &activation {
                        act.apply(&mut result);
                    }

                    let base_time = timestamp + self.task_cost(0);
                    if scatter && !routes.is_empty() {
                        self.scatter_to_dests(base_time, coord, &routes, &result);
                    } else {
                        self.broadcast_to_dests(base_time, coord, &routes, result);
                    }
                }
            }
            TaskKind::Add {
                input_slot_a,
                input_slot_b,
                output_slot,
                ref routes,
            } => {
                let pe = self.mesh.pe_mut(coord);

                if !pe.has_slot(input_slot_a) || !pe.has_slot(input_slot_b) {
                    return;
                }

                pe.counters.tasks_executed += 1;
                self.profile.total_tasks_executed += 1;

                let a = pe.read_slot(input_slot_a).clone();
                let b = pe.read_slot(input_slot_b).clone();
                pe.remove_slot(input_slot_a);
                pe.remove_slot(input_slot_b);

                debug_assert_eq!(a.len(), b.len());
                let result: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
                let elements = a.len() as u64;
                let element_cost = elements * self.config.cost_per_element;
                self.task_write_slot(timestamp + element_cost, coord, output_slot, result.clone());

                let routes = routes.clone();
                let base_time = timestamp + self.task_cost(elements);
                self.broadcast_to_dests(base_time, coord, &routes, result);
            }
            TaskKind::Softmax {
                input_slot,
                output_slot,
            } => {
                let pe = self.mesh.pe_mut(coord);
                let data = pe.read_slot(input_slot).clone();
                pe.counters.tasks_executed += 1;
                self.profile.total_tasks_executed += 1;

                let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_vals: Vec<f32> = data.iter().map(|x| (x - max_val).exp()).collect();
                let sum: f32 = exp_vals.iter().sum();
                let result: Vec<f32> = exp_vals.iter().map(|x| x / sum).collect();

                let elements = 3 * data.len() as u64;
                let element_cost = elements * self.config.cost_per_element;
                self.task_write_slot(timestamp + element_cost, coord, output_slot, result);
            }
            TaskKind::MatMul {
                matrix_slot,
                vector_slot,
                rows,
                cols,
                transpose,
                output_slot,
                ref routes,
            } => {
                let pe = self.mesh.pe_mut(coord);

                if !pe.has_slot(matrix_slot) || !pe.has_slot(vector_slot) {
                    return;
                }

                pe.counters.tasks_executed += 1;
                self.profile.total_tasks_executed += 1;

                let matrix = pe.read_slot(matrix_slot).clone();
                let vector = pe.read_slot(vector_slot).clone();
                pe.remove_slot(matrix_slot);
                pe.remove_slot(vector_slot);
                let r = rows as usize;
                let c = cols as usize;

                let result: Vec<f32> = if !transpose {
                    (0..r)
                        .map(|i| (0..c).map(|j| matrix[i * c + j] * vector[j]).sum::<f32>())
                        .collect()
                } else {
                    let mut out = vec![0.0f32; c];
                    for i in 0..r {
                        for j in 0..c {
                            out[j] += matrix[i * c + j] * vector[i];
                        }
                    }
                    out
                };

                let elements = (rows as u64) * (cols as u64);
                let element_cost = elements * self.config.cost_per_element;
                self.task_write_slot(timestamp + element_cost, coord, output_slot, result.clone());

                let routes = routes.clone();
                let base_time = timestamp + self.task_cost(elements);
                self.broadcast_to_dests(base_time, coord, &routes, result);
            }
            TaskKind::RmsNormPartialSum {
                input_slot,
                ref routes,
                slice_offset,
                slice_size,
                feature_count,
            } => {
                let pe = self.mesh.pe_mut(coord);
                let data = pe.read_slot(input_slot).clone();
                pe.counters.tasks_executed += 1;
                pe.counters.messages_sent += 1;
                self.profile.total_tasks_executed += 1;

                let fc = feature_count as usize;
                let so = slice_offset as usize;
                let ss = slice_size as usize;

                let payload = if fc > 0 && data.len() > fc {
                    debug_assert_eq!(data.len() % fc, 0);
                    let num_positions = data.len() / fc;
                    (0..num_positions)
                        .map(|p| {
                            let start = p * fc + so;
                            let slice = &data[start..start + ss];
                            slice.iter().map(|x| x * x).sum::<f32>()
                        })
                        .collect::<Vec<f32>>()
                } else if ss > 0 {
                    let slice = &data[so..so + ss];
                    vec![slice.iter().map(|x| x * x).sum::<f32>()]
                } else {
                    vec![data.iter().map(|x| x * x).sum::<f32>()]
                };

                let routes = routes.clone();
                let num_pos = if fc > 0 && data.len() > fc {
                    data.len() / fc
                } else {
                    1
                };
                let elements = (ss as u64) * (num_pos as u64);
                let send_time = timestamp + self.task_cost(elements);
                self.broadcast_to_dests(send_time, coord, &routes, payload);
            }
            TaskKind::RmsNormNormalize {
                input_slot,
                scale_slot,
                gamma_slot,
                ref routes,
                slice_offset,
                slice_size,
            } => {
                let pe = self.mesh.pe_mut(coord);
                let data = pe.read_slot(input_slot).clone();
                let scales = pe.read_slot(scale_slot).clone();
                let gamma = pe.read_slot(gamma_slot).clone();
                pe.counters.tasks_executed += 1;
                self.profile.total_tasks_executed += 1;

                let so = slice_offset as usize;
                let ss = slice_size as usize;
                let num_positions = scales.len();

                let result = if ss > 0 && data.len() > ss {
                    debug_assert_eq!(data.len() % num_positions, 0);
                    let fc = data.len() / num_positions;
                    let mut out = Vec::with_capacity(num_positions * ss);
                    for j in 0..ss {
                        for (p, &scale) in scales.iter().enumerate() {
                            let start = p * fc + so;
                            let x = data[start + j];
                            out.push(x * scale * gamma[j]);
                        }
                    }
                    out
                } else {
                    let scale = scales[0];
                    debug_assert_eq!(data.len(), gamma.len());
                    data.iter()
                        .zip(gamma.iter())
                        .map(|(x, g)| x * scale * g)
                        .collect()
                };

                let routes = routes.clone();
                let elements = (ss as u64) * (num_positions as u64);
                let base_time = timestamp + self.task_cost(elements);
                self.broadcast_to_dests(base_time, coord, &routes, result);
            }
            TaskKind::RmsNormReduce {
                num_tiles,
                feature_count,
                eps,
                ref routes,
            } => {
                let trigger_slot = self.mesh.pe(coord).tasks[task_index].trigger_slot;
                let pe = self.mesh.pe_mut(coord);

                let partial_sums = pe.read_slot(trigger_slot).clone();
                let num_positions = partial_sums.len();

                let accum_slot = Self::ACCUM_SLOT;
                let counter_slot = Self::COUNTER_SLOT;

                if !pe.has_slot(accum_slot) {
                    pe.write_slot(accum_slot, vec![0.0; num_positions]);
                    pe.write_slot(counter_slot, vec![0.0]);
                }

                let mut running = pe.read_slot(accum_slot).clone();
                for (r, &ps) in running.iter_mut().zip(partial_sums.iter()) {
                    *r += ps;
                }
                pe.write_slot(accum_slot, running);
                pe.remove_slot(trigger_slot);

                let count = pe.read_slot(counter_slot)[0] as u32 + 1;
                pe.write_slot(counter_slot, vec![count as f32]);

                if count < num_tiles {
                    return;
                }

                pe.counters.tasks_executed += 1;
                self.profile.total_tasks_executed += 1;

                let running = pe.read_slot(accum_slot).clone();
                let fc = feature_count as f32;
                let scales: Vec<f32> = running
                    .iter()
                    .map(|&s| {
                        let mean_sq = s / fc;
                        1.0 / (mean_sq + eps).sqrt()
                    })
                    .collect();

                pe.remove_slot(accum_slot);
                pe.remove_slot(counter_slot);

                let local_scale_slot = routes.first().map(|r| r.payload_slot).unwrap_or(1);
                pe.write_slot(local_scale_slot, scales.clone());

                let routes = routes.clone();
                let elements = (num_tiles as u64) * (num_positions as u64);
                let base_time = timestamp + self.task_cost(elements);
                self.broadcast_to_dests(base_time, coord, &routes, scales);
            }
        }
    }

    /// Create and enqueue a single outbound message from `source` to `dest`.
    /// Enqueues at the source PE. process_deliver uses the routing table
    /// to forward (source PE has a routing table entry from the compiler).
    /// Self-delivery (source == dest) works because process_deliver checks
    /// dest == coord and delivers locally.
    fn emit_message(
        &mut self,
        timestamp: u64,
        source: Coord,
        dest: Coord,
        payload: Vec<f32>,
        payload_slot: SlotId,
        color: u32,
    ) {
        let message = Message {
            id: self.next_message_id,
            source,
            dest,
            payload,
            payload_slot,
            timestamp,
            color,
            hop_count: 0,
        };
        self.next_message_id += 1;
        self.enqueue_deliver(timestamp, source, source, message);
    }

    /// Broadcast `payload` (cloned) to every route destination, one message
    /// per route. Messages on different colors depart in parallel;
    /// messages on the same color are serialized within that color.
    fn broadcast_to_dests(
        &mut self,
        base_time: u64,
        coord: Coord,
        routes: &[BroadcastRouteRuntime],
        payload: Vec<f32>,
    ) {
        self.mesh.pe_mut(coord).counters.messages_sent += routes.len() as u64;
        let mut color_counts: HashMap<u32, u64> = HashMap::new();
        for route in routes.iter() {
            let count = color_counts.entry(route.color).or_insert(0);
            let send_time = base_time + *count;
            *count += 1;
            self.emit_message(
                send_time,
                coord,
                route.dest,
                payload.clone(),
                route.payload_slot,
                route.color,
            );
        }
    }

    /// Scatter `result` across `routes`: row `i` (of length `result.len() /
    /// routes.len()`) goes to `routes[i]`. Messages on different colors
    /// depart in parallel; messages on the same color are serialized.
    fn scatter_to_dests(
        &mut self,
        base_time: u64,
        coord: Coord,
        routes: &[BroadcastRouteRuntime],
        result: &[f32],
    ) {
        if routes.is_empty() {
            return;
        }
        let row_size = result.len() / routes.len();
        self.mesh.pe_mut(coord).counters.messages_sent += routes.len() as u64;
        let mut color_counts: HashMap<u32, u64> = HashMap::new();
        for (i, route) in routes.iter().enumerate() {
            let count = color_counts.entry(route.color).or_insert(0);
            let send_time = base_time + *count;
            *count += 1;
            let row = result[i * row_size..(i + 1) * row_size].to_vec();
            self.emit_message(
                send_time,
                coord,
                route.dest,
                row,
                route.payload_slot,
                route.color,
            );
        }
    }

    /// Write data to a PE's SRAM slot during task execution, then schedule
    /// any co-located tasks triggered by that slot.
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
    #[allow(clippy::too_many_arguments)]
    fn process_concat_fragment(
        &mut self,
        coord: Coord,
        trigger_slot: SlotId,
        num_fragments: u32,
        total_rows: u32,
        fragment_offset: u32,
        fragment_rows: u32,
        num_positions: u32,
    ) -> Option<Vec<f32>> {
        let pe = self.mesh.pe_mut(coord);
        let fragment = pe.read_slot(trigger_slot).clone();

        let frag_len = fragment.len();
        let accum_slot = Self::ACCUM_SLOT;
        let counter_slot = Self::COUNTER_SLOT;

        let fr = fragment_rows as usize;
        let num_pos = if num_positions > 0 {
            num_positions as usize
        } else if fr > 0 && frag_len > fr {
            frag_len / fr
        } else {
            1
        };
        let total_size = total_rows as usize * num_pos;
        let offset = fragment_offset as usize * num_pos;

        if !pe.has_slot(accum_slot) {
            pe.write_slot(accum_slot, vec![0.0; total_size]);
            pe.write_slot(counter_slot, vec![0.0]);
        }

        {
            let mut buf = pe.read_slot(accum_slot).clone();
            if offset + frag_len > buf.len() {
                panic!(
                    "PE {}: concat_fragment overflow: buf.len()={}, offset={}, frag_len={}, \
                     total_rows={}, num_fragments={}, fragment_offset={}, num_pos={}, \
                     num_positions_param={}",
                    coord,
                    buf.len(),
                    offset,
                    frag_len,
                    total_rows,
                    num_fragments,
                    fragment_offset,
                    num_pos,
                    num_positions
                );
            }
            buf[offset..offset + frag_len].copy_from_slice(&fragment);
            pe.write_slot(accum_slot, buf);
            pe.remove_slot(trigger_slot);
        }

        let count = {
            let c = pe.read_slot(counter_slot)[0] as u32 + 1;
            pe.write_slot(counter_slot, vec![c as f32]);
            c
        };

        if count == num_fragments {
            pe.counters.tasks_executed += 1;
            self.profile.total_tasks_executed += 1;

            let mut result = pe.remove_slot(accum_slot).unwrap();
            pe.remove_slot(counter_slot);

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

    /// Build a point-to-point BroadcastRouteRuntime.
    fn route_p2p(dest: Coord, payload_slot: SlotId) -> BroadcastRouteRuntime {
        BroadcastRouteRuntime {
            dest,
            payload_slot,
            color: 0,
        }
    }

    fn route_p2p_colored(dest: Coord, payload_slot: SlotId, color: u32) -> BroadcastRouteRuntime {
        BroadcastRouteRuntime {
            dest,
            payload_slot,
            color,
        }
    }

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

        // 1-hop path: emit_message handles the first hop, no routing entries needed
        // Install routing table for the inject message path (0,0)->(1,0)
        // Since inject uses color 0 and source (0,0) != dest (1,0),
        // the inject message gets its first hop from XY routing.
        // No intermediate PEs on a 1-hop path.

        let result = s.run();
        assert_eq!(
            result.outputs.get(&Coord::new(1, 0)),
            Some(&vec![1.0, 2.0, 3.0])
        );
        assert_eq!(result.profile.total_hops, 1);
        assert_eq!(result.profile.total_messages, 1);
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

        // Install routing table for intermediates only (emit_message handles first hop)
        // Path: (0,0)->E->(1,0)->E->(2,0)->E->(3,0)->N->(3,1)->N->(3,2)
        // Intermediates: (1,0), (2,0), (3,0), (3,1) — source (0,0) excluded
        s.set_routing_entry(Coord::new(1, 0), 0, RouteAction::Forward(Direction::East));
        s.set_routing_entry(Coord::new(2, 0), 0, RouteAction::Forward(Direction::East));
        s.set_routing_entry(Coord::new(3, 0), 0, RouteAction::Forward(Direction::North));
        s.set_routing_entry(Coord::new(3, 1), 0, RouteAction::Forward(Direction::North));

        let result = s.run();
        assert_eq!(result.outputs.get(&Coord::new(3, 2)), Some(&vec![42.0]));
        assert_eq!(result.profile.total_hops, 5);
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
        );
    }

    #[test]
    fn add_element_wise() {
        let mut s = sim(2, 1);

        // Source PE routing entry for color 0
        s.set_routing_entry(Coord::new(0, 0), 0, RouteAction::Forward(Direction::East));

        s.add_task_direct(
            Coord::new(0, 0),
            TaskConfig {
                kind: TaskKind::Add {
                    input_slot_a: 0,
                    input_slot_b: 1,
                    output_slot: 2,
                    routes: vec![route_p2p(Coord::new(1, 0), 0)],
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

        s.write_sram(Coord::new(0, 0), 0, vec![1.0, 2.0, 3.0]);
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
    fn softmax_basic() {
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

        assert_eq!(output.len(), 4);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        for i in 1..output.len() {
            assert!(output[i] > output[i - 1]);
        }
    }

    #[test]
    fn matmul_basic() {
        let mut s = sim(2, 1);

        // Source PE routing entry for color 0
        s.set_routing_entry(Coord::new(0, 0), 0, RouteAction::Forward(Direction::East));

        for trigger in [0, 1] {
            s.add_task_direct(
                Coord::new(0, 0),
                TaskConfig {
                    kind: TaskKind::MatMul {
                        matrix_slot: 0,
                        vector_slot: 1,
                        rows: 3,
                        cols: 2,
                        transpose: false,
                        output_slot: 2,
                        routes: vec![route_p2p(Coord::new(1, 0), 0)],
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

        s.write_sram(Coord::new(0, 0), 0, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        s.inject_message(InjectMessage {
            source: Coord::new(0, 0),
            dest: Coord::new(0, 0),
            payload: vec![1.0, 1.0],
            payload_slot: 1,
        });

        let result = s.run();
        let output = result.outputs.get(&Coord::new(1, 0)).unwrap();
        assert_eq!(output, &vec![3.0, 7.0, 11.0]);
    }

    #[test]
    fn rmsnorm_full_pipeline() {
        let mut s = sim(3, 2);

        let tile0 = Coord::new(0, 0);
        let tile1 = Coord::new(0, 1);
        let reduce = Coord::new(1, 0);
        let collect0 = Coord::new(2, 0);
        let collect1 = Coord::new(2, 1);

        // Install routing tables: source PEs need first-hop entries, intermediates need forwarding.
        // tile0 -> reduce: (0,0)->E->(1,0), color 0. Source (0,0) needs entry.
        s.set_routing_entry(tile0, 0, RouteAction::Forward(Direction::East));
        // tile1 -> reduce: (0,1)->E->(1,1)->S->(1,0), color 0. Source (0,1) + intermediate (1,1).
        s.set_routing_entry(tile1, 0, RouteAction::Forward(Direction::East));
        s.set_routing_entry(Coord::new(1, 1), 0, RouteAction::Forward(Direction::South));
        // tile0 -> collect0: (0,0)->E->(1,0)->E->(2,0), color 1. Source (0,0) + intermediate (1,0).
        s.set_routing_entry(tile0, 1, RouteAction::Forward(Direction::East));
        s.set_routing_entry(Coord::new(1, 0), 1, RouteAction::Forward(Direction::East));
        // tile1 -> collect1: (0,1)->E->(1,1)->E->(2,1), color 2. Source (0,1) + intermediate (1,1).
        s.set_routing_entry(tile1, 2, RouteAction::Forward(Direction::East));
        s.set_routing_entry(Coord::new(1, 1), 2, RouteAction::Forward(Direction::East));
        // reduce -> tile0: (1,0)->W->(0,0), color 3. Source (1,0).
        s.set_routing_entry(reduce, 3, RouteAction::Forward(Direction::West));
        // reduce -> tile1: (1,0)->W->(0,0)->N->(0,1), color 4. Source (1,0) + intermediate (0,0).
        s.set_routing_entry(reduce, 4, RouteAction::Forward(Direction::West));
        s.set_routing_entry(Coord::new(0, 0), 4, RouteAction::Forward(Direction::North));

        // Tile 0: RmsNormPartialSum + RmsNormNormalize
        s.add_task_direct(
            tile0,
            TaskConfig {
                kind: TaskKind::RmsNormPartialSum {
                    input_slot: 0,
                    routes: vec![BroadcastRouteRuntime {
                        dest: reduce,
                        payload_slot: 0,
                        color: 0,
                    }],
                    slice_offset: 0,
                    slice_size: 0,
                    feature_count: 0,
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
                    routes: vec![BroadcastRouteRuntime {
                        dest: collect0,
                        payload_slot: 0,
                        color: 1,
                    }],
                    slice_offset: 0,
                    slice_size: 0,
                },
                trigger_slot: 1,
            },
        );
        s.write_sram(tile0, 2, vec![1.0, 1.0]); // gamma

        // Tile 1
        s.add_task_direct(
            tile1,
            TaskConfig {
                kind: TaskKind::RmsNormPartialSum {
                    input_slot: 0,
                    routes: vec![BroadcastRouteRuntime {
                        dest: reduce,
                        payload_slot: 1,
                        color: 0,
                    }],
                    slice_offset: 0,
                    slice_size: 0,
                    feature_count: 0,
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
                    routes: vec![BroadcastRouteRuntime {
                        dest: collect1,
                        payload_slot: 0,
                        color: 2,
                    }],
                    slice_offset: 0,
                    slice_size: 0,
                },
                trigger_slot: 1,
            },
        );
        s.write_sram(tile1, 2, vec![1.0, 1.0]); // gamma

        // Reduce PE: broadcast scale to tile0 (slot 1) and tile1 (slot 1)
        // Use separate colors for the two broadcast destinations
        for i in 0..2u32 {
            s.add_task_direct(
                reduce,
                TaskConfig {
                    kind: TaskKind::RmsNormReduce {
                        num_tiles: 2,
                        feature_count: 4,
                        eps: 1e-6,
                        routes: vec![
                            BroadcastRouteRuntime {
                                dest: tile0,
                                payload_slot: 1,
                                color: 3,
                            },
                            BroadcastRouteRuntime {
                                dest: tile1,
                                payload_slot: 1,
                                color: 4,
                            },
                        ],
                    },
                    trigger_slot: i,
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

        // Inject inputs
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

        // Verify results
        let out0 = result.outputs.get(&collect0).unwrap();
        let out1 = result.outputs.get(&collect1).unwrap();

        // scale = 1/sqrt(25/4 + 1e-6) ~ 0.4
        let expected_scale = 1.0 / (25.0f32 / 4.0 + 1e-6).sqrt();
        assert!((out0[0] - 3.0 * expected_scale).abs() < 1e-4);
        assert!((out0[1] - 4.0 * expected_scale).abs() < 1e-4);
        assert!((out1[0]).abs() < 1e-4);
        assert!((out1[1]).abs() < 1e-4);
    }

    #[test]
    fn parallel_send_different_colors_broadcast() {
        let mut s = sim(4, 1);

        for x in 1..=3 {
            s.add_task(InjectTask {
                coord: Coord::new(x, 0),
                kind: InjectTaskKind::CollectOutput { input_slot: x },
                trigger_slot: x,
            });
        }

        // Install routing tables (source + intermediate PEs)
        // Route color 0: (0,0)->E->(1,0) dest
        s.set_routing_entry(Coord::new(0, 0), 0, RouteAction::Forward(Direction::East));
        // Route color 1: (0,0)->E->(1,0)->E->(2,0) dest
        s.set_routing_entry(Coord::new(0, 0), 1, RouteAction::Forward(Direction::East));
        s.set_routing_entry(Coord::new(1, 0), 1, RouteAction::Forward(Direction::East));
        // Route color 2: (0,0)->E->(1,0)->E->(2,0)->E->(3,0) dest
        s.set_routing_entry(Coord::new(0, 0), 2, RouteAction::Forward(Direction::East));
        s.set_routing_entry(Coord::new(1, 0), 2, RouteAction::Forward(Direction::East));
        s.set_routing_entry(Coord::new(2, 0), 2, RouteAction::Forward(Direction::East));

        let routes = vec![
            route_p2p_colored(Coord::new(1, 0), 1, 0),
            route_p2p_colored(Coord::new(2, 0), 2, 1),
            route_p2p_colored(Coord::new(3, 0), 3, 2),
        ];

        s.write_sram(Coord::new(0, 0), 0, vec![10.0, 20.0]);

        let task = TaskConfig {
            kind: TaskKind::ConcatCollectForward {
                num_fragments: 1,
                total_rows: 1,
                fragment_offset: 0,
                fragment_rows: 1,
                activation: None,
                routes,
                scatter: false,
                num_positions: 0,
            },
            trigger_slot: 0,
        };
        s.add_task_direct(Coord::new(0, 0), task);

        s.inject_message(InjectMessage {
            source: Coord::new(0, 0),
            dest: Coord::new(0, 0),
            payload: vec![10.0, 20.0],
            payload_slot: 0,
        });

        let result = s.run();

        assert_eq!(
            result.outputs.get(&Coord::new(1, 0)),
            Some(&vec![10.0, 20.0])
        );
        assert_eq!(
            result.outputs.get(&Coord::new(2, 0)),
            Some(&vec![10.0, 20.0])
        );
        assert_eq!(
            result.outputs.get(&Coord::new(3, 0)),
            Some(&vec![10.0, 20.0])
        );

        assert_eq!(result.profile.color_contentions, 0);
    }
}
