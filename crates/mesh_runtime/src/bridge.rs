use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::coords::Coord;
use crate::message::SlotId;
use crate::runtime::{InjectMessage, InjectTask, InjectTaskKind, SimConfig, Simulator};

/// Task kind enum exposed to Python.
#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskKind {
    ForwardActivation = 0,
    CollectOutput = 1,
}

/// Python-facing mesh configuration.
#[pyclass]
#[derive(Debug, Clone)]
pub struct MeshConfig {
    #[pyo3(get, set)]
    pub width: u32,
    #[pyo3(get, set)]
    pub height: u32,
    #[pyo3(get, set)]
    pub hop_latency: u64,
    #[pyo3(get, set)]
    pub task_base_latency: u64,
    #[pyo3(get, set)]
    pub max_events: u64,
}

#[pymethods]
impl MeshConfig {
    #[new]
    #[pyo3(signature = (width, height, hop_latency=1, task_base_latency=1, max_events=100_000))]
    fn new(
        width: u32,
        height: u32,
        hop_latency: u64,
        task_base_latency: u64,
        max_events: u64,
    ) -> Self {
        Self {
            width,
            height,
            hop_latency,
            task_base_latency,
            max_events,
        }
    }
}

impl From<&MeshConfig> for SimConfig {
    fn from(cfg: &MeshConfig) -> Self {
        SimConfig {
            width: cfg.width,
            height: cfg.height,
            hop_latency: cfg.hop_latency,
            task_base_latency: cfg.task_base_latency,
            max_events: cfg.max_events,
        }
    }
}

/// Stored message definition for injection.
struct StoredMessage {
    source: Coord,
    dest: Coord,
    payload: Vec<f32>,
    payload_slot: SlotId,
}

/// Stored task definition for injection.
struct StoredTask {
    coord: Coord,
    kind: InjectTaskKind,
    trigger_slot: SlotId,
}

/// Python-facing simulation input collector.
#[pyclass]
pub struct SimInput {
    messages: Vec<StoredMessage>,
    tasks: Vec<StoredTask>,
}

#[pymethods]
impl SimInput {
    #[new]
    fn new() -> Self {
        Self {
            messages: Vec::new(),
            tasks: Vec::new(),
        }
    }

    /// Add a message to inject into the simulation.
    #[pyo3(signature = (source, dest, payload, payload_slot=0))]
    fn add_message(
        &mut self,
        source: (u32, u32),
        dest: (u32, u32),
        payload: Vec<f32>,
        payload_slot: SlotId,
    ) -> PyResult<()> {
        if payload.is_empty() {
            return Err(PyValueError::new_err("payload must not be empty"));
        }
        self.messages.push(StoredMessage {
            source: Coord::new(source.0, source.1),
            dest: Coord::new(dest.0, dest.1),
            payload,
            payload_slot,
        });
        Ok(())
    }

    /// Add a task to configure on a PE before simulation.
    #[pyo3(signature = (coord, kind, trigger_slot, route_dest=None))]
    fn add_task(
        &mut self,
        coord: (u32, u32),
        kind: TaskKind,
        trigger_slot: SlotId,
        route_dest: Option<(u32, u32)>,
    ) -> PyResult<()> {
        let coord = Coord::new(coord.0, coord.1);
        // M1 convention: input_slot == trigger_slot (task reads from
        // the same slot that triggers it). Can be decoupled later.
        let task_kind = match kind {
            TaskKind::ForwardActivation => {
                let dest = route_dest.ok_or_else(|| {
                    PyValueError::new_err("route_dest is required for ForwardActivation tasks")
                })?;
                InjectTaskKind::ForwardActivation {
                    input_slot: trigger_slot,
                    route_dest: Coord::new(dest.0, dest.1),
                }
            }
            TaskKind::CollectOutput => InjectTaskKind::CollectOutput {
                input_slot: trigger_slot,
            },
        };

        self.tasks.push(StoredTask {
            coord,
            kind: task_kind,
            trigger_slot,
        });
        Ok(())
    }
}

/// Per-PE profiling statistics returned from simulation.
#[pyclass]
#[derive(Debug, Clone)]
pub struct PeStats {
    #[pyo3(get)]
    pub messages_received: u64,
    #[pyo3(get)]
    pub messages_sent: u64,
    #[pyo3(get)]
    pub tasks_executed: u64,
    #[pyo3(get)]
    pub slots_written: u64,
    #[pyo3(get)]
    pub max_queue_depth: u64,
}

/// Simulation result returned to Python.
#[pyclass]
#[derive(Debug)]
pub struct SimResult {
    #[pyo3(get)]
    pub outputs: PyObject,
    #[pyo3(get)]
    pub total_hops: u64,
    #[pyo3(get)]
    pub total_messages: u64,
    #[pyo3(get)]
    pub total_events_processed: u64,
    #[pyo3(get)]
    pub total_tasks_executed: u64,
    #[pyo3(get)]
    pub final_timestamp: u64,
    #[pyo3(get)]
    pub pe_stats: PyObject,
    #[pyo3(get)]
    pub trace_events: PyObject,
    #[pyo3(get)]
    pub operator_timings: PyObject,
    #[pyo3(get)]
    pub link_counts: PyObject,
}

/// Run a simulation with the given configuration and inputs.
#[pyfunction]
pub fn run_simulation(
    py: Python<'_>,
    config: &MeshConfig,
    inputs: &SimInput,
) -> PyResult<SimResult> {
    let sim_config: SimConfig = config.into();

    // Validate coordinates against mesh dimensions
    for msg in &inputs.messages {
        validate_coord(msg.source, config.width, config.height, "message source")?;
        validate_coord(msg.dest, config.width, config.height, "message dest")?;
    }
    for task in &inputs.tasks {
        validate_coord(task.coord, config.width, config.height, "task coord")?;
        if let InjectTaskKind::ForwardActivation { route_dest, .. } = &task.kind {
            validate_coord(*route_dest, config.width, config.height, "route_dest")?;
        }
    }

    let mut sim = Simulator::new(sim_config);

    // Configure tasks first (before injecting messages)
    for task in &inputs.tasks {
        sim.add_task(InjectTask {
            coord: task.coord,
            kind: task.kind.clone(),
            trigger_slot: task.trigger_slot,
        });
    }

    // Inject messages
    for msg in &inputs.messages {
        sim.inject_message(InjectMessage {
            source: msg.source,
            dest: msg.dest,
            payload: msg.payload.clone(),
            payload_slot: msg.payload_slot,
        });
    }

    // Run simulation — catch panics from invariant violations
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sim.run())).map_err(|e| {
            let msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "simulation failed with unknown error".to_string()
            };
            PyRuntimeError::new_err(msg)
        })?;

    sim_result_to_py(py, result)
}

fn sim_result_to_py(py: Python<'_>, result: crate::runtime::SimResult) -> PyResult<SimResult> {
    let outputs_dict = pyo3::types::PyDict::new(py);
    for (coord, data) in &result.outputs {
        let key = (coord.x, coord.y);
        outputs_dict.set_item(key, data.to_vec())?;
    }

    let pe_stats_dict = pyo3::types::PyDict::new(py);
    for (coord, counters) in &result.profile.per_pe {
        let key = (coord.x, coord.y);
        let stats = PeStats {
            messages_received: counters.messages_received,
            messages_sent: counters.messages_sent,
            tasks_executed: counters.tasks_executed,
            slots_written: counters.slots_written,
            max_queue_depth: counters.max_queue_depth,
        };
        pe_stats_dict.set_item(key, stats.into_pyobject(py)?)?;
    }

    // Convert trace events: Vec<TraceEvent> -> list[dict]
    let trace_list = pyo3::types::PyList::empty(py);
    for event in &result.profile.trace_events {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("timestamp", event.timestamp)?;
        d.set_item("coord", (event.coord.x, event.coord.y))?;
        d.set_item("kind", event.kind.to_string())?;
        d.set_item("detail", &event.detail)?;
        trace_list.append(d)?;
    }

    // Convert operator timings: Vec<OperatorTiming> -> list[dict]
    let timings_list = pyo3::types::PyList::empty(py);
    for timing in &result.profile.operator_timings {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("task_kind", &timing.task_kind)?;
        d.set_item("coord", (timing.coord.x, timing.coord.y))?;
        d.set_item("start_ts", timing.start_ts)?;
        d.set_item("end_ts", timing.end_ts)?;
        timings_list.append(d)?;
    }

    // Convert link counts: HashMap<(Coord, Coord), u64> -> dict[((x1,y1),(x2,y2)), count]
    let link_counts_dict = pyo3::types::PyDict::new(py);
    for ((from, to), count) in &result.profile.link_counts {
        let key = ((from.x, from.y), (to.x, to.y));
        link_counts_dict.set_item(key, *count)?;
    }

    Ok(SimResult {
        outputs: outputs_dict.into(),
        total_hops: result.profile.total_hops,
        total_messages: result.profile.total_messages,
        total_events_processed: result.profile.total_events_processed,
        total_tasks_executed: result.profile.total_tasks_executed,
        final_timestamp: result.profile.final_timestamp,
        pe_stats: pe_stats_dict.into(),
        trace_events: trace_list.into(),
        operator_timings: timings_list.into(),
        link_counts: link_counts_dict.into(),
    })
}

fn validate_coord(coord: Coord, width: u32, height: u32, label: &str) -> PyResult<()> {
    if coord.x >= width || coord.y >= height {
        Err(PyValueError::new_err(format!(
            "{} ({}, {}) is out of bounds for {}x{} mesh",
            label, coord.x, coord.y, width, height
        )))
    } else {
        Ok(())
    }
}

/// Run a compiled program artifact with input payloads.
#[pyfunction]
#[pyo3(signature = (program_bytes, inputs))]
pub fn run_program(
    py: Python<'_>,
    program_bytes: &[u8],
    inputs: std::collections::HashMap<String, Vec<f32>>,
) -> PyResult<SimResult> {
    let loaded = crate::program::load_program(program_bytes)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let result = loaded
        .run_with_inputs(inputs)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    sim_result_to_py(py, result)
}

/// Register bridge types and functions on the PyO3 module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TaskKind>()?;
    m.add_class::<MeshConfig>()?;
    m.add_class::<SimInput>()?;
    m.add_class::<PeStats>()?;
    m.add_class::<SimResult>()?;
    m.add_function(pyo3::wrap_pyfunction!(run_simulation, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(run_program, m)?)?;
    Ok(())
}
