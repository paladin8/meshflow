//! Artifact loading — deserialize a compiled RuntimeProgram and run it.

use std::collections::HashMap;

use serde::Deserialize;

use crate::coords::{Coord, Direction};
use crate::message::SlotId;
use crate::pe::{Activation, TaskConfig, TaskKind};
use crate::runtime::{InjectMessage, SimConfig, SimResult, Simulator};

// ---------------------------------------------------------------------------
// Serde structs (mirror the Python artifact schema)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct RuntimeProgram {
    version: u32,
    mesh_config: MeshProgramConfig,
    pe_programs: Vec<PEProgram>,
    input_slots: Vec<InputSlotProgram>,
}

#[derive(Debug, Deserialize)]
struct MeshProgramConfig {
    width: u32,
    height: u32,
    hop_latency: u64,
    task_base_latency: u64,
    #[serde(default = "default_cost_per_element")]
    cost_per_element: u64,
    max_events: u64,
}

fn default_cost_per_element() -> u64 {
    1
}

#[derive(Debug, Deserialize)]
struct PEProgram {
    coord: (u32, u32),
    tasks: Vec<TaskProgram>,
    initial_sram: HashMap<u32, Vec<f32>>,
    #[serde(default)]
    sram_capacity_bytes: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind")]
enum TaskProgram {
    #[serde(rename = "forward_activation")]
    ForwardActivation {
        trigger_slot: u32,
        input_slot: u32,
        route_dest: (u32, u32),
        route_hops: Vec<String>,
        #[serde(default)]
        payload_slot: u32,
    },
    #[serde(rename = "collect_output")]
    CollectOutput { trigger_slot: u32, input_slot: u32 },
    #[serde(rename = "linear")]
    Linear {
        trigger_slot: u32,
        input_slot: u32,
        weight_slot: u32,
        bias_slot: u32,
        tile_rows: u32,
        tile_cols: u32,
        route_dest: (u32, u32),
        route_hops: Vec<String>,
        fragment_slot: u32,
        fragment_offset: u32,
    },
    #[serde(rename = "concat_collect")]
    ConcatCollect {
        trigger_slot: u32,
        num_fragments: u32,
        total_rows: u32,
        fragment_offset: u32,
        #[serde(default)]
        fragment_rows: u32,
        #[serde(default)]
        num_positions: u32,
    },
    #[serde(rename = "concat_collect_forward")]
    ConcatCollectForward {
        trigger_slot: u32,
        num_fragments: u32,
        total_rows: u32,
        fragment_offset: u32,
        activation: Option<String>,
        route_dests: Vec<((u32, u32), Vec<String>)>,
        #[serde(default)]
        payload_slots: Vec<u32>,
        #[serde(default)]
        fragment_rows: u32,
        #[serde(default)]
        num_positions: u32,
        #[serde(default)]
        scatter: bool,
    },
    #[serde(rename = "add")]
    Add {
        trigger_slot: u32,
        input_slot_a: u32,
        input_slot_b: u32,
        output_slot: u32,
        output_dests: Vec<((u32, u32), Vec<String>)>,
        #[serde(default)]
        payload_slots: Vec<u32>,
    },
    #[serde(rename = "softmax")]
    Softmax {
        trigger_slot: u32,
        input_slot: u32,
        output_slot: u32,
    },
    #[serde(rename = "mat_mul")]
    MatMul {
        trigger_slot: u32,
        matrix_slot: u32,
        vector_slot: u32,
        rows: u32,
        cols: u32,
        #[serde(default)]
        transpose: bool,
        output_slot: u32,
        output_dests: Vec<((u32, u32), Vec<String>)>,
        #[serde(default)]
        payload_slots: Vec<u32>,
    },
    #[serde(rename = "rms_norm_partial_sum")]
    RmsNormPartialSum {
        trigger_slot: u32,
        input_slot: u32,
        reduce_dest: (u32, u32),
        reduce_hops: Vec<String>,
        partial_sum_slot: u32,
        #[serde(default)]
        slice_offset: u32,
        #[serde(default)]
        slice_size: u32,
        #[serde(default)]
        feature_count: u32,
    },
    #[serde(rename = "rms_norm_normalize")]
    RmsNormNormalize {
        trigger_slot: u32,
        input_slot: u32,
        scale_slot: u32,
        gamma_slot: u32,
        output_dests: Vec<((u32, u32), Vec<String>)>,
        #[serde(default)]
        payload_slots: Vec<u32>,
        #[serde(default)]
        slice_offset: u32,
        #[serde(default)]
        slice_size: u32,
    },
    #[serde(rename = "rms_norm_reduce")]
    RmsNormReduce {
        trigger_slot: u32,
        num_tiles: u32,
        feature_count: u32,
        eps: f32,
        tile_dests: Vec<((u32, u32), Vec<String>)>,
        scale_slot: u32,
    },
}

#[derive(Debug, Deserialize)]
struct InputSlotProgram {
    name: String,
    coord: (u32, u32),
    payload_slot: u32,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum ProgramError {
    #[error("failed to deserialize artifact: {0}")]
    Deserialize(String),
    #[error("unsupported artifact version: {0}")]
    UnsupportedVersion(u32),
    #[error("invalid direction: {0:?}")]
    InvalidDirection(String),
    #[error("coordinate ({0}, {1}) out of bounds for {2}x{3} mesh")]
    OutOfBounds(u32, u32, u32, u32),
    #[error("unknown input slot name: {0:?}")]
    UnknownInputSlot(String),
    #[error("simulation failed: {0}")]
    SimulationFailed(String),
}

// ---------------------------------------------------------------------------
// LoadedProgram
// ---------------------------------------------------------------------------

/// A deserialized and validated program, ready to run with inputs.
#[derive(Debug)]
pub struct LoadedProgram {
    config: SimConfig,
    pe_configs: Vec<PEConfig>,
    input_slots: HashMap<String, Vec<InputSlotInfo>>,
}

#[derive(Debug)]
struct PEConfig {
    coord: Coord,
    tasks: Vec<TaskConfig>,
    initial_sram: HashMap<SlotId, Vec<f32>>,
    sram_capacity_bytes: Option<usize>,
}

#[derive(Debug)]
struct InputSlotInfo {
    coord: Coord,
    payload_slot: SlotId,
}

/// Deserialize and validate a compiled artifact.
pub fn load_program(bytes: &[u8]) -> Result<LoadedProgram, ProgramError> {
    let program: RuntimeProgram =
        rmp_serde::from_slice(bytes).map_err(|e| ProgramError::Deserialize(e.to_string()))?;

    if program.version != 1 {
        return Err(ProgramError::UnsupportedVersion(program.version));
    }

    let width = program.mesh_config.width;
    let height = program.mesh_config.height;

    let config = SimConfig {
        width,
        height,
        hop_latency: program.mesh_config.hop_latency,
        task_base_latency: program.mesh_config.task_base_latency,
        cost_per_element: program.mesh_config.cost_per_element,
        max_events: program.mesh_config.max_events,
    };

    // Convert PE programs
    let mut pe_configs = Vec::with_capacity(program.pe_programs.len());
    for pe in &program.pe_programs {
        let coord = Coord::new(pe.coord.0, pe.coord.1);
        validate_coord(coord, width, height)?;

        let mut tasks = Vec::with_capacity(pe.tasks.len());
        for task in &pe.tasks {
            tasks.push(convert_task(task, width, height)?);
        }

        let initial_sram: HashMap<SlotId, Vec<f32>> = pe
            .initial_sram
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect();

        pe_configs.push(PEConfig {
            coord,
            tasks,
            initial_sram,
            sram_capacity_bytes: pe.sram_capacity_bytes,
        });
    }

    // Convert input slots — group by name for broadcast support
    let mut input_slots: HashMap<String, Vec<InputSlotInfo>> = HashMap::new();
    for slot in &program.input_slots {
        let coord = Coord::new(slot.coord.0, slot.coord.1);
        validate_coord(coord, width, height)?;
        input_slots
            .entry(slot.name.clone())
            .or_default()
            .push(InputSlotInfo {
                coord,
                payload_slot: slot.payload_slot,
            });
    }

    Ok(LoadedProgram {
        config,
        pe_configs,
        input_slots,
    })
}

impl LoadedProgram {
    /// Run the loaded program with the given input payloads.
    ///
    /// `inputs` maps input slot names (from GraphIR node IDs) to payload data.
    pub fn run_with_inputs(
        &self,
        inputs: HashMap<String, Vec<f32>>,
    ) -> Result<SimResult, ProgramError> {
        let mut sim = Simulator::new(self.config.clone());

        // Configure tasks and SRAM capacity on PEs
        for pe_config in &self.pe_configs {
            if let Some(cap) = pe_config.sram_capacity_bytes {
                sim.set_sram_capacity(pe_config.coord, cap);
            }

            for task in &pe_config.tasks {
                sim.add_task_direct(pe_config.coord, task.clone());
            }

            // Pre-load initial SRAM
            for (slot, data) in &pe_config.initial_sram {
                sim.write_sram(pe_config.coord, *slot, data.clone());
            }
        }

        // Inject input messages — broadcast to all matching entries
        for (name, payload) in inputs {
            let slot_infos = self
                .input_slots
                .get(&name)
                .ok_or_else(|| ProgramError::UnknownInputSlot(name.clone()))?;

            for slot_info in slot_infos {
                sim.inject_message(InjectMessage {
                    source: slot_info.coord,
                    dest: slot_info.coord,
                    payload: payload.clone(),
                    payload_slot: slot_info.payload_slot,
                });
            }
        }

        // Run — catch panics
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sim.run())).map_err(|e| {
                let msg = if let Some(s) = e.downcast_ref::<String>() {
                    s.clone()
                } else if let Some(s) = e.downcast_ref::<&str>() {
                    s.to_string()
                } else {
                    "simulation failed with unknown error".to_string()
                };
                ProgramError::SimulationFailed(msg)
            })?;

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn validate_coord(coord: Coord, width: u32, height: u32) -> Result<(), ProgramError> {
    if coord.x >= width || coord.y >= height {
        Err(ProgramError::OutOfBounds(coord.x, coord.y, width, height))
    } else {
        Ok(())
    }
}

fn parse_activation(s: &str) -> Result<Activation, ProgramError> {
    match s {
        "relu" => Ok(Activation::ReLU),
        _ => Err(ProgramError::Deserialize(format!(
            "unknown activation: {s:?}"
        ))),
    }
}

fn parse_direction(s: &str) -> Result<Direction, ProgramError> {
    match s {
        "north" => Ok(Direction::North),
        "south" => Ok(Direction::South),
        "east" => Ok(Direction::East),
        "west" => Ok(Direction::West),
        _ => Err(ProgramError::InvalidDirection(s.to_string())),
    }
}

fn convert_route_dests(
    dests: &[((u32, u32), Vec<String>)],
    width: u32,
    height: u32,
) -> Result<Vec<(Coord, Vec<Direction>)>, ProgramError> {
    dests
        .iter()
        .map(|(coord, hops)| {
            let c = Coord::new(coord.0, coord.1);
            validate_coord(c, width, height)?;
            let h: Vec<Direction> = hops
                .iter()
                .map(|s| parse_direction(s))
                .collect::<Result<_, _>>()?;
            Ok((c, h))
        })
        .collect()
}

fn convert_task(task: &TaskProgram, width: u32, height: u32) -> Result<TaskConfig, ProgramError> {
    match task {
        TaskProgram::ForwardActivation {
            trigger_slot,
            input_slot,
            route_dest,
            route_hops,
            payload_slot,
        } => {
            let dest = Coord::new(route_dest.0, route_dest.1);
            validate_coord(dest, width, height)?;
            let hops: Vec<Direction> = route_hops
                .iter()
                .map(|s| parse_direction(s))
                .collect::<Result<_, _>>()?;
            Ok(TaskConfig {
                kind: TaskKind::ForwardActivation {
                    input_slot: *input_slot,
                    route_dest: dest,
                    hops,
                    payload_slot: *payload_slot,
                },
                trigger_slot: *trigger_slot,
            })
        }
        TaskProgram::CollectOutput {
            trigger_slot,
            input_slot,
        } => Ok(TaskConfig {
            kind: TaskKind::CollectOutput {
                input_slot: *input_slot,
            },
            trigger_slot: *trigger_slot,
        }),
        TaskProgram::Linear {
            trigger_slot,
            input_slot,
            weight_slot,
            bias_slot,
            tile_rows,
            tile_cols,
            route_dest,
            route_hops,
            fragment_slot,
            fragment_offset,
        } => {
            let dest = Coord::new(route_dest.0, route_dest.1);
            validate_coord(dest, width, height)?;
            let hops: Vec<Direction> = route_hops
                .iter()
                .map(|s| parse_direction(s))
                .collect::<Result<_, _>>()?;
            Ok(TaskConfig {
                kind: TaskKind::Linear {
                    input_slot: *input_slot,
                    weight_slot: *weight_slot,
                    bias_slot: *bias_slot,
                    tile_rows: *tile_rows,
                    tile_cols: *tile_cols,
                    route_dest: dest,
                    hops,
                    fragment_slot: *fragment_slot,
                    fragment_offset: *fragment_offset,
                },
                trigger_slot: *trigger_slot,
            })
        }
        TaskProgram::ConcatCollect {
            trigger_slot,
            num_fragments,
            total_rows,
            fragment_offset,
            fragment_rows,
            num_positions,
        } => Ok(TaskConfig {
            kind: TaskKind::ConcatCollect {
                num_fragments: *num_fragments,
                total_rows: *total_rows,
                fragment_offset: *fragment_offset,
                fragment_rows: *fragment_rows,
                num_positions: *num_positions,
            },
            trigger_slot: *trigger_slot,
        }),
        TaskProgram::ConcatCollectForward {
            trigger_slot,
            num_fragments,
            total_rows,
            fragment_offset,
            activation,
            route_dests,
            payload_slots,
            fragment_rows,
            num_positions,
            scatter,
        } => {
            let act = activation.as_deref().map(parse_activation).transpose()?;
            let dests = convert_route_dests(route_dests, width, height)?;
            Ok(TaskConfig {
                kind: TaskKind::ConcatCollectForward {
                    num_fragments: *num_fragments,
                    total_rows: *total_rows,
                    fragment_offset: *fragment_offset,
                    activation: act,
                    route_dests: dests,
                    payload_slots: payload_slots.clone(),
                    fragment_rows: *fragment_rows,
                    num_positions: *num_positions,
                    scatter: *scatter,
                },
                trigger_slot: *trigger_slot,
            })
        }
        TaskProgram::Add {
            trigger_slot,
            input_slot_a,
            input_slot_b,
            output_slot,
            output_dests,
            payload_slots,
        } => {
            let dests = convert_route_dests(output_dests, width, height)?;
            Ok(TaskConfig {
                kind: TaskKind::Add {
                    input_slot_a: *input_slot_a,
                    input_slot_b: *input_slot_b,
                    output_slot: *output_slot,
                    output_dests: dests,
                    payload_slots: payload_slots.clone(),
                },
                trigger_slot: *trigger_slot,
            })
        }
        TaskProgram::Softmax {
            trigger_slot,
            input_slot,
            output_slot,
        } => Ok(TaskConfig {
            kind: TaskKind::Softmax {
                input_slot: *input_slot,
                output_slot: *output_slot,
            },
            trigger_slot: *trigger_slot,
        }),
        TaskProgram::MatMul {
            trigger_slot,
            matrix_slot,
            vector_slot,
            rows,
            cols,
            transpose,
            output_slot,
            output_dests,
            payload_slots,
        } => {
            let dests = convert_route_dests(output_dests, width, height)?;
            Ok(TaskConfig {
                kind: TaskKind::MatMul {
                    matrix_slot: *matrix_slot,
                    vector_slot: *vector_slot,
                    rows: *rows,
                    cols: *cols,
                    transpose: *transpose,
                    output_slot: *output_slot,
                    output_dests: dests,
                    payload_slots: payload_slots.clone(),
                },
                trigger_slot: *trigger_slot,
            })
        }
        TaskProgram::RmsNormPartialSum {
            trigger_slot,
            input_slot,
            reduce_dest,
            reduce_hops,
            partial_sum_slot,
            slice_offset,
            slice_size,
            feature_count,
        } => {
            let dest = Coord::new(reduce_dest.0, reduce_dest.1);
            validate_coord(dest, width, height)?;
            let hops: Vec<Direction> = reduce_hops
                .iter()
                .map(|s| parse_direction(s))
                .collect::<Result<_, _>>()?;
            Ok(TaskConfig {
                kind: TaskKind::RmsNormPartialSum {
                    input_slot: *input_slot,
                    reduce_dest: dest,
                    reduce_hops: hops,
                    partial_sum_slot: *partial_sum_slot,
                    slice_offset: *slice_offset,
                    slice_size: *slice_size,
                    feature_count: *feature_count,
                },
                trigger_slot: *trigger_slot,
            })
        }
        TaskProgram::RmsNormNormalize {
            trigger_slot,
            input_slot,
            scale_slot,
            gamma_slot,
            output_dests,
            payload_slots,
            slice_offset,
            slice_size,
        } => {
            let dests = convert_route_dests(output_dests, width, height)?;
            Ok(TaskConfig {
                kind: TaskKind::RmsNormNormalize {
                    input_slot: *input_slot,
                    scale_slot: *scale_slot,
                    gamma_slot: *gamma_slot,
                    output_dests: dests,
                    payload_slots: payload_slots.clone(),
                    slice_offset: *slice_offset,
                    slice_size: *slice_size,
                },
                trigger_slot: *trigger_slot,
            })
        }
        TaskProgram::RmsNormReduce {
            trigger_slot,
            num_tiles,
            feature_count,
            eps,
            tile_dests,
            scale_slot,
        } => {
            let dests = convert_route_dests(tile_dests, width, height)?;
            Ok(TaskConfig {
                kind: TaskKind::RmsNormReduce {
                    num_tiles: *num_tiles,
                    feature_count: *feature_count,
                    eps: *eps,
                    tile_dests: dests,
                    scale_slot: *scale_slot,
                },
                trigger_slot: *trigger_slot,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler_test_helpers::{make_chain_artifact, make_linear_artifact};

    #[test]
    fn load_valid_artifact() {
        let bytes = make_chain_artifact(3);
        let program = load_program(&bytes).unwrap();
        assert_eq!(program.config.width, 3);
        assert_eq!(program.config.height, 1);
        assert_eq!(program.pe_configs.len(), 3);
        assert_eq!(program.input_slots.len(), 1);
        assert!(program.input_slots.contains_key("a"));
        assert_eq!(program.input_slots.get("a").unwrap().len(), 1);
    }

    #[test]
    fn load_and_run_chain() {
        let bytes = make_chain_artifact(3);
        let program = load_program(&bytes).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("a".to_string(), vec![1.0, 2.0, 3.0]);
        let result = program.run_with_inputs(inputs).unwrap();

        // Payload should arrive at the collect node (2, 0)
        assert_eq!(
            result.outputs.get(&Coord::new(2, 0)),
            Some(&vec![1.0, 2.0, 3.0])
        );
    }

    #[test]
    fn reject_malformed_bytes() {
        let err = load_program(b"not valid msgpack").unwrap_err();
        assert!(matches!(err, ProgramError::Deserialize(_)));
    }

    #[test]
    fn reject_unsupported_version() {
        let program = rmp_serde::to_vec_named(&serde_json::json!({
            "version": 99,
            "mesh_config": {"width": 1, "height": 1, "hop_latency": 1, "task_base_latency": 1, "max_events": 100},
            "pe_programs": [],
            "input_slots": []
        }))
        .unwrap();
        let err = load_program(&program).unwrap_err();
        assert!(matches!(err, ProgramError::UnsupportedVersion(99)));
    }

    #[test]
    fn reject_invalid_task_kind() {
        // With serde tagged enum, unknown kind is a deserialization error
        let program = rmp_serde::to_vec_named(&serde_json::json!({
            "version": 1,
            "mesh_config": {"width": 2, "height": 1, "hop_latency": 1, "task_base_latency": 1, "max_events": 100},
            "pe_programs": [{
                "coord": [0, 0],
                "tasks": [{"kind": "bogus", "trigger_slot": 0, "input_slot": 0}],
                "initial_sram": {}
            }],
            "input_slots": []
        }))
        .unwrap();
        let err = load_program(&program).unwrap_err();
        assert!(matches!(err, ProgramError::Deserialize(_)));
    }

    #[test]
    fn reject_invalid_direction() {
        let bytes = rmp_serde::to_vec_named(&serde_json::json!({
            "version": 1,
            "mesh_config": {"width": 2, "height": 1, "hop_latency": 1, "task_base_latency": 1, "max_events": 100},
            "pe_programs": [{
                "coord": [0, 0],
                "tasks": [{"kind": "forward_activation", "trigger_slot": 0, "input_slot": 0, "route_dest": [1, 0], "route_hops": ["east", "bogus"]}],
                "initial_sram": {}
            }],
            "input_slots": []
        }))
        .unwrap();
        let err = load_program(&bytes).unwrap_err();
        assert!(matches!(err, ProgramError::InvalidDirection(_)));
    }

    #[test]
    fn reject_out_of_bounds_coord() {
        let program = rmp_serde::to_vec_named(&serde_json::json!({
            "version": 1,
            "mesh_config": {"width": 2, "height": 1, "hop_latency": 1, "task_base_latency": 1, "max_events": 100},
            "pe_programs": [{
                "coord": [5, 0],
                "tasks": [],
                "initial_sram": {}
            }],
            "input_slots": []
        }))
        .unwrap();
        let err = load_program(&program).unwrap_err();
        assert!(matches!(err, ProgramError::OutOfBounds(5, 0, 2, 1)));
    }

    #[test]
    fn broadcast_input_slots_grouped() {
        // Duplicate names are valid — they represent broadcast targets
        let bytes = rmp_serde::to_vec_named(&serde_json::json!({
            "version": 1,
            "mesh_config": {"width": 2, "height": 1, "hop_latency": 1, "task_base_latency": 1, "max_events": 100},
            "pe_programs": [],
            "input_slots": [
                {"name": "a", "coord": [0, 0], "payload_slot": 0},
                {"name": "a", "coord": [1, 0], "payload_slot": 0}
            ]
        }))
        .unwrap();
        let program = load_program(&bytes).unwrap();
        assert_eq!(program.input_slots.get("a").unwrap().len(), 2);
    }

    #[test]
    fn reject_unknown_input_slot() {
        let bytes = make_chain_artifact(2);
        let program = load_program(&bytes).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("nonexistent".to_string(), vec![1.0]);
        let err = program.run_with_inputs(inputs).unwrap_err();
        assert!(matches!(err, ProgramError::UnknownInputSlot(_)));
    }

    #[test]
    fn load_linear_artifact() {
        // 2x2 weight matrix, 2 tiles of 1 row each, vertical layout
        let weights = vec![1.0, 2.0, 3.0, 4.0]; // [[1,2],[3,4]]
        let bias = vec![0.5, 0.5];
        let bytes = make_linear_artifact(2, 2, 2, &weights, &bias);
        let program = load_program(&bytes).unwrap();

        assert_eq!(program.config.width, 1); // single column
        assert_eq!(program.config.height, 3); // 2 tiles + 1 collect
        assert_eq!(program.pe_configs.len(), 3);
        // Broadcast input "x" to 2 tile PEs
        assert_eq!(program.input_slots.get("x").unwrap().len(), 2);
    }

    #[test]
    fn linear_compute_correctness() {
        // W = [[1, 2], [3, 4]], b = [0.5, 0.5], x = [1, 1]
        // y = W @ x + b = [1*1+2*1+0.5, 3*1+4*1+0.5] = [3.5, 7.5]
        // Vertical layout: collect at (0, 2)
        let weights = vec![1.0, 2.0, 3.0, 4.0];
        let bias = vec![0.5, 0.5];
        let bytes = make_linear_artifact(2, 2, 2, &weights, &bias);
        let program = load_program(&bytes).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), vec![1.0, 1.0]);
        let result = program.run_with_inputs(inputs).unwrap();

        let output = result.outputs.get(&Coord::new(0, 2)).unwrap();
        assert_eq!(output.len(), 2);
        assert!((output[0] - 3.5).abs() < 1e-6);
        assert!((output[1] - 7.5).abs() < 1e-6);
    }

    #[test]
    fn linear_compute_larger() {
        // W = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,1,0,0],[0,0,1,1]]
        // 6x4 identity-like, b = [0;6], x = [1,2,3,4]
        // y = [1,2,3,4,3,7]
        // Vertical layout: 3 tiles at (0,0)-(0,2), collect at (0,3)
        let weights = vec![
            1.0, 0.0, 0.0, 0.0, // row 0
            0.0, 1.0, 0.0, 0.0, // row 1
            0.0, 0.0, 1.0, 0.0, // row 2
            0.0, 0.0, 0.0, 1.0, // row 3
            1.0, 1.0, 0.0, 0.0, // row 4
            0.0, 0.0, 1.0, 1.0, // row 5
        ];
        let bias = vec![0.0; 6];
        let bytes = make_linear_artifact(4, 6, 3, &weights, &bias);
        let program = load_program(&bytes).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), vec![1.0, 2.0, 3.0, 4.0]);
        let result = program.run_with_inputs(inputs).unwrap();

        let output = result.outputs.get(&Coord::new(0, 3)).unwrap();
        assert_eq!(output, &vec![1.0, 2.0, 3.0, 4.0, 3.0, 7.0]);
    }
}
