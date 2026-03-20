//! Artifact loading — deserialize a compiled RuntimeProgram and run it.

use std::collections::HashMap;

use serde::Deserialize;

use crate::coords::{Coord, Direction};
use crate::message::SlotId;
use crate::pe::{TaskConfig, TaskKind};
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
    max_events: u64,
}

#[derive(Debug, Deserialize)]
struct PEProgram {
    coord: (u32, u32),
    tasks: Vec<TaskProgram>,
    initial_sram: HashMap<u32, Vec<f32>>,
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
    },
    #[serde(rename = "collect_output")]
    CollectOutput { trigger_slot: u32, input_slot: u32 },
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
    #[error("duplicate input slot name: {0:?}")]
    DuplicateInputSlot(String),
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
    input_slots: HashMap<String, InputSlotInfo>,
}

#[derive(Debug)]
struct PEConfig {
    coord: Coord,
    tasks: Vec<TaskConfig>,
    initial_sram: HashMap<SlotId, Vec<f32>>,
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
        });
    }

    // Convert input slots
    let mut input_slots = HashMap::new();
    for slot in &program.input_slots {
        let coord = Coord::new(slot.coord.0, slot.coord.1);
        validate_coord(coord, width, height)?;
        if input_slots.contains_key(&slot.name) {
            return Err(ProgramError::DuplicateInputSlot(slot.name.clone()));
        }
        input_slots.insert(
            slot.name.clone(),
            InputSlotInfo {
                coord,
                payload_slot: slot.payload_slot,
            },
        );
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

        // Configure tasks on PEs
        for pe_config in &self.pe_configs {
            for task in &pe_config.tasks {
                sim.add_task_direct(pe_config.coord, task.clone());
            }

            // Pre-load initial SRAM
            for (slot, data) in &pe_config.initial_sram {
                sim.write_sram(pe_config.coord, *slot, data.clone());
            }
        }

        // Inject input messages
        for (name, payload) in inputs {
            let slot_info = self
                .input_slots
                .get(&name)
                .ok_or_else(|| ProgramError::UnknownInputSlot(name.clone()))?;

            sim.inject_message(InjectMessage {
                source: slot_info.coord,
                dest: slot_info.coord,
                payload,
                payload_slot: slot_info.payload_slot,
            });
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

fn parse_direction(s: &str) -> Result<Direction, ProgramError> {
    match s {
        "north" => Ok(Direction::North),
        "south" => Ok(Direction::South),
        "east" => Ok(Direction::East),
        "west" => Ok(Direction::West),
        _ => Err(ProgramError::InvalidDirection(s.to_string())),
    }
}

fn convert_task(task: &TaskProgram, width: u32, height: u32) -> Result<TaskConfig, ProgramError> {
    match task {
        TaskProgram::ForwardActivation {
            trigger_slot,
            input_slot,
            route_dest,
            route_hops,
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler_test_helpers::make_chain_artifact;

    #[test]
    fn load_valid_artifact() {
        let bytes = make_chain_artifact(3);
        let program = load_program(&bytes).unwrap();
        assert_eq!(program.config.width, 3);
        assert_eq!(program.config.height, 1);
        assert_eq!(program.pe_configs.len(), 3);
        assert_eq!(program.input_slots.len(), 1);
        assert!(program.input_slots.contains_key("a"));
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
    fn reject_duplicate_input_slot() {
        let program = rmp_serde::to_vec_named(&serde_json::json!({
            "version": 1,
            "mesh_config": {"width": 2, "height": 1, "hop_latency": 1, "task_base_latency": 1, "max_events": 100},
            "pe_programs": [],
            "input_slots": [
                {"name": "a", "coord": [0, 0], "payload_slot": 0},
                {"name": "a", "coord": [1, 0], "payload_slot": 0}
            ]
        }))
        .unwrap();
        let err = load_program(&program).unwrap_err();
        assert!(matches!(err, ProgramError::DuplicateInputSlot(_)));
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
}
