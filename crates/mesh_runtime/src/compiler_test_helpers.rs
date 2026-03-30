//! Test helpers that build msgpack artifacts matching what the Python compiler produces.
//! Only compiled in test builds.

/// Build a msgpack artifact for an N-node forward chain: a -> b -> ... -> collect.
/// Nodes are placed sequentially on an Nx1 mesh.
/// First N-1 nodes are ForwardActivation, last is CollectOutput.
/// Input slot "a" is at (0, 0).
pub fn make_chain_artifact(n: usize) -> Vec<u8> {
    assert!(n >= 2, "chain needs at least 2 nodes");

    let mut pe_programs = Vec::new();
    for i in 0..n {
        let tasks = if i < n - 1 {
            // ForwardActivation: route to next PE via routes list
            vec![serde_json::json!({
                "kind": "forward_activation",
                "trigger_slot": 0,
                "input_slot": 0,
                "routes": [{"dest": [i + 1, 0], "payload_slot": 0, "color": 0}],
            })]
        } else {
            // CollectOutput at the end
            vec![serde_json::json!({
                "kind": "collect_output",
                "trigger_slot": 0,
                "input_slot": 0,
            })]
        };

        // Build routing table: all PEs except the collect PE forward color 0 east.
        // Source PE included since emit_message enqueues at source and
        // process_deliver uses the routing table.
        let routing_table = if i < n - 1 {
            serde_json::json!({"0": {"direction": "east"}})
        } else {
            serde_json::json!({})
        };

        pe_programs.push(serde_json::json!({
            "coord": [i, 0],
            "tasks": tasks,
            "initial_sram": {},
            "routing_table": routing_table,
        }));
    }

    let program = serde_json::json!({
        "version": 1,
        "mesh_config": {
            "width": n,
            "height": 1,
            "hop_latency": 1,
            "task_base_latency": 1,
            "max_events": 100_000,
        },
        "pe_programs": pe_programs,
        "input_slots": [{"name": "a", "coord": [0, 0], "payload_slot": 0}],
    });

    rmp_serde::to_vec_named(&program).unwrap()
}

/// Build a msgpack artifact for a tiled linear layer using vertical column layout.
///
/// Creates `num_tiles` tile PEs + 1 collect PE on a 1x(num_tiles+1) mesh.
/// Tiles are stacked vertically: tile i at (0, i), collect at (0, num_tiles).
/// Weight matrix is `(out_features, in_features)`, bias is `(out_features,)`.
/// Broadcast input "x" to all tile PEs.
pub fn make_linear_artifact(
    in_features: usize,
    out_features: usize,
    num_tiles: usize,
    weights: &[f32],
    bias: &[f32],
) -> Vec<u8> {
    use serde::Serialize;

    assert_eq!(weights.len(), out_features * in_features);
    assert_eq!(bias.len(), out_features);

    #[derive(Serialize)]
    struct Program {
        version: u32,
        mesh_config: MeshConfig,
        pe_programs: Vec<PEProg>,
        input_slots: Vec<InputSlot>,
    }
    #[derive(Serialize)]
    struct MeshConfig {
        width: u32,
        height: u32,
        hop_latency: u64,
        task_base_latency: u64,
        max_events: u64,
    }
    #[derive(Serialize)]
    struct PEProg {
        coord: (u32, u32),
        tasks: Vec<serde_json::Value>,
        initial_sram: std::collections::HashMap<u32, Vec<f32>>,
        routing_table: std::collections::HashMap<String, RouteEntry>,
    }
    #[derive(Serialize)]
    struct RouteEntry {
        direction: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        deliver_slot: Option<u32>,
    }
    #[derive(Serialize)]
    struct InputSlot {
        name: String,
        coord: (u32, u32),
        payload_slot: u32,
    }

    let base = out_features / num_tiles;
    let remainder = out_features % num_tiles;
    let height = num_tiles + 1;
    let collect_y = num_tiles as u32;

    let mut pe_programs = Vec::new();
    let mut input_slots_vec = Vec::new();

    for i in 0..num_tiles {
        let tile_rows = if i < remainder { base + 1 } else { base };
        let frag_offset = i * base + std::cmp::min(i, remainder);

        let weight_tile: Vec<f32> = (frag_offset..frag_offset + tile_rows)
            .flat_map(|r| &weights[r * in_features..(r + 1) * in_features])
            .copied()
            .collect();
        let bias_tile: Vec<f32> = bias[frag_offset..frag_offset + tile_rows].to_vec();

        let mut sram = std::collections::HashMap::new();
        sram.insert(1u32, weight_tile);
        sram.insert(2u32, bias_tile);

        // Routing table: each tile PE needs to forward color 0 north
        // (message from tile to collect PE goes north).
        // But the tile PE is the source, so it doesn't need a routing entry for
        // its own emitted messages. The intermediate PEs between tile and collect
        // need routing entries.
        // All tiles use color 0 and route north to the collect PE.
        // Intermediate PEs (tiles above this one) forward color 0 north.

        let task = serde_json::json!({
            "kind": "linear",
            "trigger_slot": 0,
            "input_slot": 0,
            "weight_slot": 1,
            "bias_slot": 2,
            "tile_rows": tile_rows as u32,
            "tile_cols": in_features as u32,
            "routes": [{"dest": [0, collect_y], "payload_slot": i as u32, "color": 0}],
            "fragment_offset": frag_offset as u32,
        });

        // Routing table for this PE: forward color 0 north.
        // All tile PEs need this entry: source tiles for their own messages,
        // and intermediate tiles for messages from tiles below.
        let mut rt = std::collections::HashMap::new();
        if (i as u32) < collect_y {
            rt.insert(
                "0".to_string(),
                RouteEntry {
                    direction: "north".to_string(),
                    deliver_slot: None,
                },
            );
        }

        pe_programs.push(PEProg {
            coord: (0, i as u32),
            tasks: vec![task],
            initial_sram: sram,
            routing_table: rt,
        });

        input_slots_vec.push(InputSlot {
            name: "x".to_string(),
            coord: (0, i as u32),
            payload_slot: 0,
        });
    }

    // Collect PE with concat_collect tasks — no routing table needed (final dest)
    let mut collect_tasks = Vec::new();
    for i in 0..num_tiles {
        let frag_offset = i * base + std::cmp::min(i, remainder);
        collect_tasks.push(serde_json::json!({
            "kind": "concat_collect",
            "trigger_slot": i as u32,
            "num_fragments": num_tiles as u32,
            "total_rows": out_features as u32,
            "fragment_offset": frag_offset as u32,
        }));
    }

    pe_programs.push(PEProg {
        coord: (0, collect_y),
        tasks: collect_tasks,
        initial_sram: std::collections::HashMap::new(),
        routing_table: std::collections::HashMap::new(),
    });

    let program = Program {
        version: 1,
        mesh_config: MeshConfig {
            width: 1,
            height: height as u32,
            hop_latency: 1,
            task_base_latency: 1,
            max_events: 100_000,
        },
        pe_programs,
        input_slots: input_slots_vec,
    };

    rmp_serde::to_vec_named(&program).unwrap()
}
