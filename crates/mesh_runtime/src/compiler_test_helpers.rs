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
            // ForwardActivation: route to next PE
            let mut hops = Vec::new();
            // From (i, 0) to (i+1, 0): one hop east
            hops.push(serde_json::json!("east"));
            vec![serde_json::json!({
                "kind": "forward_activation",
                "trigger_slot": 0,
                "input_slot": 0,
                "route_dest": [i + 1, 0],
                "route_hops": hops,
            })]
        } else {
            // CollectOutput at the end
            vec![serde_json::json!({
                "kind": "collect_output",
                "trigger_slot": 0,
                "input_slot": 0,
            })]
        };

        pe_programs.push(serde_json::json!({
            "coord": [i, 0],
            "tasks": tasks,
            "initial_sram": {},
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

    // Use serde-serializable structs so initial_sram gets integer keys in msgpack.
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
        tasks: Vec<Task>,
        initial_sram: std::collections::HashMap<u32, Vec<f32>>,
    }
    #[derive(Serialize)]
    struct Task {
        kind: String,
        trigger_slot: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        input_slot: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        weight_slot: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        bias_slot: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tile_rows: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tile_cols: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        route_dest: Option<(u32, u32)>,
        #[serde(skip_serializing_if = "Option::is_none")]
        route_hops: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        fragment_slot: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        fragment_offset: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        num_fragments: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        total_rows: Option<u32>,
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

        // Hops: north from (0, i) to (0, collect_y)
        let hops_count = collect_y as usize - i;
        let hops: Vec<String> = (0..hops_count).map(|_| "north".to_string()).collect();

        let weight_tile: Vec<f32> = (frag_offset..frag_offset + tile_rows)
            .flat_map(|r| &weights[r * in_features..(r + 1) * in_features])
            .copied()
            .collect();
        let bias_tile: Vec<f32> = bias[frag_offset..frag_offset + tile_rows].to_vec();

        let mut sram = std::collections::HashMap::new();
        sram.insert(1u32, weight_tile);
        sram.insert(2u32, bias_tile);

        pe_programs.push(PEProg {
            coord: (0, i as u32),
            tasks: vec![Task {
                kind: "linear".to_string(),
                trigger_slot: 0,
                input_slot: Some(0),
                weight_slot: Some(1),
                bias_slot: Some(2),
                tile_rows: Some(tile_rows as u32),
                tile_cols: Some(in_features as u32),
                route_dest: Some((0, collect_y)),
                route_hops: Some(hops),
                fragment_slot: Some(i as u32),
                fragment_offset: Some(frag_offset as u32),
                num_fragments: None,
                total_rows: None,
            }],
            initial_sram: sram,
        });

        input_slots_vec.push(InputSlot {
            name: "x".to_string(),
            coord: (0, i as u32),
            payload_slot: 0,
        });
    }

    // Collect PE with concat_collect tasks
    let mut collect_tasks = Vec::new();
    for i in 0..num_tiles {
        let frag_offset = i * base + std::cmp::min(i, remainder);
        collect_tasks.push(Task {
            kind: "concat_collect".to_string(),
            trigger_slot: i as u32,
            input_slot: None,
            weight_slot: None,
            bias_slot: None,
            tile_rows: None,
            tile_cols: None,
            route_dest: None,
            route_hops: None,
            fragment_slot: None,
            fragment_offset: Some(frag_offset as u32),
            num_fragments: Some(num_tiles as u32),
            total_rows: Some(out_features as u32),
        });
    }

    pe_programs.push(PEProg {
        coord: (0, collect_y),
        tasks: collect_tasks,
        initial_sram: std::collections::HashMap::new(),
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
