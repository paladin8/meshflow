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
                "route_dest": null,
                "route_hops": null,
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
        "input_slots": [{
            "name": "a",
            "coord": [0, 0],
            "payload_slot": 0,
        }],
    });

    rmp_serde::to_vec_named(&program).unwrap()
}
