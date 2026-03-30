"""Canonical benchmark for meshflow compiler performance.

Runs two transformer block configs through compile → serialize → run_program,
prints a comparison table, and optionally logs results to a JSONL file.

Usage:
    uv run python scripts/benchmark.py                     # print table
    uv run python scripts/benchmark.py --label "baseline"  # print + log
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from meshflow._mesh_runtime import run_program
from meshflow.compiler import CompilerConfig, compile
from meshflow.compiler.artifact import serialize
from meshflow.models.transformer import transformer_block, transformer_weights

CONFIGS = {
    "small": {"seq_len": 4, "d_model": 8, "d_ff": 16, "mesh_height": 6},
    "medium": {"seq_len": 8, "d_model": 16, "d_ff": 32, "mesh_height": 8},
}

WEIGHTS_SEED = 42
INPUT_SEED = 99

LOG_PATH = Path("artifacts/benchmarks/benchmark_log.jsonl")


def run_benchmark(config_name: str) -> dict:
    """Run a single benchmark config and return metrics."""
    cfg = CONFIGS[config_name]
    seq_len, d_model, d_ff = cfg["seq_len"], cfg["d_model"], cfg["d_ff"]
    mesh_height = cfg["mesh_height"]

    graph = transformer_block(seq_len, d_model, d_ff)
    weights = transformer_weights(d_model, d_ff, seed=WEIGHTS_SEED)
    compiler_config = CompilerConfig(mesh_height=mesh_height)
    program = compile(graph, compiler_config, weights=weights)
    artifact_bytes = serialize(program)

    torch.manual_seed(INPUT_SEED)
    x = torch.randn(seq_len, d_model)
    result = run_program(artifact_bytes, inputs={"input": x.flatten().tolist()})

    # Mesh shape
    mesh_w = program.mesh_config.width
    mesh_h = program.mesh_config.height
    active_pes = sum(1 for pe in program.pe_programs if pe.tasks)

    # Route analysis (Phase 3: hops no longer stored per-route, use Manhattan distance)
    total_routes = 0
    routing_table_entries = 0
    hop_counts: list[int] = []
    for pe in program.pe_programs:
        for task in pe.tasks:
            if hasattr(task, "routes"):
                for r in task.routes:
                    total_routes += 1
                    dx = abs(pe.coord[0] - r.dest[0])
                    dy = abs(pe.coord[1] - r.dest[1])
                    hop_counts.append(dx + dy)
        routing_table_entries += len(pe.routing_table)

    # PE stats
    max_sends = 0
    max_sends_pe = (0, 0)
    max_queue = 0
    max_queue_pe = (0, 0)
    for coord, stats in result.pe_stats.items():
        if stats.messages_sent > max_sends:
            max_sends = stats.messages_sent
            max_sends_pe = coord
        if stats.max_queue_depth > max_queue:
            max_queue = stats.max_queue_depth
            max_queue_pe = coord

    # Link contention
    sorted_links = sorted(result.link_counts.items(), key=lambda x: -x[1])
    hottest_link = sorted_links[0] if sorted_links else None

    return {
        "config": config_name,
        "mesh_width": mesh_w,
        "mesh_height": mesh_h,
        "total_pes": mesh_w * mesh_h,
        "active_pes": active_pes,
        "total_messages": result.total_messages,
        "total_hops": result.total_hops,
        "avg_hops_per_message": round(result.total_hops / max(result.total_messages, 1), 2),
        "total_routes": total_routes,
        "routing_table_entries": routing_table_entries,
        "max_hops": max(hop_counts) if hop_counts else 0,
        "final_timestamp": result.final_timestamp,
        "total_tasks_executed": result.total_tasks_executed,
        "max_sends": max_sends,
        "max_sends_pe": list(max_sends_pe),
        "max_queue_depth": max_queue,
        "max_queue_depth_pe": list(max_queue_pe),
        "hottest_link": f"{hottest_link[0][0]}->{hottest_link[0][1]}" if hottest_link else "",
        "hottest_link_count": hottest_link[1] if hottest_link else 0,
        "total_colors_used": result.total_colors_used,
        "max_colors_per_link": result.max_colors_per_link,
        "link_contentions": result.link_contentions,
        "total_link_wait_cycles": result.total_link_wait_cycles,
    }


def print_table(results: list[dict]) -> None:
    """Print a formatted comparison table."""
    metrics = [
        ("Mesh dimensions", lambda r: f"{r['mesh_width']}x{r['mesh_height']} = {r['total_pes']} PEs"),
        ("Active PEs", lambda r: f"{r['active_pes']} ({100 * r['active_pes'] // r['total_pes']}%)"),
        ("Total messages", lambda r: str(r["total_messages"])),
        ("Total hops", lambda r: str(r["total_hops"])),
        ("Avg hops/message", lambda r: str(r["avg_hops_per_message"])),
        ("Total routes", lambda r: str(r["total_routes"])),
        ("Routing table entries", lambda r: str(r["routing_table_entries"])),
        ("Max hops", lambda r: str(r["max_hops"])),
        ("Final timestamp", lambda r: str(r["final_timestamp"])),
        ("Tasks executed", lambda r: str(r["total_tasks_executed"])),
        ("Max sends (PE)", lambda r: f"{r['max_sends']} at {tuple(r['max_sends_pe'])}"),
        ("Max queue depth", lambda r: f"{r['max_queue_depth']} at {tuple(r['max_queue_depth_pe'])}"),
        ("Hottest link", lambda r: f"{r['hottest_link']}: {r['hottest_link_count']} msgs"),
        ("Total colors used", lambda r: str(r["total_colors_used"])),
        ("Max colors/link", lambda r: str(r["max_colors_per_link"])),
        ("Link contentions", lambda r: str(r["link_contentions"])),
        ("Link wait cycles", lambda r: str(r["total_link_wait_cycles"])),
    ]

    # Column widths
    label_w = max(len(m[0]) for m in metrics)
    col_w = 30

    # Header
    header = f"{'Metric':<{label_w}}  "
    for r in results:
        header += f"{r['config']:>{col_w}}  "
    print(header)
    print("-" * len(header))

    for label, fmt in metrics:
        row = f"{label:<{label_w}}  "
        for r in results:
            row += f"{fmt(r):>{col_w}}  "
        print(row)


def get_git_commit() -> str:
    """Get current HEAD short SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def log_results(results: list[dict], label: str) -> None:
    """Append results to the benchmark log."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "git_commit": get_git_commit(),
        "results": results,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"\nLogged to {LOG_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Meshflow benchmark")
    parser.add_argument("--label", help="Label for this run (enables logging)")
    args = parser.parse_args()

    print("Running benchmarks...\n")
    results = [run_benchmark(name) for name in CONFIGS]
    print_table(results)

    if args.label:
        log_results(results, args.label)


if __name__ == "__main__":
    main()
