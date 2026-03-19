import enum

def runtime_version() -> str: ...

class TaskKind(enum.IntEnum):
    ForwardActivation = 0
    CollectOutput = 1

class MeshConfig:
    width: int
    height: int
    hop_latency: int
    task_base_latency: int
    max_events: int
    def __init__(
        self,
        width: int,
        height: int,
        hop_latency: int = 1,
        task_base_latency: int = 1,
        max_events: int = 100_000,
    ) -> None: ...

class SimInput:
    def __init__(self) -> None: ...
    def add_message(
        self,
        source: tuple[int, int],
        dest: tuple[int, int],
        payload: list[float],
        payload_slot: int = 0,
    ) -> None: ...
    def add_task(
        self,
        coord: tuple[int, int],
        kind: TaskKind,
        trigger_slot: int,
        route_dest: tuple[int, int] | None = None,
    ) -> None: ...

class PeStats:
    messages_received: int
    messages_sent: int
    tasks_executed: int
    slots_written: int

class SimResult:
    outputs: dict[tuple[int, int], list[float]]
    total_hops: int
    total_messages: int
    total_events_processed: int
    total_tasks_executed: int
    final_timestamp: int
    pe_stats: dict[tuple[int, int], PeStats]

def run_simulation(config: MeshConfig, inputs: SimInput) -> SimResult: ...
def run_program(program_bytes: bytes, inputs: dict[str, list[float]]) -> SimResult: ...
