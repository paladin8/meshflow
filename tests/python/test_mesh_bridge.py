"""Integration tests for the PyO3 mesh simulator bridge."""

import pytest
from meshflow._mesh_runtime import MeshConfig, PeStats, SimInput, TaskKind, run_simulation


class TestSmoke:
    def test_empty_simulation(self) -> None:
        cfg = MeshConfig(width=2, height=2)
        inp = SimInput()
        result = run_simulation(config=cfg, inputs=inp)

        assert result.outputs == {}
        assert result.total_messages == 0
        assert result.total_hops == 0
        assert result.total_events_processed == 0
        assert result.final_timestamp == 0

    def test_config_defaults(self) -> None:
        cfg = MeshConfig(width=3, height=3)
        assert cfg.hop_latency == 1
        assert cfg.task_base_latency == 1
        assert cfg.max_events == 100_000

    def test_config_custom(self) -> None:
        cfg = MeshConfig(width=4, height=4, hop_latency=2, task_base_latency=3, max_events=500)
        assert cfg.hop_latency == 2
        assert cfg.task_base_latency == 3
        assert cfg.max_events == 500


class TestSingleMessage:
    def test_delivery_with_hops(self) -> None:
        cfg = MeshConfig(width=4, height=4)
        inp = SimInput()
        inp.add_message(source=(0, 0), dest=(2, 2), payload=[1.0, 2.0, 3.0])
        inp.add_task(coord=(2, 2), kind=TaskKind.CollectOutput, trigger_slot=0)

        result = run_simulation(config=cfg, inputs=inp)

        assert result.outputs[(2, 2)] == [1.0, 2.0, 3.0]
        assert result.total_hops == 4
        assert result.total_messages == 1

    def test_same_pe_delivery(self) -> None:
        cfg = MeshConfig(width=2, height=2)
        inp = SimInput()
        inp.add_message(source=(0, 0), dest=(0, 0), payload=[7.0, 8.0])
        inp.add_task(coord=(0, 0), kind=TaskKind.CollectOutput, trigger_slot=0)

        result = run_simulation(config=cfg, inputs=inp)

        assert result.outputs[(0, 0)] == [7.0, 8.0]
        assert result.total_hops == 0
        assert result.total_messages == 1


class TestForwardChain:
    def test_three_pe_chain(self) -> None:
        """(0,0) -> forward -> (2,0) -> forward -> (4,0) -> collect"""
        cfg = MeshConfig(width=5, height=1)
        inp = SimInput()

        inp.add_task(
            coord=(0, 0), kind=TaskKind.ForwardActivation, trigger_slot=0, route_dest=(2, 0)
        )
        inp.add_task(
            coord=(2, 0), kind=TaskKind.ForwardActivation, trigger_slot=0, route_dest=(4, 0)
        )
        inp.add_task(coord=(4, 0), kind=TaskKind.CollectOutput, trigger_slot=0)

        inp.add_message(source=(0, 0), dest=(0, 0), payload=[1.0, 2.0, 3.0])

        result = run_simulation(config=cfg, inputs=inp)

        assert result.outputs[(4, 0)] == [1.0, 2.0, 3.0]


class TestProfiling:
    def test_hop_counts_and_messages(self) -> None:
        cfg = MeshConfig(width=3, height=1)
        inp = SimInput()
        inp.add_message(source=(0, 0), dest=(2, 0), payload=[1.0])
        inp.add_task(coord=(2, 0), kind=TaskKind.CollectOutput, trigger_slot=0)

        result = run_simulation(config=cfg, inputs=inp)

        assert result.total_hops == 2
        assert result.total_messages == 1
        assert result.total_tasks_executed == 1
        assert result.final_timestamp == 3  # 2 hops + 1 task

    def test_per_pe_stats(self) -> None:
        cfg = MeshConfig(width=3, height=1)
        inp = SimInput()
        inp.add_message(source=(0, 0), dest=(2, 0), payload=[1.0])
        inp.add_task(coord=(2, 0), kind=TaskKind.CollectOutput, trigger_slot=0)

        result = run_simulation(config=cfg, inputs=inp)

        pe00: PeStats = result.pe_stats[(0, 0)]
        assert pe00.messages_received == 1
        assert pe00.messages_sent == 1

        pe10: PeStats = result.pe_stats[(1, 0)]
        assert pe10.messages_received == 1
        assert pe10.messages_sent == 1

        pe20: PeStats = result.pe_stats[(2, 0)]
        assert pe20.messages_received == 1
        assert pe20.messages_sent == 0
        assert pe20.tasks_executed == 1
        assert pe20.slots_written == 1

    def test_events_processed(self) -> None:
        cfg = MeshConfig(width=3, height=1)
        inp = SimInput()
        inp.add_message(source=(0, 0), dest=(2, 0), payload=[1.0])
        inp.add_task(coord=(2, 0), kind=TaskKind.CollectOutput, trigger_slot=0)

        result = run_simulation(config=cfg, inputs=inp)

        # deliver@(0,0) [fwd], deliver@(1,0) [fwd], deliver@(2,0) [final], execute collect
        assert result.total_events_processed == 4


class TestValidation:
    def test_out_of_bounds_source(self) -> None:
        cfg = MeshConfig(width=2, height=2)
        inp = SimInput()
        inp.add_message(source=(5, 0), dest=(0, 0), payload=[1.0])

        with pytest.raises(ValueError, match="out of bounds"):
            run_simulation(config=cfg, inputs=inp)

    def test_out_of_bounds_dest(self) -> None:
        cfg = MeshConfig(width=2, height=2)
        inp = SimInput()
        inp.add_message(source=(0, 0), dest=(0, 5), payload=[1.0])

        with pytest.raises(ValueError, match="out of bounds"):
            run_simulation(config=cfg, inputs=inp)

    def test_out_of_bounds_task_coord(self) -> None:
        cfg = MeshConfig(width=2, height=2)
        inp = SimInput()
        inp.add_task(coord=(5, 0), kind=TaskKind.CollectOutput, trigger_slot=0)

        with pytest.raises(ValueError, match="out of bounds"):
            run_simulation(config=cfg, inputs=inp)

    def test_out_of_bounds_route_dest(self) -> None:
        cfg = MeshConfig(width=2, height=2)
        inp = SimInput()
        inp.add_task(
            coord=(0, 0), kind=TaskKind.ForwardActivation, trigger_slot=0, route_dest=(5, 0)
        )

        with pytest.raises(ValueError, match="out of bounds"):
            run_simulation(config=cfg, inputs=inp)

    def test_invalid_task_kind_type(self) -> None:
        inp = SimInput()
        with pytest.raises(TypeError):
            inp.add_task(coord=(0, 0), kind="bogus", trigger_slot=0)  # type: ignore[arg-type]

    def test_empty_payload(self) -> None:
        inp = SimInput()
        with pytest.raises(ValueError, match="payload must not be empty"):
            inp.add_message(source=(0, 0), dest=(1, 0), payload=[])

    def test_forward_activation_missing_route_dest(self) -> None:
        inp = SimInput()
        with pytest.raises(ValueError, match="route_dest is required"):
            inp.add_task(coord=(0, 0), kind=TaskKind.ForwardActivation, trigger_slot=0)
