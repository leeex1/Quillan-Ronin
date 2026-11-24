#!/usr/bin/env python3
"""Deep correctness and safety checks for the Quantum VM master.

Verifies data integrity under awkward chunk sizes, failure isolation,
concurrency, session cleanup, and whether the reported 'boost' is a real
speed-up or only a simulated metric.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantum_master import QuantumMaster  # noqa: E402


def _echo_side_effect(master: QuantumMaster) -> AsyncMock:
    async def echo(url: str, payload: dict) -> dict:
        chunk = list(payload["data"])
        return {
            "status": "success",
            "boost_applied": 1 if chunk else 0,
            "data": chunk,
            "operation": payload["operation"],
        }

    return AsyncMock(side_effect=echo)


def test_nondivisible_chunking_preserves_all_items() -> None:
    master = QuantumMaster(copies=3)
    data = list(range(14))
    with patch.object(
        master, "send_to_instance", new=_echo_side_effect(master)
    ) as mocked:
        result = asyncio.run(
            master.coordinate_task({"data": data, "operation": "compute"})
        )
    chunks = [call.args[1]["data"] for call in mocked.call_args_list]
    assert [len(chunk) for chunk in chunks] == [5, 5, 4]
    assert sum(chunks, []) == data
    assert result["total_boost"] == 3
    assert result["results"] == data
    print("[PASS] non-divisible chunking: 14 items -> 5/5/4, no loss")


def test_more_copies_than_items_marks_empty_chunks() -> None:
    master = QuantumMaster(copies=5)
    with patch.object(
        master, "send_to_instance", new=_echo_side_effect(master)
    ) as mocked:
        result = asyncio.run(master.coordinate_task({"data": [7, 8]}))
    payloads = [call.args[1] for call in mocked.call_args_list]
    assert [len(p["data"]) for p in payloads] == [1, 1, 0, 0, 0]
    assert [p["is_empty"] for p in payloads] == [False, False, True, True, True]
    assert result["results"] == [7, 8]
    print("[PASS] copies > items: empty chunks flagged, no data loss")


def test_missing_data_and_error_isolation() -> None:
    master = QuantumMaster(copies=3)
    with patch.object(master, "send_to_instance", new=_echo_side_effect(master)):
        empty = asyncio.run(master.coordinate_task({"operation": "compute"}))
    assert empty["results"] == [] and empty["total_boost"] == 0

    async def flaky(url: str, payload: dict) -> dict:
        if payload["chunk_id"] == 1:
            raise RuntimeError("instance-2 offline")
        return {"status": "success", "boost_applied": 2, "data": payload["data"]}

    with patch.object(master, "send_to_instance", new=AsyncMock(side_effect=flaky)):
        partial = asyncio.run(master.coordinate_task({"data": list(range(9))}))
    assert partial["total_boost"] == 4
    assert len(partial["results"]) == 6
    print("[PASS] missing data + one failed instance handled without crash")


def test_concurrent_coordinate_calls_do_not_cross_contaminate() -> None:
    master = QuantumMaster(copies=3)

    async def run() -> None:
        async def echo(url: str, payload: dict) -> dict:
            await asyncio.sleep(0.001)
            return {"status": "success", "boost_applied": 1, "data": payload["data"]}

        with patch.object(master, "send_to_instance", new=AsyncMock(side_effect=echo)):
            results = await asyncio.gather(
                *[master.coordinate_task({"data": [n] * 6}) for n in range(10)]
            )
        for n, result in enumerate(results):
            assert result["results"] == [n] * 6

    asyncio.run(run())
    from quantum_master import _SESSION

    assert _SESSION.get() is None
    print("[PASS] 10 concurrent coordinations isolated; sessions closed")


def test_boost_is_a_simulated_metric_not_measured_acceleration() -> None:
    master = QuantumMaster(copies=11)
    workload = list(range(1100))

    def baseline() -> list:
        total = []
        for item in workload:
            total.append(item * item)
        return total

    with patch.object(master, "send_to_instance", new=_echo_side_effect(master)):
        start = time.perf_counter()
        result = asyncio.run(master.coordinate_task({"data": workload}))
        vm_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    expected = baseline()
    base_elapsed = time.perf_counter() - start

    assert result["results"] == workload  # coordinator only echoes chunks
    assert result["total_boost"] == 11  # a label, not measured work
    assert expected == [item * item for item in workload]
    print(
        f"[INFO] VM coordination {vm_elapsed * 1000:.2f} ms vs direct loop "
        f"{base_elapsed * 1000:.3f} ms; reported boost=11 is a simulated metric"
    )
    print("[PASS] boost semantics disclosed: no real CPU acceleration claimed")


if __name__ == "__main__":
    test_nondivisible_chunking_preserves_all_items()
    test_more_copies_than_items_marks_empty_chunks()
    test_missing_data_and_error_isolation()
    test_concurrent_coordinate_calls_do_not_cross_contaminate()
    test_boost_is_a_simulated_metric_not_measured_acceleration()
    print("[PASS] Quantum VM deep verification complete")
