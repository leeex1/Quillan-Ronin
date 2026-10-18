"""Safe, transport-neutral adapter primitives for MazeBench runs."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping


ALLOWED_ACTIONS = frozenset(
    {
        "move_forward",
        "move_backward",
        "move_left",
        "move_right",
        "camera_up",
        "camera_down",
        "camera_left",
        "camera_right",
        "undo",
        "reset_level",
        "teleport",
    }
)


class MazeBenchInputError(ValueError):
    """Raised when an observation or action violates the adapter contract."""


@dataclass(frozen=True)
class MazeBenchConfig:
    """Bounded execution settings for one local trajectory."""

    max_moves: int = 200
    max_seconds: float = 1800.0
    max_observation_bytes: int = 1_048_576
    artifacts_dir: Path = Path("benchmarks/mazebench/artifacts")

    def __post_init__(self) -> None:
        if self.max_moves <= 0:
            raise ValueError("max_moves must be positive")
        if self.max_seconds <= 0:
            raise ValueError("max_seconds must be positive")
        if self.max_observation_bytes <= 0:
            raise ValueError("max_observation_bytes must be positive")


def normalize_observation(
    observation: Mapping[str, Any],
    *,
    max_bytes: int,
) -> dict[str, Any]:
    """Validate and copy a MazeBench JSON observation with a size bound."""

    if not isinstance(observation, Mapping):
        raise MazeBenchInputError("observation must be a mapping")

    if isinstance(observation.get("json_observation"), Mapping):
        observation = observation["json_observation"]

    encoded = json.dumps(observation, ensure_ascii=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > max_bytes:
        raise MazeBenchInputError("observation exceeds configured size limit")

    normalized = dict(observation)
    normalized.setdefault("objects", {})
    if not isinstance(normalized["objects"], Mapping):
        raise MazeBenchInputError("observation.objects must be a mapping")
    return normalized


def validate_action(action: Any) -> str:
    """Return a validated action name suitable for a MazeBench transport."""

    if not isinstance(action, str) or action not in ALLOWED_ACTIONS:
        raise MazeBenchInputError(f"unsupported MazeBench action: {action!r}")
    return action


@dataclass
class TrajectoryRecorder:
    """Write bounded, non-sensitive trajectory metadata as JSON Lines."""

    config: MazeBenchConfig
    records: list[dict[str, Any]] = field(default_factory=list)

    def append(self, move: int, action: str, observation: Mapping[str, Any]) -> None:
        if move < 0 or move >= self.config.max_moves:
            raise MazeBenchInputError("move is outside configured trajectory bounds")
        self.records.append(
            {
                "move": move,
                "action": validate_action(action),
                "room": observation.get("room"),
                "camera": observation.get("camera"),
            }
        )

    def write(self, filename: str = "trajectory.jsonl") -> Path:
        """Persist the redacted trajectory under the configured artifact root."""

        if Path(filename).name != filename or not filename.endswith(".jsonl"):
            raise ValueError("filename must be a simple .jsonl name")
        root = self.config.artifacts_dir.resolve()
        root.mkdir(parents=True, exist_ok=True)
        output = (root / filename).resolve()
        if output.parent != root:
            raise ValueError("artifact path escaped configured directory")
        with output.open("w", encoding="utf-8", newline="\n") as handle:
            for record in self.records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return output


def run_policy(
    observations: list[Mapping[str, Any]],
    policy: Callable[[Mapping[str, Any]], str],
    config: MazeBenchConfig | None = None,
) -> TrajectoryRecorder:
    """Run a policy over supplied observations with move and time limits."""

    run_config = config or MazeBenchConfig()
    recorder = TrajectoryRecorder(run_config)
    started = time.monotonic()
    for move, raw_observation in enumerate(observations[: run_config.max_moves]):
        if time.monotonic() - started >= run_config.max_seconds:
            break
        observation = normalize_observation(
            raw_observation,
            max_bytes=run_config.max_observation_bytes,
        )
        recorder.append(move, validate_action(policy(observation)), observation)
    return recorder
