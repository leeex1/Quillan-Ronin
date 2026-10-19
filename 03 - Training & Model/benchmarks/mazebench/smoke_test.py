"""Dependency-free smoke tests for the MazeBench adapter contract."""

from pathlib import Path
import tempfile
import unittest

from adapter import (
    MazeBenchConfig,
    MazeBenchInputError,
    TrajectoryRecorder,
    normalize_observation,
    run_policy,
    validate_action,
)


class MazeBenchAdapterTests(unittest.TestCase):
    def test_observation_normalization_preserves_objects(self) -> None:
        observation = {"room": "level_CxD", "objects": {"player": [[1, 2, 0]]}}
        self.assertEqual(normalize_observation(observation, max_bytes=1024), observation)

    def test_rejects_unknown_action(self) -> None:
        with self.assertRaises(MazeBenchInputError):
            validate_action("launch_arbitrary_command")

    def test_rejects_oversized_observation(self) -> None:
        with self.assertRaises(MazeBenchInputError):
            normalize_observation({"objects": {"wall": ["x" * 2048]}}, max_bytes=32)

    def test_run_is_bounded_and_redacts_object_payloads(self) -> None:
        config = MazeBenchConfig(max_moves=2, max_seconds=10)
        observations = [
            {"room": "A", "camera": {"yaw": 0}, "objects": {"secret": [[0, 0, 0]]}},
            {"room": "B", "camera": {"yaw": 90}, "objects": {"secret": [[0, 0, 0]]}},
            {"room": "C", "camera": {"yaw": 180}, "objects": {"secret": [[0, 0, 0]]}},
        ]
        recorder = run_policy(observations, lambda _: "camera_right", config)
        self.assertEqual(len(recorder.records), 2)
        self.assertNotIn("secret", recorder.records[0])

        with tempfile.TemporaryDirectory() as temp_dir:
            output = TrajectoryRecorder(
                MazeBenchConfig(artifacts_dir=Path(temp_dir))
            )
            output.append(0, "move_forward", {"room": "A", "camera": {}})
            path = output.write()
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
