"""LeRobot adapter for an SO101 follower arm.

This module uses the same SO101 follower driver family as ``lerobot-record``.
Its action order matches the StarVLA SO101 training configuration.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from safety import ACTION_NAMES


class SO101RobotAdapter:
    """Read six SO101 joint positions and send absolute joint-position targets."""

    def __init__(self, port: str, robot_id: str) -> None:
        self.port = port
        self.robot_id = robot_id
        self.robot: Any | None = None

    def connect(self) -> None:
        """Create and connect the LeRobot SO101 follower."""
        try:
            from lerobot.robots.so101_follower import (
                SO101Follower,
                SO101FollowerConfig,
            )
        except ImportError as error:
            raise RuntimeError(
                "LeRobot SO101 support is not installed on this control computer. "
                "Install the same LeRobot version used for lerobot-record."
            ) from error

        config = SO101FollowerConfig(port=self.port, id=self.robot_id)
        self.robot = SO101Follower(config)
        self.robot.connect()

    def read_state(self) -> np.ndarray:
        """Return six joint positions in the StarVLA SO101 action order."""
        self._require_connection()
        observation = self.robot.get_observation()

        if "observation.state" in observation:
            state = np.asarray(observation["observation.state"], dtype=np.float32)
            if state.shape == (len(ACTION_NAMES),):
                return state

        try:
            return np.asarray(
                [observation[f"observation.{name}"] for name in ACTION_NAMES],
                dtype=np.float32,
            )
        except KeyError as error:
            available = ", ".join(sorted(observation.keys()))
            raise RuntimeError(
                "Unable to map LeRobot observation keys to the SO101 state. "
                f"Available keys: {available}"
            ) from error

    def send_joint_target(self, target: np.ndarray) -> None:
        """Send one absolute six-joint target using LeRobot's follower driver."""
        self._require_connection()
        target = np.asarray(target, dtype=np.float32)
        if target.shape != (len(ACTION_NAMES),):
            raise ValueError(
                f"Expected shape ({len(ACTION_NAMES)},), got {target.shape}"
            )

        action = {
            name: float(value)
            for name, value in zip(ACTION_NAMES, target.tolist())
        }
        self.robot.send_action(action)

    def disconnect(self) -> None:
        """Close the follower connection after a test."""
        if self.robot is not None:
            self.robot.disconnect()
            self.robot = None

    def _require_connection(self) -> None:
        if self.robot is None:
            raise RuntimeError("SO101 follower is not connected")
