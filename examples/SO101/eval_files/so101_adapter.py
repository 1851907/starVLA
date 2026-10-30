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

    def __init__(self, port: str, robot_id: str, *, use_degrees: bool) -> None:
        self.port = port
        self.robot_id = robot_id
        self.use_degrees = use_degrees
        self.robot: Any | None = None

    def connect(self) -> None:
        """Create and connect the LeRobot SO101 follower."""
        try:
            from lerobot.robots.so_follower.so_follower import SOFollower
            from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
        except ImportError as error:
            raise RuntimeError(
                "Cannot import the LeRobot 0.6.1 so_follower driver. "
                "Run this script in the Mac's lerobot environment."
            ) from error

        config = SOFollowerRobotConfig(
            port=self.port, id=self.robot_id, use_degrees=self.use_degrees
        )
        self.robot = SOFollower(config)
        # Driver connection may calibrate and always calls configure(), including
        # in the control script's dry-run mode.
        self.robot.connect()

    def read_state(self) -> np.ndarray:
        """Return six joint positions in the StarVLA SO101 action order."""
        self._require_connection()
        observation = self.robot.get_observation()

        try:
            state = np.asarray(
                [observation[name] for name in ACTION_NAMES],
                dtype=np.float32,
            )
        except KeyError as error:
            available = ", ".join(sorted(observation.keys()))
            raise RuntimeError(
                "Unable to map LeRobot observation keys to the SO101 state. "
                f"Available keys: {available}"
            ) from error
        if not np.all(np.isfinite(state)):
            raise ValueError("SO101 joint state contains non-finite values")
        return state

    def send_joint_target(self, target: np.ndarray) -> np.ndarray:
        """Send an absolute target and return the target actually sent."""
        self._require_connection()
        target = np.asarray(target, dtype=np.float32)
        if target.shape != (len(ACTION_NAMES),):
            raise ValueError(
                f"Expected shape ({len(ACTION_NAMES)},), got {target.shape}"
            )
        if not np.all(np.isfinite(target)):
            raise ValueError("SO101 target contains non-finite values")

        action = {
            name: float(value)
            for name, value in zip(ACTION_NAMES, target.tolist())
        }
        sent = self.robot.send_action(action)
        return np.asarray([sent[name] for name in ACTION_NAMES], dtype=np.float32)

    def disconnect(self) -> None:
        """Close the follower connection after a test."""
        if self.robot is not None:
            self.robot.disconnect()
            self.robot = None

    def _require_connection(self) -> None:
        if self.robot is None:
            raise RuntimeError("SO101 follower is not connected")
