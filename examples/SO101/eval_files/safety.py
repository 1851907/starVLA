"""Safety checks for SO101 joint-position commands.

Fill the limits from the calibrated SO101 follower configuration before using
``--enable-motion``. The defaults intentionally reject all motion.
"""

from __future__ import annotations

import numpy as np


ACTION_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)


class UnsafeActionError(ValueError):
    """Raised when a policy action cannot be safely sent to the robot."""


def _as_action(value: np.ndarray, name: str) -> np.ndarray:
    action = np.asarray(value, dtype=np.float32)
    if action.shape != (len(ACTION_NAMES),):
        raise UnsafeActionError(
            f"{name} must have shape ({len(ACTION_NAMES)},), got {action.shape}"
        )
    if not np.isfinite(action).all():
        raise UnsafeActionError(f"{name} contains NaN or Inf")
    return action


def validate_limits(
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    max_step_delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate a calibrated set of six-joint safety limits."""
    lower = _as_action(joint_lower, "joint_lower")
    upper = _as_action(joint_upper, "joint_upper")
    max_delta = _as_action(max_step_delta, "max_step_delta")

    if np.any(lower >= upper):
        raise UnsafeActionError("Each lower joint limit must be below its upper limit")
    if np.any(max_delta <= 0):
        raise UnsafeActionError("Each max_step_delta value must be positive")
    return lower, upper, max_delta


def make_safe_target(
    predicted_target: np.ndarray,
    current_state: np.ndarray,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    max_step_delta: np.ndarray,
) -> np.ndarray:
    """Clip an absolute six-joint target to limits and a maximum single move."""
    predicted = _as_action(predicted_target, "predicted_target")
    current = _as_action(current_state, "current_state")
    lower, upper, max_delta = validate_limits(
        joint_lower, joint_upper, max_step_delta
    )

    bounded_target = np.clip(predicted, lower, upper)
    bounded_delta = np.clip(
        bounded_target - current,
        -max_delta,
        max_delta,
    )
    return current + bounded_delta
