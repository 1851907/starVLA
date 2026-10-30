#!/usr/bin/env python3
"""SO101 receding-horizon deployment client.

The client is dry-run by default. Passing --enable-motion permits a single
filtered action per inference request. Calibrate all safety limits first.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from safety import ACTION_NAMES, UnsafeActionError, make_safe_target, validate_limits
from so101_adapter import SO101RobotAdapter


def parse_vector(text: str, name: str) -> np.ndarray:
    values = np.fromstring(text, sep=",", dtype=np.float32)
    if values.shape != (len(ACTION_NAMES),):
        raise argparse.ArgumentTypeError(
            f"{name} requires {len(ACTION_NAMES)} comma-separated values"
        )
    return values


def read_rgb(camera: cv2.VideoCapture, name: str) -> np.ndarray:
    ok, frame_bgr = camera.read()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Unable to read a frame from {name} camera")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def execute_target(robot, target, lower, upper, max_delta, speed, hz, duration):
    """Slew commanded positions; this is not a motor velocity controller."""
    period = 1.0 / hz
    previous = robot.read_state()
    deadline = time.monotonic() + duration
    sent = previous.copy()
    while time.monotonic() < deadline:
        tick = time.monotonic()
        current = robot.read_state()
        if not np.all(np.isfinite(current)) or np.any(current < lower) or np.any(current > upper):
            raise UnsafeActionError(f"Joint state outside configured bounds: {current}")
        # Bound both command slew and distance from measured position. Never
        # compensate for a delayed tick with a larger jump.
        lo = np.maximum(np.maximum(lower, previous - speed * period), current - max_delta)
        hi = np.minimum(np.minimum(upper, previous + speed * period), current + max_delta)
        if np.any(lo > hi):
            raise UnsafeActionError("Tracking error exceeds configured command limits")
        sent = robot.send_joint_target(np.clip(target, lo, hi))
        previous = sent.copy()
        time.sleep(max(0.0, period - (time.monotonic() - tick)))
    return sent, robot.read_state()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-host", required=True)
    parser.add_argument("--server-port", type=int, default=10093)
    parser.add_argument("--front-camera-index", type=int, default=0)
    parser.add_argument("--side-camera-index", type=int, default=1)
    parser.add_argument("--task", required=True)
    parser.add_argument("--unnorm-key", default="new_embodiment")
    parser.add_argument("--robot-port", required=True)
    parser.add_argument("--robot-id", default="so101_follower")
    parser.add_argument(
        "--joint-units", choices=("degrees", "normalized"), required=True,
        help="Match collection: degrees or normalized (-100..100) body joints; "
             "gripper is always 0..100.",
    )
    parser.add_argument("--control-period-s", type=float, default=0.25)
    parser.add_argument("--max-control-steps", type=int, default=100)
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument(
        "--max-joint-speed", type=lambda value: parse_vector(value, "max-joint-speed"),
        help="Enable repeated execution: six command slew limits in joint units/second.",
    )
    parser.add_argument("--servo-hz", type=float, default=20.0)
    parser.add_argument("--action-duration-s", type=float, default=1.0,
                        help="Execute each predicted first target for this long, then infer again.")
    parser.add_argument(
        "--joint-lower",
        type=lambda value: parse_vector(value, "joint-lower"),
        required=True,
    )
    parser.add_argument(
        "--joint-upper",
        type=lambda value: parse_vector(value, "joint-upper"),
        required=True,
    )
    parser.add_argument(
        "--max-step-delta",
        type=lambda value: parse_vector(value, "max-step-delta"),
        required=True,
    )
    args = parser.parse_args()

    lower, upper, max_delta = validate_limits(
        args.joint_lower, args.joint_upper, args.max_step_delta
    )
    for name in ("servo_hz", "action_duration_s", "control_period_s"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be finite and positive")
    if args.max_control_steps <= 0:
        parser.error("max-control-steps must be positive")
    if args.max_joint_speed is not None:
        if not np.all(np.isfinite(args.max_joint_speed)) or np.any(args.max_joint_speed <= 0):
            parser.error("max-joint-speed must contain six finite positive values")
    front_camera = cv2.VideoCapture(args.front_camera_index)
    side_camera = cv2.VideoCapture(args.side_camera_index)
    if not front_camera.isOpened() or not side_camera.isOpened():
        raise RuntimeError("Unable to open both front and side cameras")

    client = WebsocketClientPolicy(args.server_host, args.server_port)
    robot = SO101RobotAdapter(
        args.robot_port, args.robot_id, use_degrees=args.joint_units == "degrees"
    )
    print("Policy server metadata:", client.get_server_metadata())
    print(
        "MOTION ENABLED: executing the first predicted target with configured limits."
        if args.enable_motion
        else "DRY RUN: no policy target will be sent. Driver connection still "
             "configures motors and may calibrate or enable torque."
    )

    try:
        robot.connect()
        for control_step in range(args.max_control_steps):
            started_at = time.monotonic()
            front_rgb = read_rgb(front_camera, "front")
            side_rgb = read_rgb(side_camera, "side")
            response = client.predict_action(
                {
                    "examples": [{
                        "image": [front_rgb, side_rgb],
                        "lang": args.task,
                    }],
                    "unnorm_key": args.unnorm_key,
                    "use_ddim": True,
                    "num_ddim_steps": 20,
                }
            )
            actions = np.asarray(response["data"]["actions"], dtype=np.float32)
            if (actions.ndim != 3 or actions.shape[0] != 1
                    or actions.shape[1] == 0 or actions.shape[2] != len(ACTION_NAMES)):
                raise UnsafeActionError(f"Unexpected action shape: {actions.shape}")

            current = robot.read_state()
            raw_target = actions[0, 0]
            if np.any(current < lower) or np.any(current > upper):
                raise UnsafeActionError(f"Joint state outside configured bounds: {current}")
            safe_target = make_safe_target(
                raw_target, current, lower, upper, max_delta
            )
            if args.max_joint_speed is not None:
                safe_target = np.clip(raw_target, lower, upper)
            print(
                f"step={control_step} current={current} "
                f"raw={raw_target} safe={safe_target}"
            )
            if args.enable_motion:
                if args.max_joint_speed is None:
                    sent_target = robot.send_joint_target(safe_target)
                    print(f"step={control_step} sent={sent_target}")
                else:
                    sent_target, measured = execute_target(
                        robot, safe_target, lower, upper, max_delta,
                        args.max_joint_speed, args.servo_hz, args.action_duration_s,
                    )
                    print(f"step={control_step} sent={sent_target} measured={measured}")

            elapsed = time.monotonic() - started_at
            time.sleep(max(0.0, args.control_period_s - elapsed))
    finally:
        front_camera.release()
        side_camera.release()
        client.close()
        robot.disconnect()


if __name__ == "__main__":
    main()
