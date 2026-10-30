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
        "MOTION ENABLED: only one filtered action is sent per policy request."
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
            safe_target = make_safe_target(
                raw_target, current, lower, upper, max_delta
            )
            print(
                f"step={control_step} current={current} "
                f"raw={raw_target} safe={safe_target}"
            )
            if args.enable_motion:
                sent_target = robot.send_joint_target(safe_target)
                print(f"step={control_step} sent={sent_target}")

            elapsed = time.monotonic() - started_at
            time.sleep(max(0.0, args.control_period_s - elapsed))
    finally:
        front_camera.release()
        side_camera.release()
        client.close()
        robot.disconnect()


if __name__ == "__main__":
    main()
