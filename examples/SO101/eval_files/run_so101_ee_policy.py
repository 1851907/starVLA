#!/usr/bin/env python3
"""Synchronous EE policy client. Dry-run still connects/configures the motors."""

import argparse
import time

import cv2
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from ee_kinematics import EEKinematics
from so101_adapter import SO101RobotAdapter


def positive(value):
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Must be finite and positive")
    return number


def image(camera):
    ok, frame = camera.read()
    if not ok or frame is None:
        raise RuntimeError("Camera frame acquisition failed")
    return cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB)


def move(robot, target, hz, arm_speed, gripper_speed):
    start = robot.read_state().astype(float)
    duration = max(np.max(np.abs(target[:5] - start[:5])) / arm_speed,
                   abs(target[5] - start[5]) / gripper_speed, 1.0 / hz)
    # Smoothstep peaks at 1.5 times its average slope.
    ticks = max(1, int(np.ceil(1.5 * duration * hz)))
    for tick in range(1, ticks + 1):
        began = time.monotonic()
        alpha = tick / ticks
        alpha = alpha * alpha * (3 - 2 * alpha)
        robot.send_joint_target(start + alpha * (target - start))
        time.sleep(max(0.0, 1.0 / hz - (time.monotonic() - began)))
    return robot.read_state()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-host", required=True)
    parser.add_argument("--server-port", type=int, default=8080)
    parser.add_argument("--unnorm-key", default="new_embodiment")
    parser.add_argument("--task", required=True)
    parser.add_argument("--robot-port", required=True)
    parser.add_argument("--robot-id", default="my_awesome_follower_arm")
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--conversion", required=True)
    parser.add_argument("--front-camera-index", type=int, default=0)
    parser.add_argument("--side-camera-index", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--servo-hz", type=positive, default=30)
    parser.add_argument("--arm-speed-deg-s", type=positive, default=15)
    parser.add_argument("--gripper-speed", type=positive, default=20)
    parser.add_argument("--position-tolerance-m", type=positive, default=0.005)
    parser.add_argument("--rotation-tolerance-deg", type=positive, default=15)
    parser.add_argument("--rotation-weight", type=positive, default=0.05)
    parser.add_argument("--max-observation-age-s", type=positive, default=10)
    parser.add_argument("--max-tracking-error-deg", type=positive, default=10)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("steps must be positive")
    kin = EEKinematics(args.urdf, args.conversion)
    robot = SO101RobotAdapter(args.robot_port, args.robot_id, use_degrees=True)
    front = side = client = None
    try:
        front = cv2.VideoCapture(args.front_camera_index)
        side = cv2.VideoCapture(args.side_camera_index)
        if not front.isOpened() or not side.isOpened():
            raise RuntimeError("Unable to open both cameras")
        client = WebsocketClientPolicy(args.server_host, args.server_port)
        metadata = client.get_server_metadata()
        print("Server:", metadata)
        keys = metadata.get("action_keys")
        expected = ["action." + name for name in ("x", "y", "z", "rx", "ry", "rz", "gripper")]
        if keys != expected:
            raise ValueError(f"Expected converted EE model action_keys={expected}, got {keys}")
        if args.unnorm_key not in metadata.get("available_unnorm_keys", []):
            raise ValueError("Selected unnorm-key absent from server metadata")
        print("MOTION ENABLED" if args.enable_motion else
              "DRY RUN: no policy targets sent; driver connection may enable torque/calibrate")
        robot.connect()
        for step in range(args.steps):
            front_rgb, side_rgb = image(front), image(side)
            observed_at = time.monotonic()
            observed = robot.read_state()
            response = client.predict_action({
                "examples": [{"image": [front_rgb, side_rgb], "lang": args.task}],
                "unnorm_key": args.unnorm_key,
                "use_ddim": True, "num_ddim_steps": 20,
            })
            age = time.monotonic() - observed_at
            if age > args.max_observation_age_s:
                raise RuntimeError(f"Stale observation after inference: {age:.2f}s")
            if not response.get("ok", False):
                raise RuntimeError(f"Policy error: {response}")
            actions = np.asarray(response["data"]["actions"], dtype=float)
            if (actions.ndim != 3 or actions.shape[0] != 1
                    or actions.shape[1] < 1 or actions.shape[2] != 7
                    or not np.isfinite(actions).all()):
                raise ValueError(f"Invalid EE actions: {actions.shape}")
            action = actions[0, 0]
            target_pose = kin.target(observed[:5], action)
            current = robot.read_state()
            if np.max(np.abs(current[:5] - observed[:5])) > args.max_tracking_error_deg:
                raise RuntimeError("Arm changed pose while waiting for inference")
            joints, pos_error, rot_error = kin.solve(
                target_pose, current[:5], args.position_tolerance_m,
                args.rotation_tolerance_deg, args.rotation_weight,
            )
            target = np.r_[joints, kin.gripper(action[6])]
            print(f"step={step} inference_s={age:.3f} action={action} "
                  f"target={target} ik_position_m={pos_error:.5f} "
                  f"ik_rotation_deg={rot_error:.3f}")
            if args.enable_motion:
                measured = move(robot, target, args.servo_hz,
                                args.arm_speed_deg_s, args.gripper_speed)
                print(f"measured={measured}")
                if np.max(np.abs(measured[:5] - target[:5])) > args.max_tracking_error_deg:
                    raise RuntimeError("Arm did not track the commanded target")
    finally:
        # Disconnect before network cleanup; don't leave motors enabled if close fails.
        try:
            if robot.robot is not None:
                try:
                    if robot.robot.is_connected:
                        robot.disconnect()
                finally:
                    # A partially connected driver may not satisfy is_connected.
                    bus = getattr(robot.robot, "bus", None)
                    if bus is not None and bus.is_connected:
                        bus.disconnect(True)
        finally:
            if front is not None:
                front.release()
            if side is not None:
                side.release()
            if client is not None:
                client.close()


if __name__ == "__main__":
    main()
