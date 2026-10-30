#!/usr/bin/env python3
"""SO101 receding-horizon deployment client.

The client is dry-run by default. Chunk mode executes consecutive predicted
targets with interpolation; inference remains synchronous between chunks.
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


def execute_chunk(robot, targets, action_fps, servo_hz, transition_s):
    """Interpolate consecutive absolute targets without skipping late commands.

    Late hardware writes stretch execution instead of causing catch-up bursts.
    This controls command timing, not physical motor velocity or acceleration.
    """
    previous = robot.read_state()
    if not np.all(np.isfinite(previous)) or not np.all(np.isfinite(targets)):
        raise UnsafeActionError("Non-finite chunk or joint state")
    period = 1.0 / servo_hz
    for index, target in enumerate(targets):
        duration = transition_s if index == 0 else 1.0 / action_fps
        ticks = max(1, int(np.ceil(duration * servo_hz)))
        origin = previous.copy()
        for tick in range(1, ticks + 1):
            started = time.monotonic()
            alpha = tick / ticks
            # Smooth the connection from measured pose to the new chunk.
            if index == 0:
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            previous = robot.send_joint_target(origin + alpha * (target - origin))
            time.sleep(max(0.0, period - (time.monotonic() - started)))
    return previous, robot.read_state()


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
    parser.add_argument("--execute-chunk", action="store_true",
                        help="Execute consecutive targets with interpolation; requires direct-action.")
    parser.add_argument("--chunk-actions", type=int, default=0,
                        help="Number of targets per prediction; 0 executes the entire returned chunk.")
    parser.add_argument("--action-fps", type=float, default=30.0,
                        help="Action sample rate from training dataset metadata.")
    parser.add_argument("--chunk-transition-s", type=float, default=1.0,
                        help="Time to interpolate from measured pose to each chunk's first target.")
    parser.add_argument(
        "--direct-action", action="store_true",
        help="Send the first model target directly, bypassing script position, "
             "step-distance and slew limits. Requires --enable-motion to send.",
    )
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
    )
    parser.add_argument(
        "--joint-upper",
        type=lambda value: parse_vector(value, "joint-upper"),
    )
    parser.add_argument(
        "--max-step-delta",
        type=lambda value: parse_vector(value, "max-step-delta"),
    )
    args = parser.parse_args()

    if args.execute_chunk and not args.direct_action:
        parser.error("--execute-chunk currently requires --direct-action")
    if args.chunk_actions < 0:
        parser.error("chunk-actions must be nonnegative")
    if args.execute_chunk and args.servo_hz < args.action_fps:
        parser.error("servo-hz must be at least action-fps in chunk mode")
    if args.direct_action:
        if any(value is not None for value in (
            args.joint_lower, args.joint_upper, args.max_step_delta, args.max_joint_speed
        )):
            parser.error("--direct-action cannot be combined with joint limits or speed limits")
        lower = upper = max_delta = None
    else:
        if any(value is None for value in (
            args.joint_lower, args.joint_upper, args.max_step_delta
        )):
            parser.error("Provide joint-lower, joint-upper and max-step-delta, or --direct-action")
        lower, upper, max_delta = validate_limits(
            args.joint_lower, args.joint_upper, args.max_step_delta
        )
    for name in ("servo_hz", "action_duration_s", "control_period_s",
                 "action_fps", "chunk_transition_s"):
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
    if args.direct_action:
        print("DIRECT ACTION: script position, step-distance and slew limits are disabled.")
    print(
        "MOTION ENABLED: executing predicted targets in the selected mode."
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
            inference_started = time.monotonic()
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
            inference_s = time.monotonic() - inference_started
            actions = np.asarray(response["data"]["actions"], dtype=np.float32)
            if (actions.ndim != 3 or actions.shape[0] != 1
                    or actions.shape[1] == 0 or actions.shape[2] != len(ACTION_NAMES)):
                raise UnsafeActionError(f"Unexpected action shape: {actions.shape}")
            if not np.all(np.isfinite(actions)):
                raise UnsafeActionError("Model action chunk contains non-finite values")
            if args.execute_chunk:
                count = args.chunk_actions or actions.shape[1]
                if count > actions.shape[1]:
                    raise UnsafeActionError("chunk-actions exceeds model output horizon")
                targets = actions[0, :count]
                print(f"step={control_step} inference_s={inference_s:.3f} "
                      f"chunk_actions={count} first={targets[0]} last={targets[-1]}")
                if args.enable_motion:
                    execution_started = time.monotonic()
                    sent, measured = execute_chunk(
                        robot, targets, args.action_fps, args.servo_hz,
                        args.chunk_transition_s,
                    )
                    print(f"step={control_step} execution_s="
                          f"{time.monotonic() - execution_started:.3f} "
                          f"sent={sent} measured={measured}")
                elapsed = time.monotonic() - started_at
                time.sleep(max(0.0, args.control_period_s - elapsed))
                continue

            current = robot.read_state()
            raw_target = actions[0, 0]
            if not np.all(np.isfinite(raw_target)):
                raise UnsafeActionError("Model target contains non-finite values")
            if not np.all(np.isfinite(current)):
                raise UnsafeActionError("Joint state contains non-finite values")
            if not args.direct_action and (np.any(current < lower) or np.any(current > upper)):
                raise UnsafeActionError(f"Joint state outside configured bounds: {current}")
            safe_target = raw_target.copy() if args.direct_action else make_safe_target(
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
