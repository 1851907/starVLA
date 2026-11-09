#!/usr/bin/env python3
"""Synchronous EE policy client. Dry-run still connects/configures the motors."""

import argparse
import time
import threading
import json
from pathlib import Path

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


class FreshCamera:
    """Drain camera buffers continuously; timestamp each read's start/end."""

    def __init__(self, index):
        self.index = index
        self.camera = cv2.VideoCapture(index)
        if not self.camera.isOpened():
            self.camera.release()
            raise RuntimeError(f"Cannot open camera {index}")
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.latest = None
        self.error = None
        self.thread = threading.Thread(target=self._capture, daemon=True)
        self.thread.start()

    def _capture(self):
        failed_since = None
        try:
            while not self.stop.is_set():
                started = time.monotonic()
                ok, frame = self.camera.read()
                if not ok or frame is None or frame.size == 0:
                    now = time.monotonic()
                    if failed_since is None:
                        failed_since = now
                    if now - failed_since >= 2.0:
                        raise RuntimeError(
                            f"Camera {self.index}: frame reads failed continuously "
                            f"for {now - failed_since:.2f}s"
                        )
                    # Do not publish a frame or timestamp on a failed read.
                    self.stop.wait(0.05)
                    continue
                failed_since = None
                with self.condition:
                    self.latest = (started, time.monotonic(), frame)
                    self.condition.notify_all()
        except Exception as error:
            with self.condition:
                self.error = error
                self.condition.notify_all()
        finally:
            self.camera.release()

    def after(self, barrier, timeout):
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.latest is None or self.latest[0] <= barrier:
                if self.error is not None:
                    raise RuntimeError(f"Camera {self.index} capture stopped: {self.error}") from self.error
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Camera {self.index}: timed out waiting for a post-motion frame"
                    )
                self.condition.wait(remaining)
            if self.error is not None:
                raise RuntimeError(f"Camera {self.index} capture stopped: {self.error}") from self.error
            _, timestamp, frame = self.latest
            frame = frame.copy()
        return cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB), timestamp

    def close(self):
        self.stop.set()
        # The reader owns VideoCapture.release; avoid racing a blocked read.
        self.thread.join(timeout=2)


def move(robot, target, hz, arm_speed, gripper_speed,
         gripper_contact_motion=0.15, gripper_contact_time=0.35):
    start = robot.read_state().astype(float)
    duration = max(np.max(np.abs(target[:5] - start[:5])) / arm_speed,
                   abs(target[5] - start[5]) / gripper_speed, 1.0 / hz)
    # Smoothstep peaks at 1.5 times its average slope.
    ticks = max(1, int(np.ceil(1.5 * duration * hz)))
    closing = target[5] < start[5] - gripper_contact_motion
    gripper_stalled_since = None
    previous = start
    for tick in range(1, ticks + 1):
        began = time.monotonic()
        alpha = tick / ticks
        alpha = alpha * alpha * (3 - 2 * alpha)
        robot.send_joint_target(start + alpha * (target - start))
        measured = robot.read_state().astype(float)
        if closing:
            gripper_motion = abs(measured[5] - previous[5])
            remaining = abs(target[5] - measured[5])
            if (gripper_motion <= gripper_contact_motion
                    and remaining > gripper_contact_motion):
                if gripper_stalled_since is None:
                    gripper_stalled_since = time.monotonic()
                elif time.monotonic() - gripper_stalled_since >= gripper_contact_time:
                    return measured, True
            else:
                gripper_stalled_since = None
        previous = measured
        time.sleep(max(0.0, 1.0 / hz - (time.monotonic() - began)))
    return robot.read_state().astype(float), False


def wait_until_settled(robot, target, args):
    """Require position tolerance AND low measured motion for a continuous window."""
    began = time.monotonic()
    stable_since = None
    previous = robot.read_state().astype(float)
    while True:
        time.sleep(1.0 / args.settle_poll_hz)
        measured = robot.read_state().astype(float)
        now = time.monotonic()
        if measured.shape != (6,) or not np.isfinite(measured).all():
            raise RuntimeError("Invalid measured joint state")
        arm_error = float(np.max(np.abs(measured[:5] - target[:5])))
        grip_error = float(abs(measured[5] - target[5]))
        stationary = (
            np.max(np.abs(measured[:5] - previous[:5])) <= args.settle_motion_deg
            and abs(measured[5] - previous[5]) <= args.settle_gripper_motion
        )
        if (arm_error <= args.arrival_tolerance_deg
                and grip_error <= args.gripper_arrival_tolerance and stationary):
            if stable_since is None:
                stable_since = now
            if now - stable_since >= args.settle_time_s:
                return measured, now
        else:
            stable_since = None
        if now - began >= args.arrival_timeout_s:
            raise TimeoutError(f"Arrival timed out: arm_error={arm_error:.3f}deg "
                               f"gripper_error={grip_error:.3f}; no new inference requested")
        previous = measured


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
    parser.add_argument(
        "--replan-fraction",
        type=float,
        default=1.0,
        help="Execute this fraction of the predicted EE translation/rotation before replanning.",
    )
    parser.add_argument("--arrival-tolerance-deg", type=positive, default=2)
    parser.add_argument("--gripper-arrival-tolerance", type=positive, default=3)
    parser.add_argument("--gripper-contact-motion", type=positive, default=0.15,
                        help="Max gripper movement per control tick considered stalled.")
    parser.add_argument("--gripper-contact-time-s", type=positive, default=0.35,
                        help="Continuous stalled time before treating gripper as contacted.")
    parser.add_argument("--gripper-min-fraction", type=float, default=0.08,
                        help="Do not command a gripper fraction below this safety floor.")
    parser.add_argument("--settle-time-s", type=positive, default=0.4)
    parser.add_argument("--arrival-timeout-s", type=positive, default=10)
    parser.add_argument("--settle-poll-hz", type=positive, default=20)
    parser.add_argument("--settle-motion-deg", type=positive, default=0.3)
    parser.add_argument("--settle-gripper-motion", type=positive, default=0.5)
    parser.add_argument("--camera-settle-s", type=positive, default=0.2,
                        help="Drain camera buffers this long after measured arrival.")
    parser.add_argument("--camera-timeout-s", type=positive, default=5)
    parser.add_argument("--diagnostics-dir", type=Path,
                        help="New directory for exact request images and per-step JSON.")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("steps must be positive")
    if args.arrival_timeout_s <= args.settle_time_s:
        parser.error("arrival-timeout-s must exceed settle-time-s")
    if not 0 < args.replan_fraction <= 1:
        parser.error("replan-fraction must be in (0, 1]")
    if not 0 <= args.gripper_min_fraction < 1:
        parser.error("gripper-min-fraction must be in [0, 1)")
    kin = EEKinematics(args.urdf, args.conversion)
    if args.diagnostics_dir is not None:
        # Exclusive creation prevents overwriting a previous diagnostic run.
        args.diagnostics_dir.mkdir(parents=True, exist_ok=False)
    robot = SO101RobotAdapter(args.robot_port, args.robot_id, use_degrees=True)
    front = side = client = None
    try:
        front = FreshCamera(args.front_camera_index)
        side = FreshCamera(args.side_camera_index)
        client = WebsocketClientPolicy(args.server_host, args.server_port)
        metadata = client.get_server_metadata()
        print("Server:", metadata)
        if args.diagnostics_dir is not None:
            (args.diagnostics_dir / "run.json").write_text(json.dumps({
                "arguments": {key: str(value) if isinstance(value, Path) else value
                              for key, value in vars(args).items()},
                "server_metadata": metadata,
                "conversion": kin.config,
                "image_note": "Exact 224x224 RGB arrays submitted in examples, saved as lossless PNG. "
                              "Camera timestamps are host read-completion times, not exposure times.",
            }, indent=2) + "\n")
        keys = metadata.get("action_keys")
        expected = ["action." + name for name in ("x", "y", "z", "rx", "ry", "rz", "gripper")]
        if keys != expected:
            raise ValueError(f"Expected converted EE model action_keys={expected}, got {keys}")
        state_keys = metadata.get("state_keys")
        expected_state = [
            "state.x", "state.y", "state.z",
            "state.rx", "state.ry", "state.rz", "state.gripper",
        ]
        if state_keys != expected_state:
            raise ValueError(
                f"Expected future6 EE model state_keys={expected_state}, got {state_keys}"
            )
        if args.unnorm_key not in metadata.get("available_unnorm_keys", []):
            raise ValueError("Selected unnorm-key absent from server metadata")
        print("MOTION ENABLED" if args.enable_motion else
              "DRY RUN: no policy targets sent; driver connection may enable torque/calibrate")
        robot.connect()
        # First observation also waits for a stable measured pose.
        _, arrived_at = wait_until_settled(robot, robot.read_state(), args)
        for step in range(args.steps):
            barrier = arrived_at + args.camera_settle_s
            front_rgb, front_time = front.after(barrier, args.camera_timeout_s)
            side_rgb, side_time = side.after(barrier, args.camera_timeout_s)
            observed_at = min(front_time, side_time)
            observed = robot.read_state()
            state = kin.state_from_joint_degrees(observed)
            record = {
                "step": step, "observed_joints_deg_gripper": observed.tolist(),
                "model_state": state.tolist(),
                "front_read_monotonic_s": front_time,
                "side_read_monotonic_s": side_time,
                "image_pair_skew_s": abs(front_time - side_time),
                "task": args.task, "status": "request_pending",
            }
            record_path = None
            if args.diagnostics_dir is not None:
                record_path = args.diagnostics_dir / f"step_{step:03d}.json"
                for name, rgb in (("front", front_rgb), ("side", side_rgb)):
                    filename = f"step_{step:03d}_{name}.png"
                    if not cv2.imwrite(str(args.diagnostics_dir / filename),
                                       cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                        raise RuntimeError(f"Failed to write diagnostic image {filename}")
                    record[name + "_image"] = filename
                record_path.write_text(json.dumps(record, indent=2) + "\n")
            response = client.predict_action({
                "examples": [{
                    "image": [front_rgb, side_rgb],
                    "lang": args.task,
                    "state": state[None, :],
                }],
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
            executed_action = action.copy()
            # EE position/rotation are deltas; gripper is an absolute target
            # fraction and must not be scaled.
            executed_action[:6] *= args.replan_fraction
            observed_pose = kin.fk(kin.to_radians(observed[:5]))
            target_pose = kin.target(observed[:5], executed_action)
            print(f"TCP step={step} base_frame={kin.config['base_frame']} "
                  f"current_m={observed_pose.translation} target_m={target_pose.translation} "
                  f"predicted_delta_mm={action[:3] * 1000} "
                  f"executed_delta_mm={executed_action[:3] * 1000}")
            record.update({
                "status": "before_ik", "observation_age_s": age,
                "predicted_actions": actions[0].tolist(),
                "predicted_action": action.tolist(),
                "executed_action": executed_action.tolist(),
                "replan_fraction": args.replan_fraction,
                "base_frame": kin.config["base_frame"],
                "current_tcp_m": observed_pose.translation.tolist(),
                "target_tcp_m": target_pose.translation.tolist(),
                "current_rotation_matrix": observed_pose.rotation.tolist(),
                "target_rotation_matrix": target_pose.rotation.tolist(),
            })
            if record_path is not None:
                record_path.write_text(json.dumps(record, indent=2) + "\n")
            current = robot.read_state()
            if np.max(np.abs(current[:5] - observed[:5])) > args.max_tracking_error_deg:
                raise RuntimeError("Arm changed pose while waiting for inference")
            try:
                joints, pos_error, rot_error = kin.solve(
                    target_pose, current[:5], args.position_tolerance_m,
                    args.rotation_tolerance_deg, args.rotation_weight,
                )
            except ValueError as error:
                record.update(status="ik_rejected", error=str(error))
                if record_path is not None:
                    record_path.write_text(json.dumps(record, indent=2) + "\n")
                raise
            requested_gripper_fraction = float(executed_action[6])
            safe_gripper_fraction = max(requested_gripper_fraction,
                                        args.gripper_min_fraction)
            target = np.r_[joints, kin.gripper(safe_gripper_fraction)]
            record.update(status="ik_passed", target_joints_deg_gripper=target.tolist(),
                          requested_gripper_fraction=requested_gripper_fraction,
                          safe_gripper_fraction=safe_gripper_fraction,
                          ik_position_m=pos_error, ik_rotation_deg=rot_error)
            if record_path is not None:
                record_path.write_text(json.dumps(record, indent=2) + "\n")
            print(f"step={step} inference_s={age:.3f} "
                  f"predicted_action={action} executed_action={executed_action} "
                  f"target={target} ik_position_m={pos_error:.5f} "
                  f"ik_rotation_deg={rot_error:.3f}")
            if args.enable_motion:
                measured, gripper_contact = move(
                    robot, target, args.servo_hz, args.arm_speed_deg_s,
                    args.gripper_speed, args.gripper_contact_motion,
                    args.gripper_contact_time_s,
                )
                if gripper_contact:
                    # Hold the measured contact position; do not keep pushing
                    # toward the blocked target.
                    target = target.copy()
                    target[5] = measured[5]
                    record.update(status="gripper_contact",
                                  contact_gripper=measured[5])
                    if record_path is not None:
                        record_path.write_text(json.dumps(record, indent=2) + "\n")
                measured, arrived_at = wait_until_settled(robot, target, args)
                print(f"ARRIVED step={step} measured={measured} "
                      f"gripper_contact={gripper_contact}; next capture follows arrival")
            else:
                _, arrived_at = wait_until_settled(robot, robot.read_state(), args)
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
                front.close()
            if side is not None:
                side.close()
            if client is not None:
                client.close()


if __name__ == "__main__":
    main()
