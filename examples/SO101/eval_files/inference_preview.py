#!/usr/bin/env python3
"""Preview SO101 policy outputs from live cameras without moving the robot."""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from safety import ACTION_NAMES


def read_rgb(camera: cv2.VideoCapture, name: str) -> np.ndarray:
    ok, frame_bgr = camera.read()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Unable to read a frame from {name} camera")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def format_action(action: np.ndarray) -> str:
    return ", ".join(
        f"{name}={value:.3f}"
        for name, value in zip(ACTION_NAMES, action.tolist())
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SO101 policy inference without controlling the robot."
    )
    parser.add_argument("--server-host", required=True)
    parser.add_argument("--server-port", type=int, default=10093)
    parser.add_argument("--front-camera-index", type=int, default=0)
    parser.add_argument("--side-camera-index", type=int, default=1)
    parser.add_argument("--task", required=True)
    parser.add_argument("--unnorm-key", default="new_embodiment")
    parser.add_argument("--interval-s", type=float, default=1.0)
    args = parser.parse_args()

    front_camera = cv2.VideoCapture(args.front_camera_index)
    side_camera = cv2.VideoCapture(args.side_camera_index)
    if not front_camera.isOpened() or not side_camera.isOpened():
        raise RuntimeError("Unable to open both front and side cameras")

    client = WebsocketClientPolicy(args.server_host, args.server_port)
    print("Policy server metadata:", client.get_server_metadata())

    try:
        while True:
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
            if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[2] != 6:
                raise RuntimeError(f"Unexpected action shape: {actions.shape}")
            if not np.isfinite(actions).all():
                raise RuntimeError("Model returned NaN or Inf actions")

            print(f"\nReceived action chunk with shape {actions.shape}")
            print("First action:", format_action(actions[0, 0]))
            print("Per-joint min:", actions[0].min(axis=0))
            print("Per-joint max:", actions[0].max(axis=0))
            time.sleep(args.interval_s)
    finally:
        front_camera.release()
        side_camera.release()
        client.close()


if __name__ == "__main__":
    main()
