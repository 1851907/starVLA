"""FK and bounded numerical IK using the dataset conversion's conventions."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


class EEKinematics:
    def __init__(self, urdf, conversion):
        self.config = json.loads(Path(conversion).read_text())
        if self.config.get("status") != "complete":
            raise ValueError("Conversion report is not complete")
        if hashlib.sha256(Path(urdf).read_bytes()).hexdigest() != self.config["urdf_sha256"]:
            raise ValueError("URDF hash differs from the dataset conversion model")
        self.model = pin.buildModelFromUrdf(str(urdf))
        self.data = self.model.createData()
        self.frames = [self.model.getFrameId(self.config[key])
                       for key in ("base_frame", "tool_frame")]
        if any(i >= len(self.model.frames) for i in self.frames):
            raise ValueError("Conversion frame not found")
        self.indices = []
        for name in JOINTS:
            if not self.model.existJointName(name):
                raise ValueError(f"Missing joint {name}")
            joint = self.model.joints[self.model.getJointId(name)]
            if joint.nq != 1:
                raise ValueError(f"Expected scalar joint {name}")
            self.indices.append(joint.idx_q)
        self.signs = np.asarray(self.config["joint_signs"], dtype=float)
        self.offsets = np.asarray(self.config["joint_offsets_deg"], dtype=float)
        if (self.signs.shape != (5,) or self.offsets.shape != (5,)
                or not np.isin(self.signs, [-1, 1]).all()
                or not np.isfinite(self.offsets).all()):
            raise ValueError("Invalid conversion joint mapping")
        self.lower = self.model.lowerPositionLimit[self.indices].copy()
        self.upper = self.model.upperPositionLimit[self.indices].copy()
        if not np.isfinite([self.lower, self.upper]).all() or np.any(self.lower >= self.upper):
            raise ValueError("IK requires finite URDF joint limits")
        self.closed = float(self.config["gripper_closed"])
        self.open = float(self.config["gripper_open"])
        if not np.isfinite([self.closed, self.open]).all() or self.closed == self.open:
            raise ValueError("Invalid gripper endpoints")

    def to_radians(self, degrees):
        values = np.asarray(degrees, dtype=float)
        if values.shape != (5,) or not np.isfinite(values).all():
            raise ValueError("Expected five finite arm joint angles in degrees")
        return np.deg2rad(values * self.signs + self.offsets)

    def fk(self, radians):
        q = pin.neutral(self.model)
        q[self.indices] = radians
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return (self.data.oMf[self.frames[0]].inverse()
                * self.data.oMf[self.frames[1]]).copy()

    def target(self, measured_degrees, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("Expected finite seven-dimensional EE action")
        current = self.fk(self.to_radians(measured_degrees))
        return pin.SE3(pin.exp3(action[3:6]) @ current.rotation,
                       current.translation + action[:3])

    def solve(self, target, measured_degrees, position_tolerance=0.005,
              rotation_tolerance_deg=15.0, rotation_weight=0.05):
        initial = self.to_radians(measured_degrees)
        # Allow only the numerical starting point to be projected into the URDF
        # box. Final residuals and commanded change are checked by the caller.
        seed = np.clip(initial, self.lower, self.upper)

        def residual(q):
            pose = self.fk(q)
            return np.r_[pose.translation - target.translation,
                         rotation_weight * pin.log3(target.rotation @ pose.rotation.T),
                         1e-4 * (q - initial)]

        result = least_squares(residual, seed, bounds=(self.lower, self.upper),
                               max_nfev=150, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        pose = self.fk(result.x)
        position_error = float(np.linalg.norm(target.translation - pose.translation))
        rotation_error = float(np.rad2deg(np.linalg.norm(
            pin.log3(target.rotation @ pose.rotation.T))))
        if (not result.success or not np.isfinite(result.x).all()
                or position_error > position_tolerance
                or rotation_error > rotation_tolerance_deg):
            raise ValueError(f"IK rejected: position={position_error:.4f}m "
                             f"rotation={rotation_error:.2f}deg status={result.message}")
        degrees = (np.rad2deg(result.x) - self.offsets) / self.signs
        return degrees, position_error, rotation_error

    def gripper(self, fraction):
        return self.closed + float(np.clip(fraction, 0, 1)) * (self.open - self.closed)
