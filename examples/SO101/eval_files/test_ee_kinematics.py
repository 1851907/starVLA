"""Offline FK/IK checks using a synthetic URDF; never connects hardware."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ee_kinematics import EEKinematics, JOINTS


class IKTest(unittest.TestCase):
    def test_roundtrip_mapping_and_reject_unreachable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            urdf = root / "robot.urdf"
            xml = ['<robot name="test"><link name="base_link"/>']
            parent = "base_link"
            for i, (name, axis) in enumerate(zip(
                JOINTS, ("0 0 1", "0 1 0", "0 1 0", "0 1 0", "1 0 0")
            )):
                child = f"link{i}"
                xml.append(
                    f'<link name="{child}"/><joint name="{name}" type="revolute">'
                    f'<parent link="{parent}"/><child link="{child}"/>'
                    f'<origin xyz="0.1 0 0"/><axis xyz="{axis}"/>'
                    '<limit lower="-2.5" upper="2.5" effort="1" velocity="1"/></joint>'
                )
                parent = child
            xml.append(f'<link name="tcp"/><joint name="tool" type="fixed">'
                       f'<parent link="{parent}"/><child link="tcp"/>'
                       '<origin xyz="0.1 0 0"/></joint></robot>')
            urdf.write_text("".join(xml))
            report = {
                "status": "complete",
                "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
                "base_frame": "base_link", "tool_frame": "tcp",
                "joint_signs": [1, -1, 1, 1, 1],
                "joint_offsets_deg": [0, 5, 0, 0, 0],
                "gripper_closed": 30, "gripper_open": 0,
            }
            config = root / "conversion.json"
            config.write_text(json.dumps(report))
            kin = EEKinematics(urdf, config)
            start = np.array([0, 20, -30, 10, 0.0])
            goal = start + [1, 2, -1, 1, 2]
            target = kin.fk(kin.to_radians(goal))
            solved, p_error, r_error = kin.solve(target, start, 0.0001, 0.1)
            self.assertLess(p_error, 0.0001)
            self.assertLess(r_error, 0.1)
            self.assertEqual(solved.shape, (5,))
            np.testing.assert_allclose(
                kin.target(start, np.zeros(7)).translation,
                kin.fk(kin.to_radians(start)).translation,
            )
            self.assertEqual(kin.gripper(0), 30)
            self.assertEqual(kin.gripper(1), 0)
            target.translation = np.array([10.0, 10.0, 10.0])
            with self.assertRaises(ValueError):
                kin.solve(target, start)
            report["urdf_sha256"] = "incorrect"
            config.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                EEKinematics(urdf, config)


if __name__ == "__main__":
    unittest.main()
