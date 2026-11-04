"""Offline arrival-gate tests; no camera or robot connection."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from run_so101_ee_policy import wait_until_settled


class ArrivalTest(unittest.TestCase):
    def args(self):
        return SimpleNamespace(
            settle_poll_hz=20, settle_time_s=0.4, arrival_timeout_s=1,
            settle_motion_deg=0.3, settle_gripper_motion=0.5,
            arrival_tolerance_deg=2, gripper_arrival_tolerance=3,
        )

    def test_arrival_requires_continuous_stability(self):
        robot = SimpleNamespace(read_state=lambda: np.zeros(6))
        with patch("run_so101_ee_policy.time.sleep"), patch(
            "run_so101_ee_policy.time.monotonic", side_effect=[0, 0.1, 0.3, 0.6]
        ):
            measured, timestamp = wait_until_settled(robot, np.zeros(6), self.args())
        self.assertEqual(timestamp, 0.6)
        np.testing.assert_array_equal(measured, np.zeros(6))

    def test_gripper_not_at_target_times_out(self):
        robot = SimpleNamespace(read_state=lambda: np.zeros(6))
        target = np.array([0, 0, 0, 0, 0, 10])
        with patch("run_so101_ee_policy.time.sleep"), patch(
            "run_so101_ee_policy.time.monotonic", side_effect=[0, 0.5, 1.1]
        ), self.assertRaises(TimeoutError):
            wait_until_settled(robot, target, self.args())


if __name__ == "__main__":
    unittest.main()
