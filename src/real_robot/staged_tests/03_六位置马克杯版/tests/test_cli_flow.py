from __future__ import annotations

import sys
import unittest
from unittest import mock

import grasp_mug_test as app


class RecordingController:
    instances = []

    def __init__(self, cfg):
        self.calls = []
        self.closed = False
        self.__class__.instances.append(self)

    def recenter_arm(self):
        self.calls.append(("recenter",))

    def run_route(self, route, label):
        self.calls.append(("route", label))

    def open_gripper(self):
        self.calls.append(("open",))

    def close_gripper(self):
        self.calls.append(("close",))

    def move_arm(self, point, label):
        self.calls.append(("arm", tuple(point), label))

    def stop_safely(self):
        self.calls.append(("stop",))

    def close(self):
        self.closed = True

    def close_connection_only(self):
        self.closed = True


class RecordingLog:
    instances = []

    def __init__(self, root):
        self.events = []
        self.closed = False
        self.__class__.instances.append(self)

    def write(self, event, **details):
        self.events.append((event, details))

    def close(self):
        self.closed = True


class CommandLineFlowTests(unittest.TestCase):
    def setUp(self):
        RecordingController.instances.clear()
        RecordingLog.instances.clear()

    def run_simulated(self, *arguments):
        argv = ["grasp_mug_test.py", *arguments]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            app, "RoboMasterController", RecordingController
        ), mock.patch.object(app, "EventLog", RecordingLog), mock.patch.object(
            app.time, "sleep", return_value=None
        ):
            app.main()
        log = RecordingLog.instances[-1] if RecordingLog.instances else None
        return RecordingController.instances[-1], log

    def test_unacknowledged_nominal_points_never_connect(self):
        with mock.patch.object(
            sys, "argv", ["grasp_mug_test.py", "--execute", "--grid", "G1"]
        ), mock.patch.object(app, "RoboMasterController") as controller_type:
            with self.assertRaisesRegex(ValueError, "calibrated"):
                app.main()
            controller_type.assert_not_called()

    def test_connection_test_connects_and_closes_without_motion(self):
        controller, _ = self.run_simulated("--connection-test")
        self.assertEqual(controller.calls, [])
        self.assertTrue(controller.closed)

    def test_acknowledged_single_grid_executes_only_g1(self):
        controller, log = self.run_simulated(
            "--execute",
            "--grid",
            "G1",
            "--acknowledge-unverified-arm-points",
        )
        completed = [
            details
            for event, details in log.events
            if event == "position_completed"
        ]
        self.assertEqual(completed, [{"grid_id": "G1", "destination": "right"}])
        self.assertEqual(sum(call[0] == "close" for call in controller.calls), 1)
        self.assertTrue(controller.closed)
        self.assertTrue(log.closed)

    def test_acknowledged_full_run_executes_six_positions(self):
        controller, log = self.run_simulated(
            "--execute", "--acknowledge-unverified-arm-points"
        )
        completed = [
            (details["grid_id"], details["destination"])
            for event, details in log.events
            if event == "position_completed"
        ]
        self.assertEqual(completed, app.EXPECTED_SEQUENCE)
        self.assertEqual(sum(call[0] == "close" for call in controller.calls), 6)
        self.assertEqual(sum(call[0] == "open" for call in controller.calls), 12)
        self.assertEqual(sum(call[0] == "arm" for call in controller.calls), 36)
        self.assertTrue(controller.closed)
        self.assertTrue(log.closed)


if __name__ == "__main__":
    unittest.main()
