import pytest

from mecharm_pick_place.sorting_state_machine import SortState, SortingStateMachine
from mecharm_pick_place.sorting_types import SortingTarget


def _advance_until(machine, state):
    while machine.state is not state:
        machine.motion_succeeded()


def test_successful_target_reaches_next_target():
    machine = SortingStateMachine()
    machine.start()
    machine.set_target(
        SortingTarget("obj1", "tennis_ball", "G1", (0.2, 0.0, 0.05), (0.2, -0.1, 0.05))
    )
    _advance_until(machine, SortState.CLOSE_GRIPPER)
    machine.gripper_succeeded()
    machine.attachment_succeeded()
    _advance_until(machine, SortState.OPEN_GRIPPER)
    machine.gripper_succeeded()
    machine.placement_succeeded()
    machine.verify_place(True)
    _advance_until(machine, SortState.NEXT_TARGET)
    assert machine.state is SortState.NEXT_TARGET
    assert machine.failure_code is None


def test_unknown_class_is_recoverable_before_motion():
    machine = SortingStateMachine()
    machine.start()
    machine.set_target(
        SortingTarget("obj2", "unknown", "G2", (0.2, 0.0, 0.05), None)
    )
    assert machine.state is SortState.SAFE_STOP
    assert machine.failure_code == "UNKNOWN_CLASS"
    assert machine.motion_commands == 0


def test_motion_failure_enters_safe_stop():
    machine = SortingStateMachine()
    machine.start()
    machine.set_target(
        SortingTarget("obj3", "pencil", "G3", (0.2, 0.0, 0.05), (0.3, -0.1, 0.05))
    )
    machine.motion_failed("UNREACHABLE", "fixed point rejected")
    assert machine.state is SortState.SAFE_STOP
    assert machine.failure_code == "UNREACHABLE"


def test_terminal_state_cannot_accept_a_new_motion():
    machine = SortingStateMachine()
    machine.start()
    machine.fail("COLLISION", "contact detected")
    with pytest.raises(RuntimeError, match="terminal"):
        machine.motion_succeeded()
