from mecharm_pick_place.result_types import ErrorCode, TaskState
from mecharm_pick_place.state_machine import PickPlaceStateMachine


def test_normal_sequence_reaches_success():
    machine = PickPlaceStateMachine()
    assert machine.start() is TaskState.HOME
    while machine.active:
        machine.advance()
    assert machine.state is TaskState.SUCCESS
    assert machine.error_code is ErrorCode.NONE


def test_failure_is_terminal_and_preserves_error():
    machine = PickPlaceStateMachine()
    machine.start()
    machine.fail(ErrorCode.NO_IK, "unreachable pose")
    assert machine.state is TaskState.FAILED
    assert machine.error_code is ErrorCode.NO_IK
    assert machine.error_message == "unreachable pose"


def test_cannot_advance_terminal_state():
    machine = PickPlaceStateMachine()
    try:
        machine.advance()
    except RuntimeError as exc:
        assert "terminal state" in str(exc)
    else:
        raise AssertionError("terminal state advanced unexpectedly")

