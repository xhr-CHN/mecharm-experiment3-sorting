"""Pure fixed-point pick-and-place state machine."""

from __future__ import annotations

from .result_types import ErrorCode, TaskState


_SEQUENCE = (
    TaskState.HOME,
    TaskState.ABOVE_PICK,
    TaskState.DESCEND_PICK,
    TaskState.CLOSE_GRIPPER,
    TaskState.ATTACH_OBJECT,
    TaskState.LIFT_PICK,
    TaskState.ABOVE_PLACE,
    TaskState.DESCEND_PLACE,
    TaskState.OPEN_GRIPPER,
    TaskState.DETACH_OBJECT,
    TaskState.RETREAT,
    TaskState.RETURN_HOME,
    TaskState.SUCCESS,
)


class PickPlaceStateMachine:
    def __init__(self) -> None:
        self.state = TaskState.IDLE
        self.error_code = ErrorCode.NONE
        self.error_message = ""

    @property
    def active(self) -> bool:
        return self.state not in {TaskState.IDLE, TaskState.SUCCESS, TaskState.FAILED}

    def start(self) -> TaskState:
        if self.active:
            raise RuntimeError(f"task is already active in state {self.state.value}")
        self.error_code = ErrorCode.NONE
        self.error_message = ""
        self.state = _SEQUENCE[0]
        return self.state

    def advance(self) -> TaskState:
        if not self.active:
            raise RuntimeError(f"cannot advance terminal state {self.state.value}")
        index = _SEQUENCE.index(self.state)
        self.state = _SEQUENCE[index + 1]
        return self.state

    def fail(self, code: ErrorCode, message: str) -> TaskState:
        if code is ErrorCode.NONE:
            raise ValueError("failure requires a non-NONE error code")
        self.error_code = code
        self.error_message = message
        self.state = TaskState.FAILED
        return self.state

    def reset(self) -> TaskState:
        self.state = TaskState.IDLE
        self.error_code = ErrorCode.NONE
        self.error_message = ""
        return self.state

