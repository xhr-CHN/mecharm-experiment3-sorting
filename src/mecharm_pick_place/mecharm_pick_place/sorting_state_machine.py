"""Pure state machine for one autonomous circular sorting batch."""

from __future__ import annotations

from enum import Enum

from .sorting_types import SortingTarget


class SortState(str, Enum):
    IDLE = "IDLE"
    WAIT_DETECTIONS = "WAIT_DETECTIONS"
    SELECT_TARGET = "SELECT_TARGET"
    CHECK_TARGET = "CHECK_TARGET"
    HOME = "HOME"
    ABOVE_PICK = "ABOVE_PICK"
    DESCEND_PICK = "DESCEND_PICK"
    CLOSE_GRIPPER = "CLOSE_GRIPPER"
    ATTACH_OBJECT = "ATTACH_OBJECT"
    LIFT = "LIFT"
    ABOVE_BIN = "ABOVE_BIN"
    DESCEND_BIN = "DESCEND_BIN"
    OPEN_GRIPPER = "OPEN_GRIPPER"
    DETACH_OBJECT = "DETACH_OBJECT"
    VERIFY_PLACE = "VERIFY_PLACE"
    RETURN_HOME = "RETURN_HOME"
    NEXT_TARGET = "NEXT_TARGET"
    DONE = "DONE"
    SAFE_STOP = "SAFE_STOP"


class SortingStateMachine:
    """A small explicit transition model; ROS execution remains outside it."""

    _MOTION_NEXT = {
        SortState.HOME: SortState.ABOVE_PICK,
        SortState.ABOVE_PICK: SortState.DESCEND_PICK,
        SortState.DESCEND_PICK: SortState.CLOSE_GRIPPER,
        SortState.LIFT: SortState.ABOVE_BIN,
        SortState.ABOVE_BIN: SortState.DESCEND_BIN,
        SortState.DESCEND_BIN: SortState.OPEN_GRIPPER,
        SortState.RETURN_HOME: SortState.NEXT_TARGET,
    }

    def __init__(self) -> None:
        self.state = SortState.IDLE
        self.target: SortingTarget | None = None
        self.failure_code: str | None = None
        self.failure_message = ""
        self.motion_commands = 0
        self.motion_step_count = 1

    @property
    def terminal(self) -> bool:
        return self.state in {SortState.DONE, SortState.SAFE_STOP}

    def start(self) -> SortState:
        if self.state is not SortState.IDLE:
            raise RuntimeError(f"cannot start from state {self.state.value}")
        self.state = SortState.WAIT_DETECTIONS
        return self.state

    def set_target(self, target: SortingTarget) -> SortState:
        if self.state not in {SortState.WAIT_DETECTIONS, SortState.SELECT_TARGET}:
            raise RuntimeError(f"cannot set target from state {self.state.value}")
        self.target = target
        if target.bin_position is None:
            self.fail("UNKNOWN_CLASS", f"no bin configured for {target.class_id}")
        else:
            self.state = SortState.HOME
        return self.state

    def motion_succeeded(self) -> SortState:
        self._require_active()
        if self.state not in self._MOTION_NEXT:
            raise RuntimeError(f"motion is not expected in state {self.state.value}")
        self.motion_commands += 1
        self.state = self._MOTION_NEXT[self.state]
        return self.state

    def gripper_succeeded(self) -> SortState:
        self._require_active()
        if self.state is SortState.CLOSE_GRIPPER:
            self.state = SortState.ATTACH_OBJECT
        elif self.state is SortState.OPEN_GRIPPER:
            self.state = SortState.DETACH_OBJECT
        else:
            raise RuntimeError(f"gripper is not expected in state {self.state.value}")
        return self.state

    def attachment_succeeded(self) -> SortState:
        self._require_active()
        if self.state is not SortState.ATTACH_OBJECT:
            raise RuntimeError(f"attachment is not expected in state {self.state.value}")
        self.state = SortState.LIFT
        return self.state

    def placement_succeeded(self) -> SortState:
        self._require_active()
        if self.state is not SortState.DETACH_OBJECT:
            raise RuntimeError(f"placement is not expected in state {self.state.value}")
        self.state = SortState.VERIFY_PLACE
        return self.state

    def verify_place(self, success: bool) -> SortState:
        self._require_active()
        if self.state is not SortState.VERIFY_PLACE:
            raise RuntimeError(f"verification is not expected in state {self.state.value}")
        if success:
            self.state = SortState.RETURN_HOME
        else:
            self.fail("PLACE_FAILED", "object is outside the selected bin")
        return self.state

    def next_target(self, available: bool) -> SortState:
        self._require_active()
        if self.state is not SortState.NEXT_TARGET:
            raise RuntimeError(f"next target is not expected in state {self.state.value}")
        self.state = SortState.SELECT_TARGET if available else SortState.DONE
        return self.state

    def motion_failed(self, code: str, message: str) -> SortState:
        self._require_active()
        self.fail(code, message)
        return self.state

    def fail(self, code: str, message: str) -> SortState:
        if self.terminal:
            raise RuntimeError(f"cannot fail terminal state {self.state.value}")
        self.failure_code = str(code)
        self.failure_message = str(message)
        self.state = SortState.SAFE_STOP
        return self.state

    def _require_active(self) -> None:
        if self.terminal or self.state is SortState.IDLE:
            raise RuntimeError(f"cannot advance terminal state {self.state.value}")
