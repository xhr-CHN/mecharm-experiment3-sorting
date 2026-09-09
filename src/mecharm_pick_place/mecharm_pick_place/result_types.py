"""Stable task states and JSON result payloads."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
import time


class TaskState(str, Enum):
    IDLE = "IDLE"
    HOME = "HOME"
    ABOVE_PICK = "ABOVE_PICK"
    DESCEND_PICK = "DESCEND_PICK"
    CLOSE_GRIPPER = "CLOSE_GRIPPER"
    ATTACH_OBJECT = "ATTACH_OBJECT"
    LIFT_PICK = "LIFT_PICK"
    ABOVE_PLACE = "ABOVE_PLACE"
    DESCEND_PLACE = "DESCEND_PLACE"
    OPEN_GRIPPER = "OPEN_GRIPPER"
    DETACH_OBJECT = "DETACH_OBJECT"
    RETREAT = "RETREAT"
    RETURN_HOME = "RETURN_HOME"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class ErrorCode(str, Enum):
    NONE = "NONE"
    NO_IK = "NO_IK"
    JOINT_LIMIT = "JOINT_LIMIT"
    MOTION_FAILED = "MOTION_FAILED"
    TIMEOUT = "TIMEOUT"
    GRIPPER_FAILED = "GRIPPER_FAILED"
    OBJECT_OUT_OF_RANGE = "OBJECT_OUT_OF_RANGE"
    COLLISION = "COLLISION"
    COMMUNICATION = "COMMUNICATION"
    STOP_REQUESTED = "STOP_REQUESTED"
    INVALID_RESULT = "INVALID_RESULT"


@dataclass(frozen=True)
class TaskEvent:
    attempt_id: int
    state: str
    success: bool
    error_code: str = ErrorCode.NONE.value
    message: str = ""
    timestamp: float = 0.0

    def to_json(self) -> str:
        payload = asdict(self)
        if payload["timestamp"] == 0.0:
            payload["timestamp"] = time.time()
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "TaskEvent":
        data = json.loads(raw)
        required = {"attempt_id", "state", "success"}
        missing = required.difference(data)
        if missing:
            raise ValueError(f"missing task event fields: {sorted(missing)}")
        return cls(
            attempt_id=int(data["attempt_id"]),
            state=str(data["state"]),
            success=bool(data["success"]),
            error_code=str(data.get("error_code", ErrorCode.NONE.value)),
            message=str(data.get("message", "")),
            timestamp=float(data.get("timestamp", 0.0)),
        )

