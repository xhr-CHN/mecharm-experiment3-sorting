import json

import pytest

from mecharm_pick_place.result_types import ErrorCode, TaskEvent


def test_task_event_round_trip():
    event = TaskEvent(
        attempt_id=3,
        state="ABOVE_PICK",
        success=True,
        error_code=ErrorCode.NONE.value,
        message="planned",
        timestamp=123.5,
    )
    assert TaskEvent.from_json(event.to_json()) == event


def test_task_event_requires_core_fields():
    with pytest.raises(ValueError, match="missing task event fields"):
        TaskEvent.from_json(json.dumps({"attempt_id": 1}))

