import ast
from pathlib import Path


SOURCE = Path(
    "src/mecharm_pick_place/mecharm_pick_place/direct_motion_probe.py"
).read_text(encoding="utf-8")


def _load_smoothstep():
    tree = ast.parse(SOURCE)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_smoothstep"
    )
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<smoothstep>", "exec"), namespace)
    return namespace["_smoothstep"]


def test_probe_uses_smooth_named_joint_targets_without_moveit():
    assert '"/joint_states"' in SOURCE
    assert '"/mecharm/joint_target"' in SOURCE
    assert "_smoothstep" in SOURCE
    assert "MOTION_PROBE_SUCCESS" in SOURCE
    assert "moveit" not in SOURCE.lower()


def test_smoothstep_is_monotonic_with_zero_endpoint_acceleration():
    smoothstep = _load_smoothstep()
    samples = [smoothstep(index / 100.0) for index in range(101)]
    assert smoothstep(-1.0) == 0.0
    assert smoothstep(0.0) == 0.0
    assert smoothstep(1.0) == 1.0
    assert smoothstep(2.0) == 1.0
    assert all(left <= right for left, right in zip(samples, samples[1:]))

    step = 1e-4
    start_acceleration = (
        smoothstep(2.0 * step) - 2.0 * smoothstep(step) + smoothstep(0.0)
    ) / (step * step)
    end_acceleration = (
        smoothstep(1.0)
        - 2.0 * smoothstep(1.0 - step)
        + smoothstep(1.0 - 2.0 * step)
    ) / (step * step)
    assert abs(start_acceleration) < 0.01
    assert abs(end_acceleration) < 0.01
