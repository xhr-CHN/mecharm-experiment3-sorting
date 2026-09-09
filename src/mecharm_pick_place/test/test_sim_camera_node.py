import pytest

from mecharm_pick_place.sim_camera_node import make_rgb_frame


def test_rgb_frame_has_expected_size_and_colored_region():
    frame = make_rgb_frame(4, 3, [(1, 1, 3, 2, (10, 20, 30))])
    assert len(frame) == 4 * 3 * 3
    assert frame[(1 * 4 + 1) * 3 : (1 * 4 + 1) * 3 + 3] == bytes((10, 20, 30))
    assert frame[0:3] == bytes((0, 0, 0))


def test_rgb_frame_rejects_invalid_dimensions():
    with pytest.raises(ValueError, match="positive"):
        make_rgb_frame(0, 3)
