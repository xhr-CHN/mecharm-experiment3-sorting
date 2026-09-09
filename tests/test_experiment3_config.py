from pathlib import Path
import json
import math

import yaml


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "config" / "experiment3_sorting.yaml"


def load_config():
    with CONFIG.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)["sorting_task"]["ros__parameters"]


def test_configuration_has_six_ring_cells_and_two_classes():
    config = load_config()
    assert list(config["grid_centers"]) == ["G1", "G2", "G3", "G4", "G5", "G6"]
    assert set(config["class_bins"]) == {"tennis_ball", "pencil"}
    assert len(config["objects"]) == 6
    assert {item["class_id"] for item in config["objects"]} == {"tennis_ball", "pencil"}


def test_ring_coordinates_are_inside_table_and_have_nonzero_radius():
    config = load_config()
    assert config["ring_center"] == [0.0, 0.0]
    points = list(config["grid_centers"].values()) + list(config["class_bins"].values())
    assert all(len(point) == 3 for point in points)
    assert max(abs(point[0]) for point in points) < 0.35
    assert max(abs(point[1]) for point in points) < 0.35
    expected_radius = math.hypot(0.18, 0.08)
    radii = [math.hypot(point[0], point[1]) for point in points]
    assert all(abs(radius - expected_radius) < 1e-4 for radius in radii)
    assert config["grid_centers"]["G1"] == [0.18, 0.08, 0.025]
    angles = sorted(
        math.degrees(math.atan2(point[1], point[0])) % 360 for point in points
    )
    deltas = [
        (angles[(index + 1) % 8] - angles[index]) % 360 for index in range(8)
    ]
    assert all(abs(delta - 45.0) < 0.1 for delta in deltas)
    assert all(abs(angle - 180.0) > 1.0 for angle in angles)


def test_each_object_matches_its_world_grid_and_image_region():
    config = load_config()
    for item in config["objects"]:
        assert item["center"] == config["grid_centers"][item["grid_id"]]
        center_x, center_y = item["bbox"][:2]
        left, top, right, bottom = config["grid_regions"][item["grid_id"]]
        assert left <= center_x <= right
        assert top <= center_y <= bottom


def test_scene_builds_open_top_bins_from_a_bottom_and_four_walls():
    scene = (ROOT / "simulation" / "isaac" / "experiment3_scene.py").read_text(
        encoding="utf-8"
    )
    for part in ("bottom", "wall_pos_x", "wall_neg_x", "wall_pos_y", "wall_neg_y"):
        assert part in scene
    assert 'prim_path=f"{root}/bins/{bin_id}"' not in scene
    assert "bin_wall_thickness = 0.008" in scene
    assert "bin_wall_height = 0.045" in scene


def test_scene_builds_thin_green_wooden_pencils_as_one_rigid_body():
    scene = (ROOT / "simulation" / "isaac" / "experiment3_scene.py").read_text(
        encoding="utf-8"
    )
    for token in (
        "UsdPhysics.RigidBodyAPI",
        "UsdPhysics.CollisionAPI",
        "/shaft",
        "/wood_tip",
        "/graphite_tip",
        "UsdGeom.Cylinder",
        "0.090",
        "0.0075",
        "0.0225",
    ):
        assert token in scene
    assert "DynamicCuboid" not in scene


def test_top_camera_uses_wide_angle_focal_length_for_full_ring():
    scene = (ROOT / "simulation" / "isaac" / "experiment3_scene.py").read_text(
        encoding="utf-8"
    )
    assert "CreateFocalLengthAttr().Set(18.0)" in scene
    assert "center[1] + 0.1939596787793115" in scene
    assert "SetLookAt(" in scene


def test_simulated_pencil_boxes_are_thin_and_green():
    with CONFIG.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    pencils = [
        item
        for item in document["sorting_task"]["ros__parameters"]["objects"]
        if item["class_id"] == "pencil"
    ]
    assert all(item["bbox"][3] == 30 for item in pencils)
    assert document["sim_camera"]["ros__parameters"]["rectangles_json"].count(
        '"color":[45,160,70]'
    ) == len(pencils)


def test_pencil_shaft_and_tips_have_real_collision_geometry():
    scene = (ROOT / "simulation" / "isaac" / "experiment3_scene.py").read_text(
        encoding="utf-8"
    )
    assert "shaft.CreateHeightAttr().Set(0.090)" in scene
    assert "shaft.CreateRadiusAttr().Set(0.0075)" in scene
    assert "wood_tip.CreateRadiusAttr().Set(0.0075)" in scene
    assert "wood_tip.CreateHeightAttr().Set(0.0225)" in scene
    assert "graphite_tip.CreateHeightAttr().Set(0.0075)" in scene
    assert "SetTranslate((0.0638, 0.0, 0.0))" in scene
    assert "UsdPhysics.CollisionAPI.Apply(shaft.GetPrim())" in scene
    assert "UsdPhysics.CollisionAPI.Apply(wood_tip.GetPrim())" in scene
    assert "UsdPhysics.CollisionAPI.Apply(graphite_tip.GetPrim())" in scene


def test_pencil_shaft_uses_high_friction_physics_material():
    scene = (ROOT / "simulation" / "isaac" / "experiment3_scene.py").read_text(
        encoding="utf-8"
    )
    assert "CreateStaticFrictionAttr().Set(1.0)" in scene
    assert "CreateDynamicFrictionAttr().Set(1.0)" in scene
    assert "CreateFrictionCombineModeAttr().Set(\"max\")" in scene
    assert "MaterialBindingAPI(shaft.GetPrim()).Bind(pencil_material)" in scene
    assert '"/World/mecharm_270_pi/gripper_left1"' in scene
    assert '"/World/mecharm_270_pi/gripper_right1"' in scene


def test_tennis_balls_use_twenty_mm_radius_and_forty_pixel_boxes():
    scene = (ROOT / "simulation" / "isaac" / "experiment3_scene.py").read_text(
        encoding="utf-8"
    )
    assert "radius=0.020" in scene
    with CONFIG.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    objects = document["sorting_task"]["ros__parameters"]["objects"]
    tennis_objects = [item for item in objects if item["class_id"] == "tennis_ball"]
    assert all(item["bbox"][2:] == [40, 40] for item in tennis_objects)
    detector_records = json.loads(
        document["sim_object_detector"]["ros__parameters"]["object_records_json"]
    )
    tennis_records = [
        item for item in detector_records if item["class_id"] == "tennis_ball"
    ]
    assert all(item["width"] == 40.0 and item["height"] == 40.0 for item in tennis_records)


def test_detection_regions_cover_the_six_configured_grids():
    config = load_config()
    assert list(config["grid_regions"]) == ["G1", "G2", "G3", "G4", "G5", "G6"]
    assert all(len(region) == 4 for region in config["grid_regions"].values())
    assert config["grid_regions"] == {
        "G1": [496, 140, 596, 240],
        "G2": [423, 69, 523, 169],
        "G3": [319, 71, 419, 171],
        "G4": [255, 245, 355, 345],
        "G5": [328, 310, 428, 410],
        "G6": [426, 308, 526, 408],
    }
