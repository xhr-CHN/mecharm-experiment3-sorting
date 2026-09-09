"""Render deterministic candidate camera views for experiment three."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path


def camera_candidates(height: float = 1.10) -> list[dict]:
    candidates = [
        {
            "view_id": "vertical",
            "tilt_deg": 0.0,
            "azimuth_deg": 0.0,
            "position": [0.0, 0.0, height],
        }
    ]
    directions = ((0.0, "east"), (90.0, "north"), (180.0, "west"), (270.0, "south"))
    for tilt in (10.0, 15.0, 20.0):
        offset = height * math.tan(math.radians(tilt))
        for azimuth, label in directions:
            angle = math.radians(azimuth)
            candidates.append(
                {
                    "view_id": f"tilt{int(tilt)}_{label}",
                    "tilt_deg": tilt,
                    "azimuth_deg": azimuth,
                    "position": [
                        offset * math.cos(angle),
                        offset * math.sin(angle),
                        height,
                    ],
                }
            )
    return candidates


def is_usable_frame(frame) -> bool:
    """Reject empty or uniformly black frames before they become evidence."""
    import numpy as np

    array = np.asarray(frame)
    return bool(array.size and array.max() > 10 and array.std() > 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--frames-per-view", type=int, default=5)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    scene_path = project_root / "simulation/scenes/mecharm_pick_place.usd"
    if not scene_path.is_file():
        raise FileNotFoundError(f"scene not found: {scene_path}")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else project_root / "results/experiment3/camera_view_sweep" / stamp
    )
    frames_root = output_root / "frames"
    frames_root.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": bool(args.headless)})
    try:
        import cv2
        import omni.replicator.core as rep
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import open_stage
        from pxr import Gf, UsdGeom

        from experiment3_camera_stream import normalize_rgb_frame

        if not open_stage(str(scene_path)):
            raise RuntimeError(f"failed to open scene: {scene_path}")
        for _ in range(30):
            simulation_app.update()

        candidates = camera_candidates()
        camera_path = "/World/experiment3/top_camera"
        stage = omni.usd.get_context().get_stage()
        camera_prim = stage.GetPrimAtPath(camera_path)
        if not camera_prim.IsValid():
            raise RuntimeError(f"camera prim not found: {camera_path}")
        camera_xform = UsdGeom.Xformable(camera_prim)
        camera_xform.ClearXformOpOrder()
        camera_transform = camera_xform.AddTransformOp()

        world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / 120.0,
            rendering_dt=1.0 / 60.0,
        )
        world.reset()
        render_product = rep.create.render_product(
            camera_path, (848, 480), name="experiment3_camera_view_sweep"
        )
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach([render_product])
        world.play()
        manifest = {
            "scene": str(scene_path),
            "resolution": [848, 480],
            "focal_length_mm": 18.0,
            "horizontal_aperture_mm": 20.955,
            "look_at": [0.0, 0.0, 0.0],
            "frames_per_view": int(args.frames_per_view),
            "views": [],
            "frames": [],
        }
        for candidate in candidates:
            eye = Gf.Vec3d(*candidate["position"])
            camera_to_world = Gf.Matrix4d().SetLookAt(
                eye, Gf.Vec3d(0.0, 0.0, 0.0), Gf.Vec3d(0.0, 1.0, 0.0)
            ).GetInverse()
            camera_transform.Set(camera_to_world)
            for _ in range(20):
                world.step(render=True)
            view_dir = frames_root / candidate["view_id"]
            view_dir.mkdir(parents=True, exist_ok=True)
            manifest["views"].append(candidate)
            for frame_index in range(int(args.frames_per_view)):
                frame = None
                for _ in range(20):
                    world.step(render=True)
                    candidate_frame = normalize_rgb_frame(
                        annotator.get_data(), 848, 480
                    )
                    if is_usable_frame(candidate_frame):
                        frame = candidate_frame
                        break
                if frame is None:
                    raise RuntimeError(
                        f"camera kept returning black frames: {candidate['view_id']}"
                    )
                path = view_dir / f"frame_{frame_index + 1:02d}.png"
                success, encoded = cv2.imencode(
                    ".png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                )
                if not success:
                    raise RuntimeError(f"failed to encode camera frame: {path}")
                encoded.tofile(str(path))
                if not path.is_file() or path.stat().st_size == 0:
                    raise RuntimeError(f"camera frame was not written: {path}")
                manifest["frames"].append(
                    {
                        "view_id": candidate["view_id"],
                        "frame_index": frame_index + 1,
                        "path": str(path.relative_to(output_root)).replace("\\", "/"),
                    }
                )
                print(f"CAPTURED {candidate['view_id']} {frame_index + 1}", flush=True)
        (output_root / "view_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(f"SWEEP_OUTPUT={output_root}", flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
