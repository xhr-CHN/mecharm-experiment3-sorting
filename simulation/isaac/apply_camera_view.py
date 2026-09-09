"""Apply one evidence-backed camera sweep view to an existing USD scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--view-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_path = Path(args.scene).resolve()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    selected = next(
        (view for view in manifest["views"] if view["view_id"] == args.view_id), None
    )
    if selected is None:
        raise ValueError(f"view not found in manifest: {args.view_id}")

    simulation_app = SimulationApp({"headless": True})
    try:
        import omni.usd
        from isaacsim.core.utils.stage import open_stage
        from pxr import Gf, UsdGeom

        if not open_stage(str(scene_path)):
            raise RuntimeError(f"failed to open scene: {scene_path}")
        for _ in range(30):
            simulation_app.update()
        stage = omni.usd.get_context().get_stage()
        root_layer = stage.GetRootLayer()
        stage.SetEditTarget(root_layer)
        camera = stage.GetPrimAtPath("/World/experiment3/top_camera")
        if not camera.IsValid():
            raise RuntimeError("experiment-three top camera was not found")
        xform = UsdGeom.Xformable(camera)
        xform.ClearXformOpOrder()
        transform = Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(*selected["position"]),
            Gf.Vec3d(*manifest["look_at"]),
            Gf.Vec3d(0.0, 1.0, 0.0),
        ).GetInverse()
        xform.AddTransformOp().Set(transform)
        root_layer.Save()
        persisted = UsdGeom.Xformable(camera).ComputeLocalToWorldTransform(0.0)
        print(f"CAMERA_POSITION={list(persisted.ExtractTranslation())}", flush=True)
        print(f"APPLIED_CAMERA_VIEW={args.view_id}", flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
