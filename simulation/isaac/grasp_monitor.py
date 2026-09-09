"""Physics-only adaptive-gripper attachment monitor."""

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


class GraspMonitor:
    def __init__(
        self,
        stage,
        articulation,
        object_prim_path="/World/target_object",
        gripper_body_path="/World/mecharm_270_pi/gripper_base",
        gripper_joint_name="gripper_controller",
        closed_threshold=-0.68,
        open_threshold=0.08,
        attach_distance=0.08,
        joint_path="/World/mecharm_grasp_fixed_joint",
        object_root_path=None,
    ):
        self.stage = stage
        self.articulation = articulation
        self.object_prim_path = object_prim_path
        self.object_root_path = object_root_path
        self.gripper_body_path = gripper_body_path
        self.gripper_index = articulation.get_dof_index(gripper_joint_name)
        self.closed_threshold = closed_threshold
        self.open_threshold = open_threshold
        self.attach_distance = attach_distance
        self.joint_path = joint_path

    def _matrix(self, path):
        prim = self.stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            return None
        return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    def _remove(self, reset_object=True):
        if self.stage.GetPrimAtPath(self.joint_path).IsValid():
            self.stage.RemovePrim(self.joint_path)
        if reset_object and self.object_root_path:
            self.object_prim_path = None

    def _nearest_object(self):
        if not self.object_root_path:
            return self.object_prim_path
        gripper = self._matrix(self.gripper_body_path)
        if gripper is None:
            return None
        gripper_xyz = np.asarray(gripper.ExtractTranslation(), dtype=float)
        best_path = None
        best_distance = float("inf")
        prefix = self.object_root_path.rstrip("/") + "/"
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith(prefix) or path.count("/") != prefix.count("/"):
                continue
            target = self._matrix(path)
            if target is None:
                continue
            distance = float(
                np.linalg.norm(np.asarray(target.ExtractTranslation(), dtype=float) - gripper_xyz)
            )
            if distance < best_distance:
                best_distance = distance
                best_path = path
        return best_path

    def _attach(self):
        if self.object_root_path:
            self.object_prim_path = self._nearest_object()
        if not self.object_prim_path:
            return
        object_path = self.object_prim_path
        gripper = self._matrix(self.gripper_body_path)
        target = self._matrix(object_path)
        if gripper is None or target is None:
            return
        gripper_xyz = np.asarray(gripper.ExtractTranslation(), dtype=float)
        target_xyz = np.asarray(target.ExtractTranslation(), dtype=float)
        if np.linalg.norm(gripper_xyz - target_xyz) > self.attach_distance:
            return
        self._remove(reset_object=False)
        relative = target * gripper.GetInverse()
        translation = relative.ExtractTranslation()
        rotation = relative.ExtractRotationQuat()
        imaginary = rotation.GetImaginary()
        joint = UsdPhysics.FixedJoint.Define(self.stage, self.joint_path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(self.gripper_body_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(object_path)])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*translation))
        joint.CreateLocalRot0Attr().Set(
            Gf.Quatf(float(rotation.GetReal()), Gf.Vec3f(*imaginary))
        )
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

    def update(self):
        position = float(self.articulation.get_joint_positions()[self.gripper_index])
        attached = self.stage.GetPrimAtPath(self.joint_path).IsValid()
        if position <= self.closed_threshold and not attached:
            self._attach()
        elif position >= self.open_threshold and attached:
            self._remove()
