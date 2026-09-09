"""URDF-based numerical IK with a free wrist yaw around a vertical tool axis."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares


ARM_JOINTS = (
    "joint1_to_base",
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
)


def _numbers(text, count):
    values = [float(value) for value in (text or "").split()]
    if not values:
        return np.zeros(count, dtype=float)
    if len(values) != count:
        raise ValueError(f"expected {count} numbers, got {len(values)}")
    return np.asarray(values, dtype=float)


def _rpy_matrix(rpy):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray(((1, 0, 0), (0, cr, -sr), (0, sr, cr)), dtype=float)
    ry = np.asarray(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)), dtype=float)
    rz = np.asarray(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)), dtype=float)
    return rz @ ry @ rx


def _axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.asarray(((0, -z, y), (z, 0, -x), (-y, x, 0)), dtype=float)
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _transform(xyz, rotation):
    result = np.eye(4, dtype=float)
    result[:3, :3] = rotation
    result[:3, 3] = xyz
    return result


class UrdfNumericIK:
    def __init__(self, urdf_path, base_link="base", tip_link="gripper_tcp"):
        root = ET.parse(urdf_path).getroot()
        by_child = {}
        for joint in root.findall("joint"):
            child = joint.find("child").attrib["link"]
            by_child[child] = joint
        chain = []
        link = tip_link
        while link != base_link:
            if link not in by_child:
                raise ValueError(f"no URDF joint connects {link} to {base_link}")
            joint = by_child[link]
            chain.append(joint)
            link = joint.find("parent").attrib["link"]
        self.chain = tuple(reversed(chain))
        chain_names = tuple(
            joint.attrib["name"]
            for joint in self.chain
            if joint.attrib.get("type") in ("revolute", "continuous")
        )
        if chain_names != ARM_JOINTS:
            raise ValueError(f"unexpected active chain: {chain_names}")
        lower, upper = [], []
        for joint in self.chain:
            if joint.attrib.get("type") not in ("revolute", "continuous"):
                continue
            limit = joint.find("limit")
            lower.append(float(limit.attrib["lower"]))
            upper.append(float(limit.attrib["upper"]))
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)

    def forward(self, positions):
        values = dict(zip(ARM_JOINTS, positions))
        matrix = np.eye(4, dtype=float)
        for joint in self.chain:
            origin = joint.find("origin")
            xyz = _numbers(origin.attrib.get("xyz") if origin is not None else "", 3)
            rpy = _numbers(origin.attrib.get("rpy") if origin is not None else "", 3)
            matrix = matrix @ _transform(xyz, _rpy_matrix(rpy))
            if joint.attrib.get("type") in ("revolute", "continuous"):
                axis_element = joint.find("axis")
                axis = _numbers(
                    axis_element.attrib.get("xyz") if axis_element is not None else "0 0 1",
                    3,
                )
                matrix = matrix @ _transform(
                    np.zeros(3), _axis_angle(axis, values[joint.attrib["name"]])
                )
        return matrix

    def solve(
        self,
        target_xyz,
        seed,
        position_tolerance=0.005,
        axis_dot_min=0.98,
        tool_x_axis=None,
    ):
        target = np.asarray(target_xyz, dtype=float)
        seed = np.clip(np.asarray(seed, dtype=float), self.lower, self.upper)
        home = np.asarray((0.0, 0.0, 0.0, math.radians(-47.0), 0.0, 0.0))
        candidates = (
            seed,
            np.clip(home, self.lower, self.upper),
            np.clip(np.zeros(6), self.lower, self.upper),
            np.clip((0.5, 0.5, -1.0, -1.0, 0.0, 0.0), self.lower, self.upper),
            np.clip((-0.5, 0.5, -1.0, -1.0, 0.0, 0.0), self.lower, self.upper),
        )
        down = np.asarray((0.0, 0.0, -1.0))
        desired_tool_x = None
        if tool_x_axis is not None:
            desired_tool_x = np.asarray(tool_x_axis, dtype=float)
            desired_tool_x[2] = 0.0
            norm = np.linalg.norm(desired_tool_x)
            if norm < 1e-9:
                raise ValueError("tool_x_axis must have a non-zero horizontal projection")
            desired_tool_x /= norm
        best = None
        for initial in candidates:
            def residual(values):
                pose = self.forward(values)
                position_error = pose[:3, 3] - target
                approach_error = pose[:3, 1] - down
                yaw_error = (
                    pose[:3, 0] - desired_tool_x
                    if desired_tool_x is not None
                    else np.zeros(3)
                )
                regularization = 0.02 * (values - seed)
                # Keep vertical alignment dominant; the yaw target resolves
                # the free wrist rotation without sacrificing reachability.
                return np.concatenate(
                    (
                        20.0 * position_error,
                        approach_error,
                        0.35 * yaw_error,
                        regularization,
                    )
                )

            result = least_squares(
                residual,
                initial,
                bounds=(self.lower, self.upper),
                max_nfev=500,
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-10,
            )
            pose = self.forward(result.x)
            position_error = float(np.linalg.norm(pose[:3, 3] - target))
            axis_dot = float(np.dot(pose[:3, 1], down))
            score = position_error + max(0.0, axis_dot_min - axis_dot)
            if best is None or score < best[0]:
                best = (score, result.x, position_error, axis_dot)
            if position_error <= position_tolerance and axis_dot >= axis_dot_min:
                return tuple(float(value) for value in result.x), position_error, axis_dot
        raise RuntimeError(
            f"NUMERIC_IK_FAILED position_error={best[2]:.4f} axis_dot={best[3]:.4f}"
        )
