"""RoboMaster six-position mug-grasp test with no vision dependencies."""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "grasp_mug_config.json"
EXPECTED_SEQUENCE = [
    ("G1", "right"),
    ("G2", "right"),
    ("G3", "right"),
    ("G4", "left"),
    ("G5", "left"),
    ("G6", "left"),
]
ROBOT_AP_IP = "192.168.2.1"
ROBOT_AP_NETWORK = ipaddress.ip_network("192.168.2.0/24")


class ConfigError(ValueError):
    """The grasp-only configuration is incomplete or unsafe."""


class MotionError(RuntimeError):
    """A RoboMaster command failed or timed out."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def check_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"{label} must be finite")
    return number


def validate_arm_point(point: Any, label: str, cfg: dict[str, Any]) -> None:
    if not isinstance(point, list) or len(point) != 2:
        raise ConfigError(f"{label} must be [x_mm, y_mm]")
    x_mm = check_number(point[0], f"{label}[0]")
    y_mm = check_number(point[1], f"{label}[1]")
    limit = check_number(
        cfg["safety"]["max_abs_arm_coordinate_mm"],
        "safety.max_abs_arm_coordinate_mm",
    )
    if abs(x_mm) > limit or abs(y_mm) > limit:
        raise ConfigError(f"{label} exceeds safety.max_abs_arm_coordinate_mm")


def validate_route(route: Any, label: str, cfg: dict[str, Any]) -> None:
    if not isinstance(route, list):
        raise ConfigError(f"{label} must be a list")
    safety = cfg["safety"]
    for index, step in enumerate(route):
        if not isinstance(step, dict):
            raise ConfigError(f"{label}[{index}] must be an object")
        x_m = check_number(step.get("x_m", 0), f"{label}[{index}].x_m")
        y_m = check_number(step.get("y_m", 0), f"{label}[{index}].y_m")
        z_deg = check_number(step.get("z_deg", 0), f"{label}[{index}].z_deg")
        xy_speed = check_number(
            step.get("xy_speed_mps", safety["default_xy_speed_mps"]),
            f"{label}[{index}].xy_speed_mps",
        )
        z_speed = check_number(
            step.get("z_speed_dps", safety["default_z_speed_dps"]),
            f"{label}[{index}].z_speed_dps",
        )
        if abs(x_m) > safety["max_step_m"] or abs(y_m) > safety["max_step_m"]:
            raise ConfigError(f"{label}[{index}] exceeds safety.max_step_m")
        if abs(z_deg) > safety["max_rotation_deg"]:
            raise ConfigError(f"{label}[{index}] exceeds safety.max_rotation_deg")
        if not 0.5 <= xy_speed <= 2.0:
            raise ConfigError(f"{label}[{index}] xy speed must be in [0.5, 2.0]")
        if not 10.0 <= z_speed <= 540.0:
            raise ConfigError(f"{label}[{index}] z speed must be in [10, 540]")
        if (x_m or y_m or z_deg) and safety["chassis_motion_enabled"] is not True:
            raise ConfigError(f"{label}[{index}] requests disabled chassis motion")


def validate_config(
    cfg: dict[str, Any],
    require_arm_points: bool = False,
    require_calibrated: bool = False,
) -> None:
    required = {"robot", "safety", "layout", "grasp_test", "grids", "destinations"}
    missing = required - set(cfg)
    if missing:
        raise ConfigError("Missing sections: " + ", ".join(sorted(missing)))
    if cfg["robot"].get("conn_type") not in {"ap", "sta", "rndis"}:
        raise ConfigError("robot.conn_type must be ap, sta, or rndis")
    if cfg["robot"].get("proto_type", "udp") not in {"udp", "tcp"}:
        raise ConfigError("robot.proto_type must be udp or tcp")
    startup_delay = check_number(
        cfg["safety"].get("startup_delay_s", 0), "safety.startup_delay_s"
    )
    if not 0 <= startup_delay <= 30:
        raise ConfigError("safety.startup_delay_s must be in [0, 30]")

    grids = cfg["grids"]
    if not isinstance(grids, list) or [grid.get("id") for grid in grids] != [
        "G1", "G2", "G3", "G4", "G5", "G6"
    ]:
        raise ConfigError("grids must be ordered exactly G1 through G6")
    for grid in grids:
        center = grid.get("ground_center_m")
        if not isinstance(center, dict):
            raise ConfigError(f"{grid['id']}.ground_center_m must be an object")
        check_number(center.get("x_forward_m"), f"{grid['id']}.x_forward_m")
        check_number(center.get("y_right_m"), f"{grid['id']}.y_right_m")
        validate_route(grid.get("chassis_to_pick"), f"{grid['id']}.chassis_to_pick", cfg)
        validate_route(
            grid.get("chassis_home_from_pick"),
            f"{grid['id']}.chassis_home_from_pick",
            cfg,
        )

    sequence = cfg["grasp_test"].get("fixed_sequence")
    actual = (
        [(step.get("grid_id"), step.get("destination")) for step in sequence]
        if isinstance(sequence, list)
        else []
    )
    if actual != EXPECTED_SEQUENCE:
        raise ConfigError("fixed_sequence must send G1-G3 right and G4-G6 left")

    destinations = cfg["destinations"]
    if set(destinations) != {"left", "right"}:
        raise ConfigError("destinations must contain exactly left and right")
    for name, destination in destinations.items():
        validate_route(
            destination.get("chassis_to_destination"),
            f"destinations.{name}.chassis_to_destination",
            cfg,
        )
        validate_route(
            destination.get("chassis_home_from_destination"),
            f"destinations.{name}.chassis_home_from_destination",
            cfg,
        )

    if require_calibrated and cfg.get("calibrated") is not True:
        raise ConfigError(
            'Real motion is blocked until grasp_mug_config.json contains '
            '"calibrated": true, or the command explicitly acknowledges the '
            'unverified nominal mug points'
        )
    pickup = cfg["grasp_test"].get("pickup_arm_profile", {})
    for key in ("arm_pregrasp_mm", "arm_grasp_mm", "arm_lift_mm"):
        validate_arm_point(pickup.get(key), f"grasp_test.pickup_arm_profile.{key}", cfg)
    for name, destination in destinations.items():
        for key in ("arm_predrop_mm", "arm_drop_mm", "arm_retract_mm"):
            validate_arm_point(destination.get(key), f"destinations.{name}.{key}", cfg)


def select_sequence(
    sequence: list[dict[str, str]], grid_id: str | None
) -> list[dict[str, str]]:
    if grid_id is None:
        return sequence
    return [step for step in sequence if step["grid_id"] == grid_id]


def resolve_sdk_connection_types(
    robot_cfg: dict[str, Any], conn_module: Any
) -> tuple[str, str]:
    conn_types = {
        "ap": conn_module.CONNECTION_WIFI_AP,
        "sta": conn_module.CONNECTION_WIFI_STA,
        "rndis": conn_module.CONNECTION_USB_RNDIS,
    }
    proto_types = {
        "udp": conn_module.CONNECTION_PROTO_UDP,
        "tcp": conn_module.CONNECTION_PROTO_TCP,
    }
    return (
        conn_types[robot_cfg["conn_type"]],
        proto_types[robot_cfg.get("proto_type", "udp")],
    )


def validate_ap_local_ip(local_ip: str) -> str:
    try:
        address = ipaddress.ip_address(local_ip)
    except ValueError as error:
        raise MotionError(f"Invalid local IP address: {local_ip!r}") from error
    if (
        address not in ROBOT_AP_NETWORK
        or address == ROBOT_AP_NETWORK.network_address
        or address == ROBOT_AP_NETWORK.broadcast_address
        or str(address) == ROBOT_AP_IP
    ):
        raise MotionError(
            "AP mode requires the PC Wi-Fi address to be 192.168.2.x "
            f"(not {ROBOT_AP_IP}); current route uses {local_ip}. "
            "Connect Windows to the RoboMaster Wi-Fi first."
        )
    return str(address)


def detect_ap_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect((ROBOT_AP_IP, 9))
            local_ip = probe.getsockname()[0]
    except OSError as error:
        raise MotionError(
            "No IPv4 route to the RoboMaster AP. Connect Windows to the "
            "RoboMaster Wi-Fi first."
        ) from error
    return validate_ap_local_ip(local_ip)


class EventLog:
    def __init__(self, root: Path) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir = root / f"run_{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.path = self.run_dir / "grasp_only_log.jsonl"
        self.handle = self.path.open("w", encoding="utf-8")

    def write(self, event: str, **details: Any) -> None:
        record = {"timestamp": utc_now(), "event": event, **details}
        line = json.dumps(record, ensure_ascii=False)
        self.handle.write(line + "\n")
        self.handle.flush()
        print(line)

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.close()


class RoboMasterController:
    def __init__(self, cfg: dict[str, Any]) -> None:
        import robomaster
        from robomaster import conn, robot

        local_ip = cfg["robot"].get("local_ip")
        if cfg["robot"]["conn_type"] == "ap":
            local_ip = validate_ap_local_ip(str(local_ip)) if local_ip else detect_ap_local_ip()
        if local_ip:
            robomaster.config.LOCAL_IP_STR = str(local_ip)
        self.cfg = cfg
        self.robot = robot.Robot()
        self.chassis = None
        self.arm = None
        self.gripper = None
        conn_type, proto_type = resolve_sdk_connection_types(cfg["robot"], conn)
        kwargs: dict[str, Any] = {
            "conn_type": conn_type,
            "proto_type": proto_type,
        }
        if cfg["robot"].get("sn"):
            kwargs["sn"] = cfg["robot"]["sn"]
        try:
            initialized = self.robot.initialize(**kwargs)
            if initialized is False:
                raise MotionError("RoboMaster initialization returned False")
            self.chassis = self.robot.chassis
            self.arm = self.robot.robotic_arm
            self.gripper = self.robot.gripper
        except Exception as error:
            try:
                self.robot.close()
            except Exception:
                pass
            raise MotionError(
                "RoboMaster SDK initialization failed. Check that the robot is "
                "powered on, set to AP/direct mode, and not connected to another "
                "controller."
            ) from error

    def wait_action(self, action: Any, label: str) -> None:
        timeout = float(self.cfg["safety"]["action_timeout_s"])
        completed = action.wait_for_completed(timeout=timeout)
        if not completed or not bool(getattr(action, "has_succeeded", False)):
            reason = getattr(action, "failure_reason", None)
            raise MotionError(f"{label} failed or timed out; reason={reason!r}")

    def run_route(self, route: list[dict[str, Any]], label: str) -> None:
        assert self.chassis is not None
        safety = self.cfg["safety"]
        for index, step in enumerate(route):
            x_m = float(step.get("x_m", 0))
            y_m = float(step.get("y_m", 0))
            z_deg = float(step.get("z_deg", 0))
            if x_m == y_m == z_deg == 0:
                continue
            action = self.chassis.move(
                x=x_m,
                y=y_m,
                z=z_deg,
                xy_speed=float(
                    step.get("xy_speed_mps", safety["default_xy_speed_mps"])
                ),
                z_speed=float(
                    step.get("z_speed_dps", safety["default_z_speed_dps"])
                ),
            )
            self.wait_action(action, f"{label} step {index + 1}")

    def recenter_arm(self) -> None:
        assert self.arm is not None
        self.wait_action(self.arm.recenter(), "arm recenter")

    def move_arm(self, point: list[float], label: str) -> None:
        assert self.arm is not None
        self.wait_action(self.arm.moveto(x=point[0], y=point[1]), label)

    def open_gripper(self) -> None:
        assert self.gripper is not None
        power = int(self.cfg["safety"]["gripper_open_power"])
        if self.gripper.open(power=power) is False:
            raise MotionError("gripper open command was rejected")
        time.sleep(float(self.cfg["safety"]["gripper_open_seconds"]))
        if self.gripper.pause() is False:
            raise MotionError("gripper pause after open was rejected")
        time.sleep(0.2)

    def close_gripper(self) -> None:
        assert self.gripper is not None
        power = int(self.cfg["safety"]["gripper_close_power"])
        if self.gripper.close(power=power) is False:
            raise MotionError("gripper close command was rejected")
        time.sleep(float(self.cfg["safety"]["gripper_close_seconds"]))
        if self.gripper.pause() is False:
            raise MotionError("gripper pause after close was rejected")
        time.sleep(0.4)

    def stop_safely(self) -> None:
        if self.chassis is not None:
            try:
                self.chassis.drive_speed(x=0, y=0, z=0, timeout=1)
            except Exception:
                pass
        if self.gripper is not None:
            try:
                self.gripper.pause()
            except Exception:
                pass

    def close(self) -> None:
        self.stop_safely()
        self.robot.close()

    def close_connection_only(self) -> None:
        self.robot.close()


def execute_position(
    controller: RoboMasterController,
    grid: dict[str, Any],
    destination_name: str,
    cfg: dict[str, Any],
    event_log: EventLog,
) -> None:
    pickup = cfg["grasp_test"]["pickup_arm_profile"]
    destination = cfg["destinations"][destination_name]
    event_log.write(
        "position_started",
        grid_id=grid["id"],
        destination=destination_name,
        ground_center_m=grid["ground_center_m"],
    )

    controller.recenter_arm()
    controller.run_route(grid["chassis_to_pick"], f"route to {grid['id']}")
    controller.open_gripper()
    controller.move_arm(pickup["arm_pregrasp_mm"], f"{grid['id']} pregrasp")
    controller.move_arm(pickup["arm_grasp_mm"], f"{grid['id']} grasp")
    controller.close_gripper()
    controller.move_arm(pickup["arm_lift_mm"], f"{grid['id']} lift")
    controller.recenter_arm()
    controller.run_route(
        grid["chassis_home_from_pick"], f"return from {grid['id']}"
    )

    controller.run_route(
        destination["chassis_to_destination"],
        f"route to {destination_name}",
    )
    controller.move_arm(
        destination["arm_predrop_mm"], f"{destination_name} predrop"
    )
    controller.move_arm(destination["arm_drop_mm"], f"{destination_name} drop")
    controller.open_gripper()
    controller.move_arm(
        destination["arm_retract_mm"], f"{destination_name} retract"
    )
    controller.recenter_arm()
    controller.run_route(
        destination["chassis_home_from_destination"],
        f"return from {destination_name}",
    )
    event_log.write(
        "position_completed",
        grid_id=grid["id"],
        destination=destination_name,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute", action="store_true", help="Run real robot motion")
    parser.add_argument("--validate-config", action="store_true")
    parser.add_argument(
        "--connection-test",
        action="store_true",
        help="Connect to RoboMaster and disconnect without motion commands",
    )
    parser.add_argument(
        "--grid",
        choices=[grid_id for grid_id, _ in EXPECTED_SEQUENCE],
        help="Run only one grid; omit to run G1 through G6",
    )
    parser.add_argument(
        "--acknowledge-unverified-arm-points",
        action="store_true",
        help=(
            "Allow real motion with the supplied nominal mug points. "
            "They have not been verified on this physical robot."
        ),
    )
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.connection_test and args.execute:
        raise ConfigError("--connection-test and --execute cannot be used together")
    config_path = args.config.resolve()
    cfg = load_config(config_path)
    using_unverified_points = (
        args.execute
        and args.acknowledge_unverified_arm_points
        and cfg.get("calibrated") is not True
    )
    validate_config(
        cfg,
        require_arm_points=args.execute,
        require_calibrated=args.execute and not using_unverified_points,
    )
    sequence = select_sequence(cfg["grasp_test"]["fixed_sequence"], args.grid)

    if args.connection_test:
        controller: RoboMasterController | None = None
        try:
            controller = RoboMasterController(cfg)
            print("RoboMaster connection test: OK (no motion commands sent)")
        finally:
            if controller is not None:
                controller.close_connection_only()
        return

    if args.validate_config:
        print(
            f"Configuration is valid for "
            f"{'execution' if args.execute else 'dry-run'}: {config_path}"
        )
        if using_unverified_points:
            print("WARNING: nominal mug arm points are acknowledged but not calibrated")
        return
    if not args.execute:
        print("No robot motion. Planned fixed sequence:")
        for index, step in enumerate(sequence, start=1):
            print(f"{index}. {step['grid_id']} -> {step['destination']}")
        return

    if using_unverified_points:
        print(
            "WARNING: using nominal mug arm points that have not been verified "
            "on this physical robot"
        )

    event_log = EventLog(args.log_dir.resolve())
    controller: RoboMasterController | None = None
    try:
        controller = RoboMasterController(cfg)
        grids = {grid["id"]: grid for grid in cfg["grids"]}
        event_log.write("grasp_only_test_started", fixed_sequence=sequence)
        startup_delay = float(cfg["safety"].get("startup_delay_s", 0))
        if startup_delay > 0:
            print(f"Robot motion starts in {startup_delay:.1f} seconds...")
            time.sleep(startup_delay)
        for index, step in enumerate(sequence, start=1):
            event_log.write("sequence_step", index=index, total=len(sequence))
            execute_position(
                controller,
                grids[step["grid_id"]],
                step["destination"],
                cfg,
                event_log,
            )
        event_log.write("grasp_only_test_completed", positions_completed=len(sequence))
    except Exception as error:
        event_log.write("grasp_only_test_exception", error=repr(error))
        if controller is not None:
            controller.stop_safely()
        raise
    finally:
        if controller is not None:
            controller.close()
        event_log.close()


if __name__ == "__main__":
    try:
        main()
    except (ConfigError, MotionError) as error:
        raise SystemExit(f"ERROR: {error}") from None
