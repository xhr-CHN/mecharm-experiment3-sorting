"""Evaluate rendered camera views with the experiment-one YOLO model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, pstdev


GRID_TARGETS = {
    "G1": {"class_name": "tennis_ball", "position": [0.1800, 0.0800, 0.0250]},
    "G2": {"class_name": "pencil", "position": [0.0707, 0.1838, 0.0250]},
    "G3": {"class_name": "tennis_ball", "position": [-0.0800, 0.1800, 0.0250]},
    "G4": {"class_name": "pencil", "position": [-0.1800, -0.0800, 0.0250]},
    "G5": {"class_name": "tennis_ball", "position": [-0.0707, -0.1838, 0.0250]},
    "G6": {"class_name": "pencil", "position": [0.0800, -0.1800, 0.0250]},
}


def project_world_point(
    point: list[float],
    camera_position: list[float],
    look_at: list[float],
    width: int,
    height: int,
    focal_length_mm: float,
    horizontal_aperture_mm: float,
) -> tuple[float, float]:
    """Project a known world point using the sweep camera's look-at pose."""
    import numpy as np

    eye = np.asarray(camera_position, dtype=float)
    target = np.asarray(look_at, dtype=float)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    relative = np.asarray(point, dtype=float) - eye
    depth = float(np.dot(relative, forward))
    if depth <= 0.0:
        raise ValueError("world point is behind the camera")
    focal_pixels = float(width) * focal_length_mm / horizontal_aperture_mm
    pixel_x = width / 2.0 + focal_pixels * float(np.dot(relative, right)) / depth
    pixel_y = height / 2.0 - focal_pixels * float(np.dot(relative, up)) / depth
    return pixel_x, pixel_y


def roi_bounds(center_x: float, center_y: float, width: int, height: int, size: int = 100):
    half = size // 2
    left = max(0, int(round(center_x)) - half)
    top = max(0, int(round(center_y)) - half)
    right = min(width, left + size)
    bottom = min(height, top + size)
    left = max(0, right - size)
    top = max(0, bottom - size)
    return left, top, right, bottom


def summarize_view(frame_records: list[dict]) -> dict:
    correct = [record["correct_count"] for record in frame_records]
    pencil_recall = [record["pencil_recall"] for record in frame_records]
    tennis_recall = [record["tennis_recall"] for record in frame_records]
    confidences = [value for record in frame_records for value in record["selected_confidences"]]
    wrong = [record["wrong_count"] for record in frame_records]
    return {
        "view_id": frame_records[0]["view_id"],
        "frame_count": len(frame_records),
        "min_correct": min(correct),
        "mean_correct": mean(correct),
        "correct_std": pstdev(correct),
        "mean_pencil_recall": mean(pencil_recall),
        "mean_tennis_recall": mean(tennis_recall),
        "mean_confidence": mean(confidences) if confidences else 0.0,
        "mean_wrong": mean(wrong),
    }


def ranking_key(summary: dict) -> tuple:
    return (
        summary["min_correct"],
        summary["mean_pencil_recall"],
        summary["mean_correct"],
        summary["mean_confidence"],
        -summary["correct_std"],
        -summary["mean_wrong"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    import cv2
    import yaml
    from ultralytics import YOLO

    args = parse_args()
    result_dir = Path(args.result_dir).resolve()
    model_path = Path(args.model).resolve()
    manifest = json.loads((result_dir / "view_manifest.json").read_text(encoding="utf-8"))
    annotated_root = result_dir / "annotated"
    annotated_root.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    views = {item["view_id"]: item for item in manifest["views"]}
    image_width, image_height = map(int, manifest["resolution"])
    frame_records = []
    for frame in manifest["frames"]:
        image_path = result_dir / frame["path"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"failed to read frame: {image_path}")
        view = views[frame["view_id"]]
        crop_records = []
        crops = []
        for grid_id, target in GRID_TARGETS.items():
            center = project_world_point(
                target["position"],
                view["position"],
                manifest["look_at"],
                image_width,
                image_height,
                float(manifest["focal_length_mm"]),
                float(manifest["horizontal_aperture_mm"]),
            )
            bounds = roi_bounds(*center, image_width, image_height)
            left, top, right, bottom = bounds
            crops.append(image[top:bottom, left:right])
            crop_records.append(
                {"grid_id": grid_id, "expected_class": target["class_name"],
                 "projected_center": list(center), "roi": list(bounds)}
            )
        predictions = model.predict(
            source=crops,
            conf=float(args.confidence),
            imgsz=640,
            device=args.device,
            verbose=False,
        )
        selected_confidences = []
        correct_count = 0
        correct_by_class = {"pencil": 0, "tennis_ball": 0}
        wrong_count = 0
        annotated = image.copy()
        for crop_record, prediction in zip(crop_records, predictions):
            detections = []
            if prediction.boxes is not None:
                for box in prediction.boxes:
                    class_index = int(box.cls.item())
                    detections.append(
                        {"class_name": str(prediction.names[class_index]),
                         "confidence": float(box.conf.item()),
                         "xyxy": [float(value) for value in box.xyxy[0].tolist()]}
                    )
            best = max(detections, key=lambda item: item["confidence"], default=None)
            crop_record["detections"] = detections
            crop_record["selected"] = best
            correct = best is not None and best["class_name"] == crop_record["expected_class"]
            crop_record["correct"] = correct
            if correct:
                correct_count += 1
                correct_by_class[crop_record["expected_class"]] += 1
                selected_confidences.append(best["confidence"])
            elif best is not None:
                wrong_count += 1
            left, top, right, bottom = crop_record["roi"]
            color = (40, 210, 40) if correct else (20, 20, 230)
            cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)
            decision = "none" if best is None else f"{best['class_name']} {best['confidence']:.2f}"
            cv2.putText(
                annotated, f"{crop_record['grid_id']} {decision}",
                (left, max(16, top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                cv2.LINE_AA,
            )
        record = {
            **frame,
            "grids": crop_records,
            "correct_count": correct_count,
            "pencil_recall": correct_by_class["pencil"] / 3.0,
            "tennis_recall": correct_by_class["tennis_ball"] / 3.0,
            "wrong_count": wrong_count,
            "selected_confidences": selected_confidences,
        }
        frame_records.append(record)
        annotated_path = annotated_root / frame["view_id"] / Path(frame["path"]).name
        annotated_path.parent.mkdir(parents=True, exist_ok=True)
        success, encoded = cv2.imencode(".png", annotated)
        if not success:
            raise RuntimeError(f"failed to encode annotated frame: {annotated_path}")
        encoded.tofile(str(annotated_path))
        print(f"EVALUATED {frame['view_id']} {frame['frame_index']} correct={correct_count}")

    summaries = []
    for view in manifest["views"]:
        records = [record for record in frame_records if record["view_id"] == view["view_id"]]
        summaries.append({**view, **summarize_view(records)})
    summaries.sort(key=ranking_key, reverse=True)
    winner = summaries[0]
    (result_dir / "detections.json").write_text(
        json.dumps(frame_records, indent=2), encoding="utf-8"
    )
    with (result_dir / "view_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    recommendation = {
        "view_id": winner["view_id"],
        "position": winner["position"],
        "look_at": manifest["look_at"],
        "tilt_deg": winner["tilt_deg"],
        "azimuth_deg": winner["azimuth_deg"],
        "focal_length_mm": manifest["focal_length_mm"],
        "resolution": manifest["resolution"],
        "metrics": {key: winner[key] for key in (
            "min_correct", "mean_correct", "correct_std", "mean_pencil_recall",
            "mean_tennis_recall", "mean_confidence", "mean_wrong"
        )},
    }
    (result_dir / "recommended_camera.yaml").write_text(
        yaml.safe_dump(recommendation, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    (result_dir / "model_sha256.txt").write_text(
        f"{digest}  {model_path.name}\n", encoding="utf-8"
    )
    def grid_confidences(view_id: str) -> dict[str, float]:
        records = [record for record in frame_records if record["view_id"] == view_id]
        result = {}
        for grid_id in GRID_TARGETS:
            values = []
            for record in records:
                grid = next(item for item in record["grids"] if item["grid_id"] == grid_id)
                values.append(grid["selected"]["confidence"] if grid["correct"] else 0.0)
            result[grid_id] = mean(values)
        return result

    winner_grid = grid_confidences(winner["view_id"])
    vertical_grid = grid_confidences("vertical")
    lines = [
        "# 实验三摄像头视角测试报告",
        "",
        "## 测试条件",
        "",
        f"- 候选视角：{len(summaries)} 个；每个视角 5 帧，共 {len(frame_records)} 帧",
        "- 分辨率：848x480；焦距：18 mm；相机高度：1.10 m",
        f"- YOLO 置信度阈值：{float(args.confidence):.2f}；推理尺寸：640",
        f"- 模型：`{model_path.name}`",
        f"- 模型 SHA-256：`{digest}`",
        "- 方法：投影六个已知世界坐标，每个点裁剪 100x100 ROI；YOLO 只判断 ROI 内类别，",
        "  不参与机械臂抓取中心计算。",
        "",
        "## 结论",
        "",
        f"- 最佳视角：`{winner['view_id']}`（向北倾斜 10°）",
        f"- 相机位置：`{winner['position']}`；朝向圆心 `[0, 0, 0]`",
        f"- 最差单帧正确数：`{winner['min_correct']}/6`",
        f"- 铅笔召回率：`{winner['mean_pencil_recall']:.3f}`",
        f"- 网球召回率：`{winner['mean_tennis_recall']:.3f}`",
        f"- 六物体平均置信度：`{winner['mean_confidence']:.3f}`",
        "",
        "## 最佳视角与原垂直俯拍对比",
        "",
        "| 固定点 | 类别 | 垂直俯拍平均置信度 | 10°向北平均置信度 |",
        "|---|---|---:|---:|",
    ]
    for grid_id, target in GRID_TARGETS.items():
        lines.append(
            f"| {grid_id} | {target['class_name']} | {vertical_grid[grid_id]:.3f} "
            f"| {winner_grid[grid_id]:.3f} |"
        )
    lines.extend([
        "",
        "## 全部视角排名",
        "",
        "| 排名 | 视角 | 最差正确数 | 平均正确数 | 铅笔召回率 | 网球召回率 | 平均置信度 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ])
    for index, summary in enumerate(summaries, 1):
        lines.append(
            f"| {index} | {summary['view_id']} | {summary['min_correct']}/6 "
            f"| {summary['mean_correct']:.1f}/6 | {summary['mean_pencil_recall']:.3f} "
            f"| {summary['mean_tennis_recall']:.3f} | {summary['mean_confidence']:.3f} |"
        )
    lines.extend([
        "",
        "原始图像、逐 ROI 检测结果、标注图和 CSV 排名均保存在本目录。",
    ])
    (result_dir / "camera_view_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"RECOMMENDED_VIEW={winner['view_id']}")


if __name__ == "__main__":
    main()
