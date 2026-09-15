# Jetson 三分类验证包

该包识别以下三类目标：

- `cup`
- `mouse`
- `bottle`

文件：

- `best.pt`：三分类 YOLO11n 权重
- `run_model.py`：Jetson 摄像头推理、结果保存和统计脚本

在 Jetson 中进入本目录后运行：

```bash
python3 run_model.py --source 0 --device 0 --conf 0.5 --save
```

在检测窗口按 `q` 结束。结果保存在 `results/`：

- `realtime_detected.mp4`：带检测框、类别、置信度和 FPS 的视频
- `detections.csv`：逐框检测记录
- `run_summary.json`：帧数、速度和各类别检测统计

如需 ROS2 发布：

```bash
source /opt/ros/humble/setup.bash
python3 run_model.py --source 0 --device 0 --conf 0.5 --save --ros2
```

默认节点为 `/yolo_detector`，默认话题为 `/detections`。
