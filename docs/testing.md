# 实验三仿真测试

## 静态测试

```powershell
Set-Location 'E:\机器人集成小组项目\实验三'
$env:PYTHONPATH='E:\机器人集成小组项目\实验三\src\mecharm_pick_place'
python -m py_compile .\simulation\isaac\experiment3_scene.py .\simulation\isaac\start_simulation.py
python -m pytest .\tests .\src\mecharm_pick_place\test -q
```

## Isaac 场景

```powershell
& '.\scripts\start_isaac.ps1' -ProjectRoot 'E:\机器人集成小组项目\实验三' -Experiment3 -RebuildScene -BuildSceneOnly
& '.\scripts\start_isaac.ps1' -ProjectRoot 'E:\机器人集成小组项目\实验三' -Experiment3
```

持续运行 Isaac 的窗口保持开启，看到下面两行后，在 ROS 2 容器中使用 `start_isaac:=false` 启动 Launch：

```text
Isaac TCP joint bridge listening on 0.0.0.0:8765
Isaac camera stream listening on 0.0.0.0:8766
```

ROS 端 `virtual_camera_node` 把 8766 的真实渲染 JPEG 发布为 `/camera/image_raw`；`yolo_classifier_node` 加载 `models/pencil_tennis_yolo26n_best.pt`，通过 `/mecharm/vision_request` 接收当前固定格位。

## 验收标准

应存在 `/camera/image_raw`、`/mecharm/detections`、`/mecharm/sorting_status` 和 `/mecharm/sorting_result`。正常场景应检测 6 个物体、至少完成 5 次正确放置并返回 HOME；异常配置应记录 `EMPTY_GRID`、`UNKNOWN_CLASS`、`UNREACHABLE` 或 `SAFE_STOP`。

注意：YOLO 的框中心、框宽高不参与抓取定位。控制器只使用当前请求的固定格位编号，并从 `grid_centers[grid_id]` 读取抓取坐标。

运动验收还应确认：普通移动日志为 `20.0deg/s`，抓取/抬升/末段日志为 `3.0deg/s`；五次平滑曲线和分段策略保持不变；释放动作发生在 `z=0.133 m` 的框上方；两个目标之间没有 `RETURN_HOME` 阶段，最后一个目标完成后最终回 HOME。

完整仿真由用户执行。本目录不删除或修改实验二文件，也不包含 GitHub 同步步骤。
