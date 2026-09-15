# 实验三：桌面物体自动分类整理仿真

英文项目说明见 [README_EN.md](README_EN.md)。

本目录是实验三的独立副本，使用 Elephant Robotics mechArm 270 Pi、自适应夹爪、网球和绿色木制铅笔完成环形桌面分类。

场景中机械臂位于圆心，6 个物体和 2 个开放式分类框位于同一圆周上。分类框按类别自动选择，不需要手动指定网格、类别或放置位置。

## 目录

- `config/`：正常场景和异常场景配置
- `simulation/isaac/`：Isaac Sim 场景、TCP 桥和抓取监视器
- `simulation/urdf/`：机械臂与自适应夹爪 URDF
- `src/mecharm_pick_place/`：ROS 2 节点、Launch 和单元测试
- `tests/`：实验三配置与 Launch 合约测试
- `results/experiment3/`：本地运行结果
- `models/pencil_tennis_yolo26n_best.pt`：从实验一复制的 YOLO26n 模型
- `src/real_robot/`：整理后的 RoboMaster EP 真机程序与分阶段测试代码
- `config/real_robot/`：真机配置、地面布局和依赖清单
- `models/real_robot/`：真机 YOLO 模型
- `results/real_robot/`：真机运行记录、图像视频证据和元数据
- `report/real_robot/`：真机汇报材料源文件
- `report/presentation/`：实验三答辩 PPT 粗稿、最终稿和演讲检查材料
- `report/figures/github_commits.png`：GitHub 提交记录截图
- `results/experiment3/evidence/videos/`：仿真完整运行视频等展示证据
- `results/experiment3/camera_view_sweep/analysis/`：相机视角测试表格与图表
- `docs/real_robot/README.md`：真机资料归档说明

## 场景生成

```powershell
Set-Location 'E:\机器人集成小组项目\实验三'
& '.\scripts\start_isaac.ps1' -ProjectRoot 'E:\机器人集成小组项目\实验三' -Experiment3 -RebuildScene -BuildSceneOnly
```

## ROS 2 启动

先持续运行 Isaac，再在 Docker 容器内构建并启动：

```powershell
& '.\scripts\start_isaac.ps1' -ProjectRoot 'E:\机器人集成小组项目\实验三' -Experiment3
docker compose exec moveit bash -lc "source /opt/ros/humble/setup.bash && colcon --log-base /opt/mecharm_ws/log build --base-paths /workspace/mecharm_exp3/simulation/urdf/mycobot_description /workspace/mecharm_exp3/src --build-base /opt/mecharm_ws/build --install-base /opt/mecharm_ws/install --symlink-install && source /opt/mecharm_ws/install/setup.bash && ros2 launch mecharm_pick_place experiment3_sorting.launch.py project_root:=/workspace/mecharm_exp3 config:=/workspace/mecharm_exp3/config/experiment3_sorting.yaml start_isaac:=false"
```

Isaac 会在 TCP `8766` 提供 `/World/experiment3/top_camera` 的真实渲染图像；ROS 节点 `virtual_camera_node` 将其发布为 `/camera/image_raw`。`yolo_classifier_node` 使用实验一模型，启动后缓存连续 6 帧，对 G1–G6 的固定 ROI 分别分类，并要求至少 4/6 帧投票一致；分类结果锁存后，抓取阶段不再重复请求视觉。

控制器启动阶段只发送一次 `ALL` 批量视觉请求，收到六个格位的类别结果后按 `G1` 到 `G6` 顺序执行抓取。YOLO 返回的边界框只用于检测记录和可视化，抓取位置始终取自 `config/experiment3_sorting.yaml` 的 `grid_centers`，不会使用 YOLO 框中心。

动作流程沿用实验二的分级速度策略和五次平滑曲线；实验三当前将普通移动设为 `20°/s`，抓取、抬升和末段动作设为 `3°/s`。夹住物体后抬至 `z=0.133 m`，移动到分类框上方直接松爪自然落下；每批任务只在开始时回 HOME，目标之间不回 HOME，全部完成后最终回 HOME。

详细验收步骤见 `docs/testing.md`。本目录暂不自动同步 GitHub；后续同步时只提交实验三目录。实验二目录及其环境保持不变。

## 真机验证资料

真机部分已按项目目录重新归档，不再依赖 `real_validation` 作为项目源目录。整理后的入口见 `docs/real_robot/README.md`。`real_validation/20260915_v2/raw_package/` 仅保留为本地原始资料备份，并通过 `.gitignore` 排除，避免向 GitHub 重复上传整包压缩资料。
