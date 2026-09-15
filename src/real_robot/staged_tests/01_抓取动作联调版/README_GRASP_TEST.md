# RoboMaster 六格抓取动作测试版

这是从正式分类程序复制出来的独立测试版。原目录 `robomaster_sorter` 未被修改；本测试只使用 `robomaster_grasp_motion_test` 中的文件。

本机应使用 Conda 环境 `robomaster_env`。该环境安装的是 CPU 版 PyTorch，因此测试专用配置已设置为 `"device": "cpu"`；不要直接用默认的 Python 3.13 或 `desk-yolo` 环境启动本程序。

如果终端下载依赖时遇到 TLS/代理拦截，请按 `MANUAL_OFFLINE_INSTALL.md` 下载两个剩余 wheel 并离线安装。

## 测试目的

保留 YOLO 六格识别和日志，但识别结果不控制动作。程序固定执行：

| 顺序 | 格子 | 放置方向 |
|---:|---|---|
| 1 | G1 | 右侧 |
| 2 | G2 | 右侧 |
| 3 | G3 | 右侧 |
| 4 | G4 | 左侧 |
| 5 | G5 | 左侧 |
| 6 | G6 | 左侧 |

每个格子无论被识别为杯子、瓶子、鼠标、空格或不确定，都会执行完整动作：

1. 在原始位置识别六格并记录当前格状态；
2. 底盘移动到当前格；
3. 夹爪张开；
4. 机械臂预抓取、下降/前伸；
5. 夹爪闭合；
6. 机械臂抬起并回中；
7. 底盘回原点；
8. G1–G3 右转并到右侧放置，G4–G6 左转并到左侧放置；
9. 回到原始拍摄位，继续下一格；
10. 六格完成后再拍摄一次并结束。

因为格子可能为空，程序不会调用夹持状态验证，也不会要求源格从“有物体”变为“空”。这只是动作链路测试，不应作为识别正确率或分类成功率证据。

## 使用文件

- `grasp_motion_test.py`：本测试的入口程序；
- `grasp_test_config.json`：测试专用配置；
- `desktop_sorter.py`：复制来的相机、检测、SDK 控制基础模块；
- `tests/test_grasp_motion_test.py`：固定顺序和空抓逻辑测试。

不要使用此目录中的 `sorter_config.json` 启动测试；应使用默认的 `grasp_test_config.json`。

## 运行前标定

测试配置默认保留：

```json
"calibrated": false
```

以下机械臂点仍是 `null`，必须根据真机上电零点填写：

```text
grasp_test.pickup_arm_profile.arm_pregrasp_mm
grasp_test.pickup_arm_profile.arm_grasp_mm
grasp_test.pickup_arm_profile.arm_lift_mm

bins.cup.arm_predrop_mm
bins.cup.arm_drop_mm
bins.cup.arm_retract_mm

bins.bottle.arm_predrop_mm
bins.bottle.arm_drop_mm
bins.bottle.arm_retract_mm
```

这里只有一套通用抓取点，六格通过移动底盘把目标位置送到同一夹爪中心线。左侧沿用 `bins.cup` 的左转路线，右侧沿用 `bins.bottle` 的右转路线；这里的名称只是复用路线，不表示根据识别类别分类。

填好机械臂点、逐条空载验证底盘路线后，才可把 `grasp_test_config.json` 中的 `calibrated` 改为 `true`。

## 运行命令

仅验证配置，不连接机器人：

```bash
conda run -n robomaster_env python grasp_motion_test.py --validate-config
```

只识别一次并显示计划，不移动机器人：

```bash
conda run -n robomaster_env python grasp_motion_test.py
```

首次真机测试只执行第一个格子 G1（强烈建议先空载）：

```bash
conda run -n robomaster_env python grasp_motion_test.py --execute --max-actions 1
```

确认 G1 的去程、抓取、右侧投放和回零全部正确后，执行固定六格动作：

```bash
conda run -n robomaster_env python grasp_motion_test.py --execute
```

无显示器运行：

```bash
conda run -n robomaster_env python grasp_motion_test.py --execute --no-display
```

`--max-actions N` 可只执行固定序列的前 N 格，N 必须为 1–6。例如 `--max-actions 3` 只执行 G1、G2、G3。

如果 `conda run` 因 Windows 终端中文编码异常，直接调用环境解释器，效果相同且断网可用：

```powershell
& 'F:\Anaconda\envs\robomaster_env\python.exe' .\grasp_motion_test.py --validate-config
& 'F:\Anaconda\envs\robomaster_env\python.exe' .\grasp_motion_test.py --execute --max-actions 1
```

## 必须注意

- 识别不会阻止动作。运行前必须人工确认六条底盘通道没有人、线缆、散落物或其他障碍物。
- 即使识别到鼠标或发现格子为空，程序也会继续闭合夹爪并完成投放路线。
- 建议第一次不放任何物体，全程一人手持物理急停；确认六条路线均能回零后，再放一个轻质空物体测试。
- 只要任一步 SDK 动作失败或超时，程序就停止底盘并终止后续格子，避免在未知姿态下继续执行。
- 麦克纳姆轮属于开环位移，地面打滑会导致累计回零误差；每执行一个格子都要观察起点标记。
