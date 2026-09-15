# RoboMaster 六位置纯抓取测试

这是一个全新的独立测试目录，不会导入或调用原分类程序。程序中没有摄像头、YOLO、PyTorch、Ultralytics、OpenCV、ROI、类别判断或模型权重，只验证 RoboMaster 底盘、机械臂和夹爪的六位置动作。

## 固定动作

```text
G1 -> 右侧
G2 -> 右侧
G3 -> 右侧
G4 -> 左侧
G5 -> 左侧
G6 -> 左侧
```

每个位置都执行：回中机械臂、底盘到取物位、张开夹爪、预抓取、下降/前伸、闭合夹爪、抬起、回中、底盘回原点、转向对应分类区、放下、回到原始姿态。

格子中有没有物体都不影响动作。程序不会读取夹爪状态，也不会判断是否抓取成功。

## 文件

- `grasp_only_test.py`：唯一运行入口；
- `grasp_only_config.json`：六格路线、左右路线及机械臂标定点；
- `ground_layout.csv`：六格坐标；
- `requirements.txt`：只有 `robomaster`；
- `tests/test_grasp_only.py`：无视觉依赖与六格动作测试。

## 当前安全锁

配置默认是：

```json
"calibrated": false
```

在真机执行前，必须填写以下 9 个机械臂点：

```text
grasp_test.pickup_arm_profile.arm_pregrasp_mm
grasp_test.pickup_arm_profile.arm_grasp_mm
grasp_test.pickup_arm_profile.arm_lift_mm

destinations.left.arm_predrop_mm
destinations.left.arm_drop_mm
destinations.left.arm_retract_mm

destinations.right.arm_predrop_mm
destinations.right.arm_drop_mm
destinations.right.arm_retract_mm
```

坐标格式为 `[x_mm, y_mm]`，相对机械臂上电零点。完成逐点低速验证后，才把 `calibrated` 改成 `true`。

## 运行

安装唯一依赖：

```bash
python3 -m pip install robomaster
```

检查配置，不连接机器人：

```bash
python3 grasp_only_test.py --validate-config
```

显示六格计划，不连接机器人：

```bash
python3 grasp_only_test.py
```

执行六格真实动作：

```bash
python3 grasp_only_test.py --execute
```

## 安全说明

- 第一次运行时六格全部保持为空，只验证运动轨迹。
- 识别和避障已经完全移除，运行前必须人工清空六条底盘通道。
- 后排路线会经过对应前排通道；如果之后放置物体测试，必须确认前排物体已经被取走。
- 麦克纳姆轮开环运动存在打滑误差，应在地面标记机器人原点并逐格核对回零。
- 任意 SDK 动作失败或超过 15 秒，程序立即停止后续动作。
- 现场必须安排安全员并保留物理急停。
