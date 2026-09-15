# RoboMaster 六位置马克杯名义参数抓取测试

这是从纯抓取版复制出的独立版本，不会修改正式识别分类代码，也不包含摄像头、YOLO、PyTorch、Ultralytics 或 OpenCV。

## 固定任务

```text
G1 -> 右侧
G2 -> 右侧
G3 -> 右侧
G4 -> 左侧
G5 -> 左侧
G6 -> 左侧
```

每个位置执行：机械臂回中、底盘到取物位、张开夹爪、预抓取、下降并夹取、抬起、回中、底盘回原点、前往左右放置区、松开、回到起点。

## 已填写的马克杯初始参数

参数假设：马克杯直立放置，杯身直径约 75-90 mm，夹爪从杯身侧面夹取。

```text
预抓取: [180, 100] mm
抓取:   [205, -15] mm
抬升:   [180, 100] mm

左/右预放置: [180, 100] mm
左/右放置:   [205, -10] mm
左/右撤回:   [150, 110] mm
```

这些参数符合 RoboMaster SDK 的绝对坐标语义，并在 220 mm 软件边界内，但没有在这台实物机器人上验证。安装高度、机械臂上电零点、夹爪装配和地面高度不同都会造成误差，所以配置保留 `"calibrated": false`。

## 软件检查

```powershell
Set-Location -LiteralPath 'F:\Chatgpt\Automated Desktop Object Sorting and Classification\robomaster_six_position_grasp_mug_test'
$py = 'F:\Anaconda\envs\robomaster_env\python.exe'
& $py -B -m unittest discover -s .\tests -v
& $py -B .\grasp_mug_test.py --validate-config
& $py -B .\grasp_mug_test.py
```

连接机器人 Wi-Fi 后，先只检查 SDK 连接，不发送运动命令：

```powershell
& $py -B .\grasp_mug_test.py --connection-test
```

看到 `RoboMaster connection test: OK (no motion commands sent)` 后，才继续单格测试。

AP模式要求电脑WLAN获得 `192.168.2.x` 地址；`192.168.2.1` 是机器人地址，不能填入 `robot.local_ip`。程序会在调用SDK前检查网段，避免SDK在网络错误时产生难以理解的内部异常。

## 推荐的真机检查顺序

机器人必须连接 AP，运动通道全部清空，并安排人员随时触发急停。第一次先只运行 G1：

```powershell
& $py -B .\grasp_mug_test.py --execute --grid G1 --acknowledge-unverified-arm-points
```

确认机械臂轨迹、夹爪高度、底盘回程和右侧放置都安全后，再依次测试其他格子：

```powershell
& $py -B .\grasp_mug_test.py --execute --grid G2 --acknowledge-unverified-arm-points
& $py -B .\grasp_mug_test.py --execute --grid G3 --acknowledge-unverified-arm-points
& $py -B .\grasp_mug_test.py --execute --grid G4 --acknowledge-unverified-arm-points
& $py -B .\grasp_mug_test.py --execute --grid G5 --acknowledge-unverified-arm-points
& $py -B .\grasp_mug_test.py --execute --grid G6 --acknowledge-unverified-arm-points
```

六格全流程：

```powershell
& $py -B .\grasp_mug_test.py --execute --acknowledge-unverified-arm-points
```

连接成功后程序等待 5 秒才开始运动。每个底盘或机械臂动作最多等待 15 秒，失败后停止后续动作、停止底盘、暂停夹爪并关闭机器人连接。

## 现场修正

如果初始参数偏高或偏低，只修改 `grasp_mug_config.json` 中对应的 `[x_mm, y_mm]`。每次调整 5-10 mm，并始终先运行单格。实物验证九点后，将 `verified_on_this_robot` 和 `calibrated` 改为 `true`，之后执行时可以去掉确认参数。
