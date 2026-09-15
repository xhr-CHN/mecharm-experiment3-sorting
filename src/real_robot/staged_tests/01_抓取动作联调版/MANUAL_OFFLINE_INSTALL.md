# 两个剩余依赖的手动下载与离线安装

当前 `robomaster_env` 已能导入 RoboMaster SDK、PyTorch、OpenCV 和 Ultralytics，也已成功加载 `best.pt` 并完成本地图像推理。`pip check` 仍报告缺少两个依赖：

- `fsspec`（PyTorch 声明的依赖）；
- `ultralytics-thop>=2.0.0`（Ultralytics 声明的依赖）。

由于本机终端连接 PyPI 时出现 TLS 连接中断，请按下面步骤用浏览器下载 wheel，再离线安装。

## 1. 下载两个 wheel 文件

浏览器打开以下两个官方 PyPI 页面：

1. <https://pypi.org/project/fsspec/2024.6.1/#files>
2. <https://pypi.org/project/ultralytics-thop/2.0.14/#files>

分别在 `Download files` / `Built Distribution` 中下载：

```text
fsspec-2024.6.1-py3-none-any.whl
ultralytics_thop-2.0.14-py3-none-any.whl
```

不要下载 `.tar.gz`。这两个 `py3-none-any.whl` 都可用于当前 Windows Python 3.8 环境。

## 2. 放到项目目录

在下面目录中新建 `offline_wheels` 文件夹：

```text
F:\Chatgpt\Automated Desktop Object Sorting and Classification\robomaster_grasp_motion_test\offline_wheels
```

将两个 `.whl` 文件复制进去。最终应为：

```text
robomaster_grasp_motion_test\offline_wheels\fsspec-2024.6.1-py3-none-any.whl
robomaster_grasp_motion_test\offline_wheels\ultralytics_thop-2.0.14-py3-none-any.whl
```

## 3. 断网也可执行的安装命令

打开 PowerShell，逐行执行：

```powershell
Set-Location -LiteralPath 'F:\Chatgpt\Automated Desktop Object Sorting and Classification\robomaster_grasp_motion_test'

& 'F:\Anaconda\envs\robomaster_env\python.exe' -m pip install --no-index --no-deps '.\offline_wheels\fsspec-2024.6.1-py3-none-any.whl' '.\offline_wheels\ultralytics_thop-2.0.14-py3-none-any.whl'

& 'F:\Anaconda\envs\robomaster_env\python.exe' -m pip check
```

最后一条命令的理想输出是：

```text
No broken requirements found.
```

## 4. 安装后再做一次程序自检

```powershell
& 'F:\Anaconda\envs\robomaster_env\python.exe' .\grasp_motion_test.py --validate-config
```

预期看到 `configuration is valid for dry-run`。这只表示软件环境和配置结构通过检查，不表示机械臂坐标已经标定，也不会连接或移动机器人。

