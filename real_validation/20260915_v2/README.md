# Experiment 3 Real-Robot Validation Materials

This directory contains the complete real-robot validation package received on 2026-09-15.

## Organisation

- `raw_package/`: complete extracted source package, logs, images, videos, tests, model, and presentation materials.
- `raw_package/02_核心程序与模型/desktop_sorter.py`: final RoboMaster EP sorter.
- `raw_package/02_核心程序与模型/sorter_config.json`: camera ROIs, ground layout, routes, arm points, and safety parameters.
- `raw_package/04_完整运行记录/`: full scan images, event logs, and task summaries.
- `raw_package/05_图像与视频证据/`: six-grid recognition evidence and two-object videos.
- `raw_package/07_报告辅助材料/`: run summary, hashes, test results, and statistics.

## Evidence boundary

The real package verifies a RoboMaster EP two-object cup/bottle workflow. The representative run `run_20260913_012927_039424` completed two sorted objects (`bottle=1`, `cup=1`) and saved four scans. It does not prove a six-object real-robot acceptance run. The simulation section remains the mechArm 270 Pi six-object Isaac Sim implementation.

The package reports 88 offline unit tests passed across the final sorter and staged test variants. These tests validate program logic and do not equal 88 physical grasp attempts.
