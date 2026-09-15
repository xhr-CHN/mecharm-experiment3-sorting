# Real-robot validation materials

This directory contains the organized RoboMaster EP validation materials that complement the Isaac Sim Experiment 3 implementation.

- `../../src/real_robot/core/`: main real-robot program.
- `../../src/real_robot/staged_tests/`: staged test versions and their configurations.
- `../../config/real_robot/`: tested runtime configuration and ground-layout data.
- `../../models/real_robot/`: real-robot YOLO model (`best.pt`).
- `../../tests/real_robot/`: offline tests for the real-robot program.
- `../../results/real_robot/full_runs/`: complete run logs and summaries.
- `../../results/real_robot/evidence/`: images and videos used as evidence.
- `../../results/real_robot/metadata/`: CSV summaries, test outputs, and checksums.
- `../../report/real_robot/source_materials/`: presentation source materials and page previews.

The original extracted package remains under `real_validation/20260915_v2/raw_package/` as a rollback-only backup. It is excluded from the Git project to avoid uploading duplicate raw data; the organized copies above are the project sources.

The real-robot implementation is a RoboMaster EP Python SDK program and is separate from the ROS 2/Isaac Sim runtime. Its evidence should be described separately from the simulation evidence in the report.
