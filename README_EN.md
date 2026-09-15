# Experiment 3: Automatic Desktop Object Sorting

This repository combines the Isaac Sim simulation and RoboMaster EP real-robot validation for the desktop object sorting experiment.

## Project layout

- `simulation/`: Isaac Sim scene, camera stream, TCP joint bridge, and simulation utilities.
- `src/mecharm_pick_place/`: ROS 2 nodes, launch file, fixed-grid YOLO classification, and sorting state machine.
- `config/`: simulation and exception-case configurations.
- `src/real_robot/`: organized RoboMaster EP program and staged validation versions.
- `config/real_robot/`: real-robot configuration, ground layout, and dependency lists.
- `models/`: simulation and real-robot YOLO models.
- `results/experiment3/`: simulation logs, camera-view experiments, and simulation evidence.
- `results/real_robot/`: real-robot run logs, images, videos, summaries, and checksums.
- `report/`: LaTeX reports, figures, and presentation materials.
- `docs/real_robot/`: real-robot documentation and material index.

## Simulation workflow

The top camera publishes rendered frames through TCP port `8766`. The ROS 2 vision node classifies six fixed ROIs over six consecutive frames using majority voting. The sorting node then uses the known grid coordinates—not YOLO box centers—to execute fixed-order pick-and-place actions through the TCP joint bridge on port `8765`.

Build and launch commands are documented in `README.md` and `docs/testing.md`.

## Real-robot validation

The real-robot part uses the RoboMaster EP Python SDK and is maintained separately from the ROS 2 simulation runtime. Its source, configuration, model, tests, evidence, and report materials are placed under the corresponding project directories. The raw extracted package under `real_validation/` is a local rollback backup and is ignored by Git.

## Reproducibility and evidence

Generated build directories, logs, caches, and runtime result streams are ignored where appropriate. The complete simulation recording is stored under `results/experiment3/evidence/videos/`; because it is larger than GitHub's regular 100 MB file limit, it must be uploaded through Git LFS or another artifact store if published.
