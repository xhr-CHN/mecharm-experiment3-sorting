# Camera View Sweep Design

## Goal

Select an overhead Isaac Sim camera pose that gives the experiment-one YOLO model the most stable six-object classification performance, with special priority on pencil recall.

## Candidates

Evaluate one vertical view and four azimuths at each of 10, 15, and 20 degrees from vertical. Every view uses 848 by 480 pixels, 18 mm focal length, a 1.10 m camera height, and a look-at target at the robot origin. Capture five rendered frames per view.

## Scoring

Run `pencil_tennis_yolo26n_best.pt` at confidence 0.05 and image size 640. Cap each expected class at three true objects per frame. Rank by minimum six-object coverage across the five frames, mean pencil recall, mean total coverage, mean confidence, then fewer excess detections.

## Evidence

Save raw images, annotated images, per-frame detections, per-view summary CSV, model SHA-256, recommended pose YAML, and a Markdown conclusion under `results/experiment3/camera_view_sweep/`.

## Production Change

Apply only the winning camera pose to `/World/experiment3/top_camera`. Keep object world coordinates, fixed-grid grasp centers, YOLO model, and all robot motion parameters unchanged.
