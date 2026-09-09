# Experiment 3 Simulation Results

This directory stores local evidence for the circular tennis-ball and pencil sorting simulation.

Expected files after a run:

- `detections.jsonl`: camera detection records;
- `task_events.jsonl`: sorting state events;
- `sorting_results.jsonl`: per-object pick and placement results;
- `summary.yaml`: acceptance counts and safety flags;
- `demo.mp4`: autonomous demonstration video.

The normal acceptance scenario uses six objects distributed around the mechArm base and requires at least five correct placements. Exception scenarios are configured separately and must demonstrate at least two handled failures. No file in this directory is synchronized to GitHub automatically.
