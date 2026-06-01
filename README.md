# NeRF Lab

## Overview
This repository contains small Python helpers and experiments for NeRF / ray-marching concepts. It is part of the supporting perception layer for a robotics research portfolio focused on active sensing, 3D perception, visual state estimation, and SLAM-related representation learning.

The goal is not to present a full NeRF-SLAM system here. This is a clean public lab for understanding view synthesis, rays, image formation, and neural scene representation ideas that can support future robotic scanning work.

## Repository Role
`nerf-lab` is a support repository, not the main active-scanning contribution. It belongs near the research because active scanning depends on viewpoint choice, scene representation, visibility, and coverage.

| Related repo | Relationship |
|---|---|
| [`line-scan-mobile-manipulator-demo`](https://github.com/WikiGenius/line-scan-mobile-manipulator-demo) | Main public demo for active scanning and coverage planning. |
| [`GTSAM_SLAM_VISION`](https://github.com/WikiGenius/GTSAM_SLAM_VISION) | Factor-graph visual estimation and geometry-based SLAM experiments. |
| [`orb_slam_demo`](https://github.com/WikiGenius/orb_slam_demo) | ROS2 scaffold for visual state-estimation demos. |
| [`research-reading-map`](https://github.com/WikiGenius/research-reading-map) | Literature and architecture map that connects NeRF/view synthesis to active perception. |

## Research/Engineering Motivation
Structure-aware scanning depends on how a robot chooses viewpoints and reasons about scene geometry. NeRF-style representations are relevant because they connect camera motion, view coverage, 3D structure, and image formation.

This lab supports the broader research direction by exploring compact tools for rays, rendering intuition, and visual reconstruction concepts.

## Features
- Python package scaffold: `nerflab`.
- Small examples for NeRF / ray-marching experiments.
- Lightweight dependencies through `pyproject.toml`.
- Space for future visualizations and reconstruction notes.
- Public context linking NeRF-style perception to active scanning and SLAM.

## Method
The public lab is organized around simple, inspectable experiments:

1. Define camera rays and sample points.
2. Explore ray-marching / rendering utilities.
3. Visualize geometry and rendered outputs.
4. Connect viewpoint coverage ideas to active perception and SLAM.
5. Document what is conceptual, simplified, or not yet integrated with robotics.

## Installation
Clone and install in editable mode:

```bash
git clone https://github.com/WikiGenius/nerf-lab.git
cd nerf-lab
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

## Run
Explore the examples and experiments folders:

```bash
python examples/<example_name>.py
```

## Results
Future public results can include:

- ray visualization figures,
- rendered toy scenes,
- viewpoint coverage illustrations,
- notes connecting NeRF intuition to robot scanning.

## Limitations
- This is a learning/research lab, not a production NeRF implementation.
- It does not include private datasets or unpublished scanning experiments.
- Robotics integration is conceptual until future demos are added.
- This repo should support the research story without distracting from the main mobile-manipulation planning repos.

## Roadmap
- [ ] Add a minimal ray visualization example.
- [ ] Add toy rendering outputs.
- [ ] Add notes on view coverage and active perception.
- [ ] Add links to relevant papers in the reading map.
- [ ] Add a simple connection note to visual SLAM / factor graphs.

## Citation / Acknowledgment
Acknowledge NeRF and neural rendering papers, libraries, and tutorials used in future examples.

## Research Context
See [`docs/research-context.md`](docs/research-context.md) for how this lab relates to active scanning, SLAM, and private/public research organization.
