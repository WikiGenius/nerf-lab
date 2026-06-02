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
| [`husky-gazebo-image-capture`](https://github.com/WikiGenius/husky-gazebo-image-capture) | Gazebo Husky image/odometry snapshot capture for later visual-SLAM experiments. |
| [`research-reading-map`](https://github.com/WikiGenius/research-reading-map) | Literature and architecture map that connects NeRF/view synthesis to active perception. |

## Relation to My Research Direction

NeRF and neural scene representations are relevant to my broader robotics direction as possible future world-model layers for:

- active perception,
- view synthesis,
- 3D scene representation,
- robotic inspection,
- state estimation support,
- planning under uncertainty,
- motion planning in learned scene representations.

This repository supports those ideas as a perception and representation lab. It is not presented as a completed robotics planning contribution.

## Current Maturity

**Current status:** Supporting research lab / future extension layer.

This repository is not currently presented as a completed robotics world-model system or a validated motion-planning pipeline.

## Research/Engineering Motivation
Structure-aware scanning depends on how a robot chooses viewpoints and reasons about scene geometry. NeRF-style representations are relevant because they connect camera motion, view coverage, 3D structure, and image formation.

This lab supports the broader research direction by exploring compact tools for rays, rendering intuition, and visual reconstruction concepts.

## Features
- Python package scaffold: `nerflab`.
- Small examples for NeRF / ray-marching experiments.
- Lightweight dependencies through `pyproject.toml`.
- Space for future visualizations and reconstruction notes.
- Public context linking NeRF-style perception to active scanning and SLAM.

## Implemented Now

- [x] Repository organization
- [x] Python package scaffold: `nerflab`
- [x] Camera, world, IO, learning, and visualization helper modules
- [x] Notebook examples for camera poses, world setup, ray sampling, data loading, and exploratory training
- [ ] Confirmed fully reproducible end-to-end NeRF demo
- [ ] Robotics planning integration
- [ ] World-model planning experiment

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
Explore the notebook examples and experiments folders:

Open the `.ipynb` notebooks in `examples/` or `experiments/` using a Jupyter environment.

Helper scripts used by some notebooks are kept under `examples/helper/`.

## Results
Future public results can include:

- ray visualization figures,
- rendered toy scenes,
- viewpoint coverage illustrations,
- notes connecting NeRF intuition to robot scanning.

## Planned Later

Potential future directions include:

- simple NeRF or neural-rendering demo,
- camera-ray visualization,
- view-synthesis experiment,
- connection to active perception,
- connection to scan planning,
- possible world-model interface for motion planning.

## Limitations
- This is a learning/research lab, not a production NeRF implementation.
- It does not include private datasets or unpublished scanning experiments.
- Robotics integration is conceptual until future demos are added.
- It does not yet claim a complete robotics world model.
- It does not yet claim validated integration with mobile manipulation.
- It does not yet claim real-time planning over NeRF.
- It does not yet claim benchmarked robotic planning performance.
- It does not yet claim paper-level results.
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
