# NeRF Lab

## Overview
This repository contains small Python helpers and experiments for NeRF / ray-marching concepts. It is part of the supporting perception layer for a robotics research portfolio focused on active sensing, 3D perception, and SLAM-related representation learning.

The goal is not to present a full NeRF-SLAM system here, but to keep a clean public lab for understanding view synthesis and neural scene representation ideas that can support future robotic scanning work.

## Research/Engineering Motivation
Structure-aware scanning depends on how a robot chooses viewpoints and reasons about scene geometry. NeRF-style representations are relevant because they connect camera motion, view coverage, 3D structure, and image formation.

This lab supports the broader research direction by exploring compact tools for rays, rendering intuition, and visual reconstruction concepts.

## Features
- Python package scaffold: `nerflab`.
- Small examples for NeRF / ray-marching experiments.
- Lightweight dependencies through `pyproject.toml`.
- Space for future visualizations and reconstruction notes.

## Method
The public lab is organized around simple, inspectable experiments:

1. Define camera rays and sample points.
2. Explore ray-marching / rendering utilities.
3. Visualize geometry and rendered outputs.
4. Connect viewpoint coverage ideas to active perception and SLAM.

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

## Roadmap
- [ ] Add a minimal ray visualization example.
- [ ] Add toy rendering outputs.
- [ ] Add notes on view coverage and active perception.
- [ ] Add links to relevant papers.

## Citation / Acknowledgment
Acknowledge NeRF and neural rendering papers, libraries, and tutorials used in future examples.
