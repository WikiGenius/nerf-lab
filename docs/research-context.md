# Research Context

`nerf-lab` supports the perception side of structure-aware mobile manipulation.

## Why This Repo Belongs in the Portfolio

The main research direction is active scanning with a mobile manipulator. That problem depends on viewpoint choice, visibility, coverage, sensor geometry, and scene representation. NeRF-style and ray-based methods are relevant because they help reason about:

- how camera rays sample a scene,
- how viewpoints affect image information,
- how 3D structure can be represented from observations,
- how coverage and reconstruction quality relate to motion,
- how visual perception can support planning.

## Relationship to Other Repos

| Repo | Connection |
|---|---|
| [`line-scan-mobile-manipulator-demo`](https://github.com/WikiGenius/line-scan-mobile-manipulator-demo) | Main active-scanning public demo. `nerf-lab` can provide view-synthesis and ray-geometry intuition. |
| [`GTSAM_SLAM_VISION`](https://github.com/WikiGenius/GTSAM_SLAM_VISION) | Geometry and factor-graph side of visual state estimation. |
| [`orb_slam_demo`](https://github.com/WikiGenius/orb_slam_demo) | ROS2 visual SLAM scaffold that can eventually connect to perception demos. |
| [`research-reading-map`](https://github.com/WikiGenius/research-reading-map) | Place for NeRF, neural rendering, SLAM, active perception, and coverage papers. |

## Public/Private Boundary

Public here:

- toy ray examples,
- small generated scenes,
- simplified rendering utilities,
- conceptual notes,
- links to papers and public demos.

Private elsewhere:

- unpublished robotic scanning experiments,
- private RGB/hyperspectral data,
- paper-specific reconstruction/active perception methods,
- raw experiment logs and ablations.

## Suggested Next Public Artifacts

1. Minimal ray visualization figure.
2. Toy volumetric rendering example.
3. Short note connecting camera rays to line-scan sensing geometry.
4. Reading-map links for NeRF, NeRF-SLAM, and active view planning papers.
