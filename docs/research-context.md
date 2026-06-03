# Research Context

`nerf-lab` supports the perception side of structure-aware mobile manipulation.

The main research direction is active scanning with a mobile manipulator. That problem depends on viewpoint choice, visibility, coverage, sensor geometry, and scene representation. NeRF-style and ray-based methods are relevant because they help reason about:

- how camera rays sample a scene,
- how viewpoints affect image information,
- how dense 3D structure can be represented from observations,
- how coverage and reconstruction quality relate to motion,
- how neural scene maps may support future planning and state-estimation work.

## Relationship To Other Repos

| Repo | Connection |
|---|---|
| [`line-scan-mobile-manipulator-demo`](https://github.com/WikiGenius/line-scan-mobile-manipulator-demo) | Main active-scanning public demo. `nerf-lab` provides view-synthesis and ray-geometry intuition. |
| [`GTSAM_SLAM_VISION`](https://github.com/WikiGenius/GTSAM_SLAM_VISION) | Geometry and factor-graph side of visual state estimation. |
| [`husky-gazebo-image-capture`](https://github.com/WikiGenius/husky-gazebo-image-capture) | Public-safe image/odometry capture for visual-SLAM experiments. |
| [`research-reading-map`](https://github.com/WikiGenius/research-reading-map) | Place for NeRF, neural rendering, SLAM, active perception, and coverage papers. |

## Public / Private Boundary

Public here:

- toy ray examples,
- small generated scenes,
- simplified rendering utilities,
- controlled notebook outputs,
- public course report and presentations.

Private elsewhere:

- unpublished robotic scanning experiments,
- private RGB/hyperspectral data,
- paper-specific active perception methods,
- raw experiment logs and unpublished ablations.
