from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Literal
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from nerflab.geometry import *
from nerflab import Camera
# ============================================================
# Primitive plotters
# ============================================================

def plot_box(ax, box, alpha: float = 0.15, edgecolor: str = 'k', facecolor: str = 'C0'):
    """
    Draw an axis-aligned box.

    Parameters
    ----------
    box : has attributes min_x,max_x,min_y,max_y,min_z,max_z
    """
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = box.bounds

    x = [xmin, xmax]
    y = [ymin, ymax]
    z = [zmin, zmax]
    verts = np.array([[x[i], y[j], z[k]]
                      for i in (0, 1)
                      for j in (0, 1)
                      for k in (0, 1)], dtype=np.float32)
    faces = [
        [0, 1, 3, 2], [4, 5, 7, 6],  # bottom/top
        [0, 1, 5, 4], [2, 3, 7, 6],  # y-min/y-max
        [0, 2, 6, 4], [1, 3, 7, 5],  # x-min/x-max
    ]
    poly = Poly3DCollection([verts[f] for f in faces],
                            alpha=alpha, edgecolor=edgecolor, facecolor=facecolor)
    ax.add_collection3d(poly)


def plot_sphere(ax, sphere, alpha: float = 0.15, res: int = 30, color: str = 'C1'):
    """
    Draw a sphere.

    Parameters
    ----------
    sphere : has attributes cx, cy, cz, radius
    """
    u = np.linspace(0, 2 * np.pi, res)
    v = np.linspace(0, np.pi, res)
    cx, cy, cz = sphere.center
    x = cx + sphere.radius * np.outer(np.cos(u), np.sin(v))
    y = cy + sphere.radius * np.outer(np.sin(u), np.sin(v))
    z = cz + sphere.radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, linewidth=0, antialiased=False,
                    alpha=alpha, edgecolor='none', color=color)


# ============================================================
# Axis / grid helpers
# ============================================================

def set_3d_axes_equal(ax):
    """Force equal scale on all 3 axes (call after adding artists)."""
    xlim, ylim, zlim = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    xmid, ymid, zmid = map(np.mean, (xlim, ylim, zlim))
    max_range = max(xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]) / 2
    ax.set_xlim3d(xmid - max_range, xmid + max_range)
    ax.set_ylim3d(ymid - max_range, ymid + max_range)
    ax.set_zlim3d(zmid - max_range, zmid + max_range)


def draw_axis_triad(ax,
                    origin: Sequence[float] = (0., 0., 0.),
                    length: float = 0.5,
                    lw: float = 2.0):
    """Draw RGB axis triad at origin: X=red, Y=green, Z=blue."""
    o = np.asarray(origin, dtype=np.float32)
    axes = np.eye(3, dtype=np.float32) * length
    colors = ['r', 'g', 'b']
    for i, c in enumerate(colors):
        p = o + axes[i]
        ax.plot([o[0], p[0]], [o[1], p[1]], [o[2], p[2]], c, lw=lw)


def draw_grid_3d(ax,
                 bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
                 step: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 color: str = '#CCCCCC',
                 lw: float = 0.3,
                 alpha: float = 0.6):
    """
    Draw a simple 3D lattice/grid (because Matplotlib's built-in grid is 2D-like).

    Parameters
    ----------
    bounds : ((xmin,xmax),(ymin,ymax),(zmin,zmax))
    step   : (sx, sy, sz) spacing between grid lines
    """
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    xs = np.arange(xmin, xmax + 1e-9, step[0])
    ys = np.arange(ymin, ymax + 1e-9, step[1])
    zs = np.arange(zmin, zmax + 1e-9, step[2])

    # Lines parallel to X (vary y,z)
    for y in ys:
        for z in zs:
            ax.plot([xmin, xmax], [y, y], [z, z], color=color, lw=lw, alpha=alpha)
    # Lines parallel to Y
    for x in xs:
        for z in zs:
            ax.plot([x, x], [ymin, ymax], [z, z], color=color, lw=lw, alpha=alpha)
    # Lines parallel to Z
    for x in xs:
        for y in ys:
            ax.plot([x, x], [y, y], [zmin, zmax], color=color, lw=lw, alpha=alpha)


# ============================================================
# World plot (with optional cameras)
# ============================================================

def plot_world(world,
               cameras: Iterable["Camera"] | None = None,
               draw_rays: bool = False,
               ray_step: int = 40,
               ray_mode: Literal["lines", "quiver", "points"] = "lines",
               sample_grid: bool = False,
               grid_bounds: Tuple[Tuple[float, float], ...] = ((-1, 7), (-1, 5), (-1, 4)),
               grid_res: Tuple[int, int, int] = (80, 80, 60),
               grid_lines: bool = False,
               grid_step: Tuple[float, float, float] = (1., 1., 1.),
               invert_x: bool = False,
               figsize=(7, 6),
               elev: float = 22,
               azim: float = 35):
    """
    Plot world primitives and optionally cameras & their rays.

    Parameters
    ----------
    world : World
        Needs .shapes list and .density(x,y,z) (if sample_grid=True).
    cameras : iterable[Camera] | None
    draw_rays : bool
    ray_step : int
    ray_len : float
    ray_mode : "lines" | "quiver" | "points"
    sample_grid : bool
        If True, sample density on a voxel grid and scatter infinite-density pts.
    grid_bounds : ((xmin,xmax),(ymin,ymax),(zmin,zmax))
    grid_res : (nx,ny,nz)
    grid_lines : bool
        If True, draw a 3D lattice grid with given step.
    grid_step : (sx,sy,sz)
    invert_x : bool
        If True, flip X axis direction (X left on screen).
    figsize : tuple
    elev, azim : floats
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # ---- draw shapes ----
    for s in getattr(world, "shapes", []):
        if isinstance(s, Box):
            plot_box(ax, s)
        elif isinstance(s, Sphere):
            plot_sphere(ax, s)

    # ---- optional occupancy sampling ----
    if sample_grid:
        xs = np.linspace(*grid_bounds[0], grid_res[0])
        ys = np.linspace(*grid_bounds[1], grid_res[1])
        zs = np.linspace(*grid_bounds[2], grid_res[2])
        pts = np.array(np.meshgrid(xs, ys, zs, indexing='ij')).reshape(3, -1).T
        dens = np.array([world.density(*p) for p in pts])
        occ_pts = pts[np.isinf(dens)]
        if occ_pts.size:
            ax.scatter(occ_pts[:, 0], occ_pts[:, 1], occ_pts[:, 2],
                       s=1, c='k', alpha=0.4, label='occ')

    # ---- world triad ----
    draw_axis_triad(ax, origin=(0, 0, 0), length=0.8, lw=2)

    # ---- cameras ----
    if cameras is not None:
        for i, cam in enumerate(cameras):
            cam_pos = cam.H_wc[:3, 3]
            ray_len = cam.t_far
            ax.scatter(*cam_pos, s=50, c='red', marker='o', label=f'Cam {i}')
            Camera._draw_pose_axes(ax, cam.H_wc, scale=0.4)

            if draw_rays:
                O, D = cam.get_rays(frame="world", step=ray_step, normalize=True)
                color = f"C{i % 10}"
                if ray_mode == "quiver":
                    ax.quiver(O[:, 0], O[:, 1], O[:, 2],
                              D[:, 0], D[:, 1], D[:, 2],
                              length=ray_len, normalize=True, color=color, linewidth=0.6)
                elif ray_mode == "lines":
                    P = O + ray_len * D
                    for spt, ept in zip(O, P):
                        ax.plot([spt[0], ept[0]],
                                [spt[1], ept[1]],
                                [spt[2], ept[2]],
                                color=color, lw=0.6)
                elif ray_mode == "points":
                    ts = np.linspace(cam.t_near, cam.t_far,
                                     getattr(cam, "n_points_per_ray", 20))
                    P = (O[:, None, :] + ts[None, :, None] * D[:, None, :]).reshape(-1, 3)
                    ax.scatter(P[:, 0], P[:, 1], P[:, 2],
                               s=2, c=color, depthshade=False)
                else:
                    raise ValueError("ray_mode must be 'lines', 'quiver', or 'points'")

    # ---- labels / limits / style ----
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    if invert_x:
        ax.invert_xaxis()

    # optional 3D gridlines
    if grid_lines:
        draw_grid_3d(ax, bounds=grid_bounds, step=grid_step)

    # equalize & set view
    set_3d_axes_equal(ax)
    ax.view_init(elev=elev, azim=azim)

    # legend if needed
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc='upper right')

    ax.set_box_aspect([1,1,1])
    plt.tight_layout()
    plt.show()



def viz_sigma_scatter(pts, sigma, max_points=20000, cmap='viridis'):
    """
    3D scatter of points colored by sigma.
    pts   : (R,N,3)
    sigma : (R,N)
    """
    P = pts.reshape(-1, 3)
    S = sigma.reshape(-1)

    # Subsample if huge
    if P.shape[0] > max_points:
        idx = np.random.choice(P.shape[0], max_points, replace=False)
        P, S = P[idx], S[idx]

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(P[:,0], P[:,1], P[:,2], c=S, s=2, cmap=cmap, depthshade=False)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    fig.colorbar(sc, ax=ax, label='sigma')
    ax.set_box_aspect([1,1,1])
    plt.tight_layout()
    plt.show()

def viz_sigma_heatmap(sigma, cmap='magma'):
    """
    Show sigma per ray/sample as an image.
    sigma : (R,N)
    """
    plt.figure(figsize=(8,4))
    plt.imshow(sigma, aspect='auto', cmap=cmap)
    plt.xlabel('Sample index (N)')
    plt.ylabel('Ray index (R)')
    cbar = plt.colorbar()
    cbar.set_label('sigma')
    plt.tight_layout()
    plt.show()
