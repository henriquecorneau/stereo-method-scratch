"""Plotting helpers for disparity maps."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def show_epipolar_lines(img_left: np.ndarray, img_right: np.ndarray, step: int = 20):
    """Show both images side by side with horizontal lines every ``step`` pixels.

    On a rectified pair, the same scene point must lie on the same line in both
    images. Returns the matplotlib figure.
    """
    combined = np.hstack((img_left, img_right)).copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(combined, cmap="gray")
    for y in range(0, combined.shape[0], step):
        ax.axhline(y, color="yellow", linewidth=0.6)
    ax.set_title("Epipolar lines (rectified pair)")
    ax.axis("off")
    return fig


def compare_disparity_maps(maps: dict[str, np.ndarray], cmap: str = "jet", shared_scale: bool = True):
    """Plot several disparity maps next to each other, each with a colour bar.

    Args:
        maps: ``{title: disparity_map}``. NaN pixels (invalid) are left blank.
        shared_scale: Use the same colour range for every map, so colours are comparable.
    """
    vmin = vmax = None
    if shared_scale:
        vmin = min(np.nanmin(m) for m in maps.values())
        vmax = max(np.nanmax(m) for m in maps.values())
    fig, axes = plt.subplots(1, len(maps), figsize=(5 * len(maps), 4.5), squeeze=False)
    for ax, (title, disparity) in zip(axes[0], maps.items()):
        im = ax.imshow(disparity, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_disparity_surface(disparity: np.ndarray, step: int = 3,
                           elev: float = 80, azim: float = 300):
    """3D surface of a disparity map (subsampled every ``step`` pixels)."""
    reduced = disparity[::step, ::step].astype(float)
    h, w = reduced.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(x, y, reduced, cmap="jet", edgecolor="none", linewidth=0)
    ax.set_title("Disparity surface")
    ax.set_xlabel("x (columns)")
    ax.set_ylabel("y (rows)")
    ax.set_zlabel("Disparity")
    ax.invert_yaxis()
    ax.view_init(elev=elev, azim=azim)
    fig.colorbar(surface, shrink=0.5, aspect=5, label="Disparity")
    return fig
