"""Scanline stereo matching with dynamic programming.

Implementation of a dense stereo method inspired by Ohta & Kanade,
"Stereo by Intra- and Inter-Scanline Search Using Dynamic Programming" (1985).

Each image row (scanline) of a rectified stereo pair is solved as a shortest
path problem on a graph whose nodes are (column, disparity) pairs:

* **Intra-scanline search** - dynamic programming along the row finds the
  disparity profile that minimises the matching cost plus a penalty for every
  change of disparity between neighbouring pixels.
* **Inter-scanline consistency** - the disparity found on the row above is used
  as a prior: choosing a different disparity on the current row costs an extra
  penalty, which removes most of the horizontal streaks typical of
  row-independent DP.

All images are expected to be rectified, so that corresponding points lie on
the same row, and the left image is used as the reference.
"""

from __future__ import annotations

import numpy as np

# Moves stored in the back-pointer table (where the path came from at column x - 1).
SAME, FROM_LOWER, FROM_HIGHER = 0, 1, 2


def matching_cost(line_left: np.ndarray, line_right: np.ndarray,
                  max_disparity: int, occlusion_cost: float) -> np.ndarray:
    """Absolute intensity difference for every (column, disparity) pair.

    Pixel ``x`` of the left row is compared with pixel ``x - d`` of the right
    row. Pairs that fall outside the right image (``x < d``) get
    ``occlusion_cost``.

    Args:
        line_left: Row of the left (reference) image, shape ``(w,)``.
        line_right: Same row of the right image, shape ``(w,)``.
        max_disparity: Number of disparities searched, ``0 .. max_disparity - 1``.
        occlusion_cost: Cost assigned to pixels without a valid match.

    Returns:
        Cost table of shape ``(w, max_disparity)``.
    """
    left = np.asarray(line_left, dtype=np.float64)
    right = np.asarray(line_right, dtype=np.float64)
    w = left.shape[0]
    cost = np.full((w, max_disparity), float(occlusion_cost))

    for d in range(min(max_disparity, w)):
        cost[d:, d] = np.abs(left[d:] - right[:w - d])

    return cost


def inter_scanline_penalty(cost: np.ndarray, previous_disparity: np.ndarray,
                           penalty: float) -> np.ndarray:
    """Add a vertical smoothness penalty based on the row above.

    Every disparity that differs from the one chosen at the same column on the
    previous scanline costs ``penalty`` more.

    Args:
        cost: Cost table ``(w, max_disparity)`` of the current row.
        previous_disparity: Disparities of the previous row, shape ``(w,)``.
        penalty: Vertical penalty ``P_v``. ``0`` disables the term.

    Returns:
        A new cost table (the input is not modified).
    """
    if penalty == 0:
        return cost.copy()

    w, max_disparity = cost.shape
    prev = np.clip(np.asarray(previous_disparity, dtype=np.int64), 0, max_disparity - 1)
    out = cost + penalty
    out[np.arange(w), prev] -= penalty
    return out


def aggregate_scanline(cost: np.ndarray, penalty: float) -> tuple[np.ndarray, np.ndarray]:
    """Dynamic programming along one scanline (forward pass).

    The accumulated cost of reaching disparity ``d`` at column ``x`` is::

        C(x, d) = cost(x, d) + min(C(x-1, d),
                                   C(x-1, d-1) + P,
                                   C(x-1, d+1) + P)

    so the disparity can change by at most one level between neighbouring
    pixels, and each change costs ``P``. On ties the path keeps the same
    disparity.

    Args:
        cost: Cost table ``(w, max_disparity)``.
        penalty: Horizontal smoothness penalty ``P_h``.

    Returns:
        ``(accumulated, back_pointers)``, both of shape ``(w, max_disparity)``.
        Back-pointers hold ``SAME``, ``FROM_LOWER`` (came from ``d - 1``) or
        ``FROM_HIGHER`` (came from ``d + 1``).
    """
    w, max_disparity = cost.shape
    accumulated = np.empty_like(cost, dtype=np.float64)
    back = np.zeros((w, max_disparity), dtype=np.uint8)
    candidates = np.empty((3, max_disparity))

    accumulated[0] = cost[0]
    for x in range(1, w):
        prev = accumulated[x - 1]
        candidates[SAME] = prev
        candidates[FROM_LOWER, 0] = np.inf
        candidates[FROM_LOWER, 1:] = prev[:-1] + penalty
        candidates[FROM_HIGHER, -1] = np.inf
        candidates[FROM_HIGHER, :-1] = prev[1:] + penalty

        choice = np.argmin(candidates, axis=0)  # first minimum -> SAME wins ties
        back[x] = choice
        accumulated[x] = cost[x] + candidates[choice, np.arange(max_disparity)]

    return accumulated, back


def backtrack(accumulated: np.ndarray, back: np.ndarray) -> np.ndarray:
    """Recover the optimal disparity path from the back-pointer table.

    Args:
        accumulated: Accumulated costs from :func:`aggregate_scanline`.
        back: Back-pointers from :func:`aggregate_scanline`.

    Returns:
        Disparity of every column of the row, shape ``(w,)``, dtype ``int``.
    """
    w = accumulated.shape[0]
    disparity = np.zeros(w, dtype=np.int64)
    d = int(np.argmin(accumulated[-1]))

    for x in range(w - 1, -1, -1):
        disparity[x] = d
        move = back[x, d]
        if move == FROM_LOWER:
            d -= 1
        elif move == FROM_HIGHER:
            d += 1

    return disparity


def disparity_dp(img_left: np.ndarray, img_right: np.ndarray, max_disparity: int = 64,
                 p_horizontal: float = 25.0, p_vertical: float = 3.0,
                 occlusion_cost: float = 25.0) -> np.ndarray:
    """Dense disparity map with intra- and inter-scanline dynamic programming.

    Args:
        img_left: Rectified left (reference) image, grayscale ``(h, w)``.
        img_right: Rectified right image, grayscale ``(h, w)``.
        max_disparity: Number of disparity levels searched.
        p_horizontal: Penalty for a disparity change between horizontal neighbours.
        p_vertical: Penalty for differing from the row above. ``0`` gives plain
            row-by-row DP.
        occlusion_cost: Cost of pixels whose match would fall outside the image.

    Returns:
        Disparity map ``(h, w)``, dtype ``uint16``, values in ``[0, max_disparity)``.
    """
    left = _as_gray(img_left)
    right = _as_gray(img_right)
    if left.shape != right.shape:
        raise ValueError(f"Images must have the same shape, got {left.shape} and {right.shape}")
    if max_disparity < 1:
        raise ValueError("max_disparity must be >= 1")

    h, w = left.shape
    disparity = np.zeros((h, w), dtype=np.uint16)

    for y in range(h):
        cost = matching_cost(left[y], right[y], max_disparity, occlusion_cost)
        if y > 0 and p_vertical:
            cost = inter_scanline_penalty(cost, disparity[y - 1], p_vertical)
        accumulated, back = aggregate_scanline(cost, p_horizontal)
        disparity[y] = backtrack(accumulated, back)

    return disparity


def _as_gray(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 3:
        # BGR/RGB -> luminance without depending on OpenCV
        img = img[..., :3].astype(np.float64) @ np.array([0.114, 0.587, 0.299])
    if img.ndim != 2:
        raise ValueError("Expected a grayscale (h, w) or color (h, w, 3) image")
    return img.astype(np.float64)
