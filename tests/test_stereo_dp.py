import numpy as np
import pytest

from stereo_dp import aggregate_scanline, backtrack, disparity_dp, inter_scanline_penalty, matching_cost


def textured_pair(h=40, w=120, disparity=7, seed=0):
    """Random texture shifted by a constant disparity: right(x) = left(x + d)."""
    rng = np.random.default_rng(seed)
    left = rng.integers(0, 256, size=(h, w)).astype(np.uint8)
    right = np.zeros_like(left)
    right[:, : w - disparity] = left[:, disparity:]
    return left, right


def test_matching_cost_shape_and_occlusion():
    left = np.array([10, 20, 30, 40], dtype=np.uint8)
    right = np.array([20, 30, 40, 50], dtype=np.uint8)
    cost = matching_cost(left, right, max_disparity=3, occlusion_cost=99)
    assert cost.shape == (4, 3)
    assert cost[0, 1] == 99 and cost[0, 2] == 99 and cost[1, 2] == 99
    # left[x] vs right[x - 1] are equal for x >= 1
    assert np.all(cost[1:, 1] == 0)


def test_inter_scanline_penalty_does_not_touch_previous_disparity():
    cost = np.zeros((3, 4))
    out = inter_scanline_penalty(cost, np.array([0, 2, 3]), penalty=5)
    assert np.array_equal(out, [[0, 5, 5, 5], [5, 5, 0, 5], [5, 5, 5, 0]])
    assert np.all(cost == 0), "input must not be modified"


def test_dp_path_changes_by_at_most_one_level():
    rng = np.random.default_rng(1)
    acc, back = aggregate_scanline(rng.random((50, 16)) * 100, penalty=10)
    path = backtrack(acc, back)
    assert np.all(np.abs(np.diff(path)) <= 1)


@pytest.mark.parametrize("p_vertical", [0, 3])
def test_recovers_constant_disparity(p_vertical):
    d = 7
    left, right = textured_pair(disparity=d)
    disp = disparity_dp(left, right, max_disparity=16, p_horizontal=10, p_vertical=p_vertical, occlusion_cost=25)
    valid = disp[:, 2 * d : -2 * d]
    assert np.mean(valid == d) > 0.95


def test_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        disparity_dp(np.zeros((10, 10)), np.zeros((10, 11)))
