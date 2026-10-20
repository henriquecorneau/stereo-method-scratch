"""Compute a disparity map from a rectified stereo pair.

Example:
    python run_stereo.py images/cones_left.png images/cones_right.png --max-disparity 64 --out results
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from stereo_dp import disparity_dp  # noqa: E402
from stereo_dp.visualization import compare_disparity_maps, plot_disparity_surface  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("left", type=Path, help="rectified left image")
    parser.add_argument("right", type=Path, help="rectified right image")
    parser.add_argument("--max-disparity", type=int, default=64, help="number of disparity levels (default: 64)")
    parser.add_argument("--p-h", type=float, default=25.0, help="horizontal smoothness penalty (default: 25)")
    parser.add_argument("--p-v", type=float, default=3.0, help="inter-scanline penalty (default: 3)")
    parser.add_argument("--occlusion", type=float, default=25.0, help="occlusion cost (default: 25)")
    parser.add_argument("--out", type=Path, default=Path("results"), help="output directory (default: results)")
    parser.add_argument("--sgbm", action="store_true", help="also compute OpenCV StereoSGBM for comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left = cv2.imread(str(args.left), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(args.right), cv2.IMREAD_GRAYSCALE)
    if left is None or right is None:
        raise SystemExit("Could not read the input images")
    args.out.mkdir(parents=True, exist_ok=True)

    params = dict(max_disparity=args.max_disparity, p_horizontal=args.p_h, occlusion_cost=args.occlusion)

    t0 = time.perf_counter()
    without_is = disparity_dp(left, right, p_vertical=0, **params)
    t1 = time.perf_counter()
    with_is = disparity_dp(left, right, p_vertical=args.p_v, **params)
    t2 = time.perf_counter()
    print(f"Intra-scanline DP:           {t1 - t0:.2f} s")
    print(f"Intra + inter-scanline DP:   {t2 - t1:.2f} s")

    # Raw disparities scaled to 0-255 for viewing
    scale = 255.0 / max(args.max_disparity - 1, 1)
    cv2.imwrite(str(args.out / "disparity_dp.png"), (with_is * scale).astype("uint8"))

    maps = {"Intra-scanline only": without_is, "Intra + inter-scanline": with_is}
    if args.sgbm:
        num = ((args.max_disparity + 15) // 16) * 16
        sgbm = cv2.StereoSGBM.create(minDisparity=0, numDisparities=num, blockSize=5,
                                     P1=8 * 25, P2=32 * 25, uniquenessRatio=10, disp12MaxDiff=1)
        sgbm_disp = sgbm.compute(left, right).astype("float32") / 16.0
        sgbm_disp[sgbm_disp < 0] = float("nan")  # pixels without a valid match
        maps["OpenCV StereoSGBM"] = sgbm_disp

    compare_disparity_maps(maps).savefig(args.out / "comparison.png", dpi=150, bbox_inches="tight")
    plot_disparity_surface(with_is).savefig(args.out / "surface_3d.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"Results saved to {args.out}/")


if __name__ == "__main__":
    main()
