#!/usr/bin/env python3
"""Peak-prominence sensitivity check for histogram-based lane counting.

Replicates the lane-counting path in
``LaneDetection.lane_detection.geo_learning.GeometricLearning.run`` exactly
(group-by-id mean ``x_gps`` -> ``np.histogram(bins=50)`` ->
Gaussian smoothing -> ``scipy.signal.find_peaks``) on the saved, GPS-projected
trajectories, and sweeps the ``peak_prominence`` parameter.

Purpose: show that the fixed ``prominence=1`` setting drops sparse / low-traffic
lanes, and that lowering the (now meta-learned) ``peak_prominence`` recovers
them -- the evidence behind the dedicated ``peak_prominence`` head and the
response to reviewer comments R1.1 (lane count) and R1.2 (sparse lanes).

The saved ``clustered_id`` column is what the original ``prominence=1`` run
produced, so the ``p=1.0`` column must match it -- a built-in faithfulness check
that this script reproduces the real code path.

Usage:
    python scripts/prominence_sensitivity.py --camera US12_Monona
    python scripts/prominence_sensitivity.py --camera US12_Todd --proms 1.0,0.75,0.5,0.3
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def gaussian_filter(data, window_size=5, sigma=2):
    """Verbatim copy of GeometricLearning.gaussian_filter (kept dependency-free)."""
    kernel = np.exp(-np.linspace(-2, 2, window_size) ** 2 / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return np.convolve(data, kernel, mode="same")


def count_peaks(smoothed, prominence, distance=3):
    """Same find_peaks call as geo_learning.run(); height uses the same salience floor."""
    peaks, _ = find_peaks(smoothed, height=prominence, distance=distance, prominence=prominence)
    return peaks


def analyze(df, proms, sigma, bins):
    """Return per-contour results: counts at each prominence, peak positions, histogram."""
    df = df.dropna(subset=["x_gps", "y_gps"])
    rows = []
    for cnt, dfc in df.groupby("contour_id"):
        x_mean = dfc.groupby("id")["x_gps"].mean().values  # one point per vehicle
        if len(x_mean) < 2:
            continue
        hist_vals, bin_edges = np.histogram(x_mean.reshape(-1, 1), bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        smoothed = gaussian_filter(hist_vals, window_size=5, sigma=sigma)
        peaks_by_p = {p: count_peaks(smoothed, p) for p in proms}
        rows.append({
            "contour": int(cnt),
            "n_veh": len(x_mean),
            "old_k": int(dfc["clustered_id"].dropna().nunique()),
            "osm": int(dfc["target_lane_id"].dropna().nunique()) if "target_lane_id" in dfc else -1,
            "bin_centers": bin_centers,
            "smoothed": smoothed,
            "peaks_by_p": peaks_by_p,
            "counts": {p: len(v) for p, v in peaks_by_p.items()},
        })
    return rows


def print_table(rows, proms):
    hdr = f"{'contour':>8}{'#veh':>7}{'old_k':>7}{'OSM':>5}  |" + "".join(f"{('p='+str(p)):>8}" for p in proms)
    print(hdr)
    print("-" * len(hdr))
    faithful = True
    p1 = 1.0 if 1.0 in proms else None
    for r in rows:
        line = f"{r['contour']:>8}{r['n_veh']:>7}{r['old_k']:>7}{r['osm']:>5}  |"
        line += "".join(f"{r['counts'][p]:>8}" for p in proms)
        print(line)
        if p1 is not None and r["counts"][p1] != r["old_k"]:
            faithful = False
    if p1 is not None:
        print(f"\nFaithfulness (p=1.0 == saved old_k): {'PASS' if faithful else 'FAIL'}")
    print("Legend: old_k = lanes the saved prominence=1 run produced; "
          "OSM = #distinct target_lane_id (noisy weak-supervision reference).")


def make_figure(rows, proms, camera, out_path):
    # Feature the contour whose detected count changes most across the sweep.
    def delta(r):
        c = list(r["counts"].values())
        return max(c) - min(c)
    focus = max(rows, key=delta)
    p_hi = max(proms)
    # Gentlest reduction that recovers a lane (largest prominence below p_hi whose
    # count exceeds the count at p_hi) -- avoids highlighting the over-segmented tail.
    recovering = [p for p in proms if p < p_hi and focus["counts"][p] > focus["counts"][p_hi]]
    p_lo = max(recovering) if recovering else min(proms)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: detected lanes vs prominence, one line per contour
    xs = sorted(proms)
    for r in rows:
        axA.plot(xs, [r["counts"][p] for p in xs], marker="o", label=f"contour {r['contour']} (n={r['n_veh']})")
    axA.set_xlabel("peak_prominence  (θ_prominence)")
    axA.set_ylabel("detected lanes")
    axA.set_title(f"{camera}: detected lanes vs. peak prominence")
    axA.invert_xaxis()  # more sensitive (lower prominence) to the right
    axA.grid(True, alpha=0.3)
    axA.legend(fontsize=8)

    # Panel B: smoothed histogram of the focus contour, peaks at p_hi vs p_lo
    bc, sm = focus["bin_centers"], focus["smoothed"]
    axB.plot(bc, sm, color="0.4", lw=2, label="smoothed histogram")
    pk_hi, pk_lo = focus["peaks_by_p"][p_hi], focus["peaks_by_p"][p_lo]
    axB.plot(bc[pk_hi], sm[pk_hi], "v", ms=12, color="tab:red",
             label=f"peaks @ p={p_hi}  (n={len(pk_hi)})")
    axB.plot(bc[pk_lo], sm[pk_lo], "^", ms=9, color="tab:green",
             label=f"peaks @ p={p_lo}  (n={len(pk_lo)})")
    recovered = [i for i in pk_lo if i not in set(pk_hi.tolist())]
    if recovered:
        i0 = max(recovered, key=lambda i: sm[i])  # most salient recovered peak
        _allpk, _allpr = find_peaks(sm, prominence=0)
        prom = dict(zip(_allpk.tolist(), _allpr["prominences"])).get(i0, float("nan"))
        x_off = bc[i0] - 0.30 * (bc[-1] - bc[0])
        axB.annotate(
            f"recovered lane\nheight {sm[i0]:.2f}, prominence {prom:.2f}\n"
            f"(< {p_hi} cutoff -> dropped; kept at p={p_lo})",
            xy=(bc[i0], sm[i0]), xytext=(x_off, sm[i0] * 0.92), ha="left",
            fontsize=8.5, color="tab:green",
            arrowprops=dict(arrowstyle="->", color="tab:green"))
    axB.set_xlabel("lateral position (x_gps)")
    axB.set_ylabel("smoothed vehicle count")
    axB.set_title(f"contour {focus['contour']}: recovered lane(s) at lower prominence")
    axB.grid(True, alpha=0.3)
    axB.legend(fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to {out_path}  (focus contour {focus['contour']}: "
          f"{focus['counts'][p_hi]} lanes @ p={p_hi} -> {focus['counts'][p_lo]} @ p={p_lo})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default="US12_Monona", help="camera location name")
    ap.add_argument("--results-root", default="results/federated", help="dir holding <camera>/federated_trajectory_clustering.csv")
    ap.add_argument("--proms", default="1.0,0.75,0.5,0.3", help="comma-separated prominence values to sweep")
    ap.add_argument("--sigma", type=float, default=2.0, help="Gaussian smoothing sigma (geo_learning theta['sigma'])")
    ap.add_argument("--bins", type=int, default=50, help="histogram bins (geo_learning uses 50)")
    ap.add_argument("--out", default=None, help="output figure path (default figs/prominence_sensitivity_<camera>.png)")
    args = ap.parse_args()

    csv = Path(args.results_root) / args.camera / "federated_trajectory_clustering.csv"
    if not csv.exists():
        raise SystemExit(f"Clustering CSV not found: {csv}\n"
                         f"Run the federated pipeline for {args.camera} first, or pass --results-root.")
    proms = [float(p) for p in args.proms.split(",")]
    out_path = Path(args.out) if args.out else Path("figs") / f"prominence_sensitivity_{args.camera}.png"

    df = pd.read_csv(csv)
    rows = analyze(df, proms, args.sigma, args.bins)
    if not rows:
        raise SystemExit("No contours with >=2 vehicles found.")
    print(f"Camera: {args.camera}  |  sigma={args.sigma}  bins={args.bins}  source={csv}\n")
    print_table(rows, proms)
    make_figure(rows, proms, args.camera, out_path)


if __name__ == "__main__":
    main()
