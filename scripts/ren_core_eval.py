"""External baseline: the trajectory-clustering CORE of Ren et al. 2014
(doi 10.1155/2014/156296), our implementation, on OUR detections.

The full Ren system includes its own detection front end (activity map,
virtual detection lines, Kalman graph tracking) with no public code; we
implement the comparable lane-extraction core faithfully to the paper's
description: incremental (leader-follower) clustering of tracked vehicle
trajectories under Hausdorff distance, followed by k-means style
reassignment, cluster mean polylines as lane centers. Run on the same
tracked trajectories our system consumes (detector-controlled), scored
under the same annotation protocol as every other table row.

The cluster admission threshold is SELF-CALIBRATED per camera as
TAU_WIDTHS x the median vehicle pixel width (never tuned on annotations).

Usage: uv run python scripts/ren_core_eval.py [--cameras ...]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial.distance import directed_hausdorff

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))

from adaptation_curve import build_args, load_annotation_lanes, annotation_gps, _to_m, M_LAT
from strategy_annotation_eval import prf
from LaneDetection.osm_extraction.utils import trajectory_calibration, interpolate_edge

SEEN = ['US12_Todd', 'US12_Monona', 'US12_Yahara', 'US12_CountyAB', 'US12_Mineral', 'US12_University']
UNSEEN = ['US12_Park', 'US12_Greenway', 'I43_Keefe', 'I43_Walnut']

K_RESAMPLE = 20        # points per trajectory for Hausdorff/averaging
MIN_TRACK_PTS = 10     # tracked frames for a usable trajectory
MIN_TRACK_LEN_PX = 60  # minimum travelled distance in pixels
TAU_WIDTHS = 1.75      # admission threshold = TAU_WIDTHS x median vehicle width
# Noise floor per lane: relative, since absolute counts are meaningless across
# sites whose traffic spans two orders of magnitude — a genuine lane on a
# monitored road carries a substantive share of the observed traffic. Chosen
# a priori (share-of-traffic rationale), not tuned against annotations.
MIN_CLUSTER_ABS = 5
MIN_CLUSTER_FRAC = 0.03
KMEANS_ITERS = 3       # refinement passes after the leader-follower pass


def resample(P, k=K_RESAMPLE):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    if d[-1] <= 0:
        return None
    t = np.linspace(0, d[-1], k)
    return np.stack([np.interp(t, d, P[:, 0]), np.interp(t, d, P[:, 1])], axis=1)


def hausdorff(A, B):
    return max(directed_hausdorff(A, B)[0], directed_hausdorff(B, A)[0])


def orient_like(P, ref):
    """Flip P if its travel direction opposes ref's (index-aligned averaging)."""
    return P if np.dot(P[-1] - P[0], ref[-1] - ref[0]) >= 0 else P[::-1]


def ren_core(trajs, tau):
    """Leader-follower incremental clustering + k-means refinement (Ren 2014 core)."""
    min_members = max(MIN_CLUSTER_ABS, int(MIN_CLUSTER_FRAC * len(trajs)))
    centers, members = [], []
    for P in trajs:  # temporal order, as in the incremental scheme
        if not centers:
            centers.append(P.copy()); members.append([P]); continue
        d = [hausdorff(P, C) for C in centers]
        j = int(np.argmin(d))
        if d[j] < tau:
            members[j].append(orient_like(P, centers[j]))
            centers[j] = np.mean(np.stack(members[j]), axis=0)
        else:
            centers.append(P.copy()); members.append([P])
    for _ in range(KMEANS_ITERS):  # k-means style reassignment on Hausdorff
        buckets = [[] for _ in centers]
        for P in trajs:
            j = int(np.argmin([hausdorff(P, C) for C in centers]))
            buckets[j].append(orient_like(P, centers[j]))
        centers = [np.mean(np.stack(b), axis=0) if b else c for b, c in zip(buckets, centers)]
        members = buckets
    return [c for c, m in zip(centers, members) if len(m) >= min_members]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cameras', nargs='+', default=SEEN + UNSEEN)
    ap.add_argument('--taus', type=float, nargs='+', default=[3.0, 5.0])
    ap.add_argument('--tau_widths', type=float, default=None, help='sensitivity override for TAU_WIDTHS')
    ap.add_argument('--floor_frac', type=float, default=None, help='sensitivity override for MIN_CLUSTER_FRAC')
    ap.add_argument('--out', default='results/ren_core', help='output dir (sensitivity runs must NOT clobber the canonical CSV)')
    opts = ap.parse_args()
    global TAU_WIDTHS, MIN_CLUSTER_FRAC
    if opts.tau_widths is not None:
        TAU_WIDTHS = opts.tau_widths
    if opts.floor_frac is not None:
        MIN_CLUSTER_FRAC = opts.floor_frac
    print(f'config: TAU_WIDTHS={TAU_WIDTHS} MIN_CLUSTER_FRAC={MIN_CLUSTER_FRAC}')
    args = build_args()
    out_dir = Path(opts.out); out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cam in opts.cameras:
        ann_px = load_annotation_lanes(cam)
        if not ann_px:
            print(f'{cam}: no annotation, skipped'); continue
        df = pl.read_csv(ROOT / 'results' / 'preprocess' / cam / 'trajectory.csv')
        trajs, widths = [], []
        for _, g in sorted(df.group_by('id'), key=lambda kv: kv[1]['frame_num'].min()):
            if len(g) < MIN_TRACK_PTS:
                continue
            P = np.stack([g['x'].to_numpy(), g['y'].to_numpy()], axis=1).astype(float)
            if np.linalg.norm(P[-1] - P[0]) < MIN_TRACK_LEN_PX:
                continue
            R = resample(P)
            if R is not None:
                trajs.append(R)
                widths.append(float(np.median(g['w'].to_numpy())))
        if len(trajs) < MIN_CLUSTER_ABS:
            print(f'{cam}: too few usable trajectories ({len(trajs)})'); continue
        tau = TAU_WIDTHS * float(np.median(widths))
        lanes_px = ren_core(trajs, tau)
        # pixel -> GPS through the site homography (same as every other row)
        ann_g = annotation_gps(cam, ann_px, args.dataset_path)
        lat0 = float(np.mean([a[:, 0].mean() for a in ann_g]))
        lon0 = float(np.mean([a[:, 1].mean() for a in ann_g]))
        m_lon = M_LAT * np.cos(np.deg2rad(lat0))
        ann_m = [_to_m(a, lat0, lon0, m_lon) for a in ann_g]
        det_m = []
        for P in lanes_px:
            dense = np.asarray(interpolate_edge(P, num_points=60), dtype=float)
            gps, _ = trajectory_calibration(dense, Path(args.dataset_path, '511calibration'), cam)
            det_m.append(_to_m(np.asarray(gps, float), lat0, lon0, m_lon))
        for tau_m in opts.taus:
            p, r, f, g, tp, nd, na = prf(det_m, ann_m, tau_m)
            rows.append(dict(camera=cam, split='seen' if cam in SEEN else 'unseen', tau=tau_m,
                             precision=p, recall=r, f1=f, geo_err_tp=g, tp=tp, n_det=nd, n_ann=na,
                             n_trajs=len(trajs), tau_px=round(tau, 1)))
            print(f'{cam:<16} tau={tau_m:.0f}  P={p:.2f} R={r:.2f} F1={f:.2f}  det={nd} ann={na}  (trajs={len(trajs)}, tau_px={tau:.0f})')

    with open(out_dir / 'ren_core_eval.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out_dir}/ren_core_eval.csv")
    for tau_m in opts.taus:
        for split in ['seen', 'unseen']:
            rs = [r for r in rows if r['split'] == split and r['tau'] == tau_m]
            if not rs:
                continue
            tp = sum(r['tp'] for r in rs); nd = sum(r['n_det'] for r in rs); na = sum(r['n_ann'] for r in rs)
            P = tp / nd if nd else 0; R = tp / na if na else 0
            F = 2 * P * R / (P + R) if (P + R) else 0
            print(f'REN-CORE {split:<7} tau={tau_m:.0f}: P={P:.3f} R={R:.3f} microF1={F:.3f}')


if __name__ == '__main__':
    main()
