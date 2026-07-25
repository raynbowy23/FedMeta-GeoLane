"""Self-calibration prototype: registration of the detected lane bundle to the
map reference, validated leave-one-lane-out.

Mechanism: a systematic offset/rotation between the detected trajectory bundle
and the map-derived reference is calibration error, not lane geometry. A rigid
transform (rotation + translation, Umeyama) estimated from the bundle recovers
it. Validation is leave-one-lane-out so the correction never grades itself:
for each detected lane, the transform is estimated from the OTHER lanes only
and evaluated on the held-out lane's centerline deviation.

Detection uses the fixed baseline theta so the demonstrated correction is
model-independent (it composes with any strategy).

Usage: uv run python scripts/self_calibration.py [--cameras CAM ...]
Output: per-camera raw vs corrected held-out deviation + estimated calibration
error (offset meters, rotation degrees), CSV in results/self_calibration/.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from adaptation_curve import BASELINE_THETA, build_args, build_processed
from LaneDetection.lane_detection.geo_learning import GeometricLearning
from LaneDetection.osm_extraction.utils import interpolate_edge

M_LAT = 111_320.0


def to_m(pts, lat0, lon0, m_lon):
    """(N,2) lat/lon degrees -> local meters (x=east, y=north)."""
    return np.stack([(pts[:, 1] - lon0) * m_lon, (pts[:, 0] - lat0) * M_LAT], axis=1)


def nearest_points(P, Q):
    """For each point of P, its nearest point on polyline Q. Returns (targets, dists)."""
    d = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2)
    idx = d.argmin(axis=1)
    return Q[idx], d[np.arange(len(P)), idx]


def umeyama_rigid(src, dst):
    """Rigid transform (R, t) minimizing ||R@src + t - dst||."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    cov = (dst - mu_d).T @ (src - mu_s) / len(src)
    U, _, Vt = np.linalg.svd(cov)
    S = np.diag([1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ S @ Vt
    t = mu_d - R @ mu_s
    return R, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cameras', nargs='+',
                    default=['US12_Whitney', 'US12_Todd', 'US12_Mineral', 'US12_Stoughton', 'US12_Monona'])
    ap.add_argument('--corr_cap_m', type=float, default=25.0,
                    help='Discard correspondences farther than this (outlier/mismatch guard)')
    opts = ap.parse_args()

    args = build_args()
    out_dir = Path('results/self_calibration')
    out_dir.mkdir(parents=True, exist_ok=True)
    saving_file_path = Path(args.saving_path, 'federated')
    rows = []

    for cam in opts.cameras:
        processed = build_processed(args, cam, saving_file_path)
        geo = GeometricLearning(args, saving_file_path)
        for k, v in BASELINE_THETA.items():
            geo.theta[k] = torch.tensor(float(v))
        traj_df, bounds = geo.run(c_epoch=0, g_epoch=0, traj_df=processed['gps_df'],
                                  camera_loc=cam, trial='0', is_save=False)
        detected = [d['center'] for by_id in bounds.values() for d in by_id.values()]
        sumo_node, _ = processed.get('sumo_graph', ([], []))
        refs = []
        for group in sumo_node:
            for line in group:
                a = np.asarray(line, dtype=float)
                if len(a) >= 2 and np.abs(np.diff(a, axis=0)).sum() > 0:
                    refs.append(np.asarray(interpolate_edge(a, num_points=240), dtype=float))
        if len(detected) < 3 or not refs:
            print(f'{cam}: skipped (lanes={len(detected)}, refs={len(refs)})')
            continue

        lat0 = float(np.mean([d[:, 0].mean() for d in detected]))
        lon0 = float(np.mean([d[:, 1].mean() for d in detected]))
        m_lon = M_LAT * np.cos(np.deg2rad(lat0))
        det_m = [to_m(np.asarray(d, dtype=float), lat0, lon0, m_lon) for d in detected]
        ref_m = [to_m(r, lat0, lon0, m_lon) for r in refs]

        # Match each detected lane to its reference by mean nearest-point distance
        pairs = []
        for D in det_m:
            dists = [nearest_points(D, R)[1].mean() for R in ref_m]
            pairs.append(ref_m[int(np.argmin(dists))])

        # Leave-one-lane-out: transform from the other lanes, evaluated on lane i
        raw_all, cor_all = [], []
        for i in range(len(det_m)):
            src, dst = [], []
            for j in range(len(det_m)):
                if j == i:
                    continue
                tgt, dist = nearest_points(det_m[j], pairs[j])
                keep = dist < opts.corr_cap_m
                src.append(det_m[j][keep])
                dst.append(tgt[keep])
            src, dst = np.concatenate(src), np.concatenate(dst)
            if len(src) < 20:
                continue
            R, t = umeyama_rigid(src, dst)
            held = det_m[i]
            held_corrected = held @ R.T + t
            raw_all.append(nearest_points(held, pairs[i])[1].mean())
            cor_all.append(nearest_points(held_corrected, pairs[i])[1].mean())

        # Whole-bundle transform = the site's estimated calibration error
        src, dst = [], []
        for j in range(len(det_m)):
            tgt, dist = nearest_points(det_m[j], pairs[j])
            keep = dist < opts.corr_cap_m
            src.append(det_m[j][keep])
            dst.append(tgt[keep])
        R, t = umeyama_rigid(np.concatenate(src), np.concatenate(dst))
        rot_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
        offset = float(np.linalg.norm(t))

        raw, cor = float(np.mean(raw_all)), float(np.mean(cor_all))
        rows.append(dict(camera=cam, lanes=len(det_m), raw_m=raw, corrected_m=cor,
                         improvement=raw / cor if cor > 0 else float('inf'),
                         est_offset_m=offset, est_rot_deg=rot_deg))
        print(f'{cam}: held-out deviation {raw:.2f} -> {cor:.2f} m ({raw/cor:.2f}x), '
              f'estimated calibration error: offset {offset:.2f} m, rotation {rot_deg:.2f} deg '
              f'({len(det_m)} lanes)')

    out = out_dir / 'self_calibration.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
