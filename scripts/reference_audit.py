"""Reference zero-point audit against human lane annotations.

Decomposes each site's apparent geometric error into two parts using the
lanelet_annotator ground truth (dataset/preprocess/<cam>/annotation.json,
absolute pixel waypoints):

  detection error   = detected centerlines vs annotation lanes, BOTH projected
                      through the same pixel->GPS homography, so calibration
                      error cancels and this isolates what the detector gets
                      wrong about the road it can see.
  reference offset  = annotation lanes vs the OSM/SUMO reference lanes the
                      paper evaluates against. Large values mean the reference,
                      not the detector, owns the site's error.

Detection uses the fixed baseline theta (model-independent). Annotations are
used for EVALUATION ONLY; the method stays weakly supervised.

Usage: uv run python scripts/reference_audit.py [--cameras CAM ...]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from adaptation_curve import BASELINE_THETA, build_args, build_processed
from LaneDetection.lane_detection.geo_learning import GeometricLearning
from LaneDetection.osm_extraction.utils import interpolate_edge, trajectory_calibration
from self_calibration import M_LAT, nearest_points, to_m


def load_annotation_lanes(cam):
    p = ROOT / 'dataset' / 'preprocess' / cam / 'annotation.json'
    if not p.exists():
        return None
    doc = json.load(open(p))
    lanes = []
    for g in doc.get('lane_groups', []):
        for lane in g.get('lanes', []):
            pts = np.array([[w['x'], w['y']] for w in lane.get('waypoints', [])], dtype=float)
            if len(pts) >= 2:
                lanes.append(pts)
    return lanes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cameras', nargs='+',
                    default=['US12_Todd', 'US12_Monona', 'US12_Yahara', 'US12_Stoughton', 'US12_Mineral',
                             'US12_University', 'US12_CountyAB', 'US12_Park', 'US12_Greenway',
                             'I43_Keefe', 'I43_Walnut', 'US12_Whitney'])
    ap.add_argument('--match_cap_m', type=float, default=15.0)
    opts = ap.parse_args()

    args = build_args()
    sfp = Path(args.saving_path, 'federated')
    out_rows = []

    for cam in opts.cameras:
        ann_px = load_annotation_lanes(cam)
        if not ann_px:
            print(f'{cam:<16} no annotation')
            continue
        processed = build_processed(args, cam, sfp)
        geo = GeometricLearning(args, sfp)
        for k, v in BASELINE_THETA.items():
            geo.theta[k] = torch.tensor(float(v))
        _, bounds = geo.run(0, 0, processed['gps_df'], cam, '0', False)
        detected = [np.asarray(d['center'], float) for by_id in bounds.values() for d in by_id.values()]

        # Annotation pixels -> GPS through the SAME homography as trajectories
        ann_gps = []
        for pts in ann_px:
            dense = np.asarray(interpolate_edge(pts, num_points=60), dtype=float)
            gps, _ = trajectory_calibration(dense, Path(args.dataset_path, '511calibration'), cam)
            ann_gps.append(np.asarray(gps, dtype=float))

        sumo_node, _ = processed.get('sumo_graph', ([], []))
        refs = []
        for g in sumo_node:
            for line in g:
                a = np.asarray(line, dtype=float)
                if len(a) >= 2 and np.abs(np.diff(a, axis=0)).sum() > 0:
                    refs.append(np.asarray(interpolate_edge(a, num_points=240), dtype=float))

        all_pts = ann_gps + detected
        lat0 = float(np.mean([a[:, 0].mean() for a in all_pts]))
        lon0 = float(np.mean([a[:, 1].mean() for a in all_pts]))
        m_lon = M_LAT * np.cos(np.deg2rad(lat0))
        ann_m = [to_m(a, lat0, lon0, m_lon) for a in ann_gps]
        det_m = [to_m(d, lat0, lon0, m_lon) for d in detected]
        ref_m = [to_m(r, lat0, lon0, m_lon) for r in refs]

        # detection error: each detected lane vs its nearest annotation lane
        det_err = []
        for D in det_m:
            cands = [nearest_points(D, A)[1].mean() for A in ann_m]
            best = min(cands)
            if best < opts.match_cap_m:
                det_err.append(best)
        # reference offset: each annotation lane vs its nearest OSM reference
        ref_err = []
        for A in ann_m:
            if not ref_m:
                continue
            cands = [nearest_points(A, R)[1].mean() for R in ref_m]
            ref_err.append(min(cands))

        de = float(np.mean(det_err)) if det_err else float('nan')
        re = float(np.mean(ref_err)) if ref_err else float('nan')
        out_rows.append(dict(camera=cam, ann_lanes=len(ann_m), det_lanes=len(det_m),
                             det_matched=len(det_err), detection_err_m=de, reference_offset_m=re))
        print(f'{cam:<16} annotation lanes={len(ann_m):2d} detected={len(det_m):2d} '
              f'| DETECTION err {de:6.2f} m ({len(det_err)} matched) | OSM REFERENCE offset {re:6.2f} m')

    out = Path('results/reference_audit'); out.mkdir(parents=True, exist_ok=True)
    with open(out / 'reference_audit.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    print(f'\nwrote {out / "reference_audit.csv"}')
    d = [r['detection_err_m'] for r in out_rows if np.isfinite(r['detection_err_m'])]
    r = [r['reference_offset_m'] for r in out_rows if np.isfinite(r['reference_offset_m'])]
    print(f'MEANS across sites: detection {np.mean(d):.2f} m vs OSM reference offset {np.mean(r):.2f} m')


if __name__ == '__main__':
    main()
