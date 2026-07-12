"""OSM-as-a-row: score the public map's lane geometry itself against the human
annotations, same Hungarian matcher and thresholds as every method row. Turns
the map-vs-method decomposition into a table row: the reference everyone would
use by default, evaluated as if it were a detector.

Usage: uv run python scripts/osm_row_eval.py [--cameras ...]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))

from adaptation_curve import (build_args, build_processed, load_annotation_lanes,
                              annotation_gps, _to_m, M_LAT)
from strategy_annotation_eval import prf
from LaneDetection.osm_extraction.utils import interpolate_edge

SEEN = ['US12_Todd', 'US12_Monona', 'US12_Yahara', 'US12_CountyAB', 'US12_Mineral', 'US12_University']
UNSEEN = ['US12_Park', 'US12_Greenway', 'I43_Keefe', 'I43_Walnut']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cameras', nargs='+', default=SEEN + UNSEEN)
    ap.add_argument('--taus', type=float, nargs='+', default=[3.0, 5.0])
    ap.add_argument('--out', default='results/osm_row', help='output dir')
    opts = ap.parse_args()
    args = build_args()
    sfp = Path(args.saving_path, 'federated')
    out_dir = Path(opts.out); out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cam in opts.cameras:
        ann_px = load_annotation_lanes(cam)
        if not ann_px:
            print(f'{cam}: no annotation, skipped'); continue
        processed = build_processed(args, cam, sfp)
        ann_g = annotation_gps(cam, ann_px, args.dataset_path)
        lat0 = float(np.mean([a[:, 0].mean() for a in ann_g]))
        lon0 = float(np.mean([a[:, 1].mean() for a in ann_g]))
        m_lon = M_LAT * np.cos(np.deg2rad(lat0))
        ann_m = [_to_m(a, lat0, lon0, m_lon) for a in ann_g]
        # the SUMO/OSM reference lanes, exactly as the training loss sees them
        sumo_node, _ = processed.get('sumo_graph', ([], []))
        ref_m = []
        for group in sumo_node:
            for line in group:
                a = np.asarray(line, dtype=float)
                if len(a) >= 2 and np.abs(np.diff(a, axis=0)).sum() > 0:
                    dense = np.asarray(interpolate_edge(a, num_points=60), dtype=float)
                    ref_m.append(_to_m(dense, lat0, lon0, m_lon))
        for tau in opts.taus:
            p, r, f, g, tp, nd, na = prf(ref_m, ann_m, tau)
            rows.append(dict(camera=cam, split='seen' if cam in SEEN else 'unseen', tau=tau,
                             precision=p, recall=r, f1=f, geo_err_tp=g, tp=tp, n_det=nd, n_ann=na))
            print(f'{cam:<16} tau={tau:.0f}  P={p:.2f} R={r:.2f} F1={f:.2f}  osm_lanes={nd} ann={na}')

    with open(out_dir / 'osm_row_eval.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out_dir}/osm_row_eval.csv")
    for tau in opts.taus:
        for split in ['seen', 'unseen']:
            rs = [r for r in rows if r['split'] == split and r['tau'] == tau]
            if not rs:
                continue
            tp = sum(r['tp'] for r in rs); nd = sum(r['n_det'] for r in rs); na = sum(r['n_ann'] for r in rs)
            P = tp / nd if nd else 0; R = tp / na if na else 0
            F = 2 * P * R / (P + R) if (P + R) else 0
            print(f'OSM {split:<7} tau={tau:.0f}: P={P:.3f} R={R:.3f} microF1={F:.3f}')


if __name__ == '__main__':
    main()
