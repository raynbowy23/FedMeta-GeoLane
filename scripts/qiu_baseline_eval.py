"""External baseline: Qiu et al. 2024 multiple-ROI lane learning, evaluated
under our annotation protocol on OUR detections (detector-controlled).

Their Cycle_learning_multiple_ROI consumes the same accumulated-detections
format our preprocess stores (shared codebase lineage: collect_cars 6-tuples),
so we replay their cycle protocol offline on results/preprocess/<cam> data,
extract their final lane geometry in pixel space, project it through the same
site homography as our detections and the annotations (calibration cancels),
and score precision/recall/F1 with the same Hungarian matcher as every other
row of the table.

Their code is used from an external clone (no license file in the repo, so it
is not vendored). Clone: https://github.com/qiumei1101/Multiple_ROI_lane_learning_system_for_Highway

Usage:
  uv run python scripts/qiu_baseline_eval.py --qiu_path <clone dir> [--cameras ...] [--debug]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))

from adaptation_curve import (build_args, load_annotation_lanes, annotation_gps,
                              _to_m, M_LAT)
from strategy_annotation_eval import prf
from LaneDetection.osm_extraction.utils import trajectory_calibration, interpolate_edge

SEEN = ['US12_Todd', 'US12_Monona', 'US12_Yahara', 'US12_CountyAB', 'US12_Mineral', 'US12_University']
UNSEEN = ['US12_Park', 'US12_Greenway', 'I43_Keefe', 'I43_Walnut']


def polyline_from_boundaries(lane_dict, W, H):
    """Their lane geometry parameterizes boundary x per y row via per-ROI
    polynomials, which extrapolate wildly outside the ROI's valid rows (x in
    the thousands on a 1920 px frame). Keep only rows where both boundaries
    land in-frame, so their method is scored on its valid geometry rather
    than on extrapolation artifacts. Centerline = boundary midpoint per row."""
    pts = []
    for y, b in lane_dict.items():
        try:
            yv = float(y)
            if not (0 <= yv < H) or not isinstance(b, dict):
                continue
            lx = b.get('left_boundary_x'); rx = b.get('right_boundary_x')
            if lx is None or rx is None:
                continue
            lx, rx = float(lx), float(rx)
            if not (0 <= lx <= W and 0 <= rx <= W):
                continue
            pts.append(((lx + rx) / 2.0, yv))
        except (TypeError, ValueError):
            continue
    pts.sort(key=lambda p: p[1])
    return np.array(pts, dtype=float)


def boundaries_from_lane(lane_dict, W, H):
    """One Qiu lane dict -> (center, left, right) pixel polylines, in-frame
    rows only (their per-ROI polynomials extrapolate wildly outside)."""
    c, l, r = [], [], []
    for y, b in lane_dict.items():
        try:
            yv = float(y)
            if not (0 <= yv < H) or not isinstance(b, dict):
                continue
            lx, rx = b.get('left_boundary_x'), b.get('right_boundary_x')
            if lx is None or rx is None:
                continue
            lx, rx = float(lx), float(rx)
            if not (0 <= lx <= W and 0 <= rx <= W):
                continue
            c.append(((lx + rx) / 2.0, yv)); l.append((lx, yv)); r.append((rx, yv))
        except (TypeError, ValueError):
            continue
    srt = lambda P: np.array(sorted(P, key=lambda p: p[1]), dtype=float)
    return srt(c), srt(l), srt(r)


def run_qiu(cam, qiu_path, workdir, n_cycles=3, debug=False, return_boundaries=False):
    """Replay their cycle protocol on our saved detections. Returns a list of
    centerline pixel polylines (or {center,left,right} dicts when
    return_boundaries=True), or raises on failure."""
    sys.path.insert(0, str(qiu_path))
    import matplotlib
    matplotlib.use('Agg')
    # their code targets numpy<1.24 (np.float/np.int aliases); shim instead of
    # editing the external clone
    for alias, typ in [('float', float), ('int', int), ('bool', bool), ('object', object)]:
        if not hasattr(np, alias):
            setattr(np, alias, typ)
    from Cycle_Learning_multiple_ROI import Cycle_learning_multiple_ROI

    pre = ROOT / 'results' / 'preprocess' / cam
    frame = np.load(pre / 'last_frame.npy')
    H, W = frame.shape[:2]
    # their heatmap indexes dicts by integer pixel row (their detector stored
    # ints); our preprocess stores floats — cast and clip to frame bounds
    cars = []
    for x, y, w, h, fid, conf in np.load(pre / 'collect_cars.npy').tolist():
        cars.append((int(np.clip(x, 0, W - 1)), int(np.clip(y, 0, H - 1)),
                     int(w), int(h), int(fid), float(conf)))
    cars.sort(key=lambda v: v[4])  # chronological by frame id

    out = None
    fp = workdir / cam
    fp.mkdir(parents=True, exist_ok=True)
    for cyc in range(1, n_cycles + 1):
        chunk = cars[: max(1, len(cars) * cyc // n_cycles)]
        out = Cycle_learning_multiple_ROI(chunk, frame.copy(), str(fp), cyc, 120)
    (_, road_status, lane_annotations, lane_centers, ROIs,
     all_centers_x, roi_idx, lane_boundaries, lines) = out
    if debug:
        print(f'  status={road_status}, lanes annotated={lane_annotations}')
        print(f'  boundaries type={type(lane_boundaries)}, len={len(lane_boundaries) if hasattr(lane_boundaries, "__len__") else "?"}')
        if isinstance(lane_boundaries, (list, tuple)) and lane_boundaries:
            print(f'  first element type={type(lane_boundaries[0])}: {str(lane_boundaries[0])[:300]}')
        elif isinstance(lane_boundaries, dict):
            k = next(iter(lane_boundaries))
            print(f'  first key={k}: {str(lane_boundaries[k])[:300]}')

    polylines = []
    items = lane_boundaries.values() if isinstance(lane_boundaries, dict) else lane_boundaries
    for lane in items:
        if isinstance(lane, dict):
            if return_boundaries:
                c, l, r = boundaries_from_lane(lane, W, H)
                if len(c) >= 2:
                    polylines.append(dict(center=c, left=l, right=r))
            else:
                P = polyline_from_boundaries(lane, W, H)
                if len(P) >= 2:
                    polylines.append(P)
    return polylines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qiu_path', required=True)
    ap.add_argument('--cameras', nargs='+', default=SEEN + UNSEEN)
    ap.add_argument('--taus', type=float, nargs='+', default=[3.0, 5.0])
    ap.add_argument('--n_cycles', type=int, default=3)
    ap.add_argument('--debug', action='store_true')
    ap.add_argument('--out', default='results/qiu_baseline', help='output dir (diagnostic runs must NOT clobber the canonical CSV)')
    opts = ap.parse_args()
    args = build_args()
    workdir = Path(opts.out); workdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cam in opts.cameras:
        ann_px = load_annotation_lanes(cam)
        if not ann_px:
            print(f'{cam}: no annotation, skipped'); continue
        try:
            polylines = run_qiu(cam, Path(opts.qiu_path), workdir, opts.n_cycles, opts.debug)
        except Exception as e:
            print(f'{cam}: QIU FAILED — {type(e).__name__}: {e}')
            if opts.debug:
                import traceback; traceback.print_exc()
            rows.append(dict(camera=cam, split='seen' if cam in SEEN else 'unseen',
                             tau=np.nan, precision=0, recall=0, f1=0, geo_err_tp=np.nan, tp=0, n_det=0,
                             n_ann=len(ann_px), note=f'failed:{type(e).__name__}'))
            continue
        # pixel -> GPS through the site homography (same as annotations/detections)
        ann_g = annotation_gps(cam, ann_px, args.dataset_path)
        lat0 = float(np.mean([a[:, 0].mean() for a in ann_g]))
        lon0 = float(np.mean([a[:, 1].mean() for a in ann_g]))
        m_lon = M_LAT * np.cos(np.deg2rad(lat0))
        ann_m = [_to_m(a, lat0, lon0, m_lon) for a in ann_g]
        det_m = []
        for P in polylines:
            dense = np.asarray(interpolate_edge(P, num_points=60), dtype=float)
            gps, _ = trajectory_calibration(dense, Path(args.dataset_path, '511calibration'), cam)
            det_m.append(_to_m(np.asarray(gps, float), lat0, lon0, m_lon))
        for tau in opts.taus:
            p, r, f, g, tp, nd, na = prf(det_m, ann_m, tau)
            rows.append(dict(camera=cam, split='seen' if cam in SEEN else 'unseen', tau=tau,
                             precision=p, recall=r, f1=f, geo_err_tp=g, tp=tp, n_det=nd, n_ann=na, note=''))
            print(f'{cam:<16} tau={tau:.0f}  P={p:.2f} R={r:.2f} F1={f:.2f}  det={nd} ann={na}')

    with open(workdir / 'qiu_baseline_eval.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {workdir}/qiu_baseline_eval.csv")
    for tau in opts.taus:
        for split in ['seen', 'unseen']:
            rs = [r for r in rows if r['split'] == split and r['tau'] == tau]
            if not rs:
                continue
            tp = sum(r['tp'] for r in rs); nd = sum(r['n_det'] for r in rs); na = sum(r['n_ann'] for r in rs)
            P = tp / nd if nd else 0; R = tp / na if na else 0
            F = 2 * P * R / (P + R) if (P + R) else 0
            fails = sum(1 for r in rs if r['note'])
            print(f'QIU {split:<7} tau={tau:.0f}: P={P:.3f} R={R:.3f} microF1={F:.3f}  ({fails} site failures counted as zero recall)')


if __name__ == '__main__':
    main()
