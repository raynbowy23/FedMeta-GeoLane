"""Per-strategy lane-detection F1 against human annotations (the honest primary
metric). For each strategy we take the theta it actually deploys at a site (the
best theta from its training buffer, i.e. what _best_theta_from_buffer returns;
baseline uses the fixed theta), run detection, and score detected lanes against
the lanelet annotations with one-to-one Hungarian matching.

precision = matched / detected, recall = matched / annotated, F1 = harmonic mean,
at a lateral threshold TAU (meters). Calibration cancels because detections and
annotations pass through the same homography. Annotations are EVALUATION ONLY.

Deployment-only, trains nothing. Run AFTER a full baseline/meta/federated pass.
Usage: uv run python scripts/strategy_annotation_eval.py [--taus 3 5] [--cameras ...]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adaptation_curve import (BASELINE_THETA, build_args, build_processed, evaluate,
                              load_annotation_lanes, annotation_gps, _to_m, M_LAT)
from LaneDetection.lane_detection.geo_learning import GeometricLearning

SEEN = ['US12_Todd', 'US12_Monona', 'US12_Yahara', 'US12_Stoughton',
        'US12_CountyAB', 'US12_Mineral', 'US12_University']
FED_CKPT = 'results/federated/training_results/federated_model_perfedavg.pth'


def deployed_theta(strategy, cam, fed_ckpt=FED_CKPT, meta_dir='results/meta/training_results'):
    """The theta the strategy deploys at this seen site (best from its buffer).
    fed_ckpt / meta_dir let this point at a scarcity-sweep snapshot dir."""
    if strategy == 'baseline':
        return dict(BASELINE_THETA)
    if strategy == 'meta':
        p = Path(meta_dir) / f'meta_model_{cam}.pth'
        if not p.exists():
            return None
        buf = torch.load(p, map_location='cpu', weights_only=False).get('client_buffer', [])
    else:  # federated
        ck = torch.load(fed_ckpt, map_location='cpu', weights_only=False)
        buf = ck.get('client_data_buffer', {}).get(cam, [])
    fin = [b for b in buf if np.isfinite(b.get('best_loss', float('inf'))) and b.get('best_theta')]
    if not fin:
        return None
    return min(fin, key=lambda b: b['best_loss'])['best_theta']


def bounds_to_centerlines_m(bounds, ann_m_ref):
    det = [np.asarray(d['center'], float) for by in bounds.values() for d in by.values()]
    if not det:
        return []
    lat0, lon0 = ann_m_ref
    m_lon = M_LAT * np.cos(np.deg2rad(lat0))
    return [_to_m(d, lat0, lon0, m_lon) for d in det]


def det_to_ann_dist(D, A):
    """Mean nearest-point distance from detected lane D to annotation A (meters).
    Directed D->A, robust to the annotation covering a longer extent, same basis
    as reference_audit's detection error."""
    d = np.linalg.norm(D[:, None, :] - A[None, :, :], axis=2)
    return d.min(axis=1).mean()


def prf(det_m, ann_m, tau):
    """One-to-one Hungarian match; pairs with cost < tau are true positives."""
    if not det_m or not ann_m:
        return 0.0, 0.0, 0.0, float('nan'), 0, len(det_m), len(ann_m)
    C = np.array([[det_to_ann_dist(D, A) for A in ann_m] for D in det_m])
    ri, ci = linear_sum_assignment(C)
    tp_costs = [C[i, j] for i, j in zip(ri, ci) if C[i, j] < tau]
    tp = len(tp_costs)
    precision = tp / len(det_m)
    recall = tp / len(ann_m)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    geo = float(np.mean(tp_costs)) if tp_costs else float('nan')
    return precision, recall, f1, geo, tp, len(det_m), len(ann_m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--taus', type=float, nargs='+', default=[3.0, 5.0])
    ap.add_argument('--cameras', nargs='+', default=SEEN)
    ap.add_argument('--fed_ckpt', default=FED_CKPT, help='federated checkpoint (point at a scarcity snapshot)')
    ap.add_argument('--meta_ckpt_dir', default='results/meta/training_results', help='dir of meta_model_<cam>.pth')
    ap.add_argument('--tag', default='', help='label column for the output rows (e.g. f0.25_s43)')
    opts = ap.parse_args()
    args = build_args()
    sfp = Path(args.saving_path, 'federated')
    strategies = ['baseline', 'meta', 'federated']

    rows = []
    for cam in opts.cameras:
        ann_px = load_annotation_lanes(cam)
        if not ann_px:
            print(f'{cam}: no annotation, skipped')
            continue
        processed = build_processed(args, cam, sfp)
        ann_gps = annotation_gps(cam, ann_px, args.dataset_path)
        # shared meter frame anchored on the annotations
        lat0 = float(np.mean([a[:, 0].mean() for a in ann_gps]))
        lon0 = float(np.mean([a[:, 1].mean() for a in ann_gps]))
        m_lon = M_LAT * np.cos(np.deg2rad(lat0))
        ann_m = [_to_m(a, lat0, lon0, m_lon) for a in ann_gps]
        geo = GeometricLearning(args, sfp)
        for strat in strategies:
            theta = deployed_theta(strat, cam, opts.fed_ckpt, opts.meta_ckpt_dir)
            if theta is None:
                print(f'  {cam} {strat}: no deployed theta, skipped')
                continue
            _, _, bounds = evaluate(geo, processed, cam, theta)
            det_m = bounds_to_centerlines_m(bounds, (lat0, lon0))
            for tau in opts.taus:
                p, r, f, g, tp, nd, na = prf(det_m, ann_m, tau)
                rows.append(dict(tag=opts.tag, camera=cam, strategy=strat, tau=tau, precision=p,
                                 recall=r, f1=f, geo_err_tp=g, tp=tp, n_det=nd, n_ann=na))
                print(f'  {cam:<16} {strat:<10} tau={tau:.0f}  P={p:.2f} R={r:.2f} F1={f:.2f}  '
                      f'det={nd} ann={na} tp={tp}  geoErr={g:.2f}m')

    out = Path('results/strategy_annotation_eval'); out.mkdir(parents=True, exist_ok=True)
    with open(out / 'strategy_annotation_eval.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f'\nwrote {out}/strategy_annotation_eval.csv ({len(rows)} rows)')

    # micro-average across cameras (sum tp/det/ann), and macro F1
    print('\n=== micro-averaged over cameras ===')
    for tau in opts.taus:
        print(f'  TAU = {tau:.0f} m')
        for strat in strategies:
            rs = [r for r in rows if r['strategy'] == strat and r['tau'] == tau]
            if not rs:
                continue
            tp = sum(r['tp'] for r in rs); nd = sum(r['n_det'] for r in rs); na = sum(r['n_ann'] for r in rs)
            P = tp / nd if nd else 0; R = tp / na if na else 0
            F = 2 * P * R / (P + R) if (P + R) else 0
            macroF = np.mean([r['f1'] for r in rs])
            geo = np.nanmean([r['geo_err_tp'] for r in rs])
            print(f'    {strat:<10} P={P:.3f} R={R:.3f}  microF1={F:.3f}  macroF1={macroF:.3f}  geoErr(TP)={geo:.2f}m')


if __name__ == '__main__':
    main()
