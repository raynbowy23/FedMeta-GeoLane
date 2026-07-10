"""Scarcity analysis: annotation-F1 / recall vs data fraction, meta vs federated.

Scores every scarcity-sweep snapshot (results/_scarcity/<model>_f<frac>_s<seed>)
against the human annotations. Seen sites deploy their buffer-best theta; unseen
sites deploy the model prediction (federated = global model, meta = nearest-scene
donor), which is where federation's low-data advantage should show. Processed
data is built once per camera and reused across snapshots.

The hypothesis: as data shrinks, the federated curve should hold up better than
the per-camera curve (federation borrows strength across sites). GT-grounded, so
it can't be gamed by the OSM reference.

Usage: uv run python scripts/scarcity_analysis.py [--taus 5] [--fracs ...] [--seeds ...]
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))

from adaptation_curve import (build_args, build_processed, evaluate, load_annotation_lanes,
                              annotation_gps, _to_m, M_LAT, load_model, predict_theta)
from strategy_annotation_eval import prf, bounds_to_centerlines_m, deployed_theta
from LaneDetection.lane_detection.utils import SceneFeatureExtractor
from LaneDetection.lane_detection.geo_learning import GeometricLearning

SEEN = ['US12_Todd', 'US12_Monona', 'US12_Yahara', 'US12_CountyAB', 'US12_Mineral', 'US12_University']
UNSEEN = ['US12_Park', 'US12_Greenway', 'I43_Keefe', 'I43_Walnut']


def nearest_donor_theta(snap, cam_feats):
    """Meta unseen deployment: nearest-scene seen donor predicts theta."""
    best_d, best_p = float('inf'), None
    for s in SEEN:
        p = snap / f'meta_model_{s}.pth'
        if not p.exists():
            continue
        buf = torch.load(p, map_location='cpu', weights_only=False).get('client_buffer', [])
        fin = [b for b in buf if b.get('scene_features') is not None]
        if not fin:
            continue
        df = torch.stack([b['scene_features'].squeeze() for b in fin]).mean(0)
        d = float(torch.norm(df - cam_feats.squeeze().cpu()))
        if d < best_d:
            best_d, best_p = d, p
    return predict_theta(load_model(best_p), cam_feats) if best_p else None


def unseen_theta(model, snap, cam_feats):
    if model == 'federated':
        return predict_theta(load_model(snap / 'federated_model.pth'), cam_feats)
    return nearest_donor_theta(snap, cam_feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--taus', type=float, nargs='+', default=[5.0])
    ap.add_argument('--fracs', type=float, nargs='+', default=[1.0, 0.5, 0.25, 0.1])
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--cameras', nargs='+', default=UNSEEN + SEEN)
    opts = ap.parse_args()
    args = build_args()
    sfp = Path(args.saving_path, 'federated')

    rows = []
    for cam in opts.cameras:
        ann_px = load_annotation_lanes(cam)
        if not ann_px:
            print(f'{cam}: no annotation, skipped'); continue
        processed = build_processed(args, cam, sfp)
        feats = SceneFeatureExtractor.extract_features(processed)
        ann_gps = annotation_gps(cam, ann_px, args.dataset_path)
        lat0 = float(np.mean([a[:, 0].mean() for a in ann_gps]))
        lon0 = float(np.mean([a[:, 1].mean() for a in ann_gps]))
        m_lon = M_LAT * np.cos(np.deg2rad(lat0))
        ann_m = [_to_m(a, lat0, lon0, m_lon) for a in ann_gps]
        geo = GeometricLearning(args, sfp)
        split = 'unseen' if cam in UNSEEN else 'seen'
        for model in ['meta', 'federated']:
            for frac in opts.fracs:
                for seed in opts.seeds:
                    snap = Path(f'results/_scarcity/{model}_f{frac}_s{seed}')
                    if not snap.exists():
                        continue
                    try:
                        if split == 'unseen':
                            theta = unseen_theta(model, snap, feats)
                        else:
                            theta = deployed_theta(model, cam, snap / 'federated_model.pth', str(snap))
                        if theta is None:
                            continue
                        _, _, bounds = evaluate(geo, processed, cam, theta)
                        det_m = bounds_to_centerlines_m(bounds, (lat0, lon0))
                    except Exception as e:
                        print(f'  {cam} {model} f{frac} s{seed}: FAILED {e}')
                        continue
                    for tau in opts.taus:
                        p, r, f, g, tp, nd, na = prf(det_m, ann_m, tau)
                        rows.append(dict(split=split, camera=cam, model=model, frac=frac, seed=seed,
                                         tau=tau, precision=p, recall=r, f1=f, tp=tp, n_det=nd, n_ann=na))
        print(f'{cam} ({split}) done')

    out = Path('results/scarcity_analysis'); out.mkdir(parents=True, exist_ok=True)
    with open(out / 'scarcity_analysis.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f'\nwrote {out}/scarcity_analysis.csv ({len(rows)} rows)')

    # aggregate: F1 and recall vs frac, meta vs fed, per split (mean +/- std over seeds, micro over cams)
    for tau in opts.taus:
        for split in ['unseen', 'seen']:
            print(f'\n=== {split} sites, TAU={tau:.0f}m: micro-F1 (and recall) by fraction, mean+/-std over seeds ===')
            print(f"  {'frac':>5}  {'meta F1':>14}  {'fed F1':>14}   {'meta R':>7} {'fed R':>7}")
            for frac in sorted(opts.fracs, reverse=True):
                line = f'  {frac:>5.2f}  '
                stats = {}
                for model in ['meta', 'federated']:
                    perseed = []
                    rec = []
                    for seed in opts.seeds:
                        rs = [x for x in rows if x['split'] == split and x['model'] == model
                              and x['frac'] == frac and x['seed'] == seed and x['tau'] == tau]
                        if not rs:
                            continue
                        tp = sum(x['tp'] for x in rs); nd = sum(x['n_det'] for x in rs); na = sum(x['n_ann'] for x in rs)
                        P = tp / nd if nd else 0; R = tp / na if na else 0
                        perseed.append(2 * P * R / (P + R) if (P + R) else 0); rec.append(R)
                    stats[model] = (np.mean(perseed) if perseed else float('nan'),
                                    np.std(perseed) if perseed else float('nan'),
                                    np.mean(rec) if rec else float('nan'))
                m, f = stats['meta'], stats['federated']
                print(f"  {frac:>5.2f}  {m[0]:.3f}+/-{m[1]:.3f}  {f[0]:.3f}+/-{f[1]:.3f}   {m[2]:>7.3f} {f[2]:>7.3f}"
                      + ('   <-- fed>meta' if f[0] > m[0] else ''))


if __name__ == '__main__':
    main()
