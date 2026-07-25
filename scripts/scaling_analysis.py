"""Fleet-scaling analysis: unseen annotation-F1 vs number of training cameras.

Scores every scaling-sweep snapshot (results/_scaling/<model>_n<k>_d<draw>) on
the 4 unseen cameras using zero-shot init predictions (fed = global model,
meta = nearest donor among THAT snapshot's trained cameras, from cameras.txt).
Adds the n=6 point from results/_final_batch (3 seeds). Curves: mean +/- std
over draws (n=6: over seeds).

Usage: uv run python scripts/scaling_analysis.py [--taus 3 5]
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))

from adaptation_curve import (build_args, build_processed, evaluate, load_annotation_lanes,
                              annotation_gps, _to_m, M_LAT, load_model, predict_theta)
from strategy_annotation_eval import prf, bounds_to_centerlines_m
from LaneDetection.lane_detection.utils import SceneFeatureExtractor
from LaneDetection.lane_detection.geo_learning import GeometricLearning

UNSEEN = ['US12_Park', 'US12_Greenway', 'I43_Keefe', 'I43_Walnut']


def donor_theta(snap, feats, allowed):
    best_d, best_p = float('inf'), None
    for p in snap.glob('meta_model_*.pth'):
        cam = p.stem.replace('meta_model_', '')
        if cam not in allowed:
            continue
        buf = torch.load(p, map_location='cpu', weights_only=False).get('client_buffer', [])
        fin = [b for b in buf if b.get('scene_features') is not None]
        if not fin:
            continue
        df = torch.stack([b['scene_features'].squeeze() for b in fin]).mean(0)
        d = float(torch.norm(df - feats.squeeze().cpu()))
        if d < best_d:
            best_d, best_p = d, p
    return predict_theta(load_model(best_p), feats) if best_p else None


def collect_snapshots():
    """[(model, n_train, label, snapdir, allowed_cams)] from _scaling + _final_batch."""
    out = []
    scaling = ROOT / 'results/_scaling'
    for d in sorted(scaling.glob('*_n*_d*')) if scaling.exists() else []:
        if not (d / '.done').exists():
            continue
        model = 'meta' if d.name.startswith('meta') else 'federated'
        n = int(d.name.split('_n')[1].split('_')[0])
        cams = open(d / 'cameras.txt').read().strip().split(',') if (d / 'cameras.txt').exists() else None
        out.append((model, n, d.name, d, cams))
    final = ROOT / 'results/_final_batch'
    full = ['US12_Todd', 'US12_Monona', 'US12_Yahara', 'US12_CountyAB', 'US12_Mineral', 'US12_University']
    for tag, model in [('meta', 'meta'), ('fed_perfedavg', 'federated')]:
        for seed in [42, 43, 44]:
            d = final / f'{tag}_s{seed}'
            if (d / '.done').exists():
                out.append((model, 6, f'{tag}_n6_s{seed}', d, full))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--taus', type=float, nargs='+', default=[3.0, 5.0])
    opts = ap.parse_args()
    args = build_args()
    sfp = Path(args.saving_path, 'federated')
    snaps = collect_snapshots()
    print(f'{len(snaps)} snapshots found')

    rows = []
    for cam in UNSEEN:
        ann_px = load_annotation_lanes(cam)
        if not ann_px:
            print(f'{cam}: no annotation, skipped'); continue
        processed = build_processed(args, cam, sfp)
        feats = SceneFeatureExtractor.extract_features(processed)
        ann_g = annotation_gps(cam, ann_px, args.dataset_path)
        lat0 = float(np.mean([a[:, 0].mean() for a in ann_g]))
        lon0 = float(np.mean([a[:, 1].mean() for a in ann_g]))
        m_lon = M_LAT * np.cos(np.deg2rad(lat0))
        ann_m = [_to_m(a, lat0, lon0, m_lon) for a in ann_g]
        geo = GeometricLearning(args, sfp)
        for model, n, label, snap, allowed in snaps:
            try:
                if model == 'federated':
                    theta = predict_theta(load_model(snap / 'federated_model.pth'), feats)
                else:
                    theta = donor_theta(snap, feats, allowed or [])
                if theta is None:
                    continue
                _, _, bounds = evaluate(geo, processed, cam, theta)
                det_m = bounds_to_centerlines_m(bounds, (lat0, lon0))
            except Exception as e:
                print(f'  {cam} {label}: FAILED {e}'); continue
            for tau in opts.taus:
                p, r, f, g, tp, nd, na = prf(det_m, ann_m, tau)
                rows.append(dict(camera=cam, model=model, n_train=n, label=label, tau=tau,
                                 precision=p, recall=r, f1=f, tp=tp, n_det=nd, n_ann=na))
        print(f'{cam} done')

    out = Path('results/scaling_analysis'); out.mkdir(parents=True, exist_ok=True)
    with open(out / 'scaling_analysis.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f'wrote {out}/scaling_analysis.csv ({len(rows)} rows)')

    ns = sorted(set(r['n_train'] for r in rows))
    for tau in opts.taus:
        print(f'\n=== UNSEEN micro-F1 vs #training cameras (TAU={tau:.0f}m; mean+/-std over draws) ===')
        print(f"  {'model':<11}" + ''.join(f'   n={n}'.ljust(17) for n in ns))
        for model in ['meta', 'federated']:
            line = f'  {model:<11}'
            for n in ns:
                labels = sorted(set(r['label'] for r in rows if r['model'] == model and r['n_train'] == n))
                fvals = []
                for lb in labels:
                    rs = [r for r in rows if r['label'] == lb and r['tau'] == tau]
                    tp = sum(r['tp'] for r in rs); nd = sum(r['n_det'] for r in rs); na = sum(r['n_ann'] for r in rs)
                    P = tp / nd if nd else 0; R = tp / na if na else 0
                    fvals.append(2 * P * R / (P + R) if (P + R) else 0)
                line += (f'{np.mean(fvals):.3f}+/-{np.std(fvals):.3f}  ' if fvals else '      n/a        ')
            print(line)


if __name__ == '__main__':
    main()
