"""Final-batch annotation-F1 aggregation: every snapshot in results/_final_batch
scored against human annotations, mean +/- std over seeds.

Seen sites deploy exactly what the pipeline deploys (buffer-best via the unified
trial_score; baseline = fixed theta). Unseen sites use the zero-shot init
prediction (fed = global model, meta = nearest-scene donor from the same
snapshot) — the deployed calibration thetas are not persisted in checkpoints, so
this is an init-quality comparison, identical budget by construction (zero).
"""
import argparse
import csv, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))

from adaptation_curve import (build_args, build_processed, evaluate, load_annotation_lanes,
                              annotation_gps, _to_m, M_LAT, load_model, predict_theta)
from strategy_annotation_eval import (prf, geometry_decomp, bounds_to_centerlines_m,
                                       deployed_theta, BASELINE_THETA, SEEN)
from LaneDetection.lane_detection.utils import SceneFeatureExtractor
from LaneDetection.lane_detection.geo_learning import GeometricLearning

UNSEEN = ['US12_Park', 'US12_Greenway', 'I43_Keefe', 'I43_Walnut']
SEEDS = [42, 43, 44]
CONFIGS = ['baseline', 'meta', 'fed_perfedavg', 'fed_fedavg']
SNAP = ROOT / 'results/_final_batch'
TAUS = [3.0, 5.0]


def donor_theta(snap, feats):
    best_d, best_p = float('inf'), None
    for p in snap.glob('meta_model_*.pth'):
        cam = p.stem.replace('meta_model_', '')
        if cam not in SEEN:
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


def config_theta(config, seed, cam, split, feats):
    if config == 'baseline':
        return dict(BASELINE_THETA)
    snap = SNAP / f'{config}_s{seed}'
    if config == 'meta':
        if split == 'seen':
            return deployed_theta('meta', cam, '', str(snap))
        return donor_theta(snap, feats)
    ckpt = snap / 'federated_model.pth'
    if split == 'seen':
        return deployed_theta('federated', cam, str(ckpt), '')
    return predict_theta(load_model(ckpt), feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS,
                    help='seeds to score (baseline is always seed 42, deterministic)')
    ap.add_argument('--cameras', nargs='+', default=SEEN + UNSEEN,
                    help='subset of cameras for a quick check')
    ap.add_argument('--configs', nargs='+', default=CONFIGS, help='subset of methods')
    ap.add_argument('--out', default='results/final_batch_f1/final_batch_f1.csv',
                    help='output CSV; point a quick check elsewhere so the full table is not clobbered')
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
        ann_g = annotation_gps(cam, ann_px, args.dataset_path)
        lat0 = float(np.mean([a[:, 0].mean() for a in ann_g]))
        lon0 = float(np.mean([a[:, 1].mean() for a in ann_g]))
        m_lon = M_LAT * np.cos(np.deg2rad(lat0))
        ann_m = [_to_m(a, lat0, lon0, m_lon) for a in ann_g]
        geo = GeometricLearning(args, sfp)
        split = 'seen' if cam in SEEN else 'unseen'
        for config in opts.configs:
            seeds = [42] if config == 'baseline' else opts.seeds  # baseline deterministic
            for seed in seeds:
                try:
                    theta = config_theta(config, seed, cam, split, feats)
                    if theta is None:
                        continue
                    _, metrics, bounds = evaluate(geo, processed, cam, theta)
                    det_m = bounds_to_centerlines_m(bounds, (lat0, lon0))
                except Exception as e:
                    print(f'  {cam} {config} s{seed}: FAILED {e}'); continue
                # width is referenced to the SUMO net width tag (annotations carry no
                # boundaries); model-independent (width_scale divided out in compute_loss)
                width_sumo = float(metrics.get('geo_width_m', float('nan')))
                for tau in TAUS:
                    p, r, f, g, tp, nd, na = prf(det_m, ann_m, tau)
                    gd = geometry_decomp(det_m, ann_m, tau)
                    rows.append(dict(split=split, camera=cam, config=config, seed=seed, tau=tau,
                                     precision=p, recall=r, f1=f, geo_err_tp=g, tp=tp, n_det=nd, n_ann=na,
                                     geo_consistency_m=gd['geo_consistency_m'],
                                     geo_centerline_m=gd['geo_centerline_m'],
                                     geo_coverage_m=gd['geo_coverage_m'],
                                     geo_width_sumo_m=width_sumo,
                                     lane_count_err=gd['lane_count_err']))
        print(f'{cam} ({split}) done')

    if not rows:
        print('no rows produced (check --cameras/--configs/--seeds)'); return
    out = Path(opts.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f'wrote {out} ({len(rows)} rows)')

    for tau in TAUS:
        for split in ['seen', 'unseen']:
            print(f'\n=== {split.upper()}  TAU={tau:.0f}m  (micro over cams; mean+/-std over seeds) ===')
            print(f"  {'config':<15}{'P':>8}{'R':>8}{'F1':>16}{'geoErr':>8}")
            for config in opts.configs:
                per = []
                pr = rr = ge = []
                pvals, rvals, fvals, gvals = [], [], [], []
                seeds = [42] if config == 'baseline' else opts.seeds
                for seed in seeds:
                    rs = [x for x in rows if x['split'] == split and x['config'] == config
                          and x['seed'] == seed and x['tau'] == tau]
                    if not rs:
                        continue
                    tp = sum(x['tp'] for x in rs); nd = sum(x['n_det'] for x in rs); na = sum(x['n_ann'] for x in rs)
                    P = tp / nd if nd else 0; R = tp / na if na else 0
                    F = 2 * P * R / (P + R) if (P + R) else 0
                    pvals.append(P); rvals.append(R); fvals.append(F)
                    gvals.append(np.nanmean([x['geo_err_tp'] for x in rs]))
                if not fvals:
                    continue
                print(f"  {config:<15}{np.mean(pvals):8.3f}{np.mean(rvals):8.3f}"
                      f"{np.mean(fvals):8.3f}+/-{np.std(fvals):5.3f}{np.nanmean(gvals):8.2f}")

    # geometry decomposition against annotations (meters), the geometry-forward table
    for tau in TAUS:
        for split in ['seen', 'unseen']:
            print(f"\n=== {split.upper()}  TAU={tau:.0f}m  GEOMETRY vs annotations (mean over seeds, m) ===")
            print(f"  {'config':<15}{'consist':>9}{'center':>9}{'coverage':>9}{'width*':>9}{'countErr':>9}")
            for config in opts.configs:
                seeds = [42] if config == 'baseline' else opts.seeds
                cons, cent, cov, wid, cnt = [], [], [], [], []
                for seed in seeds:
                    rs = [x for x in rows if x['split'] == split and x['config'] == config
                          and x['seed'] == seed and x['tau'] == tau]
                    if not rs:
                        continue
                    cons.append(np.nanmean([x['geo_consistency_m'] for x in rs]))
                    cent.append(np.nanmean([x['geo_centerline_m'] for x in rs]))
                    cov.append(np.nanmean([x['geo_coverage_m'] for x in rs]))
                    wid.append(np.nanmean([x['geo_width_sumo_m'] for x in rs]))
                    cnt.append(np.mean([x['lane_count_err'] for x in rs]))
                if not cons:
                    continue
                print(f"  {config:<15}{np.nanmean(cons):9.2f}{np.nanmean(cent):9.2f}"
                      f"{np.nanmean(cov):9.2f}{np.nanmean(wid):9.2f}{np.nanmean(cnt):9.2f}")
            print("  * width referenced to SUMO net width (annotations have no boundaries)")


if __name__ == '__main__':
    main()
