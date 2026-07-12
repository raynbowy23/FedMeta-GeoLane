"""Qualitative side-by-side lane detection comparison, one figure per camera.

Panels: [annotations + trajectory context] [fixed baseline] [Meta] [FedMeta]
[Qiu 2024, authors' code on our detections] [Ren-core 2014, our impl.].
Every panel overlays the human annotations (black dashed) and shows that
method's detected centerlines, in local meters on identical axes, titled with
its P/R/F1 at the 5 m threshold. Thetas come from the final-batch seed-42
snapshots via the same deployment rules as the quantitative tables.

Usage:
  uv run python scripts/qualitative_compare.py --qiu_path <clone> [--cameras ...]
Outputs results/qualitative/<camera>.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from adaptation_curve import (build_args, build_processed, evaluate, load_annotation_lanes,
                              annotation_gps, _to_m, M_LAT)
from strategy_annotation_eval import prf, bounds_to_centerlines_m
from final_f1_aggregate import config_theta, SEEN, UNSEEN
from LaneDetection.lane_detection.utils import SceneFeatureExtractor
from LaneDetection.lane_detection.geo_learning import GeometricLearning

METHOD_COLORS = {'baseline': '0.35', 'meta': 'tab:blue', 'fed_perfedavg': 'tab:orange',
                 'qiu': 'tab:purple', 'ren': 'tab:red'}
METHOD_LABELS = {'baseline': 'Fixed baseline', 'meta': 'Meta (per-camera)',
                 'fed_perfedavg': 'FedMeta', 'qiu': 'Qiu et al. [28]', 'ren': 'Ren core [29]'}


def ren_lanes(cam, lat0, lon0, m_lon):
    import polars as pl
    from ren_core_eval import (resample, ren_core, MIN_TRACK_PTS, MIN_TRACK_LEN_PX, TAU_WIDTHS)
    from LaneDetection.osm_extraction.utils import trajectory_calibration, interpolate_edge
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
            trajs.append(R); widths.append(float(np.median(g['w'].to_numpy())))
    lanes_px = ren_core(trajs, TAU_WIDTHS * float(np.median(widths)))
    out = []
    for P in lanes_px:
        dense = np.asarray(interpolate_edge(P, num_points=60), dtype=float)
        gps, _ = trajectory_calibration(dense, Path('./dataset/511calibration'), cam)
        out.append(_to_m(np.asarray(gps, float), lat0, lon0, m_lon))
    return out


def qiu_lanes(cam, qiu_path, lat0, lon0, m_lon):
    from qiu_baseline_eval import run_qiu
    from LaneDetection.osm_extraction.utils import trajectory_calibration, interpolate_edge
    polylines = run_qiu(cam, Path(qiu_path), Path('results/qualitative/_qiu_work'), 3, False)
    out = []
    for P in polylines:
        dense = np.asarray(interpolate_edge(P, num_points=60), dtype=float)
        gps, _ = trajectory_calibration(dense, Path('./dataset/511calibration'), cam)
        out.append(_to_m(np.asarray(gps, float), lat0, lon0, m_lon))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qiu_path', default=None, help='Qiu clone dir; panel skipped if absent')
    ap.add_argument('--cameras', nargs='+', default=SEEN + UNSEEN)
    opts = ap.parse_args()
    args = build_args()
    sfp = Path(args.saving_path, 'federated')
    out_dir = Path('results/qualitative'); out_dir.mkdir(parents=True, exist_ok=True)

    for cam in opts.cameras:
        ann_px = load_annotation_lanes(cam)
        if not ann_px:
            print(f'{cam}: no annotation, skipped'); continue
        split = 'seen' if cam in SEEN else 'unseen'
        processed = build_processed(args, cam, sfp)
        feats = SceneFeatureExtractor.extract_features(processed)
        ann_g = annotation_gps(cam, ann_px, args.dataset_path)
        lat0 = float(np.mean([a[:, 0].mean() for a in ann_g]))
        lon0 = float(np.mean([a[:, 1].mean() for a in ann_g]))
        m_lon = M_LAT * np.cos(np.deg2rad(lat0))
        ann_m = [_to_m(a, lat0, lon0, m_lon) for a in ann_g]
        geo = GeometricLearning(args, sfp)

        results = {}
        for cfg in ['baseline', 'meta', 'fed_perfedavg']:
            try:
                theta = config_theta(cfg, 42, cam, split, feats)
                _, _, bounds = evaluate(geo, processed, cam, theta)
                results[cfg] = bounds_to_centerlines_m(bounds, (lat0, lon0))
            except Exception as e:
                print(f'  {cam} {cfg}: FAILED {e}'); results[cfg] = []
        try:
            results['ren'] = ren_lanes(cam, lat0, lon0, m_lon)
        except Exception as e:
            print(f'  {cam} ren: FAILED {e}'); results['ren'] = []
        if opts.qiu_path:
            try:
                results['qiu'] = qiu_lanes(cam, opts.qiu_path, lat0, lon0, m_lon)
            except Exception as e:
                print(f'  {cam} qiu: FAILED {e}'); results['qiu'] = []

        # trajectory context (subsampled points, local meters)
        gdf = processed['gps_df']
        pts = np.stack([gdf['x_gps'].to_numpy(), gdf['y_gps'].to_numpy()], axis=1)
        pts = pts[np.isfinite(pts).all(axis=1)][::25]
        pts_m = _to_m(pts, lat0, lon0, m_lon)

        panels = ['context', 'baseline', 'meta', 'fed_perfedavg'] + \
                 (['qiu'] if 'qiu' in results else []) + ['ren']
        # roads are long flat strips: stacked rows with shared axes use the
        # canvas far better than a grid
        fig, axes = plt.subplots(len(panels), 1, figsize=(14, 2.1 * len(panels)),
                                 sharex=True, sharey=True)
        axes = np.atleast_1d(axes)
        for ax, key in zip(axes, panels):
            for A in ann_m:
                ax.plot(A[:, 0], A[:, 1], 'k--', lw=1.2, alpha=0.85,
                        label='annotation' if A is ann_m[0] else None)
            if key == 'context':
                ax.scatter(pts_m[:, 0], pts_m[:, 1], s=1, c='0.75', zorder=0)
                ax.set_title(f'{cam} ({split}) — annotations + trajectories')
            else:
                dets = results.get(key, [])
                for D in dets:
                    ax.plot(D[:, 0], D[:, 1], color=METHOD_COLORS[key], lw=2.0, alpha=0.9)
                p, r, f, g, tp, nd, na = prf(dets, ann_m, 5.0)
                ax.set_title(f'{METHOD_LABELS[key]}  P={p:.2f} R={r:.2f} F1={f:.2f} ({nd}/{na})')
            ax.set_aspect('equal'); ax.grid(alpha=0.2)
            ax.set_xticklabels([]); ax.set_yticklabels([])
        fig.suptitle(f'Detected lane centerlines vs human annotations — {cam}', y=0.995)
        fig.tight_layout()
        fig.savefig(out_dir / f'{cam}.png', dpi=140)
        plt.close(fig)
        print(f'{cam}: wrote results/qualitative/{cam}.png')


if __name__ == '__main__':
    main()
