"""Camera-frame overlay comparison: the site's last video frame as background,
lane data drawn on it in pixel space. Panels: annotations, fixed baseline,
Meta, FedMeta, Qiu [28], Ren core [29]. Our detections are back-projected
from GPS through the inverse of the same site homography; annotations, Qiu,
and Ren are pixel-native. Same thetas and deployment rules as the tables.

Usage:
  uv run python scripts/qualitative_overlay.py --qiu_path <clone> [--cameras ...]
Outputs results/qualitative/<camera>_overlay.png
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from adaptation_curve import build_args, build_processed, evaluate, load_annotation_lanes
from strategy_annotation_eval import bounds_to_centerlines_m  # noqa: F401 (parity import)
from final_f1_aggregate import config_theta, SEEN, UNSEEN
from LaneDetection.lane_detection.utils import SceneFeatureExtractor
from LaneDetection.lane_detection.geo_learning import GeometricLearning
from LaneDetection.osm_extraction.utils import trajectory_calibration

METHODS = [('baseline', 'Fixed baseline', (0.17, 0.63, 0.17)),
           ('meta', 'Meta (per-camera)', (0.12, 0.47, 0.71)),
           ('fed_fedavg', 'FedMeta', (1.00, 0.50, 0.05)),
           ('qiu', 'Qiu et al. [28]', (0.58, 0.40, 0.74)),
           ('ren', 'Ren core [29]', (0.84, 0.15, 0.16))]


def gps_to_pixel_fn(cam, ann_px, dataset_path):
    """Inverse of the site homography, with the GPS column order verified by
    round-tripping the annotation points (the pipeline's lat/lon ordering has
    bitten before, so it is checked, not assumed)."""
    pts = np.vstack(ann_px)[:60].astype(float)
    gps, H = trajectory_calibration(pts, Path(dataset_path, '511calibration'), cam)
    gps = np.asarray(gps, float)
    Hinv = np.linalg.inv(np.asarray(H, float))

    def back(g, swap):
        q = g[:, ::-1] if swap else g
        p = cv2.perspectiveTransform(q.reshape(-1, 1, 2).astype(np.float64), Hinv).reshape(-1, 2)
        return p
    errs = [float(np.mean(np.linalg.norm(back(gps, s) - pts, axis=1))) for s in (False, True)]
    swap = bool(np.argmin(errs))
    assert min(errs) < 5.0, f'homography round-trip failed ({errs})'
    return lambda g: back(np.asarray(g, float), swap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qiu_path', default=None)
    ap.add_argument('--cameras', nargs='+', default=SEEN + UNSEEN)
    ap.add_argument('--panels', action='store_true',
                    help='also export each panel as a separate label-free PNG at native frame resolution (results/qualitative/panels/)')
    opts = ap.parse_args()
    args = build_args()
    sfp = Path(args.saving_path, 'federated')
    out_dir = Path('results/qualitative'); out_dir.mkdir(parents=True, exist_ok=True)

    for cam in opts.cameras:
        ann_px = load_annotation_lanes(cam)
        if not ann_px:
            print(f'{cam}: no annotation, skipped'); continue
        split = 'seen' if cam in SEEN else 'unseen'
        frame = np.load(ROOT / 'results' / 'preprocess' / cam / 'last_frame.npy')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        H, W = frame.shape[:2]
        processed = build_processed(args, cam, sfp)
        feats = SceneFeatureExtractor.extract_features(processed)
        geo = GeometricLearning(args, sfp)
        to_px = gps_to_pixel_fn(cam, ann_px, args.dataset_path)

        lanes_px = {}
        for cfg in ['baseline', 'meta', 'fed_fedavg']:
            try:
                theta = config_theta(cfg, 42, cam, split, feats)
                _, _, bounds = evaluate(geo, processed, cam, theta)
                lanes_px[cfg] = [dict(center=to_px(np.asarray(d['center'], float)),
                                      left=to_px(np.asarray(d['left'], float)),
                                      right=to_px(np.asarray(d['right'], float)))
                                 for by in bounds.values() for d in by.values()]
            except Exception as e:
                print(f'  {cam} {cfg}: FAILED {e}'); lanes_px[cfg] = []
        try:
            import polars as pl
            from ren_core_eval import resample, ren_core, MIN_TRACK_PTS, MIN_TRACK_LEN_PX, TAU_WIDTHS
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
            lanes_px['ren'] = [dict(center=np.asarray(C, float)) for C in ren_core(trajs, TAU_WIDTHS * float(np.median(widths)))]
        except Exception as e:
            print(f'  {cam} ren: FAILED {e}'); lanes_px['ren'] = []
        if opts.qiu_path:
            try:
                from qiu_baseline_eval import run_qiu
                lanes_px['qiu'] = run_qiu(cam, Path(opts.qiu_path),
                                          Path('results/qualitative/_qiu_work'), 3, False,
                                          return_boundaries=True)
            except Exception as e:
                print(f'  {cam} qiu: FAILED {e}'); lanes_px['qiu'] = []

        methods = [(k, lbl, c) for k, lbl, c in METHODS if k in lanes_px]
        panels = [('annotations', 'Human annotations', None)] + methods
        fig, axes = plt.subplots(2, 3, figsize=(19, 7.5))
        axes = axes.ravel()
        for ax in axes[len(panels):]:
            ax.axis('off')
        for ax, (key, label, color) in zip(axes, panels):
            ax.imshow(frame)
            for A in ann_px:
                ax.plot(A[:, 0], A[:, 1], '-', color='black', lw=1.6, alpha=0.9)
            if key != 'annotations':
                def clip(P):
                    P = np.asarray(P, float)
                    k = (P[:, 0] > -50) & (P[:, 0] < W + 50) & (P[:, 1] > -50) & (P[:, 1] < H + 50)
                    return P[k]
                nb = 0
                for L in lanes_px.get(key, []):
                    C = clip(L['center'])
                    if len(C) >= 2:
                        ax.plot(C[:, 0], C[:, 1], color=color, lw=2.4, alpha=0.95)
                    # boundaries drawn only when the method estimates them, so
                    # the capability gap is visible in the figure itself
                    for side in ('left', 'right'):
                        if side in L and L[side] is not None and len(L[side]) >= 2:
                            B = clip(L[side])
                            if len(B) >= 2:
                                ax.plot(B[:, 0], B[:, 1], color=color, lw=1.1,
                                        alpha=0.75, linestyle='-')
                                nb += 1
                cap = 'center+boundaries' if nb else 'centerline only'
                ax.set_title(f'{label} ({len(lanes_px.get(key, []))} lanes, {cap})')
            else:
                ax.set_title(f'{cam} ({split}) — human annotations (black solid)')
            ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')
        fig.suptitle(f'Detected lanes over the camera view — {cam}', y=0.99)
        fig.tight_layout()
        fig.savefig(out_dir / f'{cam}_overlay.png', dpi=140)
        plt.close(fig)
        print(f'{cam}: wrote results/qualitative/{cam}_overlay.png')

        if opts.panels:
            # label-free single panels at native resolution for manuscript
            # composition: raw frame, annotations-only, and one per method
            pdir = out_dir / 'panels'; pdir.mkdir(exist_ok=True)
            def save_panel(name, draw):
                pfig = plt.figure(figsize=(W / 100.0, H / 100.0), dpi=100)
                pax = pfig.add_axes([0, 0, 1, 1])
                pax.imshow(frame)
                draw(pax)
                pax.set_xlim(0, W); pax.set_ylim(H, 0); pax.axis('off')
                pfig.savefig(pdir / f'{cam}_{name}.png', dpi=100)
                plt.close(pfig)
            def draw_ann(pax):
                for A in ann_px:
                    pax.plot(A[:, 0], A[:, 1], '-', color='black', lw=2.0, alpha=0.9)
            save_panel('frame', lambda pax: None)
            save_panel('annotations', draw_ann)
            for key, label, color in methods:
                def draw_m(pax, key=key, color=color):
                    draw_ann(pax)
                    for L in lanes_px.get(key, []):
                        C = clip(L['center'])
                        if len(C) >= 2:
                            pax.plot(C[:, 0], C[:, 1], color=color, lw=3.0, alpha=0.95)
                        for side in ('left', 'right'):
                            if side in L and L[side] is not None and len(L[side]) >= 2:
                                B = clip(L[side])
                                if len(B) >= 2:
                                    pax.plot(B[:, 0], B[:, 1], color=color, lw=1.5, alpha=0.75)
                save_panel(key, draw_m)
            print(f'{cam}: wrote {2 + len(methods)} label-free panels to results/qualitative/panels/')


if __name__ == '__main__':
    main()
