"""Manuscript qualitative figure: method rows x site columns, each cell drawing that
method's detected lanes on the camera frame, one colour per lane, bold centerlines,
dashed boundaries (where the method estimates them), faint black GT. Rows are the
fixed baseline, the per-camera meta model, the federated meta model, and the two
external baselines Qiu et al. and Ren core. Uses current checkpoints and the same
deployed thetas as the tables (config_theta, seed 42).

--cameras chooses the columns (default is the candidate set for selection).
--qiu_path <clone> adds the Qiu row (skipped if absent). --out overrides the path;
default writes a preview, not the manuscript figs/, so the installed figure is not
clobbered until the columns are chosen.

Usage: uv run python scripts/qualitative_grid.py --qiu_path ../third_party/qiu_mroi
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

from qualitative_overlay import gps_to_pixel_fn
from adaptation_curve import build_args, build_processed, evaluate, load_annotation_lanes
from strategy_annotation_eval import SEEN
from final_f1_aggregate import config_theta
from LaneDetection.lane_detection.utils import SceneFeatureExtractor
from LaneDetection.lane_detection.geo_learning import GeometricLearning
import polars as pl

# candidate columns (Rei picks four of these)
DEFAULT_SITES = ['US12_Yahara', 'US12_Mineral', 'US12_CountyAB', 'US12_University', 'I43_Keefe']
# Row display order: external baselines first, then ours (baseline labelled as ours).
METHODS = [('qiu', 'Qiu et al.'), ('ren', 'Ren'), ('baseline', 'Baseline (ours)'),
           ('meta', 'Meta'), ('fed_fedavg', 'FedMeta')]
LANE_COLORS = [(0.84, 0.15, 0.16), (0.12, 0.47, 0.71), (0.17, 0.63, 0.17),
               (1.00, 0.50, 0.05), (0.58, 0.40, 0.74), (0.55, 0.34, 0.29),
               (0.89, 0.47, 0.76), (0.50, 0.50, 0.50), (0.74, 0.74, 0.13),
               (0.09, 0.75, 0.81), (0.99, 0.75, 0.35), (0.40, 0.65, 0.85)]


def gt_group_lanes(cam):
    """Ground-truth lanes in pixel coordinates, each tagged with its annotation lane
    group id, so a detected lane can be assigned to the group of its nearest GT lane.
    Colouring is then done left-to-right within each group."""
    import json
    p = ROOT / 'dataset' / 'preprocess' / cam / 'annotation.json'
    if not p.exists():
        return []
    doc = json.load(open(p))
    out = []
    for gi, g in enumerate(doc.get('lane_groups', [])):
        for lane in g.get('lanes', []):
            pts = np.array([[w['x'], w['y']] for w in lane.get('waypoints', [])], float)
            if len(pts) >= 2:
                out.append((pts, gi))
    return out


def _group_clouds(gt):
    """Per-group stacked GT point cloud, for nearest-group tests."""
    clouds = {}
    for P, g in gt:
        clouds.setdefault(g, []).append(P)
    return {g: np.vstack(v) for g, v in clouds.items()}


def group_of(L, gt):
    """Annotation lane-group id where the majority of this detection's points fall
    (per-point nearest group), so the assignment agrees with where the lane's body sits
    and a lane that only grazes another carriageway is not misassigned to it."""
    C = np.asarray(L['center'], float)
    if len(C) < 2 or not gt:
        return -1
    clouds = _group_clouds(gt)
    near = [min(clouds, key=lambda g: np.linalg.norm(clouds[g] - p, axis=1).min()) for p in C]
    vals, counts = np.unique(near, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def clip_to_group(P, clouds, gid, gap=2):
    """Keep the longest run of the polyline that stays in group gid, so a detected lane
    sweeping across the median into the opposing carriageway is drawn only over the
    extent that belongs to its own direction. Isolated out-of-group points (up to `gap`
    long) are bridged first, so that where perspective makes carriageways converge near
    the vanishing point a single ambiguous point does not chop a valid lane short; a
    genuine multi-point excursion still breaks the run and is trimmed."""
    P = np.asarray(P, float)
    if len(P) < 2 or gid not in clouds or len(clouds) < 2:
        return P
    keep = np.array([min(clouds, key=lambda g: np.linalg.norm(clouds[g] - p, axis=1).min()) == gid
                     for p in P])
    # bridge short out-of-group gaps that are flanked by in-group points on both sides
    i = 0
    while i < len(keep):
        if not keep[i]:
            j = i
            while j < len(keep) and not keep[j]:
                j += 1
            if 0 < i and j < len(keep) and (j - i) <= gap:
                keep[i:j] = True
            i = j
        else:
            i += 1
    best = (0, 0); i = 0
    while i < len(keep):
        if keep[i]:
            j = i
            while j < len(keep) and keep[j]:
                j += 1
            if j - i > best[1] - best[0]:
                best = (i, j)
            i = j
        else:
            i += 1
    return P[best[0]:best[1]] if best[1] - best[0] >= 2 else P


def ours(cam, cfg, geo, proc, feats, to_px, split):
    _, _, bounds = evaluate(geo, proc, cam, config_theta(cfg, 42, cam, split, feats))
    return [dict(center=to_px(np.asarray(d['center'], float)),
                 left=to_px(np.asarray(d['left'], float)) if d.get('left') is not None else None,
                 right=to_px(np.asarray(d['right'], float)) if d.get('right') is not None else None)
            for by in bounds.values() for d in by.values()]


def ren_lanes(cam):
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
    return [dict(center=np.asarray(C, float), left=None, right=None)
            for C in ren_core(trajs, TAU_WIDTHS * float(np.median(widths)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cameras', nargs='+', default=DEFAULT_SITES)
    ap.add_argument('--qiu_path', default=None)
    ap.add_argument('--out', default=str(ROOT / 'results' / 'qualitative' / 'qualitative_grid_preview.png'))
    opts = ap.parse_args()
    args = build_args(); sfp = Path(args.saving_path, 'federated')
    sites = opts.cameras
    data = {}
    for cam in sites:
        ann_px = load_annotation_lanes(cam)
        frame = cv2.cvtColor(np.load(ROOT / 'results' / 'preprocess' / cam / 'last_frame.npy'), cv2.COLOR_BGR2RGB)
        HW = frame.shape[:2]
        proc = build_processed(args, cam, sfp)
        feats = SceneFeatureExtractor.extract_features(proc)
        to_px = gps_to_pixel_fn(cam, ann_px, args.dataset_path)
        split = 'seen' if cam in SEEN else 'unseen'
        lanes = {}
        for cfg in ['baseline', 'meta', 'fed_fedavg']:
            try:
                lanes[cfg] = ours(cam, cfg, geo := GeometricLearning(args, sfp), proc, feats, to_px, split)
            except Exception as e:
                print(f'  {cam} {cfg}: FAILED {e}'); lanes[cfg] = []
        try:
            lanes['ren'] = ren_lanes(cam)
        except Exception as e:
            print(f'  {cam} ren: FAILED {e}'); lanes['ren'] = []
        if opts.qiu_path:
            try:
                from qiu_baseline_eval import run_qiu
                lanes['qiu'] = run_qiu(cam, Path(opts.qiu_path), ROOT / 'results/qualitative/_qiu_work', 3, False,
                                       return_boundaries=True)
            except Exception as e:
                print(f'  {cam} qiu: FAILED {e}'); lanes['qiu'] = []
        else:
            lanes['qiu'] = []
        data[cam] = dict(frame=frame, ann=ann_px, lanes=lanes, HW=HW, split=split, gt=gt_group_lanes(cam))
        print(f'{cam} computed')

    rows = [m for m in METHODS if m[0] != 'qiu' or opts.qiu_path]
    fig, axes = plt.subplots(len(rows), len(sites), figsize=(4.6 * len(sites), 2.75 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (cfg, mlabel) in enumerate(rows):
        for c, cam in enumerate(sites):
            ax = axes[r][c]; d = data[cam]; H, W = d['HW']
            ax.imshow(d['frame'])
            for A in d['ann']:
                ax.plot(A[:, 0], A[:, 1], '-', color='black', lw=0.8, alpha=0.42)

            def clip(P):
                P = np.asarray(P, float)
                k = (P[:, 0] > -40) & (P[:, 0] < W + 40) & (P[:, 1] > -40) & (P[:, 1] < H + 40)
                return P[k]
            lane_list = d['lanes'].get(cfg, [])
            clouds = _group_clouds(d['gt'])
            # assign each lane to its GT lane group, then colour left-to-right within group
            def _meanx(L):
                C = np.asarray(L['center'], float)
                return float(C[:, 0].mean()) if len(C) else 1e9
            gid_of = {i: group_of(L, d['gt']) for i, L in enumerate(lane_list)}
            by_group = {}
            for i in range(len(lane_list)):
                by_group.setdefault(gid_of[i], []).append(i)
            rank = {}
            for members in by_group.values():
                for k, i in enumerate(sorted(members, key=lambda i: _meanx(lane_list[i]))):
                    rank[i] = k
            for i, L in enumerate(lane_list):
                col = LANE_COLORS[rank[i] % len(LANE_COLORS)]
                gid = gid_of[i]
                C = clip(clip_to_group(L['center'], clouds, gid))
                if len(C) >= 2:
                    ax.plot(C[:, 0], C[:, 1], color=col, lw=2.5, alpha=0.97)
                for side in ('left', 'right'):
                    if L.get(side) is not None:
                        B = clip(clip_to_group(L[side], clouds, gid))
                        if len(B) >= 2:
                            ax.plot(B[:, 0], B[:, 1], color=col, lw=0.8, ls=(0, (4, 3)), alpha=0.7)
            ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                tag = cam.replace('US12_', '').replace('I43_', '')
                ax.set_title(f"{tag}" + (" (held-out)" if d['split'] == 'unseen' else ""), fontsize=13)
            if c == 0:
                ax.set_ylabel(mlabel, fontsize=12)
    fig.tight_layout()
    fig.savefig(opts.out, dpi=170, bbox_inches='tight'); plt.close(fig)
    print(f'\nwrote {opts.out}')


if __name__ == '__main__':
    main()
