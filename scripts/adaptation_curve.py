"""Adaptation-budget curves at unseen sites.

For each unseen camera and each initialization (FedMeta global model, Meta
nearest-scene donor, fixed baseline theta), evaluate the raw geometric error
after k weakly-supervised calibration trials, k in {0, 1, 2, 4, 8}. The
meta-learning claim is that a better initialization adapts faster, so the
FedMeta curve sitting at or below the Meta-donor curve at small k is the
figure this produces evidence for.

Deployment-only: consumes trained checkpoints and saved preprocess data,
trains nothing. Run AFTER a full federated + meta training pass.

Usage:
    uv run python scripts/adaptation_curve.py [--reps 3] [--budgets 0 1 2 4 8]

Outputs results/adaptation_curve/adaptation_curve.csv with columns
camera,init,budget,rep,score,lane_count_err plus a summary printout.
"""
import argparse
import csv
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import polars as pl
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from LaneDetection.lane_detection.geo_learning import GeometricLearning
from LaneDetection.lane_detection.meta_federated_lane_detection import MetaMLModel
from LaneDetection.lane_detection.utils import (SceneFeatureExtractor, perturb_theta,
                                                trial_score, no_detection_result)
from LaneDetection.osm_extraction.connect_to_osm import OSMConnection
from data_pipeline import OrbitDataPipeline
from geolearning_pipeline import process_camera_data
from geolearning_system import SEEN_CLIENTS, UNSEEN_CLIENTS
from utils import compute_loss_for_baseline

BASELINE_THETA = {
    'angle_penalty': 0.5, 'width_scale': 1.0, 'consistency_weight': 0.5,
    'triplet_margin': 0.8, 'smoothing_factor': 10.0, 'edge_trim_ratio': 0.1,
    'peak_prominence': 0.5, 'weight_lane_count': 1.0, 'weight_consistency': 1.0,
    'weight_triplet': 1.0, 'weight_geometry': 1.0,
}


def build_args():
    return Namespace(
        dataset_path='./dataset/', video_path='./dataset/511video',
        saving_path='./results', osm_path='./dataset/sumo/', model='federated',
        T=60, is_save=False, conf_thre=0.25, cnts_threshold=0, num_grids=50,
        lambda_thres=120, seed=42,
    )


def load_model(path):
    m = MetaMLModel()
    m.device = torch.device('cpu')
    sd = torch.load(path, map_location='cpu', weights_only=False)
    m.load_state_dict(sd['model_state_dict'] if isinstance(sd, dict) and 'model_state_dict' in sd else sd)
    m.eval()
    return m


def predict_theta(model, feats):
    with torch.no_grad():
        out = model(feats)
    return {k: (float(v.squeeze()) if isinstance(v, torch.Tensor) else float(v)) for k, v in out.items()}


def build_processed(args, camera_loc, saving_file_path):
    pre = Path('results/preprocess', camera_loc)
    frame = np.load(pre / 'last_frame.npy')
    cars = [tuple(x) for x in np.load(pre / 'collect_cars.npy').tolist()]
    dots = [tuple(x) for x in np.load(pre / 'collect_det_dots_including_truck.npy').tolist()]
    traj = pl.read_csv(pre / 'trajectory.csv', schema_overrides={'target_lane_id': pl.Utf8})
    data_pipeline = OrbitDataPipeline(args, saving_file_path)
    osm = OSMConnection(args, saving_file_path)
    return process_camera_data(data_pipeline, osm, camera_loc, c_epoch=0,
                               frame=frame, collect_cars=cars, collect_dots=dots, traj_df=traj)


def evaluate(geo, processed, camera_loc, theta):
    for k, v in theta.items():
        geo.theta[k] = torch.tensor(float(v))
    traj_df, bounds = geo.run(c_epoch=0, g_epoch=0, traj_df=processed['gps_df'],
                              camera_loc=camera_loc, trial='0', is_save=False)
    detected = sum(len(b) for b in bounds.values())
    if detected == 0:
        loss, metrics = no_detection_result(processed)
    else:
        loss, metrics = compute_loss_for_baseline(geo, traj_df, bounds, processed, camera_loc)
    return trial_score(loss, metrics), metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--budgets', type=int, nargs='+', default=[0, 1, 2, 4, 8])
    ap.add_argument('--cameras', nargs='+', default=None)
    opts = ap.parse_args()

    args = build_args()
    cameras = opts.cameras or UNSEEN_CLIENTS
    out_dir = Path('results/adaptation_curve')
    out_dir.mkdir(parents=True, exist_ok=True)
    saving_file_path = Path(args.saving_path, 'federated')

    fed = load_model('results/federated/training_results/federated_model.pth')
    donors = {}
    for cam in SEEN_CLIENTS:
        p = Path(f'results/meta/training_results/meta_model_{cam}.pth')
        if not p.exists():
            continue
        ck = torch.load(p, map_location='cpu', weights_only=False)
        buf = ck.get('client_buffer', [])
        if buf:
            feats = torch.stack([b['scene_features'].squeeze() for b in buf]).mean(dim=0)
            donors[cam] = (load_model(p), feats)
    print(f'{len(donors)} donor models loaded: {sorted(donors)}')

    rows = []
    for cam in cameras:
        t0 = time.time()
        processed = build_processed(args, cam, saving_file_path)
        feats = SceneFeatureExtractor.extract_features(processed)
        geo = GeometricLearning(args, saving_file_path)

        donor_id = min(donors, key=lambda d: float(torch.norm(donors[d][1] - feats.squeeze().cpu())))
        inits = {
            'fedmeta': predict_theta(fed, feats),
            'meta_donor': predict_theta(donors[donor_id][0], feats),
            'baseline': dict(BASELINE_THETA),
        }
        print(f'{cam}: donor={donor_id}, processed in {time.time()-t0:.0f}s')

        for init_name, theta0 in inits.items():
            s0, m0 = evaluate(geo, processed, cam, theta0)
            for budget in opts.budgets:
                reps = 1 if budget == 0 else opts.reps
                for rep in range(reps):
                    rng = np.random.default_rng(hash((cam, init_name, budget, rep)) % 2**32)
                    np.random.seed(rng.integers(2**31))
                    torch.manual_seed(rng.integers(2**31))
                    best_s, best_m = s0, m0
                    for _ in range(budget):
                        cand = perturb_theta(theta0)
                        s, m = evaluate(geo, processed, cam, cand)
                        if s < best_s:
                            best_s, best_m = s, m
                    rows.append({'camera': cam, 'init': init_name, 'budget': budget, 'rep': rep,
                                 'score': best_s, 'lane_count_err': best_m.get('lane_count_err', float('nan'))})
                    print(f'  {init_name:<11} k={budget} rep={rep}: score={best_s:.2f}')

    out_csv = out_dir / 'adaptation_curve.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['camera', 'init', 'budget', 'rep', 'score', 'lane_count_err'])
        w.writeheader()
        w.writerows(rows)
    print(f'\nwrote {out_csv} ({len(rows)} rows)')

    print('\nmean score by init and budget (over cameras and reps):')
    for init_name in ['baseline', 'meta_donor', 'fedmeta']:
        line = f'  {init_name:<11}'
        for budget in opts.budgets:
            vals = [r['score'] for r in rows if r['init'] == init_name and r['budget'] == budget and np.isfinite(r['score'])]
            line += f'  k={budget}:{np.mean(vals):7.2f}' if vals else f'  k={budget}:    n/a'
        print(line)


if __name__ == '__main__':
    main()
