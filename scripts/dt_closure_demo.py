"""R1.4 closed-loop dynamic-update demo: detection -> change flag -> TraCI
lane closure -> re-simulation, one scenario (default US12_Monona).

The closure is SYNTHESIZED in the input data and disclosed as such: window A
detects on the full trajectories, window B on the trajectories with one
detected lane's vehicles removed (a closed lane carries no traffic). What the
demo exercises for real is the LOOP the reviewer asked about: the change
detector flags the disappeared lane, the flag maps to a SUMO lane id through
the same net the pipeline trains against, the closure is pushed into the
RUNNING network via TraCI (no netconvert, no re-import), and the scene is
re-simulated with the osmWebWizard demand shipped with the net (fixed seed).

Outputs results/dt_demo/: candidates table, change log, before/after metrics
(JSON + CSV), overlay + metrics figure (provisional styling; the manuscript
figure is restyled later).

Usage:
  uv run python scripts/dt_closure_demo.py [--camera US12_Monona]
      [--close_lane N] [--sim_end 600] [--seed 42]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))

import sumolib
import traci

from adaptation_curve import build_args, BASELINE_THETA, build_processed, _to_m, M_LAT
from strategy_annotation_eval import det_to_ann_dist
from LaneDetection.lane_detection.geo_learning import GeometricLearning
import polars as pl


def detect(geo, processed, cam, gps_df):
    for k, v in BASELINE_THETA.items():
        geo.theta[k] = torch.tensor(float(v))
    traj_df, bounds = geo.run(0, 0, gps_df, cam, '0', False)
    tp = traj_df.to_pandas() if hasattr(traj_df, 'to_pandas') else traj_df
    lanes = []
    for cnt_id, by_id in bounds.items():
        for lane_id, d in by_id.items():
            sel = tp[(tp['contour_id'] == cnt_id) & (tp['clustered_id'] == lane_id)]
            # mean member travel vector in GPS (first->last per id), for the
            # direction constraint in net mapping (never map to the opposing
            # carriageway of the schematic OSM offset)
            vecs = []
            for vid, gg in sel.groupby('id'):
                gg = gg.sort_values('time') if 'time' in gg else gg
                vecs.append([gg['x_gps'].iloc[-1] - gg['x_gps'].iloc[0],
                             gg['y_gps'].iloc[-1] - gg['y_gps'].iloc[0]])
            heading = np.mean(np.asarray(vecs), axis=0) if vecs else np.zeros(2)
            lanes.append(dict(center=np.asarray(d['center'], float),
                              ids=set(sel['id'].unique().tolist()),
                              heading_gps=heading,
                              key=(cnt_id, lane_id)))
    return lanes


def map_lanes_to_net(net, lanes):
    """Detected-lane -> net-lane mapping robust to the schematic OSM offset.

    Absolute distances cannot resolve within-edge lane indices under the
    warped 15-20 m reference offset (R1.3 decomposition), but two structures
    survive any smooth warp: travel DIRECTION (separates carriageways) and
    LATERAL ORDER (separates lanes within an edge). So: (1) assign each
    detected lane to its direction-consistent nearest EDGE, (2) within each
    edge group, rank-match detected lanes to SUMO lane indices by lateral
    position (SUMO index 0 = rightmost). 'exact' marks groups whose detected
    count equals the edge's lane count, where the index mapping is
    unambiguous; auto-pick only closes lanes from exact groups."""
    det_pts = [np.asarray([net.convertLonLat2XY(lon, lat) for lat, lon in L['center'][::3]], float)
               for L in lanes]
    scene_c = np.mean(np.vstack(det_pts), axis=0)
    headings = []
    for L in lanes:
        h = L['heading_gps']
        lat_m = float(np.mean(L['center'][:, 0])); lon_m = float(np.mean(L['center'][:, 1]))
        p0 = np.asarray(net.convertLonLat2XY(lon_m, lat_m))
        p1 = np.asarray(net.convertLonLat2XY(lon_m + h[1], lat_m + h[0]))
        v = p1 - p0
        headings.append(v / (np.linalg.norm(v) + 1e-12))
    edges = [e for e in net.getEdges() if e.getFunction() != 'internal'
             and np.linalg.norm(np.asarray(e.getShape(), float).mean(axis=0) - scene_c) < 300]
    # (1) direction-consistent nearest edge per detected lane
    groups = {}
    for i, D in enumerate(det_pts):
        best = (None, float('inf'))
        for e in edges:
            shape = np.asarray(e.getShape(), float)
            ev = shape[-1] - shape[0]; ev = ev / (np.linalg.norm(ev) + 1e-12)
            if float(np.dot(headings[i], ev)) < 0:
                continue
            d = float(np.mean([np.min(np.linalg.norm(shape - p, axis=1)) for p in D]))
            if d < best[1]:
                best = (e, d)
        lanes[i]['edge_d'] = best[1]
        groups.setdefault(best[0].getID() if best[0] else None, []).append(i)
    # (2) rank match by lateral position within each edge group
    for L in lanes:
        L['sumo'], L['exact'] = None, False
    for eid, idxs in groups.items():
        if eid is None:
            continue
        e = net.getEdge(eid)
        shape = np.asarray(e.getShape(), float)
        ev = shape[-1] - shape[0]; ev = ev / (np.linalg.norm(ev) + 1e-12)
        perp = np.array([-ev[1], ev[0]])  # +90 deg: leftward of travel
        det_s = sorted(idxs, key=lambda i: float(np.dot(det_pts[i].mean(axis=0), perp)))
        nl = e.getLanes()  # index 0 = rightmost, increasing leftward
        net_s = sorted(nl, key=lambda l: float(np.dot(np.asarray(l.getShape(), float).mean(axis=0), perp)))
        exact = len(det_s) == len(net_s)
        if exact:
            pairs = zip(det_s, net_s)
        else:
            # best contiguous window of net lanes for the detected block
            k, m = len(det_s), len(net_s)
            if k > m:
                pairs = zip(det_s[:m], net_s)
            else:
                costs = []
                for off in range(m - k + 1):
                    c = sum(np.min(np.linalg.norm(
                        np.asarray(net_s[off + r].getShape(), float)
                        - det_pts[i].mean(axis=0), axis=1))
                        for r, i in enumerate(det_s))
                    costs.append(c)
                off = int(np.argmin(costs))
                pairs = zip(det_s, net_s[off:off + k])
        for i, lane in pairs:
            lanes[i]['sumo'] = lane.getID()
            lanes[i]['exact'] = exact


def run_sim(sumocfg, sim_end, seed, closed_lane=None, sample_every=10, watch_edge=None, scale=1.0):
    cmd = ['sumo', '-c', str(sumocfg), '--seed', str(seed), '--end', str(sim_end),
           '--scale', str(scale),
           '--no-warnings', 'true', '--no-step-log', 'true', '--duration-log.disable', 'true']
    traci.start(cmd)
    if closed_lane is not None:
        traci.lane.setAllowed(closed_lane, [])  # allow nothing = closed
    speeds, arrived, waiting = [], 0, []
    while traci.simulation.getMinExpectedNumber() > 0 and traci.simulation.getTime() < sim_end:
        traci.simulationStep()
        arrived += traci.simulation.getArrivedNumber()
        t = traci.simulation.getTime()
        if watch_edge and int(t) % sample_every == 0:
            speeds.append(traci.edge.getLastStepMeanSpeed(watch_edge))
            waiting.append(traci.edge.getWaitingTime(watch_edge))
    traci.close()
    return dict(mean_speed=float(np.mean([s for s in speeds if s >= 0]) if speeds else np.nan),
                arrived=int(arrived),
                mean_waiting=float(np.mean(waiting)) if waiting else 0.0,
                n_samples=len(speeds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--camera', default='US12_Monona')
    ap.add_argument('--close_lane', type=int, default=None,
                    help='index from the candidates table; default = most-supported unambiguous lane')
    ap.add_argument('--sim_end', type=int, default=600)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--match_m', type=float, default=2.5, help='A/B centerline match threshold (m)')
    ap.add_argument('--scale', type=float, default=3.0, help='SUMO demand scale (wizard trips are sparse)')
    opts = ap.parse_args()
    cam = opts.camera
    args = build_args()
    sfp = Path(args.saving_path, 'federated')
    out = Path('results/dt_demo'); out.mkdir(parents=True, exist_ok=True)
    net_dir = Path('dataset/sumo') / cam
    net = sumolib.net.readNet(str(net_dir / 'osm.net.xml'))

    # ---- window A: detect on full data, map lanes to the net ----
    processed = build_processed(args, cam, sfp)
    geo = GeometricLearning(args, sfp)
    lanes_a = detect(geo, processed, cam, processed['gps_df'])
    map_lanes_to_net(net, lanes_a)
    print(f'\nWindow A: {len(lanes_a)} detected lanes. Candidates (offset-corrected mapping):')
    print(f"{'idx':>4} {'vehicles':>9} {'sumo lane':>22} {'edge dist m':>12} {'index exact':>12}")
    for i, L in enumerate(lanes_a):
        print(f"{i:>4} {len(L['ids']):>9} {str(L['sumo']):>22} {L.get('edge_d', float('nan')):>12.2f} {str(L['exact']):>12}")

    # ---- choose the lane to close ----
    if opts.close_lane is not None:
        ci = opts.close_lane
    else:
        ok = [i for i, L in enumerate(lanes_a) if L['sumo'] is not None and L['exact']]
        if not ok:
            ok = [i for i, L in enumerate(lanes_a) if L['sumo'] is not None]
            print('NOTE: no exact-count edge group (detected fewer lanes than the net carries);')
            print('      index alignment used the best contiguous window - confirm the SUMO lane')
            print('      in the table above and override with --close_lane if it looks wrong.')
        if not ok:
            sys.exit('no mapped lane at all; inspect the scene')
        ci = max(ok, key=lambda i: len(lanes_a[i]['ids']))
    closed = lanes_a[ci]
    print(f'\nClosing detected lane {ci}: {len(closed["ids"])} vehicles -> SUMO lane {closed["sumo"]} '
          f'(rank-matched{", exact group" if closed["exact"] else ""})')

    # ---- window B: same scene minus the closed lane's traffic ----
    gps_b = processed['gps_df'].filter(~pl.col('id').is_in(list(closed['ids'])))
    lanes_b = detect(geo, processed, cam, gps_b)

    # ---- change detector: which window-A lanes have no window-B match ----
    lat0 = float(np.nanmean(processed['gps_df']['x_gps'].to_numpy()))
    lon0 = float(np.nanmean(processed['gps_df']['y_gps'].to_numpy()))
    m_lon = M_LAT * np.cos(np.deg2rad(lat0))
    a_m = [_to_m(L['center'], lat0, lon0, m_lon) for L in lanes_a]
    b_m = [_to_m(L['center'], lat0, lon0, m_lon) for L in lanes_b]
    flagged = []
    for i, A in enumerate(a_m):
        dmin = min((det_to_ann_dist(A, B) for B in b_m), default=float('inf'))
        if dmin > opts.match_m:
            flagged.append(i)
            print(f'[CHANGE] window-A lane {i} has no window-B match (nearest {dmin:.1f} m) '
                  f'-> SUMO lane {lanes_a[i]["sumo"]}')
    if ci not in flagged:
        print('WARNING: the synthesized closure was NOT flagged — inspect before using the demo')

    # ---- twin update + re-simulation (before vs after) ----
    watch_edge = closed['sumo'].rsplit('_', 1)[0]
    sumocfg = net_dir / 'osm.sumocfg'
    print(f'\nSimulating {opts.sim_end}s (seed {opts.seed}), watching edge {watch_edge} ...')
    before = run_sim(sumocfg, opts.sim_end, opts.seed, closed_lane=None, watch_edge=watch_edge, scale=opts.scale)
    after = run_sim(sumocfg, opts.sim_end, opts.seed, closed_lane=closed['sumo'], watch_edge=watch_edge, scale=opts.scale)
    print(f"{'':12}{'mean speed m/s':>15}{'arrived veh':>13}{'mean wait s':>13}")
    print(f"{'before':<12}{before['mean_speed']:>15.2f}{before['arrived']:>13}{before['mean_waiting']:>13.1f}")
    print(f"{'after':<12}{after['mean_speed']:>15.2f}{after['arrived']:>13}{after['mean_waiting']:>13.1f}")

    # ---- artifacts ----
    result = dict(camera=cam, closed_detected_lane=ci, closed_vehicle_count=len(closed['ids']),
                  sumo_lane=closed['sumo'], exact_group=closed['exact'], watch_edge=watch_edge,
                  flagged=flagged, closure_flagged=ci in flagged, sim_end=opts.sim_end, scale=opts.scale,
                  seed=opts.seed, before=before, after=after,
                  lanes_window_a=len(lanes_a), lanes_window_b=len(lanes_b))
    (out / 'dt_closure_demo.json').write_text(json.dumps(result, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for i, L in enumerate(lanes_a):
        c = L['center']
        ax1.plot(c[:, 1], c[:, 0], color='0.6', lw=1.5)
    for L in lanes_b:
        c = L['center']
        ax1.plot(c[:, 1], c[:, 0], color='tab:blue', lw=1.0, alpha=0.8)
    cc = closed['center']
    ax1.plot(cc[:, 1], cc[:, 0], color='tab:red', lw=2.5, ls='--', label=f'closed lane {ci}')
    ax1.set_title(f'{cam}: window A (gray), window B (blue), closure (red)')
    ax1.legend(); ax1.set_xlabel('lon'); ax1.set_ylabel('lat')
    x = np.arange(2)
    ax2.bar(x - 0.2, [before['mean_speed'], after['mean_speed']][0:1] + [np.nan], 0.4)
    ax2.bar(x, [before['mean_speed'], after['mean_speed']], 0.35, color=['0.5', 'tab:red'])
    ax2.set_xticks(x, ['before', 'after'])
    ax2.set_ylabel(f'mean speed on {watch_edge} (m/s)')
    ax2.set_title(f"arrived: {before['arrived']} -> {after['arrived']}")
    fig.tight_layout()
    fig.savefig(out / 'dt_closure_demo.png', dpi=150)
    print(f'\nwrote {out}/dt_closure_demo.json and dt_closure_demo.png')


if __name__ == '__main__':
    main()
