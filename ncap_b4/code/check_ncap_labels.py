'''Checks of the per-step NCAP labels (ncap_labels.py) and their statistics, before they are used for training.

    python check_ncap_labels.py --latents <latents/tag> --ncap-out <neuro-ncap/outputs/TIME> --out labels.csv

Coordinates:
  1. target actor from actors.json + ego_poses.json in NCAP's ego frame == metrics.json target_actor_in_ego
  2. lidar2ego = ego2world^-1 @ lidar2world (latents) is the same at every step (fixed calibration)
  3. target actor in the LiDAR frame vs the closest UniAD track box (detection error, should be ~< 1 m)
Labels: CollisionLoss > 0 for the first 6 and 4 plan steps at delta 0 / 0.5 / 1.0, every step of every run
(phase = priming / auto as NCAP logs them), compared with the run outcome (any_collide@0.0s).
'''
import argparse
import csv
import glob
import json
import os.path as osp
import re
from collections import defaultdict

import numpy as np
import torch

from ncap_labels import DELTAS, actor_pose, collision_losses, lidar2world, load_actors, world_to_lidar

STEP_RE = re.compile(r'_run(\d+)_step(\d+)\.pt$')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--latents', required=True)
    p.add_argument('--ncap-out', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    rows, err_ego, err_det, l2e, outside = [], [], [], [], []
    for combo_dir in sorted(glob.glob(osp.join(args.latents, '*-*'))):
        combo = osp.basename(combo_dir)
        by_run = defaultdict(list)
        for f in glob.glob(osp.join(combo_dir, '*_run*_step*.pt')):
            r, k = map(int, STEP_RE.search(f).groups())
            by_run[r].append((k, f))
        for r, files in sorted(by_run.items()):
            d = osp.join(args.ncap_out, combo, f'run_{r}')
            if not osp.exists(osp.join(d, 'metrics.json')):
                continue
            actors = load_actors(d)
            target = [a for a in actors if a['target']]
            metrics = json.load(open(osp.join(d, 'metrics.json')))
            ego_poses = {int(k): np.array(v) for k, v in json.load(open(osp.join(d, 'ego_poses.json'))).items()}
            logged = set(json.load(open(osp.join(d, 'trajectories.json'))))
            # 1. NCAP's own target pose in its ego frame
            for ri in metrics['recall_info']:
                if ri and target and ri['timestamp'] in ego_poses:
                    tp = actor_pose(target[0], ri['timestamp'])
                    if tp is None:
                        outside.append(combo)
                        continue
                    x, y, _ = tp
                    e = np.linalg.inv(ego_poses[ri['timestamp']]) @ [x, y, 0, 1]
                    err_ego.append(np.hypot(*(e[:2] - np.array(ri['target_actor_in_ego'][0])[:2, 3])))
            for k, f in sorted(files):
                rec = torch.load(f, map_location='cpu')
                ts = round(rec['timestamp'] * 1e6)
                if ts in ego_poses:  # 2. calibration
                    m = np.linalg.inv(ego_poses[ts]) @ lidar2world(rec)
                    l2e.append(np.r_[m[:3, 3], np.degrees(np.arctan2(m[1, 0], m[0, 0]))])
                if target and len(rec['track_boxes']) > 1:  # 3. detection of the target actor
                    tp = actor_pose(target[0], ts)
                    if tp is not None:
                        xy = world_to_lidar(rec, np.array(tp[:2]))
                        dist = (rec['track_boxes'][:-1, :2].double().numpy() - xy).__pow__(2).sum(1) ** 0.5
                        if np.hypot(*xy) < 35:
                            err_det.append(dist.min())
                row = dict(combo=combo, run=r, step=k, phase='auto' if str(ts) in logged else 'priming',
                           run_collided=int(metrics['any_collide@0.0s']))
                for n in (6, 4):
                    losses = collision_losses(rec, actors, n)
                    for dl in DELTAS:
                        row[f'col_n{n}_d{dl}'] = '' if losses is None else losses[dl]
                # for reference: the executed plan (after the collision optimisation; == plan_preopt when it is off)
                losses = collision_losses({**rec, 'plan_preopt': rec['plan']}, actors, 6)
                for dl in DELTAS:
                    row[f'colexec_n6_d{dl}'] = '' if losses is None else losses[dl]
                rows.append(row)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    q = lambda v: np.percentile(v, [50, 90, 100]).round(3).tolist()
    print(f'1. target in NCAP ego frame vs target_actor_in_ego: {len(err_ego)} poses, error m [p50, p90, max] {q(err_ego)}; '
          f'target outside its actors.json time range at {len(outside)} logged steps: {sorted(set(outside))}')
    l2e = np.array(l2e)
    print(f'2. lidar2ego over {len(l2e)} steps: mean (x, y, z, yaw deg) {l2e.mean(0).round(3).tolist()}, '
          f'std {l2e.std(0).round(4).tolist()}')
    print(f'3. target vs closest UniAD track box within 35 m: {len(err_det)} steps, m [p50, p90, max] {q(err_det)}')
    print('\nlabels (auto steps; - = future past the end of the scene)')
    auto = [r for r in rows if r['phase'] == 'auto']
    for n in (6, 4):
        for dl in DELTAS:
            key = f'col_n{n}_d{dl}'
            valid = [r for r in auto if r[key] != '']
            pos = [r for r in valid if r[key] > 0]
            by = {s: [r for r in valid if r['combo'].startswith(s)] for s in ('frontal', 'side', 'stationary')}
            agree = np.mean([(r[key] > 0) == r['run_collided'] for r in valid])
            print(f'  N={n} delta={dl}: {len(valid)}/{len(auto)} valid, positive {100 * len(pos) / len(valid):.1f} % '
                  + ' '.join(f'{s} {100 * np.mean([r[key] > 0 for r in v]):.1f} %' for s, v in by.items())
                  + f' | same as run outcome {100 * agree:.1f} %')
    # collided runs: positives near the end?  per run: fraction of positive auto steps, collided vs not
    key = 'col_n6_d1.0'
    for c in (1, 0):
        fr = defaultdict(list)
        for r in auto:
            if r['run_collided'] == c and r[key] != '':
                fr[(r['combo'], r['run'])].append(r[key] > 0)
        last = [v[-1] for v in fr.values()]
        print(f'  runs collided={c}: {len(fr)} runs, mean positive fraction {100 * np.mean([np.mean(v) for v in fr.values()]):.1f} %, '
              f'last auto step positive in {100 * np.mean(last):.1f} % of runs, runs with no positive step '
              f'{sum(not any(v) for v in fr.values())}')
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
