'''NCAP per-step latents (UniAD/inference/runner_v2.py) -> inputs of every Table 2 method, in the formats of
riskmonitor_g (CachedBEVDataset caches, rebuttal2 functions), plus the training / test configs.

    python ncap_to_cache.py --latents <latents/tag> --ncap-out <neuro-ncap/outputs/TIME> --out <out dir>

Fields as uniad_rebuttal2/tools/arr/export_eval_tokens.py (checked 2026-11-01): plan_org = planning attn_module input,
traj = last-layer vehicle traj_query, traj_all = last-layer traj_query of all tracked agents, tracked / motion_all /
motion_scores_all of all tracked agents (sdc removed), pre = plan before the collision optimisation.
Outputs:
  cache_base    RiskMonitor-T (plan_org, vehicle traj)       cache_c2full  RiskMonitor-G (traj_all + geometry G1)
  cache_plan    LP-FCN (plan_org only)
  lists/        train / val / test, and train_m1..4 (bagging members: member i leaves out quarter i of the training
                runs, stratified by the run outcome; the data are balanced, so not the nuScenes rule of all
                positives + a quarter of the negatives)
  configs/      {T,G,LPFCN}_m{i}.json (training, checkpoints every 2 epochs), {T,G}_{val,test}.json (a1_test_export)
  samples.csv   one row per labelled step: split (or fold), scene, label, and the scores that need no training:
                rbd, gmm (rebuttal2/v2fuse_baselines.one) over the 6 plan steps, rbd4, gmm4 over the first 4 steps
                (the label horizon with --label-steps 4), mcd (trace of the endpoint covariance of the first 4
                MC Dropout passes, rebuttal2/mcd_evalmode.py)
--split cv: scene-disjoint cross-validation (plan §5.4 option 2): the scenes are spread over --folds folds with
balanced sample counts; fold k tests on its scenes, selects epochs on the scenes of fold k + 1 and trains on the
rest. Each fold is written to <out>/f<k>/ with the layout above.
Samples: every step of a run that NCAP logged (auto mode; the priming steps replay the log), each with its own
label: UniAD's CollisionLoss (delta 1.0 m) of the plan before the collision optimisation against NCAP's actor
boxes at t + 0.5 s ... t + N * 0.5 s > 0 (ncap_labels.py; N = --label-steps, 6 as planning.loss_collision_aa_2).
Steps whose future runs past the end of the scene have no label and are left out.
Provisional rules (to be settled later):
  - split (--split), by run (all steps of a run in one split): scene = scenes 0103 / 0106 / 0108 / 0110 train
    (old split, plan §4.3), the other runs val : test = 1 : 2 per run outcome; random = train / val / test 6:1:3
    over runs, stratified by the run outcome (any_collide@0.0s)
  - training as the NCAP RiskMonitor of the paper (focal alpha 0.5, gamma 2, lr 1e-3, 20 epochs) for T, G and LP-FCN
    (LP-FCN with the 'riskmonitor' lr schedule of lpfcn_train.py; its 'plan' schedule reaches only ~1.5e-6), 4
    members each; the epoch is chosen on val (score_cv.py)
'''
import argparse
import copy
import glob
import json
import os
import os.path as osp
import re
import sys
from collections import defaultdict

import numpy as np
import torch

RM = '/nobackup/proj/disk/cvls/personal/shipengl/riskmonitor_g'  # copy of naiss2024-6-300/shared/riskmonitor_g
sys.path.insert(0, RM)
sys.path.insert(0, osp.join(RM, 'rebuttal2'))
from c2full_data import N_G1, geometry  # noqa: E402
from common import LABEL_KEY  # noqa: E402
from mcd_evalmode import endpoint_cov  # noqa: E402
import baselines_rbd_gmm  # noqa: E402
from v2fuse_baselines import one as rbd_gmm  # noqa: E402
from v2fuse_data import cache_path  # noqa: E402

from ncap_labels import collision_losses, load_actors  # noqa: E402

TRAIN_SCENES = {'0103', '0106', '0108', '0110'}
LABEL_DELTA = 1.0  # planning.loss_collision_aa_2; --label-delta / the plan's label key can change it
STEP_RE = re.compile(r'_run(\d+)_step(\d+)\.pt$')


def runs_of(combo_dir, ncap_dir, label_steps, max_runs=None):
    '''yields (run, [(step, rec, label)] of the labelled auto-mode steps, collided); runs 0 .. max_runs - 1'''
    files = glob.glob(osp.join(combo_dir, '*_run*_step*.pt'))
    by_run = {}
    for p in files:
        r, k = map(int, STEP_RE.search(p).groups())
        by_run.setdefault(r, []).append((k, p))
    for r in sorted(by_run):
        if max_runs is not None and r >= max_runs:
            continue
        d = osp.join(ncap_dir, f'run_{r}')
        if not osp.isdir(d):
            d = ncap_dir  # runs == 1: no run_<i> subfolder
        if not osp.exists(osp.join(d, 'metrics.json')):
            print(f'  {osp.basename(combo_dir)} run {r}: no NCAP metrics (unfinished run), skipped')
            continue
        logged = set(json.load(open(osp.join(d, 'trajectories.json'))))
        collided = bool(json.load(open(osp.join(d, 'metrics.json')))['any_collide@0.0s'])
        actors = load_actors(d)
        steps = []
        for k, p in sorted(by_run[r]):
            x = torch.load(p, map_location='cpu')
            if str(round(x['timestamp'] * 1e6)) not in logged:
                continue
            losses = collision_losses(x, actors, label_steps)
            if losses is not None:
                steps.append((k, x, losses[LABEL_DELTA] > 0))
        yield r, steps, collided


def fields(x):
    '''the export_eval_tokens.py fields of one step'''
    boxes = x['track_boxes'][:-1]                                          # sdc box is the last row
    A = len(boxes)
    traj = x['traj_query'][-1, 0].float()
    return dict(plan_org=x['plan_query_org'].float().reshape(1, -1), traj=traj.flatten(1), nq_track=len(traj),
                traj_all=x['traj_query_all'][0, :-1].float().flatten(1),
                tracked=torch.stack([boxes[:, 0], boxes[:, 1], boxes[:, 3], boxes[:, 4], boxes[:, 6],
                                     x['track_labels'].float(), x['track_scores'].float()], 1).float(),
                motion_all=x['traj_preds'][-1, 0, :A, :, :6, :2].float(),
                motion_scores_all=x['traj_scores'][-1, 0, :A].float(),
                pre=x['plan_preopt'][0, :6, :2].float())


def items(token, scene, x, label):
    f = fields(x)
    lab = torch.tensor([float(label)])
    A = len(f['tracked'])
    common = {'label_path': token, LABEL_KEY: lab, 'plan_org': f['plan_org']}
    base = {**common, 'nq_track': f['nq_track'], 'traj': f['traj']}
    g = geometry(f)[:, :N_G1] if A else np.zeros((0, N_G1), np.float32)
    g_item = {**common, 'nq_track': A, 'traj': f['traj_all'], 'track_pose': torch.from_numpy(g)}
    plan_item = {**common, 'nq_track': 0, 'traj': torch.zeros(0, 1)}   # as rebuttal2/lpfcn.py data
    s = rbd_gmm({**f, 'sample_token': token, 'scene_token': scene, 'col_pre_d1.0': float(label)})
    s4 = rbd_gmm({**f, 'pre': f['pre'][:4], 'motion_all': f['motion_all'][..., :4, :],
                  'sample_token': token, 'scene_token': scene, 'col_pre_d1.0': float(label)})
    mcd = float(np.trace(endpoint_cov(x['mcd_plans'][:4].numpy()))) if 'mcd_plans' in x else float('nan')
    row = dict(token=token, scene=scene, label=int(label), n_vehicles=f['nq_track'], n_agents=A,
               rbd=s['rbd'], gmm=s['gmm'], rbd4=s4['rbd'], gmm4=s4['gmm'], mcd=mcd)
    return base, g_item, plan_item, row


def split_runs(runs, how, seed):
    '''split name per run'''
    rng = np.random.default_rng(seed)
    out = [None] * len(runs)
    if how == 'scene':
        rest = []
        for i, (combo, *_) in enumerate(runs):
            if combo.split('-')[1] in TRAIN_SCENES:
                out[i] = 'train'
            else:
                rest.append(i)
        fracs = (0.0, 1 / 3)  # val : test = 1 : 2
    else:
        rest = list(range(len(runs)))
        fracs = (0.6, 0.1)
    for label in (False, True):
        idx = [i for i in rest if runs[i][2] == label]
        idx = [idx[j] for j in rng.permutation(len(idx))]
        n_train, n_val = round(fracs[0] * len(idx)), round(fracs[1] * len(idx))
        for j, i in enumerate(idx):
            out[i] = 'train' if j < n_train else 'val' if j < n_train + n_val else 'test'
    return out


def scene_folds(runs, k):
    '''fold per scene: scenes by sample count (largest first), each to the fold with the fewest samples'''
    n = defaultdict(int)
    for combo, _, _, steps in runs:
        n[combo.split('-')[1]] += len(steps)
    size, fold = [0] * k, {}
    for sc in sorted(n, key=lambda sc: (-n[sc], sc)):
        fold[sc] = int(np.argmin(size))
        size[fold[sc]] += n[sc]
    return fold


def members(runs, train_idx, n, seed):
    '''bagging: member i leaves out quarter i of the training runs (stratified by the run outcome)'''
    rng = np.random.default_rng(seed)
    part = {}
    for c in (False, True):
        idx = [i for i in train_idx if runs[i][2] == c]
        for j, i in enumerate(rng.permutation(idx)):
            part[i] = j % n
    return [[i for i in train_idx if part[i] != m] for m in range(n)]


def configs(out, epochs, n_members):
    '''training configs per member and test configs: RiskMonitor-T (V2fuse_all/base), -G (V2fuse_all/F3),
    LP-FCN (LPFCN/uniad/plan); all with the NCAP training of the paper (final.tex L642)'''
    src = osp.join(RM, 'configs/rebuttal2')
    os.makedirs(osp.join(out, 'configs'), exist_ok=True)
    for run, cache, pose in (('T', 'cache_base', False), ('G', 'cache_c2full', True)):
        d = osp.join(src, 'V2fuse_all', 'base' if run == 'T' else 'F3')
        te = json.load(open(osp.join(d, 'config_mixup_ensemble_test_eval.json')))
        te['val_loader']['args'].update(cache_dir=osp.join(out, cache), with_pose=pose, max_num_agents=256)
        for split in ('val', 'test'):
            te['val_loader']['args']['data_dir'] = osp.join(out, f'lists/{split}.txt')
            te['trainer']['save_dir'] = osp.join(out, run, 'test')
            json.dump(te, open(osp.join(out, 'configs', f'{run}_{split}.json'), 'w'), indent=4)
        for m in range(1, n_members + 1):
            tr = json.load(open(osp.join(d, f'config_mixup_ensemble{m}.json')))
            if run == 'T':
                tr['Trainer']['alpha'] = 1.0  # mixup (provisional; nuScenes base uses 3, F3 none)
            tr['name'] = f'NCAP_{run}_m{m}'
            tr['train_loader']['args'].update(data_dir=osp.join(out, f'lists/train_m{m}.txt'),
                                              cache_dir=osp.join(out, cache), with_pose=pose, max_num_agents=256)
            tr['loss']['args']['alpha'] = 0.5
            tr['trainer'].update(epochs=epochs, save_period=2, save_dir=osp.join(out, run, f'm{m}', 'saved'))
            json.dump(tr, open(osp.join(out, 'configs', f'{run}_m{m}.json'), 'w'), indent=4)
    for m in range(1, n_members + 1):
        tr = json.load(open(osp.join(src, f'LPFCN/uniad/plan/config_member{m}.json')))
        tr['name'] = f'NCAP_LPFCN_m{m}'
        tr['Trainer']['lr_schedule'] = 'riskmonitor'
        tr['train_loader']['args'].update(data_dir=osp.join(out, f'lists/train_m{m}.txt'),
                                          cache_dir=osp.join(out, 'cache_plan'))
        tr['loss']['args']['alpha'] = 0.5
        tr['trainer'].update(epochs=epochs, save_period=2, save_dir=osp.join(out, 'LPFCN', f'm{m}', 'saved'))
        json.dump(tr, open(osp.join(out, 'configs', f'LPFCN_m{m}.json'), 'w'), indent=4)


def write_split(out, runs, split_idx, args):
    '''lists, caches and configs of one train / val / test split (+ the bagging members of train)'''
    lists = dict(split_idx)
    for m, idx in enumerate(members(runs, split_idx['train'], args.members, args.seed), 1):
        lists[f'train_m{m}'] = idx
    os.makedirs(osp.join(out, 'lists'), exist_ok=True)
    for name, idx in lists.items():
        base, g, plan = ([it[j] for i in idx for it in runs[i][3]] for j in range(3))
        lf = osp.join(out, 'lists', f'{name}.txt')
        with open(lf, 'w') as f:
            f.writelines(it['label_path'] + '\n' for it in base)
        for d, suffix, its in (('cache_base', '', base), ('cache_c2full', '__pose', g), ('cache_plan', '', plan)):
            os.makedirs(osp.join(out, d), exist_ok=True)
            torch.save(its, cache_path(osp.join(out, d), lf, suffix))
        pos = sum(int(it[LABEL_KEY]) for it in base)
        print(f'  {name}: {len(idx)} runs, {len(base)} samples, {pos} positive ({100 * pos / max(len(base), 1):.1f} %), '
              f'scenes {sorted({runs[i][0].split("-")[1] for i in idx})}, '
              f'{sum(it["nq_track"] == 0 for it in g)} without agents')
    configs(out, args.epochs, args.members)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--latents', required=True, help='latents/<tag> with one folder per scenario-sequence')
    p.add_argument('--ncap-out', required=True, help='neuro-ncap/outputs/<TIME>')
    p.add_argument('--out', required=True)
    p.add_argument('--epochs', type=int, default=20, help='RiskMonitor-T / -G (NCAP: 20, final.tex L642)')
    p.add_argument('--split', choices=['scene', 'random', 'cv', 'plan'], default='scene')
    p.add_argument('--plan', help='ncap_plan.py output (--split plan): scene -> set, runs per pair, label key')
    p.add_argument('--label-delta', type=float, default=1.0, help='safe distance of the label and of RBD (m)')
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--members', type=int, default=4)
    p.add_argument('--label-steps', type=int, default=6, help='plan steps checked for the label (6 or 4)')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    plan = json.load(open(args.plan)) if args.split == 'plan' else None
    if plan:  # label key col_n<steps>_d<delta>
        m = re.fullmatch(r'col_n(\d+)_d([\d.]+)', plan['key'])
        args.label_steps, args.label_delta = int(m.group(1)), float(m.group(2))
    global LABEL_DELTA
    LABEL_DELTA = args.label_delta
    # RBD under the label rule (baselines_rbd_gmm: ego box of CollisionLoss with its DELTA)
    baselines_rbd_gmm.EGO_W, baselines_rbd_gmm.EGO_L = 1.85 + LABEL_DELTA, 4.084 + LABEL_DELTA
    print(f'label: CollisionLoss over {args.label_steps} steps, delta {LABEL_DELTA} m')
    runs = []  # (combo, run, collided, [(base, G, plan, row)])
    for combo_dir in sorted(glob.glob(osp.join(args.latents, '*-*'))):
        combo = osp.basename(combo_dir)
        scene = combo.split('-')[1]
        if plan and combo not in plan['runs']:
            continue
        max_runs = plan['runs'][combo] if plan else None
        for r, steps, collided in runs_of(combo_dir, osp.join(args.ncap_out, combo), args.label_steps, max_runs):
            runs.append((combo, r, collided,
                         [items(f'{combo}_run{r}_step{k}', scene, x, label) for k, x, label in steps]))
    import pandas as pd
    rows = []
    if args.split == 'cv':
        fold = scene_folds(runs, args.folds)
        for k in range(args.folds):
            role = {f: 'test' if f == k else 'val' if f == (k + 1) % args.folds else 'train' for f in range(args.folds)}
            split_idx = {s: [i for i, r in enumerate(runs) if role[fold[r[0].split('-')[1]]] == s]
                         for s in ('train', 'val', 'test')}
            print(f'fold {k}:')
            write_split(osp.join(args.out, f'f{k}'), runs, split_idx, args)
        for combo, r, collided, steps in runs:
            for *_, row in steps:
                rows.append(dict(fold=fold[combo.split('-')[1]], combo=combo, run=r, run_collided=int(collided), **row))
    else:
        split = ([plan['split'][c.split('-')[1]] for c, *_ in runs] if plan else split_runs(runs, args.split, args.seed))
        write_split(args.out, runs, {s: [i for i in range(len(runs)) if split[i] == s] for s in ('train', 'val', 'test')},
                    args)
        for (combo, r, collided, steps), sp in zip(runs, split):
            for *_, row in steps:
                rows.append(dict(split=sp, combo=combo, run=r, run_collided=int(collided), **row))
    pd.DataFrame(rows).to_csv(osp.join(args.out, 'samples.csv'), index=False)

if __name__ == '__main__':
    main()
