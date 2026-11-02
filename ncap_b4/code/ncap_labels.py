'''Per-step collision labels on NCAP, defined as planning.loss_collision_aa_{0,1,2} on nuScenes: UniAD's
CollisionLoss (uniad_rebuttal2/projects/mmdet3d_plugin/losses/planning_loss.py, delta 0 / 0.5 / 1.0 m) between
the plan before the collision optimisation and the boxes of all actors at t + 0.5 s ... t + 3.0 s, in the LiDAR
frame at t (PlanningHead.loss pairs plan step i with future frame i + 1). Label = loss > 0.

The actor boxes are NCAP's ground truth (actors.json of the run: every actor's world poses and w / l / h; NCAP
actors do not react to the ego vehicle, so their future does not depend on the plan), not UniAD's detections.
The ego box heading at each step is the direction of the plan (nuScenes uses the logged ego heading, which does
not exist after a run ends); CollisionLoss only uses axis-aligned extents, so the sign of the heading is irrelevant.
'''
import importlib.util
import json
import os.path as osp

import numpy as np
import torch

UNIAD = '/nobackup/proj/disk/cvls/personal/shipengl/uniad_rebuttal2'  # copy of naiss2024-6-300/shared/uniad_rebuttal2
_spec = importlib.util.spec_from_file_location(
    'uniad_planning_loss', osp.join(UNIAD, 'projects/mmdet3d_plugin/losses/planning_loss.py'))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
DELTAS = (0.0, 0.5, 1.0)
COLLISION_LOSSES = {d: _mod.CollisionLoss(delta=d, weight=1.0) for d in DELTAS}
STEP_US = 500_000  # UniAD plans at 2 Hz


class Boxes:
    '''the two attributes of LiDARInstance3DBoxes that CollisionLoss reads (its BEV corners are corners[:, [0, 3, 4, 7]])'''

    def __init__(self, bev):  # (N, 4, 2)
        self.tensor = torch.zeros(len(bev), 7)
        self.corners = torch.zeros(len(bev), 8, 3)
        self.corners[:, [0, 3, 4, 7], :2] = bev


def load_actors(run_dir):
    return [dict(t=np.array(a['timestamps'], np.int64), poses=np.array(a['poses'], np.float64),
                 dims=np.array(a['dims'], np.float64), target=a['is_target_actor'])
            for a in json.load(open(osp.join(run_dir, 'actors.json')))]


def scene_end(actors):
    return max(int(a['t'][-1]) for a in actors)


def actor_pose(a, t_us):
    '''(x, y, yaw) in the world at t_us, linear in position and yaw (NCAP interpolates the same way); None if the
    actor does not exist then (NCAP's collision check skips such actors too)'''
    ts = a['t']
    if not ts[0] <= t_us <= ts[-1]:
        return None
    yaws = np.arctan2(a['poses'][:, 1, 0], a['poses'][:, 0, 0])
    if len(ts) == 1:
        return a['poses'][0, 0, 3], a['poses'][0, 1, 3], yaws[0]
    i = min(int(np.searchsorted(ts, t_us, side='right')) - 1, len(ts) - 2)
    f = (t_us - ts[i]) / (ts[i + 1] - ts[i])
    xy = (1 - f) * a['poses'][i, :2, 3] + f * a['poses'][i + 1, :2, 3]
    dyaw = np.arctan2(np.sin(yaws[i + 1] - yaws[i]), np.cos(yaws[i + 1] - yaws[i]))
    return xy[0], xy[1], yaws[i] + f * dyaw


def actor_corners(actors, t_us):
    '''(N, 4, 2) world BEV corners of the actors that exist at t_us (NCAP pose: x along the length)'''
    out = []
    for a in actors:
        p = actor_pose(a, t_us)
        if p is None:
            continue
        x, y, yaw = p
        w, l = a['dims'][:2]
        local = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        out.append(local @ rot.T + [x, y])
    return np.array(out).reshape(-1, 4, 2)


def lidar2world(rec):
    '''runner_v2 saves UniAD's lidar_pose input, whose rotation is transposed (UniAD data prep, inference/server.py)'''
    m = rec['lidar_pose'].double().numpy().copy()
    m[:3, :3] = m[:3, :3].T
    return m


def world_to_lidar(rec, xy):
    m = lidar2world(rec)
    pts = np.concatenate([xy.reshape(-1, 2), np.full((xy.size // 2, 1), m[2, 3])], 1)  # at the LiDAR height
    return ((pts - m[:3, 3]) @ m[:3, :3])[:, :2].reshape(xy.shape)


def plan_heading(plan):
    '''CollisionLoss yaw of each plan step: angle from the LiDAR +y axis (forward), from the previous point'''
    d = np.diff(np.concatenate([np.zeros((1, 2)), plan]), axis=0)
    yaw, out = 0.0, []
    for dx, dy in d:
        if np.hypot(dx, dy) > 1e-2:
            yaw = np.arctan2(dx, dy)
        out.append(yaw)
    return np.array(out)


def collision_losses(rec, actors, n_steps=6):
    '''{delta: CollisionLoss} of the plan before the collision optimisation over its first n_steps steps;
    None if t + n_steps * 0.5 s is past the end of the scene (actors would be missing)'''
    t0 = round(rec['timestamp'] * 1e6)
    if t0 + n_steps * STEP_US > scene_end(actors):
        return None
    plan = rec['plan_preopt'][0, :n_steps, :2].float()
    gt = torch.zeros(1, n_steps, 3)
    gt[0, :, 2] = torch.from_numpy(plan_heading(plan.double().numpy())).float()
    boxes = [Boxes(torch.from_numpy(world_to_lidar(rec, actor_corners(actors, t0 + k * STEP_US))).float())
             for k in range(1, n_steps + 1)]
    with torch.no_grad():
        return {d: float(loss(plan[None], gt, None, boxes)) for d, loss in COLLISION_LOSSES.items()}
