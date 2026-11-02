'''Dataset plan for NCAP as the paper's appendix B.4 ILP + the 6:1:3 split of Sec. 5.4.1, from pilot statistics.

    python ncap_plan.py --steps <check_ncap_labels.py csv> --key col_n6_d1.0 --target 0.5 --out plan.json

Per scene-sequence pair i (pilot runs): t_i = labelled timestamps per run, c_i = collision timestamps per run
(label column --key > 0, auto steps). B.4: integer runs r_i >= 45, sum r_i <= 1200, collision share of the
timestamps in [target - band, target + band] (the equations use 0.46 / 0.54 for 50 %), minimise the largest
difference D between the r_i. Here additionally (paper Sec. 5.4.1):
  - split: whole scenes go to train / val / test (no scene in two sets, so no sequence either), every set
    contains all three scenario types, expected timestamps 6:1:3 (+- --ratio-tol);
  - every set's collision share within target +- set band, the first feasible of --set-bands (so that test and
    train have similar shares; the paper constrains only the total);
  - balance between pairs: max r_i <= --spread * min r_i;
  - ties in D are broken towards more runs (objective D - 1e-4 sum r_i).
The candidate splits are the scene assignments (3^14) closest to 6:1:3 with equal runs; each is solved and the
smallest D wins. If the target is not feasible, --fallback min raises the target in steps of 1 % (same bands)
until it is: the lowest collision share reachable under these balance conditions.
'''
import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

SETS = ('train', 'val', 'test')


def pair_stats(path, key):
    df = pd.read_csv(path, dtype={'combo': str})
    df = df[(df.phase == 'auto') & df[key].notna()]
    g = df.groupby('combo')
    st = pd.DataFrame({'runs': g.run.nunique(), 'run_collided': g.apply(lambda d: d.groupby('run').run_collided.first().mean()),
                       'steps': g.size(), 'pos': g[key].apply(lambda v: int((v > 0).sum()))})
    st['t'] = st.steps / st.runs
    st['c'] = st.pos / st.runs
    st['rate'] = st.pos / st.steps
    st['type'] = [c.split('-')[0] for c in st.index]
    st['scene'] = [c.split('-')[1] for c in st.index]
    return st


def candidate_splits(st, ratio, n_keep):
    scenes = sorted(st.scene.unique())
    w = np.array([st.t[st.scene == s].sum() for s in scenes])  # expected timestamps per scene, equal runs
    types = sorted(st.type.unique())
    has = np.array([[((st.scene == s) & (st.type == ty)).any() for ty in types] for s in scenes], float)
    n = len(scenes)
    a = np.stack(np.unravel_index(np.arange(3 ** n), (3,) * n), 1).astype(np.int8)  # (3^n, n): 0 train 1 val 2 test
    score = np.zeros(len(a))
    ok = np.ones(len(a), bool)
    for k in range(3):
        m = (a == k).astype(np.float32)
        score += np.abs(m @ w / w.sum() - ratio[k])
        ok &= ((m @ has) > 0).all(1)
    idx = np.flatnonzero(ok)
    idx = idx[np.argsort(score[idx], kind='stable')[:n_keep]]
    return [{s: SETS[a[i, j]] for j, s in enumerate(scenes)} for i in idx], scenes


def solve(st, split, target, band, set_band, args, feasibility=False):
    '''r_i per pair (Series) or None'''
    n = len(st)
    t, c = st.t.values, st.c.values
    member = {s: np.array([split[sc] == s for sc in st.scene]) for s in SETS}
    rows, lo, hi = [], [], []

    def add(coef_r, l, u, coef_lohi=(0.0, 0.0)):
        rows.append(np.r_[coef_r, coef_lohi])
        lo.append(l)
        hi.append(u)

    for i in range(n):  # lo <= r_i <= hi
        e = np.zeros(n)
        e[i] = 1
        add(e, -np.inf, 0, (0, -1))
        add(e, 0, np.inf, (-1, 0))
    add(np.ones(n), -np.inf, args.max_total)
    add(np.zeros(n), -np.inf, 0, (-args.spread, 1))  # hi <= spread * lo
    for m, b in [(m, set_band) for m in member.values()] + [(np.ones(n, bool), band)]:
        add(m * (c - (target + b) * t), -np.inf, 0)
        add(m * (c - max(target - b, 0) * t), 0, np.inf)
    for k, s in enumerate(SETS):
        rho = args.ratio[k]
        add(member[s] * t - (rho + args.ratio_tol) * t, -np.inf, 0)
        add(member[s] * t - (rho - args.ratio_tol) * t, 0, np.inf)
    obj = np.zeros(n + 2) if feasibility else np.r_[-1e-4 * np.ones(n), -1.0, 1.0]
    res = milp(obj, constraints=LinearConstraint(np.array(rows), lo, hi), integrality=np.ones(n + 2),
               bounds=Bounds(np.r_[np.full(n, args.min_runs), args.min_runs, args.min_runs],
                             np.r_[np.full(n, args.max_runs), args.max_runs, args.max_runs]))
    if res.status != 0:
        return None
    return pd.Series(np.round(res.x[:n]).astype(int), index=st.index)


def summary(st, split, r):
    out = {}
    for s in SETS + ('all',):
        m = np.ones(len(st), bool) if s == 'all' else np.array([split[sc] == s for sc in st.scene])
        T, C = float((st.t * r)[m].sum()), float((st.c * r)[m].sum())
        out[s] = dict(runs=int(r[m].sum()), timestamps=round(T), collision_share=C / T,
                      scenes=sorted({sc for sc, x in zip(st.scene, m) if x}))
    total = out['all']['timestamps']
    for s in SETS:
        out[s]['timestamp_share'] = out[s]['timestamps'] / total
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', required=True)
    p.add_argument('--key', required=True, help='label column, e.g. col_n6_d1.0')
    p.add_argument('--target', type=float, required=True)
    p.add_argument('--band', type=float, default=0.04)
    p.add_argument('--min-runs', type=int, default=45)
    p.add_argument('--max-runs', type=int, default=400)
    p.add_argument('--max-total', type=int, default=1200)
    p.add_argument('--ratio', type=float, nargs=3, default=(0.6, 0.1, 0.3))
    p.add_argument('--ratio-tol', type=float, default=0.03)
    p.add_argument('--candidates', type=int, default=200)
    p.add_argument('--set-bands', type=float, nargs='+', default=(0.08, 0.12, 0.16))
    p.add_argument('--spread', type=float, default=2.0)
    p.add_argument('--fallback', choices=['none', 'min'], default='none')
    p.add_argument('--out')
    args = p.parse_args()
    st = pair_stats(args.steps, args.key)
    print(st[['type', 'scene', 'runs', 'run_collided', 't', 'c', 'rate']].round(3).to_string())
    splits, scenes = candidate_splits(st, args.ratio, args.candidates)
    print(f'{len(st)} pairs, {len(scenes)} scenes, {len(splits)} candidate splits')

    def search(target):
        for sb in args.set_bands:
            best = None
            for split in splits:
                r = solve(st, split, target, args.band, sb, args)
                if r is not None and (best is None or r.max() - r.min() < best[1].max() - best[1].min()):
                    best = (split, r)
            if best is not None:
                return best + (sb,)
        return None

    mode, target = 'target', args.target
    best = search(target)
    while best is None and args.fallback == 'min' and target < 0.95:
        mode, target = 'min', round(target + 0.01, 2)
        best = search(target)
    if mode == 'min' and best is not None:
        print(f'target {args.target} not feasible; lowest feasible target {target}')
    if best is None:
        print('infeasible')
        return
    split, r, set_band = best
    summ = summary(st, split, r)
    print(f'mode {mode}: target {target} +- {args.band} (sets +- {set_band}): D = {r.max() - r.min()}, '
          f'runs per pair {r.min()}..{r.max()}, total {r.sum()}')
    for s, v in summ.items():
        print(f"  {s}: {v['runs']} runs, {v['timestamps']} timestamps"
              + (f" ({100 * v['timestamp_share']:.1f} %)" if s != 'all' else '')
              + f", expected collision share {100 * v['collision_share']:.1f} %, scenes {v['scenes']}")
    print('  runs: ' + ', '.join(f'{k} {v}' for k, v in r.items()))
    if args.out:
        json.dump(dict(key=args.key, target=target, requested_target=args.target, band=args.band, set_band=set_band,
                       mode=mode, split=split,
                       runs={k: int(v) for k, v in r.items()}, summary=summ,
                       pilot=st[['runs', 't', 'c', 'rate']].round(4).to_dict('index'), args=vars(args)),
                  open(args.out, 'w'), indent=1)
        print(f'-> {args.out}')


if __name__ == '__main__':
    main()
