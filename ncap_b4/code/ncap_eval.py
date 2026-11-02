'''Table 2 (NCAP, UniAD) on the test split: AUROC, AP, Pr30/50/70/100 (common.metrics) with scene-bootstrap 95 % CIs
(2000 resamples of the test scenes, seed 0, as rebuttal2/bootstrap_compare.py) and paired differences with their
two-sided bootstrap p-values (2 x the share of resamples on the other side of 0; p < 1/2000 shows as 0).

    python ncap_eval.py --out <out dir>      reads samples.csv, {T,G}/scores_test.csv, LPFCN/scores_test.csv
With scene cross-validation (samples.csv has a fold column) the test scores of all folds f<k>/ are pooled (every
sample is tested once) and the per-fold metrics are also reported (mean +- std over folds, folds.csv).
Samples a method cannot score (no agents for RiskMonitor-T / -G) get the lowest score (riskmonitor_g README).
'''
import argparse
import glob
import os.path as osp
import sys

import numpy as np
import pandas as pd

RM = '/nobackup/proj/disk/cvls/personal/shipengl/riskmonitor_g'
sys.path.insert(0, osp.join(RM, 'rebuttal2'))
import bootstrap_compare  # noqa: E402
from common import metrics  # noqa: E402

bootstrap_compare.RECALLS = (0.3, 0.5, 0.7, 1.0)   # Table 2 reports Pr30/50/70/100
METHODS = {'MC Dropout': 'mcd', 'GMM': 'gmm', 'GMM-4': 'gmm4', 'RBD': 'rbd', 'RBD-4': 'rbd4', 'LP-FCN': 'lpfcn',
           'RiskMonitor-T': 'T', 'RiskMonitor-G': 'G'}
PAIRS = [('G', 'T'), ('G', 'rbd'), ('T', 'rbd'), ('G', 'rbd4')]
KEYS = ['AUROC', 'AP', 'Pr30', 'Pr50', 'Pr70', 'Pr100']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--n-boot', type=int, default=2000)
    args = p.parse_args()
    df = pd.read_csv(osp.join(args.out, 'samples.csv'), dtype={'scene': str})
    cv = 'fold' in df
    dirs = sorted(glob.glob(osp.join(args.out, 'f[0-9]*'))) if cv else [args.out]
    if not cv:
        df = df[df.split == 'test'].reset_index(drop=True)
    for col, f in (('T', 'T/scores_test.csv'), ('G', 'G/scores_test.csv'), ('lpfcn', 'LPFCN/scores_test.csv')):
        s = pd.concat([pd.read_csv(osp.join(d, f)) for d in dirs])[['file_key', 'score']]
        assert s.file_key.is_unique
        df = df.merge(s.rename(columns={'file_key': 'token', 'score': col}), on='token', how='left')
        print(f'{col}: {int(df[col].isna().sum())} of {len(df)} test samples unscored -> lowest score')
        df[col] = df[col].fillna(-1.0)
    methods = {m: c for m, c in METHODS.items() if df[c].notna().all()}
    for m in set(METHODS) - set(methods):
        print(f'{m}: missing scores, left out')
    df.to_csv(osp.join(args.out, 'test_scores.csv'), index=False)

    y = df.label.values.astype(int)
    print(f'test: {len(df)} samples, {y.sum()} positive ({100 * y.mean():.1f} %), '
          f'{df.combo.nunique()} scenario-sequences, {df.scene.nunique()} scenes')
    point = pd.DataFrame({c: metrics(df[c], y) for c in methods.values()}).T[KEYS]
    if cv:
        folds = pd.DataFrame([dict(fold=k, method=m,
                                   **metrics(df[df.fold == k][c], df[df.fold == k].label.values)) for k in sorted(df.fold.unique())
                              for m, c in methods.items()])
        folds.to_csv(osp.join(args.out, 'folds.csv'), index=False)
        agg = folds.groupby('method', sort=False)[KEYS].agg(['mean', 'std'])
        print('per fold (mean +- std over folds):')
        print(pd.DataFrame({k: agg[k]['mean'].round(1).astype(str) + ' +- ' + agg[k]['std'].round(1).astype(str)
                            for k in KEYS}).to_string())

    # scene bootstrap (bootstrap_compare.compare with group = NCAP scene); resamples with one class are redrawn
    S = {c: df[c].values.astype(float) for c in methods.values()}
    pairs = [(a, b) for a, b in PAIRS if a in S and b in S]
    groups = [np.flatnonzero(df.scene.values == sc) for sc in df.scene.unique()]
    rng = np.random.default_rng(0)
    boot, redrawn = [], 0
    while len(boot) < args.n_boot:
        ii = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        if y[ii].min() == y[ii].max():
            redrawn += 1
            continue
        st = bootstrap_compare._stats({c: v[ii] for c, v in S.items()}, y[ii])
        for a, b in pairs:
            for k in KEYS:
                st[f'{a} - {b}|{k}'] = st[f'{a}|{k}'] - st[f'{b}|{k}']
        boot.append(st)
    boot = pd.DataFrame(boot)
    ci = lambda k: np.percentile(boot[k], [2.5, 97.5])
    rows = []
    for m, c in methods.items():
        for k in KEYS:
            lo, hi = ci(f'{c}|{k}')
            rows.append(dict(method=m, metric=k, value=point.loc[c, k], lo=lo, hi=hi))
    for a, b in pairs:
        for k in KEYS:
            d = boot[f'{a} - {b}|{k}']
            lo, hi = ci(f'{a} - {b}|{k}')
            # two-sided bootstrap p-value of the paired difference: twice the smaller tail mass beyond 0
            p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
            rows.append(dict(method=f'{a} - {b}', metric=k, value=point.loc[a, k] - point.loc[b, k], lo=lo, hi=hi, p=p))
    res = pd.DataFrame(rows)
    res.to_csv(osp.join(args.out, 'table2.csv'), index=False)
    print(f'scene bootstrap: {args.n_boot} resamples of {len(groups)} scenes (seed 0), {redrawn} one-class resamples redrawn')
    pd.set_option('display.width', 250)
    print(point.rename(index={c: m for m, c in methods.items()}).round(2).to_string())
    txt = res.assign(txt=res.apply(lambda r: f'{r.value:6.2f} [{r.lo:6.2f}, {r.hi:6.2f}]', axis=1))
    print(txt.pivot(index='method', columns='metric', values='txt')[KEYS].to_string())
    d = res.dropna(subset=['p'])
    print('paired differences, two-sided scene-bootstrap p-values:')
    print(d.pivot(index='method', columns='metric', values='p')[KEYS].to_string(float_format=lambda v: f'{v:.4f}'))


if __name__ == '__main__':
    main()
