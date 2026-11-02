'''Epoch selection on val and test scores of the trained methods of one split (ncap_to_cache.py layout).

    python score_cv.py --out <out dir or out/f<k>>

For RiskMonitor-T / -G (rebuttal2/a1_test_export.py, which skips samples without agents) and LP-FCN
(score_lpfcn.py): the ensemble of the members' checkpoints at each saved epoch scores val, the epoch with the
highest val AUROC (the earliest on ties) scores test. Writes <run>/epoch_selection.csv and <run>/scores_test.csv.
'''
import argparse
import glob
import json
import os
import os.path as osp
import sys
from argparse import Namespace

import pandas as pd
from sklearn.metrics import roc_auc_score

RM = '/nobackup/proj/disk/cvls/personal/shipengl/riskmonitor_g'
sys.path.insert(0, RM)
sys.path.insert(0, osp.join(RM, 'rebuttal2'))
import a1_test_export  # noqa: E402

from score_lpfcn import score as score_lpfcn  # noqa: E402

CACHE = {'T': 'cache_base', 'G': 'cache_c2full'}


def member_dirs(out, run):
    '''latest checkpoint dir of each member'''
    dirs = []
    for m in sorted(glob.glob(osp.join(out, run, 'm*'))):
        dirs.append(sorted(glob.glob(osp.join(m, 'saved', 'models', '*', '*/')))[-1])
    return dirs


def scores(out, run, dirs, split, epoch):
    if run == 'LPFCN':
        cfg = json.load(open(osp.join(out, 'configs', 'LPFCN_m1.json')))
        return score_lpfcn(cfg, dirs, osp.join(out, 'lists', f'{split}.txt'), epoch)
    path = osp.join(out, run, 'sel', f'{split}_e{epoch}.csv')
    a1_test_export.main(Namespace(config=osp.join(out, 'configs', f'{run}_{split}.json'), ckpt_dirs=dirs,
                                  epoch=epoch, cache_dir=osp.join(out, CACHE[run]), out=path))
    return pd.read_csv(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--runs', nargs='+', default=['T', 'G', 'LPFCN'])
    args = p.parse_args()
    for run in args.runs:
        dirs = member_dirs(args.out, run)
        epochs = sorted(int(f.rsplit('epoch', 1)[1][:-4]) for f in glob.glob(osp.join(dirs[0], 'checkpoint-epoch*.pth')))
        os.makedirs(osp.join(args.out, run, 'sel'), exist_ok=True)
        sel = []
        for e in epochs:
            v = scores(args.out, run, dirs, 'val', e)
            sel.append(dict(epoch=e, n=len(v), val_auroc=100 * roc_auc_score(v.label, v.score)))
        sel = pd.DataFrame(sel)
        best = int(sel.epoch[sel.val_auroc.idxmax()])
        sel['selected'] = sel.epoch == best
        sel.to_csv(osp.join(args.out, run, 'epoch_selection.csv'), index=False)
        scores(args.out, run, dirs, 'test', best).to_csv(osp.join(args.out, run, 'scores_test.csv'), index=False)
        print(f'{run}: {len(dirs)} members, val AUROC per epoch '
              + ' '.join(f'{e}:{a:.1f}' for e, a in zip(sel.epoch, sel.val_auroc)) + f' -> epoch {best}', flush=True)


if __name__ == '__main__':
    main()
