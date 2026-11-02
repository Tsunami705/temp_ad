'''LP-FCN scores on a list, as rebuttal2/lpfcn.py export (UQNet members, mean of the sigmoid outputs, every sample).

    python score_lpfcn.py --config <out>/configs/LPFCN_m1.json --ckpt-dirs <member dirs> --list <out>/lists/test.txt \
        --epoch 20 --out scores.csv
'''
import argparse
import json
import os.path as osp
import sys

import numpy as np
import pandas as pd
import torch

RM = '/nobackup/proj/disk/cvls/personal/shipengl/riskmonitor_g'
sys.path.insert(0, RM)
import data_loader.data_loaders as module_data  # noqa: E402
import model.model as module_arch  # noqa: E402


def score(c, ckpt_dirs, list_file, epoch):
    '''c: a member's training config (arch, keys, cache_dir); ckpt_dirs: one checkpoint dir per member'''
    label_key = c['Trainer']['label_key']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = []
    for d in ckpt_dirs:
        m = module_arch.UQNet(**c['arch']['args'])
        m.load_state_dict(torch.load(osp.join(d, f'checkpoint-epoch{epoch}.pth'), map_location='cpu')['state_dict'])
        models.append(m.to(device).eval())
    ds = module_data.CachedBEVLoader(batch_size=256, shuffle=False, num_workers=0, data_dir=list_file, mode='val',
                                     cache_dir=c['train_loader']['args']['cache_dir'], label_key=label_key).dataset
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False, drop_last=False)
    rows = []
    with torch.no_grad():
        for d in loader:
            s = torch.stack([m(d[c['Trainer']['q_key']].to(device)).reshape(-1) for m in models], 1)
            s = s.cpu().numpy().astype(np.float64).mean(1)
            for lp, y, sc in zip(d['label_path'], d[label_key].reshape(-1).numpy(), s):
                rows.append({'file_key': lp, 'label': int(y > 0), 'score': sc})
    df = pd.DataFrame(rows)
    assert len(df) == len(open(list_file).read().split())
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--ckpt-dirs', nargs='+', required=True)
    p.add_argument('--list', required=True)
    p.add_argument('--epoch', type=int, default=20)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    df = score(json.load(open(args.config)), args.ckpt_dirs, args.list, args.epoch)
    df.to_csv(args.out, index=False)
    print(f'{len(df)} samples -> {args.out}')


if __name__ == '__main__':
    main()
