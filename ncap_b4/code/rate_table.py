'''Markdown table of the collision rates per scene-sequence pair from check_ncap_labels.py output.

    python3 rate_table.py <steps.csv> [--deltas 0.0 0.5 1.0] [--exec]

Columns: runs, run collision rate (NCAP any_collide), and per delta the share of auto steps whose plan before the
collision optimisation collides within 6 / 4 steps (CollisionLoss > 0); --exec adds the executed plan (6 steps).
Standard library only (runs on the login node).
'''
import argparse
import csv
from collections import defaultdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument('steps')
    p.add_argument('--deltas', nargs='+', default=['0.0', '0.5', '1.0'])
    p.add_argument('--exec', action='store_true')
    args = p.parse_args()
    rows = [r for r in csv.DictReader(open(args.steps)) if r['phase'] == 'auto']
    cols = [(f'{n} 步 δ={d}', f'col_n{n}_d{d}') for d in args.deltas for n in (6, 4)]
    if args.exec:
        cols += [(f'执行轨迹 6 步 δ={d}', f'colexec_n6_d{d}') for d in args.deltas]
    groups = defaultdict(list)
    for r in rows:
        typ = r['combo'].split('-')[0]
        for g in (r['combo'], f'**{typ} 合计**', '**全部**'):
            groups[g].append(r)
    order = sorted(k for k in groups if not k.startswith('**')) + \
        [f'**{t} 合计**' for t in ('frontal', 'side', 'stationary') if f'**{t} 合计**' in groups] + ['**全部**']
    print('| 组合 | run 数 | run 碰撞率 | ' + ' | '.join(c for c, _ in cols) + ' |')
    print('|---' * (3 + len(cols)) + '|')
    for g in order:
        rs = groups[g]
        runs = {(r['combo'], r['run']): int(r['run_collided']) for r in rs}
        cells = []
        for _, key in cols:
            v = [float(r[key]) > 0 for r in rs if r.get(key, '') != '']
            cells.append(f'{100 * sum(v) / len(v):.1f}%' if v else '—')
        print(f'| {g} | {len(runs)} | {100 * sum(runs.values()) / len(runs):.0f}% | ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    main()
