"""Official DAVIS-2017 val J&F evaluation for our AOT(+GC) runs.

Wraps the `davis2017-evaluation` package (region similarity J = Jaccard and
boundary similarity F) so the numbers are directly comparable to the AOT
README table (AOTT/PRE_YTB_DAV: J&F=79.2, J=76.5, F=81.9).
"""
import os
import sys
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'davis2017-evaluation'))

from davis2017.evaluation import DAVISEvaluation  # noqa: E402

DAVIS_ROOT = os.path.join(HERE, 'aot-benchmark', 'datasets', 'DAVIS')
RES_ROOT = os.path.join(HERE, 'aot-benchmark', 'results', 'davis2017')

# label -> exp_name used during inference
RUNS = {
    'noGC':     'nogcfull',
    'legacy20': 'legacy20full',
    'legacy20_k1': 'legacy20k1full',
    'faithful20_k1': 'faithful20k1full',
    'adk1_20':    'adk1_20full',
    'adk1_200':   'adk1_200full',
    'adk1_2k':    'adk1_2000full',
    'adk1_10k':   'adk1_10000full',
    'adk1_30k':   'adk1_30000full',
    'legacy200': 'legacy200full',
    'legacy1k': 'legacy1000full',
    'legacy5k': 'legacy5000full',
    'legacy20k': 'legacy20000full',
    'a2000':    'gc2000full',
    'a5000':    'gc5000full',
    'a10000':   'gc10000full',
    'a30000':   'gc30000full',
    'a50000':   'gc50000full',
    'adapt5k_g1':  'adapt5000g1',
    'adapt10k_g1': 'adapt10000g1',
    'adapt30k_g1': 'adapt30000g1',
    'adapt30k_g2': 'adapt30000g2',
}


def res_path(exp_name):
    return os.path.join(
        RES_ROOT, 'davis2017_val_{}_AOTT_PRE_ckpt_unknown'.format(exp_name),
        'Annotations', '480p')


def eval_one(exp_name):
    rp = res_path(exp_name)
    if not os.path.isdir(rp):
        return None
    ev = DAVISEvaluation(davis_root=DAVIS_ROOT, task='semi-supervised',
                         gt_set='val')
    metrics = ev.evaluate(rp)
    J = metrics['J']['M']
    F = metrics['F']['M']
    Jm, Fm = float(np.mean(J)), float(np.mean(F))
    return {
        'J': Jm, 'F': Fm, 'JF': (Jm + Fm) / 2.0,
        'per_obj_J': metrics['J']['M_per_object'],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', default=list(RUNS.keys()),
                    help='subset of run labels to evaluate')
    args = ap.parse_args()

    results = {}
    for label in args.runs:
        exp = RUNS.get(label, label)
        print('Evaluating %-8s (exp_name=%s) ...' % (label, exp), flush=True)
        r = eval_one(exp)
        if r is None:
            print('  -> MISSING results, skipped')
            continue
        results[label] = r

    if not results:
        print('No runs evaluated.')
        return

    print()
    print('=' * 56)
    print('DAVIS-2017 val  (official J&F, x100)')
    print('%-10s %8s %8s %8s %10s' % ('run', 'J&F', 'J', 'F', 'dJ&F'))
    print('-' * 56)
    base = results.get('noGC', {}).get('JF')
    for label in args.runs:
        if label not in results:
            continue
        r = results[label]
        d = '' if base is None else '%+10.2f' % (100 * (r['JF'] - base))
        print('%-10s %8.2f %8.2f %8.2f %s' % (
            label, 100 * r['JF'], 100 * r['J'], 100 * r['F'], d))
    print('=' * 56)

    # per-object J delta vs noGC for the hardest sequences
    if 'noGC' in results:
        print('\nPer-object J (x100) vs noGC, sorted by noGC ascending:')
        keys = sorted(results['noGC']['per_obj_J'],
                      key=lambda k: results['noGC']['per_obj_J'][k])
        hdr = '%-24s' % 'obj' + ''.join(
            '%9s' % l for l in args.runs if l in results)
        print(hdr)
        for k in keys:
            line = '%-24s' % k
            for l in args.runs:
                if l in results:
                    line += '%9.2f' % (100 * results[l]['per_obj_J'].get(
                        k, float('nan')))
            print(line)


if __name__ == '__main__':
    main()
