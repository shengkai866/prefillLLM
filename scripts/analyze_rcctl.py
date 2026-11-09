#!/usr/bin/env python
"""0.6B->8B: base vs control (rcctl_06: fine-tune without recompute) vs rc_06, benchmark paired vs base."""
import json, random, statistics as S
W = '/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-moe/results'
M = '/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/results'


def boot(d, n=4000):
    rng = random.Random(0)
    bs = sorted(S.mean(rng.choices(d, k=len(d))) for _ in range(n))
    return bs[int(.025 * n)], bs[int(.975 * n)]


def gens(p, cfg):
    d = {}
    for l in open(p):
        r = json.loads(l)
        if r['config'] != cfg:
            continue
        if r['task'] == 'qa':
            d[('qa', r['bucket'], r['source'], r['idx'])] = r['f1']
        elif r['task'] == 'ruler' and not str(r.get('sub', '')).startswith('_'):
            d[('ruler', f"{r['length'] // 1024}k", r['sub'], r['idx'])] = r['score']
    return d


def summ(r, c):
    x = [e['accuracy'] for e in r.get('gsm8k', []) if e['config'] == c]
    y = [e['exact_match'] for e in r.get('code', []) if e['config'] == c]
    z = [e['accuracy'] for e in r.get('niah', []) if e['config'] == c]
    return (x[0] if x else float('nan'), S.mean(y) if y else float('nan'), S.mean(z) if z else float('nan'))


runs = {'base': f'{M}/fused_06_main', 'ctl': f'{W}/rcctl_06_main', 'rc': f'{W}/rc_06_main'}
cov = {r['i']: r['coverage'] for r in map(json.loads, open(f'{W}/rc_qa_06/per_example.jsonl'))}
for c in ('hyb', 'attn_small:0.1:sent'):
    g = {k: gens(f'{v}/generations.jsonl', c) for k, v in runs.items()}
    print(f'=== {c}')
    for task, b in (('qa', '4k'), ('qa', '8k'), ('ruler', '4k'), ('ruler', '8k')):
        ks = [k for k in g['ctl'] if k[0] == task and k[1] == b and k in g['base'] and k in g['rc']]
        out = f' {task}{b} n={len(ks)} ' + ' '.join(f'{t} {S.mean(g[t][k] for k in ks):.3f}' for t in runs)
        for t in ('ctl', 'rc'):
            d = [g[t][k] - g['base'][k] for k in ks]; lo, hi = boot(d)
            out += f' | {t}-base {S.mean(d):+.3f} [{lo:+.3f},{hi:+.3f}]'
        d = [g['rc'][k] - g['ctl'][k] for k in ks]; lo, hi = boot(d)
        print(out + f' | rc-ctl {S.mean(d):+.3f} [{lo:+.3f},{hi:+.3f}]')
        if task == 'qa' and c != 'hyb':
            for cv in ('covered', 'missed'):
                kk = [k for k in ks if cov.get(k[3]) == cv]
                print(f'    {cv:8} n={len(kk)} ' + ' '.join(f'{t} {S.mean(g[t][k] for k in kk):.3f}' for t in runs))
    print(' GSM/code/niah ' + ' | '.join(f'{t} ' + '/'.join(f'{x:.3f}' for x in summ(json.load(open(f'{v}/results.json')), c)) for t, v in runs.items()))
