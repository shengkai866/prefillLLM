#!/usr/bin/env python
"""base vs rc vs rcn (imperfect training selection) for the pairs of slurm/run_rcn.slurm: sd_mixtest KL (mean / median /
top-1), QA-prompt diagnostic by answer coverage, and the benchmark (ours 10%) with paired bootstrap vs base."""
import json, random, statistics as S
W = '/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-moe/results'
M = '/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/results'
pairs = [('ll1to8', 'fused_ll1to8_main'), ('ll1toq8', 'll1toq8_main'), ('17', 'fused_17_main')]
C = 'attn_small:0.1:sent'


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
    o = {}
    x = [e['accuracy'] for e in r.get('gsm8k', []) if e['config'] == c]
    o['GSM'] = x[0] if x else float('nan')
    x = [e['exact_match'] for e in r.get('code', []) if e['config'] == c]
    o['code'] = S.mean(x) if x else float('nan')
    return o


for tag, base in pairs:
    print(f'\n######## {tag}')
    rows = [json.loads(l) for l in open(f'{W}/rcn_kl_{tag}/per_example.jsonl')]
    for b in ('4k', '8k'):
        rs = [r for r in rows if r['bucket'] == b]
        line = []
        for k in ('base@0', 'rc@0', 'rcn@0', 'base@0.1', 'rc@0.1', 'rcn@0.1'):
            line.append(f"{k} {S.mean(r['kl'][k] for r in rs):.4f}/{S.median(r['kl'][k] for r in rs):.4f}/{S.mean(r['agree'][k] for r in rs):.3f}")
        print(f' sdKL {b} mean/median/top1: ' + ' | '.join(line))
        qa = [r for r in rs if r['type'] == 'qa']
        print(f'   qa type @0.1: base {S.mean(r["kl"]["base@0.1"] for r in qa):.3f} rc {S.mean(r["kl"]["rc@0.1"] for r in qa):.3f} rcn {S.mean(r["kl"]["rcn@0.1"] for r in qa):.3f}')
    # QA diagnostic: rc file (base vs rc) and rcn file (base vs rcn), same prompts
    rc = {r['i']: r for r in map(json.loads, open(f'{W}/rc_qa_{tag}/per_example.jsonl'))}
    rn = {r['i']: r for r in map(json.loads, open(f'{W}/rc_qa_rcn_{tag}/per_example.jsonl'))}
    gb, gr, gn = gens(f'{M}/{base}/generations.jsonl', C), gens(f'{W}/rc_{tag}_main/generations.jsonl', C), gens(f'{W}/rcn_{tag}_main/generations.jsonl', C)
    for b in ('4k', '8k'):
        ii = [i for i in rn if rn[i]['bucket'] == b and i in rc]
        for kk in ('0', '0.1'):
            vb = [rn[i]['kl'][f'base@{kk}'] for i in ii]; vr = [rc[i]['kl'][f'rc@{kk}'] for i in ii]; vn = [rn[i]['kl'][f'rc@{kk}'] for i in ii]
            tb = S.mean(rn[i]['top1'][f'base@{kk}'] for i in ii); tr = S.mean(rc[i]['top1'][f'rc@{kk}'] for i in ii); tn = S.mean(rn[i]['top1'][f'rc@{kk}'] for i in ii)
            print(f' QAdiag {b} @{kk}: KL mean base {S.mean(vb):.3f} rc {S.mean(vr):.3f} rcn {S.mean(vn):.3f} | dKL median rc {S.median([x - y for x, y in zip(vr, vb)]):+.4f} rcn {S.median([x - y for x, y in zip(vn, vb)]):+.4f}'
                  f' | better/worse rc {S.mean(x - y < -.01 for x, y in zip(vr, vb)):.2f}/{S.mean(x - y > .01 for x, y in zip(vr, vb)):.2f} rcn {S.mean(x - y < -.01 for x, y in zip(vn, vb)):.2f}/{S.mean(x - y > .01 for x, y in zip(vn, vb)):.2f}'
                  f' | top1 base {tb:.3f} rc {tr:.3f} rcn {tn:.3f}')
        for cov in ('covered', 'missed', 'absent'):
            c = [i for i in ii if rn[i]['coverage'] == cov]
            ks = [('qa', b, rn[i]['source'], i) for i in c]
            ks = [k for k in ks if k in gb and k in gr and k in gn]
            dr = [gr[k] - gb[k] for k in ks]; dn = [gn[k] - gb[k] for k in ks]
            lr, hr = boot(dr); ln, hn = boot(dn)
            print(f'   {cov:8} n={len(ks):3} F1 base {S.mean(gb[k] for k in ks):.3f} | rc {S.mean(dr):+.3f} [{lr:+.3f},{hr:+.3f}] | rcn {S.mean(dn):+.3f} [{ln:+.3f},{hn:+.3f}]'
                  f' | KL@.1 base {S.mean(rn[i]["kl"]["base@0.1"] for i in c):.3f} rc {S.mean(rc[i]["kl"]["rc@0.1"] for i in c):.3f} rcn {S.mean(rn[i]["kl"]["rc@0.1"] for i in c):.3f}')
    # benchmark totals, paired vs base
    for task, b in (('qa', '4k'), ('qa', '8k'), ('ruler', '4k'), ('ruler', '8k')):
        ks = [k for k in gn if k[0] == task and k[1] == b and k in gb and k in gr]
        dr = [gr[k] - gb[k] for k in ks]; dn = [gn[k] - gb[k] for k in ks]; dnr = [gn[k] - gr[k] for k in ks]
        print(f' bench {task}{b} n={len(ks)} base {S.mean(gb[k] for k in ks):.3f} rc {S.mean(gr[k] for k in ks):.3f} rcn {S.mean(gn[k] for k in ks):.3f}'
              f' | rcn-base {S.mean(dn):+.3f} [{boot(dn)[0]:+.3f},{boot(dn)[1]:+.3f}] | rcn-rc {S.mean(dnr):+.3f} [{boot(dnr)[0]:+.3f},{boot(dnr)[1]:+.3f}]')
    jb, jr, jn = (json.load(open(p)) for p in (f'{M}/{base}/results.json', f'{W}/rc_{tag}_main/results.json', f'{W}/rcn_{tag}_main/results.json'))
    for c in ('hyb', C):
        sb, sr, sn = summ(jb, c), summ(jr, c), summ(jn, c)
        print(f' {c:20} GSM {sb["GSM"]:.3f}/{sr["GSM"]:.3f}/{sn["GSM"]:.3f}  code {sb["code"]:.3f}/{sr["code"]:.3f}/{sn["code"]:.3f}')
