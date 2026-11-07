#!/usr/bin/env python
"""Offline analysis of results/rc_qa_<tag> (scripts/diag_rc_qa.py) joined with the QA benchmark F1 of base (main repo
results/fused_<tag>_main) and rc (results/rc_<tag>_main), ours 10%: by answer coverage, and the shape of the KL change."""
import json, statistics as S, collections, random
W='/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-moe/results'; M='/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/results'
pairs=[('06','fused_06_main'),('17','fused_17_main'),('ll1to8','fused_ll1to8_main'),('ll1toq8','ll1toq8_main'),('q17toll8','fused_q17toll8_main')]
def load(p,cfg):
    d={}
    for l in open(p):
        r=json.loads(l)
        if r['task']=='qa' and r['config']==cfg: d[(r['source'],r['bucket'],r['idx'])]=r
    return d
def boot(x,n=2000):
    rng=random.Random(0); m=[S.mean(rng.choices(x,k=len(x))) for _ in range(n)]; m.sort(); return m[int(.025*n)],m[int(.975*n)]
K=['base@0','rc@0','base@0.1','rc@0.1']
for tag,base in pairs:
    rows=[json.loads(l) for l in open(f'{W}/rc_qa_{tag}/per_example.jsonl')]
    b=load(f'{M}/{base}/generations.jsonl','attn_small:0.1:sent'); n=load(f'{W}/rc_{tag}_main/generations.jsonl','attn_small:0.1:sent')
    print(f'===== {tag}')
    for bk in ('4k','8k'):
        rs=[r for r in rows if r['bucket']==bk]
        print(f' {bk} n={len(rs)} KL  '+'  '.join(f"{k}={S.mean(r['kl'][k] for r in rs):.3f}" for k in K)+' | kl_first '+' '.join(f"{k}={S.mean(r['kl_first'][k] for r in rs):.3f}" for k in K))
        for cov in ('covered','missed','absent'):
            c=[r for r in rs if r['coverage']==cov]
            if not c: continue
            ks=[(r['source'],r['bucket'],r['i']) for r in c]
            ks=[k for k in ks if k in b and k in n]
            d=[n[k]['f1']-b[k]['f1'] for k in ks]
            dkl=[r['kl']['rc@0.1']-r['kl']['base@0.1'] for r in c]
            lo,hi=boot(d) if d else (0,0)
            print(f'   {cov:8} n={len(c):3}  KL@.1 base {S.mean(r["kl"]["base@0.1"] for r in c):.3f} rc {S.mean(r["kl"]["rc@0.1"] for r in c):.3f} (rc<base {S.mean(x<0 for x in dkl):.2f}) | @0 base {S.mean(r["kl"]["base@0"] for r in c):.3f} rc {S.mean(r["kl"]["rc@0"] for r in c):.3f}'
                  f' | F1 base {S.mean(b[k]["f1"] for k in ks):.3f} rc {S.mean(n[k]["f1"] for k in ks):.3f} d={S.mean(d):+.3f} [{lo:+.3f},{hi:+.3f}]')

print("\n# distribution of the KL change at 10%, top-1, F1 changes")
for tag,base in pairs:
    rows=[json.loads(l) for l in open(f'{W}/rc_qa_{tag}/per_example.jsonl')]
    b=load(f'{M}/{base}/generations.jsonl','attn_small:0.1:sent'); n=load(f'{W}/rc_{tag}_main/generations.jsonl','attn_small:0.1:sent')
    for bk in ('4k','8k'):
        rs=[r for r in rows if r['bucket']==bk]
        t1=lambda k:S.mean(r['top1'][k] for r in rs)
        d=[r['kl']['rc@0.1']-r['kl']['base@0.1'] for r in rs]
        med=S.median(d); better=S.mean(x< -0.01 for x in d); worse=S.mean(x>0.01 for x in d)
        ks=[(r['source'],r['bucket'],r['i']) for r in rs]
        df=[n[k]['f1']-b[k]['f1'] for k in ks]
        chg=[(x,y) for x,y in zip(d,df) if y!=0]
        # F1 changes: when rc F1 worse, mean KL delta
        fw=[x for x,y in zip(d,df) if y<0]; fb=[x for x,y in zip(d,df) if y>0]
        # sum of top-5 KL improvements share
        imp=sorted(d)[:10]
        print(f"{tag:9}{bk} top1@.1 base {t1('base@0.1'):.3f} rc {t1('rc@0.1'):.3f} @0 {t1('base@0'):.3f}/{t1('rc@0'):.3f} | dKL mean {S.mean(d):+.3f} median {med:+.4f} better {better:.2f} worse {worse:.2f} | top10 gains sum {sum(imp):+.2f} of total {sum(d):+.2f}"
              f" | F1 worse n={len(fw)} (dKL {S.mean(fw) if fw else 0:+.3f}) better n={len(fb)} (dKL {S.mean(fb) if fb else 0:+.3f})")
