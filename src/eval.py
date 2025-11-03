import json, glob, re, os
from evaluate import load
rouge = load("rouge")
def slug(s): return re.sub(r'[^a-z0-9]+','', s.lower())
gold = {slug(os.path.basename(p).replace('.json','')):p for p in glob.glob('data/gold_examples/*.json')}
pred = {slug(os.path.basename(p).replace('.json','')):p for p in glob.glob('outputs_fixed/*.json')}
common = set(gold.keys()) & set(pred.keys())
print("common:", common)
gs=[]; ps=[]
for k in sorted(common):
    g=json.load(open(gold[k],'r',encoding='utf-8')); p=json.load(open(pred[k],'r',encoding='utf-8'))
    gs.append(g.get('summary','')); ps.append(p.get('summary',''))
if gs:
    r = rouge.compute(predictions=ps, references=gs)
    print("ROUGE result:", r)
else:
    print("No summaries")
