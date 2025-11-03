import json, glob, os
print("GOLD (data/gold_examples):")
for p in sorted(glob.glob("data/gold_examples/*.json")):
    j=json.load(open(p,'r',encoding='utf-8'))
    name=os.path.basename(p).replace(".json","")
    s=j.get("summary","")
    print(f"  {name}  summary_len={len(s)}  keys={list(j.keys())[:8]}")

print("\nPRED (outputs):")
for p in sorted(glob.glob("outputs/*.json")):
    j=json.load(open(p,'r',encoding='utf-8'))
    name=os.path.basename(p).replace(".json","")
    s=j.get("summary","")
    print(f"  {name}  summary_len={len(s)}  keys={list(j.keys())[:8]}")

# show intersection that your eval will use
gold_keys = {os.path.basename(p).replace('.json','') for p in glob.glob("data/gold_examples/*.json")}
pred_keys = {os.path.basename(p).replace('.json','') for p in glob.glob("outputs/*.json")}
print("\nGold keys:", gold_keys)
print("Pred keys:", pred_keys)
print("Intersection:", gold_keys & pred_keys)
