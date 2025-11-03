# scripts/split_train_to_gold.py
import json, os, re
from pathlib import Path

def slug(s):
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "topic"

os.makedirs("data/gold_examples", exist_ok=True)
lines = open("data/gold_examples/train.jsonl", "r", encoding="utf-8").read().strip().splitlines()
for i, line in enumerate(lines, 1):
    try:
        obj = json.loads(line)
        out_str = obj.get("output", "")
        # output field may be a JSON string; parse if necessary
        if isinstance(out_str, str):
            out_json = json.loads(out_str)
        else:
            out_json = out_str
        topic = out_json.get("topic", f"topic_{i}")
        filename = f"data/gold_examples/{slug(topic)}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(out_json, f, ensure_ascii=False, indent=2)
        print("Wrote", filename)
    except Exception as e:
        print("Error on line", i, e)
