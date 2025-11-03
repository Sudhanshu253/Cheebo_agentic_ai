# scripts/generate_from_gold.py
import json
import glob
import os
from src.inference import generate_study_guide

OUT_DIR = "outputs"
GOLD_DIR = "data/gold_examples"
TOP_K = 6

os.makedirs(OUT_DIR, exist_ok=True)

gold_files = glob.glob(os.path.join(GOLD_DIR, "*.json"))
if not gold_files:
    raise SystemExit(f"No gold examples found in {GOLD_DIR}")

print(f"Found {len(gold_files)} gold examples.\n")

success, failed = 0, 0

for gf in gold_files:
    try:
        g = json.load(open(gf, "r", encoding="utf-8"))
        topic = g.get("topic") or os.path.splitext(os.path.basename(gf))[0]
        safe_name = topic.replace(" ", "_").replace("/", "_").replace(":", "_")
        print(f"🔹 Generating for topic: {topic}")

        # Call your RAG + generation pipeline
        out = generate_study_guide(topic, top_k=TOP_K, save=True)

        # Explicitly save (some inference functions only print)
        out_path = os.path.join(OUT_DIR, f"{safe_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        print(f"✅ Saved: {out_path}\n")
        success += 1

    except Exception as e:
        print(f"❌ Failed for {gf}: {e}\n")
        failed += 1

print(f"\n=== SUMMARY ===")
print(f"Total gold examples: {len(gold_files)}")
print(f"✅ Successful generations: {success}")
print(f"❌ Failed: {failed}")
print(f"Outputs saved in: {os.path.abspath(OUT_DIR)}")
