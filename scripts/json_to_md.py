# scripts/json_to_md.py
import json, sys, os

def json_to_md(path):
    data = json.load(open(path, "r", encoding="utf-8"))
    topic = data.get("topic", "Untitled")
    md = [f"# {topic}\n"]

    md.append("## Summary")
    md.append(data.get("summary", "").strip())

    if "key_points" in data:
        md.append("\n## Key Points")
        for k in data["key_points"]:
            md.append(f"- {k}")

    if data.get("formulas"):
        md.append("\n## Formulas")
        for f in data["formulas"]:
            md.append(f"- **{f['name']}** ({f['latex']}) — {f['meaning']} [{f['units']}]")

    if data.get("important_questions"):
        md.append("\n## Important Questions")
        for q in data["important_questions"]:
            md.append(f"- **Q:** {q['q']}\n  - *Why important:* {q['why_important']}  \n  - *Difficulty:* {q['difficulty']}")

    if data.get("solved_examples"):
        md.append("\n## Solved Examples")
        for ex in data["solved_examples"]:
            md.append(f"### {ex['question']}")
            steps = ex.get("solution_steps", [])
            if steps:
                md.extend([f"1. {s}" for s in steps])
            md.append(f"**Final Answer:** {ex.get('final_answer', '')}\n")

    return "\n".join(md)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/json_to_md.py <path_to_json>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        sys.exit(f"File not found: {path}")

    md = json_to_md(path)
    out = os.path.splitext(path)[0] + ".md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"✅ Markdown saved to: {out}")
