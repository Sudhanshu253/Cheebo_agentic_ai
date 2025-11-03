# src/prompts.py
import json

# Short, strict instructions that match your gold_example schema exactly.
SCHEMA_INSTRUCTIONS = (
    "Return EXACTLY one JSON object with these keys: "
    "topic, summary, key_points, formulas, important_questions, solved_examples.\n\n"
    "Constraints:\n"
    "- summary: 3-6 concise sentences.\n"
    "- key_points: list of short strings (6-10 preferred).\n"
    "- formulas: list of objects {\"latex\",\"name\",\"meaning\",\"units\"} (can be []).\n"
    "- important_questions: list of objects {\"q\",\"why_important\",\"difficulty\"}.\n"
    "- solved_examples: list of objects {\"question\",\"solution_steps\",\"final_answer\"}.\n\n"
    "MUSTs:\n"
    "1) Output JSON ONLY. No explanation, no commentary, no extra text.\n"
    "2) The JSON MUST begin immediately after the marker <<<BEGIN_JSON>>> on a new line with '{'.\n"
    "3) Use double quotes for all keys and string values (valid JSON).\n"
)

# A compact example the model can imitate (few-shot)
EXAMPLE = {
    "topic": "EXAMPLE_TOPIC",
    "summary": "A short 3-sentence summary showing the style and concision required.",
    "key_points": ["Point A", "Point B", "Point C"],
    "formulas": [
        {"latex": "V=IR", "name": "Ohm's law", "meaning": "Relates voltage, current and resistance", "units": "V,A,Ohm"}
    ],
    "important_questions": [
        {"q": "State Ohm's law.", "why_important": "Fundamental relation used in circuits", "difficulty": "easy"}
    ],
    "solved_examples": [
        {"question": "Find I when V=10V and R=2Ω", "solution_steps": ["I = V/R", "I = 10/2 = 5 A"], "final_answer": "5 A"}
    ]
}
EXAMPLE_TEXT = json.dumps(EXAMPLE, ensure_ascii=False)

def build_prompt(topic: str, context: str) -> str:
    """
    Produce a strict prompt for inference. Keep context reasonably short before passing to model.
    The function returns the prompt string — feed it into the tokenizer exactly as-is.
    """
    prompt = (
        "You are Cheebo — a concise study-guide generator.\n\n"
        "Context (source notes):\n"
        f"{context}\n\n"
        f"{SCHEMA_INSTRUCTIONS}\n"
        "Example of the exact JSON format to output (copy structure):\n\n"
        f"{EXAMPLE_TEXT}\n\n"
        "Now produce the study-guide for the requested topic.\n"
        "You MUST output JSON ONLY. Begin the JSON object immediately after the marker below.\n"
        "DO NOT write anything before the marker.\n\n"
        "<<<BEGIN_JSON>>>\n"
        "{\n"
        '  "topic": "",\n'
        '  "summary": "",\n'
        '  "key_points": [],\n'
        '  "formulas": [],\n'
        '  "important_questions": [],\n'
        '  "solved_examples": []\n'
        "}\n"
    )
    return prompt
