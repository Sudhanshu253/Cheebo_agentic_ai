# src/inference.py
import json
import os
import re
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from src.indexer import query_index
from src.prompts import build_prompt

# CONFIG
BASE_MODEL = "gpt2"          # must match model used during fine-tuning
LORA_DIR = "models/lora"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Keep max tokens moderate to avoid OOM or positional errors
MAX_NEW_TOKENS = 200

# cache
_CACHED = {"tokenizer": None, "model": None, "model_max_pos": None}


def load_model(base_model: str = BASE_MODEL, lora_dir: str = LORA_DIR):
    """
    Load tokenizer (prefer from LORA_DIR to pick up added tokens), load base, resize embeddings
    and attach LoRA adapter. Also compute model's max position embeddings.
    """
    global _CACHED
    if _CACHED["tokenizer"] is not None and _CACHED["model"] is not None:
        return _CACHED["tokenizer"], _CACHED["model"], _CACHED["model_max_pos"]

    # 1) tokenizer (prefer adapter folder so added tokens are present)
    try:
        tokenizer = AutoTokenizer.from_pretrained(lora_dir, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        added_tokens_path = os.path.join(lora_dir, "added_tokens.json")
        if os.path.exists(added_tokens_path):
            try:
                with open(added_tokens_path, "r", encoding="utf-8") as f:
                    added = json.load(f)
                if isinstance(added, dict) and "added_tokens" in added:
                    tokenizer.add_tokens(added["added_tokens"])
            except Exception:
                pass

    # ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # 2) base model
    base = AutoModelForCausalLM.from_pretrained(base_model, low_cpu_mem_usage=False)
    base.to(DEVICE)

    # 3) resize embeddings to tokenizer length (handles added tokens)
    base.resize_token_embeddings(len(tokenizer))

    # 4) attach LoRA adapter
    model = PeftModel.from_pretrained(base, lora_dir)
    model.to(DEVICE)
    model.eval()

    # 5) infer model max positional embeddings
    cfg = getattr(model, "config", base.config)
    # fallback keys often present: n_positions, max_position_embeddings, n_ctx
    model_max_pos = getattr(cfg, "n_positions", None) or getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "n_ctx", 1024)
    model_max_pos = int(model_max_pos)

    _CACHED["tokenizer"] = tokenizer
    _CACHED["model"] = model
    _CACHED["model_max_pos"] = model_max_pos
    return tokenizer, model, model_max_pos


def extract_json_from_text(text: str) -> dict:
    """Try to extract first JSON object; fallback to raw_output."""
    # Balanced braces search
    start_idx = None
    stack = 0
    for i, ch in enumerate(text):
        if ch == "{":
            if start_idx is None:
                start_idx = i
            stack += 1
        elif ch == "}":
            stack -= 1
            if stack == 0 and start_idx is not None:
                candidate = text[start_idx : i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    start_idx = None
                    stack = 0
    # regex fallback
    m = re.search(r"(\{(?:.|\n)*\})", text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    return {"raw_output": text}


def safe_tokenize_truncate(tokenizer, text: str, model_max_pos: int, max_new_tokens: int):
    """
    Tokenize and truncate the input text so that:
        len(input_ids) + max_new_tokens <= model_max_pos
    Returns tokenized dict ready for model.generate.
    """
    # Conservative allowed input length
    allowed_input_len = max(1, model_max_pos - max_new_tokens)
    tok = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=allowed_input_len)
    return tok


def generate_study_guide(topic: str, top_k: int = 5, save: bool = True, model_override: Optional[str] = None):
    """
    RAG + LoRA generation pipeline:
     - retrieve relevant chunks
     - build prompt
     - load model & tokenizer (with added tokens handling)
     - safely tokenize + truncate prompt
     - generate with a safe max_new_tokens
     - extract JSON and save
    """
    # 1) retrieve
    chunks = query_index(topic, k=top_k)
    context = "\n\n----\n\n".join(chunks)

    # 2) prompt
    prompt = build_prompt(topic, context)

    # 3) load model/tokenizer & model position limit
    tokenizer, model, model_max_pos = load_model(base_model=(model_override or BASE_MODEL), lora_dir=LORA_DIR)

    # 4) tokenize safely (truncate to allowed length)
    tokenized = safe_tokenize_truncate(tokenizer, prompt, model_max_pos, MAX_NEW_TOKENS)
    input_ids = tokenized["input_ids"].to(DEVICE)
    attention_mask = tokenized.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(DEVICE)

    # 5) generate (avoid unsupported or ignored args)
    # Make generation deterministic (no sampling)
    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 6) parse JSON and save
    parsed = extract_json_from_text(text)
    os.makedirs("outputs", exist_ok=True)
    outpath = os.path.join("outputs", f"{topic.replace(' ', '_')}.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)

    return parsed


if __name__ == "__main__":
    test_topic = "Short Line Model"
    print(f"Generating study guide for: {test_topic}")
    out = generate_study_guide(test_topic, top_k=6)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Saved to outputs/{test_topic.replace(' ', '_')}.json")
