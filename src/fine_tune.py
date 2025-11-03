# src/fine_tune.py
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
import torch

# -----------------------
# CONFIG - adjust these
# -----------------------
MODEL_REF = "gpt2"          # lightweight example. Change if you prefer another model.
OUT_DIR = "models/lora"
JSONL_PATH = "data/gold_examples/train.jsonl"
MAX_LENGTH = 1024
# -----------------------

def load_jsonl(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Create your train.jsonl first.")
    return load_dataset("json", data_files=path)["train"]

def preprocess(dataset, tokenizer, max_length=MAX_LENGTH):
    """
    expects examples with keys: 'instruction','input','output'
    returns tokenized dict with 'input_ids', 'attention_mask', 'labels'
    """
    def fn(examples):
        inputs = []
        for instr, inp, out in zip(
            examples.get("instruction", []),
            examples.get("input", []),
            examples.get("output", []),
        ):
            # Build the textual example seen by the model during training.
            # Keep formatting consistent with your inference prompt style.
            text = instr.strip() + "\n" + inp.strip() + "\n\n" + out.strip()
            inputs.append(text)

        # Tokenize the batch
        tok = tokenizer(
            inputs,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

        # For causal LM, labels == input_ids but mask pad tokens with -100
        labels = []
        for ids in tok["input_ids"]:
            lab = ids.copy()
            # mask padding tokens
            lab = [(-100 if (token == tokenizer.pad_token_id) else token) for token in lab]
            labels.append(lab)

        tok["labels"] = labels
        return tok

    tokenized = dataset.map(fn, batched=True, remove_columns=dataset.column_names)
    return tokenized

def train(jsonl_path=JSONL_PATH, out_dir=OUT_DIR):
    print("Generating train split from:", jsonl_path)
    ds = load_jsonl(jsonl_path)

    print("Loading tokenizer and base model:", MODEL_REF)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REF, use_fast=True)
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    # Set padding side to right for causal LM
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_REF)
    # resize embeddings if tokenizer changed
    base_model.resize_token_embeddings(len(tokenizer))

    # LoRA config
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn", "c_proj"] if "gpt2" in MODEL_REF else None,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, peft_config)

    print("Tokenizing dataset...")
    tokenized = preprocess(ds, tokenizer, max_length=MAX_LENGTH)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        logging_steps=10,
        save_strategy="no",            # small dataset — avoid many checkpoint files
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model to", out_dir)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print("Done.")

if __name__ == "__main__":
    train()
