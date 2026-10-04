#!/usr/bin/env python3
"""Season s17: LoRA fine-tune Qwen3-4B on JEV-labeled native Malay text.

Plain transformers + peft + bnb-4bit. Train split only; labels are the real
JEV choices; labels masked to the answer word; merged save is impossible on
bnb-4bit, so the LoRA adapter is saved for vLLM --enable-lora serving.

Usage (from the unsloth venv):
  python finetune_qwen_s17.py --data data/work/s17_labeled.jsonl \
    --out runs/qwen_s17_adapter
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("finetune_qwen_s17")

SYSTEM = ("Classify the sentiment of the user's message for a support triage "
          "system. Reply with exactly one word: positive, neutral, or "
          "negative. No other text.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--system", default=SYSTEM)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines()
            if line.strip()]
    train = [{"messages": [{"role": "system", "content": args.system},
                           {"role": "user", "content": r["text"]},
                           {"role": "assistant", "content": r["choice"]}]}
             for r in rows if r.get("choice")]
    logger.info("training rows: %d", len(train))

    bnb = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = tokenizer = None
    for attempt in range(3):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen3-4B-Instruct-2507", quantization_config=bnb,
                device_map={"": 0}, torch_dtype=torch.bfloat16)
            break
        except (ValueError, RuntimeError, OSError) as err:
            logger.warning("load attempt %d failed: %s", attempt + 1, err)
            time.sleep(90)
    if model is None:
        raise SystemExit("FATAL: model load failed after 3 attempts (shared GPU race)")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def encode(ex: dict) -> dict:
        prompt = tokenizer.apply_chat_template(ex["messages"][:2],
                                               tokenize=False,
                                               add_generation_prompt=True)
        full = tokenizer.apply_chat_template(ex["messages"], tokenize=False)
        p_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        f_ids = tokenizer(full, add_special_tokens=False).input_ids
        f_ids = f_ids + [tokenizer.eos_token_id]
        labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
        return {"input_ids": f_ids[:512], "labels": labels[:512]}

    dataset = Dataset.from_list(train).map(encode, remove_columns=["messages"])
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100)

    train_args = TrainingArguments(
        output_dir="runs/s17_ckpt", num_train_epochs=args.epochs,
        per_device_train_batch_size=8, gradient_accumulation_steps=2,
        learning_rate=2e-4, warmup_ratio=0.03, lr_scheduler_type="cosine",
        logging_steps=20, seed=args.seed, bf16=True, report_to=[],
        save_strategy="no", optim="paged_adamw_8bit")
    trainer = Trainer(model=model, args=train_args, train_dataset=dataset,
                      data_collator=collator)
    trainer.train()

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    logger.info("adapter saved -> %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
