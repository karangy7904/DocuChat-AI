from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DocuChat with QLoRA")
    parser.add_argument("--dataset", default="data/training_data.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", default="models/docuchat-lora")
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required. Run this script in Google Colab.")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        ),
        device_map="auto",
        use_cache=False,
    )
    dataset = load_dataset("json", data_files=args.dataset, split="train")

    def format_record(record):
        messages = [
            {"role": "system", "content": "Answer only from the supplied document context."},
            {
                "role": "user",
                "content": f"Context:\n{record['context']}\n\nQuestion:\n{record['instruction']}",
            },
            {"role": "assistant", "content": record["response"]},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    config = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_seq_length=512,
        logging_steps=5,
        save_strategy="epoch",
        fp16=True,
        gradient_checkpointing=True,
        report_to="none",
        seed=42,
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        peft_config=LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
        formatting_func=format_record,
        processing_class=tokenizer,
    )
    trainer.train()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved LoRA adapter to {args.output}")


if __name__ == "__main__":
    main()

