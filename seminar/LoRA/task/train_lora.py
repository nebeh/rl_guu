"""Обучение LoRA. Цикл Trainer уже собран.

Вам нужно описать сам адаптер и то, какой пример видит модель:
реплика «напиши рассказ» и текст из корпуса. Базовые веса замораживает PEFT,
отдельно это делать не нужно.
"""

from __future__ import annotations

import os

# httpx не принимает схему socks:// (её часто ставит локальный прокси).
# До импорта huggingface её нужно заменить на socks5://.
for _key, _value in list(os.environ.items()):
    if _key.lower().endswith("proxy") and isinstance(_value, str) and _value.startswith("socks://"):
        os.environ[_key] = "socks5://" + _value[len("socks://") :]

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


def make_messages(story: str) -> list[dict]:
    """Соберите диалог, на котором учится адаптер.

    Генерация в generate.py использует тот же вид диалога: одна реплика user,
    без системного промпта. Не добавляйте здесь роль system.
    """
    # TODO: верните список сообщений ролей user и assistant.
    # user просит короткий рассказ, assistant — это аргумент story.
    raise NotImplementedError("соберите обучающий диалог")


def build_lora() -> LoraConfig:
    """Опишите LoRA. Оба автора обучаются с этой одной конфигурацией."""
    # TODO: task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05.
    # target_modules — проекции attention и MLP модели Qwen:
    # q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj.
    raise NotImplementedError("соберите LoraConfig")


def load_stories(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class StoryDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        messages = make_messages(self.rows[index]["text"])
        full = list(
            self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False, return_dict=False
            )
        )[: self.max_length]
        prompt = self.tokenizer.apply_chat_template(
            messages[:1], tokenize=True, add_generation_prompt=True, return_dict=False
        )
        prompt_len = min(len(prompt), len(full))
        label = full.copy()
        # TODO: в label замените токены реплики пользователя на -100,
        # чтобы loss считался только по словам рассказа.
        # Срез уже посчитан: это первые prompt_len позиций.
        return {
            "input_ids": full,
            "labels": label,
            "attention_mask": [1] * len(full),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--max-length", type=int, default=768)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.config.use_cache = False
    model = get_peft_model(model, build_lora())
    model.print_trainable_parameters()

    stories = load_stories(args.data)
    tokenized = StoryDataset(stories, tokenizer, args.max_length)
    collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, label_pad_token_id=-100)
    train_args = TrainingArguments(
        output_dir=str(args.out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
    )
    Trainer(model=model, args=train_args, train_dataset=tokenized, data_collator=collator).train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"адаптер сохранён в {args.out}")


if __name__ == "__main__":
    main()
