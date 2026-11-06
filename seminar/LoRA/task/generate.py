"""Сравнение трёх прогонов одним промптом: база, адаптер Хармса, адаптер Чехова.

База уже загружается в main. Допишите тему и подключение адаптера.
Параметры генерации общие для всех трёх вызовов.
"""

from __future__ import annotations

import os

# httpx не принимает схему socks:// (её часто ставит локальный прокси).
# До импорта huggingface её нужно заменить на socks5://.
for _key, _value in list(os.environ.items()):
    if (
        _key.lower().endswith("proxy")
        and isinstance(_value, str)
        and _value.startswith("socks://")
    ):
        os.environ[_key] = "socks5://" + _value[len("socks://") :]

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-1.5B-Instruct"

# TODO: замените текст на свой бытовой сюжет. Начните фразу с «Напиши рассказ о ».
PROMPT = ""

RUNS = [
    ("Базовая модель", None),
    ("LoRA: Хармс", Path("../output/harms")),
    ("LoRA: Чехов", Path("../output/chekhov")),
]


def attach_adapter(model, adapter_dir: Path):
    """Наденьте сохранённый LoRA на уже загруженную базу.

    Базу заново не загружайте: адаптер должен лечь поверх тех же весов.
    """
    # TODO: PeftModel.from_pretrained(model, adapter_dir)
    raise NotImplementedError("подключите адаптер")


def generate(
    model, tokenizer, prompt: str, temperature: float, num_predict: int
) -> str:
    # Тот же вид диалога, что в train_lora.make_messages: одна реплика user, без system.
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=False
    )
    input_ids = torch.tensor([input_ids], device=model.device)
    attention_mask = torch.ones_like(input_ids)
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=num_predict,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = output[0, input_ids.shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=BASE)
    parser.add_argument("--out", type=Path, default=Path("generations.md"))
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--num-predict", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not PROMPT.startswith("Напиши рассказ о "):
        raise SystemExit("PROMPT должен начинаться с «Напиши рассказ о »")
    if len(RUNS) != 3:
        raise SystemExit("в RUNS нужны ровно три прогона")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    lines = [f"Промпт: {PROMPT}", ""]
    for label, adapter in RUNS:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        current = model if adapter is None else attach_adapter(model, adapter)
        print(f"=== {label} ===", flush=True)
        text = generate(current, tokenizer, PROMPT, args.temperature, args.num_predict)
        lines += [f"### {label}", text, ""]
        if adapter is not None:
            model = current.unload()
    args.out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"записано {args.out}")


if __name__ == "__main__":
    main()
