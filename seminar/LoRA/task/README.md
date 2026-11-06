# Лабораторная: два LoRA на одной базе

Вы учите два адаптера на одной базе и сравниваете три рассказа: база, адаптер по Хармсу, адаптер по Чехову. Цикл обучения уже написан. Дописать нужно только места с `TODO` — так видно, какие решения вообще делают эту работу, а не обвязку.

Корпуса качаются скриптом, в git их класть не нужно.

1. `prepare_data.py` — кодировка lib.ru, склейка переносов, какие куски страницы являются рассказом.
2. `train_lora.py` — диалог для обучения, параметры LoRA, маска loss (учимся на тексте рассказа).
3. `generate.py` — бытовой промпт и подключение сохранённого адаптера к той же базе через `PeftModel`.

Оба автора вызываются одной и той же командой обучения, меняется только файл данных.

```bash
pip install -r ../requirements.txt
python prepare_data.py --author harms --out ../data/harms.jsonl
python prepare_data.py --author chekhov --out ../data/chekhov.jsonl
python train_lora.py --data ../data/harms.jsonl --out ../output/harms
python train_lora.py --data ../data/chekhov.jsonl --out ../output/chekhov
python generate.py --out generations.md
```

В разборе процитируйте свои три текста: чем ответ уехал от базы и где адаптер не сдвинул стиль.
