"""Скачать страницы lib.ru и нарезать их на рассказы в JSONL.

Каркас уже качает страницы и пишет JSONL. Хармс — одна страница, рассказы
между <h2>. Чехов — десять томов с az.lib.ru, рассказ начинается с якоря.
Вам нужно дописать три места, помеченные TODO: кодировку, склейку переносов
и отбор рассказов.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

SOURCES = {
    "harms": ["https://www.lib.ru/HARMS/harms.txt"],
    # Тома 1–10 полного собрания. text_0032 — подписи к рисункам, не проза.
    "chekhov": [
        f"http://az.lib.ru/c/chehow_a_p/text_{number:04d}.shtml"
        for number in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
    ],
}

# TODO: у авторов разные кодировки. Хармс на lib.ru — KOI8-R.
# Тома Чехова на az.lib.ru — windows-1251. Подставьте имена кодировок.
ENCODINGS = {"harms": None, "chekhov": None}

H2_RE = re.compile(r"<h2>(.*?)</h2>", re.IGNORECASE | re.DOTALL)
ANCHOR_RE = re.compile(r'<a name="(\d+)"></a>', re.IGNORECASE)
TITLE_RE = re.compile(r"<FONT color=darkred><B>(.*?)</B>", re.IGNORECASE | re.DOTALL)
H4_TITLE_RE = re.compile(r"<h4>.*?<b>(.*?)</b>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
MIN_CHARS = 200


def download(url: str, encoding: str) -> str:
    # curl -4: у lib.ru есть AAAA, а маршрута IPv6 часто нет.
    completed = subprocess.run(
        ["curl", "-4", "-fsSL", "--retry", "3", "--max-time", "180", url],
        check=True,
        capture_output=True,
    )
    if not encoding:
        raise NotImplementedError("укажите кодировку страницы в ENCODINGS")
    return completed.stdout.decode(encoding, errors="replace")


def html_to_text(fragment: str) -> str:
    text = TAG_RE.sub("", fragment)
    text = text.replace("&quot;", '"').replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("\xa0", " ")
    # TODO: в конце строки lib.ru часто стоит перенос слова: "изобрета-\nтельностью".
    # Замените такой перенос на склейку букв (без дефиса и без перевода строки).
    # Обычные строки ниже уже склеиваются пробелом.
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


def keep_story(title: str, text: str) -> bool:
    """Решите, класть ли этот кусок в обучение.

    В страницы входят титул, оглавление, «От редакции», примечания и письма.
    Для адаптера нужны сами рассказы. Регистр сравнивайте через title.casefold():
    флаг re.IGNORECASE здесь не меняет кириллицу. Обрывки короче MIN_CHARS отбросьте.
    """
    # TODO: верните True только для рассказа.
    raise NotImplementedError("опишите, какой фрагмент считается рассказом")


def split_by_h2(page: str) -> list[dict]:
    """Куски до первого <h2> отбрасываются. Так нарезана страница Хармса."""
    parts = H2_RE.split(page)
    stories = []
    for title_html, body_html in zip(parts[1::2], parts[2::2]):
        title = html_to_text(title_html)
        text = html_to_text(body_html)
        if keep_story(title, text):
            stories.append({"title": title, "text": text})
    return stories


def split_by_anchor(page: str) -> list[dict]:
    """Тома Чехова: каждый рассказ начинается с <a name="N"> и заголовка."""
    parts = ANCHOR_RE.split(page)
    stories = []
    for body_html in parts[2::2]:
        title_match = TITLE_RE.search(body_html) or H4_TITLE_RE.search(body_html)
        if title_match is None:
            continue
        title = html_to_text(title_match.group(1))
        text = html_to_text(body_html)
        if keep_story(title, text):
            stories.append({"title": title, "text": text})
    return stories


def split_stories(page: str) -> list[dict]:
    by_heading = split_by_h2(page)
    by_anchor = split_by_anchor(page)
    if len(by_anchor) >= len(by_heading) and by_anchor:
        return by_anchor
    return by_heading


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author", choices=sorted(SOURCES), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    stories = []
    seen_titles = set()
    for url in SOURCES[args.author]:
        print(f"качаю {url}", flush=True)
        page = download(url, ENCODINGS[args.author])
        for story in split_stories(page):
            key = story["title"].casefold()
            if key in seen_titles:
                continue
            seen_titles.add(key)
            stories.append(story)
    if not stories:
        raise SystemExit("после чистки не осталось рассказов, проверьте кодировку и нарезку")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for story in stories:
            fh.write(json.dumps(story, ensure_ascii=False) + "\n")
    chars = sum(len(s["text"]) for s in stories)
    print(f"{args.author}: {len(stories)} рассказов, {chars} символов -> {args.out}")


if __name__ == "__main__":
    main()
