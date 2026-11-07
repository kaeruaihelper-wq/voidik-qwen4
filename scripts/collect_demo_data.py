"""Collect a small bilingual story corpus for quick local demos.

The demo mixes:
  * TinyStories (English, simple children's stories)
  * ru_stories (Russian, five-sentence stories)

The output is one story per line, alternating languages.  ``prepare_data.py``
can consume the mixed file directly.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path


def clean_text(text: str) -> str:
    text = str(text).replace("\r", " ").replace("\n", " ")
    return " ".join(text.split())


def iter_tinystories(limit: int):
    from datasets import load_dataset

    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    seen = 0
    for example in dataset:
        text = clean_text(example.get("text", ""))
        if len(text) < 80:
            continue
        yield text
        seen += 1
        if seen >= limit:
            break


def iter_ru_stories(limit: int):
    from datasets import load_dataset

    dataset = load_dataset("inkoziev/ru_stories", split="train", streaming=True)
    seen = 0
    for example in dataset:
        sentences = [
            clean_text(example.get(f"sentence{idx}", "")) for idx in range(1, 6)
        ]
        text = " ".join(sentence for sentence in sentences if sentence)
        if len(text) < 80:
            continue
        yield text
        seen += 1
        if seen >= limit:
            break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/demo/raw")
    parser.add_argument("--max-per-language", type=int, default=50_000)
    args = parser.parse_args()

    try:
        import datasets  # noqa: F401
    except ImportError as exc:
        print("Install data dependencies first: pip install -e .[data]", file=sys.stderr)
        raise SystemExit(1) from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    en_path = output_dir / "en.txt"
    ru_path = output_dir / "ru.txt"
    mixed_path = output_dir / "mixed.txt"

    en_count = 0
    ru_count = 0
    with (
        open(en_path, "w", encoding="utf-8", newline="\n") as en_file,
        open(ru_path, "w", encoding="utf-8", newline="\n") as ru_file,
        open(mixed_path, "w", encoding="utf-8", newline="\n") as mixed_file,
    ):
        en_iter = iter_tinystories(args.max_per_language)
        ru_iter = iter_ru_stories(args.max_per_language)
        for en_text, ru_text in itertools.zip_longest(en_iter, ru_iter):
            if en_text:
                en_file.write(en_text + "\n")
                mixed_file.write(en_text + "\n")
                en_count += 1
            if ru_text:
                ru_file.write(ru_text + "\n")
                mixed_file.write(ru_text + "\n")
                ru_count += 1
            if (en_count + ru_count) % 10_000 == 0:
                print(f"collected en={en_count:,} ru={ru_count:,}", flush=True)

    print(f"English stories: {en_count:,} -> {en_path}")
    print(f"Russian stories: {ru_count:,} -> {ru_path}")
    print(f"Mixed corpus:    {mixed_path} ({mixed_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
