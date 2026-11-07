"""Tokenize a HuggingFace dataset into packed uint16/uint32 shards.

Examples
--------
FineWeb-Edu:
    python scripts/prepare_data.py --dataset HuggingFaceFW/fineweb-edu \
        --split train --val-split validation --tokens 20e9 --val-tokens 20e6 \
        --tokenizer Qwen/Qwen3-0.6B --output data/fineweb

Local text files:
    python scripts/prepare_data.py --dataset text --data-files "corpus/*.txt" \
        --split train --tokens 1e9 --output data/local
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def parse_count(value: str) -> int:
    value = str(value).strip().lower().replace("_", "")
    if value.endswith("b"):
        return int(float(value[:-1]) * 1_000_000_000)
    if value.endswith("m"):
        return int(float(value[:-1]) * 1_000_000)
    if value.endswith("k"):
        return int(float(value[:-1]) * 1_000)
    return int(float(value))


def tokenize_split(
    dataset_name: str,
    split: str,
    text_column: str,
    tokenizer,
    output_dir: Path,
    max_tokens: int,
    shard_tokens: int,
    eos_id: int,
    data_files: str | None,
    min_chars: int,
    max_docs: int | None,
    prefix: str | None = None,
    skip_docs: int = 0,
) -> dict:
    from datasets import load_dataset

    prefix = prefix or ("train" if split == "train" else "val")

    if dataset_name == "text":
        if not data_files:
            raise ValueError("--data-files is required when --dataset text")
        dataset = load_dataset(
            "text", data_files=data_files, split=split, streaming=True
        )
    else:
        dataset = load_dataset(dataset_name, split=split, streaming=True)
    if skip_docs > 0:
        dataset = dataset.skip(skip_docs)

    output_dir.mkdir(parents=True, exist_ok=True)
    for old_shard in output_dir.glob(f"{prefix}_*.bin"):
        old_shard.unlink()
    dtype = np.uint16 if tokenizer.get_vocab_size() <= 65_535 else np.uint32
    chars_per_token = 3.5
    expected = int(max_tokens / chars_per_token) if max_tokens else 0

    shards: list[dict] = []
    current_fh = None
    current_count = 0
    shard_index = 0
    total_tokens = 0
    docs = 0
    started = time.time()

    def open_shard() -> None:
        nonlocal current_fh, current_count, shard_index
        path = output_dir / f"{prefix}_{shard_index:05d}.bin"
        current_fh = open(path, "wb")
        shards.append({"path": path.name, "tokens": 0})
        current_count = 0
        shard_index += 1

    open_shard()
    buffer: list[int] = []
    buffer_tokens = 0
    try:
        for batch in dataset.iter(batch_size=512):
            texts = [t for t in batch[text_column] if t and len(t) >= min_chars]
            if not texts:
                continue
            encodings = tokenizer.encode_batch(texts)
            for encoding in encodings:
                buffer.extend(encoding.ids)
                buffer.append(eos_id)
            buffer_tokens += sum(len(e.ids) + 1 for e in encodings)
            docs += len(texts)

            if buffer_tokens >= 1_000_000 or max_tokens and total_tokens + buffer_tokens >= max_tokens:
                tokens = np.asarray(buffer, dtype=dtype)
                buffer.clear()
                buffer_tokens = 0
                pos = 0
                while pos < len(tokens):
                    remaining = max_tokens - total_tokens if max_tokens else None
                    if remaining is not None and remaining <= 0:
                        break
                    take = min(
                        len(tokens) - pos,
                        shard_tokens - current_count,
                        remaining if remaining is not None else len(tokens) - pos,
                    )
                    current_fh.write(tokens[pos : pos + take].tobytes())
                    current_count += take
                    shards[-1]["tokens"] += take
                    total_tokens += take
                    pos += take
                    if current_count >= shard_tokens:
                        current_fh.close()
                        open_shard()
                if max_tokens and total_tokens >= max_tokens:
                    break

            if docs % 200_000 < 512:
                elapsed = max(time.time() - started, 1e-6)
                current_total = total_tokens + buffer_tokens
                rate = current_total / elapsed
                pct = 100.0 * current_total / max_tokens if max_tokens else 0.0
                print(
                    f"[{split}] docs={docs:,} tokens={current_total:,} "
                    f"({pct:.1f}%) {rate:,.0f} tok/s",
                    flush=True,
                )
            if max_docs is not None and docs >= max_docs:
                break
    finally:
        if buffer:
            tokens = np.asarray(buffer, dtype=dtype)
            pos = 0
            while pos < len(tokens):
                take = min(len(tokens) - pos, shard_tokens - current_count)
                current_fh.write(tokens[pos : pos + take].tobytes())
                current_count += take
                shards[-1]["tokens"] += take
                total_tokens += take
                pos += take
                if current_count >= shard_tokens:
                    current_fh.close()
                    open_shard()
        if current_fh is not None:
            current_fh.close()

    # Remove empty shards created by a shutdown at a boundary.
    for shard in shards:
        if shard["tokens"] == 0:
            (output_dir / shard["path"]).unlink(missing_ok=True)
    shards = [s for s in shards if s["tokens"] > 0]

    return {
        "split": split,
        "dataset": dataset_name,
        "tokenizer": tokenizer_name_global,
        "dtype": np.dtype(dtype).name,
        "vocab_size": tokenizer.get_vocab_size(),
        "eos_id": eos_id,
        "tokens": total_tokens,
        "docs": docs,
        "shards": shards,
        "expected_docs": expected,
    }


tokenizer_name_global = ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--val-dataset", default=None)
    parser.add_argument("--data-files", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--val-split",
        default="train",
        help="Use the same value as --split to hold out the first --val-tokens from "
        "the training stream (no overlap), or name a separate validation split.",
    )
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output", default="data/fineweb")
    parser.add_argument("--tokens", default="20e9")
    parser.add_argument("--val-tokens", default="20e6")
    parser.add_argument("--shard-tokens", default="1e9")
    parser.add_argument("--eos-id", type=int, default=None)
    parser.add_argument("--min-chars", type=int, default=64)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        from datasets import load_dataset  # noqa: F401
        from tokenizers import Tokenizer
    except ImportError as exc:
        print(
            "Install data dependencies first: uv pip install -e .[data]",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    global tokenizer_name_global
    tokenizer_name_global = args.tokenizer
    output_dir = Path(args.output)
    meta_path = output_dir / "meta.json"
    if meta_path.exists() and not args.overwrite:
        print(f"{meta_path} already exists; pass --overwrite to regenerate")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer.from_pretrained(args.tokenizer)
    tokenizer.no_truncation()
    tokenizer.no_padding()
    eos_id = args.eos_id
    if eos_id is None:
        eos_id = tokenizer.token_to_id("<|endoftext|>")
        if eos_id is None:
            eos_id = tokenizer.token_to_id("</s>")
        if eos_id is None:
            raise ValueError("Could not determine EOS id; pass --eos-id")

    max_tokens = parse_count(args.tokens)
    val_tokens = parse_count(args.val_tokens)
    shard_tokens = parse_count(args.shard_tokens)
    if max_tokens <= 0:
        raise ValueError("--tokens must be positive")

    val_dataset = args.val_dataset or args.dataset
    same_stream = val_dataset == args.dataset and args.val_split == args.split

    val_record = None
    train_skip_docs = 0
    if val_tokens > 0:
        if same_stream:
            # Reserve the first docs of the training stream for validation,
            # then skip exactly those docs when materialising the train shards.
            val_record = tokenize_split(
                val_dataset,
                args.val_split,
                args.text_column,
                tokenizer,
                output_dir,
                val_tokens,
                shard_tokens,
                eos_id,
                args.data_files,
                args.min_chars,
                None,
                prefix="val",
            )
            train_skip_docs = int(val_record["docs"])
        else:
            val_record = tokenize_split(
                val_dataset,
                args.val_split,
                args.text_column,
                tokenizer,
                output_dir,
                val_tokens,
                shard_tokens,
                eos_id,
                None,
                args.min_chars,
                None,
                prefix="val",
            )

    train_record = tokenize_split(
        args.dataset,
        args.split,
        args.text_column,
        tokenizer,
        output_dir,
        max_tokens,
        shard_tokens,
        eos_id,
        args.data_files,
        args.min_chars,
        args.max_docs,
        prefix="train",
        skip_docs=train_skip_docs,
    )
    records = [train_record]
    if val_record is not None:
        records.append(val_record)

    meta = {
        "tokenizer": args.tokenizer,
        "vocab_size": tokenizer.get_vocab_size(),
        "dtype": records[0]["dtype"],
        "eos_id": eos_id,
        "splits": records,
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
