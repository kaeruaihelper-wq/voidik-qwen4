"""Train a compact byte-level BPE tokenizer for local demos.

Example:
    python scripts/train_tokenizer.py \
        --input data/demo/raw/mixed.txt \
        --output data/demo/tokenizer \
        --vocab-size 16000
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path


def iter_lines(paths: list[str], max_bytes: int):
    consumed = 0
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                consumed += len(line.encode("utf-8")) + 1
                yield line
                if max_bytes > 0 and consumed >= max_bytes:
                    return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/demo/raw/mixed.txt")
    parser.add_argument("--output", default="data/demo/tokenizer")
    parser.add_argument("--vocab-size", type=int, default=16_000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--sample-mb", type=float, default=30.0)
    parser.add_argument(
        "--special-tokens",
        default="<|endoftext|>,<|pad|>",
        help="Comma-separated special tokens.",
    )
    args = parser.parse_args()

    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    except ImportError as exc:
        print("Install data dependencies first: pip install -e .[data]")
        raise SystemExit(1) from exc

    paths = sorted(glob.glob(args.input))
    if not paths:
        raise SystemExit(f"No input files matched {args.input!r}")

    special_tokens = [token.strip() for token in args.special_tokens.split(",") if token.strip()]
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=True
    )
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    max_bytes = int(args.sample_mb * 1_000_000)
    print(f"training tokenizer on {', '.join(paths)} (up to {args.sample_mb:g} MB)")
    tokenizer.train_from_iterator(iter_lines(paths, max_bytes), trainer=trainer)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_dir / "tokenizer.json"))
    (output_dir / "tokenizer_config.json").write_text(
        '{"tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": 2048}\n',
        encoding="utf-8",
    )

    print(f"vocab size: {tokenizer.get_vocab_size():,}")
    for token in special_tokens:
        print(f"{token!r} -> {tokenizer.token_to_id(token)}")
    for text in ("Once upon a time, there was a little girl.", "Однажды в лесу жил зайчик."):
        ids = tokenizer.encode(text).ids
        print(f"sample: {text!r} -> {len(ids)} tokens -> {tokenizer.decode(ids)!r}")


if __name__ == "__main__":
    main()
