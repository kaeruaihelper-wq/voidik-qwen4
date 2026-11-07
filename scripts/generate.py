"""Generate samples from a trained Voidik checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def load_tokenizer(path: str):
    from tokenizers import Tokenizer

    tokenizer_path = Path(path)
    if tokenizer_path.is_dir():
        tokenizer_path = tokenizer_path / "tokenizer.json"
    return Tokenizer.from_file(str(tokenizer_path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="data/demo/tokenizer")
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    from voidik.config import Config
    from voidik.model import VoidikForCausalLM

    config = Config.load(args.config)
    tokenizer = load_tokenizer(args.tokenizer)
    eos_id = tokenizer.token_to_id("<|endoftext|>")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VoidikForCausalLM(config.model)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    prompts = args.prompt or [
        "Once upon a time, there was a little girl named Lily.",
        "Однажды в маленьком лесу жил маленький зайчик.",
        "The little cat wanted to play with a red ball.",
        "Маленький мальчик нашёл на дороге красивую ракушку.",
    ]

    lines: list[str] = []
    for prompt in prompts:
        torch.manual_seed(args.seed)
        ids = tokenizer.encode(prompt).ids
        if not ids:
            ids = [eos_id]
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        output = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            eos_token_id=eos_id,
        )
        text = tokenizer.decode(output[0].tolist()).strip()
        block = f"PROMPT: {prompt}\n{text}\n"
        print(block)
        lines.append(block)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"saved samples to {output_path}")


if __name__ == "__main__":
    main()
