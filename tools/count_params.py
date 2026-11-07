"""Count total and active parameters for a Voidik configuration."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

from voidik.config import Config
from voidik.model import VoidikForCausalLM


def categorize(name: str) -> str:
    if name.startswith("embed_tokens"):
        return "token_embedding"
    if name.startswith("ngram"):
        return "ngram_embedding"
    if name.startswith("layers."):
        parts = name.split(".")
        if len(parts) > 2:
            if parts[2] == "mixer":
                return f"layers.{parts[2]}.{parts[3].split('.')[0]}"
            if parts[2] == "moe":
                return "layers.moe"
            if parts[2] == "residual_bank":
                return "layers.residual_bank"
            if parts[2] in ("input_norm", "post_moe_norm"):
                return "layers.norm"
    if name.startswith("mtp_modules"):
        return "mtp"
    if name.startswith("bank_merge") or name.startswith("bank_final_norm"):
        return "bank_merge"
    if name.startswith("lm_head"):
        return "lm_head"
    if name.startswith("norm"):
        return "final_norm"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config = Config.load(args.config)
    model = VoidikForCausalLM(config.model)
    breakdown: dict[str, int] = defaultdict(int)
    for name, parameter in model.named_parameters():
        breakdown[categorize(name)] += parameter.numel()

    total = model.num_parameters()
    active = model.active_parameters()
    token_embedding = breakdown.get("token_embedding", 0)
    ngram_embedding = breakdown.get("ngram_embedding", 0)
    result = {
        "total": total,
        "active": active,
        "inactive": total - active,
        "token_embedding": token_embedding,
        "ngram_embedding": ngram_embedding,
        "non_embedding": total - token_embedding - ngram_embedding,
        "active_non_embedding": active - token_embedding - ngram_embedding,
        "breakdown": dict(sorted(breakdown.items())),
    }
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(f"total parameters:          {total:>15,}")
    print(f"active parameters/token:   {active:>15,}")
    print(f"inactive expert params:    {total - active:>15,}")
    print(f"token embedding:           {token_embedding:>15,}")
    print(f"n-gram embedding:          {ngram_embedding:>15,}")
    print(f"non-embedding:             {result['non_embedding']:>15,}")
    print(f"active non-embedding:      {result['active_non_embedding']:>15,}")
    print("\nbreakdown:")
    for key, value in result["breakdown"].items():
        print(f"  {key:<28} {value:>15,}")


if __name__ == "__main__":
    main()
