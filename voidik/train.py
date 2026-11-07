"""Training entry point for Voidik-0.8B.

Single GPU:
    python -m voidik.train --config configs/voidik_0.8b.yaml

Multi-GPU (one node):
    torchrun --standalone --nproc_per_node=8 -m voidik.train --config configs/voidik_0.8b.yaml

The script uses DDP, bf16 autocast, gradient checkpointing, and either AdamW or
Adafactor.  A 0.8B model does not fit a full AdamW training step on an 8 GB
consumer GPU; see README.md for the recommended hardware envelope.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import random
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from voidik.config import Config
from voidik.data import build_dataloader
from voidik.model import VoidikForCausalLM


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def setup_distributed() -> tuple[int, int, int, torch.device]:
    rank = get_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() and os.name != "nt" else "gloo"
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(hours=2),
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, world_size, local_rank, device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def lr_at_step(step: int, config: Config) -> float:
    train = config.train
    if step < train.warmup_steps:
        return train.lr * (step + 1) / max(train.warmup_steps, 1)
    if train.max_steps <= train.warmup_steps:
        return train.min_lr
    progress = (step - train.warmup_steps) / (train.max_steps - train.warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return train.min_lr + (train.lr - train.min_lr) * cosine


def build_optimizer(model: torch.nn.Module, config: Config) -> torch.optim.Optimizer:
    decay_names = []
    no_decay_names = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2:
            no_decay_names.append(name)
        elif any(
            marker in name
            for marker in (
                "embed_tokens",
                "ngram.embeddings",
                "a_log",
                "dt_bias",
                "read_gate",
                "write_gate",
            )
        ):
            no_decay_names.append(name)
        else:
            decay_names.append(name)

    named_parameters = dict(model.named_parameters())
    decay_params = [named_parameters[name] for name in decay_names]
    no_decay_params = [named_parameters[name] for name in no_decay_names]
    groups = [
        {"params": decay_params, "weight_decay": config.train.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    if config.train.optimizer == "adafactor":
        return torch.optim.Adafactor(
            groups,
            lr=config.train.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )
    kwargs = {}
    if config.train.fused_adamw and torch.cuda.is_available():
        kwargs["fused"] = True
    return torch.optim.AdamW(
        groups,
        lr=config.train.lr,
        betas=(config.train.beta1, config.train.beta2),
        eps=config.train.eps,
        **kwargs,
    )


def compute_loss(
    outputs,
    input_ids: torch.Tensor,
    config: Config,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    block_size = config.data.block_size
    if input_ids.shape[1] < block_size + 1:
        raise ValueError("batch is shorter than data.block_size + 1")

    main_logits = outputs.logits[:, :block_size]
    main_targets = input_ids[:, 1 : block_size + 1]
    main_loss = F.cross_entropy(
        main_logits.reshape(-1, main_logits.shape[-1]),
        main_targets.reshape(-1),
    )
    total = main_loss
    mtp_total = main_loss.new_zeros(())
    for depth, mtp_logits in enumerate(outputs.mtp_logits, start=1):
        logits = mtp_logits[:, :block_size]
        targets = input_ids[:, depth + 1 : depth + 1 + block_size]
        if logits.shape[1] != targets.shape[1]:
            keep = min(logits.shape[1], targets.shape[1])
            logits = logits[:, :keep]
            targets = targets[:, :keep]
        depth_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )
        mtp_total = mtp_total + depth_loss
        total = total + config.model.mtp.loss_weight * depth_loss

    aux_loss = outputs.aux_loss
    z_loss = outputs.z_loss
    if aux_loss is not None:
        total = total + config.model.moe.aux_loss_coef * aux_loss
    if z_loss is not None:
        total = total + config.model.moe.router_z_loss_coef * z_loss

    metrics = {
        "main_loss": main_loss.detach(),
        "mtp_loss": mtp_total.detach(),
        "total_loss": total.detach(),
    }
    if aux_loss is not None:
        metrics["aux_loss"] = aux_loss.detach()
    if z_loss is not None:
        metrics["z_loss"] = z_loss.detach()
    return total, metrics


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    config: Config,
    device: torch.device,
    autocast_context,
    rank: int,
    world_size: int,
) -> float | None:
    if loader is None:
        return None
    model.eval()
    losses = []
    iterator = iter(loader)
    for _ in range(config.train.eval_batches):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        batch = move_batch(batch, device)
        with autocast_context():
            outputs = model(batch["input_ids"])
            _, metrics = compute_loss(outputs, batch["input_ids"], config)
        losses.append(metrics["main_loss"])
    if not losses:
        model.train()
        return None
    value = torch.stack(losses).mean()
    if world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.AVG)
    model.train()
    return float(value.item())


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    module = model.module if isinstance(model, DDP) else model
    while hasattr(module, "_orig_mod"):
        module = module._orig_mod
    return module


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    step: int,
    config: Config,
    output_dir: Path,
    best_val: float | None,
) -> None:
    if get_rank() != 0:
        return
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    raw_model = unwrap_model(model)
    state = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "best_val": best_val,
        "config": config.to_dict(),
        "rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    latest = checkpoint_dir / "latest.pt"
    temporary = checkpoint_dir / "latest.pt.tmp"
    torch.save(state, temporary)
    temporary.replace(latest)
    # Keep the latest N numbered checkpoints. The latest.pt file is always kept.
    numbered = sorted(checkpoint_dir.glob("step_*.pt"))
    for old in (
        numbered[: -config.train.keep_last_checkpoints]
        if config.train.keep_last_checkpoints > 0
        else []
    ):
        old.unlink(missing_ok=True)


def make_checkpoint_copy(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    step: int,
    config: Config,
    output_dir: Path,
    best_val: float | None,
) -> None:
    """Create a numbered hardlink to latest.pt (fallback: copy)."""
    del model, optimizer, scaler, config, best_val
    if get_rank() != 0:
        return
    checkpoint_dir = output_dir / "checkpoints"
    latest = checkpoint_dir / "latest.pt"
    if not latest.exists():
        return
    path = checkpoint_dir / f"step_{step:08d}.pt"
    if path.exists():
        path.unlink()
    try:
        os.link(latest, path)
    except OSError:
        import shutil

        shutil.copyfile(latest, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler,
    device: torch.device,
) -> tuple[int, float | None]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    raw_model = unwrap_model(model)
    raw_model.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    if "rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["rng_state"].cpu().to(torch.uint8))
    if torch.cuda.is_available() and "cuda_rng_state" in checkpoint:
        torch.cuda.set_rng_state_all(
            [state.cpu().to(torch.uint8) for state in checkpoint["cuda_rng_state"]]
        )
    return int(checkpoint.get("step", 0)), checkpoint.get("best_val")


def write_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--toy-data",
        action="store_true",
        help="Use random tokens instead of token shards (pipeline smoke test only).",
    )
    args = parser.parse_args()

    config = Config.load(args.config)
    if args.max_steps is not None:
        config.train.max_steps = args.max_steps
    if args.output_dir is not None:
        config.train.output_dir = args.output_dir
    resume_path = args.resume or config.train.resume

    rank, world_size, local_rank, device = setup_distributed()
    seed_everything(config.train.seed + rank)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    output_dir = Path(config.train.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        config.save(output_dir / "config.yaml")
    if world_size > 1:
        dist.barrier()

    model = VoidikForCausalLM(config.model)
    model.to(device)
    if config.train.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if config.train.compile:
        model = torch.compile(model)

    optimizer = build_optimizer(model, config)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(config.train.dtype == "float16" and torch.cuda.is_available())
    )

    start_step = 0
    best_val = None
    if resume_path:
        start_step, best_val = load_checkpoint(
            resume_path, model, optimizer, scaler, device
        )
        if rank == 0:
            print(f"Resumed from {resume_path} at step {start_step}")

    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            find_unused_parameters=True,
        )

    mtp_depth = config.model.mtp.depth if config.model.mtp.enabled else 0
    toy_vocab = config.model.vocab_size if args.toy_data else None
    train_loader = build_dataloader(
        config.data,
        "train",
        mtp_depth=mtp_depth,
        batch_size=config.train.batch_size,
        seed=config.train.seed,
        distributed=world_size > 1,
        rank=rank,
        world_size=world_size,
        toy_vocab_size=toy_vocab,
        model_vocab_size=config.model.vocab_size,
    )
    val_loader = None
    if not args.toy_data:
        try:
            val_loader = build_dataloader(
                config.data,
                "val",
                mtp_depth=mtp_depth,
                batch_size=config.train.batch_size,
                seed=config.train.seed,
                distributed=world_size > 1,
                rank=rank,
                world_size=world_size,
                model_vocab_size=config.model.vocab_size,
            )
        except FileNotFoundError:
            if rank == 0:
                print("No validation shards found; evaluation disabled.")

    use_autocast = device.type == "cuda" and config.train.dtype != "float32"
    autocast_dtype = (
        torch.bfloat16
        if config.train.dtype == "bfloat16"
        else torch.float16
        if config.train.dtype == "float16"
        else torch.float32
    )

    def autocast_context():
        if use_autocast:
            return torch.autocast(device_type="cuda", dtype=autocast_dtype)
        return contextlib.nullcontext()

    if rank == 0:
        raw_model = unwrap_model(model)
        print(json.dumps(config.to_dict(), indent=2, default=str))
        print(
            f"parameters: total={raw_model.num_parameters():,} "
            f"active={raw_model.active_parameters():,}"
        )
        print(
            f"world_size={world_size} device={device} "
            f"tokens/step={config.train.batch_size * config.train.grad_accum * config.data.block_size * world_size:,}"
        )

    train_loader_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    step = start_step
    tokens_seen = (
        start_step
        * config.train.batch_size
        * config.train.grad_accum
        * config.data.block_size
        * world_size
    )
    last_log_time = time.time()
    last_log_tokens = tokens_seen
    progress = tqdm(
        total=config.train.max_steps,
        initial=start_step,
        disable=rank != 0,
        dynamic_ncols=True,
    )

    while step < config.train.max_steps:
        # Linear warmup + cosine decay, updated once per optimizer step.
        current_lr = lr_at_step(step, config)
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        step_metrics: dict[str, float] = {}
        micro_batches = []
        for _ in range(config.train.grad_accum):
            try:
                batch = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(train_loader)
                batch = next(train_loader_iter)
            micro_batches.append(batch)

        for micro_idx, batch in enumerate(micro_batches):
            batch = move_batch(batch, device)
            sync_context = (
                model.no_sync()
                if world_size > 1 and (micro_idx + 1) < config.train.grad_accum
                else contextlib.nullcontext()
            )
            with sync_context:
                with autocast_context():
                    outputs = model(batch["input_ids"])
                    loss, metrics = compute_loss(outputs, batch["input_ids"], config)
                scaled_loss = loss / config.train.grad_accum
                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
            for key, value in metrics.items():
                step_metrics[key] = step_metrics.get(key, 0.0) + float(value.item())

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.train.grad_clip
        )
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        progress.update(1)
        tokens_seen += (
            config.train.batch_size
            * config.train.grad_accum
            * config.data.block_size
            * world_size
        )

        if step % config.train.log_every == 0 or step == 1:
            now = time.time()
            elapsed = max(now - last_log_time, 1e-6)
            tokens_per_sec = (tokens_seen - last_log_tokens) / elapsed
            last_log_time = now
            last_log_tokens = tokens_seen
            averaged = {key: value / config.train.grad_accum for key, value in step_metrics.items()}
            message = {
                "step": step,
                "tokens": tokens_seen,
                "lr": round(current_lr, 8),
                "grad_norm": round(float(gradient_norm), 4),
                "tok/s": round(tokens_per_sec, 1),
                **{key: round(value, 6) for key, value in averaged.items()},
            }
            if rank == 0:
                tqdm.write(json.dumps(message))
                write_jsonl(output_dir / "train_log.jsonl", message)

        if config.train.eval_every > 0 and step % config.train.eval_every == 0:
            val_loss = evaluate(
                model,
                val_loader,
                config,
                device,
                autocast_context,
                rank,
                world_size,
            )
            if rank == 0 and val_loss is not None:
                record = {
                    "step": step,
                    "val_loss": val_loss,
                    "val_ppl": math.exp(min(val_loss, 20.0)),
                }
                tqdm.write(json.dumps(record))
                write_jsonl(output_dir / "eval_log.jsonl", record)
                if best_val is None or val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(
                        model,
                        optimizer,
                        scaler,
                        step,
                        config,
                        output_dir,
                        best_val,
                    )

        if config.train.save_every > 0 and step % config.train.save_every == 0:
            save_checkpoint(
                model,
                optimizer,
                scaler,
                step,
                config,
                output_dir,
                best_val,
            )
            if rank == 0:
                make_checkpoint_copy(
                    model,
                    optimizer,
                    scaler,
                    step,
                    config,
                    output_dir,
                    best_val,
                )
            if world_size > 1:
                dist.barrier()

    progress.close()
    save_checkpoint(
        model, optimizer, scaler, step, config, output_dir, best_val
    )
    if rank == 0:
        make_checkpoint_copy(
            model, optimizer, scaler, step, config, output_dir, best_val
        )
        print(f"Training finished at step {step}. Checkpoints: {output_dir / 'checkpoints'}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
