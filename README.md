# Voidik-0.8B

> **Status: research prototype.** No trained weights are released. The 0.8B
> reference model has not been trained. A separate 58.9M compact configuration
> completed a 1,000-step pilot; its early loss and inference results are
> documented below. These limited results do not establish full-scale training
> stability or competitive language quality.

[Русская версия](README.ru.md)

Voidik-0.8B is an experimental decoder-only language model design with about
799M parameters (about 587M active per token). It combines Gated DeltaNet
(GDN), block-sparse attention (QSA), sparse MoE, hashed n-gram embeddings,
Expanded Residual / GR-style read-write gates, and Multi-Token Prediction
(MTP).

This repository contains model and training code, data-preparation utilities,
configuration files, parameter counting, and tests. The main configuration
targets multi-GPU training on a GPU cluster. A single 8 GB GPU is not suitable
for training the full-size model.

## Research status and limitations

- The project is an implementation experiment, not a released pretrained
  model. The 0.8B reference model has not been trained, and no checkpoint is
  published.
- A separate 58.9M model completed a 1,000-step pilot on FineWeb-Edu. Its
  measured validation loss and sample generations are reported below; the run
  is too small to establish useful general language ability.
- The smoke configuration uses a tiny model and toy data; it does not assess
  the 0.8B configuration or resulting language-model quality.
- Some architectural details were inferred because the source design
  description was underspecified. This implementation should not be treated as
  an exact reproduction.
- QSA uses a fixed causal block-sparse pattern (local window plus sink tokens);
  it does not learn which blocks to attend to.
- GR-style read/write is implemented as a per-token gated latent bank. The
  bank does not carry state across tokens, so this is not a fully recurrent
  interpretation.
- N-gram embeddings are hashed, so collisions are possible.
- `generate()` is a reference implementation without a KV cache and is not
  optimized for inference.
- MTP is configured with depth 1. Validation loss is computed for the main
  prediction head only.
- Hardware, runtime, and cost figures below are rough planning estimates, not
  measured results for this implementation.

Contributions, issue reports, and independent testing are welcome. Please
include hardware details, configuration, and reproducible steps when reporting
training or performance results.

## Architecture

The 20-layer pattern repeats **GDN → GDN → GDN → QSA** five times.

| Component | Configuration |
|---|---|
| Hidden size | 1536 |
| Layers | 20: 15 GDN + 5 QSA |
| QSA | Causal block-sparse attention: 512-token local window + 4 sink tokens, block size 64, GQA with 12 Q / 4 KV heads, head dim 128, QK-norm, RoPE |
| GDN | Chunk-parallel Gated DeltaNet, 4 K heads / 12 V heads, head dim 128, convolution kernel 4, chunk size 64 |
| MoE | 8 routed experts + 1 shared expert, top-2 routing, FFN size 384, dropless routing, auxiliary loss 0.01 and router z-loss 0.001 |
| Expanded Residual / GR | 4 latent slots × 128 per token; read/write gates around each mixer, followed by a final merge into the residual stream |
| N-gram embeddings | Hashed 2- and 3-grams, 131,072 buckets per order, dimension 32; injected at layer 2 (zero-based `inject_layer: 1`) |
| MTP | One next-next-token prediction module, loss weight 0.1 |
| Vocabulary | 151,936 (Qwen3-compatible size; `Qwen/Qwen3-0.6B` tokenizer has 151,669 base tokens) |
| Training context | 2,048 tokens |

### Parameter counts

```text
total parameters:              799,098,117
active parameters/token:       586,761,477
inactive expert params:        212,336,640
token embedding:               233,373,696   (tied with lm_head)
n-gram embedding:                 8,437,793
non-embedding:                 557,286,628
active non-embedding:          344,949,988
```

Recalculate locally:

```bash
python -m tools.count_params --config configs/voidik_0.8b.yaml
```

## Requirements and installation

Python 3.10+ and PyTorch 2.5+ are required. CUDA-capable PyTorch is needed for
GPU training; the tiny smoke configuration can also be used for limited CPU
checks.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -e .
# Optional dependencies for data preparation
pip install -e ".[data]"
# Test dependencies
pip install -e ".[dev]"
```

Check the PyTorch installation and CUDA availability:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Data preparation

`scripts/prepare_data.py` tokenizes a stream of documents and writes packed
`train_*.bin` / `val_*.bin` shards plus `meta.json`. The files contain a flat
stream of token IDs (`uint32` for the Qwen vocabulary), from which training
samples random crops.

### FineWeb-Edu

```bash
python scripts/prepare_data.py \
  --dataset HuggingFaceFW/fineweb-edu \
  --split train \
  --val-split train \
  --tokens 20e9 \
  --val-tokens 20e6 \
  --shard-tokens 1e9 \
  --tokenizer Qwen/Qwen3-0.6B \
  --output data/fineweb
```

With `--val-split` equal to `--split`, the first `--val-tokens` documents in
the stream are used for validation, and training shards start after them.
Twenty billion `uint32` tokens require about 80 GB of disk space; 100 billion
require about 400 GB. Dataset access and use are subject to the dataset's own
terms.

### Local text corpus

```bash
python scripts/prepare_data.py \
  --dataset text \
  --data-files "corpus/*.txt" \
  --split train \
  --val-split train \
  --tokens 1e9 \
  --val-tokens 1e6 \
  --tokenizer Qwen/Qwen3-0.6B \
  --output data/local
```

Set the corresponding paths in the configuration:

```yaml
data:
  train_glob: data/local/train_*.bin
  val_glob: data/local/val_*.bin
```

Other useful options include `--text-column`, `--min-chars`, `--max-docs`,
`--eos-id`, and `--overwrite`. The tokenizer vocabulary must not exceed
`model.vocab_size` (151,936). Tokenizer assets are downloaded separately; use
them according to their applicable license and terms.

### Compact single-GPU experiment (~59M)

The [configs/voidik_58m.yaml](configs/voidik_58m.yaml) configuration retains
GDN, QSA, MoE, n-gram embeddings, the residual bank, and MTP while reducing the
vocabulary to 16,384 tokens. It uses the public FineWeb-Edu English dataset
both to train a compact BPE tokenizer and to prepare disjoint training and
validation streams:

```powershell
pip install -e ".[data]"
python scripts/train_tokenizer.py --dataset HuggingFaceFW/fineweb-edu --split train --max-docs 50000 --output data/voidik_58m/tokenizer --vocab-size 16384 --sample-mb 50
python scripts/prepare_data.py --dataset HuggingFaceFW/fineweb-edu --split train --val-split train --tokens 40m --val-tokens 1m --shard-tokens 10m --tokenizer data/voidik_58m/tokenizer --output data/voidik_58m/tokens
python -m tools.count_params --config configs/voidik_58m.yaml
python -m voidik.train --config configs/voidik_58m.yaml
```

The dataset is streamed rather than downloaded in full. Review FineWeb-Edu's
license and terms before using it.

This configuration targets an 8 GB RTX 3070 with batch size 1, context 256,
and gradient accumulation 8. The 1,000-step pilot processes about 2.05 million
training tokens. It tests the training loop and early loss trend, not final
language quality. The estimated run time is about 75 minutes on that GPU based
on the initial smoke run; evaluation may add time, and real throughput may
differ.

Generate samples after training:

```powershell
python scripts/generate.py --config configs/voidik_58m.yaml --checkpoint runs/voidik-58m-1000/checkpoints/latest.pt --tokenizer data/voidik_58m/tokenizer --prompt "Once upon a time"
```

This is next-token training on ordinary text, not instruction tuning. Early
generations may be incoherent; inspect `train_log.jsonl` and `eval_log.jsonl`,
comparing the same prompts across checkpoints.

#### Recorded 58M pilot and inference check

The completed run processed 2,048,000 training tokens on one RTX 3070. Main
validation loss was **6.21** at step 500 and **5.92** at step 1,000
(perplexity **371** at step 1,000). These are measurements from this pilot,
not a benchmark or evidence of general language competence.

With prompt `Hello!`, temperature `0.1` repeatedly generated `||`. At
temperature `0.7`, repetition stopped, but the result was still mostly
incoherent:

```text
Hello!
|Pani||Cigb]
- Dicmaka is a result of the number of "pentical� (p. 1)
- "Petu
```

Reproduce the inference check on the trained checkpoint:

```powershell
python scripts/generate.py --config configs/voidik_58m.yaml --checkpoint runs/voidik-58m-1000/checkpoints/latest.pt --tokenizer data/voidik_58m/tokenizer --prompt "Hello!" --max-new-tokens 40 --temperature 0.7 --top-k 40
```

The tokenizer, dataset shards, and checkpoint are local experiment artifacts
and are not included in this repository.

## Training

### Smoke test (toy data, tiny configuration)

```bash
python -m voidik.train --config configs/smoke.yaml --toy-data --max-steps 10
```

This is a software smoke test only. It does not train or evaluate the
0.8B-parameter configuration.

### Resume the compact run

```bash
python -m voidik.train --config configs/voidik_58m.yaml \
  --resume runs/voidik-58m-1000/checkpoints/latest.pt
```

### Single GPU

```bash
python -m voidik.train --config configs/voidik_0.8b.yaml \
  --output-dir runs/voidik-0.8b-single
```

### Distributed training (8 GPUs)

```bash
torchrun --standalone --nproc_per_node=8 -m voidik.train \
  --config configs/voidik_0.8b.yaml
```

The main configuration uses batch size 1 per GPU, gradient accumulation 32,
and sequence length 2,048. On 8 GPUs that is 524,288 tokens per optimizer
step; 40,000 steps correspond to about 21B tokens. These are configuration
arithmetic, not a claim that such a run has been completed.

Key settings include AdamW, learning rate 6e-4 with cosine decay to 6e-5,
2,000 warmup steps, weight decay 0.1, gradient clipping 1.0, bf16 autocast,
and gradient checkpointing.

### Resume from checkpoint

```bash
python -m voidik.train --config configs/voidik_0.8b.yaml \
  --resume runs/voidik-0.8b/checkpoints/latest.pt
```

Checkpoints are saved under `<output_dir>/checkpoints/` (`latest.pt` and
numbered `step_*.pt` files, limited by `keep_last_checkpoints`). Training and
evaluation logs are written to `<output_dir>/train_log.jsonl` and
`eval_log.jsonl`; `config.yaml` is saved alongside them.

The script detects rank and world size through `torchrun`; no additional DDP
environment setup is required. The main configuration is not practical to
train on CPU.

## Monitoring and evaluation

- `train_log.jsonl`: loss, learning rate, step time, and throughput.
- `eval_log.jsonl`: validation loss for the main head only (without MTP).
- Checkpoints store model, optimizer, scaler, and RNG state for resuming.
- TensorBoard logging is not currently implemented, despite the
  `train.tensorboard` config option.

## Tests

```bash
pytest
```

The tests cover QSA shapes and causality, chunk-parallel GDN equivalence,
n-gram hashing, configuration handling, and model forward/backward behavior.
Passing tests is not a substitute for full-scale training or evaluation.

## Hardware and rough planning estimates

- **Memory:** approximately 13–16 GB per replica for bf16 weights, gradients,
  and AdamW states, plus activations. Gradient checkpointing may make
  24–40 GB GPUs more practical. 8 × A100 (40/80 GB) or 8 × H100 is a possible
  cluster target, not a tested requirement.
- **Parallelism:** DDP replicates the full model on every GPU; FSDP is not
  implemented.
- **Runtime:** 2–5 days for 21B tokens on 8 × A100 is a rough estimate only;
  actual throughput has not been benchmarked.
- **Cost:** the prior rough estimate of $400–2,000 assumes rented A100s at
  around $1–2 per GPU-hour. Provider, utilization, and throughput can change
  this substantially; treat it as illustrative, not a quote.

## Repository layout

```text
configs/
  voidik_0.8b.yaml       # Main 0.8B configuration
  voidik_58m.yaml        # Compact single-GPU experiment
  smoke.yaml             # Tiny configuration for pipeline checks
voidik/
  config.py               # Dataclass configuration and YAML loader
  train.py                # DDP training, evaluation, checkpoint/resume
  data/dataset.py         # Packed memmap dataset
  model/                  # Model and component implementations
scripts/
  prepare_data.py         # Tokenize documents into shards
  generate.py             # Reference generation utility
tools/
  count_params.py         # Total/active parameter counts
tests/                    # Unit and smoke tests
```

## Licensing

The original Voidik source code in this repository is licensed under the MIT
License; see [LICENSE](LICENSE).

The MIT License applies only to the project materials covered by this
repository's license. It does not relicense third-party software, datasets, or
other external assets. The Qwen Community License 1.0 applies to software
released by Qwen, not to this independently authored Voidik code. This
repository does not include Qwen model weights. If you download the
`Qwen/Qwen3-0.6B` tokenizer or other external assets, review and comply with
the license and terms published for those specific assets. Dataset terms
apply separately as well.
