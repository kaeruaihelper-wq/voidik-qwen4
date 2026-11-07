# Voidik-0.8B

> **Статус: исследовательский прототип.** Обученных весов в репозитории нет.
> Референсная модель 0.8B не обучалась. Отдельная компактная конфигурация
> 58,9M прошла пилот на 1 000 шагов; ранние метрики и результаты генерации
> приведены ниже. Этих данных недостаточно для выводов о масштабируемости
> обучения или качестве универсальной языковой модели.

English version: [README.md](README.md).

Гибридная decoder-only языковая модель на **~0.8 млрд параметров** (799 098 117
total, 586 761 477 active на токен) с чередованием Gated DeltaNet (GDN) и
block-sparse attention (QSA), разреженным MoE, n-gram embedding, Expanded
Residual / GR Read-Write и Multi-Token Prediction (MTP).

Проект содержит полный цикл: подготовка токенизированных шардов, обучение с
DDP, validation, checkpoint/resume, подсчёт параметров и smoke-тесты. Полный
прогон на 20+ млрд токенов рассчитан на GPU-кластер; на одной RTX 3070 8 ГБ
обучить модель такого размера нельзя.

## Архитектура

Схема повторяется 5 раз: **GDN → GDN → GDN → QSA**, всего 20 слоёв.

| Компонент | Реализация |
|---|---|
| Размер hidden | 1536 |
| Слои | 20: 15 × GDN + 5 × QSA |
| QSA | causal block-sparse attention: локальное окно 512 токенов + 4 sink-токена, block size 64, GQA 12 Q / 4 KV головы, head_dim 128, QK-norm, RoPE |
| GDN | chunk-parallel Gated DeltaNet, 4 K-головы / 12 V-голов, head_dim 128, conv kernel 4, chunk 64 |
| MoE | 8 routed experts + 1 shared expert, top-2, FFN 384, dropless routing, aux loss 0.01 и router z-loss 0.001 |
| Expanded Residual / GR | 4 latent slots × 128 на токен; read/write-гейты перед/после каждого миксера, финальный merge в residual stream |
| n-gram embedding | хэшированные 2- и 3-граммы, 131 072 бакета на порядок, dim 32, добавляются только на Layer 2 (zero-based `inject_layer: 1`) |
| MTP | 1 модуль предсказания следующего-следующего токена, loss weight 0.1 |
| Vocab | 151 936 (как у Qwen3; токенизатор `Qwen/Qwen3-0.6B` содержит 151 669 базовых токенов) |
| Контекст обучения | 2048 токенов |

### Параметры

```
total parameters:              799,098,117
active parameters/token:       586,761,477
inactive expert params:        212,336,640
token embedding:               233,373,696   (tied с lm_head)
n-gram embedding:                8,437,793
non-embedding:                 557,286,628
active non-embedding:          344,949,988
```

Проверить самостоятельно:

```bash
python -m tools.count_params --config configs/voidik_0.8b.yaml
```

### Отличия от исходной схемы

Скриншот не задаёт точную математику нескольких блоков, поэтому приняты
следующие приближения:

- **QSA** реализован как фиксированный causal block-sparse паттерн: локальное
  окно + sink-токены. Обучаемого выбора блоков нет.
- **GR Read/Write** реализован как gated latent bank на каждый токен. Банк не
  переносит состояние между токенами; это упрощает teacher forcing и
  инкрементальный декодинг. Полностью рекуррентная семантика «GR» со скриншота
  не воспроизводится.
- **n-gram embedding** использует хэширование с коллизиями, а не точные
  словари n-грамм.
- **MTP** имеет глубину 1 (два токена вперёд); конфиг допускает увеличение
  `mtp.depth`.
- `generate()` — референсный декодер без KV-cache, предназначен для
  smoke-проверок, а не для быстрого инференса.

## Установка

Требуется Python 3.10+ и PyTorch с поддержкой CUDA (для обучения).

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux
source .venv/bin/activate

pip install -e .
# зависимости для подготовки данных
pip install -e ".[data]"
# тесты
pip install -e ".[dev]"
```

Проверка окружения:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Подготовка данных

`scripts/prepare_data.py` токенизирует поток документов и пишет packed-шарды
`train_*.bin` / `val_*.bin` плюс `meta.json`. Формат — плоский поток token id
(`uint32` для словаря Qwen), из которого обучение берёт случайные crop'ы.

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

`--val-split` равен `--split`: первые `--val-tokens` документов потока уходят в
validation, а train-шарды начинаются ровно после них — пересечения нет.

20 млрд токенов в `uint32` занимают примерно **80 ГБ** на диске. Для 100 млрд
токенов нужно ~400 ГБ.

### Локальный корпус

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

Затем в конфиге укажите:

```yaml
data:
  train_glob: data/local/train_*.bin
  val_glob: data/local/val_*.bin
```

Другие полезные флаги: `--text-column`, `--min-chars`, `--max-docs`,
`--eos-id`, `--overwrite`. Токенизатор должен иметь `vocab_size` не больше
`model.vocab_size` (151 936), иначе загрузка данных остановится с ошибкой.

### Компактный эксперимент на одной GPU (~59M)

Конфигурация [configs/voidik_58m.yaml](configs/voidik_58m.yaml) сохраняет GDN,
QSA, MoE, n-граммы, residual bank и MTP, но уменьшает словарь до 16 384
токенов. Ниже показан запуск на публичном английском корпусе FineWeb-Edu:
сначала потоково обучается компактный BPE-токенизатор, затем готовятся
непересекающиеся обучающий и валидационный потоки.

```powershell
pip install -e ".[data]"
python scripts/train_tokenizer.py --dataset HuggingFaceFW/fineweb-edu --split train --max-docs 50000 --output data/voidik_58m/tokenizer --vocab-size 16384 --sample-mb 50
python scripts/prepare_data.py --dataset HuggingFaceFW/fineweb-edu --split train --val-split train --tokens 40m --val-tokens 1m --shard-tokens 10m --tokenizer data/voidik_58m/tokenizer --output data/voidik_58m/tokens
python -m tools.count_params --config configs/voidik_58m.yaml
python -m voidik.train --config configs/voidik_58m.yaml
```

Датасет читается потоком, а не скачивается целиком. Перед использованием
проверьте лицензию и условия FineWeb-Edu.

Для RTX 3070 с 8 ГБ памяти заданы batch 1, контекст 256 и накопление
градиента 8. Пилот на 1 000 шагов обработает около 2,05 млн обучающих
токенов. Это проверка цикла обучения и начальной динамики loss, не гарантия
качества языка. По начальному smoke-замеру обучение займёт около 75 минут;
валидация добавит время, а реальная скорость может отличаться.

После обучения сгенерируйте примеры:

```powershell
python scripts/generate.py --config configs/voidik_58m.yaml --checkpoint runs/voidik-58m-1000/checkpoints/latest.pt --tokenizer data/voidik_58m/tokenizer --prompt "Однажды в лесу"
```

Это обучение next-token на обычном тексте, не instruction tuning. Первые
генерации могут быть несвязными; следите за `train_log.jsonl` и
`eval_log.jsonl`, сравнивая один и тот же prompt на разных чекпоинтах.

#### Результаты пилота и проверка inference

Завершён запуск на 1 RTX 3070: **1 000 шагов**, 2 048 000 обучающих токенов.
Validation loss составил **6,21** на шаге 500 и **5,92** на шаге 1 000
(perplexity **371** на шаге 1 000). Это метрики небольшого эксперимента, а не
бенчмарк и не доказательство способности модели хорошо понимать язык.

На prompt `Hello!` с temperature `0.1` модель зациклилась на `||`. С
temperature `0.7` повтор исчез, но генерация осталась в основном
бессвязной:

```text
Hello!
|Pani||Cigb]
- Dicmaka is a result of the number of "pentical� (p. 1)
- "Petu
```

Повторить проверку можно на локальном чекпоинте:

```powershell
python scripts/generate.py --config configs/voidik_58m.yaml --checkpoint runs/voidik-58m-1000/checkpoints/latest.pt --tokenizer data/voidik_58m/tokenizer --prompt "Hello!" --max-new-tokens 40 --temperature 0.7 --top-k 40
```

Токенизатор, шарды датасета и чекпоинт остаются локальными файлами и не
включены в репозиторий.

## Обучение

### Smoke-проверка (без данных, случайные токены)

```bash
python -m voidik.train --config configs/smoke.yaml --toy-data --max-steps 10
```

### Продолжить обучение после прерывания

```bash
python -m voidik.train --config configs/voidik_58m.yaml \
  --resume runs/voidik-58m-1000/checkpoints/latest.pt
```

### Один GPU

```bash
python -m voidik.train --config configs/voidik_0.8b.yaml \
  --output-dir runs/voidik-0.8b-single
```

### DDP, 8 GPU

```bash
torchrun --standalone --nproc_per_node=8 -m voidik.train \
  --config configs/voidik_0.8b.yaml
```

Основной конфиг рассчитан на 8 GPU: batch 1 на GPU × grad accumulation 32 ×
2048 токенов = **524 288 токенов на шаг**. За 40 000 шагов модель увидит
примерно **21 млрд токенов**.

Ключевые гиперпараметры: AdamW, lr 6e-4 с cosine decay до 6e-5, warmup 2000,
weight decay 0.1, gradient clipping 1.0, bf16 autocast, gradient checkpointing.

### Resume

```bash
python -m voidik.train --config configs/voidik_0.8b.yaml \
  --resume runs/voidik-0.8b/checkpoints/latest.pt
```

Чекпоинты лежат в `<output_dir>/checkpoints/` (`latest.pt` и нумерованные
`step_*.pt`, хранятся последние `keep_last_checkpoints`). Логи пишутся в
`<output_dir>/train_log.jsonl` и `eval_log.jsonl`, рядом сохраняется
`config.yaml`.

### Переменные окружения DDP

Скрипт сам определяет rank/world size через `torchrun`; дополнительная
настройка не нужна. Для отладки на CPU можно запустить smoke-конфиг без
`torchrun`, но основной конфиг на CPU практически не обучается.

## Мониторинг и оценка

- `train_log.jsonl` — loss, learning rate, время шага, throughput;
- `eval_log.jsonl` — validation loss (только main head, без MTP);
- чекпоинты содержат model/optimizer/scaler/RNG state, поэтому resume
  продолжает обучение детерминированно;
- TensorBoard-логи в текущей версии не включены, несмотря на флаг
  `train.tensorboard` в конфиге.

## Тесты

```bash
pytest
```

Тесты проверяют форму и причинность QSA, эквивалентность chunk-parallel GDN,
n-gram хэширование, конфиги и forward/backward модели.

## Требования к железу и бюджет

- **Память.** ~0.8B параметров в bf16 + grads + AdamW состояния ≈ 13–16 ГБ на
  реплику, плюс активации. С gradient checkpointing комфортно на 24–40 ГБ;
  рекомендуется 8 × A100 40/80 ГБ или 8 × H100.
- **DDP** реплицирует полный модель на каждый GPU; FSDP не используется.
  Для 0.8B этого достаточно.
- **Время.** 21 млрд токенов на 8 × A100 — ориентировочно 2–5 суток в
  зависимости от throughput и interconnect.
- **Стоимость.** При аренде ~$1–2 за A100-час полный прогон ориентировочно
  $400–2000; 8 × H100 обычно быстрее, но дороже за час. Оценка очень грубая:
  итог зависит от провайдера, загрузки и реального tokens/sec.
- Для заметно более качественной модели рассматривайте 60–100+ млрд токенов;
  бюджет и время растут линейно.

## Структура проекта

```
configs/
  voidik_0.8b.yaml      # основной 0.8B конфиг
  voidik_58m.yaml       # компактный конфиг для одной GPU
  smoke.yaml            # tiny-конфиг для проверки пайплайна
voidik/
  config.py             # dataclass-конфиги + YAML loader
  train.py              # DDP-обучение, eval, checkpoint/resume
  data/dataset.py       # packed memmap dataset
  model/
    model.py            # VoidikForCausalLM, n-gram injection, MTP, generate
    block.py            # HybridBlock: residual bank + mixer + MoE
    gdn.py              # chunk-parallel Gated DeltaNet
    attention.py        # block-sparse QSA
    moe.py              # dropless top-k MoE
    ngram.py            # hashed n-gram embeddings
    residual_bank.py    # Expanded Residual / GR read-write
    mtp.py              # Multi-Token Prediction
scripts/prepare_data.py # токенизация в шарды
tools/count_params.py   # total/active параметры
tests/                  # unit/smoke тесты
```

## Ограничения

- Полное обучение не проверено на реальном масштабе: smoke-прогоны подтверждают
  корректность пайплайна, но не качество финальной модели.
- `generate()` не использует KV-cache и не оптимизирован для инференса.
- Реализация GR и QSA — обоснованные приближения, а не точная копия
  неоднозначной схемы.
- MTP-логиты считаются только в режиме `training`; validation loss — только по
  основной голове.

## Лицензирование

Оригинальный исходный код Voidik в этом репозитории распространяется по
лицензии MIT; см. [LICENSE](LICENSE).

Лицензия MIT распространяется только на материалы проекта, охваченные
лицензией репозитория. Она не меняет лицензии стороннего ПО, датасетов и других
внешних ресурсов. Qwen Community License 1.0 распространяется на ПО,
выпущенное Qwen, а не на независимо написанный код Voidik. В репозитории нет
весов Qwen. Если вы отдельно скачиваете токенизатор `Qwen/Qwen3-0.6B` или
другие внешние ресурсы, проверьте и соблюдайте лицензии и условия, опубликованные
для этих ресурсов. Условия датасетов также применяются отдельно.
