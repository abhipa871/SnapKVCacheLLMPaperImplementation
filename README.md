# SnapKV KV cache (Qwen3.5)

This repo implements **SnapKV** prompt KV compression inside Hugging Face **dense Qwen3.5** models: `SnapKVCache` plus `SnapKVQwen3_5ForCausalLM` in [`modify_qwen.py`](modify_qwen.py). It is **not** a generic drop-in for other architectures.

SnapKV scores attention during prefill, keeps high-importance prefix tokens plus a recent window, and tracks logical positions for left-padded batched generation. The policy follows the SnapKV paper; this code paths it through Qwen3.5’s attention stack and `DynamicCache`.

## Layout

- [`modify_qwen.py`](modify_qwen.py) — `SnapKVCache`, `SnapKVQwen3_5Attention` / `SnapKVQwen3_5ForCausalLM`, mask helpers (`build_snapkv_kv_valid_mask`, `make_snapkv_causal_mask`, `left_pad_logical_positions`).
- [`test_modify_qwen_snapkv.py`](test_modify_qwen_snapkv.py) — SnapKV test suite (padding, cache unit, masks, one-shot prefill integration, optional CUDA).
- [`pytest.ini`](pytest.ini) — markers: `gpu`, `padding`, `snapkv`, `integration`.
- [`snapkv_longbench_qwen35_colab.ipynb`](snapkv_longbench_qwen35_colab.ipynb) — LongBench baseline vs SnapKV eval on `Qwen/Qwen3.5-0.8B`.
- [`longbench_config/`](longbench_config/) — vendored SnapKV LongBench prompt/maxlen JSON.
- [`longbench_metrics.py`](longbench_metrics.py) — vendored LongBench task metrics and `scorer()`.
- [`h2o_gsm8k_eval_colab.ipynb`](h2o_gsm8k_eval_colab.ipynb) — legacy GSM8K accuracy notebook (H2O naming; predates SnapKV rename).

## Requirements

- Python 3, **PyTorch**, **Transformers** (tested with 5.6.x; use a matching or newer version locally).
- **CUDA** optional; GPU-only tests are marked `gpu`.
- For tests: `pytest`.

```bash
pip install -U "transformers>=4.53.0" torch datasets accelerate matplotlib pandas tqdm pytest
```

## Using `SnapKVQwen3_5ForCausalLM` in Python

`modify_qwen.py` reads **`window_size`** and **`max_capacity_prompt`** from the model config (via each attention layer’s `SnapKVCache`). After prefill, the compressed prompt cache holds at most **`max_capacity_prompt`** KV slots (top-scored prefix + trailing **`window_size`** recent tokens).

Set those fields on **both** the top-level `config` and **`config.text_config`** when it exists (same pattern as `_patch_snapkv_budget` in the tests).

Use **`attn_implementation="eager"`** with this stack. For **batched** generation, set **`tokenizer.padding_side = "left"`** (see `SnapKVCache` docstring). Call **`model.reset_snapkv_state()`** when starting a new unrelated sequence (also done automatically in `prepare_inputs_for_generation` when the cache is empty).

```python
import copy
import torch
from transformers import AutoConfig, AutoTokenizer

from modify_qwen import SnapKVQwen3_5ForCausalLM


def patch_snapkv_budget(cfg, max_capacity_prompt: int, window_size: int):
    """Mirror test_modify_qwen_snapkv._patch_snapkv_budget: set on root + text_config."""
    c = copy.deepcopy(cfg)
    for obj in (c, getattr(c, "text_config", None)):
        if obj is None:
            continue
        setattr(obj, "max_capacity_prompt", max_capacity_prompt)
        setattr(obj, "window_size", window_size)
    return c


MODEL_ID = "Qwen/Qwen3.5-0.8B"

base_cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
cfg = patch_snapkv_budget(base_cfg, max_capacity_prompt=512, window_size=64)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.padding_side = "left"

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
device_kw = {} if torch.cuda.is_available() else {"device_map": "cpu"}
if torch.cuda.is_available():
    device_kw["device_map"] = "cuda:0"

model = (
    SnapKVQwen3_5ForCausalLM.from_pretrained(
        MODEL_ID,
        config=cfg,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
        **device_kw,
    )
    .eval()
)

batch = tokenizer("5+7=?", return_tensors="pt")

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch = {k: v.to(dev) for k, v in batch.items()}

with torch.inference_mode():
    ids = model.generate(
        **batch,
        max_new_tokens=24,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )

print(tokenizer.decode(ids[0], skip_special_tokens=True))
```

### Prefill assumption

The full prompt is processed in **one** prefill forward (`past_key_values.get_seq_length() == 0` at the start of that forward). SnapKV scoring and compression run on that pass only. Multi-chunk / `prefill_chunk_size` prefill is not supported by this port.

## LongBench evaluation

[`snapkv_longbench_qwen35_colab.ipynb`](snapkv_longbench_qwen35_colab.ipynb) reproduces the minimal SnapKV [LongBench](https://arxiv.org/abs/2308.14508) pipeline from [pred_snap.py](https://github.com/FasterDecoding/SnapKV/blob/main/experiments/LongBench/pred_snap.py), using `SnapKVQwen3_5ForCausalLM` on `Qwen/Qwen3.5-0.8B`.

1. Open the notebook on a GPU runtime (Colab or local).
2. Ensure the repo root (with `modify_qwen.py`) is on `PYTHONPATH`.
3. For a quick smoke test, set `SMOKE_TEST = True` (2 samples, `qasper` only).
4. Default run: baseline Full-KV vs SnapKV at `max_capacity_prompt` ∈ {1024, 2048, 4096} (`window_size=32`, `kernel_size=7`, `maxpool`) on `qasper`, `hotpotqa`, `multifieldqa_en`.

Artifacts:

- `pred/{run_name}/{dataset}.jsonl` — raw predictions
- `results/{run_name}/result.json` — per-task scores (0–100)
- `results/comparison.csv` — baseline vs SnapKV c1024/c2048/c4096 table

## Tests

From the repo root:

```bash
# All CPU tests
python -m pytest test_modify_qwen_snapkv.py -m "not gpu" -q

# CUDA smoke tests (needs a GPU)
python -m pytest test_modify_qwen_snapkv.py -m gpu -q
```

### Test sections (`test_modify_qwen_snapkv.py`)

| Marker | Coverage |
|--------|----------|
| `padding` | `left_pad_logical_positions`, `build_snapkv_kv_valid_mask`, causal masks |
| `snapkv` | `SnapKVCache` lifecycle, compression, position tracking, attention mask paths |
| `integration` | Mini-model one-shot prefill and left-pad eviction |
| `gpu` | `Qwen/Qwen3.5-0.8B` generate, long prefill, padded decode, dynamic cache |

Filter examples:

```bash
python -m pytest test_modify_qwen_snapkv.py -m "snapkv and not gpu" -q
```

## Troubleshooting

- **`unexpected keyword argument 'padding_attention_mask'`** — Qwen decoder layers must thread `padding_attention_mask` through to `self_attn` (see notebook sanity checks if using the Colab eval).
- **KV length / `position_ids` mismatches** — compare `snap_kv_cache.position_ids` to `key_states.shape[2]` and `_build_additive_attention_mask` in [`modify_qwen.py`](modify_qwen.py).
- **Left-pad batches** — invalid padding slots must not receive valid SnapKV scores; see `build_snapkv_kv_valid_mask` and GPU `test_gpu_greedy_padded_decode_position_ids_stable`.
- **OOM on long prompts** — process the full prompt in one forward; use a smaller `max_capacity_prompt` SnapKV budget or cap input length (see LongBench notebook notes).

## Reference

Implements compression behavior in the spirit of **SnapKV** (observation-window scoring + prefix top-k retention) for efficient long-context inference.
