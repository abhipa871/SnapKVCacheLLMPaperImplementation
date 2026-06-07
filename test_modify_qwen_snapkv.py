"""
CPU: python -m pytest test_modify_qwen_snapkv.py -m "not gpu" -q
GPU: python -m pytest test_modify_qwen_snapkv.py -m gpu -q

SnapKV / padding / chunked-prefill tests for ``modify_qwen``.
"""

import copy
import math
import sys
from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]
import torch

_D = Path(__file__).resolve().parent
if str(_D) not in sys.path:
    sys.path.insert(0, str(_D))

from modify_qwen import (  # noqa: E402
    SnapKVCache,
    SnapKVQwen3_5Attention,
    SnapKVQwen3_5ForCausalLM,
    SnapKVQwen3_5TextModel,
    build_snapkv_kv_valid_mask,
    dense_kv_logical_positions,
    left_pad_logical_positions,
    make_snapkv_causal_mask,
    prefill_prompt_complete,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class KVPast:
    """Tuple past substitute for ``SnapKVCache.__call__`` unit tests."""

    __slots__ = ("snap_next_position", "_k", "_v")

    def __init__(self, k: torch.Tensor, v: torch.Tensor):
        self.snap_next_position = None
        self._k = k
        self._v = v

    def __getitem__(self, i: int):
        return self._k if i == 0 else self._v


class _FakePast:
    """Sentinel so ``past_key_values is not None`` in mask-builder tests."""


def _mini_snap_cfg(
    *,
    window_size: int = 2,
    max_capacity_prompt: int = 6,
    num_hidden_layers: int = 1,
) -> "object":
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    return Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        layer_types=["full_attention"] * num_hidden_layers,
        pad_token_id=0,
        window_size=window_size,
        max_capacity_prompt=max_capacity_prompt,
    )


_MiniSnapCfg = _mini_snap_cfg()


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: CUDA + HF Qwen3.5-0.8B")
    config.addinivalue_line("markers", "padding: mask and logical position helpers")
    config.addinivalue_line("markers", "snapkv: SnapKVCache and attention mask paths")
    config.addinivalue_line("markers", "chunk_prefill: multi-forward prefill / gate tests")
    config.addinivalue_line("markers", "integration: mini-model forwards")


@pytest.fixture(scope="session")
def qwen_cfg():
    from transformers import AutoConfig

    return AutoConfig.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)


@pytest.fixture
def snap_attn():
    return SnapKVQwen3_5Attention(_mini_snap_cfg(), layer_idx=0).eval()


@pytest.fixture
def mini_text_model():
    model = SnapKVQwen3_5TextModel(_mini_snap_cfg(window_size=2, max_capacity_prompt=4)).eval()
    torch.manual_seed(0)
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, 0.02)
    return model


def _ninf(dtype):
    return torch.finfo(dtype).min


def _patch_snapkv_budget(cfg, max_capacity_prompt: int, window_size: int):
    c = copy.deepcopy(cfg)
    for obj in filter(None, (c, getattr(c, "text_config", None))):
        setattr(obj, "max_capacity_prompt", max_capacity_prompt)
        setattr(obj, "window_size", window_size)
    return c


def _snapkv_attn_weights(
    batch: int,
    num_attn_heads: int,
    window_size: int,
    seq: int,
    spike_indices: list[int] | None = None,
) -> torch.Tensor:
    attn = torch.zeros(batch, num_attn_heads, window_size, seq)
    if spike_indices:
        for idx in spike_indices:
            attn[..., idx] = 1000.0 + float(idx)
    return attn


def _snapkv_left_pad_attn_b2(seq: int = 30) -> torch.Tensor:
    attn = torch.zeros(2, seq, dtype=torch.bool)
    attn[0, 15:] = True
    attn[1, 10:] = True
    return attn


def _contract_b_chunk_columns(attn_mask: torch.Tensor, tokens_per_chunk: int) -> list[list[int]]:
    """Column index lists per chunk (row-0 real columns; tests use aligned real spans)."""

    cols = attn_mask[0].nonzero(as_tuple=True)[0].tolist()
    return [cols[i : i + tokens_per_chunk] for i in range(0, len(cols), tokens_per_chunk)]


def _contract_b_chunk_columns_for_row(
    attn_mask: torch.Tensor,
    row_idx: int,
    tokens_per_chunk: int,
) -> list[list[int]]:
    """Column index lists per chunk using one batch row's real (non-pad) columns.

    Chunks are rectangular across the batch: shorter rows may see pad/dummy columns in
    a chunk (not true per-row Contract B). Use only when testing batch-level gates.
    """

    cols = attn_mask[row_idx].nonzero(as_tuple=True)[0].tolist()
    return [cols[i : i + tokens_per_chunk] for i in range(0, len(cols), tokens_per_chunk)]


def _run_chunked_prefill(
    model: SnapKVQwen3_5TextModel,
    input_ids_full: torch.Tensor,
    attn_mask: torch.Tensor,
    chunk_columns: list[list[int]],
):
    pst = None
    out = None
    for cols in chunk_columns:
        chunk_ids = input_ids_full[:, cols]
        out = model(
            input_ids=chunk_ids,
            attention_mask=attn_mask,
            past_key_values=pst,
            use_cache=True,
        )
        pst = out.past_key_values
    return out


def _run_hf_style_chunked_prefill(
    model: SnapKVQwen3_5TextModel,
    input_ids_full: torch.Tensor,
    attention_mask_full: torch.Tensor,
    chunk_size: int,
):
    """HF prefill_chunk_size emulation: prefix-sliced masks + explicit flags per chunk."""

    pst = None
    out = None
    L = input_ids_full.shape[-1]

    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)

        chunk_ids = input_ids_full[:, start:end]
        prefix_mask = attention_mask_full[:, :end]
        final = end >= L

        out = model(
            input_ids=chunk_ids,
            attention_mask=prefix_mask,
            past_key_values=pst,
            use_cache=True,
            hf_chunked_prefill=True,
            prefill_final_chunk=final,
        )
        pst = out.past_key_values

        if not final:
            assert not getattr(pst, "snap_prefill_complete", False)
            sa = _first_snapkv_self_attn_from_text(model)
            assert sa.snap_kv_cache.snap_score is None

    return out


def _first_snapkv_self_attn_layer(model):
    lm = model.model.language_model
    for lyr in lm.layers:
        sa = getattr(lyr, "self_attn", None)
        if sa is not None and sa.__class__.__name__ == "SnapKVQwen3_5Attention":
            return lyr
    raise AssertionError("no SnapKVQwen3_5Attention layer found")


def _first_snapkv_self_attn_from_text(model: SnapKVQwen3_5TextModel):
    for lyr in model.layers:
        sa = getattr(lyr, "self_attn", None)
        if sa is not None and isinstance(sa, SnapKVQwen3_5Attention):
            return sa
    raise AssertionError("no SnapKVQwen3_5Attention in text model")


# ---------------------------------------------------------------------------
# A. Padding / position helpers
# ---------------------------------------------------------------------------


@pytest.mark.padding
def test_left_pad_logical_positions_examples():
    mask = torch.tensor(
        [
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    got = left_pad_logical_positions(mask)
    assert got[0].tolist() == [0, 0, 0, 1, 2]
    assert got[1].tolist() == [0, 0, 1, 2, 3]
    assert got[2].tolist() == [0, 1, 2, 3, 4]


@pytest.mark.padding
def test_left_pad_logical_positions_past_seen_tensor():
    mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]], dtype=torch.bool)
    past_seen = torch.tensor([10, 20], dtype=torch.long)
    got = left_pad_logical_positions(mask, past_seen)
    assert got[0].tolist() == [10, 10, 10, 11]
    assert got[1].tolist() == [20, 20, 21, 22]


@pytest.mark.padding
def test_dense_kv_logical_positions_rebuilds_arange():
    tail = torch.tensor([[2, 3]], dtype=torch.long)
    got = dense_kv_logical_positions(4, 2, tail, device=torch.device("cpu"))
    assert got[0].tolist() == [0, 1, 2, 3]


@pytest.mark.padding
def test_kv_valid_inside_bounds_reflects_gathered_padding_row():
    B, nk, L = 1, 2, 8
    attn = torch.zeros(B, L, dtype=torch.bool)
    attn[:, 6:] = True
    kv_pid = torch.tensor([[4, 5, 6, 7]], dtype=torch.long).view(B, 1, -1).expand(B, nk, -1)
    vm = build_snapkv_kv_valid_mask(kv_pid, attn)
    assert (~vm[..., :2]).all() and vm[..., 2:].all()


@pytest.mark.padding
def test_kv_valid_oob_assumes_generation_valid():
    B, nk, L = 2, 1, 6
    attn = torch.ones(B, L, dtype=torch.bool)
    kv_pid = torch.tensor([[[L - 1, L + 100]], [[0, L + 999]]])
    vm = build_snapkv_kv_valid_mask(kv_pid, attn)
    assert vm[:, :, 1].all()


@pytest.mark.padding
def test_causal_masks_where_kv_strictly_after_any_query_blocked():
    B, nk, _, kvlen = 1, 2, 2, 5
    q_arange = torch.tensor([[9, 11]], dtype=torch.long)
    kv_arange = torch.tensor([[[8, 9, 10, 11, 12]]], dtype=torch.long).expand(B, nk, kvlen)
    nh = nk * 6
    dtype = torch.float32
    m = make_snapkv_causal_mask(
        q_arange,
        kv_arange,
        torch.ones(B, 256, dtype=torch.bool),
        num_attention_heads=nh,
        dtype=dtype,
        device=torch.device("cpu"),
    )
    causal = kv_arange[:, :, None, :].repeat_interleave(nh // nk, dim=1) <= q_arange[:, None, :, None]
    forbid = torch.where(~causal, m, torch.zeros_like(m))
    assert torch.all(forbid[~causal] == _ninf(dtype))


@pytest.mark.padding
def test_causal_plus_padding_blocked():
    B, nk, _, kl = 2, 2, 1, 5
    L = 96
    attn = torch.zeros(B, L, dtype=torch.bool)
    attn[0, 90:] = True
    attn[1, 88:] = True
    q_arange = torch.tensor([[92], [91]], dtype=torch.long)
    rows = [
        torch.arange(82, 82 + kl, dtype=torch.long).view(1, 1, -1),
        torch.arange(80, 80 + kl, dtype=torch.long).view(1, 1, -1),
    ]
    kv_arange = torch.cat(rows, dim=0).expand(-1, nk, -1)
    nh = 8
    dtype = torch.float64
    m = make_snapkv_causal_mask(
        q_arange,
        kv_arange,
        attn,
        num_attention_heads=nh,
        dtype=dtype,
        device=torch.device("cpu"),
    )
    vm = build_snapkv_kv_valid_mask(kv_arange, attn).repeat_interleave(nh // nk, dim=1)
    causal = kv_arange[:, :, None, :].repeat_interleave(nh // nk, dim=1) <= q_arange[:, None, :, None]
    allowed = causal & vm[:, :, None, :]
    assert torch.all(torch.where(~allowed, m, torch.zeros_like(m)) == _ninf(dtype))


# ---------------------------------------------------------------------------
# B. Prefill gate
# ---------------------------------------------------------------------------


@pytest.mark.chunk_prefill
def test_prefill_prompt_complete_false_mid_chunk():
    attn = torch.tensor([[0, 0, 1, 1, 1, 1]], dtype=torch.bool)
    assert not prefill_prompt_complete(attn, past_len=0, chunk_seq_len=2)


@pytest.mark.chunk_prefill
def test_prefill_prompt_complete_true_final_chunk():
    attn = torch.tensor([[0, 0, 1, 1, 1, 1]], dtype=torch.bool)
    assert prefill_prompt_complete(attn, past_len=2, chunk_seq_len=2)


@pytest.mark.chunk_prefill
def test_prefill_prompt_complete_left_pad_uses_real_token_count():
    attn = torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.bool)
    assert not prefill_prompt_complete(attn, past_len=0, chunk_seq_len=2)
    assert prefill_prompt_complete(attn, past_len=1, chunk_seq_len=2)


@pytest.mark.chunk_prefill
def test_prefill_prompt_complete_heterogeneous_waits_for_longest_row():
    """Batch gate uses ``.all()``: shorter rows finishing early must not mark prefill complete."""

    attn = torch.zeros(2, 10, dtype=torch.bool)
    attn[0, 4:] = True  # 6 real tokens
    attn[1, 3:] = True  # 7 real tokens (longest row)

    assert int(attn.sum(dim=-1)[0]) == 6
    assert int(attn.sum(dim=-1)[1]) == 7

    # Two chunks of 3: cumulative 6 covers row 0 but not row 1.
    assert not prefill_prompt_complete(attn, past_len=3, chunk_seq_len=3)
    assert prefill_prompt_complete(attn[0:1], past_len=3, chunk_seq_len=3)
    assert not prefill_prompt_complete(attn[1:2], past_len=3, chunk_seq_len=3)

    # One more real token in the final chunk completes the longest row (and batch).
    assert prefill_prompt_complete(attn, past_len=6, chunk_seq_len=1)


# ---------------------------------------------------------------------------
# C. SnapKVCache unit
# ---------------------------------------------------------------------------


@pytest.mark.snapkv
def test_score_gqa_group_sum():
    nk, grp, qt, ks = 2, 2, 2, 17
    win = 2
    c = SnapKVCache(
        window_size=win,
        max_capacity_prompt=8,
        layer_idx=0,
        num_attention_heads=nk * grp,
        num_key_value_heads=nk,
        num_key_value_groups=grp,
    )
    z = torch.randn(1, nk * grp, qt, ks)
    c._update_snap_score(z)
    gold = z.view(1, nk, grp, qt, ks)[..., -win:, :-win].sum(dim=(2, 3))
    assert torch.allclose(c.snap_score, gold)


@pytest.mark.snapkv
def test_snap_score_raises_on_second_update():
    c = SnapKVCache(window_size=2, max_capacity_prompt=8, layer_idx=0)
    z = torch.randn(1, 2, 2, 5)
    c._update_snap_score(z)
    with pytest.raises(ValueError, match="snap_score is already set"):
        c._update_snap_score(z)


@pytest.mark.snapkv
def test_snapkv_append_positions_left_pad_contract_a():
    B, nk, seq, win, nh = 1, 2, 5, 2, 4
    attn = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.bool)
    expected = left_pad_logical_positions(attn)
    cache = SnapKVCache(
        window_size=win,
        max_capacity_prompt=8,
        layer_idx=0,
        num_attention_heads=nh,
        num_key_value_heads=nk,
        num_key_value_groups=nh // nk,
    )
    weights = torch.ones(B, nh, win, seq)
    cache._update_snap_score(weights)
    cache._append_positions(weights, padding_attention_mask=attn)
    assert cache.position_ids is not None
    assert cache.position_ids[0, 0].tolist() == expected[0].tolist()


@pytest.mark.snapkv
def test_snapkv_append_positions_dense_chunk_contract_b():
    B, nk, kv_len, q_len, win, nh = 1, 2, 4, 2, 2, 4
    cache = SnapKVCache(
        window_size=win,
        max_capacity_prompt=8,
        layer_idx=0,
        num_attention_heads=nh,
        num_key_value_heads=nk,
        num_key_value_groups=nh // nk,
    )
    weights = torch.ones(B, nh, q_len, kv_len)
    cache._update_snap_score(weights)
    chunk_pos = torch.tensor([[2, 3]], dtype=torch.long)
    cache._append_positions(weights, text_position_ids=chunk_pos)
    want = dense_kv_logical_positions(kv_len, q_len, chunk_pos, device=weights.device)
    assert cache.position_ids[0, 0].tolist() == want[0].tolist()


@pytest.mark.snapkv
def test_append_new_positions_with_text_ids():
    B, nk = 1, 2
    cache = SnapKVCache(window_size=2, max_capacity_prompt=8, num_key_value_heads=nk)
    cache.position_ids = torch.tensor([[[48, 49]]], dtype=torch.long).expand(B, nk, 2)
    cache.append_new_positions(torch.tensor([[50, 51, 52]], dtype=torch.long), target_kv_len=5)
    assert cache.position_ids.shape[-1] == 5
    assert cache.position_ids[0, 0, -2:].tolist() == [51, 52]


@pytest.mark.snapkv
def test_append_new_positions_monotonic_fallback():
    B, nk = 1, 2
    cache = SnapKVCache(window_size=2, max_capacity_prompt=8, num_key_value_heads=nk)
    cache.position_ids = torch.tensor([[[10, 11]]], dtype=torch.long).expand(B, nk, 2)
    cache.append_new_positions(None, target_kv_len=5)
    assert cache.position_ids[0, 0].tolist() == [10, 11, 12, 13, 14]


def _snapkv_budget_same_setup_batched(attn, nk, seq, Dh, max_cap, win, num_heads=8, num_kv_groups=8):
    B = attn.shape[0]
    keys = torch.arange(seq, dtype=torch.float32).view(1, 1, seq, 1).expand(B, nk, seq, Dh)
    past = KVPast(keys.clone(), keys.clone())
    cache = SnapKVCache(
        window_size=win,
        max_capacity_prompt=max_cap,
        kernel_size=1,
        layer_idx=0,
        num_attention_heads=num_heads,
        num_key_value_heads=nk,
        num_key_value_groups=num_kv_groups,
    )
    return cache, past


@pytest.mark.snapkv
def test_snapkv_compression_recent_and_topk():
    B, nh_kv, Dh = 1, 2, 6
    seq, max_cap, win = 13, 6, 4
    keys = torch.empty(B, nh_kv, seq, Dh)
    for t in range(seq):
        keys[:, :, t] = float(t)
    cache = SnapKVCache(
        window_size=win,
        max_capacity_prompt=max_cap,
        kernel_size=1,
        layer_idx=0,
        num_attention_heads=8,
        num_key_value_heads=nh_kv,
        num_key_value_groups=4,
    )
    ph = seq - win
    past = KVPast(keys.clone(), keys.clone())
    attn = _snapkv_attn_weights(B, 8, win, seq, spike_indices=[ph - 3, ph - 1])
    out = cache(past, attn)
    new_k = out[0].clone()
    assert new_k.shape[2] == max_cap
    want = sorted([ph - 3, ph - 1] + list(range(ph, seq)))
    assert sorted(new_k[0, 0, :, 0].round().long().tolist()) == want


@pytest.mark.snapkv
def test_snapkv_prefers_valid_prefix_over_padding():
    B, nk, seq, Dh = 1, 1, 12, 4
    prefix_keep, win = 3, 2
    max_cap = prefix_keep + win
    attn_mask = torch.zeros(B, seq, dtype=torch.bool)
    attn_mask[:, 6:] = True
    pref = seq - win
    keys = torch.arange(seq, dtype=torch.float32).reshape(1, 1, seq, 1).expand(B, nk, seq, Dh)
    past = KVPast(keys.clone(), keys.clone())
    cache = SnapKVCache(
        window_size=win,
        max_capacity_prompt=max_cap,
        kernel_size=1,
        layer_idx=0,
        num_attention_heads=8,
        num_key_value_heads=nk,
        num_key_value_groups=8,
    )
    weights = torch.zeros(B, 8, win, seq)
    weights[..., :] = torch.arange(seq, dtype=torch.float).reshape(1, 1, 1, seq)
    weights[..., :, :6] -= 4444
    weights[..., :, 7] += 9900
    weights[..., :, 8] += 9800
    weights[..., :, 9] += 9777
    out = cache(past, weights, padding_attention_mask=attn_mask)[0]
    kept = sorted(out[0, 0, :, 0].round().long().tolist())
    assert kept[-win:] == list(range(pref, seq))
    prefix_set = set(kept[:prefix_keep])
    assert prefix_set.issubset({7, 8, 9}) and prefix_set.issuperset({9})


@pytest.mark.snapkv
def test_snapkv_budget_uses_valid_counts_not_physical_seq_batch_two_rows():
    B, nk, seq, Dh = 2, 1, 14, 2
    prefix_keep, win = 4, 3
    max_cap = prefix_keep + win
    attn = torch.zeros(B, seq, dtype=torch.bool)
    for b in range(B):
        attn[b, seq - 7 :] = True
    cache, past = _snapkv_budget_same_setup_batched(attn, nk, seq, Dh, max_cap, win)
    weights = torch.ones(B, 8, win, seq)
    out_kept = cache(past, weights, padding_attention_mask=attn)[0]
    assert out_kept.shape[2] == seq
    naive_cache, naive_past = _snapkv_budget_same_setup_batched(attn, nk, seq, Dh, max_cap, win)
    naive_k = naive_cache(naive_past, weights, padding_attention_mask=None)[0]
    assert naive_k.shape[2] == max_cap


@pytest.mark.snapkv
def test_snapkv_batch_masked_padding_never_kept_in_prefix():
    B, nk, seq, Dh = 2, 1, 30, 2
    prefix_keep, win = 5, 10
    max_cap = prefix_keep + win
    attn = _snapkv_left_pad_attn_b2(seq)
    prefix_hi = seq - win
    cache, past = _snapkv_budget_same_setup_batched(attn, nk, seq, Dh, max_cap, win)
    big = torch.finfo(torch.float32).max / 4096
    weights = torch.zeros(B, 8, win, seq)
    for b in range(B):
        for t in range(prefix_hi):
            if not bool(attn[b, t]):
                weights[b, :, :, t] = big * (prefix_hi - t + 1)
            else:
                weights[b, :, :, t] = float(t) * 0.003 + float(b)
    out_k = cache(past, weights, padding_attention_mask=attn)[0]
    want_recent = tuple(range(seq - win, seq))
    for b in range(B):
        row = out_k[b, 0].clone()
        for k in range(prefix_keep):
            slot_idx = int(round(float(row[k, 0])))
            assert bool(attn[b, slot_idx]), f"SnapKV kept padding slot idx={slot_idx} for batch={b}"
        recent_part = tuple(int(round(float(row[k, 0]))) for k in range(prefix_keep, max_cap))
        assert recent_part == want_recent


@pytest.mark.snapkv
def test_snapkv_heterogeneous_batch_distinct_logical_positions():
    B, nk, seq, win, nh = 2, 1, 5, 2, 4
    attn = torch.zeros(B, seq, dtype=torch.bool)
    attn[0, 2:] = True
    attn[1, 1:] = True
    text_pos = left_pad_logical_positions(attn)
    keys = torch.arange(seq, dtype=torch.float32).view(1, 1, seq, 1).expand(B, nk, seq, 1)
    past = KVPast(keys.clone(), keys.clone())
    cache = SnapKVCache(
        window_size=win,
        max_capacity_prompt=seq,
        kernel_size=1,
        layer_idx=0,
        num_attention_heads=nh,
        num_key_value_heads=nk,
        num_key_value_groups=nh // nk,
    )
    weights = torch.ones(B, nh, win, seq)
    cache(past, weights, padding_attention_mask=attn, text_position_ids=text_pos)
    assert not torch.equal(cache.position_ids[0, 0], cache.position_ids[1, 0])


# ---------------------------------------------------------------------------
# D. Attention mask builder
# ---------------------------------------------------------------------------


@pytest.mark.snapkv
def test_snap_scoring_pass_kv_arange_no_raise(snap_attn):
    attn = snap_attn
    B, nh, nk = 1, _MiniSnapCfg.num_attention_heads, _MiniSnapCfg.num_key_value_heads
    window = attn.snap_kv_cache.window_size
    kv_len, q_len = 8, window
    hd = attn.head_dim
    q = torch.zeros(B, nh, q_len, hd)
    k = torch.zeros(B, nk, kv_len, hd)
    text_pos = torch.arange(q_len, dtype=torch.long).view(1, -1)
    mask = attn._build_additive_attention_mask(
        q, k, text_pos, None, _FakePast(), snap_scoring_pass=True
    )
    assert mask.shape == (B, nh, q_len, kv_len)


@pytest.mark.snapkv
def test_snap_scoring_pass_left_pad_blocks_physical_pad_columns(snap_attn):
    attn = snap_attn
    B, nh, nk = 1, _MiniSnapCfg.num_attention_heads, _MiniSnapCfg.num_key_value_heads
    window = attn.snap_kv_cache.window_size
    seq = 6
    hd = attn.head_dim
    pad_mask = torch.zeros(B, seq, dtype=torch.bool)
    pad_mask[0, 2:] = True
    q = torch.zeros(B, nh, window, hd)
    k = torch.zeros(B, nk, seq, hd)
    text_pos = left_pad_logical_positions(pad_mask)[:, -window:]
    mask = attn._build_additive_attention_mask(
        q, k, text_pos, pad_mask, _FakePast(), snap_scoring_pass=True
    )
    assert mask[0, 0, 0, 0] == _ninf(q.dtype)
    assert mask[0, 0, 0, 1] == _ninf(q.dtype)


@pytest.mark.snapkv
def test_main_prefill_left_pad_logical_q_kv_causal(snap_attn):
    attn = snap_attn
    B, nh, nk = 1, _MiniSnapCfg.num_attention_heads, _MiniSnapCfg.num_key_value_heads
    seq = 5
    hd = attn.head_dim
    pad_mask = torch.zeros(B, seq, dtype=torch.bool)
    pad_mask[0, 2:] = True
    q_pos = left_pad_logical_positions(pad_mask)
    q = torch.zeros(B, nh, seq, hd)
    k = torch.zeros(B, nk, seq, hd)
    mask = attn._build_additive_attention_mask(q, k, q_pos, pad_mask, _FakePast())
    assert mask[0, 0, 2, 0] == _ninf(q.dtype)
    assert mask[0, 0, 2, 1] == _ninf(q.dtype)
    assert mask[0, 0, 4, 4] == 0.0


@pytest.mark.snapkv
def test_dense_real_kv_no_gather_over_full_mask(snap_attn):
    attn = snap_attn
    B, nh, nk = 1, _MiniSnapCfg.num_attention_heads, _MiniSnapCfg.num_key_value_heads
    kv_len, q_len = 2, 2
    hd = attn.head_dim
    full_mask = torch.ones(B, 6, dtype=torch.bool)
    q = torch.zeros(B, nh, q_len, hd)
    k = torch.zeros(B, nk, kv_len, hd)
    text_pos = torch.tensor([[0, 1]], dtype=torch.long)
    mask = attn._build_additive_attention_mask(q, k, text_pos, full_mask, _FakePast())
    assert mask.shape == (B, nh, q_len, kv_len)
    assert mask[0, 0, 1, 0] == 0.0


@pytest.mark.snapkv
def test_decode_kv_arange_cat_prev_and_cur(snap_attn):
    attn = snap_attn
    B, nh, nk = 1, _MiniSnapCfg.num_attention_heads, _MiniSnapCfg.num_key_value_heads
    hd = attn.head_dim
    prev_len, q_len = 3, 1
    kv_len = prev_len + q_len
    attn.snap_kv_cache.position_ids = torch.tensor([[[0, 1, 2]]], dtype=torch.long).expand(
        B, nk, prev_len
    )
    q = torch.zeros(B, nh, q_len, hd)
    k = torch.zeros(B, nk, kv_len, hd)
    text_pos = torch.tensor([[3]], dtype=torch.long)
    mask = attn._build_additive_attention_mask(q, k, text_pos, None, _FakePast())
    assert mask.shape == (B, nh, q_len, kv_len)
    assert mask[0, 0, 0, kv_len - 1] == 0.0


def test_attn_scaling_matches_head_dim_inverse_sqrt(qwen_cfg):
    txt = getattr(qwen_cfg, "text_config", None) or qwen_cfg
    att = SnapKVQwen3_5Attention(copy.deepcopy(txt), layer_idx=0)
    assert math.isclose(att.scaling, att.head_dim**-0.5)


def test_eager_attn_matches_scaled_softmax(qwen_cfg):
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        eager_attention_forward,
        repeat_kv,
    )

    txt = getattr(qwen_cfg, "text_config", None) or qwen_cfg
    cfg = copy.deepcopy(txt)
    attn = SnapKVQwen3_5Attention(cfg, layer_idx=0).float().eval()
    torch.manual_seed(11)
    for p in attn.parameters():
        torch.nn.init.normal_(p, 0.02)
    bs, nt = 1, cfg.num_attention_heads
    hd = attn.head_dim
    nl = 5
    q = torch.randn(bs, nt, nl, hd, dtype=torch.float32)
    k = torch.randn(bs, cfg.num_key_value_heads, nl, hd, dtype=torch.float32)
    v = torch.randn(bs, cfg.num_key_value_heads, nl, hd, dtype=torch.float32)
    addm = torch.zeros(bs, nt, nl, nl)
    yo, _ = eager_attention_forward(attn, q, k, v, addm, attn.scaling, dropout=0.0)
    kxp = repeat_kv(k, attn.num_key_value_groups)
    vxp = repeat_kv(v, attn.num_key_value_groups)
    sc = torch.matmul(q, kxp.transpose(-2, -1)).float().mul(attn.scaling)
    pr = torch.softmax(sc + addm, dim=-1, dtype=torch.float32).to(q.dtype)
    yref = torch.matmul(pr, vxp).transpose(1, 2).contiguous()
    assert torch.max(torch.abs(yo.float() - yref.float())).item() < 3e-2


# ---------------------------------------------------------------------------
# E/F. Chunked prefill integration
# ---------------------------------------------------------------------------


@pytest.mark.chunk_prefill
@pytest.mark.integration
def test_chunked_intermediate_forward_no_snap_compress(mini_text_model):
    B, L = 1, 6
    attn = torch.zeros(B, L, dtype=torch.bool)
    attn[:, 2:] = True
    full_ids = torch.arange(10, 10 + L).view(1, -1)
    chunks = _contract_b_chunk_columns(attn, tokens_per_chunk=2)
    out1 = mini_text_model(
        input_ids=full_ids[:, chunks[0]],
        attention_mask=attn,
        use_cache=True,
    )
    pst = out1.past_key_values
    assert pst is not None
    assert not getattr(pst, "snap_prefill_complete", False)
    assert pst.get_seq_length() == 2
    sa = _first_snapkv_self_attn_from_text(mini_text_model)
    assert sa.snap_kv_cache.snap_score is None


@pytest.mark.chunk_prefill
@pytest.mark.integration
def test_chunked_final_forward_triggers_snap():
    model = SnapKVQwen3_5TextModel(_mini_snap_cfg(window_size=2, max_capacity_prompt=4)).eval()
    torch.manual_seed(1)
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, 0.02)
    B, L = 1, 8
    attn = torch.ones(B, L, dtype=torch.bool)
    full_ids = torch.arange(20, 20 + L).view(1, -1)
    chunks = _contract_b_chunk_columns(attn, tokens_per_chunk=2)
    out = _run_chunked_prefill(model, full_ids, attn, chunks)
    pst = out.past_key_values
    assert getattr(pst, "snap_prefill_complete", False)
    sa = _first_snapkv_self_attn_from_text(model)
    assert sa.snap_kv_cache.snap_score is not None
    cap = sa.snap_kv_cache.max_capacity_prompt
    li = sa.layer_idx
    assert pst.layers[li].keys.shape[2] <= cap


@pytest.mark.chunk_prefill
@pytest.mark.integration
def test_chunked_left_pad_dense_kv_positions(mini_text_model):
    B, L = 1, 6
    attn = torch.zeros(B, L, dtype=torch.bool)
    attn[:, 2:] = True
    full_ids = torch.arange(30, 30 + L).view(1, -1)
    chunks = _contract_b_chunk_columns(attn, tokens_per_chunk=2)
    out = _run_chunked_prefill(mini_text_model, full_ids, attn, chunks)
    sa = _first_snapkv_self_attn_from_text(mini_text_model)
    pid = sa.snap_kv_cache.position_ids
    assert pid is not None
    real_count = int(attn.sum().item())
    assert pid.shape[-1] <= sa.snap_kv_cache.max_capacity_prompt
    assert pid[0, 0].max().item() == real_count - 1


@pytest.mark.chunk_prefill
@pytest.mark.snapkv
@pytest.mark.integration
def test_chunked_left_pad_snapkv_keeps_only_valid_prefix_slots():
    """Homogeneous left-pad batch: both rows share the same real-token count and chunk geometry."""

    model = SnapKVQwen3_5TextModel(_mini_snap_cfg(window_size=2, max_capacity_prompt=4)).eval()
    torch.manual_seed(2)
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, 0.02)
    B, L = 2, 10
    attn = torch.zeros(B, L, dtype=torch.bool)
    attn[0, 4:] = True
    attn[1, 4:] = True
    full_ids = torch.stack(
        [
            torch.arange(10, 10 + L),
            torch.arange(20, 20 + L),
        ]
    )
    chunks = _contract_b_chunk_columns(attn, tokens_per_chunk=3)
    out = _run_chunked_prefill(model, full_ids, attn, chunks)
    sa = _first_snapkv_self_attn_from_text(model)
    li = sa.layer_idx
    k = out.past_key_values.layers[li].keys
    assert k is not None
    assert k.shape[2] <= sa.snap_kv_cache.max_capacity_prompt


@pytest.mark.chunk_prefill
@pytest.mark.snapkv
@pytest.mark.integration
def test_chunked_heterogeneous_left_pad_snapkv_deferred_until_longest_row():
    """Deferred batch-level SnapKV gating for heterogeneous left-pad prefill.

    Chunks follow the longest row (``row_idx=1``), so the first chunk includes column 3,
    which is padding for row 0. This does **not** prove per-row Contract B (real-token-only
    chunks for every row); it only proves SnapKV stays off until ``prefill_prompt_complete``
    sees all rows done under the batch-level ``.all()`` gate.
    """

    model = SnapKVQwen3_5TextModel(_mini_snap_cfg(window_size=2, max_capacity_prompt=16)).eval()
    torch.manual_seed(3)
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, 0.02)

    B, L = 2, 10
    attn = torch.zeros(B, L, dtype=torch.bool)
    attn[0, 4:] = True  # 6 real tokens
    attn[1, 3:] = True  # 7 real tokens (longest)
    full_ids = torch.stack(
        [
            torch.arange(10, 10 + L),
            torch.arange(20, 20 + L),
        ]
    )
    chunks = _contract_b_chunk_columns_for_row(attn, row_idx=1, tokens_per_chunk=3)
    assert chunks == [[3, 4, 5], [6, 7, 8], [9]]

    out_mid = _run_chunked_prefill(model, full_ids, attn, chunks[:2])
    pst_mid = out_mid.past_key_values
    sa = _first_snapkv_self_attn_from_text(model)

    assert pst_mid.get_seq_length() == 6
    assert not prefill_prompt_complete(attn, past_len=3, chunk_seq_len=3)
    assert not getattr(pst_mid, "snap_prefill_complete", False)
    assert not getattr(pst_mid, "last_chunk_processed", False)
    assert sa.snap_kv_cache.snap_score is None

    out_final = model(
        input_ids=full_ids[:, chunks[2]],
        attention_mask=attn,
        past_key_values=pst_mid,
        use_cache=True,
    )
    pst_final = out_final.past_key_values

    assert prefill_prompt_complete(attn, past_len=6, chunk_seq_len=1)
    assert getattr(pst_final, "snap_prefill_complete", False)
    assert sa.snap_kv_cache.snap_score is not None


@pytest.mark.chunk_prefill
def test_prepare_inputs_for_generation_preserves_hf_prefill_flags(qwen_cfg):
    cfg = _patch_snapkv_budget(qwen_cfg, max_capacity_prompt=16, window_size=8)
    model = SnapKVQwen3_5ForCausalLM(cfg).eval()

    input_ids = torch.ones(1, 2, dtype=torch.long)
    attention_mask = torch.ones(1, 2, dtype=torch.long)

    model_inputs = model.prepare_inputs_for_generation(
        input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        hf_chunked_prefill=True,
        prefill_final_chunk=False,
    )

    assert model_inputs["hf_chunked_prefill"] is True
    assert model_inputs["prefill_final_chunk"] is False


@pytest.mark.chunk_prefill
@pytest.mark.integration
def test_hf_style_prefill_chunk_size_prefix_sliced_mask_triggers_snap_on_final():
    """HF prefill_chunk_size: prefix-sliced masks defer SnapKV to the final chunk flag."""

    model = SnapKVQwen3_5TextModel(
        _mini_snap_cfg(window_size=2, max_capacity_prompt=4)
    ).eval()

    torch.manual_seed(4)
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, 0.02)

    B, L = 1, 8
    input_ids = torch.arange(10, 10 + L).view(1, -1)
    attention_mask = torch.ones(B, L, dtype=torch.bool)

    out = _run_hf_style_chunked_prefill(
        model,
        input_ids,
        attention_mask,
        chunk_size=2,
    )

    pst = out.past_key_values
    assert getattr(pst, "snap_prefill_complete", False)

    sa = _first_snapkv_self_attn_from_text(model)
    assert sa.snap_kv_cache.snap_score is not None

    li = sa.layer_idx
    assert pst.layers[li].keys.shape[2] <= sa.snap_kv_cache.max_capacity_prompt


@pytest.mark.chunk_prefill
@pytest.mark.integration
def test_hf_style_prefix_sliced_mask_does_not_infer_final_without_flag():
    """First HF chunk (prefix mask width == chunk length) must not trigger SnapKV."""

    model = SnapKVQwen3_5TextModel(
        _mini_snap_cfg(window_size=2, max_capacity_prompt=4)
    ).eval()

    torch.manual_seed(5)
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, 0.02)

    B, L = 1, 8
    input_ids = torch.arange(30, 30 + L).view(1, -1)
    attention_mask = torch.ones(B, L, dtype=torch.bool)

    # Old code incorrectly treated this as final because sum(mask) == chunk length.
    out = model(
        input_ids=input_ids[:, :2],
        attention_mask=attention_mask[:, :2],
        use_cache=True,
        hf_chunked_prefill=True,
        prefill_final_chunk=False,
    )

    pst = out.past_key_values
    assert not getattr(pst, "snap_prefill_complete", False)

    sa = _first_snapkv_self_attn_from_text(model)
    assert sa.snap_kv_cache.snap_score is None


# ---------------------------------------------------------------------------
# G. GPU HF tests
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize("batch_size", [1, 2])
def test_gpu_generate_and_reset_cache(qwen_cfg, batch_size):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)
    tok.padding_side = "left"
    cfg = _patch_snapkv_budget(qwen_cfg, max_capacity_prompt=16, window_size=8)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = SnapKVQwen3_5ForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B",
        config=cfg,
        torch_dtype=dtype,
        device_map="cuda:0",
        trust_remote_code=True,
        attn_implementation="eager",
    ).eval()
    if batch_size == 1:
        batch = tok("5+7=", return_tensors="pt").to("cuda")
    else:
        batch = tok(
            ["9+1=", "short question one word sky color"],
            padding=True,
            return_tensors="pt",
        ).to("cuda")
    with torch.inference_mode():
        out = model.generate(
            **batch,
            max_new_tokens=8,
            do_sample=False,
            use_cache=True,
            pad_token_id=tok.pad_token_id,
        )
    assert out.shape[0] == batch_size
    assert torch.isfinite(out.float()).all()
    model.reset_snapkv_state()
    for lyr in model.model.language_model.layers:
        sa = getattr(lyr, "self_attn", None)
        if sa is None or sa.__class__.__name__ != "SnapKVQwen3_5Attention":
            continue
        assert sa.snap_kv_cache.snap_score is None and sa.snap_kv_cache.position_ids is None


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_gpu_long_prefill_past_within_cache_cap(qwen_cfg):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)
    tok.padding_side = "left"
    cfg = _patch_snapkv_budget(qwen_cfg, max_capacity_prompt=24, window_size=12)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = SnapKVQwen3_5ForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B",
        config=cfg,
        torch_dtype=dtype,
        device_map="cuda:0",
        trust_remote_code=True,
        attn_implementation="eager",
    ).eval()
    long_text = "counting " + " ".join(str(i) for i in range(220))
    batch = tok(long_text, return_tensors="pt").to("cuda")
    model.reset_snapkv_state()
    with torch.inference_mode():
        out = model(**batch, use_cache=True, return_dict=True)
    pst = out.past_key_values
    lm = model.model.language_model
    for li, lyr in enumerate(lm.layers):
        if (
            not hasattr(lyr, "self_attn")
            or lyr.self_attn.__class__.__name__ != "SnapKVQwen3_5Attention"
        ):
            continue
        cap = lyr.self_attn.snap_kv_cache.max_capacity_prompt
        layer = pst.layers[li]
        if layer.keys is None:
            continue
        assert layer.keys.shape[2] <= cap


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_gpu_greedy_padded_decode_position_ids_stable(qwen_cfg):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)
    tok.padding_side = "left"
    cfg = _patch_snapkv_budget(qwen_cfg, max_capacity_prompt=192, window_size=96)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = SnapKVQwen3_5ForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B",
        config=cfg,
        torch_dtype=dtype,
        device_map="cuda:0",
        trust_remote_code=True,
        attn_implementation="eager",
    ).eval()
    prompts = ["9+1=", "short question sky color yes"]
    batch = tok(prompts, padding=True, return_tensors="pt").to("cuda")
    ref_sa = _first_snapkv_self_attn_layer(model).self_attn
    li = ref_sa.layer_idx

    def verify(past, attn_cuda: torch.Tensor, tag: str):
        lay = past.layers[li]
        kv_len = lay.keys.shape[2]
        pid = ref_sa.snap_kv_cache.position_ids
        assert pid is not None and pid.shape[-1] == kv_len
        assert kv_len == past.get_seq_length(), tag
        snap_next = past.snap_next_position
        pred_next = pid[:, :, -1].max(dim=1).values.to(dtype=torch.long) + 1
        assert torch.equal(snap_next.detach().cpu(), pred_next.detach().cpu()), tag
        pid_cpu = pid.detach().cpu()
        am_bool = attn_cuda.detach().cpu().bool()
        if kv_len == am_bool.shape[-1]:
            vm = build_snapkv_kv_valid_mask(pid_cpu, am_bool)
            assert vm is not None
            in_bounds = pid_cpu < am_bool.shape[-1]
            safe = pid_cpu.clamp(0, am_bool.shape[-1] - 1).long()
            expanded = am_bool.unsqueeze(1).expand_as(pid_cpu)
            gathered = expanded.gather(2, safe)
            pad_slot = ~gathered
            assert not (vm & in_bounds & pad_slot).any(), tag

    model.reset_snapkv_state()
    inp_ids, attn = batch["input_ids"], batch["attention_mask"]
    with torch.inference_mode():
        out = model(input_ids=inp_ids, attention_mask=attn, use_cache=True, return_dict=True)
        pst = out.past_key_values
        verify(pst, attn, "prefill")
        for step in range(5):
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            attn = torch.cat([attn, torch.ones_like(nxt, dtype=attn.dtype)], dim=-1)
            out = model(
                input_ids=nxt,
                attention_mask=attn,
                past_key_values=pst,
                use_cache=True,
                return_dict=True,
            )
            pst = out.past_key_values
            verify(pst, attn, f"decode_{step}")


@pytest.mark.gpu
@pytest.mark.chunk_prefill
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_gpu_chunked_prefill_snapkv_dynamic_cache(qwen_cfg):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)
    tok.padding_side = "left"
    cfg = _patch_snapkv_budget(qwen_cfg, max_capacity_prompt=32, window_size=8)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = SnapKVQwen3_5ForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B",
        config=cfg,
        torch_dtype=dtype,
        device_map="cuda:0",
        trust_remote_code=True,
        attn_implementation="eager",
    ).eval()
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    batch = tok(text, return_tensors="pt").to("cuda")
    attn = batch["attention_mask"].bool()
    cols = attn[0].nonzero(as_tuple=True)[0].tolist()
    mid = len(cols) // 2
    chunk_cols = [cols[:mid], cols[mid:]]
    model.reset_snapkv_state()
    pst = None
    with torch.inference_mode():
        for i, cc in enumerate(chunk_cols):
            chunk = {k: v for k, v in batch.items()}
            chunk["input_ids"] = batch["input_ids"][:, cc]
            out = model(**chunk, past_key_values=pst, use_cache=True, return_dict=True)
            pst = out.past_key_values
            if i == 0:
                assert not getattr(pst, "snap_prefill_complete", False)
            else:
                assert getattr(pst, "snap_prefill_complete", False)
    sa = _first_snapkv_self_attn_layer(model).self_attn
    cap = sa.snap_kv_cache.max_capacity_prompt
    li = sa.layer_idx
    assert pst.layers[li].keys.shape[2] <= cap
