import torch
import torch.nn.functional as F
from collections.abc import Callable
from torch import nn

from transformers.cache_utils import Cache, DynamicCache, DynamicLayer
from transformers.integrations import use_kernelized_func
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5ForCausalLM,
    Qwen3_5ModelOutputWithPast,
    Qwen3_5PreTrainedModel,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, is_torch_xpu_available, logging
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.import_utils import is_torch_greater_or_equal
from transformers.utils.output_capturing import capture_outputs

_IS_TORCH_GE_2_6 = is_torch_greater_or_equal("2.6", accept_dev=True)
_IS_TORCH_XPU_AVAILABLE = is_torch_xpu_available()
logger = logging.get_logger(__name__)
__all__ = [
    "SnapKVCache",
    "SnapKVQwen3_5Attention",
    "SnapKVQwen3_5TextModel",
    "SnapKVQwen3_5Model",
    "SnapKVQwen3_5ForCausalLM",
    "build_snapkv_kv_valid_mask",
    "left_pad_logical_positions",
    "make_snapkv_causal_mask",
]

class SnapKVCache:
    """Snap KV cache.

    For batched decoder generation, use left padding (`tokenizer.padding_side = 'left'`) 
    """

    def __init__(
        self,
        window_size=512,
        max_capacity_prompt=2048,
        kernel_size=3,
        k_seq_dim=2,
        v_seq_dim=2,
        layer_idx: int | None = None,
        num_attention_heads: int | None = None,
        num_key_value_heads: int | None = None,
        num_key_value_groups: int | None = None,    
        pooling: str = "avgpool",
    ):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.kernel_size = kernel_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.layer_idx = layer_idx
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_key_value_groups
        self.snap_score = None
        self.position_ids = None
        self.pooling = pooling

    def _get_kv_tensors(self, past_key_values: Cache | tuple):
        """Return (keys, values) tensor pair for this layer (Cache uses ``layer_idx``)."""
        if isinstance(past_key_values, Cache):
            if self.layer_idx is None or self.layer_idx >= len(past_key_values.layers):
                return None, None
            layer = past_key_values.layers[self.layer_idx]
            if not isinstance(layer, DynamicLayer) or not layer.is_initialized:
                return None, None
            if layer.keys is None or layer.values is None or layer.keys.numel() == 0:
                return None, None
            return layer.keys, layer.values
        keys, values = past_key_values[0], past_key_values[1]
        return keys, values

    def _append_positions(
        self,
        attn_score_cache,
        text_position_ids: torch.Tensor | None = None,
        padding_attention_mask: torch.Tensor | None = None,
    ):
        """Track logical text positions of every token currently in the KV cache."""
        bsz, _, q_len, kv_len = attn_score_cache.shape
        n_heads = self.snap_score.shape[1]
        device = attn_score_cache.device
        dtype = torch.long

        if self.position_ids is None:
            if (
                padding_attention_mask is not None
                and padding_attention_mask.shape[-1] == kv_len
            ):
                pos = left_pad_logical_positions(
                    padding_attention_mask.to(device=device)
                )
            elif text_position_ids is not None:
                pos = text_position_ids.to(device=device, dtype=dtype)
                if pos.ndim == 3:
                    pos = pos[0]
                if pos.shape[-1] != kv_len:
                    raise ValueError(
                        f"SnapKV initial text_position_ids length {pos.shape[-1]} != kv_len {kv_len}"
                    )
            else:
                pos = torch.arange(kv_len, device=device, dtype=dtype).view(1, -1).expand(bsz, -1)

            self.position_ids = pos[:, None, :].expand(bsz, n_heads, kv_len).clone()
        else:
            if text_position_ids is not None:
                pos = text_position_ids.to(device=device, dtype=dtype)
                if pos.ndim == 3:
                    pos = pos[0]
                pos = pos[:, -q_len:]
                if pos.shape[-1] != q_len:
                    raise ValueError(
                        f"SnapKV append text_position_ids length {pos.shape[-1]} != q_len {q_len}"
                    )
                new = pos[:, None, :].expand(bsz, n_heads, q_len)
            else:
                last = self.position_ids[:, :, -1:] + 1
                step = torch.arange(q_len, device=device, dtype=dtype).view(1, 1, -1)
                new = last + step

            self.position_ids = torch.cat([self.position_ids, new], dim=-1)

    def append_new_positions(self, text_position_ids: torch.Tensor | None, target_kv_len: int) -> None:
        """
        After SnapKV prefill compression, generation appends new KV tokens to the cache.
        This keeps self.position_ids aligned with the physical KV cache length.
        """
        if self.position_ids is None:
            return

        cur_len = self.position_ids.shape[-1]
        missing = target_kv_len - cur_len

        if missing <= 0:
            return

        bsz, n_heads, _ = self.position_ids.shape
        device = self.position_ids.device
        dtype = self.position_ids.dtype

        if text_position_ids is not None:
            new_pos = text_position_ids.to(device=device, dtype=dtype)

            if new_pos.ndim == 3:
                # If full Qwen-style position_ids accidentally arrives, use text component.
                new_pos = new_pos[0]

            new_pos = new_pos[:, -missing:]

            if new_pos.shape[-1] != missing:
                raise ValueError(
                    f"Cannot append SnapKV positions: need {missing} new positions, "
                    f"got {new_pos.shape[-1]}."
                )

            new_pos = new_pos[:, None, :].expand(bsz, n_heads, missing)

        else:
            # Fallback: infer monotonic positions from the last tracked position.
            step = torch.arange(1, missing + 1, device=device, dtype=dtype).view(1, 1, -1)
            new_pos = self.position_ids[:, :, -1:] + step

        self.position_ids = torch.cat([self.position_ids, new_pos], dim=-1)

    def _assign_kv_tensors(self, past_key_values: Cache | tuple, k_new: torch.Tensor, v_new: torch.Tensor):
        if isinstance(past_key_values, Cache):
            layer = past_key_values.layers[self.layer_idx]
            layer.keys = k_new
            layer.values = v_new
            return past_key_values
        return (k_new, v_new)

    def _write_snap_next_position(self, past_key_values: Cache) -> None:
        if self.position_ids is None or self.position_ids.numel() == 0:
            return

        new_next = (
            self.position_ids[:, :, -1].max(dim=1).values.detach() + 1
        ).to(device=self.position_ids.device, dtype=torch.long)

        past_key_values.snap_next_position = new_next

    def _mask_snap_scores_with_valid_positions(
        self,
        select_snap_scores: torch.Tensor,
        valid_kv_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Mask padding KV slots so they are not chosen for retention; preserves tensor shape."""
        if valid_kv_mask is None:
            return select_snap_scores
        score_width = select_snap_scores.shape[-1]
        valid_score_mask = valid_kv_mask[:, :, :score_width]
        if valid_score_mask.shape[-1] != score_width:
            raise ValueError(
                f"valid_kv_mask slice width {valid_score_mask.shape[-1]} != snap score width {score_width}"
            )
        out = select_snap_scores.masked_fill(
            ~valid_score_mask,
            torch.finfo(select_snap_scores.dtype).min,
        )
        if (valid_score_mask.sum(dim=-1) < self.max_capacity_prompt - self.window_size).any():
            logger.warning_once(
                "Some SnapKV heads have fewer valid candidates than the prompt capacity budget; "
                "topk may include masked fallback indices."
            )
        return out

    def __call__(
        self,
        past_key_values,
        attn_score_cache,
        padding_attention_mask=None,
        text_position_ids: torch.Tensor | None = None,
    ):
        if attn_score_cache is None:
            return past_key_values
        self._update_snap_score(attn_score_cache)
        self._append_positions(
            attn_score_cache,
            text_position_ids=text_position_ids,
            padding_attention_mask=padding_attention_mask,
        )
        if past_key_values is None:
            return None

        keys, values = self._get_kv_tensors(past_key_values)
        if keys is None:
            return past_key_values

        if isinstance(past_key_values, Cache) and self.layer_idx is None:
            raise ValueError("SnapKVCache(layer_idx=...) is required when past_key_values is a Cache.")

        bsz, num_heads, _, head_dim = keys.shape
        seq_len = keys.size(self.k_seq_dim)

        valid_kv_mask = None
        if padding_attention_mask is not None:
            if padding_attention_mask.shape[-1] == seq_len:
                valid_kv_mask = padding_attention_mask[:, None, :].expand(bsz, num_heads, seq_len)
            elif seq_len < padding_attention_mask.shape[-1]:
                valid_kv_mask = torch.ones(
                    bsz, num_heads, seq_len, dtype=torch.bool, device=keys.device
                )
            elif self.position_ids is not None:
                valid_kv_mask = build_snapkv_kv_valid_mask(
                    self.position_ids, padding_attention_mask
                )

        if valid_kv_mask is not None:
            valid_counts = valid_kv_mask.sum(dim=-1)
            needs_eviction = (valid_counts > self.max_capacity_prompt).any()
        else:
            needs_eviction = seq_len > self.max_capacity_prompt

        if not needs_eviction:
            self._write_snap_next_position(past_key_values)
            return past_key_values

        select_snap_scores = self.snap_score
        select_snap_scores = self._mask_snap_scores_with_valid_positions(select_snap_scores, valid_kv_mask)
        if self.pooling == 'avgpool':
            select_snap_scores = F.avg_pool1d(select_snap_scores, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        elif self.pooling == 'maxpool':
            select_snap_scores = F.max_pool1d(select_snap_scores, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        else:
            raise ValueError('Pooling method not supported')
        _, keep_top_k = torch.topk(select_snap_scores, self.max_capacity_prompt - self.window_size, dim=-1)
        keep_top_k = keep_top_k.sort().values
        keep_recent = torch.arange(
            seq_len - self.window_size, seq_len, device=keep_top_k.device
        ).view(1, 1, -1).expand(bsz, num_heads, -1)
        keep_idx = torch.cat([keep_top_k, keep_recent], dim=-1)
        gather_idx = keep_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        k_hh_recent = torch.gather(keys, dim=self.k_seq_dim, index=gather_idx)
        v_hh_recent = torch.gather(values, dim=self.v_seq_dim, index=gather_idx)
        if self.position_ids is not None:
            self.position_ids = torch.gather(self.position_ids, dim=-1, index=keep_idx)
        self._write_snap_next_position(past_key_values)
        return self._assign_kv_tensors(past_key_values, k_hh_recent, v_hh_recent)

    def _update_snap_score(self, attn_score_cache):
        num_new_tokens = attn_score_cache.shape[2]
        if (self.num_key_value_heads is not None and self.num_key_value_groups is not None and attn_score_cache.ndim == 4):
            bsz, n_attn_heads, _, kv_len = attn_score_cache.shape
            n_kv_heads = self.num_key_value_heads
            n_groups = self.num_key_value_groups

            if n_attn_heads == n_kv_heads * n_groups:
                scores = attn_score_cache.reshape(
                    bsz, n_kv_heads, n_groups, num_new_tokens, kv_len
                )
                
                scores = scores[..., -self.window_size:, :-self.window_size].sum(dim=2).sum(dim=2)  # (bsz, n_kv_heads, kv_len)
            else:
                scores = attn_score_cache[..., -self.window_size:, :-self.window_size].sum(dim=2)
        else:
            # non-GQA
            scores = attn_score_cache[..., -self.window_size:, :-self.window_size].sum(dim=2)

        if self.snap_score is None:
            self.snap_score = scores
        else:
            raise ValueError("snap_score is already set. DO NOT RUN MORE THAN ONCE")

    def get_absolute_position_ids(self):
        return self.position_ids

    def _clean_scores(self):
        self.snap_score = None
        self.position_ids = None


@use_kernelized_func(apply_rotary_pos_emb)
class SnapKVQwen3_5Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.window_size = getattr(config, "window_size", 512)
        self.is_causal = True
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim * 2, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.snap_kv_cache = SnapKVCache(
            window_size=getattr(config, "window_size", 512),
            max_capacity_prompt=getattr(config, "max_capacity_prompt", 2048),
            kernel_size=getattr(config, "kernel_size", 3),
            k_seq_dim=2,
            v_seq_dim=2,
            num_attention_heads=getattr(config, "num_attention_heads", None),
            num_key_value_heads=getattr(config, "num_key_value_heads", None),
            num_key_value_groups=self.num_key_value_groups,
            layer_idx=layer_idx,
        )

    def _build_additive_attention_mask(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        text_position_ids: torch.LongTensor | None,
        padding_attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
        snap_scoring_pass: bool = False,
    ) -> torch.Tensor:
        batch, _, q_len, _ = query_states.shape
        kv_len = key_states.shape[2]
        device = query_states.device
        dtype = query_states.dtype
        num_kv_heads = self.config.num_key_value_heads

        if text_position_ids is None:
            q_arange = torch.arange(q_len, device=device, dtype=torch.long).view(1, -1).expand(batch, -1)
        else:
            q_arange = text_position_ids.to(device=device, dtype=torch.long)
            if q_arange.shape[0] != batch:
                q_arange = q_arange.expand(batch, -1)

        if past_key_values is None:
            kv_arange = torch.arange(kv_len, device=device, dtype=q_arange.dtype).view(1, 1, -1).expand(batch, num_kv_heads, -1)
            return make_snapkv_causal_mask(
                q_arange=q_arange,
                kv_arange=kv_arange,
                attention_mask=padding_attention_mask,
                num_attention_heads=self.config.num_attention_heads,
                dtype=dtype,
                device=device,
            )

        if snap_scoring_pass:
            if (
                padding_attention_mask is not None
                and padding_attention_mask.shape[-1] == kv_len
            ):
                kv_pos = left_pad_logical_positions(
                    padding_attention_mask.to(device=device)
                ).to(dtype=q_arange.dtype)
                kv_arange = kv_pos[:, None, :].expand(batch, num_kv_heads, kv_len)
            else:
                kv_arange = torch.arange(
                    kv_len, device=device, dtype=q_arange.dtype
                ).view(1, 1, -1).expand(batch, num_kv_heads, -1)
        elif self.snap_kv_cache.position_ids is None:
            if (
                padding_attention_mask is not None
                and padding_attention_mask.shape[-1] == kv_len
            ):
                kv_pos = left_pad_logical_positions(
                    padding_attention_mask.to(device=device)
                ).to(dtype=q_arange.dtype)
                kv_arange = kv_pos[:, None, :].expand(batch, num_kv_heads, kv_len)
            else:
                kv_arange = torch.arange(
                    kv_len, device=device, dtype=q_arange.dtype
                ).view(1, 1, -1).expand(batch, num_kv_heads, -1)
        else:
            prev = self.snap_kv_cache.position_ids.to(device=device, dtype=q_arange.dtype)
            prev_len = prev.shape[-1]

            if prev_len == kv_len:
                # Cache consists exactly of tracked SnapKV positions.
                kv_arange = prev

            elif prev_len + q_len == kv_len:
                # Cache contains tracked compressed KV + current newly appended KV.
                cur = q_arange[:, None, :].expand(batch, num_kv_heads, q_len)
                kv_arange = torch.cat([prev, cur], dim=-1)

            else:
                raise ValueError(
                    f"SnapKV position_ids length {prev_len}, q_len {q_len}, kv_len {kv_len} mismatch "
                    f"at layer {self.layer_idx}. Cannot safely build causal mask."
                )

        if kv_arange.shape[-1] != kv_len:
            raise ValueError(
                f"SnapKV kv_arange length {kv_arange.shape[-1]} != key_states seq len {kv_len} "
                f"(layer {self.layer_idx}). KV position tracking may be out of sync."
            )

        dense_real_kv = (
            padding_attention_mask is not None
            and kv_len < padding_attention_mask.shape[-1]
            and self.snap_kv_cache.position_ids is None
        )
        use_logical_kv_positions = (
            snap_scoring_pass
            or self.snap_kv_cache.position_ids is not None
            or (
                padding_attention_mask is not None
                and padding_attention_mask.shape[-1] == kv_len
            )
        )
        if dense_real_kv:
            mask_for_causal = None
        elif use_logical_kv_positions:
            mask_for_causal = None
        else:
            mask_for_causal = padding_attention_mask

        additive = make_snapkv_causal_mask(
            q_arange=q_arange,
            kv_arange=kv_arange,
            attention_mask=mask_for_causal,
            num_attention_heads=self.config.num_attention_heads,
            dtype=dtype,
            device=device,
        )

        if (
            use_logical_kv_positions
            and not dense_real_kv
            and padding_attention_mask is not None
            and kv_len == padding_attention_mask.shape[-1]
        ):
            physical_valid = padding_attention_mask.to(device=device, dtype=torch.bool)
            physical_valid = physical_valid[:, None, None, :].expand(
                batch,
                self.config.num_attention_heads,
                q_len,
                kv_len,
            )
            additive = additive.masked_fill(
                ~physical_valid,
                torch.finfo(dtype).min,
            )

        return additive

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        position_ids: torch.LongTensor | None = None,
        padding_attention_mask: torch.Tensor | None = None,
        
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        attn_weights_snap = None
        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
            )

            if getattr(past_key_values, "snap_prefill_complete", False):
                self.snap_kv_cache.append_new_positions(
                    position_ids,
                    key_states.shape[2],
                )
                self.snap_kv_cache._write_snap_next_position(past_key_values)

        if getattr(past_key_values, "last_chunk_processed", False) and not getattr(past_key_values, "snap_prefill_complete", False):
            additive_attention_mask = self._build_additive_attention_mask(
                query_states[..., -self.window_size:, :],
                key_states,
                position_ids[..., -self.window_size:],
                padding_attention_mask,
                past_key_values,
                snap_scoring_pass=True,
            )

            _, attn_weights_snap = eager_attention_forward(
                self,
                query_states[..., -self.window_size:, :],
                key_states,
                value_states,
                additive_attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs
            )

        additive_attention_mask = self._build_additive_attention_mask(
            query_states,
            key_states,
            position_ids,
            padding_attention_mask,
            past_key_values,
        )

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            additive_attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs
        )

        if past_key_values is not None and attn_weights_snap is not None:
            if getattr(past_key_values, "last_chunk_processed", False) and not getattr(past_key_values, "snap_prefill_complete", False):
                past_key_values = self.snap_kv_cache(
                    past_key_values,
                    attn_weights_snap.detach().clone(),
                    padding_attention_mask=padding_attention_mask,
                    text_position_ids=position_ids,
                )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class SnapKVQwen3_5TextModel(Qwen3_5PreTrainedModel):
    config: Qwen3_5TextConfig

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        for layer_idx, layer in enumerate(self.layers):
            if self.config.layer_types[layer_idx] != "linear_attention":
                layer.self_attn = SnapKVQwen3_5Attention(config, layer_idx)
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
            past_key_values.snap_prefill_complete = False
        # the hard coded `4` is for text, temporal, height and width.
        if position_ids is None:
            batch = inputs_embeds.shape[0]
            seq_len = inputs_embeds.shape[1]
            device = inputs_embeds.device

            if past_key_values is not None and hasattr(past_key_values, "snap_next_position"):
                past_seen_tokens = past_key_values.snap_next_position
            else:
                past_seen_tokens = (
                    past_key_values.get_seq_length()
                    if past_key_values is not None
                    else 0
                )

            if (
                attention_mask is not None
                and seq_len == attention_mask.shape[-1]
            ):
                text_pos = left_pad_logical_positions(
                    attention_mask.to(device=device)
                )
                position_ids = text_pos.unsqueeze(0).expand(4, batch, -1)

            elif isinstance(past_seen_tokens, torch.Tensor):
                past_seen_tokens = past_seen_tokens.to(device=device)

                if past_seen_tokens.ndim == 0:
                    step = torch.arange(seq_len, device=device, dtype=past_seen_tokens.dtype)
                    position_ids = step + past_seen_tokens
                    position_ids = position_ids.view(1, 1, -1).expand(4, batch, -1)

                else:
                    # Per-sample offsets, expected shape: (batch,)
                    past_seen_tokens = past_seen_tokens.reshape(-1)

                    if past_seen_tokens.shape[0] != batch:
                        raise ValueError(
                            f"Expected past_seen_tokens to have shape ({batch},), "
                            f"got {tuple(past_seen_tokens.shape)}"
                        )

                    step = torch.arange(seq_len, device=device, dtype=past_seen_tokens.dtype)
                    position_ids = past_seen_tokens.unsqueeze(-1) + step
                    position_ids = position_ids.unsqueeze(0).expand(4, -1, -1)

            else:
                step = torch.arange(seq_len, device=device)
                position_ids = step + past_seen_tokens
                position_ids = position_ids.view(1, 1, -1).expand(4, batch, -1)

        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        if attention_mask is not None:
            padding_attention_mask_bool = attention_mask.to(
                device=inputs_embeds.device, dtype=torch.bool
            )
        else:
            padding_attention_mask_bool = None

        linear_attn_mask = self._update_linear_attn_mask(attention_mask, past_key_values)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        prefill_len_before_forward = (
            past_key_values.get_seq_length()
            if past_key_values is not None
            else 0
        )
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if self.config.layer_types[i] == "linear_attention":
                layer_mask = linear_attn_mask
                decoder_kwargs = dict(kwargs)
            else:
                layer_mask = None
                decoder_kwargs = {**kwargs, "padding_attention_mask": padding_attention_mask_bool}
            if (
                use_cache
                and past_key_values is not None
                and attention_mask is not None
                and not getattr(past_key_values, "last_chunk_processed", False)
                and prefill_len_before_forward == 0
            ):
                past_key_values.last_chunk_processed = True
                decoder_kwargs["last_chunk_processed"] = True
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **decoder_kwargs,
            )

        if (
            use_cache
            and past_key_values is not None
            and attention_mask is not None
            and not getattr(past_key_values, "snap_prefill_complete", False)
            and getattr(past_key_values, "last_chunk_processed", False)
        ):
            # last_chunk_processed already validated prefill completion this forward.
            past_key_values.snap_prefill_complete = True

        hidden_states = self.norm(hidden_states)

        return Qwen3_5ModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def _update_linear_attn_mask(self, attention_mask, past_key_values):
        """
        NOTE: Left-padding is used for linear attention mask.
        No need for zeroing states when
            1. Cached forward
            2. Attending to all inputs
        """
        linear_attn_mask = attention_mask
        has_linear_layers = any(t == "linear_attention" for t in self.config.layer_types)
        linear_cache_active = False
        if has_linear_layers and past_key_values is not None:
            try:
                linear_cache_active = past_key_values.has_previous_state()
            except ValueError:
                linear_cache_active = False
        if linear_cache_active or (
            attention_mask is not None and torch.all(attention_mask == 1)
        ):
            linear_attn_mask = None
        return linear_attn_mask


class SnapKVQwen3_5Model(Qwen3_5PreTrainedModel):
    """Wraps the SnapKV text tower under ``language_model`` for multimodal checkpoint key layout."""

    def __init__(self, config):
        super().__init__(config)
        text_config = config.text_config if hasattr(config, "text_config") else config
        self.language_model = SnapKVQwen3_5TextModel(text_config)
        self.post_init()

    def forward(self, *args, **kwargs):
        return self.language_model(*args, **kwargs)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)


class SnapKVQwen3_5ForCausalLM(Qwen3_5ForCausalLM):

    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self, config):
        Qwen3_5PreTrainedModel.__init__(self, config)
        text_config = config.text_config if hasattr(config, "text_config") else config
        self.text_config = text_config

        self.model = SnapKVQwen3_5Model(config)

        self.vocab_size = text_config.vocab_size
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)
        self.post_init()

    def reset_snapkv_state(self):
        text_model = self.model.language_model
        inner_cfg = text_model.config
        for layer_idx, layer in enumerate(text_model.layers):
            if inner_cfg.layer_types[layer_idx] != "linear_attention":
                layer.self_attn.snap_kv_cache._clean_scores()

    def prepare_inputs_for_generation(self, *args, **kwargs):
        past_key_values = kwargs.get("past_key_values", None)

        is_new_sequence = past_key_values is None or (
            hasattr(past_key_values, "get_seq_length")
            and past_key_values.get_seq_length() == 0
        )
        if is_new_sequence:
            self.reset_snapkv_state()

        return super().prepare_inputs_for_generation(*args, **kwargs)


def left_pad_logical_positions(
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Convert a left-padded 2D attention mask into logical text positions.

    Example:
        [0, 0, 1, 1, 1] -> [0, 0, 0, 1, 2]
        [0, 1, 1, 1, 1] -> [0, 0, 1, 2, 3]
        [1, 1, 1, 1, 1] -> [0, 1, 2, 3, 4]
    """
    mask = attention_mask.bool()
    batch, seq_len = mask.shape
    device = mask.device
    dtype = torch.long

    real_lens = mask.sum(dim=-1, keepdim=True)
    col = torch.arange(seq_len, device=device, dtype=dtype).view(1, -1)
    offset = seq_len - real_lens
    local = (col - offset).clamp(min=0) * mask.to(dtype=dtype)

    return local


def build_snapkv_kv_valid_mask(
    kv_position_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    """
    Per-KV-slot validity from absolute positions and the 2D padding mask.

    kv_position_ids:
        Shape ``(batch, kv_heads, kv_len)`` — absolute token index for each KV slot.
    attention_mask:
        Shape ``(batch, full_len)`` — True = real token, False = padding.

    Returns:
        Shape ``(batch, kv_heads, kv_len)``, True where the slot is valid (non-padding).
        Positions beyond ``attention_mask.shape[-1]`` are treated as valid (generated tokens).

    Returns ``None`` when ``attention_mask`` is ``None`` (caller falls back to physical seq length).
    """
    if attention_mask is None:
        return None

    device = kv_position_ids.device
    attention_mask = attention_mask.to(device=device, dtype=torch.bool)

    in_bounds = kv_position_ids < attention_mask.shape[-1]

    safe_index = kv_position_ids.clamp(
        min=0,
        max=attention_mask.shape[-1] - 1,
    ).long()

    expanded_padding = attention_mask[:, None, :].expand(
        -1,
        kv_position_ids.shape[1],
        -1,
    )

    gathered_padding = expanded_padding.gather(
        dim=-1,
        index=safe_index,
    )

    valid_kv_mask = torch.where(
        in_bounds,
        gathered_padding,
        torch.ones_like(gathered_padding, dtype=torch.bool),
    )

    return valid_kv_mask


def make_snapkv_causal_mask(
    q_arange: torch.Tensor,
    kv_arange: torch.Tensor,
    attention_mask: torch.Tensor | None,
    num_attention_heads: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Build an additive causal attention mask for SnapKV KV cache.
    """

    q_arange = q_arange.to(device=device)
    kv_arange = kv_arange.to(device=device)

    causal = kv_arange[:, :, None, :] <= q_arange[:, None, :, None]
    if attention_mask is not None:
        kv_padding_valid = build_snapkv_kv_valid_mask(kv_arange, attention_mask)
        if kv_padding_valid is not None:
            causal = causal & kv_padding_valid[:, :, None, :]

    if causal.shape[1] != num_attention_heads:
        if num_attention_heads % causal.shape[1] != 0:
            raise ValueError(
                f"Cannot repeat SnapKV mask from {causal.shape[1]} KV heads "
                f"to {num_attention_heads} attention heads."
            )

        repeat = num_attention_heads // causal.shape[1]
        causal = causal.repeat_interleave(repeat, dim=1)


    additive_mask = torch.zeros(
        causal.shape,
        dtype=dtype,
        device=device,
    )

    additive_mask = additive_mask.masked_fill(
        ~causal,
        torch.finfo(dtype).min,
    )
    return additive_mask