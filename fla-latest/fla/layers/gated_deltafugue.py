# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

from fla.layers.utils import get_layer_cache, get_unpad_data, index_first_axis, pad_input, update_layer_cache
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.ops.deltafugue import fused_delay_shift
from fla.ops.delta_rule import chunk_delta_rule, fused_recurrent_delta_rule
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


class GatedDeltaFugue(nn.Module):
    """
    GatedDeltaFugue: combines Gated DeltaNet's forget-gated associative memory
    with DeltaFugue's bounded 1D tape memory pointer -- the main associative
    memory reads/writes `v_aug = v - shifted_tape_read` (the tape's own
    prediction error) instead of `v` directly, exactly as in plain DeltaFugue,
    but the associative-memory update itself is the gated delta rule (forget
    gate `g` via `A_log`/`dt_bias`) rather than the plain delta rule.

    The tape mechanism (dual read/write pointers, Laplacian-kernel addressing,
    tape delta rule, `fused_delay_shift`) is the same as in
    `fla.layers.delta_fugue.DeltaFugue`; the associative memory is called the
    same way `fla.layers.gated_deltanet.GatedDeltaNet` calls the gated delta
    rule (gate and beta sigmoid applied inside the kernel).
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 2.0,
        head_dim: int = 256,
        num_heads: int = 6,
        num_v_heads: int = None,
        mode: str = 'chunk',
        use_gate: bool = True,
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        tape_size: int = 8,
        pointer_gamma_init: float = 2.0,
        **kwargs,
    ) -> GatedDeltaFugue:
        super().__init__()

        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.hidden_size = hidden_size
        self.expand_v = expand_v

        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads

        self.head_k_dim = head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        self.layer_idx = layer_idx

        # Consistency check: Ensure expand_v produces integer values
        if not math.isclose(self.num_v_heads * self.head_dim * expand_v, self.value_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by key_dim={self.key_dim}."
            )
        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError(f"num_v_heads={self.num_v_heads} must be divisible by num_heads={self.num_heads}.")

        assert mode in ['chunk', 'fused_recurrent'], f"Not supported mode `{mode}`."

        # Base Projections
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        # Gated DeltaNet Specific Projections
        self.a_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)

        A = torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(
            torch.rand(self.num_v_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min),
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # Tape Specific Parameters
        self.tape_size = tape_size
        self.orig_tape_size = tape_size
        self.pointer_gamma = nn.Parameter(torch.tensor([float(pointer_gamma_init)]))  # sharpness = softplus(pointer_gamma)

        # Note: Tape projections align with num_v_heads because the tape operates on the value dimension
        self.read_action_proj = nn.Linear(hidden_size, self.num_v_heads * 3, bias=True)
        self.write_action_proj = nn.Linear(hidden_size, self.num_v_heads * 3, bias=True)
        self.beta_tape_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)

        with torch.no_grad():
            for h in range(self.num_v_heads):
                dominant_action = h % 3
                self.read_action_proj.bias[h * 3 + dominant_action] += 2.0
                self.write_action_proj.bias[h * 3 + dominant_action] += 2.0

        if use_short_conv:
            self.q_conv1d = ShortConvolution(hidden_size=self.key_dim, kernel_size=conv_size, bias=conv_bias, activation='silu')
            self.k_conv1d = ShortConvolution(hidden_size=self.key_dim, kernel_size=conv_size, bias=conv_bias, activation='silu')
            self.v_conv1d = ShortConvolution(hidden_size=self.value_dim, kernel_size=conv_size, bias=conv_bias, activation='silu')
        else:
            warnings.warn("ShortConvolution is crucial. Do not turn it off unless you know what you are doing.")

        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps, dtype=torch.float32)

        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def _apply_laplacian_pointer(self, pos: torch.Tensor) -> torch.Tensor:
        indices = torch.arange(self.tape_size, device=pos.device, dtype=torch.float32).view(1, 1, 1, -1)
        pos = pos.unsqueeze(-1).to(torch.float32)
        gamma = F.softplus(self.pointer_gamma).to(torch.float32)
        distances = -gamma * torch.abs(indices - pos)
        return F.softmax(distances, dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:

        batch_size, q_len, _ = hidden_states.shape
        mode = 'fused_recurrent' if (q_len <= 64 and not self.training) else self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        last_state = get_layer_cache(self, past_key_values)

        assoc_state, tape_state, ptr_states, prev_tape_read = None, None, None, None
        conv_state_q, conv_state_k, conv_state_v = None, None, None

        if last_state is not None:
            if 'recurrent_state' in last_state and last_state['recurrent_state'] is not None:
                assoc_state, tape_state, ptr_states, prev_tape_read = last_state['recurrent_state']
            if self.use_short_conv and 'conv_state' in last_state:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']

        ptr_state_read, ptr_state_write = ptr_states if ptr_states is not None else (None, None)
        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None

        if attention_mask is not None and q_len > 1:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        # 1. Convolutions & Base Projections
        if self.use_short_conv:
            q, conv_state_q = self.q_conv1d(x=self.q_proj(hidden_states), cache=conv_state_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            k, conv_state_k = self.k_conv1d(x=self.k_proj(hidden_states), cache=conv_state_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            v, conv_state_v = self.v_conv1d(x=self.v_proj(hidden_states), cache=conv_state_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

        if self.num_v_heads > self.num_heads:
            q, k = map(lambda x: repeat(x, '... h d -> ... (h g) d', g=self.num_v_heads // self.num_heads), (q, k))

        # 2. Gated DeltaNet Variables (raw; sigmoid/gate math now happens inside the kernel)
        beta_assoc = self.b_proj(hidden_states)
        g_assoc = self.a_proj(hidden_states)

        # 3. Tape Variables & Pointers
        beta_tape = self.beta_tape_proj(hidden_states).sigmoid()
        if self.allow_neg_eigval:
            beta_tape = beta_tape * 2.0

        def compute_pointer(action_logits, ptr_state):
            action_probs = F.softmax(action_logits, dim=-1, dtype=torch.float32)
            p_right, p_left, p_stay = action_probs.unbind(dim=-1)

            delta_t = p_right - p_left
            active_ptr_state = ptr_state.to(torch.float32) if ptr_state is not None else None

            if ptr_state is not None and q_len == 1 and indices is not None:
                active_ptr_state = ptr_state[indices].to(torch.float32)

            if q_len == 1:
                pos_t = active_ptr_state.view(batch_size, 1, self.num_v_heads) + delta_t if active_ptr_state is not None else delta_t
                return pos_t, pos_t[:, -1]

            pos_t = torch.empty_like(delta_t)
            next_ptr = None

            if cu_seqlens is not None:
                seqlens_list = cu_seqlens.tolist()
                next_ptrs = []
                for i in range(len(seqlens_list) - 1):
                    start, end = seqlens_list[i], seqlens_list[i+1]
                    chunk_delta = delta_t[:, start:end, :]
                    chunk_pos = torch.cumsum(chunk_delta, dim=1)
                    if active_ptr_state is not None:
                        chunk_pos += active_ptr_state[i:i+1].view(1, 1, self.num_v_heads)
                    pos_t[:, start:end, :] = chunk_pos
                    next_ptrs.append(chunk_pos[:, -1])
                if use_cache:
                    next_ptr = torch.cat(next_ptrs, dim=0)
            else:
                pos_t = torch.cumsum(delta_t, dim=1)
                if active_ptr_state is not None:
                    pos_t += active_ptr_state.view(batch_size, 1, self.num_v_heads)
                if use_cache:
                    next_ptr = pos_t[:, -1]

            return pos_t, next_ptr

        read_logits = rearrange(self.read_action_proj(hidden_states), '... (h a) -> ... h a', h=self.num_v_heads, a=3)
        pos_read, next_ptr_read = compute_pointer(read_logits, ptr_state_read)
        q_tape = self._apply_laplacian_pointer(pos_read).to(v.dtype).contiguous()

        write_logits = rearrange(self.write_action_proj(hidden_states), '... (h a) -> ... h a', h=self.num_v_heads, a=3)
        pos_write, next_ptr_write = compute_pointer(write_logits, ptr_state_write)
        k_tape = self._apply_laplacian_pointer(pos_write).to(v.dtype).contiguous()

        # 4. Tape Delta Rule Update
        if mode == "fused_recurrent":
            v_tape_read, new_tape_state = fused_recurrent_delta_rule(
                q=q_tape, k=k_tape, v=v, beta=beta_tape, initial_state=tape_state,
                output_final_state=use_cache, cu_seqlens=cu_seqlens, use_qk_l2norm_in_kernel=False
            )
        else:
            v_tape_read, new_tape_state = chunk_delta_rule(
                q=q_tape, k=k_tape, v=v, beta=beta_tape, initial_state=tape_state,
                output_final_state=use_cache, cu_seqlens=cu_seqlens, use_qk_l2norm_in_kernel=False
            )

        # 5. Tape Shift & Augmented Value
        next_prev_tape_read = None
        if use_cache:
            if cu_seqlens is not None:
                next_prev_tape_read = v_tape_read[0, cu_seqlens[1:] - 1].unsqueeze(1).contiguous()
            else:
                next_prev_tape_read = v_tape_read[:, -1:].contiguous()

        if q_len == 1:
            shifted_val = prev_tape_read.squeeze(1)[indices].unsqueeze(0) if indices is not None else prev_tape_read
            v_aug = v - shifted_val if shifted_val is not None else v
        else:
            v_aug = fused_delay_shift(v, v_tape_read, cu_seqlens, prev_tape_read)

        # 6. Gated Delta Rule Update (Main Associative Memory)
        if mode == 'chunk':
            o, new_assoc_state = chunk_gated_delta_rule(
                q=q, k=k, v=v_aug, g=g_assoc, beta=beta_assoc,
                A_log=self.A_log, dt_bias=self.dt_bias,
                initial_state=assoc_state, output_final_state=use_cache, cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
                allow_neg_eigval=self.allow_neg_eigval, state_v_first=True,
            )
        else:
            o, new_assoc_state = fused_recurrent_gated_delta_rule(
                q=q, k=k, v=v_aug, g=g_assoc, beta=beta_assoc,
                A_log=self.A_log, dt_bias=self.dt_bias,
                initial_state=assoc_state, output_final_state=use_cache, cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
                allow_neg_eigval=self.allow_neg_eigval, state_v_first=True,
            )

        # 7. Update Cache
        recurrent_cache = (
            new_assoc_state.contiguous() if new_assoc_state is not None else None,
            new_tape_state.contiguous() if new_tape_state is not None else None,
            (next_ptr_read, next_ptr_write),
            next_prev_tape_read
        )

        update_layer_cache(
            self,
            past_key_values,
            recurrent_state=recurrent_cache,
            conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
            offset=q_len,
        )

        # 8. Output Gates and Projection
        if self.use_gate:
            g_out = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g_out)
        else:
            o = self.o_norm(o)

        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        if attention_mask is not None and q_len > 1:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values

    def set_tape_size(self, new_size: int):
        self.orig_tape_size = self.tape_size
        self.tape_size = new_size

    def reset_tape_size(self):
        self.tape_size = self.orig_tape_size
