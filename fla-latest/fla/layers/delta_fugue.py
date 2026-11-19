# Copyright (c) 2026

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.ops.delta_rule import chunk_delta_rule, fused_recurrent_delta_rule
from fla.ops.deltafugue import fused_delay_shift

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack
    from fla.models.utils import Cache


def elu_p1(x):
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)


class DeltaFugue(nn.Module):
    """
    DeltaFugue: DeltaNet augmented with a bounded 1D tape memory.

    The tape is a delta-rule memory with `tape_size` slots, addressed by two
    independent pointers (read and write). Each pointer moves by a soft
    right / left / stay action per token; its position is the cumulative sum
    of (p_right - p_left), decoded into a soft address over the slots with a
    Laplacian kernel `softmax(-softplus(pointer_gamma) * |slot - pos|)`. The
    associative (DeltaNet) memory then stores the tape's prediction error:
    its value is `v - tape_read_{t-1}` (`fused_delay_shift`).

    `tape_size` can be changed at eval time with `set_tape_size` /
    `reset_tape_size` (e.g. train at 60 slots, evaluate at 256), since the
    pointer and kernel do not depend on it.
    """
    def __init__(
        self,
        mode: str = "chunk",
        d_model: int = None,
        hidden_size: int = 1024,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 4,
        use_beta: bool = True,
        use_gate: bool = False,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        allow_neg_eigval: bool = False,
        layer_idx: int = None,
        qk_activation: str = "silu",
        qk_norm: str = "l2",
        norm_eps: float = 1e-5,
        tape_size: int = 8,
        pointer_gamma_init: float = 2.0,
        **kwargs,
    ) -> DeltaFugue:
        super().__init__()

        self.tape_size = tape_size
        self.orig_tape_size = tape_size
        self.pointer_gamma = nn.Parameter(torch.tensor([float(pointer_gamma_init)]))  # sharpness = softplus(pointer_gamma)

        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm

        assert self.qk_activation in ["silu", "relu", "elu", "identity"]
        assert self.qk_norm in ["l2", "sum"]

        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.layer_idx = layer_idx

        assert mode in ["chunk", "fused_recurrent"], f"Not supported mode `{mode}`."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        # 1. Associative Projections
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        
        # 2. Dual-Head Tape Projections (3 Actions: Right, Left, Stay)
        self.read_action_proj = nn.Linear(hidden_size, self.num_heads * 3, bias=True)
        self.write_action_proj = nn.Linear(hidden_size, self.num_heads * 3, bias=True)
        self.beta_tape_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        # ---------------------------------------------------------
        # Orthogonal Action Basis Initialization
        # ---------------------------------------------------------
        with torch.no_grad():
            for h in range(self.num_heads):
                # Cycle through the 3 actions (0=Right, 1=Left, 2=Stay)
                dominant_action = h % 3 
                
                self.read_action_proj.bias[h * 3 + dominant_action] += 2.0
                self.write_action_proj.bias[h * 3 + dominant_action] += 2.0

        self.use_beta = use_beta
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(hidden_size=self.key_dim, kernel_size=conv_size, bias=conv_bias, activation="silu" if qk_activation == "silu" else None)
            self.k_conv1d = ShortConvolution(hidden_size=self.key_dim, kernel_size=conv_size, bias=conv_bias, activation="silu" if qk_activation == "silu" else None)
            self.v_conv1d = ShortConvolution(hidden_size=self.value_dim, kernel_size=conv_size, bias=conv_bias, activation="silu")

        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps, dtype=torch.float32)

        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)


    def _apply_laplacian_pointer(self, pos: torch.Tensor) -> torch.Tensor:
        B, T, H = pos.shape
        indices = torch.arange(self.tape_size, device=pos.device, dtype=torch.float32)
        indices = indices.view(1, 1, 1, -1)
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
        mode = "fused_recurrent" if q_len <= 64 else self.mode

        # ---------------------------------------------------------
        # A. State Retrieval (Dual-Head Cache)
        # ---------------------------------------------------------
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        assoc_state, tape_state, ptr_states, prev_tape_read = None, None, None, None
        conv_state_q, conv_state_k, conv_state_v = None, None, None
        
        if last_state is not None:
            assoc_state, tape_state, ptr_states, prev_tape_read = last_state["recurrent_state"]
            if self.use_short_conv:
                conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]

        ptr_state_read, ptr_state_write = ptr_states if ptr_states is not None else (None, None)

        # ---------------------------------------------------------
        # B. Sequence Unpadding
        # ---------------------------------------------------------
        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None

        # Single-token decoding runs unpadded, without cu_seqlens.
        if q_len == 1:
            cu_seqlens = None
            attention_mask = None
        elif attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(
                rearrange(hidden_states, "b s ... -> (b s) ..."), indices
            ).unsqueeze(0)

        # ---------------------------------------------------------
        # C. Process Associative Queries, Keys, and Values
        # ---------------------------------------------------------
        if self.use_short_conv:
            q_assoc, conv_state_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_state_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            k_assoc, conv_state_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_state_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            v, conv_state_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_state_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        else:
            q_assoc, k_assoc, v = self.q_proj(hidden_states), self.k_proj(hidden_states), F.silu(self.v_proj(hidden_states))
            if self.qk_activation == "silu":
                q_assoc, k_assoc = F.silu(q_assoc), F.silu(k_assoc)

        q_assoc = rearrange(q_assoc, 'b t (h d) -> b t h d', h=self.num_heads, d=self.head_k_dim)
        k_assoc = rearrange(k_assoc, 'b t (h d) -> b t h d', h=self.num_heads, d=self.head_k_dim)
        v = rearrange(v, 'b t (h d) -> b t h d', h=self.num_heads, d=self.head_v_dim)

        beta_assoc = self.b_proj(hidden_states).sigmoid() if self.use_beta else torch.ones_like(q_assoc[..., 0])
        beta_tape = self.beta_tape_proj(hidden_states).sigmoid()

        # allow_neg_eigval widens both write strengths from (0, 1) to (0, 2).
        # The formal-language experiments set it (models/fla.py); the HF / LM
        # configs leave it off.
        if self.allow_neg_eigval:
            beta_assoc = beta_assoc * 2.0
            beta_tape = beta_tape * 2.0

        if self.qk_norm == "sum":
            q_assoc = sum_norm(q_assoc).to(q_assoc)
            k_assoc = sum_norm(k_assoc).to(k_assoc)

        # ---------------------------------------------------------
        # D. Dual-Head Tape Logic (Cumulative Sum Integration)
        # ---------------------------------------------------------
        def compute_pointer(action_logits, ptr_state):
            # Softmax and cumsum in fp32, so the pointer does not drift on long sequences.
            action_probs = F.softmax(action_logits, dim=-1, dtype=torch.float32)
            p_right, p_left, p_stay = action_probs.unbind(dim=-1)

            delta_t = p_right - p_left
            active_ptr_state = ptr_state.to(torch.float32) if ptr_state is not None else None

            if ptr_state is not None and q_len == 1 and indices is not None:
                active_ptr_state = ptr_state[indices].to(torch.float32)

            # --- Autoregressive Decoding ---
            # The cached pointer is [B, H]; this step's position is [B, 1, H].
            if q_len == 1:
                if active_ptr_state is not None:
                    pos_t = active_ptr_state.view(batch_size, 1, self.num_heads) + delta_t
                else:
                    pos_t = delta_t
                return pos_t, pos_t[:, -1]

            # --- Prefill / Chunk Mode ---
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
                        chunk_pos += active_ptr_state[i:i+1].view(1, 1, self.num_heads)

                    pos_t[:, start:end, :] = chunk_pos
                    next_ptrs.append(chunk_pos[:, -1])

                if use_cache:
                    next_ptr = torch.cat(next_ptrs, dim=0)
            else:
                pos_t = torch.cumsum(delta_t, dim=1)
                if active_ptr_state is not None:
                    pos_t += active_ptr_state.view(batch_size, 1, self.num_heads)

                if use_cache:
                    next_ptr = pos_t[:, -1]

            return pos_t, next_ptr

        # Decode Read Head (3 Actions)
        read_logits = rearrange(self.read_action_proj(hidden_states), 'b t (h a) -> b t h a', h=self.num_heads, a=3)
        pos_read, next_ptr_read = compute_pointer(read_logits, ptr_state_read)
        q_tape = self._apply_laplacian_pointer(pos_read).to(v.dtype)

        # Decode Write Head (3 Actions)
        write_logits = rearrange(self.write_action_proj(hidden_states), 'b t (h a) -> b t h a', h=self.num_heads, a=3)
        pos_write, next_ptr_write = compute_pointer(write_logits, ptr_state_write)
        k_tape = self._apply_laplacian_pointer(pos_write).to(v.dtype)

        # ---------------------------------------------------------
        # E. Sequential Delta Rule Evaluation
        # ---------------------------------------------------------
        if mode == "fused_recurrent":
            v_tape_read, new_tape_state = fused_recurrent_delta_rule(
                q=q_tape, k=k_tape, v=v, beta=beta_tape, initial_state=tape_state, output_final_state=use_cache, cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=False,  # tape keys/queries are soft one-hot addresses; no l2 norm
            )
        else:
            v_tape_read, new_tape_state = chunk_delta_rule(
                q=q_tape, k=k_tape, v=v, beta=beta_tape, initial_state=tape_state, output_final_state=use_cache, cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=False,  # tape keys/queries are soft one-hot addresses; no l2 norm
            )

        # ---------------------------------------------------------
        # F. Extract KV-Cache state for the next step
        # ---------------------------------------------------------
        next_prev_tape_read = None
        if use_cache:
            if cu_seqlens is not None:
                next_prev_tape_read = v_tape_read[0, cu_seqlens[1:] - 1].unsqueeze(1).contiguous()
            else:
                next_prev_tape_read = v_tape_read[:, -1:].contiguous()

        # ---------------------------------------------------------
        # G. Augment V (Fused Triton Delay Shift)
        # ---------------------------------------------------------
        if q_len == 1:
            if prev_tape_read is not None:
                if indices is not None: 
                    shifted_val = prev_tape_read.squeeze(1)[indices].unsqueeze(0)
                else:
                    shifted_val = prev_tape_read
                v_aug = v - shifted_val
            else:
                v_aug = v 
        else:
            # Drop in the fully-fused Triton operation
            v_aug = fused_delay_shift(v, v_tape_read, cu_seqlens, prev_tape_read)

        if mode == "fused_recurrent":
            o, new_assoc_state = fused_recurrent_delta_rule(
                q=q_assoc, k=k_assoc, v=v_aug, beta=beta_assoc, initial_state=assoc_state, output_final_state=use_cache, cu_seqlens=cu_seqlens, use_qk_l2norm_in_kernel=(self.qk_norm == "l2")
            )
        else:
            o, new_assoc_state = chunk_delta_rule(
                q=q_assoc, k=k_assoc, v=v_aug, beta=beta_assoc, initial_state=assoc_state, output_final_state=use_cache, cu_seqlens=cu_seqlens, use_qk_l2norm_in_kernel=(self.qk_norm == "l2")
            )

        # ---------------------------------------------------------
        # F. Cache Update & Gate Projection
        # ---------------------------------------------------------
        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=(new_assoc_state, new_tape_state, (next_ptr_read, next_ptr_write), next_prev_tape_read),
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q_len,
            )

        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
            
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values
        
    def set_tape_size(self, new_size: int):
        self.orig_tape_size = self.tape_size
        self.tape_size = new_size

    def reset_tape_size(self):
        self.tape_size = self.orig_tape_size
