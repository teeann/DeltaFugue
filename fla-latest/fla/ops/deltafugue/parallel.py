import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_delay_shift_kernel(
    v_ptr, v_tape_ptr, prev_tape_ptr, v_aug_ptr,
    is_start_ptr, seq_idx_ptr,
    total_tokens,
    stride_v_t, stride_v_h, stride_v_d,
    stride_vt_t, stride_vt_h, stride_vt_d,
    stride_p_b, stride_p_h, stride_p_d,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_PREV: tl.constexpr
):
    pid_t = tl.program_id(0)
    head = tl.program_id(1)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < total_tokens
    offs_d = tl.arange(0, BLOCK_D)

    # 1. Load V with explicit strides
    off_v = offs_t[:, None] * stride_v_t + head * stride_v_h + offs_d[None, :] * stride_v_d
    v_val = tl.load(v_ptr + off_v, mask=mask_t[:, None])

    # 2. Sequence Boundaries
    is_start = tl.load(is_start_ptr + offs_t, mask=mask_t, other=False)

    # 3. Safe Shift (Prevent out-of-bounds pointer calculation)
    prev_idx = tl.maximum(offs_t - 1, 0)
    off_prev_t = prev_idx[:, None] * stride_vt_t + head * stride_vt_h + offs_d[None, :] * stride_vt_d
    
    valid_prev = mask_t & (~is_start)
    tape_val = tl.load(v_tape_ptr + off_prev_t, mask=valid_prev[:, None], other=0.0)

    # 4. KV-Cache Routing
    if HAS_PREV:
        seq_idx = tl.load(seq_idx_ptr + offs_t, mask=mask_t, other=0)
        off_p = seq_idx[:, None] * stride_p_b + head * stride_p_h + offs_d[None, :] * stride_p_d
        valid_p = mask_t & is_start
        prev_val = tl.load(prev_tape_ptr + off_p, mask=valid_p[:, None], other=0.0)
        sub_val = tl.where(is_start[:, None], prev_val, tape_val)
    else:
        # THE FIX: If there is no KV-cache, the start token MUST subtract 0.0
        sub_val = tl.where(is_start[:, None], 0.0, tape_val)

    # 5. Compute and Store
    v_aug = v_val - sub_val
    tl.store(v_aug_ptr + off_v, v_aug, mask=mask_t[:, None])


@triton.jit
def _bwd_delay_shift_kernel(
    dv_aug_ptr, dv_tape_ptr,
    is_start_ptr,
    total_tokens,
    stride_dva_t, stride_dva_h, stride_dva_d,
    stride_dvt_t, stride_dvt_h, stride_dvt_d,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    pid_t = tl.program_id(0)
    head = tl.program_id(1)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < total_tokens
    offs_d = tl.arange(0, BLOCK_D)

    # Safe forward-shift for gradients
    is_next_start = tl.load(is_start_ptr + offs_t + 1, mask=(offs_t + 1 < total_tokens), other=True)
    next_idx = tl.minimum(offs_t + 1, total_tokens - 1)
    
    off_next_t = next_idx[:, None] * stride_dva_t + head * stride_dva_h + offs_d[None, :] * stride_dva_d
    valid_next = mask_t & (~is_next_start)
    
    dv_aug_next = tl.load(dv_aug_ptr + off_next_t, mask=valid_next[:, None], other=0.0)
    dv_tape = -dv_aug_next
    
    off_v = offs_t[:, None] * stride_dvt_t + head * stride_dvt_h + offs_d[None, :] * stride_dvt_d
    tl.store(dv_tape_ptr + off_v, dv_tape, mask=mask_t[:, None])


class FusedDelayShiftFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, v_tape_read, cu_seqlens=None, prev_tape_read=None):
        # Force contiguous memory to prevent stride mismatches
        v = v.contiguous()
        v_tape_read = v_tape_read.contiguous()
        
        if cu_seqlens is not None:
            total_tokens = v.shape[1]
            is_start = torch.zeros(total_tokens, dtype=torch.bool, device=v.device)
            is_start[cu_seqlens[:-1]] = True
            
            seq_idx = torch.zeros(total_tokens, dtype=torch.int32, device=v.device)
            seq_idx[cu_seqlens[1:-1]] = 1
            seq_idx = torch.cumsum(seq_idx, dim=0)
        else:
            B, T, H, D = v.shape
            total_tokens = B * T
            is_start = torch.zeros((B, T), dtype=torch.bool, device=v.device)
            is_start[:, 0] = True
            is_start = is_start.flatten()
            seq_idx = torch.arange(B, device=v.device).unsqueeze(1).expand(B, T).flatten().to(torch.int32)

        H, D = v.shape[-2], v.shape[-1]
        v_aug = torch.empty_like(v)

        v_flat = v.view(total_tokens, H, D)
        v_tape_flat = v_tape_read.view(total_tokens, H, D)
        v_aug_flat = v_aug.view(total_tokens, H, D)

        BLOCK_T = 128
        BLOCK_D = triton.next_power_of_2(D)
        grid = (triton.cdiv(total_tokens, BLOCK_T), H)

        HAS_PREV = prev_tape_read is not None
        if HAS_PREV:
            prev_tape_read = prev_tape_read.contiguous()
            stride_p_b = prev_tape_read.stride(0)
            stride_p_h = prev_tape_read.stride(2) # Dimension 2 is H
            stride_p_d = prev_tape_read.stride(3) # Dimension 3 is D
        else:
            stride_p_b, stride_p_h, stride_p_d = 0, 0, 0

        _fwd_delay_shift_kernel[grid](
            v_flat, v_tape_flat, prev_tape_read, v_aug_flat,
            is_start, seq_idx, total_tokens,
            v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
            v_tape_flat.stride(0), v_tape_flat.stride(1), v_tape_flat.stride(2),
            stride_p_b, stride_p_h, stride_p_d,
            BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D, HAS_PREV=HAS_PREV
        )

        ctx.save_for_backward(is_start)
        ctx.total_tokens = total_tokens
        ctx.H = H
        ctx.BLOCK_T = BLOCK_T
        ctx.BLOCK_D = BLOCK_D

        return v_aug

    @staticmethod
    def backward(ctx, dv_aug):
        is_start, = ctx.saved_tensors
        dv_aug = dv_aug.contiguous()
        
        dv = dv_aug 
        dv_tape_read = torch.empty_like(dv_aug)
        
        dv_aug_flat = dv_aug.view(ctx.total_tokens, ctx.H, -1)
        dv_tape_flat = dv_tape_read.view(ctx.total_tokens, ctx.H, -1)
        
        grid = (triton.cdiv(ctx.total_tokens, ctx.BLOCK_T), ctx.H)
        
        _bwd_delay_shift_kernel[grid](
            dv_aug_flat, dv_tape_flat, is_start,
            ctx.total_tokens,
            dv_aug_flat.stride(0), dv_aug_flat.stride(1), dv_aug_flat.stride(2),
            dv_tape_flat.stride(0), dv_tape_flat.stride(1), dv_tape_flat.stride(2),
            BLOCK_T=ctx.BLOCK_T, BLOCK_D=ctx.BLOCK_D
        )
        
        return dv, dv_tape_read, None, None

def fused_delay_shift(v, v_tape_read, cu_seqlens=None, prev_tape_read=None):
    return FusedDelayShiftFunction.apply(v, v_tape_read, cu_seqlens, prev_tape_read)
