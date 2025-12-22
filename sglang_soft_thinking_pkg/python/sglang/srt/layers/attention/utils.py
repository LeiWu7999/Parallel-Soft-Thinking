import triton
import triton.language as tl


@triton.jit
def create_flashinfer_kv_indices_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_indptr,
    kv_start_idx,
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)

    # find the req pool idx, this is for batch to token
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr + pid)

    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start
    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < kv_end - kv_start
        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + kv_start
            + offset,
            mask=mask,
        )
        tl.store(kv_indices_ptr + kv_indices_offset + offset, data, mask=mask)


@triton.jit
def compute_masked_kv_lens_triton(
    page_kernel_lens_ptr,  # [bs]
    kv_start_idx_ptr,  # [bs] or None
    kv_mask_starts_ptr,  # [bs, MAX_MASK_RANGES]
    kv_mask_ends_ptr,  # [bs, MAX_MASK_RANGES]
    out_kv_lens_ptr,  # [bs]
    kv_mask_ptr_stride: tl.constexpr,
    MAX_MASK_RANGES: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    kv_start = tl.cast(0, tl.int32)
    if kv_start_idx_ptr:
        kv_start = tl.load(kv_start_idx_ptr + pid).to(tl.int32)

    kv_len = tl.load(page_kernel_lens_ptr + pid).to(tl.int32)
    kv_end = kv_start + kv_len

    masked_len = tl.cast(0, tl.int32)
    for rid in tl.static_range(MAX_MASK_RANGES):
        mask_start = tl.load(kv_mask_starts_ptr + pid * kv_mask_ptr_stride + rid).to(
            tl.int32
        )
        mask_end = tl.load(kv_mask_ends_ptr + pid * kv_mask_ptr_stride + rid).to(
            tl.int32
        )

        # Unused slots are filled with -1.
        valid = mask_start >= 0

        start = tl.maximum(mask_start, kv_start)
        end = tl.minimum(mask_end + 1, kv_end)
        overlap = tl.maximum(0, end - start)
        masked_len += tl.where(valid, overlap, 0)

    out_len = tl.maximum(kv_len - masked_len, 0)
    tl.store(out_kv_lens_ptr + pid, out_len)


@triton.jit
def create_flashinfer_kv_indices_masked_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_indptr,
    kv_start_idx,  # [bs] or None
    kv_mask_starts_ptr,  # [bs, MAX_MASK_RANGES]
    kv_mask_ends_ptr,  # [bs, MAX_MASK_RANGES]
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
    kv_mask_ptr_stride: tl.constexpr,
    MAX_MASK_RANGES: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)

    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    out_offset = tl.load(kv_indptr + pid).to(tl.int32)

    kv_start = tl.cast(0, tl.int32)
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)

    kv_len = tl.load(page_kernel_lens_ptr + pid).to(tl.int32)
    kv_end = kv_start + kv_len

    cur = kv_start
    out_pos = out_offset

    # Copy "keep" segments between masked ranges into a compacted kv_indices.
    for rid in tl.static_range(MAX_MASK_RANGES):
        mask_start = tl.load(kv_mask_starts_ptr + pid * kv_mask_ptr_stride + rid).to(
            tl.int32
        )
        mask_end = tl.load(kv_mask_ends_ptr + pid * kv_mask_ptr_stride + rid).to(
            tl.int32
        )
        valid = mask_start >= 0

        skip_start = tl.maximum(mask_start, kv_start)
        skip_end = tl.minimum(mask_end + 1, kv_end)
        overlap = valid & (skip_start < skip_end)

        keep_len = tl.where(overlap, tl.maximum(skip_start - cur, 0), 0)
        num_loop = tl.cdiv(keep_len, BLOCK_SIZE)
        for i in range(num_loop):
            offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
            mask = offset < keep_len
            data = tl.load(
                req_to_token_ptr
                + req_pool_index * req_to_token_ptr_stride
                + cur
                + offset,
                mask=mask,
            )
            tl.store(kv_indices_ptr + out_pos + offset, data, mask=mask)

        out_pos += keep_len
        cur = tl.where(overlap & (skip_end > cur), skip_end, cur)

    tail_len = kv_end - cur
    num_loop = tl.cdiv(tail_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < tail_len
        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + cur
            + offset,
            mask=mask,
        )
        tl.store(kv_indices_ptr + out_pos + offset, data, mask=mask)


@triton.jit
def create_flashmla_kv_indices_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_start_idx,
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
    kv_indices_ptr_stride: tl.constexpr,
    PAGED_SIZE: tl.constexpr = 64,
):
    BLOCK_SIZE: tl.constexpr = 4096
    NUM_PAGE_PER_BLOCK: tl.constexpr = 64
    pid = tl.program_id(axis=0)

    # find the req pool idx, this is for batch to token
    req_pool_index = tl.load(req_pool_indices_ptr + pid)

    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start

    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    num_paged = tl.cdiv(kv_end - kv_start, PAGED_SIZE)
    num_pages_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)

    for i in range(num_pages_loop):
        paged_offset = (
            tl.arange(0, NUM_PAGE_PER_BLOCK) + i * NUM_PAGE_PER_BLOCK
        ) * PAGED_SIZE
        paged_offset_out = tl.arange(0, NUM_PAGE_PER_BLOCK) + i * NUM_PAGE_PER_BLOCK

        mask = paged_offset <= num_paged * PAGED_SIZE
        mask_out = paged_offset_out <= num_paged

        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + kv_start
            + paged_offset,
            mask=mask,
        )
        tl.store(
            kv_indices_ptr + pid * kv_indices_ptr_stride + paged_offset_out,
            data // PAGED_SIZE,
            mask=mask_out,
        )
