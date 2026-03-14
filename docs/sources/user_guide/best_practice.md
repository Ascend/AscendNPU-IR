# Best practices

## Performance optimization

## Tiling strategy

### Case description

When migrating Triton kernels from GPU to NPU, the number of logical cores launched is often much larger than the number of physical cores, causing significant launch and scheduling overhead. When porting, adjust the tiling strategy to reduce the number of cores so that the number of logical cores is close to the number of physical cores. This example uses Triton.

```python
out = torch.gather(x, dim=1, index=idx)
```

Input:

| Input | Shape  |
|-------|--------|
| x     | (B, C) |
| idx   | (B, K) |

Output:

| Output | Shape  |
|--------|--------|
| out   | (B, K) |

### NPU vs CUDA code diff

```diff
@triton.jit
def gather_dim1_kernel(
        x_ptr,  # *x  [B, C]
        idx_ptr,  # *idx[B, K]
        out_ptr,  # *out[B, K]
        stride_xb, stride_xc,
        stride_ib, stride_ik,
        stride_ob, stride_ok,
        B, K,
        BLOCK_B: tl.constexpr,
        BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)  # 1 block per batch row
-   # GPU implementation
-   pid_k = tl.program_id(1)  # 1 block per K-tile
-   k_off = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
-   mask = k_off < K
-   idx = tl.load(idx_ptr + pid_b * stride_ib + k_off * stride_ik, mask=mask)  # [BLOCK_K]
-   x_val = tl.load(x_ptr + pid_b * stride_xb + idx * stride_xc, mask=mask)
-   tl.store(out_ptr + pid_b * stride_ob + k_off * stride_ok, x_val, mask=mask)

+   # NPU implementation: loop over K dimension
+   b_idx = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
+   b_mask = b_idx < B
+   for k_start in range(0, K, BLOCK_K):
+       ks = tl.arange(0, BLOCK_K)
+       k_mask = ks < K - k_start
+       idx_off = (b_idx[:, None] * stride_ib +
+                  (k_start + ks)[None, :] * stride_ik)
+       col_idx = tl.load(idx_ptr + idx_off, mask=b_mask[:, None] & k_mask)
+       x_off = (b_idx[:, None] * stride_xb +
+                col_idx * stride_xc)
+       x_val = tl.load(x_ptr + x_off, mask=b_mask[:, None] & k_mask)
+       out_off = (b_idx[:, None] * stride_ob +
+                  (k_start + ks)[None, :] * stride_ok)
+       tl.store(out_ptr + out_off, x_val, mask=b_mask[:, None] & k_mask)

# Invocation
B = 128  # batch dim
K = 64
BLOCK_B = 4
BLOCK_K = 128

- # GPU
- grid = (B, triton.cdiv(K, BLOCK_K))
+ # NPU
+ grid = (triton.cdiv(B, BLOCK_B),)

gather_dim1_kernel[grid](
    x, idx, out,
    x.stride(0), x.stride(1),
    idx.stride(0), idx.stride(1),
    out.stride(0), out.stride(1),
    B, K,
    BLOCK_B=BLOCK_B,
    BLOCK_K=BLOCK_K,
)
```

## Ascend-friendly kernel rewrite (vectorized compare)

### Case description

In the original GPU flow, i64/i32 compare operations cannot use the vector unit on NPU and fall back to scalar computation, reducing efficiency. Converting to fp32 and using vec_cast and vec_cmp enables vectorized execution. Note: masks in `tl.load` and `tl.store` are often auto-optimized to vector ops by the compiler; in this example `tl.where` requires manual conversion. This case uses LayerNorm to illustrate vectorized compare for tail-block handling.

### NPU vs CUDA code diff

```diff
-   xbar = tl.where(cols < N, X - mean, 0.0)
+   # Change cols (i64) to cols_cmp (f32) to enable vector processing
+   cols_cmp = cols.to(tl.float32)
+   xbar = tl.where(cols_cmp < N, x - mean, 0.0)
```

---

## Function and precision cases

This section covers common function or precision issues.

## Hang / timeout

### Delimitation

- **Symptom**: Kernel hang or timeout. Some hangs are related to hardware synchronization (intra-core, inter-core, or pipeline sync). If you encounter a hang, you can try passing the following options when invoking the kernel to change the binary’s sync behavior and avoid the hang.
- **Example**:

| Compile option | Value | Description |
|----------------|-------|-------------|
| **inject_barrier_all** | false (default). Set to true; if the hang disappears, intra-core sync is likely the cause. Applies to mix/aic/aiv kernels. |
| **inject_block_all**  | false (default). Set to true; if the hang disappears, inter-core sync is likely the cause. Applies to mix kernels. |

Example: for the GDN network’s `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` kernel, original invocation:

```python
chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
    k=k, v=u, w=w, v_new=v_new, g=g, gk=gk, h=h,
    h0=initial_state, ht=final_state,
    cu_seqlens=cu_seqlens, chunk_offsets=chunk_offsets,
    T=T, H=H, K=K, V=V, BT=BT,
)
```

Invocation with sync options to avoid hang:

```python
chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
    k=k, v=u, w=w, v_new=v_new, g=g, gk=gk, h=h,
    h0=initial_state, ht=final_state,
    cu_seqlens=cu_seqlens, chunk_offsets=chunk_offsets,
    T=T, H=H, K=K, V=V, BT=BT,
    inject_block_all=True,   # enable inter-core sync
    inject_barrier_all=True  # enable intra-core sync
)
```

### Invalid varlen arguments

For varlen-style kernels that randomly sample indices in seqlen, ensure indices are valid: strictly increasing and in the range [0, seqlen].

---

## UB overflow

### Triton argmax: 32B alignment then axis fusion wastes UB

MLIR snippet:

```mlir
%reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [256, 9, 11], strides: [99, 11, 1] : memref<?xi8, #hivm.address_space<gm>> to memref<256x9x11xi8, strided<[99, 11, 1]>, #hivm.address_space<gm>>
%2 = hivm.hir.pointer_cast(%c0_i64) : memref<256x32x11x1xi8, #hivm.address_space<ub>>
%subview = memref.subview %2[0, 0, 0, 0] [256, 9, 11, 1] [1, 1, 1, 1] : memref<256x32x11x1xi8, #hivm.address_space<ub>> to memref<256x9x11xi8, strided<[352, 11, 1]>, #hivm.address_space<ub>>
%collapse_shape = memref.collapse_shape %reinterpret_cast [[0], [1, 2]] : memref<256x9x11xi8, ...> into memref<256x99xi8, ...>
%collapse_shape_0 = memref.collapse_shape %subview [[0], [1, 2]] : memref<256x9x11xi8, ...> into memref<256x99xi8, ...>
hivm.hir.load ins(%collapse_shape : ...) outs(%collapse_shape_0 : ...) ...
```

- **Analysis**: Line 1: original data size 256x9x11xi8 in GM (kernel arg %arg3). Line 2: UB allocation 256x32x11x1xi8 for copying from GM to UB; the first axis is 32-byte aligned and an extra dimension is added on the last axis. Line 3: subview of the UB shape 256x32x11x1xi8 to get 256x9x11xi8. Lines 4–5: collapse_shape to 256x99xi8. Line 6: copy from GM 256x99xi8 to UB 256x99xi8.

- **Summary**: Original data 256x9x11xi8 is 25344 B; after loading from GM to UB, UB usage (256x32x11x1xi8) is 90112 B, about 3.5× the original size.

### Triton Not op: unreasonable lowering wastes UB

The Triton Not op is lowered in NPU IR to a sequence of VOR, VAND, VNOT, VAND; in many cases a single VNOT is sufficient. MLIR snippet:

```mlir
%2 = hivm.hir.pointer_cast(%c0_i64) : memref<65536xi8, #hivm.address_space<ub>>
hivm.hir.load ins(%reinterpret_cast : ...) outs(%2 : ...)
%3 = hivm.hir.pointer_cast(%c131072_i64) : memref<65536xi8, #hivm.address_space<ub>>
hivm.hir.vbrc ins(%c-1_i8 : i8) outs(%3 : ...)
%4 = hivm.hir.pointer_cast(%c65536_i64) : memref<65536xi8, #hivm.address_space<ub>>
hivm.hir.vor ins(%2, %3 : ...) outs(%4 : ...)
%5 = hivm.hir.pointer_cast(%c0_i64) : memref<65536xi8, #hivm.address_space<ub>>
hivm.hir.vand ins(%2, %3 : ...) outs(%5 : ...)
hivm.hir.vnot ins(%5 : ...) outs(%5 : ...)
%6 = hivm.hir.pointer_cast(%c65536_i64) : memref<65536xi8, #hivm.address_space<ub>>
hivm.hir.vand ins(%5, %4 : ...) outs(%6 : ...)
```

- **Analysis**: Input 65536xi8 in GM; copy to UB; allocate more UB; fill -1; then OR, AND, NOT, AND sequence. So a simple Not on input is implemented as (input|(-1)) & (!(input&(-1))), using 5×65536 B of UB.

- **Summary**: Prefer an implementation that lowers to a single VNOT where possible.

### Triton max_dim0 (int64): PlanMemory before HIVMLowerToLoops wastes UB

MLIR snippet:

```mlir
%2 = hivm.hir.pointer_cast(%c0_i64) : memref<2x4912xi64, #hivm.address_space<ub>>
%3 = hivm.hir.pointer_cast(%c78592_i64) : memref<1x4912xi64, #hivm.address_space<ub>>
%4 = hivm.hir.pointer_cast(%c117888_i64) : memref<9824xi64, #hivm.address_space<ub>>
hivm.hir.vreduce {already_initialize_init} <max> ins(%2 : ...) outs(%3 : ...) temp_buffer(%4 : ...) reduce_dims = [0]
```

- **Analysis**: Line 1: input 2x4912xi64 in UB (from GM). Line 2: output 1x4912xi64 in UB. Line 3: temp_buffer 9824xi64 for vreduce. Line 4: for int64 input, vreduce is later lowered to loop scalar ops and temp_buffer is removed.

- **Summary**: PlanMemory accounts for temp_buffer that is not used in the final computation, which can cause false UB overflow. The temp_buffer allocation rule needs to be adjusted before PlanMemory (e.g. do not reserve UB for temp_buffer that is later removed).

---

## D-cache

### Invalid address access

- **Symptom**: Inputs are valid and on the same device ID, but the kernel’s device ID is set incorrectly, so data cannot be read and D-cache read/write errors occur.
- **Wrong**: `A = torch.empty(shape, dtype)` (not on NPU).
- **Correct**: `A = torch.empty(shape, dtype).npu()` or `A = torch.empty(shape, dtype, device="npu:0").npu()`.

### Negative or overflowing offset

- **Symptom**: The offset value is a computed value in the IR.
- **Examples**: (1) The computed offset is negative, leading to incorrect read address. (2) The operator’s offset is represented as int32; if the value exceeds that range, i32 overflows.

### Use non-negative loop iter as memory index

- **Symptom**: The compiler analyzes and optimizes memory access; if the index involves complex control flow (e.g. loop indices causing out-of-bounds access), the compiler may not fully handle it. Prefer using non-negative for-loop iteration arguments as memory indices.

---

## Memory access

### Load introducing unexpected vtranspose and UB overflow

- **Symptom**: Compile or precision error; a sign of implicit transpose is innermost stride ≠ 1 and outer stride = 1.
- **Wrong example**:

```python
K_block_ptr = tl.make_block_ptr(
    base=K, shape=(HEAD_DIM, N_CTX), stride=(kk, kn),
    offsets=(0, 0), block_shape=(HEAD_DIM, BLOCK_N), order=(0, 1),
)
k = tl.load(K_block_ptr)
```

- **Correct example**:

```python
K_block_ptr = tl.make_block_ptr(
    base=K, shape=(N_CTX, HEAD_DIM), stride=(kn, kk),
    offsets=(0, 0), block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0),
)
k = tl.load(K_block_ptr)
trans_k = tl.trans(k)
```

### Implicit transpose in load

“Implicit transpose” means the load or store performs a transpose in one go, avoiding a separate transpose kernel or explicit data reorder. It is usually done by adjusting pointer shape and strides so that the access pattern swaps dimensions. You can use `tl.make_block_ptr(base, shape, strides, offsets, block_shape, order)` with `order` or strides to achieve this. For a matrix transpose with input A (M, K) and output B (K, M), each block can process a block of B and load the corresponding transposed block from A (e.g. with swapped strides), or load a normal block of A and use `tl.trans` before storing to B.

**Example** (transpose kernel with implicit transpose load):

```python
@triton.jit
def transpose_kernel(x_ptr, y_ptr, M, N,
                     stride_xm, stride_xn, stride_ym, stride_yn,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Matrix transpose Y = X^T; X (M,N), Y (N,M). Each block handles a (BLOCK_N, BLOCK_M) block of Y."""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    bn = pid_n * BLOCK_N
    bm = pid_m * BLOCK_M
    x_ptr_t = tl.make_block_ptr(base=x_ptr, shape=(N, M), strides=(stride_xn, stride_xm),
                                offsets=(bn, bm), block_shape=(BLOCK_N, BLOCK_M), order=(1, 0))
    y_ptr_b = tl.make_block_ptr(base=y_ptr, shape=(N, M), strides=(stride_ym, stride_yn),
                                offsets=(bn, bm), block_shape=(BLOCK_N, BLOCK_M), order=(1, 0))
    x_tile = tl.load(x_ptr_t, boundary_check=(0, 1))
    tl.store(y_ptr_b, x_tile, boundary_check=(0, 1))
```

### Use mayDiscretememaccess to avoid UB overflow

- **Symptom**: UB overflow can be caused by large tensor size (e.g. beyond 192 KB UB) or by non-contiguous access that expands the last axis (e.g. `<Nx1xf32>` becomes `<Nx8xf32>` for 32B alignment). Adding the `mayDiscretememaccess` compile hint degenerates the tensor op to scalar access and can avoid UB overflow.

- **Usage**: Add the hint on the value used in load/store:

```python
# For load: add hint on the loaded value
value = tl.load(pointer)
tl.compile_hint(value, "mayDiscretememaccess")

# For store: add hint on the value being stored
tl.compile_hint(value, "mayDiscretememaccess")
tl.store(pointer, value)
```

- **Example 1**:

```python
b_x = tl.load(x + o_t * D + o_d[:, None], mask=(m_t & m_d[:, None]), other=0)
tl.compile_hint(b_x, "mayDiscretememaccess")
```

- **Example 2** (column-major to row-major with Ascend extension):

```python
import triton.language.extra.cann.extension as extension
a = tl.load(A_block_ptr, boundary_check=(0, 1))
extension.compile_hint(a, "mayDiscretememaccess")
tl.store(B_block_ptr, a, boundary_check=(0, 1))
```

Before the hint, the IR uses tensor load/store and `linalg.transpose`; after the hint, it is lowered to scalar loops (e.g. `scf.for` with `tensor.extract`/`tensor.insert` and `DiscreteMemAccess` / `ExtractedLoadOrStore`).

---

## Baseline (compute)

- **TODO**: TRITON_INTERPRET mode (compute), GPU-specific logic (compute), to be added.

---

## Scenario-based debugging

This section gives guidance on Triton NPU kernel performance tuning.

### Use bitwise_mask for mask memory

#### Problem

On Ascend, boolean (i1) tensors are stored in GM as i8 (one byte). Triton Ascend loads i1 as i8; when the value is used as a condition mask (e.g. in `tl.where`), it may be converted back to i1, causing extra conversions and overhead. The `compile_hint("bitwise_mask")` tells the compiler to treat the tensor as a bitmask and use bitwise ops, avoiding conversions.

#### Usage

Add the hint on the condition used in `tl.where`:

```python
mask = tl.where(cond, value1, value2)
tl.compile_hint(cond, "bitwise_mask")
```

Mask pointer offsets must be computed correctly for the bitmask layout. See the figures in the Chinese version for layout.

#### Example

See [Ascend where kernel](https://gitcode.com/Ascend/triton-ascend/blob/master/ascend/examples/pytest_ut/test_where_lt.py). For i8 bitwise mask input, add the hint to the result of `tl.where`:

```python
res = tl.where(cond, in1, in0)
tl.extra.cann.extension.compile_hint(cond, "bitwise_mask")
tl.store(out_ptr0 + (xindex), res, xmask)
```

Because bitwise mask packs 8 i8 booleans into one i8, the mask assembly logic must be updated accordingly (e.g. expand byte to 8 bits when comparing with torch reference). See the Chinese version for the full test_where_lt_case1 example.

#### Limit

Only i8 mask is supported. Using bitwise_mask on other types (e.g. i16/i32) can hurt performance, so this feature is limited to i8.

### Dynamically generated mask

#### Problem

A common pattern is `tl.arange` followed by compare to form a lower-triangular mask. The hardware may not support i32/i64 compare and falls back to scalar. See the figures in the Chinese version.

#### Example (e.g. diffusion_attention-style)

```python
for idx_ingroup in range(GROUP_SIZE):
    idx_n = idx_group * GROUP_SIZE + idx_ingroup
    offs_r_local = tl.arange(0, BLOCK_C)[:, None]
    offs_c_local = tl.arange(0, BLOCK_C)[None, :]
    chunk_idx_r = offs_r_local // BLOCK_SIZE
    chunk_idx_c = offs_c_local // BLOCK_SIZE
    block_mask_bool = (
        (chunk_idx_r > chunk_idx_c)
        & (seq_st + idx_c * BLOCK_C + offs_r_local < seq_ed)
        & (seq_st + idx_c * BLOCK_C + offs_c_local < seq_ed)
    )
```

---

## Cube–Vector (CV)

### Use tile_cube_loop to avoid L1 overflow

#### Problem

The compiler currently analyzes tiling for a single matmul and does not consider the lifetime of other matmuls. When matmuls are triggered multiple times (e.g. `cube -> vector -> cube`), overlapping lifetimes can cause L1 overflow at runtime. Until lifetime analysis is improved, use the `tile_cube_loop` compile hint so the compiler can apply sub-tiling to the relevant matmul.

#### Usage

Add the hint on the result of the dot that needs sub-tiling:

```python
res = tl.dot(lhs, rhs)
tl.compile_hint(res, "tile_cube_loop", 2)
```

Example (Flash Attention–style): first dot `qk = tl.dot(q, trans_k)`, then vector ops (e.g. softmax), then `pv = tl.dot(p, v)`. The second dot should be hinted so that cube tiling accounts for L1:

```python
qk = tl.dot(q, trans_k)
# softmax, etc.
p = tl.math.exp(qk)
pv = tl.dot(p, v)
tl.compile_hint(pv, "tile_cube_loop", 2)
```

### Compile options (reference)

| Option | Meaning | Values |
|--------|---------|--------|
| multibuffer | Enable ping-pong pipeline | False (default), True |
| limit_auto_multi_buffer_of_local_buffer | Scope of ping-pong on-chip (L1, L0, UB). "no-limit" = no restriction; "no-l0c" = only outside L0 cache (default) | "no-limit", "no-l0c" |
| unit_flag | Cube output by block (alignment scenarios only) | False (default), True |
| limit_auto_multi_buffer_only_for_local_buffer | Enable CV pipeline in GM workspace. False = enable (default). Interface may change. | False, True |
| set_workspace_multibuffer | When above is false: CV parallelism degree. Ensure no data dependency. N = N CV ops in parallel. | 2 (default), 4 |
| tile_mix_vector_loop | When above is false: vector tile count (can autotune). | 1 (default), 2, 4 |
| tile_mix_cube_loop | When above is false: cube tile count (can autotune). | 1 (default), 2, 4 |

### Kernel split (load balance)

For attention, workload can be imbalanced across cores (e.g. lower-triangular mask so later cores do more work). Prefer splitting so that lighter and heavier work are grouped on the same core where possible. See [matrix multiplication optimized](https://gitee.com/guangpengz/triton-ascend/blob/master/ascend/examples/tutorials/13-matrix-multiplication-optimized.py).

### Kernel options to avoid timeout

#### Problem

Some hangs are related to hardware sync (intra-core, inter-core, or pipeline). You can try the following options when invoking the kernel to avoid hang:

```python
# Sync options
inject_block_all = True   # inter-core sync
inject_barrier_all = True # intra-core sync
# Pipeline options
limit_auto_multi_buffer_only_for_local_buffer = True  # disable CV pipeline in GM
multibuffer = False  # disable ping-pong
```

#### Example (GDN)

Original call:

```python
chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](k=k, v=u, w=w, v_new=v_new, g=g, gk=gk, h=h,
    h0=initial_state, ht=final_state, cu_seqlens=cu_seqlens, chunk_offsets=chunk_offsets,
    T=T, H=H, K=K, V=V, BT=BT)
```

With CV pipeline in GM disabled to avoid hang:

```python
chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
    k=k, v=u, w=w, v_new=v_new, g=g, gk=gk, h=h,
    h0=initial_state, ht=final_state, cu_seqlens=cu_seqlens, chunk_offsets=chunk_offsets,
    T=T, H=H, K=K, V=V, BT=BT,
    limit_auto_multi_buffer_only_for_local_buffer=True,
)
```

---

## Triton NPU programming tutorials

Triton NPU programming reference:

[https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/README.zh.md](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/README.zh.md)
