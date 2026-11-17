/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// A6 (dav-920r1) MTE gather-direction templates (design doc §3.4):
//
//   simd_gather_out_to_ub    GM scattered read  -> UB dense  (MTE2)
//   simd_indirect_load       x[off[i]] with mask/other       (MTE2)
//   simd_stride_load         base + i*S affine read           (MTE2)
//   simd_embedding_gather    whole-row indexed fetch          (MTE2)
//
// All four digest their op semantics while building the per-chunk s32/s64
// byte offset table (invalid lanes = -1, skipped by the engine) and then
// converge on gather_gm_to_ubuf through the shared skeleton in
// RegBase/MTEGatherScatter.h. Row-major contiguous dst tiles are written
// densely by the engine; non-contiguous (strided / broadcast) dst tiles take
// a per-lane scalar fallback identical to the SIMT core's addressing.
//
// The whole TU is 920r1-only: the C310 variant of meta_op.aiv compiles it
// as an empty translation unit (design doc §3.2).

#include "ArchDefs.h"
#if BISHENG_TPL_REGBASE

#include "RegBase/MTEGatherScatter.h"

#if BISHENG_TPL_920R1

// ---------------------------------------------------------------------------
// Cores: raw pointers + flat shape arrays, shared by all ranks.
// ---------------------------------------------------------------------------

// Scalar fallback for a non-contiguous dst tile: the engine path writes dst
// densely, which scrambles strided / broadcast tiles. This mirrors the engine
// path lane-for-lane -- same index resolution, window checks and address
// folding -- and only swaps the landing for per-dim stride addressing.
// Skipped lanes keep 0 (zero-fill contract).
template <typename DTYPE, typename ITYPE, int DIM>
__aiv__ __attribute__((always_inline)) static void
SimdGatherScalar(__gm__ DTYPE *src, __ubuf__ ITYPE *idx, __ubuf__ DTYPE *dst,
                 const int32_t *idx_sizes, const int32_t *idx_strides,
                 const int32_t *dst_strides, const int64_t index_boundary,
                 const int32_t dimension, const int64_t *src_strides,
                 const int32_t *end_offsets, const int32_t *start_offsets,
                 const uint32_t total_lanes) {
  MteGsDrainMte2();
  MteGsStridedFill<DTYPE, DIM>(dst, idx_sizes, dst_strides,
                               static_cast<DTYPE>(0));
  int32_t coords[5] = {0, 0, 0, 0, 0};
  for (uint32_t lane = 0; lane < total_lanes; ++lane) {
    int64_t elem_off = 0;
    if (MteGsGatherElemOff<ITYPE, DIM>(coords, idx, idx_strides, index_boundary,
                                       dimension, src_strides, end_offsets,
                                       start_offsets, elem_off)) {
      dst[MteGsStridedOffset(coords, dst_strides, DIM)] = src[elem_off];
    }
    MteGsAdvanceCoords(coords, idx_sizes, DIM);
  }
}

// gather (GM scattered read -> UB dense), design doc §3.4:
//   table[lane] = resolve(idx[lane]) landing address × elemBytes;
//   out-of-range / out-of-window lanes get -1.
// 1D ignores the stride/offset family (design doc §3.6); 2D+ folds all dims
// into one byte offset: the indexed dim multiplies srcStrides[dimension],
// the other dims multiply their own stride, the last-dim step is always 1
// (srcStrides[DIM-1] is not consumed).
template <typename DTYPE, typename ITYPE, int DIM>
__aiv__ __attribute__((always_inline)) static void
SimdGatherCore(__gm__ DTYPE *src, __ubuf__ ITYPE *idx, __ubuf__ DTYPE *dst,
               const int32_t *idx_sizes, const int32_t *idx_strides,
               const int32_t *dst_strides, const int64_t index_boundary,
               const int32_t dimension, const int64_t *src_strides,
               const int32_t *end_offsets, const int32_t *start_offsets,
               const bool const_pad_ctl, const uint8_t l2_cache_ctl,
               __ubuf__ MteGsOffsetType<ITYPE> *table) {
  static_assert(DIM >= 1 && DIM <= 5, "DIM must be in [1, 5]");
  using OffsetT = MteGsOffsetType<ITYPE>;
  constexpr uint32_t elem_bytes = sizeof(DTYPE);

  // Guard mirrors SimdIndexPutCore: an out-of-range indexed dim would make
  // every lane silently fall back to coords-only addressing.
  if (dimension < 0 || dimension >= DIM) {
    return;
  }

  uint64_t total = 1;
  for (int32_t d = 0; d < DIM; ++d) {
    total *= static_cast<uint64_t>(idx_sizes[d]);
  }
  const uint32_t total_lanes = MteGsClampLanes(total); // early-out when empty
  if (total_lanes == 0) {
    return;
  }

  if (!MteGsIsContiguous(idx_sizes, dst_strides, DIM)) {
    SimdGatherScalar<DTYPE, ITYPE, DIM>(
        src, idx, dst, idx_sizes, idx_strides, dst_strides, index_boundary,
        dimension, src_strides, end_offsets, start_offsets, total_lanes);
    return;
  }

  // Pre-pass: zero-fill the whole dst tile; skipped lanes keep 0
  // (zero-fill contract).
  MteGsDrainMte2();
  MteGsZeroFill(dst, total_lanes);

  // Chunked loop: scalar table build -> table visible -> MTE2 copy -> wait.
  // The scratch table tile is the caller-provided temp_buffer (see
  // MTEGatherScatter.h); it is planner-owned UB and cannot alias idx / dst.
  int32_t coords[5] = {0, 0, 0, 0, 0};
  for (uint32_t base = 0; base < total_lanes; base += MTE_GS_MAX_CHUNK) {
    const uint32_t n = total_lanes - base < MTE_GS_MAX_CHUNK
                           ? total_lanes - base
                           : MTE_GS_MAX_CHUNK;
    for (uint32_t lane = 0; lane < n; ++lane) {
      int64_t elem_off = 0;
      if (MteGsGatherElemOff<ITYPE, DIM>(
              coords, idx, idx_strides, index_boundary, dimension, src_strides,
              end_offsets, start_offsets, elem_off)) {
        table[lane] = static_cast<OffsetT>(elem_off * elem_bytes);
      } else {
        table[lane] = static_cast<OffsetT>(-1);
      }
      MteGsAdvanceCoords(coords, idx_sizes, DIM);
    }
    // Pad the tail chunk with -1 entries up to a 32B-aligned count; the
    // compact-mode tail padding is eliminated at the source (design §3.4).
    const uint32_t pad = MteGsTailPadEntries(n, elem_bytes);
    MteGsPadTableSkip(table, n, pad);
    MteGsGatherChunk(dst + base, src, table, n + pad, elem_bytes, elem_bytes,
                     const_pad_ctl, l2_cache_ctl);
  }
}

// Scalar fallback for a non-contiguous dst tile: the engine path writes dst
// densely, which scrambles strided / broadcast tiles. This mirrors the engine
// path lane-for-lane -- prefill `other` through the dst strides, then land
// mask / idx valid lanes with the same strided addressing -- and only swaps
// the landing for per-dim stride addressing. Negative idx entries are
// skipped, like the engine path's -1 table entries.
template <typename DTYPE, typename ITYPE, int DIM>
__aiv__ __attribute__((always_inline)) static void
SimdIndirectLoadScalar(__gm__ DTYPE *src, __ubuf__ ITYPE *idx,
                       __ubuf__ DTYPE *dst, const __ubuf__ bool *mask,
                       const __ubuf__ DTYPE *other, const int32_t *idx_sizes,
                       const int32_t *idx_strides, const int32_t *mask_strides,
                       const int32_t *other_strides, const int32_t *dst_strides,
                       const uint32_t total_lanes) {
  MteGsDrainMte2();
  {
    int32_t coords[5] = {0, 0, 0, 0, 0};
    for (uint32_t lane = 0; lane < total_lanes; ++lane) {
      dst[MteGsStridedOffset(coords, dst_strides, DIM)] =
          other[MteGsStridedOffset(coords, other_strides, DIM)];
      MteGsAdvanceCoords(coords, idx_sizes, DIM);
    }
  }
  int32_t coords[5] = {0, 0, 0, 0, 0};
  for (uint32_t lane = 0; lane < total_lanes; ++lane) {
    int32_t flat = 0;
    for (int32_t d = 0; d < DIM; ++d) {
      flat += coords[d] * idx_strides[d];
    }
    bool valid = true;
    int32_t mask_flat = 0;
    for (int32_t d = 0; d < DIM; ++d) {
      mask_flat += coords[d] * mask_strides[d];
    }
    if (!mask[mask_flat]) {
      valid = false;
    }
    if (valid) {
      const int64_t off = static_cast<int64_t>(idx[flat]);
      if (off >= 0) {
        dst[MteGsStridedOffset(coords, dst_strides, DIM)] = src[off];
      }
    }
    MteGsAdvanceCoords(coords, idx_sizes, DIM);
  }
}

// indirect load: dst[i] = mask[i] ? src[idx[i]] : other[i].
// Offsets are raw element offsets (no boundary in the ABI); mask=false lanes
// get a -1 table entry and keep the prefilled `other` (design doc §3.4).
template <typename DTYPE, typename ITYPE, int DIM>
__aiv__ __attribute__((always_inline)) static void
SimdIndirectLoadCore(__gm__ DTYPE *src, __ubuf__ ITYPE *idx,
                     __ubuf__ DTYPE *dst, const __ubuf__ bool *mask,
                     const __ubuf__ DTYPE *other, const int32_t *idx_sizes,
                     const int32_t *idx_strides, const int32_t *mask_strides,
                     const int32_t *other_strides, const int32_t *dst_strides,
                     __ubuf__ MteGsOffsetType<ITYPE> *table) {
  static_assert(DIM >= 1 && DIM <= 5, "DIM must be in [1, 5]");
  using OffsetT = MteGsOffsetType<ITYPE>;
  constexpr uint32_t elem_bytes = sizeof(DTYPE);

  uint64_t total = 1;
  for (int32_t d = 0; d < DIM; ++d) {
    total *= static_cast<uint64_t>(idx_sizes[d]);
  }
  const uint32_t total_lanes = MteGsClampLanes(total);
  if (total_lanes == 0) {
    return;
  }

  if (!MteGsIsContiguous(idx_sizes, dst_strides, DIM)) {
    SimdIndirectLoadScalar<DTYPE, ITYPE, DIM>(
        src, idx, dst, mask, other, idx_sizes, idx_strides, mask_strides,
        other_strides, dst_strides, total_lanes);
    return;
  }

  // Pre-pass: prefill the whole dst tile with `other` (replaces gather's
  // zero-fill).
  MteGsDrainMte2();
  MteGsPrefillOther<DTYPE, DIM>(dst, other, idx_sizes, other_strides,
                                total_lanes);

  int32_t coords[5] = {0, 0, 0, 0, 0};
  for (uint32_t base = 0; base < total_lanes; base += MTE_GS_MAX_CHUNK) {
    const uint32_t n = total_lanes - base < MTE_GS_MAX_CHUNK
                           ? total_lanes - base
                           : MTE_GS_MAX_CHUNK;
    for (uint32_t lane = 0; lane < n; ++lane) {
      int32_t flat = 0;
      for (int32_t d = 0; d < DIM; ++d) {
        flat += coords[d] * idx_strides[d];
      }
      bool valid = true;
      int32_t mask_flat = 0;
      for (int32_t d = 0; d < DIM; ++d) {
        mask_flat += coords[d] * mask_strides[d];
      }
      if (!mask[mask_flat]) {
        valid = false;
      }
      if (valid) {
        const int64_t off = static_cast<int64_t>(idx[flat]) * elem_bytes;
        table[lane] = static_cast<OffsetT>(off);
      } else {
        table[lane] = static_cast<OffsetT>(-1);
      }
      MteGsAdvanceCoords(coords, idx_sizes, DIM);
    }
    const uint32_t pad = MteGsTailPadEntries(n, elem_bytes);
    MteGsPadTableSkip(table, n, pad);
    // volatile / nonvolatile symbols behave identically; l2CacheCtl keeps
    // its default (the volatile peripheral-parameter mapping pends ISA
    // confirmation, design doc §3.4).
    MteGsGatherChunk(dst + base, src, table, n + pad, elem_bytes, elem_bytes,
                     /*const_pad_ctl=*/false, /*l2_cache_ctl=*/0);
  }
}

// Scalar fallback for a non-contiguous dst tile: the engine path writes dst
// densely, which scrambles strided / broadcast tiles. This mirrors the engine
// path lane-for-lane -- same bounded-load validity and affine address folding
// -- and only swaps the landing for per-dim stride addressing. Out-of-bound
// lanes keep the prefilled `other`.
template <typename DTYPE, typename ITYPE, int DIM>
__aiv__ __attribute__((always_inline)) static void
SimdStrideLoadScalar(__gm__ DTYPE *src, __ubuf__ DTYPE *dst,
                     const int64_t offset, const DTYPE other,
                     const int64_t *strides, const int64_t *load_sizes,
                     const int32_t *dst_sizes, const int32_t *dst_strides,
                     const uint32_t total_lanes) {
  MteGsDrainMte2();
  MteGsStridedFill<DTYPE, DIM>(dst, dst_sizes, dst_strides, other);
  int32_t coords[5] = {0, 0, 0, 0, 0};
  for (uint32_t lane = 0; lane < total_lanes; ++lane) {
    bool valid = true;
    int64_t elem_off = offset;
    for (int32_t d = 0; d < DIM; ++d) {
      if (static_cast<int64_t>(coords[d]) >= load_sizes[d]) {
        valid = false;
        break;
      }
      elem_off += static_cast<int64_t>(coords[d]) * strides[d];
    }
    if (valid) {
      dst[MteGsStridedOffset(coords, dst_strides, DIM)] = src[elem_off];
    }
    MteGsAdvanceCoords(coords, dst_sizes, DIM);
  }
}

// stride load: dst[i...] = src[offset + Σ coords_d × stride_d], other prefill.
// Affine table: no index memory is read (design doc §3.4). Lanes beyond the
// per-dim bounded load size are invalid and keep the prefilled `other`.
template <typename DTYPE, typename ITYPE, int DIM>
__aiv__ __attribute__((always_inline)) static void
SimdStrideLoadCore(__gm__ DTYPE *src, __ubuf__ DTYPE *dst, const int64_t offset,
                   const DTYPE other, const int64_t *strides,
                   const int64_t *numels, const int32_t *dst_sizes,
                   const int32_t *dst_strides,
                   __ubuf__ MteGsOffsetType<ITYPE> *table) {
  static_assert(DIM >= 1 && DIM <= 3, "DIM must be in [1, 3]");
  using OffsetT = MteGsOffsetType<ITYPE>;
  constexpr uint32_t elem_bytes = sizeof(DTYPE);

  uint64_t total = 1;
  for (int32_t d = 0; d < DIM; ++d) {
    total *= static_cast<uint64_t>(dst_sizes[d]);
  }
  const uint32_t total_lanes = MteGsClampLanes(total);
  if (total_lanes == 0) {
    return;
  }

  int64_t load_sizes[DIM];
  for (int32_t d = 0; d < DIM; ++d) {
    load_sizes[d] = MteGsStrideBoundedSize(strides[d], numels[d], dst_sizes[d]);
  }

  if (!MteGsIsContiguous(dst_sizes, dst_strides, DIM)) {
    SimdStrideLoadScalar<DTYPE, ITYPE, DIM>(src, dst, offset, other, strides,
                                            load_sizes, dst_sizes, dst_strides,
                                            total_lanes);
    return;
  }

  bool need_pad = false;
  for (int32_t d = 0; d < DIM; ++d) {
    if (load_sizes[d] < dst_sizes[d]) {
      need_pad = true;
    }
  }
  if (need_pad) {
    MteGsDrainMte2();
    MteGsZeroFill(dst, total_lanes);
    for (uint32_t lane = 0; lane < total_lanes; ++lane) {
      dst[lane] = other;
    }
  }

  // Row-burst fast path: when the innermost dim is a full unit-stride
  // segment (no bounded truncation on it), each dst row is one contiguous GM
  // run, so issue one table entry per row with burst_len = rowBytes instead
  // of one entry per element -- the op's segment semantics is preserved
  // rather than split into per-element bursts.
  const int64_t row_elems = dst_sizes[DIM - 1];
  if (DIM >= 2 && strides[DIM - 1] == 1 &&
      load_sizes[DIM - 1] == dst_sizes[DIM - 1]) {
    const uint32_t burst_bytes = static_cast<uint32_t>(row_elems) * elem_bytes;
    const uint32_t total_rows = total_lanes / static_cast<uint32_t>(row_elems);
    int32_t coords[5] = {0, 0, 0, 0, 0};
    for (uint32_t base = 0; base < total_rows; base += MTE_GS_MAX_CHUNK) {
      const uint32_t n = total_rows - base < MTE_GS_MAX_CHUNK
                             ? total_rows - base
                             : MTE_GS_MAX_CHUNK;
      for (uint32_t lane = 0; lane < n; ++lane) {
        bool valid = true;
        int64_t elem_off = offset;
        for (int32_t d = 0; d < DIM - 1; ++d) {
          if (static_cast<int64_t>(coords[d]) >= load_sizes[d]) {
            valid = false;
            break;
          }
          elem_off += static_cast<int64_t>(coords[d]) * strides[d];
        }
        table[lane] = valid ? static_cast<OffsetT>(elem_off * elem_bytes)
                            : static_cast<OffsetT>(-1);
        MteGsAdvanceCoords(coords, dst_sizes, DIM - 1);
      }
      const uint32_t pad = MteGsTailPadEntries(n, burst_bytes);
      MteGsPadTableSkip(table, n, pad);
      MteGsGatherChunk(dst + base * row_elems, src, table, n + pad, burst_bytes,
                       elem_bytes, /*const_pad_ctl=*/false,
                       /*l2_cache_ctl=*/0);
    }
    return;
  }

  int32_t coords[5] = {0, 0, 0, 0, 0};
  for (uint32_t base = 0; base < total_lanes; base += MTE_GS_MAX_CHUNK) {
    const uint32_t n = total_lanes - base < MTE_GS_MAX_CHUNK
                           ? total_lanes - base
                           : MTE_GS_MAX_CHUNK;
    for (uint32_t lane = 0; lane < n; ++lane) {
      bool valid = true;
      int64_t elem_off = offset;
      for (int32_t d = 0; d < DIM; ++d) {
        if (static_cast<int64_t>(coords[d]) >= load_sizes[d]) {
          valid = false;
          break;
        }
        elem_off += static_cast<int64_t>(coords[d]) * strides[d];
      }
      table[lane] = valid ? static_cast<OffsetT>(elem_off * elem_bytes)
                          : static_cast<OffsetT>(-1);
      MteGsAdvanceCoords(coords, dst_sizes, DIM);
    }
    const uint32_t pad = MteGsTailPadEntries(n, elem_bytes);
    MteGsPadTableSkip(table, n, pad);
    MteGsGatherChunk(dst + base, src, table, n + pad, elem_bytes, elem_bytes,
                     /*const_pad_ctl=*/false, /*l2_cache_ctl=*/0);
  }
}

// Scalar fallback for a non-contiguous dst tile: the engine path packs whole
// rows densely, which scrambles strided / broadcast tiles. This mirrors the
// engine path lane-for-lane -- same row index resolution and window checks --
// and only swaps the landing for per-dim stride addressing. Skipped rows keep
// 0 (zero-fill contract).
template <typename EMBTY, typename IDXTY, int DIM>
__aiv__ __attribute__((always_inline)) static void SimdEmbeddingGatherScalar(
    __gm__ EMBTY *src_base, __ubuf__ IDXTY *idx, __ubuf__ EMBTY *dst,
    const int32_t *idx_sizes, const int32_t *idx_strides,
    const int32_t *dst_strides, const int64_t bound, const int64_t *offsets,
    const int64_t *numels, const int32_t *full_sizes, const int32_t row_elems,
    const uint32_t total_rows) {
  MteGsDrainMte2();
  MteGsStridedFill<EMBTY, DIM + 1>(dst, full_sizes, dst_strides,
                                   static_cast<EMBTY>(0));
  int32_t coords[2] = {0, 0};
  for (uint32_t lane = 0; lane < total_rows; ++lane) {
    int32_t idx_flat = 0;
    for (int32_t d = 0; d < DIM; ++d) {
      idx_flat += coords[d] * idx_strides[d];
    }
    IDXTY idx_val = idx[idx_flat];
    const bool idx_valid = MteGsResolveIndex(idx_val, bound);

    bool in_window = true;
    for (int32_t d = 0; d < DIM; ++d) {
      const int32_t g = coords[d] + static_cast<int32_t>(offsets[d]);
      if (static_cast<uint32_t>(g) >= static_cast<uint32_t>(numels[d])) {
        in_window = false;
        break;
      }
    }
    if (in_window && idx_valid) {
      const int64_t row_base = static_cast<int64_t>(idx_val) * numels[DIM];
      const int64_t dst_row = MteGsStridedOffset(coords, dst_strides, DIM);
      // Per-column bound (SIMT: offset_last + x < numel_last): columns past
      // the row end keep the prefilled 0.
      for (int32_t x = 0; x < row_elems; ++x) {
        if (offsets[DIM] + static_cast<int64_t>(x) < numels[DIM]) {
          dst[dst_row + static_cast<int64_t>(x) * dst_strides[DIM]] =
              src_base[row_base + x];
        }
      }
    }
    MteGsAdvanceCoords(coords, idx_sizes, DIM);
  }
}

// embedding gather: one burst fetches a whole row of row_elems elements.
// table[lane] = resolve(idx[lane]) × numel_last × elemBytes from the column
// offset base; burstLen = rowBytes and the engine classifies the padding
// mode by row width (design doc §3.4; no special-casing here).
template <typename EMBTY, typename IDXTY, int DIM>
__aiv__ __attribute__((always_inline)) static void
SimdEmbeddingGatherCore(__gm__ EMBTY *emb, __ubuf__ IDXTY *idx,
                        __ubuf__ EMBTY *dst, const int32_t *idx_sizes,
                        const int32_t *idx_strides, const int32_t *dst_sizes,
                        const int32_t *dst_strides, const int64_t bound,
                        const int64_t *offsets, const int64_t *numels,
                        __ubuf__ MteGsOffsetType<IDXTY> *table) {
  static_assert(DIM == 1 || DIM == 2, "DIM must be 1 or 2");
  using OffsetT = MteGsOffsetType<IDXTY>;
  constexpr uint32_t elem_bytes = sizeof(EMBTY);

  uint64_t rows = 1;
  for (int32_t d = 0; d < DIM; ++d) {
    rows *= static_cast<uint64_t>(idx_sizes[d]);
  }
  const uint32_t total_rows = MteGsClampLanes(rows);
  if (total_rows == 0) {
    return;
  }
  const int32_t row_elems = dst_sizes[DIM];
  if (row_elems <= 0) {
    return;
  }
  const uint32_t row_bytes = static_cast<uint32_t>(row_elems) * elem_bytes;
  // Column offset folds into the GM base; the indexed row stride is the full
  // embedding row length (SIMT: emb[index * numel_last + offset_last + x]).
  __gm__ EMBTY *src_base = emb + offsets[DIM];

  // Full dst shape: idx rows x row_elems (the engine path packs rows
  // densely; the contiguity test below uses the same shape).
  int32_t full_sizes[3] = {1, 1, 1};
  for (int32_t d = 0; d < DIM; ++d) {
    full_sizes[d] = idx_sizes[d];
  }
  full_sizes[DIM] = row_elems;

  // Column bound: each row burst reads row_elems elements starting at
  // offsets[DIM]; when offsets[DIM] + row_elems crosses numels[DIM] the read
  // runs past the embedding row end, and the engine cannot mask a partial
  // row (one burst per lane, no column granularity). Route the whole tile to
  // the scalar fallback, which checks per column (SIMT parity).
  const bool col_overflow =
      offsets[DIM] + static_cast<int64_t>(row_elems) > numels[DIM];

  if (col_overflow || !MteGsIsContiguous(full_sizes, dst_strides, DIM + 1)) {
    SimdEmbeddingGatherScalar<EMBTY, IDXTY, DIM>(
        src_base, idx, dst, idx_sizes, idx_strides, dst_strides, bound, offsets,
        numels, full_sizes, row_elems, total_rows);
    return;
  }

  // Pre-pass: zero-fill the whole dst tile (out-of-range/window rows keep 0).
  const uint64_t dst_elems =
      static_cast<uint64_t>(total_rows) * static_cast<uint32_t>(row_elems);
  MteGsDrainMte2();
  for (uint64_t i = 0; i < dst_elems; ++i) {
    dst[i] = static_cast<EMBTY>(0);
  }

  int32_t coords[2] = {0, 0};
  for (uint32_t base = 0; base < total_rows; base += MTE_GS_MAX_CHUNK) {
    const uint32_t n = total_rows - base < MTE_GS_MAX_CHUNK ? total_rows - base
                                                            : MTE_GS_MAX_CHUNK;
    for (uint32_t lane = 0; lane < n; ++lane) {
      int32_t idx_flat = 0;
      for (int32_t d = 0; d < DIM; ++d) {
        idx_flat += coords[d] * idx_strides[d];
      }
      IDXTY idx_val = idx[idx_flat];
      const bool idx_valid = MteGsResolveIndex(idx_val, bound);

      bool in_window = true;
      for (int32_t d = 0; d < DIM; ++d) {
        const int32_t g = coords[d] + static_cast<int32_t>(offsets[d]);
        if (static_cast<uint32_t>(g) >= static_cast<uint32_t>(numels[d])) {
          in_window = false;
          break;
        }
      }
      if (in_window && idx_valid) {
        table[lane] = static_cast<OffsetT>(static_cast<int64_t>(idx_val) *
                                           numels[DIM] * elem_bytes);
      } else {
        table[lane] = static_cast<OffsetT>(-1);
      }
      MteGsAdvanceCoords(coords, idx_sizes, DIM);
    }
    const uint32_t pad = MteGsTailPadEntries(n, row_bytes);
    MteGsPadTableSkip(table, n, pad);
    // burstLen = rowBytes: one burst moves a whole row; the engine picks the
    // padding mode from the row width.
    MteGsGatherChunk(dst + static_cast<uint64_t>(base) * row_elems, src_base,
                     table, n + pad, row_bytes, row_bytes,
                     /*const_pad_ctl=*/false, /*l2_cache_ctl=*/0);
  }
}

// ---------------------------------------------------------------------------
// ciface adapters: unpack memrefs, forward to the cores.
// ---------------------------------------------------------------------------

template <int DIM, typename DTYPE, typename ITYPE>
__aiv__ __attribute__((always_inline)) void SimdGatherOutToUB(
    memref_t<__gm__ DTYPE, 1> *src, memref_t<__ubuf__ ITYPE, DIM> *idx,
    memref_t<__ubuf__ DTYPE, DIM> *dst, const int64_t *src_strides,
    const int32_t *end_offsets, const int32_t *start_offsets,
    const int32_t dimension, const int64_t index_boundary,
    const PadMode pad_mode, const PadValueT<DTYPE> pad_value,
    const EvictionPolicy eviction_policy, memref_t<__ubuf__ ITYPE, 1> *table) {
  static_assert(std::is_same<DTYPE, float>::value ||
                    std::is_same<DTYPE, float16_t>::value ||
                    std::is_same<DTYPE, bfloat16_t>::value ||
                    std::is_same<DTYPE, half>::value,
                "Currently dtype of data supports only "
                "float|float16_t|bfloat16_t|half");
  static_assert(std::is_same<ITYPE, int32_t>::value ||
                    std::is_same<ITYPE, int64_t>::value,
                "Currently dtype of index supports only int32|int64.");
  // Engine state setup: the pad register and the tail-padding source follow
  // padMode (design doc §3.4).
  const bool const_pad_ctl = MteGsSetupPadMode<DTYPE>(pad_mode, pad_value);
  const uint8_t l2_cache_ctl = MteGsL2CacheCtl(eviction_policy);

  int32_t idx_sizes[5];
  MteGsFillSizes<DIM>(idx->sizes, idx_sizes);
  int32_t idx_strides[5];
  MteGsFillSizes<DIM>(idx->strides, idx_strides);
  int32_t dst_strides[5];
  MteGsFillSizes<DIM>(dst->strides, dst_strides);
  SimdGatherCore<DTYPE, ITYPE, DIM>(
      reinterpret_cast<__gm__ DTYPE *>(src->aligned),
      reinterpret_cast<__ubuf__ ITYPE *>(idx->aligned + idx->offset),
      reinterpret_cast<__ubuf__ DTYPE *>(dst->aligned + dst->offset), idx_sizes,
      idx_strides, dst_strides, index_boundary, dimension, src_strides,
      end_offsets, start_offsets, const_pad_ctl, l2_cache_ctl,
      reinterpret_cast<__ubuf__ MteGsOffsetType<ITYPE> *>(table->aligned +
                                                          table->offset));
}

template <int DIM, typename DTYPE, typename ITYPE>
__aiv__ __attribute__((always_inline)) void SimdIndirectLoad(
    memref_t<__gm__ DTYPE, 1> *src, memref_t<__ubuf__ ITYPE, DIM> *idx,
    memref_t<__ubuf__ DTYPE, DIM> *dst, memref_t<__ubuf__ bool, DIM> *mask,
    memref_t<__ubuf__ DTYPE, DIM> *other, memref_t<__ubuf__ ITYPE, 1> *table) {
  static_assert(std::is_same<ITYPE, int32_t>::value ||
                    std::is_same<ITYPE, int64_t>::value,
                "Currently dtype of index supports only int32|int64.");
  int32_t idx_sizes[5];
  MteGsFillSizes<DIM>(idx->sizes, idx_sizes);
  int32_t idx_strides[5];
  MteGsFillSizes<DIM>(idx->strides, idx_strides);
  int32_t mask_strides[5];
  MteGsFillSizes<DIM>(mask->strides, mask_strides);
  int32_t other_strides[5];
  MteGsFillSizes<DIM>(other->strides, other_strides);
  int32_t dst_strides[5];
  MteGsFillSizes<DIM>(dst->strides, dst_strides);
  SimdIndirectLoadCore<DTYPE, ITYPE, DIM>(
      reinterpret_cast<__gm__ DTYPE *>(src->aligned),
      reinterpret_cast<__ubuf__ ITYPE *>(idx->aligned + idx->offset),
      reinterpret_cast<__ubuf__ DTYPE *>(dst->aligned + dst->offset),
      reinterpret_cast<__ubuf__ bool *>(mask->aligned + mask->offset),
      reinterpret_cast<__ubuf__ DTYPE *>(other->aligned + other->offset),
      idx_sizes, idx_strides, mask_strides, other_strides, dst_strides,
      reinterpret_cast<__ubuf__ MteGsOffsetType<ITYPE> *>(table->aligned +
                                                          table->offset));
}

template <int DIM, typename DTYPE, typename ITYPE>
__aiv__ __attribute__((always_inline)) void
SimdStrideLoad(memref_t<__gm__ DTYPE, 1> *src,
               memref_t<__ubuf__ DTYPE, DIM> *dst, const ITYPE offset,
               const DTYPE other, const ITYPE *strides, const ITYPE *numels,
               memref_t<__ubuf__ ITYPE, 1> *table) {
  int32_t dst_sizes[5];
  MteGsFillSizes<DIM>(dst->sizes, dst_sizes);
  int32_t dst_strides[5];
  MteGsFillSizes<DIM>(dst->strides, dst_strides);
  int64_t strides64[DIM];
  int64_t numels64[DIM];
  for (int32_t d = 0; d < DIM; ++d) {
    strides64[d] = static_cast<int64_t>(strides[d]);
    numels64[d] = static_cast<int64_t>(numels[d]);
  }
  SimdStrideLoadCore<DTYPE, ITYPE, DIM>(
      reinterpret_cast<__gm__ DTYPE *>(src->aligned + src->offset),
      reinterpret_cast<__ubuf__ DTYPE *>(dst->aligned + dst->offset),
      static_cast<int64_t>(offset), other, strides64, numels64, dst_sizes,
      dst_strides,
      reinterpret_cast<__ubuf__ MteGsOffsetType<ITYPE> *>(table->aligned +
                                                          table->offset));
}

template <int DIM, typename EMBTY, typename IDXTY>
__aiv__ __attribute__((always_inline)) void
SimdEmbeddingGather(memref_t<__gm__ EMBTY, 1> *emb,
                    memref_t<__ubuf__ IDXTY, DIM> *idx,
                    memref_t<__ubuf__ EMBTY, DIM + 1> *dst, const IDXTY bound,
                    const IDXTY *off_num, memref_t<__ubuf__ IDXTY, 1> *table) {
  static_assert(std::is_same<EMBTY, float>::value,
                "Currently dtype of embedding supports only float");
  static_assert(std::is_same<IDXTY, int32_t>::value ||
                    std::is_same<IDXTY, int64_t>::value,
                "Currently dtype of index supports only int32|int64");
  int32_t idx_sizes[5];
  MteGsFillSizes<DIM>(idx->sizes, idx_sizes);
  int32_t idx_strides[5];
  MteGsFillSizes<DIM>(idx->strides, idx_strides);
  int32_t dst_sizes[5];
  MteGsFillSizes<DIM + 1>(dst->sizes, dst_sizes);
  int32_t dst_strides[5];
  MteGsFillSizes<DIM + 1>(dst->strides, dst_strides);
  int64_t offsets[DIM + 1];
  int64_t numels[DIM + 1];
  for (int32_t d = 0; d < DIM + 1; ++d) {
    offsets[d] = static_cast<int64_t>(off_num[2 * d]);
    numels[d] = static_cast<int64_t>(off_num[2 * d + 1]);
  }
  SimdEmbeddingGatherCore<EMBTY, IDXTY, DIM>(
      reinterpret_cast<__gm__ EMBTY *>(emb->aligned),
      reinterpret_cast<__ubuf__ IDXTY *>(idx->aligned + idx->offset),
      reinterpret_cast<__ubuf__ EMBTY *>(dst->aligned + dst->offset), idx_sizes,
      idx_strides, dst_sizes, dst_strides, static_cast<int64_t>(bound), offsets,
      numels,
      reinterpret_cast<__ubuf__ MteGsOffsetType<IDXTY> *>(table->aligned +
                                                          table->offset));
}

// ---------------------------------------------------------------------------
// Symbol declarations / registrations. ABI is parameter-identical to the
// corresponding SIMT templates, only the symbol name gains the simd_ prefix
// (design doc §3.3); gather additionally carries pad/eviction trailing
// params with defaults, mirroring the load templates. Every symbol takes one
// extra trailing `table` parameter: the planner-owned UB scratch tile for
// the MTE offset table (the IR side's temp_buffer operand, always the last
// forwarded operand -- inserted before the defaulted pad/eviction params).
// ---------------------------------------------------------------------------

#define DECLARE_SIMD_GATHER_OUT_TO_UB(dim, dtype, itype, ...)                  \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_simd_gather_out_to_ub_##dim##d_##dtype##_##itype(           \
          memref_t<__gm__ dtype, 1> *src, memref_t<__ubuf__ itype, dim> *idx,  \
          memref_t<__ubuf__ dtype, dim> *dst, const int64_t indexBoundary,     \
          const int32_t dimension, __VA_ARGS__)

#define DECLARE_SIMD_GATHER_OUT_TO_UB_1D(dim, dtype, itype)                    \
  DECLARE_SIMD_GATHER_OUT_TO_UB(                                               \
      dim, dtype, itype, const int64_t srcStride0, const int32_t endOffset0,   \
      const int32_t startOffset0, memref_t<__ubuf__ itype, 1> *table,          \
      const PadMode pad_mode = PadMode::Null,                                  \
      const PadValueT<dtype> pad_value = 0,                                    \
      const EvictionPolicy eviction_policy = EvictionPolicy::EvictFirst)

#define REGISTER_SIMD_GATHER_OUT_TO_UB_1D(dim, dtype, itype)                   \
  DECLARE_SIMD_GATHER_OUT_TO_UB_1D(dim, dtype, itype) {                        \
    int64_t srcStrides[1] = {srcStride0};                                      \
    int32_t endOffsets[1] = {endOffset0};                                      \
    int32_t startOffsets[1] = {startOffset0};                                  \
    SimdGatherOutToUB<dim, dtype, itype>(                                      \
        src, idx, dst, srcStrides, endOffsets, startOffsets, dimension,        \
        indexBoundary, pad_mode, pad_value, eviction_policy, table);           \
  }

#define DECLARE_SIMD_GATHER_OUT_TO_UB_2D(dim, dtype, itype)                    \
  DECLARE_SIMD_GATHER_OUT_TO_UB(                                               \
      dim, dtype, itype, const int64_t srcStride0, const int64_t srcStride1,   \
      const int32_t endOffset0, const int32_t endOffset1,                      \
      const int32_t startOffset0, const int32_t startOffset1,                  \
      memref_t<__ubuf__ itype, 1> *table,                                      \
      const PadMode pad_mode = PadMode::Null,                                  \
      const PadValueT<dtype> pad_value = 0,                                    \
      const EvictionPolicy eviction_policy = EvictionPolicy::EvictFirst)

#define REGISTER_SIMD_GATHER_OUT_TO_UB_2D(dim, dtype, itype)                   \
  DECLARE_SIMD_GATHER_OUT_TO_UB_2D(dim, dtype, itype) {                        \
    int64_t srcStrides[2] = {srcStride0, srcStride1};                          \
    int32_t endOffsets[2] = {endOffset0, endOffset1};                          \
    int32_t startOffsets[2] = {startOffset0, startOffset1};                    \
    SimdGatherOutToUB<dim, dtype, itype>(                                      \
        src, idx, dst, srcStrides, endOffsets, startOffsets, dimension,        \
        indexBoundary, pad_mode, pad_value, eviction_policy, table);           \
  }

#define DECLARE_SIMD_GATHER_OUT_TO_UB_3D(dim, dtype, itype)                    \
  DECLARE_SIMD_GATHER_OUT_TO_UB(                                               \
      dim, dtype, itype, const int64_t srcStride0, const int64_t srcStride1,   \
      const int64_t srcStride2, const int32_t endOffset0,                      \
      const int32_t endOffset1, const int32_t endOffset2,                      \
      const int32_t startOffset0, const int32_t startOffset1,                  \
      const int32_t startOffset2, memref_t<__ubuf__ itype, 1> *table,          \
      const PadMode pad_mode = PadMode::Null,                                  \
      const PadValueT<dtype> pad_value = 0,                                    \
      const EvictionPolicy eviction_policy = EvictionPolicy::EvictFirst)

#define REGISTER_SIMD_GATHER_OUT_TO_UB_3D(dim, dtype, itype)                   \
  DECLARE_SIMD_GATHER_OUT_TO_UB_3D(dim, dtype, itype) {                        \
    int64_t srcStrides[3] = {srcStride0, srcStride1, srcStride2};              \
    int32_t endOffsets[3] = {endOffset0, endOffset1, endOffset2};              \
    int32_t startOffsets[3] = {startOffset0, startOffset1, startOffset2};      \
    SimdGatherOutToUB<dim, dtype, itype>(                                      \
        src, idx, dst, srcStrides, endOffsets, startOffsets, dimension,        \
        indexBoundary, pad_mode, pad_value, eviction_policy, table);           \
  }

#define DECLARE_SIMD_GATHER_OUT_TO_UB_4D(dim, dtype, itype)                    \
  DECLARE_SIMD_GATHER_OUT_TO_UB(                                               \
      dim, dtype, itype, const int64_t srcStride0, const int64_t srcStride1,   \
      const int64_t srcStride2, const int64_t srcStride3,                      \
      const int32_t endOffset0, const int32_t endOffset1,                      \
      const int32_t endOffset2, const int32_t endOffset3,                      \
      const int32_t startOffset0, const int32_t startOffset1,                  \
      const int32_t startOffset2, const int32_t startOffset3,                  \
      memref_t<__ubuf__ itype, 1> *table,                                      \
      const PadMode pad_mode = PadMode::Null,                                  \
      const PadValueT<dtype> pad_value = 0,                                    \
      const EvictionPolicy eviction_policy = EvictionPolicy::EvictFirst)

#define REGISTER_SIMD_GATHER_OUT_TO_UB_4D(dim, dtype, itype)                   \
  DECLARE_SIMD_GATHER_OUT_TO_UB_4D(dim, dtype, itype) {                        \
    int64_t srcStrides[4] = {srcStride0, srcStride1, srcStride2, srcStride3};  \
    int32_t endOffsets[4] = {endOffset0, endOffset1, endOffset2, endOffset3};  \
    int32_t startOffsets[4] = {startOffset0, startOffset1, startOffset2,       \
                               startOffset3};                                  \
    SimdGatherOutToUB<dim, dtype, itype>(                                      \
        src, idx, dst, srcStrides, endOffsets, startOffsets, dimension,        \
        indexBoundary, pad_mode, pad_value, eviction_policy, table);           \
  }

#define DECLARE_SIMD_GATHER_OUT_TO_UB_5D(dim, dtype, itype)                    \
  DECLARE_SIMD_GATHER_OUT_TO_UB(                                               \
      dim, dtype, itype, const int64_t srcStride0, const int64_t srcStride1,   \
      const int64_t srcStride2, const int64_t srcStride3,                      \
      const int64_t srcStride4, const int32_t endOffset0,                      \
      const int32_t endOffset1, const int32_t endOffset2,                      \
      const int32_t endOffset3, const int32_t endOffset4,                      \
      const int32_t startOffset0, const int32_t startOffset1,                  \
      const int32_t startOffset2, const int32_t startOffset3,                  \
      const int32_t startOffset4, memref_t<__ubuf__ itype, 1> *table,          \
      const PadMode pad_mode = PadMode::Null,                                  \
      const PadValueT<dtype> pad_value = 0,                                    \
      const EvictionPolicy eviction_policy = EvictionPolicy::EvictFirst)

#define REGISTER_SIMD_GATHER_OUT_TO_UB_5D(dim, dtype, itype)                   \
  DECLARE_SIMD_GATHER_OUT_TO_UB_5D(dim, dtype, itype) {                        \
    int64_t srcStrides[5] = {srcStride0, srcStride1, srcStride2, srcStride3,   \
                             srcStride4};                                      \
    int32_t endOffsets[5] = {endOffset0, endOffset1, endOffset2, endOffset3,   \
                             endOffset4};                                      \
    int32_t startOffsets[5] = {startOffset0, startOffset1, startOffset2,       \
                               startOffset3, startOffset4};                    \
    SimdGatherOutToUB<dim, dtype, itype>(                                      \
        src, idx, dst, srcStrides, endOffsets, startOffsets, dimension,        \
        indexBoundary, pad_mode, pad_value, eviction_policy, table);           \
  }

#define DECLARE_SIMD_INDIRECT_LOAD(dim, dtype, itype)                          \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_simd_indirect_load_##dim##d_##dtype##_##itype(              \
          memref_t<__gm__ dtype, 1> *src, memref_t<__ubuf__ itype, dim> *idx,  \
          memref_t<__ubuf__ dtype, dim> *dst,                                  \
          memref_t<__ubuf__ bool, dim> *mask,                                  \
          memref_t<__ubuf__ dtype, dim> *other,                                \
          memref_t<__ubuf__ itype, 1> *table)

#define DECLARE_SIMD_INDIRECT_LOAD_NONVOLATILE(dim, dtype, itype)              \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_simd_indirect_load_nonvolatile_##dim##d_##dtype##_##itype(  \
          memref_t<__gm__ dtype, 1> *src, memref_t<__ubuf__ itype, dim> *idx,  \
          memref_t<__ubuf__ dtype, dim> *dst,                                  \
          memref_t<__ubuf__ bool, dim> *mask,                                  \
          memref_t<__ubuf__ dtype, dim> *other,                                \
          memref_t<__ubuf__ itype, 1> *table)

// The volatile-specific peripheral parameters pend ISA confirmation; until
// then both symbol flavors behave identically (design doc §3.4).
#define REGISTE_SIMD_INDIRECT_LOAD(dim, dtype, itype)                          \
  DECLARE_SIMD_INDIRECT_LOAD(dim, dtype, itype) {                              \
    SimdIndirectLoad<dim, dtype, itype>(src, idx, dst, mask, other, table);    \
  }                                                                            \
  DECLARE_SIMD_INDIRECT_LOAD_NONVOLATILE(dim, dtype, itype) {                  \
    SimdIndirectLoad<dim, dtype, itype>(src, idx, dst, mask, other, table);    \
  }

#define DECLARE_SIMD_STRIDE_LOAD(dim, dtype, itype, ...)                       \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_simd_stride_load_##dim##d_##dtype##_##itype(                \
          memref_t<__gm__ dtype, 1> *src, memref_t<__ubuf__ dtype, dim> *dst,  \
          __VA_ARGS__)

#define DECLARE_SIMD_STRIDE_LOAD_1D(dtype, itype)                              \
  DECLARE_SIMD_STRIDE_LOAD(1, dtype, itype, itype offset, dtype other,         \
                           itype stride, itype numel,                          \
                           memref_t<__ubuf__ itype, 1> *table)

#define DECLARE_SIMD_STRIDE_LOAD_2D(dtype, itype)                              \
  DECLARE_SIMD_STRIDE_LOAD(2, dtype, itype, itype offset, dtype other,         \
                           itype stride0, itype stride1, itype numel0,         \
                           itype numel1, memref_t<__ubuf__ itype, 1> *table)

#define DECLARE_SIMD_STRIDE_LOAD_3D(dtype, itype)                              \
  DECLARE_SIMD_STRIDE_LOAD(3, dtype, itype, itype offset, dtype other,         \
                           itype stride0, itype stride1, itype stride2,        \
                           itype numel0, itype numel1, itype numel2,           \
                           memref_t<__ubuf__ itype, 1> *table)

#define REGISTE_SIMD_STRIDE_LOAD_1D(dtype, itype)                              \
  DECLARE_SIMD_STRIDE_LOAD_1D(dtype, itype) {                                  \
    const itype strides[1] = {stride};                                         \
    const itype numels[1] = {numel};                                           \
    SimdStrideLoad<1, dtype, itype>(src, dst, offset, other, strides, numels,  \
                                    table);                                    \
  }

#define REGISTE_SIMD_STRIDE_LOAD_2D(dtype, itype)                              \
  DECLARE_SIMD_STRIDE_LOAD_2D(dtype, itype) {                                  \
    const itype strides[2] = {stride0, stride1};                               \
    const itype numels[2] = {numel0, numel1};                                  \
    SimdStrideLoad<2, dtype, itype>(src, dst, offset, other, strides, numels,  \
                                    table);                                    \
  }

#define REGISTE_SIMD_STRIDE_LOAD_3D(dtype, itype)                              \
  DECLARE_SIMD_STRIDE_LOAD_3D(dtype, itype) {                                  \
    const itype strides[3] = {stride0, stride1, stride2};                      \
    const itype numels[3] = {numel0, numel1, numel2};                          \
    SimdStrideLoad<3, dtype, itype>(src, dst, offset, other, strides, numels,  \
                                    table);                                    \
  }

#define DECLARE_SIMD_EMBEDDINGGATHER(dim, embty, idxty, ...)                   \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_simd_embedding_gather_##dim##d_##embty##_##idxty(           \
          memref_t<__gm__ embty, 1> *emb, memref_t<__ubuf__ idxty, dim> *idx,  \
          memref_t<__ubuf__ embty, (dim) + 1> *dst, idxty bound, __VA_ARGS__)

#define DECLARE_SIMD_EMBEDDINGGATHER_1D(embty, idxty)                          \
  DECLARE_SIMD_EMBEDDINGGATHER(1, embty, idxty, idxty offset0, idxty offset1,  \
                               idxty numel0, idxty numel1,                     \
                               memref_t<__ubuf__ idxty, 1> *table)

#define DECLARE_SIMD_EMBEDDINGGATHER_2D(embty, idxty)                          \
  DECLARE_SIMD_EMBEDDINGGATHER(2, embty, idxty, idxty offset0, idxty offset1,  \
                               idxty offset2, idxty numel0, idxty numel1,      \
                               idxty numel2,                                   \
                               memref_t<__ubuf__ idxty, 1> *table)

#define REGISTE_SIMD_EMBEDDINGGATHER_1D(embty, idxty)                          \
  DECLARE_SIMD_EMBEDDINGGATHER_1D(embty, idxty) {                              \
    const idxty offNum[4] = {offset0, numel0, offset1, numel1};                \
    SimdEmbeddingGather<1, embty, idxty>(emb, idx, dst, bound, offNum, table); \
  }

#define REGISTE_SIMD_EMBEDDINGGATHER_2D(embty, idxty)                          \
  DECLARE_SIMD_EMBEDDINGGATHER_2D(embty, idxty) {                              \
    const idxty offNum[6] = {                                                  \
        offset0, numel0, offset1, numel1, offset2, numel2};                    \
    SimdEmbeddingGather<2, embty, idxty>(emb, idx, dst, bound, offNum, table); \
  }

// Per-op registration lists, dtype coverage identical to the SIMT templates
// (design doc §3.3 matrix; §3.6 keeps the old templates' dim coverage).
#define SIMD_GS_REG_GATHER_DIM(dim, itype)                                     \
  REGISTER_SIMD_GATHER_OUT_TO_UB_##dim##D(dim, float, itype);                  \
  REGISTER_SIMD_GATHER_OUT_TO_UB_##dim##D(dim, float16_t, itype);              \
  REGISTER_SIMD_GATHER_OUT_TO_UB_##dim##D(dim, bfloat16_t, itype);             \
  REGISTER_SIMD_GATHER_OUT_TO_UB_##dim##D(dim, half, itype);

#define REGISTE_SIMD_INDIRECT_LOAD_ALL_DTYPES(dim, itype)                      \
  REGISTE_SIMD_INDIRECT_LOAD(dim, float, itype);                               \
  REGISTE_SIMD_INDIRECT_LOAD(dim, half, itype);                                \
  REGISTE_SIMD_INDIRECT_LOAD(dim, bfloat16_t, itype);                          \
  REGISTE_SIMD_INDIRECT_LOAD(dim, int8_t, itype);                              \
  REGISTE_SIMD_INDIRECT_LOAD(dim, int16_t, itype);                             \
  REGISTE_SIMD_INDIRECT_LOAD(dim, int32_t, itype);                             \
  REGISTE_SIMD_INDIRECT_LOAD(dim, int64_t, itype);                             \
  REGISTE_SIMD_INDIRECT_LOAD(dim, uint8_t, itype);                             \
  REGISTE_SIMD_INDIRECT_LOAD(dim, uint16_t, itype);                            \
  REGISTE_SIMD_INDIRECT_LOAD(dim, uint32_t, itype);                            \
  REGISTE_SIMD_INDIRECT_LOAD(dim, uint64_t, itype);                            \
  REGISTE_SIMD_INDIRECT_LOAD(dim, bool, itype);                                \
  REGISTE_SIMD_INDIRECT_LOAD(dim, hifloat8_t, itype);                          \
  REGISTE_SIMD_INDIRECT_LOAD(dim, hifloat4x2_t, itype);                        \
  REGISTE_SIMD_INDIRECT_LOAD(dim, float8_e4m3_t, itype);                       \
  REGISTE_SIMD_INDIRECT_LOAD(dim, float8_e5m2_t, itype);

#define REGISTE_SIMD_STRIDE_LOAD_ALL_DTYPES_1D(dtype_itype)                    \
  REGISTE_SIMD_STRIDE_LOAD_1D dtype_itype;
#define REGISTE_SIMD_STRIDE_LOAD_ALL_DTYPES_2D(dtype_itype)                    \
  REGISTE_SIMD_STRIDE_LOAD_2D dtype_itype;
#define REGISTE_SIMD_STRIDE_LOAD_ALL_DTYPES_3D(dtype_itype)                    \
  REGISTE_SIMD_STRIDE_LOAD_3D dtype_itype;

#define SIMD_GS_STRIDE_DTYPES_X(dim_macro)                                     \
  dim_macro((float, int32_t)) dim_macro((half, int32_t)) dim_macro(            \
      (bfloat16_t, int32_t)) dim_macro((int8_t, int32_t))                      \
      dim_macro((int16_t, int32_t)) dim_macro((int32_t, int32_t)) dim_macro(   \
          (int64_t, int32_t)) dim_macro((uint8_t, int32_t))                    \
          dim_macro((uint16_t, int32_t)) dim_macro((uint32_t, int32_t))        \
              dim_macro((uint64_t, int32_t)) dim_macro((float, int64_t))       \
                  dim_macro((half, int64_t)) dim_macro((bfloat16_t, int64_t))  \
                      dim_macro((int8_t, int64_t)) dim_macro(                  \
                          (int16_t, int64_t)) dim_macro((int32_t, int64_t))    \
                          dim_macro((int64_t, int64_t))                        \
                              dim_macro((uint8_t, int64_t))                    \
                                  dim_macro((uint16_t, int64_t))               \
                                      dim_macro((uint32_t, int64_t))           \
                                          dim_macro((uint64_t, int64_t))

extern "C" {

// gather / scatter dtype: float|float16_t|bfloat16_t|half, itype: int32|int64.
SIMD_GS_REG_GATHER_DIM(1, int32_t)
SIMD_GS_REG_GATHER_DIM(1, int64_t)
SIMD_GS_REG_GATHER_DIM(2, int32_t)
SIMD_GS_REG_GATHER_DIM(2, int64_t)
SIMD_GS_REG_GATHER_DIM(3, int32_t)
SIMD_GS_REG_GATHER_DIM(3, int64_t)
SIMD_GS_REG_GATHER_DIM(4, int32_t)
SIMD_GS_REG_GATHER_DIM(4, int64_t)
SIMD_GS_REG_GATHER_DIM(5, int32_t)
SIMD_GS_REG_GATHER_DIM(5, int64_t)

// indirect load: all b8~b64 dtypes, 1D~5D.
REGISTE_SIMD_INDIRECT_LOAD_ALL_DTYPES(1, int32_t)
REGISTE_SIMD_INDIRECT_LOAD_ALL_DTYPES(2, int32_t)
REGISTE_SIMD_INDIRECT_LOAD_ALL_DTYPES(3, int32_t)
REGISTE_SIMD_INDIRECT_LOAD_ALL_DTYPES(4, int32_t)
REGISTE_SIMD_INDIRECT_LOAD_ALL_DTYPES(5, int32_t)
REGISTE_SIMD_INDIRECT_LOAD_ALL_DTYPES(1, int64_t)
REGISTE_SIMD_INDIRECT_LOAD_ALL_DTYPES(2, int64_t)
REGISTE_SIMD_INDIRECT_LOAD_ALL_DTYPES(3, int64_t)
REGISTE_SIMD_INDIRECT_LOAD_ALL_DTYPES(4, int64_t)
REGISTE_SIMD_INDIRECT_LOAD_ALL_DTYPES(5, int64_t)

// stride load: 1D~3D, itype: int32|int64.
SIMD_GS_STRIDE_DTYPES_X(REGISTE_SIMD_STRIDE_LOAD_ALL_DTYPES_1D)
SIMD_GS_STRIDE_DTYPES_X(REGISTE_SIMD_STRIDE_LOAD_ALL_DTYPES_2D)
SIMD_GS_STRIDE_DTYPES_X(REGISTE_SIMD_STRIDE_LOAD_ALL_DTYPES_3D)

// embedding gather: float, 1D/2D.
REGISTE_SIMD_EMBEDDINGGATHER_1D(float, int32_t);
REGISTE_SIMD_EMBEDDINGGATHER_1D(float, int64_t);
REGISTE_SIMD_EMBEDDINGGATHER_2D(float, int32_t);
REGISTE_SIMD_EMBEDDINGGATHER_2D(float, int64_t);

} // extern "C"

#endif // BISHENG_TPL_920R1
#endif // BISHENG_TPL_REGBASE
