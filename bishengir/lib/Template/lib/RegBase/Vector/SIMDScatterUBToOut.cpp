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

// A6 (dav-920r1) MTE scatter-direction templates (design doc §3.4):
//
//   simd_scatter_ub_to_out    UB dense -> GM scattered write  (MTE3)
//   simd_indirect_store       dst[idx[i]] = src[i] (+mask)    (MTE3)
//   simd_stride_store         base + i*S affine write         (MTE3)
//   simd_index_put            flattened index-scatter write   (MTE3)
//
// All four digest their op semantics while building the per-chunk s32/s64
// byte offset table (invalid lanes = -1, skipped by the engine: nothing is
// written to GM) and then converge on scatter_ubuf_to_gm through the shared
// skeleton in RegBase/MTEGatherScatter.h. The UB value tile is consumed
// through per-dim strides (design doc §3.4): row-major contiguous tiles feed
// the engine densely, non-contiguous (strided / broadcast) tiles take a
// per-lane scalar fallback identical to the SIMT core's addressing; there is
// no dst prefill or zero-padding contract on the write direction.
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

// Scalar fallback for a non-contiguous src tile: the engine path reads src
// densely, which scrambles strided / broadcast tiles. This mirrors the engine
// path lane-for-lane -- same index resolution, window checks and address
// folding -- and only swaps the source consumption for per-dim stride
// addressing.
template <typename DTYPE, typename ITYPE, int DIM>
__aiv__ __attribute__((always_inline)) static void
SimdScatterScalar(__gm__ DTYPE *dst, __ubuf__ DTYPE *src, __ubuf__ ITYPE *idx,
                  const int32_t *idx_sizes, const int32_t *idx_strides,
                  const int32_t *src_strides, const int64_t index_boundary,
                  const int32_t dimension, const int64_t *dst_strides,
                  const int32_t *end_offsets, const int32_t *start_offsets,
                  const uint32_t total_lanes) {
  MteGsDrainMte3();
  int32_t coords[5] = {0, 0, 0, 0, 0};
  for (uint32_t lane = 0; lane < total_lanes; ++lane) {
    int64_t elem_off = 0;
    if (MteGsScatterElemOff<ITYPE, DIM>(coords, idx, idx_strides,
                                        index_boundary, dimension, dst_strides,
                                        end_offsets, start_offsets, elem_off)) {
      dst[elem_off] = src[MteGsStridedOffset(coords, src_strides, DIM)];
    }
    MteGsAdvanceCoords(coords, idx_sizes, DIM);
  }
}

// scatter (UB dense -> GM scattered write), design doc §3.4:
//   table[lane] = resolve(idx[lane]) landing address × elemBytes;
//   out-of-range / out-of-window lanes get -1 and the engine skips them,
//   writing nothing to GM.
// 1D consumes dstStride0 and the window args (SIMT 1D semantics:
// dst[idx*dstStride0]); 2D+ folds dims: the indexed dim multiplies
// dstStrides[dimension], the other dims multiply their own stride, the
// last-dim step is always 1 (dstStrides[DIM-1] is not consumed).
template <typename DTYPE, typename ITYPE, int DIM>
__aiv__ __attribute__((always_inline)) static void
SimdScatterCore(__gm__ DTYPE *dst, __ubuf__ DTYPE *src, __ubuf__ ITYPE *idx,
                const int32_t *idx_sizes, const int32_t *idx_strides,
                const int32_t *src_strides, const int64_t index_boundary,
                const int32_t dimension, const int64_t *dst_strides,
                const int32_t *end_offsets, const int32_t *start_offsets,
                __ubuf__ MteGsOffsetType<ITYPE> *table) {
  static_assert(DIM >= 1 && DIM <= 5, "DIM must be in [1, 5]");
  using OffsetT = MteGsOffsetType<ITYPE>;
  constexpr uint32_t elem_bytes = sizeof(DTYPE);

  uint64_t total = 1;
  for (int32_t d = 0; d < DIM; ++d) {
    total *= static_cast<uint64_t>(idx_sizes[d]);
  }
  const uint32_t total_lanes = MteGsClampLanes(total); // early-out when empty
  if (total_lanes == 0) {
    return;
  }

  if (!MteGsIsContiguous(idx_sizes, src_strides, DIM)) {
    SimdScatterScalar<DTYPE, ITYPE, DIM>(
        dst, src, idx, idx_sizes, idx_strides, src_strides, index_boundary,
        dimension, dst_strides, end_offsets, start_offsets, total_lanes);
    return;
  }

  // Drain MTE3 so earlier in-flight transfers cannot race the UB table.
  MteGsDrainMte3();

  // Chunked loop: scalar table build -> table visible -> MTE3 copy -> wait
  // (design doc §3.4 three-phase handshake). The scratch table tile is the
  // caller-provided temp_buffer (see MTEGatherScatter.h); it is planner-owned
  // UB and cannot alias idx / src.
  int32_t coords[5] = {0, 0, 0, 0, 0};
  for (uint32_t base = 0; base < total_lanes; base += MTE_GS_MAX_CHUNK) {
    const uint32_t n = total_lanes - base < MTE_GS_MAX_CHUNK
                           ? total_lanes - base
                           : MTE_GS_MAX_CHUNK;
    for (uint32_t lane = 0; lane < n; ++lane) {
      int64_t elem_off = 0;
      if (MteGsScatterElemOff<ITYPE, DIM>(
              coords, idx, idx_strides, index_boundary, dimension, dst_strides,
              end_offsets, start_offsets, elem_off)) {
        table[lane] = static_cast<OffsetT>(elem_off * elem_bytes);
      } else {
        table[lane] = static_cast<OffsetT>(-1);
      }
      MteGsAdvanceCoords(coords, idx_sizes, DIM);
    }
    // Pad the tail chunk with -1 entries up to a 32B-aligned count; the
    // engine skips them, producing no GM writes (design doc §3.4).
    const uint32_t pad = MteGsTailPadEntries(n, elem_bytes);
    MteGsPadTableSkip(table, n, pad);
    MteGsScatterChunk(dst, src + base, table, n + pad, elem_bytes, elem_bytes);
  }
}

// Scalar fallback for a non-contiguous src tile: the engine path reads src
// densely, which scrambles strided / broadcast tiles. This mirrors the engine
// path lane-for-lane -- same mask / idx resolution -- and only swaps the
// source consumption for per-dim stride addressing. Negative idx entries are
// skipped, like the engine path's -1 table entries.
template <typename DTYPE, typename ITYPE, int DIM, bool HAS_MASK>
__aiv__ __attribute__((always_inline)) static void
SimdIndirectStoreScalar(__gm__ DTYPE *dst, __ubuf__ ITYPE *idx,
                        __ubuf__ DTYPE *src, const __ubuf__ bool *mask,
                        const int32_t *idx_sizes, const int32_t *idx_strides,
                        const int32_t *mask_strides, const int32_t *src_strides,
                        const uint32_t total_lanes) {
  MteGsDrainMte3();
  int32_t coords[5] = {0, 0, 0, 0, 0};
  for (uint32_t lane = 0; lane < total_lanes; ++lane) {
    int32_t flat = 0;
    for (int32_t d = 0; d < DIM; ++d) {
      flat += coords[d] * idx_strides[d];
    }
    bool valid = true;
    if constexpr (HAS_MASK) {
      int32_t mask_flat = 0;
      for (int32_t d = 0; d < DIM; ++d) {
        mask_flat += coords[d] * mask_strides[d];
      }
      valid = mask[mask_flat];
    }
    if (valid) {
      const int64_t idx_off = static_cast<int64_t>(idx[flat]);
      if (idx_off >= 0) {
        dst[idx_off] = src[MteGsStridedOffset(coords, src_strides, DIM)];
      }
    }
    MteGsAdvanceCoords(coords, idx_sizes, DIM);
  }
}

// indirect store: dst[idx[i]] = src[i]; invalid lanes (mask=false, or the
// no_mask variant) get a -1 entry and are skipped. Offsets are raw element
// offsets (no boundary in the ABI, same as indirect load).
template <typename DTYPE, typename ITYPE, int DIM, bool HAS_MASK>
__aiv__ __attribute__((always_inline)) static void
SimdIndirectStoreCore(__gm__ DTYPE *dst, __ubuf__ ITYPE *idx,
                      __ubuf__ DTYPE *src, const __ubuf__ bool *mask,
                      const int32_t *idx_sizes, const int32_t *idx_strides,
                      const int32_t *mask_strides, const int32_t *src_strides,
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

  if (!MteGsIsContiguous(idx_sizes, src_strides, DIM)) {
    SimdIndirectStoreScalar<DTYPE, ITYPE, DIM, HAS_MASK>(
        dst, idx, src, mask, idx_sizes, idx_strides, mask_strides, src_strides,
        total_lanes);
    return;
  }

  MteGsDrainMte3();

  // The scratch table tile is the caller-provided temp_buffer (see
  // MTEGatherScatter.h); it is planner-owned UB and cannot alias idx / src.
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
      if constexpr (HAS_MASK) {
        int32_t mask_flat = 0;
        for (int32_t d = 0; d < DIM; ++d) {
          mask_flat += coords[d] * mask_strides[d];
        }
        valid = mask[mask_flat];
      }
      if (valid) {
        // Negative index guard: -1 is a common "skip" convention, but a raw
        // negative entry would target GM below dst (this is the only write
        // direction). Skip such lanes instead, like the other scatter cores.
        const int64_t idx_off = static_cast<int64_t>(idx[flat]);
        table[lane] = idx_off < 0 ? static_cast<OffsetT>(-1)
                                  : static_cast<OffsetT>(idx_off * elem_bytes);
      } else {
        table[lane] = static_cast<OffsetT>(-1);
      }
      MteGsAdvanceCoords(coords, idx_sizes, DIM);
    }
    const uint32_t pad = MteGsTailPadEntries(n, elem_bytes);
    MteGsPadTableSkip(table, n, pad);
    MteGsScatterChunk(dst, src + base, table, n + pad, elem_bytes, elem_bytes);
  }
}

// Scalar fallback for a non-contiguous src tile: the engine path reads src
// densely, which scrambles strided / broadcast tiles. This mirrors the engine
// path lane-for-lane -- same bounded-store validity and affine address folding
// -- and only swaps the source consumption for per-dim stride addressing.
template <typename DTYPE, typename ITYPE, int DIM>
__aiv__ __attribute__((always_inline)) static void
SimdStrideStoreScalar(__gm__ DTYPE *dst, __ubuf__ DTYPE *src,
                      const int64_t offset, const int64_t *strides,
                      const int64_t *store_sizes, const int32_t *src_sizes,
                      const int32_t *src_strides, const uint32_t total_lanes) {
  MteGsDrainMte3();
  int32_t coords[5] = {0, 0, 0, 0, 0};
  for (uint32_t lane = 0; lane < total_lanes; ++lane) {
    bool valid = true;
    int64_t elem_off = offset;
    for (int32_t d = 0; d < DIM; ++d) {
      if (static_cast<int64_t>(coords[d]) >= store_sizes[d]) {
        valid = false;
        break;
      }
      elem_off += static_cast<int64_t>(coords[d]) * strides[d];
    }
    if (valid) {
      dst[elem_off] = src[MteGsStridedOffset(coords, src_strides, DIM)];
    }
    MteGsAdvanceCoords(coords, src_sizes, DIM);
  }
}

// stride store: dst[offset + Σ coords_d × stride_d] = src[lane] (affine
// table, no index memory read). Lanes beyond the per-dim bounded store size
// get -1 and write nothing (mirrors the SIMT storeSize semantics); no
// other/pad contract.
template <typename DTYPE, typename ITYPE, int DIM>
__aiv__ __attribute__((always_inline)) static void SimdStrideStoreCore(
    __gm__ DTYPE *dst, __ubuf__ DTYPE *src, const int64_t offset,
    const int64_t *strides, const int64_t *numels, const int32_t *src_sizes,
    const int32_t *src_strides, __ubuf__ MteGsOffsetType<ITYPE> *table) {
  static_assert(DIM >= 1 && DIM <= 3, "DIM must be in [1, 3]");
  using OffsetT = MteGsOffsetType<ITYPE>;
  constexpr uint32_t elem_bytes = sizeof(DTYPE);

  uint64_t total = 1;
  for (int32_t d = 0; d < DIM; ++d) {
    total *= static_cast<uint64_t>(src_sizes[d]);
  }
  const uint32_t total_lanes = MteGsClampLanes(total);
  if (total_lanes == 0) {
    return;
  }

  int64_t store_sizes[DIM];
  for (int32_t d = 0; d < DIM; ++d) {
    store_sizes[d] =
        MteGsStrideBoundedSize(strides[d], numels[d], src_sizes[d]);
  }

  if (!MteGsIsContiguous(src_sizes, src_strides, DIM)) {
    SimdStrideStoreScalar<DTYPE, ITYPE, DIM>(dst, src, offset, strides,
                                             store_sizes, src_sizes,
                                             src_strides, total_lanes);
    return;
  }

  MteGsDrainMte3();

  // Row-burst fast path: when the innermost dim is a full unit-stride
  // segment (no bounded truncation on it), each src row is one contiguous GM
  // run, so issue one table entry per row with burst_len = rowBytes instead
  // of one entry per element -- the op's segment semantics is preserved
  // rather than split into per-element bursts.
  const int64_t row_elems = src_sizes[DIM - 1];
  if (DIM >= 2 && strides[DIM - 1] == 1 &&
      store_sizes[DIM - 1] == src_sizes[DIM - 1]) {
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
          if (static_cast<int64_t>(coords[d]) >= store_sizes[d]) {
            valid = false;
            break;
          }
          elem_off += static_cast<int64_t>(coords[d]) * strides[d];
        }
        table[lane] = valid ? static_cast<OffsetT>(elem_off * elem_bytes)
                            : static_cast<OffsetT>(-1);
        MteGsAdvanceCoords(coords, src_sizes, DIM - 1);
      }
      const uint32_t pad = MteGsTailPadEntries(n, burst_bytes);
      MteGsPadTableSkip(table, n, pad);
      MteGsScatterChunk(dst, src + base * row_elems, table, n + pad,
                        burst_bytes, elem_bytes);
    }
    return;
  }

  // The scratch table tile is the caller-provided temp_buffer (see
  // MTEGatherScatter.h); it is planner-owned UB and cannot alias src.
  int32_t coords[5] = {0, 0, 0, 0, 0};
  for (uint32_t base = 0; base < total_lanes; base += MTE_GS_MAX_CHUNK) {
    const uint32_t n = total_lanes - base < MTE_GS_MAX_CHUNK
                           ? total_lanes - base
                           : MTE_GS_MAX_CHUNK;
    for (uint32_t lane = 0; lane < n; ++lane) {
      bool valid = true;
      int64_t elem_off = offset;
      for (int32_t d = 0; d < DIM; ++d) {
        if (static_cast<int64_t>(coords[d]) >= store_sizes[d]) {
          valid = false;
          break;
        }
        elem_off += static_cast<int64_t>(coords[d]) * strides[d];
      }
      table[lane] = valid ? static_cast<OffsetT>(elem_off * elem_bytes)
                          : static_cast<OffsetT>(-1);
      MteGsAdvanceCoords(coords, src_sizes, DIM);
    }
    const uint32_t pad = MteGsTailPadEntries(n, elem_bytes);
    MteGsPadTableSkip(table, n, pad);
    MteGsScatterChunk(dst, src + base, table, n + pad, elem_bytes, elem_bytes);
  }
}

// Scalar fallback for a non-contiguous val tile: the engine path reads val
// densely, which scrambles strided / broadcast tiles. This mirrors the engine
// path lane-for-lane -- same index resolution, window checks and address
// folding -- and only swaps the source consumption for per-dim stride
// addressing.
template <typename SRCTY, typename IDXTY, int DIM>
__aiv__ __attribute__((always_inline)) static void
SimdIndexPutScalar(__gm__ SRCTY *dst, __ubuf__ SRCTY *val, __ubuf__ IDXTY *idx,
                   const int32_t *val_sizes, const int32_t *val_strides,
                   const int64_t index_boundary, const int32_t scatter_dim,
                   const int32_t idx_stride0, const int64_t *end_offsets,
                   const int64_t *start_offsets, const int64_t *dst_strides,
                   const uint32_t total_lanes) {
  MteGsDrainMte3();
  int32_t coords[5] = {0, 0, 0, 0, 0};
  for (uint32_t lane = 0; lane < total_lanes; ++lane) {
    IDXTY idx_val = idx[coords[scatter_dim] * idx_stride0];
    if (MteGsResolveIndex(idx_val, index_boundary)) {
      int64_t elem_off = 0;
      bool valid = true;
      for (int32_t d = 0; d < DIM; ++d) {
        const int64_t g = (d == scatter_dim) ? static_cast<int64_t>(idx_val)
                                             : coords[d] + start_offsets[d];
        if (static_cast<uint64_t>(g) >= static_cast<uint64_t>(end_offsets[d])) {
          valid = false;
          break;
        }
        elem_off += g * dst_strides[d];
      }
      if (valid) {
        int64_t val_off = 0;
        for (int32_t d = 0; d < DIM; ++d) {
          val_off += static_cast<int64_t>(coords[d]) * val_strides[d];
        }
        dst[elem_off] = val[val_off];
      }
    }
    MteGsAdvanceCoords(coords, val_sizes, DIM);
  }
}

// index put: val is consumed densely lane by lane (design doc §3.4); the
// address folding mirrors SIMT:
//   didx[d] = (d == scatterDim) ? resolve(idx[vidx[scatterDim]])
//                               : vidx[d] + startOffsets[d]
//   table[lane] = Σ didx[d] × dstStrides[d] × elemBytes (all DIM strides
//   consumed, same as the SIMT core); out-of-window/range lanes get -1.
// A non-contiguous val tile (strided view or broadcast) cannot feed the
// engine, which reads the source densely; such tiles take a scalar-store
// fallback that consumes val through its per-dim strides, lane-for-lane
// identical to the SIMT core's val[Σ vidx[d] × val_strides[d]] addressing.
template <typename SRCTY, typename IDXTY, int DIM>
__aiv__ __attribute__((always_inline)) static void
SimdIndexPutCore(__gm__ SRCTY *dst, __ubuf__ SRCTY *val, __ubuf__ IDXTY *idx,
                 const int32_t *val_sizes, const int32_t *val_strides,
                 const int64_t index_boundary, const int32_t scatter_dim,
                 const int32_t idx_stride0, const int64_t *end_offsets,
                 const int64_t *start_offsets, const int64_t *dst_strides,
                 __ubuf__ MteGsOffsetType<IDXTY> *table) {
  static_assert(DIM >= 2 && DIM <= 5, "DIM must be in [2, 5]");
  using OffsetT = MteGsOffsetType<IDXTY>;
  constexpr uint32_t elem_bytes = sizeof(SRCTY);

  if (scatter_dim < 0 || scatter_dim >= DIM) {
    return;
  }

  uint64_t total = 1;
  for (int32_t d = 0; d < DIM; ++d) {
    total *= static_cast<uint64_t>(val_sizes[d]);
  }
  const uint32_t total_lanes = MteGsClampLanes(total);
  if (total_lanes == 0) {
    return;
  }

  if (!MteGsIsContiguous(val_sizes, val_strides, DIM)) {
    SimdIndexPutScalar<SRCTY, IDXTY, DIM>(
        dst, val, idx, val_sizes, val_strides, index_boundary, scatter_dim,
        idx_stride0, end_offsets, start_offsets, dst_strides, total_lanes);
    return;
  }

  MteGsDrainMte3();

  // The scratch table tile is the caller-provided temp_buffer (see
  // MTEGatherScatter.h); it is planner-owned UB and cannot alias idx / val.
  // Chunked loop: build table -> table visible -> MTE3 write -> wait.
  int32_t coords[5] = {0, 0, 0, 0, 0};
  for (uint32_t base = 0; base < total_lanes; base += MTE_GS_MAX_CHUNK) {
    const uint32_t n = total_lanes - base < MTE_GS_MAX_CHUNK
                           ? total_lanes - base
                           : MTE_GS_MAX_CHUNK;
    for (uint32_t lane = 0; lane < n; ++lane) {
      IDXTY idx_val = idx[coords[scatter_dim] * idx_stride0];
      bool valid = MteGsResolveIndex(idx_val, index_boundary);

      int64_t elem_off = 0;
      if (valid) {
        for (int32_t d = 0; d < DIM; ++d) {
          const int64_t g = (d == scatter_dim) ? static_cast<int64_t>(idx_val)
                                               : coords[d] + start_offsets[d];
          if (static_cast<uint64_t>(g) >=
              static_cast<uint64_t>(end_offsets[d])) {
            valid = false;
            break;
          }
          elem_off += g * dst_strides[d];
        }
      }
      table[lane] = valid ? static_cast<OffsetT>(elem_off * elem_bytes)
                          : static_cast<OffsetT>(-1);
      MteGsAdvanceCoords(coords, val_sizes, DIM);
    }
    const uint32_t pad = MteGsTailPadEntries(n, elem_bytes);
    MteGsPadTableSkip(table, n, pad);
    MteGsScatterChunk(dst, val + base, table, n + pad, elem_bytes, elem_bytes);
  }
}

// ---------------------------------------------------------------------------
// ciface adapters: unpack memrefs, forward to the cores. Pointer conventions
// mirror the SIMT templates: scatter / indirect store / index put take the
// GM dst without offset, stride store applies dst->offset; UB operands use
// aligned + offset.
// ---------------------------------------------------------------------------

template <int DIM, typename DTYPE, typename ITYPE>
__aiv__ __attribute__((always_inline)) void SimdScatterUBToOut(
    memref_t<__gm__ DTYPE, 1> *dst, memref_t<__ubuf__ DTYPE, DIM> *src,
    memref_t<__ubuf__ ITYPE, DIM> *index, const int64_t indexBoundary,
    const int32_t dim, const int64_t *dstStrides, const int32_t *endOffsets,
    const int32_t *startOffsets, memref_t<__ubuf__ ITYPE, 1> *table) {
  static_assert(std::is_same<DTYPE, float>::value ||
                    std::is_same<DTYPE, float16_t>::value ||
                    std::is_same<DTYPE, bfloat16_t>::value ||
                    std::is_same<DTYPE, half>::value,
                "Currently DTYPE of data supports only "
                "float|float16_t|bfloat16_t|half.");
  static_assert(std::is_same<ITYPE, int32_t>::value ||
                    std::is_same<ITYPE, int64_t>::value,
                "Currently DTYPE of indices supports only int32|int64.");
  int32_t idx_sizes[5];
  MteGsFillSizes<DIM>(index->sizes, idx_sizes);
  int32_t idx_strides[5];
  MteGsFillSizes<DIM>(index->strides, idx_strides);
  int32_t src_strides[5];
  MteGsFillSizes<DIM>(src->strides, src_strides);
  SimdScatterCore<DTYPE, ITYPE, DIM>(
      reinterpret_cast<__gm__ DTYPE *>(dst->aligned),
      reinterpret_cast<__ubuf__ DTYPE *>(src->aligned + src->offset),
      reinterpret_cast<__ubuf__ ITYPE *>(index->aligned + index->offset),
      idx_sizes, idx_strides, src_strides, indexBoundary, dim, dstStrides,
      endOffsets, startOffsets,
      reinterpret_cast<__ubuf__ MteGsOffsetType<ITYPE> *>(table->aligned +
                                                          table->offset));
}

template <int DIM, typename DTYPE, typename ITYPE, bool HAS_MASK>
__aiv__ __attribute__((always_inline)) void SimdIndirectStore(
    memref_t<__gm__ DTYPE, 1> *dst, memref_t<__ubuf__ ITYPE, DIM> *idx,
    memref_t<__ubuf__ DTYPE, DIM> *src, memref_t<__ubuf__ bool, DIM> *mask,
    memref_t<__ubuf__ ITYPE, 1> *table) {
  static_assert(std::is_same<DTYPE, float>::value ||
                    std::is_same<DTYPE, half>::value ||
                    std::is_same<DTYPE, bfloat16_t>::value ||
                    std::is_same<DTYPE, int8_t>::value ||
                    std::is_same<DTYPE, int16_t>::value ||
                    std::is_same<DTYPE, int32_t>::value ||
                    std::is_same<DTYPE, int64_t>::value ||
                    std::is_same<DTYPE, uint8_t>::value ||
                    std::is_same<DTYPE, uint16_t>::value ||
                    std::is_same<DTYPE, uint32_t>::value ||
                    std::is_same<DTYPE, uint64_t>::value ||
                    std::is_same<DTYPE, bool>::value ||
                    std::is_same<DTYPE, hifloat8_t>::value ||
                    std::is_same<DTYPE, hifloat4x2_t>::value ||
                    std::is_same<DTYPE, float8_e4m3_t>::value ||
                    std::is_same<DTYPE, float8_e5m2_t>::value,
                "Currently dtype of data supports only "
                "float|half|bfloat16_t|int8_t|int16_t|int32_t|int64_t|"
                "bool|hifloat8_t|hifloat4x2_t|float8_e4m3_t|float8_e5m2_t.");
  static_assert(std::is_same<ITYPE, int32_t>::value ||
                    std::is_same<ITYPE, int64_t>::value,
                "Currently dtype of index supports only int32|int64.");
  int32_t idx_sizes[5];
  MteGsFillSizes<DIM>(idx->sizes, idx_sizes);
  int32_t idx_strides[5];
  MteGsFillSizes<DIM>(idx->strides, idx_strides);
  int32_t mask_strides[5] = {0, 0, 0, 0, 0};
  __ubuf__ bool *mask_ptr = nullptr;
  if constexpr (HAS_MASK) {
    MteGsFillSizes<DIM>(mask->strides, mask_strides);
    mask_ptr = reinterpret_cast<__ubuf__ bool *>(mask->aligned + mask->offset);
  }
  int32_t src_strides[5];
  MteGsFillSizes<DIM>(src->strides, src_strides);
  SimdIndirectStoreCore<DTYPE, ITYPE, DIM, HAS_MASK>(
      reinterpret_cast<__gm__ DTYPE *>(dst->aligned),
      reinterpret_cast<__ubuf__ ITYPE *>(idx->aligned + idx->offset),
      reinterpret_cast<__ubuf__ DTYPE *>(src->aligned + src->offset), mask_ptr,
      idx_sizes, idx_strides, mask_strides, src_strides,
      reinterpret_cast<__ubuf__ MteGsOffsetType<ITYPE> *>(table->aligned +
                                                          table->offset));
}

template <int DIM, typename DTYPE, typename ITYPE>
__aiv__ __attribute__((always_inline)) void
SimdStrideStore(memref_t<__gm__ DTYPE, 1> *dst,
                memref_t<__ubuf__ DTYPE, DIM> *src, const ITYPE offset,
                const ITYPE *strides, const ITYPE *numels,
                memref_t<__ubuf__ ITYPE, 1> *table) {
  int32_t src_sizes[5];
  MteGsFillSizes<DIM>(src->sizes, src_sizes);
  int32_t src_strides[5];
  MteGsFillSizes<DIM>(src->strides, src_strides);
  int64_t strides64[DIM];
  int64_t numels64[DIM];
  for (int32_t d = 0; d < DIM; ++d) {
    strides64[d] = static_cast<int64_t>(strides[d]);
    numels64[d] = static_cast<int64_t>(numels[d]);
  }
  SimdStrideStoreCore<DTYPE, ITYPE, DIM>(
      reinterpret_cast<__gm__ DTYPE *>(dst->aligned + dst->offset),
      reinterpret_cast<__ubuf__ DTYPE *>(src->aligned + src->offset),
      static_cast<int64_t>(offset), strides64, numels64, src_sizes, src_strides,
      reinterpret_cast<__ubuf__ MteGsOffsetType<ITYPE> *>(table->aligned +
                                                          table->offset));
}

template <int DIM, typename SRCTY, typename IDXTY>
__aiv__ __attribute__((always_inline)) void
SimdIndexPut(memref_t<__ubuf__ SRCTY, DIM> *val,
             memref_t<__ubuf__ IDXTY, 1> *idx, memref_t<__gm__ SRCTY, 1> *dst,
             const int32_t scatterDim, const int64_t indexBoundary,
             const IDXTY *endOffsets, const IDXTY *startOffsets,
             const IDXTY *dstStrides, memref_t<__ubuf__ IDXTY, 1> *table) {
  static_assert(
      std::is_same<SRCTY, float>::value || std::is_same<SRCTY, half>::value ||
          std::is_same<SRCTY, bfloat16_t>::value,
      "Currently dtype of source supports only float32|half|bfloat16");
  static_assert(std::is_same<IDXTY, int32_t>::value ||
                    std::is_same<IDXTY, int64_t>::value,
                "Currently dtype of index supports only int32|int64");
  int32_t val_sizes[5];
  MteGsFillSizes<DIM>(val->sizes, val_sizes);
  int32_t val_strides[5];
  MteGsFillSizes<DIM>(val->strides, val_strides);
  int64_t end64[DIM];
  int64_t start64[DIM];
  int64_t strides64[DIM];
  for (int32_t d = 0; d < DIM; ++d) {
    end64[d] = static_cast<int64_t>(endOffsets[d]);
    start64[d] = static_cast<int64_t>(startOffsets[d]);
    strides64[d] = static_cast<int64_t>(dstStrides[d]);
  }
  SimdIndexPutCore<SRCTY, IDXTY, DIM>(
      reinterpret_cast<__gm__ SRCTY *>(dst->aligned),
      reinterpret_cast<__ubuf__ SRCTY *>(val->aligned + val->offset),
      reinterpret_cast<__ubuf__ IDXTY *>(idx->aligned + idx->offset), val_sizes,
      val_strides, indexBoundary, scatterDim,
      static_cast<int32_t>(idx->strides[0]), end64, start64, strides64,
      reinterpret_cast<__ubuf__ MteGsOffsetType<IDXTY> *>(table->aligned +
                                                          table->offset));
}

// ---------------------------------------------------------------------------
// Symbol declarations / registrations. ABI is parameter-identical to the
// corresponding SIMT templates, only the symbol name gains the simd_ prefix
// (design doc §3.3); the macros are self-contained and need no VecUtils.h
// change. Every symbol takes one extra trailing `table` parameter: the
// planner-owned UB scratch tile for the MTE offset table (the IR side's
// temp_buffer operand, always the last forwarded operand).
// ---------------------------------------------------------------------------

#define DECLARE_SIMD_SCATTER_UB_TO_OUT(DIM, dataty, idxty, ...)                \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_simd_scatter_ub_to_out_##DIM##d_##dataty##_##idxty(         \
          memref_t<__gm__ dataty, 1> *dst,                                     \
          memref_t<__ubuf__ dataty, DIM> *src,                                 \
          memref_t<__ubuf__ idxty, DIM> *index, const int64_t indexBoundary,   \
          const int32_t dim, __VA_ARGS__)

#define DECLARE_SIMD_SCATTER_UB_TO_OUT_1D(DIM, dataty, idxty)                  \
  DECLARE_SIMD_SCATTER_UB_TO_OUT(                                              \
      DIM, dataty, idxty, const int64_t dstStride0, const int32_t endOffset0,  \
      const int32_t startOffset0, memref_t<__ubuf__ idxty, 1> *table)

#define REGISTE_SIMD_SCATTER_UB_TO_OUT_1D(DIM, dataty, idxty)                  \
  DECLARE_SIMD_SCATTER_UB_TO_OUT_1D(DIM, dataty, idxty) {                      \
    int64_t dstStrides[1] = {dstStride0};                                      \
    int32_t endOffsets[1] = {endOffset0};                                      \
    int32_t startOffsets[1] = {startOffset0};                                  \
    SimdScatterUBToOut<1, dataty, idxty>(dst, src, index, indexBoundary, dim,  \
                                         dstStrides, endOffsets, startOffsets, \
                                         table);                               \
  }

#define DECLARE_SIMD_SCATTER_UB_TO_OUT_2D(DIM, dataty, idxty)                  \
  DECLARE_SIMD_SCATTER_UB_TO_OUT(                                              \
      DIM, dataty, idxty, const int64_t dstStride0, const int64_t dstStride1,  \
      const int32_t endOffset0, const int32_t endOffset1,                      \
      const int32_t startOffset0, const int32_t startOffset1,                  \
      memref_t<__ubuf__ idxty, 1> *table)

#define REGISTE_SIMD_SCATTER_UB_TO_OUT_2D(DIM, dataty, idxty)                  \
  DECLARE_SIMD_SCATTER_UB_TO_OUT_2D(DIM, dataty, idxty) {                      \
    int64_t dstStrides[2] = {dstStride0, dstStride1};                          \
    int32_t endOffsets[2] = {endOffset0, endOffset1};                          \
    int32_t startOffsets[2] = {startOffset0, startOffset1};                    \
    SimdScatterUBToOut<2, dataty, idxty>(dst, src, index, indexBoundary, dim,  \
                                         dstStrides, endOffsets, startOffsets, \
                                         table);                               \
  }

#define DECLARE_SIMD_SCATTER_UB_TO_OUT_3D(DIM, dataty, idxty)                  \
  DECLARE_SIMD_SCATTER_UB_TO_OUT(                                              \
      DIM, dataty, idxty, const int64_t dstStride0, const int64_t dstStride1,  \
      const int64_t dstStride2, const int32_t endOffset0,                      \
      const int32_t endOffset1, const int32_t endOffset2,                      \
      const int32_t startOffset0, const int32_t startOffset1,                  \
      const int32_t startOffset2, memref_t<__ubuf__ idxty, 1> *table)

#define REGISTE_SIMD_SCATTER_UB_TO_OUT_3D(DIM, dataty, idxty)                  \
  DECLARE_SIMD_SCATTER_UB_TO_OUT_3D(DIM, dataty, idxty) {                      \
    int64_t dstStrides[3] = {dstStride0, dstStride1, dstStride2};              \
    int32_t endOffsets[3] = {endOffset0, endOffset1, endOffset2};              \
    int32_t startOffsets[3] = {startOffset0, startOffset1, startOffset2};      \
    SimdScatterUBToOut<3, dataty, idxty>(dst, src, index, indexBoundary, dim,  \
                                         dstStrides, endOffsets, startOffsets, \
                                         table);                               \
  }

#define DECLARE_SIMD_SCATTER_UB_TO_OUT_4D(DIM, dataty, idxty)                  \
  DECLARE_SIMD_SCATTER_UB_TO_OUT(                                              \
      DIM, dataty, idxty, const int64_t dstStride0, const int64_t dstStride1,  \
      const int64_t dstStride2, const int64_t dstStride3,                      \
      const int32_t endOffset0, const int32_t endOffset1,                      \
      const int32_t endOffset2, const int32_t endOffset3,                      \
      const int32_t startOffset0, const int32_t startOffset1,                  \
      const int32_t startOffset2, const int32_t startOffset3,                  \
      memref_t<__ubuf__ idxty, 1> *table)

#define REGISTE_SIMD_SCATTER_UB_TO_OUT_4D(DIM, dataty, idxty)                  \
  DECLARE_SIMD_SCATTER_UB_TO_OUT_4D(DIM, dataty, idxty) {                      \
    int64_t dstStrides[4] = {dstStride0, dstStride1, dstStride2, dstStride3};  \
    int32_t endOffsets[4] = {endOffset0, endOffset1, endOffset2, endOffset3};  \
    int32_t startOffsets[4] = {startOffset0, startOffset1, startOffset2,       \
                               startOffset3};                                  \
    SimdScatterUBToOut<4, dataty, idxty>(dst, src, index, indexBoundary, dim,  \
                                         dstStrides, endOffsets, startOffsets, \
                                         table);                               \
  }

#define DECLARE_SIMD_SCATTER_UB_TO_OUT_5D(DIM, dataty, idxty)                  \
  DECLARE_SIMD_SCATTER_UB_TO_OUT(                                              \
      DIM, dataty, idxty, const int64_t dstStride0, const int64_t dstStride1,  \
      const int64_t dstStride2, const int64_t dstStride3,                      \
      const int64_t dstStride4, const int32_t endOffset0,                      \
      const int32_t endOffset1, const int32_t endOffset2,                      \
      const int32_t endOffset3, const int32_t endOffset4,                      \
      const int32_t startOffset0, const int32_t startOffset1,                  \
      const int32_t startOffset2, const int32_t startOffset3,                  \
      const int32_t startOffset4, memref_t<__ubuf__ idxty, 1> *table)

#define REGISTE_SIMD_SCATTER_UB_TO_OUT_5D(DIM, dataty, idxty)                  \
  DECLARE_SIMD_SCATTER_UB_TO_OUT_5D(DIM, dataty, idxty) {                      \
    int64_t dstStrides[5] = {dstStride0, dstStride1, dstStride2, dstStride3,   \
                             dstStride4};                                      \
    int32_t endOffsets[5] = {endOffset0, endOffset1, endOffset2, endOffset3,   \
                             endOffset4};                                      \
    int32_t startOffsets[5] = {startOffset0, startOffset1, startOffset2,       \
                               startOffset3, startOffset4};                    \
    SimdScatterUBToOut<5, dataty, idxty>(dst, src, index, indexBoundary, dim,  \
                                         dstStrides, endOffsets, startOffsets, \
                                         table);                               \
  }

#define DECLARE_SIMD_INDIRECT_STORE(dim, dtype, itype)                         \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_simd_indirect_store_##dim##d_##dtype##_##itype(             \
          memref_t<__gm__ dtype, 1> *dst, memref_t<__ubuf__ itype, dim> *idx,  \
          memref_t<__ubuf__ dtype, dim> *src,                                  \
          memref_t<__ubuf__ bool, dim> *mask,                                  \
          memref_t<__ubuf__ itype, 1> *table)

#define DECLARE_SIMD_INDIRECT_STORE_NO_MASK(dim, dtype, itype)                 \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_simd_indirect_store_no_mask_##dim##d_##dtype##_##itype(     \
          memref_t<__gm__ dtype, 1> *dst, memref_t<__ubuf__ itype, dim> *idx,  \
          memref_t<__ubuf__ dtype, dim> *src,                                  \
          memref_t<__ubuf__ itype, 1> *table)

// One registration emits both the mask / no_mask symbols (aligned with the
// SIMT registration matrix).
#define REGISTE_SIMD_INDIRECT_STORE(dim, dtype, itype)                         \
  DECLARE_SIMD_INDIRECT_STORE(dim, dtype, itype) {                             \
    SimdIndirectStore<dim, dtype, itype, true>(dst, idx, src, mask, table);    \
  }                                                                            \
  DECLARE_SIMD_INDIRECT_STORE_NO_MASK(dim, dtype, itype) {                     \
    SimdIndirectStore<dim, dtype, itype, false>(                               \
        dst, idx, src, static_cast<memref_t<__ubuf__ bool, dim> *>(nullptr),   \
        table);                                                                \
  }

#define DECLARE_SIMD_STRIDE_STORE(dim, dtype, itype, ...)                      \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_simd_stride_store_##dim##d_##dtype##_##itype(               \
          memref_t<__gm__ dtype, 1> *dst, memref_t<__ubuf__ dtype, dim> *src,  \
          __VA_ARGS__)

#define DECLARE_SIMD_STRIDE_STORE_1D(dtype, itype)                             \
  DECLARE_SIMD_STRIDE_STORE(1, dtype, itype, itype offset, itype stride,       \
                            itype numel, memref_t<__ubuf__ itype, 1> *table)

#define DECLARE_SIMD_STRIDE_STORE_2D(dtype, itype)                             \
  DECLARE_SIMD_STRIDE_STORE(2, dtype, itype, itype offset, itype stride0,      \
                            itype stride1, itype numel0, itype numel1,         \
                            memref_t<__ubuf__ itype, 1> *table)

#define DECLARE_SIMD_STRIDE_STORE_3D(dtype, itype)                             \
  DECLARE_SIMD_STRIDE_STORE(3, dtype, itype, itype offset, itype stride0,      \
                            itype stride1, itype stride2, itype numel0,        \
                            itype numel1, itype numel2,                        \
                            memref_t<__ubuf__ itype, 1> *table)

#define REGISTE_SIMD_STRIDE_STORE_1D(dtype, itype)                             \
  DECLARE_SIMD_STRIDE_STORE_1D(dtype, itype) {                                 \
    const itype strides[1] = {stride};                                         \
    const itype numels[1] = {numel};                                           \
    SimdStrideStore<1, dtype, itype>(dst, src, offset, strides, numels,        \
                                     table);                                   \
  }

#define REGISTE_SIMD_STRIDE_STORE_2D(dtype, itype)                             \
  DECLARE_SIMD_STRIDE_STORE_2D(dtype, itype) {                                 \
    const itype strides[2] = {stride0, stride1};                               \
    const itype numels[2] = {numel0, numel1};                                  \
    SimdStrideStore<2, dtype, itype>(dst, src, offset, strides, numels,        \
                                     table);                                   \
  }

#define REGISTE_SIMD_STRIDE_STORE_3D(dtype, itype)                             \
  DECLARE_SIMD_STRIDE_STORE_3D(dtype, itype) {                                 \
    const itype strides[3] = {stride0, stride1, stride2};                      \
    const itype numels[3] = {numel0, numel1, numel2};                          \
    SimdStrideStore<3, dtype, itype>(dst, src, offset, strides, numels,        \
                                     table);                                   \
  }

#define DECLARE_SIMD_INDEX_PUT(dim, srcty, idxty, ...)                         \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_simd_index_put_##dim##d_##srcty##_##idxty(                  \
          memref_t<__gm__ srcty, 1> *dst, memref_t<__ubuf__ idxty, 1> *idx,    \
          memref_t<__ubuf__ srcty, dim> *val, const int32_t scatterDim,        \
          const int64_t indexBoundary, __VA_ARGS__)

#define DECLARE_SIMD_INDEX_PUT_2D(DIM, srcty, idxty)                           \
  DECLARE_SIMD_INDEX_PUT(DIM, srcty, idxty, const idxty endOffset0,            \
                         const idxty endOffset1, const idxty startOffset0,     \
                         const idxty startOffset1, const idxty dstStride0,     \
                         const idxty dstStride1,                               \
                         memref_t<__ubuf__ idxty, 1> *table)

#define REGISTER_SIMD_INDEX_PUT_2D(DIM, srcty, idxty)                          \
  DECLARE_SIMD_INDEX_PUT_2D(DIM, srcty, idxty) {                               \
    const idxty endOffsets[2] = {endOffset0, endOffset1};                      \
    const idxty startOffsets[2] = {startOffset0, startOffset1};                \
    const idxty dstStrides[2] = {dstStride0, dstStride1};                      \
    SimdIndexPut<2, srcty, idxty>(val, idx, dst, scatterDim, indexBoundary,    \
                                  endOffsets, startOffsets, dstStrides,        \
                                  table);                                      \
  }

#define DECLARE_SIMD_INDEX_PUT_3D(DIM, srcty, idxty)                           \
  DECLARE_SIMD_INDEX_PUT(DIM, srcty, idxty, const idxty endOffset0,            \
                         const idxty endOffset1, const idxty endOffset2,       \
                         const idxty startOffset0, const idxty startOffset1,   \
                         const idxty startOffset2, const idxty dstStride0,     \
                         const idxty dstStride1, const idxty dstStride2,       \
                         memref_t<__ubuf__ idxty, 1> *table)

#define REGISTER_SIMD_INDEX_PUT_3D(DIM, srcty, idxty)                          \
  DECLARE_SIMD_INDEX_PUT_3D(DIM, srcty, idxty) {                               \
    const idxty endOffsets[3] = {endOffset0, endOffset1, endOffset2};          \
    const idxty startOffsets[3] = {startOffset0, startOffset1, startOffset2};  \
    const idxty dstStrides[3] = {dstStride0, dstStride1, dstStride2};          \
    SimdIndexPut<3, srcty, idxty>(val, idx, dst, scatterDim, indexBoundary,    \
                                  endOffsets, startOffsets, dstStrides,        \
                                  table);                                      \
  }

#define DECLARE_SIMD_INDEX_PUT_4D(DIM, srcty, idxty)                           \
  DECLARE_SIMD_INDEX_PUT(                                                      \
      DIM, srcty, idxty, const idxty endOffset0, const idxty endOffset1,       \
      const idxty endOffset2, const idxty endOffset3,                          \
      const idxty startOffset0, const idxty startOffset1,                      \
      const idxty startOffset2, const idxty startOffset3,                      \
      const idxty dstStride0, const idxty dstStride1, const idxty dstStride2,  \
      const idxty dstStride3, memref_t<__ubuf__ idxty, 1> *table)

#define REGISTER_SIMD_INDEX_PUT_4D(DIM, srcty, idxty)                          \
  DECLARE_SIMD_INDEX_PUT_4D(DIM, srcty, idxty) {                               \
    const idxty endOffsets[4] = {endOffset0, endOffset1, endOffset2,           \
                                 endOffset3};                                  \
    const idxty startOffsets[4] = {startOffset0, startOffset1, startOffset2,   \
                                   startOffset3};                              \
    const idxty dstStrides[4] = {dstStride0, dstStride1, dstStride2,           \
                                 dstStride3};                                  \
    SimdIndexPut<4, srcty, idxty>(val, idx, dst, scatterDim, indexBoundary,    \
                                  endOffsets, startOffsets, dstStrides,        \
                                  table);                                      \
  }

#define DECLARE_SIMD_INDEX_PUT_5D(DIM, srcty, idxty)                           \
  DECLARE_SIMD_INDEX_PUT(                                                      \
      DIM, srcty, idxty, const idxty endOffset0, const idxty endOffset1,       \
      const idxty endOffset2, const idxty endOffset3, const idxty endOffset4,  \
      const idxty startOffset0, const idxty startOffset1,                      \
      const idxty startOffset2, const idxty startOffset3,                      \
      const idxty startOffset4, const idxty dstStride0,                        \
      const idxty dstStride1, const idxty dstStride2, const idxty dstStride3,  \
      const idxty dstStride4, memref_t<__ubuf__ idxty, 1> *table)

#define REGISTER_SIMD_INDEX_PUT_5D(DIM, srcty, idxty)                          \
  DECLARE_SIMD_INDEX_PUT_5D(DIM, srcty, idxty) {                               \
    const idxty endOffsets[5] = {endOffset0, endOffset1, endOffset2,           \
                                 endOffset3, endOffset4};                      \
    const idxty startOffsets[5] = {startOffset0, startOffset1, startOffset2,   \
                                   startOffset3, startOffset4};                \
    const idxty dstStrides[5] = {dstStride0, dstStride1, dstStride2,           \
                                 dstStride3, dstStride4};                      \
    SimdIndexPut<5, srcty, idxty>(val, idx, dst, scatterDim, indexBoundary,    \
                                  endOffsets, startOffsets, dstStrides,        \
                                  table);                                      \
  }

// Per-op registration lists, dtype coverage identical to the SIMT templates
// (design doc §3.3 matrix; §3.6 keeps the old templates' dim coverage).
#define SIMD_GS_REG_SCATTER_DIM(dim, itype)                                    \
  REGISTE_SIMD_SCATTER_UB_TO_OUT_##dim##D(dim, float, itype);                  \
  REGISTE_SIMD_SCATTER_UB_TO_OUT_##dim##D(dim, float16_t, itype);              \
  REGISTE_SIMD_SCATTER_UB_TO_OUT_##dim##D(dim, bfloat16_t, itype);             \
  REGISTE_SIMD_SCATTER_UB_TO_OUT_##dim##D(dim, half, itype);

#define REGISTE_SIMD_INDIRECT_STORE_ALL_DTYPES(dim, itype)                     \
  REGISTE_SIMD_INDIRECT_STORE(dim, float, itype);                              \
  REGISTE_SIMD_INDIRECT_STORE(dim, half, itype);                               \
  REGISTE_SIMD_INDIRECT_STORE(dim, bfloat16_t, itype);                         \
  REGISTE_SIMD_INDIRECT_STORE(dim, int8_t, itype);                             \
  REGISTE_SIMD_INDIRECT_STORE(dim, int16_t, itype);                            \
  REGISTE_SIMD_INDIRECT_STORE(dim, int32_t, itype);                            \
  REGISTE_SIMD_INDIRECT_STORE(dim, int64_t, itype);                            \
  REGISTE_SIMD_INDIRECT_STORE(dim, uint8_t, itype);                            \
  REGISTE_SIMD_INDIRECT_STORE(dim, uint16_t, itype);                           \
  REGISTE_SIMD_INDIRECT_STORE(dim, uint32_t, itype);                           \
  REGISTE_SIMD_INDIRECT_STORE(dim, uint64_t, itype);                           \
  REGISTE_SIMD_INDIRECT_STORE(dim, bool, itype);                               \
  REGISTE_SIMD_INDIRECT_STORE(dim, hifloat8_t, itype);                         \
  REGISTE_SIMD_INDIRECT_STORE(dim, hifloat4x2_t, itype);                       \
  REGISTE_SIMD_INDIRECT_STORE(dim, float8_e4m3_t, itype);                      \
  REGISTE_SIMD_INDIRECT_STORE(dim, float8_e5m2_t, itype);

#define REGISTE_SIMD_STRIDE_STORE_ALL_DTYPES_1D(dtype_itype)                   \
  REGISTE_SIMD_STRIDE_STORE_1D dtype_itype;
#define REGISTE_SIMD_STRIDE_STORE_ALL_DTYPES_2D(dtype_itype)                   \
  REGISTE_SIMD_STRIDE_STORE_2D dtype_itype;
#define REGISTE_SIMD_STRIDE_STORE_ALL_DTYPES_3D(dtype_itype)                   \
  REGISTE_SIMD_STRIDE_STORE_3D dtype_itype;

#define SIMD_GS_STRIDE_STORE_DTYPES_X(dim_macro)                               \
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

#define SIMD_GS_REG_INDEX_PUT_DIM(dim, itype)                                  \
  REGISTER_SIMD_INDEX_PUT_##dim##D(dim, float, itype);                         \
  REGISTER_SIMD_INDEX_PUT_##dim##D(dim, half, itype);                          \
  REGISTER_SIMD_INDEX_PUT_##dim##D(dim, bfloat16_t, itype);

extern "C" {

// scatter dtype: float|float16_t|bfloat16_t|half, itype: int32|int64.
SIMD_GS_REG_SCATTER_DIM(1, int32_t)
SIMD_GS_REG_SCATTER_DIM(1, int64_t)
SIMD_GS_REG_SCATTER_DIM(2, int32_t)
SIMD_GS_REG_SCATTER_DIM(2, int64_t)
SIMD_GS_REG_SCATTER_DIM(3, int32_t)
SIMD_GS_REG_SCATTER_DIM(3, int64_t)
SIMD_GS_REG_SCATTER_DIM(4, int32_t)
SIMD_GS_REG_SCATTER_DIM(4, int64_t)
SIMD_GS_REG_SCATTER_DIM(5, int32_t)
SIMD_GS_REG_SCATTER_DIM(5, int64_t)

// indirect store (mask + no_mask dual symbols): all b8~b64 dtypes, 1D~5D.
REGISTE_SIMD_INDIRECT_STORE_ALL_DTYPES(1, int32_t)
REGISTE_SIMD_INDIRECT_STORE_ALL_DTYPES(2, int32_t)
REGISTE_SIMD_INDIRECT_STORE_ALL_DTYPES(3, int32_t)
REGISTE_SIMD_INDIRECT_STORE_ALL_DTYPES(4, int32_t)
REGISTE_SIMD_INDIRECT_STORE_ALL_DTYPES(5, int32_t)
REGISTE_SIMD_INDIRECT_STORE_ALL_DTYPES(1, int64_t)
REGISTE_SIMD_INDIRECT_STORE_ALL_DTYPES(2, int64_t)
REGISTE_SIMD_INDIRECT_STORE_ALL_DTYPES(3, int64_t)
REGISTE_SIMD_INDIRECT_STORE_ALL_DTYPES(4, int64_t)
REGISTE_SIMD_INDIRECT_STORE_ALL_DTYPES(5, int64_t)

// stride store: 1D~3D, itype: int32|int64.
SIMD_GS_STRIDE_STORE_DTYPES_X(REGISTE_SIMD_STRIDE_STORE_ALL_DTYPES_1D)
SIMD_GS_STRIDE_STORE_DTYPES_X(REGISTE_SIMD_STRIDE_STORE_ALL_DTYPES_2D)
SIMD_GS_STRIDE_STORE_DTYPES_X(REGISTE_SIMD_STRIDE_STORE_ALL_DTYPES_3D)

// index put: float|half|bfloat16_t, 2D~5D.
SIMD_GS_REG_INDEX_PUT_DIM(2, int32_t)
SIMD_GS_REG_INDEX_PUT_DIM(2, int64_t)
SIMD_GS_REG_INDEX_PUT_DIM(3, int32_t)
SIMD_GS_REG_INDEX_PUT_DIM(3, int64_t)
SIMD_GS_REG_INDEX_PUT_DIM(4, int32_t)
SIMD_GS_REG_INDEX_PUT_DIM(4, int64_t)
SIMD_GS_REG_INDEX_PUT_DIM(5, int32_t)
SIMD_GS_REG_INDEX_PUT_DIM(5, int64_t)

} // extern "C"

#endif // BISHENG_TPL_920R1
#endif // BISHENG_TPL_REGBASE
