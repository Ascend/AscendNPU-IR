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

// A6 (dav-920r1) MTE gather/scatter shared skeleton.
//
// Design doc: A6 gather/scatter MTE-fallback design, §3.4. The 8 MTE wrappers
// (read: simd_gather / simd_indirect_load / simd_stride_load /
// simd_embedding_gather; write: simd_scatter / simd_indirect_store /
// simd_stride_store / simd_index_put) all converge on the same two hardware
// instructions and share the chunked table-driven handshake loop provided
// here. Op semantics are digested while building the per-chunk offset table;
// invalid lanes are encoded as -1 and skipped by the engine.
//
// The whole header content is 920r1-only: the C310 variant of meta_op.aiv
// compiles the including .cpp as an empty TU.

#ifndef HIVM_MLIR_TEMPLATE_MTE_GATHER_SCATTER_H
#define HIVM_MLIR_TEMPLATE_MTE_GATHER_SCATTER_H

#include "ArchDefs.h"
#if BISHENG_TPL_REGBASE
#include "RegBase/DMAUtils.h"
#include "Utils.h"
#include <type_traits>

#if BISHENG_TPL_920R1

// ---------------------------------------------------------------------------
// Intrinsic notes (design doc §3.1 / adaptation doc §6.1, §6.2).
//
// gather_gm_to_ubuf (MTE2): copy `offsetNum` bursts from GM to UB, one burst
//   per offset-table entry; dstStride is the UB-side landing gap.
//   Signature (ccec 920r1 builtin):
//   gather_gm_to_ubuf(__ubuf__ uint8_t *dst, __gm__ uint8_t *baseSrc,
//                     uint8_t sid, uint8_t offset_type,
//                     bool constant_padding_ctl, __ubuf__ void *offsetSrc,
//                     uint8_t l2_cache_ctl, uint32_t burst_len,
//                     uint32_t dst_stride, uint32_t offset_num)
// scatter_ubuf_to_gm (MTE3): mirror direction, no padding-content select.
//   Signature (ccec 920r1 builtin):
//   scatter_ubuf_to_gm(__gm__ uint8_t *dst, __ubuf__ uint8_t *dataSrc,
//                      uint8_t sid, uint8_t offset_type,
//                      __ubuf__ void *offsetSrc, uint8_t l2_cache_ctl,
//                      uint32_t burst_len, uint32_t src_stride,
//                      uint32_t offset_num)
//
// No in-tree prototypes are declared: ccec 920r1 provides the builtin
// declarations itself (confirmed by a minimal compile test - without any
// hand-written prototype the calls are inlined and no undefined symbols are
// emitted). offsetType encodes the offset-table entry width: 0 = s32, 1 = s64
// (s64 encoding value pending ISA confirmation, design doc §3.1 case (3)').
// The s64 table is passed through the same pointer parameter with a cast.
// ---------------------------------------------------------------------------

// Offset-table entry width encoding (s64 value pending ISA confirmation).
constexpr uint8_t MTE_GS_OFFSET_TYPE_S32 = 0;
constexpr uint8_t MTE_GS_OFFSET_TYPE_S64 = 1;

// Chunking (design doc §3.4): single call carries at most 64 table entries
// (s32 table 256B / s64 512B). The chunked loops carry any total lane count.
constexpr uint32_t MTE_GS_MAX_CHUNK = 64;

// The offset-table scratch tile is caller-provided: the IR side attaches a
// planner-owned UB memref as the op's temp_buffer (ExtraBufferOpInterface,
// MTE_GS_MAX_CHUNK entries of the index element type), and the ciface layer
// passes it through as the trailing `table` parameter. A compiler-local
// __ubuf__ array previously allocated the table with no contract, which let
// the ccec-assigned slot alias the op's idx / value tiles (observed as table
// writes landing on top of the idx region). The planner-allocated tile can
// never alias them.

// Offset-table entry type follows the index type at compile time
// (design doc §3.1 case (3)'): int32 instances build s32 tables, int64
// instances build s64 tables (supporting byte spans beyond 2GB).
template <typename ITYPE> struct MteGsOffsetT {
  using type = int32_t;
};
template <> struct MteGsOffsetT<int64_t> {
  using type = int64_t;
};
template <typename ITYPE>
using MteGsOffsetType = typename MteGsOffsetT<ITYPE>::type;

template <typename OffsetT>
__aiv__ __attribute__((always_inline)) static inline constexpr uint8_t
MteGsOffsetTypeCode() {
  return std::is_same<OffsetT, int64_t>::value ? MTE_GS_OFFSET_TYPE_S64
                                               : MTE_GS_OFFSET_TYPE_S32;
}

// ---------------------------------------------------------------------------
// Small helpers shared by all 8 wrappers.
// ---------------------------------------------------------------------------

// EvictionPolicy -> l2CacheCtl (isomorphic encodings, design doc §3.1).
__aiv__ __attribute__((always_inline)) static inline uint8_t
MteGsL2CacheCtl(EvictionPolicy eviction_policy) {
  return eviction_policy == EvictionPolicy::EvictLast ? 1 : 0;
}

// Program the MTE pad-value SPR and report the constantPaddingCtl bit
// (design doc §3.1 case (2)): PadMode::Value routes the intrinsic tail
// padding to the register (ctl=1); Null clears the register and keeps the
// default copy source (ctl=0).
template <typename DTYPE>
__aiv__ __attribute__((always_inline)) static inline bool
MteGsSetupPadMode(PadMode pad_mode, PadValueT<DTYPE> pad_value) {
  uint32_t bits = 0;
  bool from_register = false;
  if (pad_mode == PadMode::Value) {
    from_register = true;
    using PadT = PadValueT<DTYPE>;
    if constexpr (sizeof(PadT) == 1) {
      bits = *reinterpret_cast<uint8_t *>(&pad_value);
    } else if constexpr (sizeof(PadT) == 2) {
      bits = *reinterpret_cast<uint16_t *>(&pad_value);
    } else {
      bits = *reinterpret_cast<uint32_t *>(&pad_value);
    }
  }
  INTRINSIC(set_mov_pad_val, bits);
  return from_register;
}

// Index resolution (design doc §1): negative indices wrap once against the
// boundary; anything still outside [0, boundary) is an invalid lane.
template <typename ITYPE>
__aiv__ __attribute__((always_inline)) static inline bool
MteGsResolveIndex(ITYPE &idx, int64_t index_boundary) {
  if (idx < 0) {
    idx += static_cast<ITYPE>(index_boundary);
  }
  const int64_t v = static_cast<int64_t>(idx);
  return v >= 0 && v < index_boundary;
}

// Advance a row-major coordinate vector by one lane.
__aiv__ __attribute__((always_inline)) static inline void
MteGsAdvanceCoords(int32_t *coords, const int32_t *sizes, int32_t rank) {
  for (int32_t i = rank - 1; i >= 0; --i) {
    if (++coords[i] < sizes[i]) {
      return;
    }
    coords[i] = 0;
  }
}

// Per-lane gather index math, shared by the engine table build and the
// scalar fallback (design doc §3.4): reads the lane's index through
// idx_strides, applies the out-of-range guard and the window guard, and
// folds the ND source offset. Returns false for lanes the engine marks
// -1 and the scalar fallback leaves untouched. The window guard runs on
// every dim, 1D included (SIMT SimtGather1D: i0 + startOffset0 < endOffset0
// -- skipping it for 1D would silently ignore the ABI window args and fill
// out-of-window lanes with src[idx], breaking the zero-fill contract);
// the last dim never multiplies a stride.
template <typename ITYPE, int DIM>
__aiv__ __attribute__((always_inline)) static inline bool
MteGsGatherElemOff(const int32_t *coords, __ubuf__ ITYPE *idx,
                   const int32_t *idx_strides, const int64_t index_boundary,
                   const int32_t dimension, const int64_t *src_strides,
                   const int32_t *end_offsets, const int32_t *start_offsets,
                   int64_t &elem_off) {
  int32_t idx_flat = 0;
  for (int32_t d = 0; d < DIM; ++d) {
    idx_flat += coords[d] * idx_strides[d];
  }
  ITYPE idx_val = idx[idx_flat];
  if (!MteGsResolveIndex(idx_val, index_boundary)) {
    return false;
  }
  // Window guard on every dim, 1D included (matching SIMT and the scatter
  // side): coords[d] + start_offsets[d] must stay below end_offsets[d].
  for (int32_t d = 0; d < DIM; ++d) {
    const int32_t g = coords[d] + start_offsets[d];
    if (static_cast<uint32_t>(g) >= static_cast<uint32_t>(end_offsets[d])) {
      return false;
    }
  }
  if constexpr (DIM == 1) {
    elem_off = static_cast<int64_t>(idx_val);
    return true;
  }
  elem_off = 0;
  for (int32_t d = 0; d < DIM - 1; ++d) {
    const int32_t g = (d == dimension) ? static_cast<int32_t>(idx_val)
                                       : coords[d] + start_offsets[d];
    elem_off += static_cast<int64_t>(g) * src_strides[d];
  }
  elem_off += (dimension == DIM - 1) ? static_cast<int64_t>(idx_val)
                                     : coords[DIM - 1] + start_offsets[DIM - 1];
  return true;
}

// Per-lane scatter index math, shared by the engine table build and the
// scalar fallback: same guard structure as MteGsGatherElemOff, but the
// window guard applies on every dim (SIMT 1D semantics: dst[idx*dstStride0])
// and the folding targets the GM dst strides.
template <typename ITYPE, int DIM>
__aiv__ __attribute__((always_inline)) static inline bool
MteGsScatterElemOff(const int32_t *coords, __ubuf__ ITYPE *idx,
                    const int32_t *idx_strides, const int64_t index_boundary,
                    const int32_t dimension, const int64_t *dst_strides,
                    const int32_t *end_offsets, const int32_t *start_offsets,
                    int64_t &elem_off) {
  int32_t idx_flat = 0;
  for (int32_t d = 0; d < DIM; ++d) {
    idx_flat += coords[d] * idx_strides[d];
  }
  ITYPE idx_val = idx[idx_flat];
  if (!MteGsResolveIndex(idx_val, index_boundary)) {
    return false;
  }
  // Window guard: check coords[d]+start[d] on every dim (matching SIMT: the
  // indexed dim is checked against its raw coordinate, not the substituted
  // index).
  for (int32_t d = 0; d < DIM; ++d) {
    const int32_t g = coords[d] + start_offsets[d];
    if (static_cast<uint32_t>(g) >= static_cast<uint32_t>(end_offsets[d])) {
      return false;
    }
  }
  if constexpr (DIM == 1) {
    elem_off = static_cast<int64_t>(idx_val) * dst_strides[0];
    return true;
  }
  elem_off = 0;
  for (int32_t d = 0; d < DIM - 1; ++d) {
    const int32_t g = (d == dimension) ? static_cast<int32_t>(idx_val)
                                       : coords[d] + start_offsets[d];
    elem_off += static_cast<int64_t>(g) * dst_strides[d];
  }
  elem_off += (dimension == DIM - 1) ? static_cast<int64_t>(idx_val)
                                     : coords[DIM - 1] + start_offsets[DIM - 1];
  return true;
}

// Greatest common divisor (small inputs).
__aiv__ __attribute__((always_inline)) static inline uint32_t
MteGsGcd(uint32_t a, uint32_t b) {
  while (b != 0) {
    const uint32_t t = a % b;
    a = b;
    b = t;
  }
  return a;
}

// Number of -1 table entries appended to a tail chunk so that
// n * burstBytes rounds up to a 32B boundary (design doc §3.4): the engine
// skips the -1 bursts, which removes the compact-mode tail padding at the
// source. A full 64-entry chunk is always 32B aligned and pads nothing.
__aiv__ __attribute__((always_inline)) static inline uint32_t
MteGsTailPadEntries(uint32_t n, uint32_t burst_bytes) {
  if (burst_bytes % UB_ALIGN_BYTES == 0) {
    return 0;
  }
  const uint32_t period =
      UB_ALIGN_BYTES / MteGsGcd(burst_bytes, UB_ALIGN_BYTES);
  return (period - n % period) % period;
}

// Fill table[n .. n+pad) with skip markers.
template <typename OffsetT>
__aiv__ __attribute__((always_inline)) static inline void
MteGsPadTableSkip(__ubuf__ OffsetT *table, uint32_t n, uint32_t pad) {
  for (uint32_t i = n; i < n + pad; ++i) {
    table[i] = static_cast<OffsetT>(-1);
  }
}

// Drain outstanding MTE traffic before scalar writes into a shared UB region
// (the leading handshake of load_gm_to_ubuf_*_by_scalar in DMAUtils.h).
__aiv__ __attribute__((always_inline)) static inline void MteGsDrainMte2() {
  INTRINSIC(set_flag, PIPE_MTE2, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_MTE2, PIPE_S, LIB_EVENT_ID0);
}

__aiv__ __attribute__((always_inline)) static inline void MteGsDrainMte3() {
  INTRINSIC(set_flag, PIPE_MTE3, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_MTE3, PIPE_S, LIB_EVENT_ID0);
}

// ---------------------------------------------------------------------------
// Chunked MTE transfers: publish the freshly built table to the engine,
// issue the intrinsic, then wait for completion so the next chunk may reuse
// the table and the scalar side may observe dst (design doc §3.4,
// three-phase handshake).
//
// Element-wise wrappers pass burstLen = dstStride/srcStride = sizeof(DTYPE)
// (compact mode); the embedding wrapper passes rowBytes and lets the engine
// classify the padding mode by row width.
// ---------------------------------------------------------------------------
template <typename DTYPE, typename OffsetT>
__aiv__ __attribute__((always_inline)) static inline void
MteGsGatherChunk(__ubuf__ DTYPE *dst, __gm__ DTYPE *src,
                 __ubuf__ OffsetT *table, uint32_t offset_num,
                 uint32_t burst_len, uint32_t dst_stride, bool const_pad_ctl,
                 uint8_t l2_cache_ctl) {
  INTRINSIC(set_flag, PIPE_S, PIPE_MTE2, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_MTE2, LIB_EVENT_ID0);
  // The scalar config operands (constantPaddingCtl / l2CacheCtl) must reach
  // the intrinsic as compile-time literals: the builtin lowering keys its
  // variants on operand immediateness, and a register-carried flag makes the
  // frontend synthesize a variant the backend does not register (observed
  // as an unresolvable llvm.hivm.GATHER.OUT.TO.UB.* symbol at kernel link).
  // Both flags span tiny value sets, so dispatch on the runtime values and
  // instantiate literal call sites -- the exact operand profile the
  // ESL-validated 1D spike used.
  if (const_pad_ctl) {
    if (l2_cache_ctl == 0) {
      INTRINSIC(gather_gm_to_ubuf, reinterpret_cast<__ubuf__ uint8_t *>(dst),
                reinterpret_cast<__gm__ uint8_t *>(src), /*sid=*/0,
                MteGsOffsetTypeCode<OffsetT>(), /*constantPaddingCtl=*/true,
                reinterpret_cast<__ubuf__ void *>(table), /*l2CacheCtl=*/0,
                burst_len, dst_stride, offset_num);
    } else {
      INTRINSIC(gather_gm_to_ubuf, reinterpret_cast<__ubuf__ uint8_t *>(dst),
                reinterpret_cast<__gm__ uint8_t *>(src), /*sid=*/0,
                MteGsOffsetTypeCode<OffsetT>(), /*constantPaddingCtl=*/true,
                reinterpret_cast<__ubuf__ void *>(table), /*l2CacheCtl=*/1,
                burst_len, dst_stride, offset_num);
    }
  } else {
    if (l2_cache_ctl == 0) {
      INTRINSIC(gather_gm_to_ubuf, reinterpret_cast<__ubuf__ uint8_t *>(dst),
                reinterpret_cast<__gm__ uint8_t *>(src), /*sid=*/0,
                MteGsOffsetTypeCode<OffsetT>(), /*constantPaddingCtl=*/false,
                reinterpret_cast<__ubuf__ void *>(table), /*l2CacheCtl=*/0,
                burst_len, dst_stride, offset_num);
    } else {
      INTRINSIC(gather_gm_to_ubuf, reinterpret_cast<__ubuf__ uint8_t *>(dst),
                reinterpret_cast<__gm__ uint8_t *>(src), /*sid=*/0,
                MteGsOffsetTypeCode<OffsetT>(), /*constantPaddingCtl=*/false,
                reinterpret_cast<__ubuf__ void *>(table), /*l2CacheCtl=*/1,
                burst_len, dst_stride, offset_num);
    }
  }
  INTRINSIC(set_flag, PIPE_MTE2, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_MTE2, PIPE_S, LIB_EVENT_ID0);
}

template <typename DTYPE, typename OffsetT>
__aiv__ __attribute__((always_inline)) static inline void
MteGsScatterChunk(__gm__ DTYPE *dst, __ubuf__ DTYPE *src,
                  __ubuf__ OffsetT *table, uint32_t offset_num,
                  uint32_t burst_len, uint32_t src_stride) {
  INTRINSIC(set_flag, PIPE_S, PIPE_MTE3, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_MTE3, LIB_EVENT_ID0);
  INTRINSIC(scatter_ubuf_to_gm, reinterpret_cast<__gm__ uint8_t *>(dst),
            reinterpret_cast<__ubuf__ uint8_t *>(src), /*sid=*/0,
            MteGsOffsetTypeCode<OffsetT>(),
            reinterpret_cast<__ubuf__ void *>(table), /*l2CacheCtl=*/0,
            burst_len, src_stride, offset_num);
  INTRINSIC(set_flag, PIPE_MTE3, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_MTE3, PIPE_S, LIB_EVENT_ID0);
}

// ---------------------------------------------------------------------------
// Scalar prefill helpers (gather-family preprocessing, design doc §3.4).
// ---------------------------------------------------------------------------

// Zero-fill a dense UB tile (zero-fill contract: skipped lanes keep 0).
template <typename DTYPE>
__aiv__ __attribute__((always_inline)) static inline void
MteGsZeroFill(__ubuf__ DTYPE *dst, uint32_t total_lanes) {
  for (uint32_t i = 0; i < total_lanes; ++i) {
    dst[i] = static_cast<DTYPE>(0);
  }
}

// Prefill a dense UB dst tile from a strided `other` tile (indirect-load /
// stride-load other contract); `rank`-dim coordinates walk both tiles.
template <typename DTYPE, int DIM>
__aiv__ __attribute__((always_inline)) static inline void
MteGsPrefillOther(__ubuf__ DTYPE *dst, const __ubuf__ DTYPE *other,
                  const int32_t *dst_sizes, const int32_t *other_strides,
                  uint32_t total_lanes) {
  int32_t coords[DIM] = {0};
  for (uint32_t lane = 0; lane < total_lanes; ++lane) {
    int32_t other_off = 0;
    for (int32_t d = 0; d < DIM; ++d) {
      other_off += coords[d] * other_strides[d];
    }
    dst[lane] = other[other_off];
    MteGsAdvanceCoords(coords, dst_sizes, DIM);
  }
}

// Flatten a memref's sizes into int32[5] (padded with 1).
template <int DIM>
__aiv__ __attribute__((always_inline)) static inline void
MteGsFillSizes(const int64_t *memref_sizes, int32_t *sizes) {
  for (int32_t d = 0; d < DIM; ++d) {
    sizes[d] = static_cast<int32_t>(memref_sizes[d]);
  }
  for (int32_t d = DIM; d < 5; ++d) {
    sizes[d] = 1;
  }
}

// Row-major contiguity test: strides[d] must equal the product of the
// trailing sizes (unit innermost stride, no gaps, no broadcasts). The MTE
// engine reads / writes the UB data tile densely; non-contiguous tiles
// (strided views, broadcasts) cannot feed the engine and take the per-lane
// scalar fallbacks in the wrapper cores (SIMT parity).
__aiv__ __attribute__((always_inline)) static inline bool
MteGsIsContiguous(const int32_t *sizes, const int32_t *strides, int32_t rank) {
  int64_t expected = 1;
  for (int32_t d = rank - 1; d >= 0; --d) {
    if (static_cast<int64_t>(strides[d]) != expected) {
      return false;
    }
    expected *= sizes[d];
  }
  return true;
}

// Lane offset of a coordinate vector under per-dim strides.
__aiv__ __attribute__((always_inline)) static inline int64_t
MteGsStridedOffset(const int32_t *coords, const int32_t *strides,
                   int32_t rank) {
  int64_t off = 0;
  for (int32_t d = 0; d < rank; ++d) {
    off += static_cast<int64_t>(coords[d]) * strides[d];
  }
  return off;
}

// Fill every lane of a sizes-shaped tile with `value` through per-dim
// strides (strided counterpart of MteGsZeroFill).
template <typename DTYPE, int DIM>
__aiv__ __attribute__((always_inline)) static inline void
MteGsStridedFill(__ubuf__ DTYPE *dst, const int32_t *sizes,
                 const int32_t *strides, DTYPE value) {
  uint64_t total = 1;
  for (int32_t d = 0; d < DIM; ++d) {
    total *= static_cast<uint64_t>(sizes[d]);
  }
  int32_t coords[5] = {0, 0, 0, 0, 0};
  for (uint64_t lane = 0; lane < total; ++lane) {
    dst[MteGsStridedOffset(coords, strides, DIM)] = value;
    MteGsAdvanceCoords(coords, sizes, DIM);
  }
}

// Lane-count helper: 0 only when the tile is empty. No truncation -- the
// chunked loops carry any total (a single transfer is bounded by
// MTE_GS_MAX_CHUNK table entries), and a silent cap would drop tail lanes
// on the scatter side. A UB-resident tile is bounded by the UB size, so
// the count always fits uint32_t.
__aiv__ __attribute__((always_inline)) static inline uint32_t
MteGsClampLanes(uint64_t total) {
  return total == 0 ? 0 : static_cast<uint32_t>(total);
}

// Bounded per-dim load/store size (mirrors the SIMT
// getStrideLoad/StoreBoundedSize semantics):
// stride<0 / numel<=0 / size<=0 -> 0; stride==0 -> size==1 ? 1 : 0;
// otherwise min(numel, size). Shared by simd_stride_load / simd_stride_store.
__aiv__ __attribute__((always_inline)) static inline int64_t
MteGsStrideBoundedSize(const int64_t stride, const int64_t numel,
                       const int64_t size) {
  if (stride < 0 || numel <= 0 || size <= 0) {
    return 0;
  }
  if (stride == 0) {
    return size == 1 ? 1 : 0;
  }
  return numel < size ? numel : size;
}

#endif // BISHENG_TPL_920R1
#endif // BISHENG_TPL_REGBASE
#endif // HIVM_MLIR_TEMPLATE_MTE_GATHER_SCATTER_H
