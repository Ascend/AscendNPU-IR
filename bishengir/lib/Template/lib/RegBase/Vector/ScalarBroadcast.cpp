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

/// Scalar-broadcast library functions for the reg-based (Ascend910D /
/// Ascend960, dav-c310-vec) template library.
///
/// `convert-hivm-to-standard-regbase` lowers `hivm.hir.vbrc` with a scalar
/// source to `_mlir_ciface_broadcast_scalar_<type>_to_<dim>d` libcalls. The
/// mem-based (c220) template library provides them from
/// lib/Vector/Broadcast/BrcSinglePoint.cpp, but that family relies on the
/// `vector_dup`/`vcopy` repeat-mode intrinsics which do not exist on
/// dav-c310-vec, leaving the c310/920r1 meta_op bitcode without any
/// broadcast symbols. This file provides the same C ABI on the reg-based
/// arches using `vbr` + `vsts`, which are fully supported there.
///
/// Pipeline discipline: every entry wraps its whole fill in one
/// `set_flag/wait_flag` pair on entry (V -> S: prior vector accesses
/// complete before the stores) and one on exit (S -> V: subsequent vector
/// accesses observe the stores), mirroring the mem-based `broadcast_scalar`
/// (BrcSinglePoint.cpp) and RegBase/Vector/Sort1D.cpp. No per-element
/// synchronization is emitted.
///
/// NOTE (device-verified, Ascend950PR): moving the pair inside the
/// scalar-pipe branches only (so that pure-vector fills synchronize
/// nothing) corrupts the strided scalar stores under the fused compile --
/// they land at a 4x-scaled byte address (only elements with a byte
/// offset congruent to 0 mod 32 receive the fill). The payload IR is
/// byte-exact either way; the corruption appears only in the fused
/// pipeline (two-level addr-transform inconsistency, tracked as F1).
/// The entry-level pair is therefore kept on every path.
///
/// Alignment: the masked `vsts` fast path (NORM_B8/B16/B32) requires the
/// actual start address of each store run (`dst->aligned + dst->offset`
/// elements, and per-row for 2-D row fills) to be 32-byte aligned. Runs
/// starting at an unaligned address take a plain scalar fill: the
/// unaligned vector store `vstore(..., ADDRESS_UNALIGNED)` is byte-exact
/// in the payload IR but drops the view offset in the same fused pipeline
/// for the <= 4-byte widths, so it is not used there. 64-bit types always
/// use `vstore(..., ADDRESS_UNALIGNED)` through
/// `scalar_brc_fill_vstore_unaligned` because they have no masked `vsts`
/// support (`VectorReg<int64_t>` is the pair register
/// `vector_2xvl_s64`); that route is device-verified, including unaligned
/// view starts.
///
/// bool is bit-packed: offset and strides are in bits, sizes count logical
/// bool elements, LSB-first within each byte (same convention as
/// `get_condition_bit`/`set_result_bit` in Vector/CmpSel/Cmp1D.cpp and
/// `view_as` in Vector/VecUtils.h). Sixteen bools share one uint16 word, so
/// the word-pattern fast path is only taken when every 16-bit group of the
/// destination view is fully contained in the view (contiguous last dim,
/// 16-aligned bit offset, 16-aligned extents and row strides). Otherwise a
/// per-bit read-modify-write on the packed bytes is used, which only
/// touches bits belonging to the view.
///
/// The bitcode built from this file is linked into
/// meta_op.aiv.c310 / meta_op.aiv.920r1 (RegBase/Vector/CMakeLists.txt).

#include "ArchDefs.h"
#include "RegBase/ReductionUtils.h"
#include "RegBase/VecUtils.h"
#include "Utils.h"
#include "Vector/VecUtils.h"
#include <type_traits>

#if BISHENG_TPL_REGBASE

namespace {

/// Element count of one vector register for element type T.
template <typename T>
__aiv__ __attribute__((always_inline)) constexpr uint32_t
scalar_brc_ele_per_vl() {
  return VL_IN_BYTE / sizeof(T);
}

/// bool elements packed into one uint16 word.
constexpr int64_t kScalarBrcBitsPerU16Word = 16;

/// Sync helpers around scalar-pipe stores, shared by all scalar-store
/// fallbacks (strided fill, bool RMW). Same event discipline as
/// RegBase/Vector/Sort1D.cpp and the mem-based broadcast_scalar:
/// V -> S before writing, S -> V after. Contiguous fills emit only
/// V-pipe instructions and synchronize nothing.
__aiv__ __attribute__((always_inline)) void
scalar_brc_sync_before_scalar_store() {
  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
}

__aiv__ __attribute__((always_inline)) void
scalar_brc_sync_after_scalar_store() {
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
}

/// Plain scalar fill of `count` contiguous elements. The caller is
/// responsible for the pipeline synchronization around it.
template <typename T>
__aiv__ __attribute__((always_inline)) void
scalar_brc_fill_scalar(T src_val, __ubuf__ T *dst_ptr, int64_t count) {
  for (int64_t i = 0; i < count; ++i)
    dst_ptr[i] = src_val;
}

/// Vectorized fill through the unaligned vector store. Broadcasts `src_val`
/// with `vbr` and writes `count` contiguous elements starting at `dst_ptr`,
/// which may be arbitrarily aligned: `vstore(..., ADDRESS_UNALIGNED)` emits
/// `vstus_post`/`vstas` and writes exactly `elementCount` elements from the
/// given address, with no 32-byte alignment requirement and no mask, so no
/// element outside the view is touched (sentinel-safe). This covers every
/// broadcast width: `vstore` has overloads for s8/u8/s16/u16/bf16/f16/s32/
/// u32/f32 and for the 64-bit pair registers `vector_2xvl_s64`/`u64`
/// (`VectorReg<int64_t>`/`VectorReg<uint64_t>`), and `vbr` likewise accepts
/// every width including full 64-bit scalars.
///
/// `ElemsPerCall` is the largest element count one `vstore` call may write:
/// the intrinsic stores at most one vector register per `vstus_post`, so a
/// single-register type caps at `VL_IN_BYTE / sizeof(T)` elements and a
/// 64-bit pair register at twice that. Full-register runs are chunked into
/// `<= 65535`-iteration loops (ccec requires uint16_t induction variables
/// inside `__VEC_SCOPE__`), and every address is computed in 64-bit first
/// (a UB-resident view keeps each offset well inside 32 bits).
///
/// No pipeline synchronization here: only V-pipe instructions are emitted;
/// the entries wrap the whole fill (this helper included) in one
/// synchronization pair.
template <typename T, int64_t ElemsPerCall>
__aiv__ __attribute__((always_inline)) void
scalar_brc_fill_vstore_unaligned(T src_val, __ubuf__ T *dst_ptr,
                                 int64_t count) {
  if (count <= 0)
    return;
  constexpr int64_t kItersPerVecLoop = 65535;
  const int64_t fullCalls = count / ElemsPerCall;
  const int64_t tailCnt = count % ElemsPerCall;

  for (int64_t chunk = 0; chunk < fullCalls; chunk += kItersPerVecLoop) {
    const int64_t iters = (fullCalls - chunk < kItersPerVecLoop)
                              ? fullCalls - chunk
                              : kItersPerVecLoop;
    __VEC_SCOPE__ {
      VectorReg<T> val;
      vbr(val, src_val);
      for (uint16_t r = 0; r < (uint16_t)iters; ++r) {
        const uint32_t off = (uint32_t)((chunk + r) * ElemsPerCall);
        vstore(dst_ptr + off, val, (uint32_t)ElemsPerCall, ADDRESS_UNALIGNED);
      }
    }
  }

  if (tailCnt != 0) {
    const uint32_t tailOff = (uint32_t)(fullCalls * ElemsPerCall);
    __VEC_SCOPE__ {
      VectorReg<T> val;
      vbr(val, src_val);
      vstore(dst_ptr + tailOff, val, (uint32_t)tailCnt, ADDRESS_UNALIGNED);
    }
  }
}

/// Vectorized contiguous 1d fill: broadcast `src_val` over `count` elements
/// starting at `dst_ptr` (element stride 1).
///
/// 64-bit types have no masked `vsts` support on the reg-based arch
/// (`VectorReg<int64_t>` is the pair register `vector_2xvl_s64`), but
/// `vbr` + `vstore(..., ADDRESS_UNALIGNED)` cover them fully; each call
/// writes one full pair register. For the other widths the aligned path
/// stores with NORM_B8/B16/B32, which requires the actual start address of
/// the run to be 32-byte aligned; a run starting at an unaligned address
/// uses the unaligned vector store instead of scalar stores. Both unaligned
/// routes go through `scalar_brc_fill_vstore_unaligned`, which writes
/// exactly `count` elements.
///
/// No pipeline synchronization here: only V-pipe instructions are emitted;
/// the entries wrap the whole fill (this helper included) in one
/// synchronization pair.
template <typename T>
__aiv__ __attribute__((always_inline)) void
scalar_brc_1d_core(T src_val, __ubuf__ T *dst_ptr, int64_t count) {
  if (count <= 0)
    return;

  if constexpr (sizeof(T) == 8) {
    // One vstore call writes at most one vector_2xvl register, i.e. two
    // vector registers' worth of 8-byte elements.
    scalar_brc_fill_vstore_unaligned<T,
                                     2 * (int64_t)scalar_brc_ele_per_vl<T>()>(
        src_val, dst_ptr, count);
  } else if (!isAddress32ByteAligned(dst_ptr)) {
    // Keep scalar fallback for <=32-bit unaligned stores: A5 fused compilation
    // currently mishandles view offsets with vstore(ADDRESS_UNALIGNED).
    // The 64-bit pair-register path is device-verified.
    // Revisit vectorization after the fusion toolchain issue is fixed.
    scalar_brc_fill_scalar<T>(src_val, dst_ptr, count);
  } else {
    const uint32_t elePerVL = scalar_brc_ele_per_vl<T>();
    // Keep the loop bookkeeping in 64-bit: `count` can be any destination
    // extent, so neither the register count nor the element offsets may be
    // narrowed before the multiplication. ccec requires uint16_t induction
    // variables inside __VEC_SCOPE__, so the full-register runs are chunked
    // into <= 65535-iteration loops, and every vsts offset operand is
    // computed in 64-bit first (a UB-resident view keeps each offset well
    // inside 32 bits).
    const int64_t fullRegs = count / elePerVL;
    const int64_t tailCnt = count % elePerVL;
    constexpr int64_t kItersPerVecLoop = 65535;
    uint32_t fullMaskCnt = elePerVL;

    for (int64_t chunk = 0; chunk < fullRegs; chunk += kItersPerVecLoop) {
      const int64_t iters = (fullRegs - chunk < kItersPerVecLoop)
                                ? fullRegs - chunk
                                : kItersPerVecLoop;
      __VEC_SCOPE__ {
        VectorReg<T> val;
        vbr(val, src_val);

        vector_bool mFull;
        CREATE_MASK_BY_SIZE(mFull, T, fullMaskCnt);
        for (uint16_t r = 0; r < (uint16_t)iters; ++r) {
          const uint32_t off = (uint32_t)((chunk + r) * elePerVL);
          if constexpr (sizeof(T) == 1)
            vsts(val, dst_ptr, off, NORM_B8, mFull);
          else if constexpr (sizeof(T) == 2)
            vsts(val, dst_ptr, off, NORM_B16, mFull);
          else
            vsts(val, dst_ptr, off, NORM_B32, mFull);
        }
      }
    }

    if (tailCnt != 0) {
      uint32_t tailMaskCnt = (uint32_t)tailCnt;
      const uint32_t tailOff = (uint32_t)(fullRegs * elePerVL);
      __VEC_SCOPE__ {
        VectorReg<T> val;
        vbr(val, src_val);

        vector_bool mTail;
        CREATE_MASK_BY_SIZE(mTail, T, tailMaskCnt);
        if constexpr (sizeof(T) == 1)
          vsts(val, dst_ptr, tailOff, NORM_B8, mTail);
        else if constexpr (sizeof(T) == 2)
          vsts(val, dst_ptr, tailOff, NORM_B16, mTail);
        else
          vsts(val, dst_ptr, tailOff, NORM_B32, mTail);
      }
    }
  }
}

/// Scalar fill of a strided run, used for non-contiguous tails and views.
/// No pipeline synchronization here; entries wrap the whole fill once.
template <typename T>
__aiv__ __attribute__((always_inline)) void
scalar_brc_by_scalar(T src_val, __ubuf__ T *dst_ptr, int64_t count,
                     int64_t stride) {
  for (int64_t i = 0; i < count; ++i)
    *(dst_ptr + i * stride) = src_val;
}

/// 1d entry. Mirrors the mem-based `broadcast_scalar_1d` semantics: any
/// stride other than 1 takes the scalar path. The whole fill is wrapped in
/// one V -> S / S -> V synchronization pair so the scalar stores cannot
/// race with surrounding vector traffic.
///
/// NOTE (device-verified, Ascend950PR): scoping the pair to the scalar
/// branch only was measured to corrupt the strided scalar stores under the
/// fused compile (stores land at a 4x-scaled byte address; see the F1
/// two-level addr-transform note). The entry-level pair is the only
/// placement that is correct on device, so it is kept for every path.
template <typename T>
__aiv__ __attribute__((always_inline)) void
scalar_brc_1d(T src_val, memref_t<__ubuf__ T, 1> *dst) {
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  // Keep entry-level V/S sync: sinking it into scalar-only branches caused
  // incorrect results for mixed vector -> strided-scalar -> vector execution
  // under A5 fused compilation.
  scalar_brc_sync_before_scalar_store();
  if (dst->strides[0] == 1)
    scalar_brc_1d_core<T>(src_val, dst_ptr, dst->sizes[0]);
  else
    scalar_brc_by_scalar<T>(src_val, dst_ptr, dst->sizes[0], dst->strides[0]);
  scalar_brc_sync_after_scalar_store();
}

/// 2d entry. Mirrors the mem-based `broadcast_scalar_2d` semantics:
/// * fully contiguous (strides[0] == sizes[1], strides[1] == 1) is flattened
///   to a single 1d fill;
/// * row-contiguous (strides[1] == 1) fills each row with the 1d core, which
///   handles per-row start alignment (all of its paths are vectorized);
/// * otherwise the scalar path.
/// The whole fill is wrapped in one V -> S / S -> V synchronization pair
/// (same placement discipline and device evidence as scalar_brc_1d).
template <typename T>
__aiv__ __attribute__((always_inline)) void
scalar_brc_2d(T src_val, memref_t<__ubuf__ T, 2> *dst) {
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  // Keep entry-level V/S sync for the same reason as scalar_brc_1d.
  scalar_brc_sync_before_scalar_store();
  if (dst->strides[1] != 1) {
    for (int64_t i = 0; i < dst->sizes[0]; ++i)
      scalar_brc_by_scalar<T>(src_val, dst_ptr + i * dst->strides[0],
                              dst->sizes[1], dst->strides[1]);
    scalar_brc_sync_after_scalar_store();
    return;
  }
  if (dst->strides[0] == dst->sizes[1]) {
    scalar_brc_1d_core<T>(src_val, dst_ptr, dst->sizes[0] * dst->sizes[1]);
    scalar_brc_sync_after_scalar_store();
    return;
  }
  for (int64_t i = 0; i < dst->sizes[0]; ++i)
    scalar_brc_1d_core<T>(src_val, dst_ptr + i * dst->strides[0],
                          dst->sizes[1]);
  scalar_brc_sync_after_scalar_store();
}

/// bfloat16_t has no `vbr` overload on the reg-based arch; drive it through
/// the bit-identical uint16_t path (plain reinterpretation of the scalar and
/// of the memref, both 2-byte elements).
template <typename T> struct ScalarBrcAsU16 : std::false_type {};
template <> struct ScalarBrcAsU16<bfloat16_t> : std::true_type {};

template <typename T>
__aiv__ __attribute__((always_inline)) uint16_t
scalar_brc_scalar_as_u16(const T &val) {
  static_assert(sizeof(T) == sizeof(uint16_t),
                "unexpected scalar width for u16 reinterpretation");
  return *reinterpret_cast<const uint16_t *>(&val);
}

/// Per-bit read-modify-write fallback for bit-packed bool, 1d. Only bits
/// belonging to the logical view are modified; bits of the packed words
/// outside the view are preserved. Same byte/bit order as set_result_bit in
/// Vector/CmpSel/Cmp1D.cpp (offset and strides are in bits, LSB-first).
/// The caller wraps this in the V -> S / S -> V synchronization pair, which
/// also orders the read-modify-write accesses against vector traffic.
__aiv__ __attribute__((always_inline)) void
scalar_brc_bool_rmw_1d(bool src_val, memref_t<__ubuf__ bool, 1> *dst) {
  __ubuf__ uint8_t *base = reinterpret_cast<__ubuf__ uint8_t *>(dst->aligned);
  for (int64_t i = 0; i < dst->sizes[0]; ++i) {
    const int64_t absBit = dst->offset + i * dst->strides[0];
    const int64_t bytePos = absBit >> 3;
    const int64_t bitInByte = absBit & 7;
    if (src_val)
      base[bytePos] |= (uint8_t)(1u << bitInByte);
    else
      base[bytePos] &= (uint8_t) ~(1u << bitInByte);
  }
}

/// Per-bit read-modify-write fallback for bit-packed bool, 2d.
__aiv__ __attribute__((always_inline)) void
scalar_brc_bool_rmw_2d(bool src_val, memref_t<__ubuf__ bool, 2> *dst) {
  __ubuf__ uint8_t *base = reinterpret_cast<__ubuf__ uint8_t *>(dst->aligned);
  for (int64_t i = 0; i < dst->sizes[0]; ++i) {
    for (int64_t j = 0; j < dst->sizes[1]; ++j) {
      const int64_t absBit =
          dst->offset + i * dst->strides[0] + j * dst->strides[1];
      const int64_t bytePos = absBit >> 3;
      const int64_t bitInByte = absBit & 7;
      if (src_val)
        base[bytePos] |= (uint8_t)(1u << bitInByte);
      else
        base[bytePos] &= (uint8_t) ~(1u << bitInByte);
    }
  }
}

/// 1d bool entry. Fast path: the view is contiguous, starts on a 16-bit
/// boundary and covers whole 16-bit groups, so every written uint16 word
/// lies fully inside the view and a word-pattern fill is exact; the fill
/// goes through the uint16 1d entry and its synchronization pair. All
/// other layouts take the per-bit RMW fallback, which never touches bits
/// outside the view and synchronizes around its scalar stores.
__aiv__ __attribute__((always_inline)) void
scalar_brc_bool_1d(bool src_val, memref_t<__ubuf__ bool, 1> *dst) {
  if (dst->strides[0] == 1 && dst->offset % kScalarBrcBitsPerU16Word == 0 &&
      dst->sizes[0] % kScalarBrcBitsPerU16Word == 0) {
    memref_t<__ubuf__ uint16_t, 1> dstU16;
    dstU16.allocated = reinterpret_cast<__ubuf__ uint16_t *>(dst->allocated);
    dstU16.aligned = reinterpret_cast<__ubuf__ uint16_t *>(dst->aligned);
    dstU16.offset = dst->offset / kScalarBrcBitsPerU16Word;
    dstU16.sizes[0] = dst->sizes[0] / kScalarBrcBitsPerU16Word;
    dstU16.strides[0] = 1;
    scalar_brc_1d<uint16_t>(src_val ? 0xFFFFu : 0u, &dstU16);
    return;
  }
  scalar_brc_sync_before_scalar_store();
  scalar_brc_bool_rmw_1d(src_val, dst);
  scalar_brc_sync_after_scalar_store();
}

/// 2d bool entry. Fast path: the last dim is contiguous, the bit offset and
/// every row stride are multiples of 16 and the last-dim extent is a
/// multiple of 16, so rows map onto whole uint16 words (a row stride equal
/// to the row length gives the fully-contiguous flatten inside
/// scalar_brc_2d); the fill goes through the uint16 entry and its
/// synchronization pair. All other layouts take the per-bit RMW fallback
/// with its own synchronization pair.
__aiv__ __attribute__((always_inline)) void
scalar_brc_bool_2d(bool src_val, memref_t<__ubuf__ bool, 2> *dst) {
  if (dst->strides[1] == 1 && dst->offset % kScalarBrcBitsPerU16Word == 0 &&
      dst->sizes[1] % kScalarBrcBitsPerU16Word == 0 &&
      dst->strides[0] % kScalarBrcBitsPerU16Word == 0) {
    memref_t<__ubuf__ uint16_t, 2> dstU16;
    dstU16.allocated = reinterpret_cast<__ubuf__ uint16_t *>(dst->allocated);
    dstU16.aligned = reinterpret_cast<__ubuf__ uint16_t *>(dst->aligned);
    dstU16.offset = dst->offset / kScalarBrcBitsPerU16Word;
    dstU16.sizes[0] = dst->sizes[0];
    dstU16.strides[0] = dst->strides[0] / kScalarBrcBitsPerU16Word;
    dstU16.sizes[1] = dst->sizes[1] / kScalarBrcBitsPerU16Word;
    dstU16.strides[1] = 1;
    scalar_brc_2d<uint16_t>(src_val ? 0xFFFFu : 0u, &dstU16);
    return;
  }
  scalar_brc_sync_before_scalar_store();
  scalar_brc_bool_rmw_2d(src_val, dst);
  scalar_brc_sync_after_scalar_store();
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
scalar_brc_dispatch_1d(T src_val, memref_t<__ubuf__ T, 1> *dst) {
  if constexpr (ScalarBrcAsU16<T>::value) {
    uint16_t raw = scalar_brc_scalar_as_u16(src_val);
    scalar_brc_1d<uint16_t>(
        raw, reinterpret_cast<memref_t<__ubuf__ uint16_t, 1> *>(dst));
  } else {
    scalar_brc_1d<T>(src_val, dst);
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
scalar_brc_dispatch_2d(T src_val, memref_t<__ubuf__ T, 2> *dst) {
  if constexpr (ScalarBrcAsU16<T>::value) {
    uint16_t raw = scalar_brc_scalar_as_u16(src_val);
    scalar_brc_2d<uint16_t>(
        raw, reinterpret_cast<memref_t<__ubuf__ uint16_t, 2> *>(dst));
  } else {
    scalar_brc_2d<T>(src_val, dst);
  }
}

} // namespace

extern "C" {

//===-------------------------------------------------------------------===//
// broadcast scalar to 1d
//===-------------------------------------------------------------------===//
#define BISHENG_TPL_SCALAR_BRC_1D(TYPE)                                        \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_broadcast_scalar_##TYPE##_to_1d(                            \
          TYPE scalar, memref_t<__ubuf__ TYPE, 1> *dst) {                      \
    scalar_brc_dispatch_1d<TYPE>(scalar, dst);                                 \
  }

BISHENG_TPL_SCALAR_BRC_1D(int8_t)
BISHENG_TPL_SCALAR_BRC_1D(uint8_t)
BISHENG_TPL_SCALAR_BRC_1D(int16_t)
BISHENG_TPL_SCALAR_BRC_1D(uint16_t)
BISHENG_TPL_SCALAR_BRC_1D(bfloat16_t)
BISHENG_TPL_SCALAR_BRC_1D(half)
BISHENG_TPL_SCALAR_BRC_1D(int32_t)
BISHENG_TPL_SCALAR_BRC_1D(uint32_t)
BISHENG_TPL_SCALAR_BRC_1D(float)
BISHENG_TPL_SCALAR_BRC_1D(int64_t)
BISHENG_TPL_SCALAR_BRC_1D(uint64_t)

//===-------------------------------------------------------------------===//
// broadcast scalar to 2d
//===-------------------------------------------------------------------===//
#define BISHENG_TPL_SCALAR_BRC_2D(TYPE)                                        \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_broadcast_scalar_##TYPE##_to_2d(                            \
          TYPE scalar, memref_t<__ubuf__ TYPE, 2> *dst) {                      \
    scalar_brc_dispatch_2d<TYPE>(scalar, dst);                                 \
  }

BISHENG_TPL_SCALAR_BRC_2D(int8_t)
BISHENG_TPL_SCALAR_BRC_2D(uint8_t)
BISHENG_TPL_SCALAR_BRC_2D(int16_t)
BISHENG_TPL_SCALAR_BRC_2D(uint16_t)
BISHENG_TPL_SCALAR_BRC_2D(bfloat16_t)
BISHENG_TPL_SCALAR_BRC_2D(half)
BISHENG_TPL_SCALAR_BRC_2D(int32_t)
BISHENG_TPL_SCALAR_BRC_2D(uint32_t)
BISHENG_TPL_SCALAR_BRC_2D(float)
BISHENG_TPL_SCALAR_BRC_2D(int64_t)
BISHENG_TPL_SCALAR_BRC_2D(uint64_t)

//===-------------------------------------------------------------------===//
// bool entries
//===-------------------------------------------------------------------===//
__aiv__ __attribute__((always_inline)) void
_mlir_ciface_broadcast_scalar_bool_to_1d(bool scalar,
                                         memref_t<__ubuf__ bool, 1> *dst) {
  scalar_brc_bool_1d(scalar, dst);
}

__aiv__ __attribute__((always_inline)) void
_mlir_ciface_broadcast_scalar_bool_to_2d(bool scalar,
                                         memref_t<__ubuf__ bool, 2> *dst) {
  scalar_brc_bool_2d(scalar, dst);
}

} // extern "C"

#endif // BISHENG_TPL_REGBASE
