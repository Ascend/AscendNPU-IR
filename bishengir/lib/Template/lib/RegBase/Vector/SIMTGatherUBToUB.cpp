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

#if defined(__DAV_C310__)

#include "RegBase/SIMTGatherUBToUB.h"

// UB-to-UB SIMT gather kernels.
//   - src resides in UB as a multi-dim memref with sizes/strides; the
//     dispatcher uses `src->aligned + src->offset` as base and forwards
//     all strides from src->strides[].
//   - Loop bounds come from indices->sizes[]; no external startOffset/endOffset.
//   - indexBoundary = src->sizes[dimension], injected by the dispatcher.
//     Negative indices are folded via `indexVal += indexBoundary`; still
//     out-of-range after folding → `continue`.

constexpr uint32_t THREAD_DIM_1024 = 1024;

// Normalize negative index into [0, indexBoundary) and check validity,
// using branch-free arithmetic + a single unsigned comparison:
//   - Arithmetic right-shift extracts the sign mask (all-1s if negative,
//     0 if non-negative); AND with indexBoundary and add back to indexVal
//     — equivalent to "negative += indexBoundary, non-negative unchanged",
//     eliminating the conditional branch.
//   - A single `(uint64_t)indexVal >= indexBoundary` check: negative values
//     cast to huge unsigned numbers, covering both "still negative after
//     folding" and ">= indexBoundary" in one comparison.
// Returns true if index is valid (folded in-place), false if out-of-range
// (caller should `continue`).
template <typename ITYPE>
__simt_callee__ __aiv__ __attribute__((always_inline)) static bool
normalizeAndCheckIndex(ITYPE &indexVal, uint32_t indexBoundary) {
  indexVal += (indexVal >> (sizeof(ITYPE) * 8 - 1)) &
              static_cast<ITYPE>(indexBoundary);
  return static_cast<uint64_t>(indexVal) < indexBoundary;
}

template <typename DTYPE, typename ITYPE>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM_1024) __aiv__
    __attribute__((always_inline)) static void SimtGatherUB1D(
        __ubuf__ DTYPE *src, __ubuf__ ITYPE *indices, __ubuf__ DTYPE *dst,
        const uint32_t indexBoundary, const uint32_t size0,
        const uint32_t idxStride0, const uint32_t dstStride0,
        const uint32_t srcStride0) {
  for (uint32_t i0 = threadIdx.x; i0 < size0; i0 += blockDim.x) {
    ITYPE indexVal = indices[i0 * idxStride0];
    if (!normalizeAndCheckIndex(indexVal, indexBoundary)) {
      continue;
    }
    dst[i0 * dstStride0] = src[indexVal * srcStride0];
  }
}

template <typename DTYPE, typename ITYPE>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM_1024) __aiv__
    __attribute__((always_inline)) static void SimtGatherUB2D(
        __ubuf__ DTYPE *src, __ubuf__ ITYPE *indices, __ubuf__ DTYPE *dst,
        const uint32_t indexBoundary, const uint32_t numElems,
        const uint32_t shift1, const uint32_t m1, const uint32_t size1,
        const uint32_t idxStride0, const uint32_t idxStride1,
        const uint32_t dstStride0, const uint32_t dstStride1,
        const uint32_t srcStride0, const uint32_t srcStride1,
        const uint32_t dimension) {
  for (uint32_t elemIdx = threadIdx.x; elemIdx < numElems;
       elemIdx += blockDim.x) {
    uint32_t tyDiv = UintDivImpl<uint32_t>(elemIdx, m1, shift1);
    uint32_t ty = elemIdx - tyDiv * size1;
    uint32_t tx = tyDiv;
    ITYPE indexVal = indices[tx * idxStride0 + ty * idxStride1];
    if (!normalizeAndCheckIndex(indexVal, indexBoundary)) {
      continue;
    }
    const uint32_t dstIdx = tx * dstStride0 + ty * dstStride1;
    if (dimension == 0) {
      dst[dstIdx] = src[indexVal * srcStride0 + ty * srcStride1];
    } else if (dimension == 1) {
      dst[dstIdx] = src[tx * srcStride0 + indexVal * srcStride1];
    }
  }
}

template <typename DTYPE, typename ITYPE>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM_1024) __aiv__
    __attribute__((always_inline)) static void SimtGatherUB3D(
        __ubuf__ DTYPE *src, __ubuf__ ITYPE *indices, __ubuf__ DTYPE *dst,
        const uint32_t indexBoundary, const uint32_t numElems,
        const uint32_t shift1, const uint32_t m1, const uint32_t shift2,
        const uint32_t m2, const uint32_t size1, const uint32_t size2,
        const uint32_t idxStride0, const uint32_t idxStride1,
        const uint32_t idxStride2, const uint32_t dstStride0,
        const uint32_t dstStride1, const uint32_t dstStride2,
        const uint32_t srcStride0, const uint32_t srcStride1,
        const uint32_t srcStride2, const uint32_t dimension) {
  for (uint32_t elemIdx = threadIdx.x; elemIdx < numElems;
       elemIdx += blockDim.x) {
    uint32_t tzDiv = UintDivImpl<uint32_t>(elemIdx, m2, shift2);
    uint32_t tz = elemIdx - tzDiv * size2;
    uint32_t tyDiv = UintDivImpl<uint32_t>(tzDiv, m1, shift1);
    uint32_t ty = tzDiv - tyDiv * size1;
    uint32_t tx = tyDiv;
    ITYPE indexVal =
        indices[tx * idxStride0 + ty * idxStride1 + tz * idxStride2];
    if (!normalizeAndCheckIndex(indexVal, indexBoundary)) {
      continue;
    }
    const uint32_t dstIdx =
        tx * dstStride0 + ty * dstStride1 + tz * dstStride2;
    if (dimension == 0) {
      dst[dstIdx] = src[indexVal * srcStride0 + ty * srcStride1 +
                        tz * srcStride2];
    } else if (dimension == 1) {
      dst[dstIdx] = src[tx * srcStride0 + indexVal * srcStride1 +
                        tz * srcStride2];
    } else if (dimension == 2) {
      dst[dstIdx] = src[tx * srcStride0 + ty * srcStride1 +
                        indexVal * srcStride2];
    }
  }
}

template <typename DTYPE, typename ITYPE>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM_1024) __aiv__
    __attribute__((always_inline)) static void SimtGatherUB4D(
        __ubuf__ DTYPE *src, __ubuf__ ITYPE *indices, __ubuf__ DTYPE *dst,
        const uint32_t indexBoundary, const uint32_t numElems,
        const uint32_t shift1, const uint32_t m1, const uint32_t shift2,
        const uint32_t m2, const uint32_t shift3, const uint32_t m3,
        const uint32_t size1, const uint32_t size2, const uint32_t size3,
        const uint32_t idxStride0, const uint32_t idxStride1,
        const uint32_t idxStride2, const uint32_t idxStride3,
        const uint32_t dstStride0, const uint32_t dstStride1,
        const uint32_t dstStride2, const uint32_t dstStride3,
        const uint32_t srcStride0, const uint32_t srcStride1,
        const uint32_t srcStride2, const uint32_t srcStride3,
        const uint32_t dimension) {
  for (uint32_t elemIdx = threadIdx.x; elemIdx < numElems;
       elemIdx += blockDim.x) {
    uint32_t tkDiv = UintDivImpl<uint32_t>(elemIdx, m3, shift3);
    uint32_t tk = elemIdx - tkDiv * size3;
    uint32_t tzDiv = UintDivImpl<uint32_t>(tkDiv, m2, shift2);
    uint32_t tz = tkDiv - tzDiv * size2;
    uint32_t tyDiv = UintDivImpl<uint32_t>(tzDiv, m1, shift1);
    uint32_t ty = tzDiv - tyDiv * size1;
    uint32_t tx = tyDiv;
    const uint32_t idxOff = tx * idxStride0 + ty * idxStride1 +
                            tz * idxStride2 + tk * idxStride3;
    ITYPE indexVal = indices[idxOff];
    if (!normalizeAndCheckIndex(indexVal, indexBoundary)) {
      continue;
    }
    const uint32_t dstIdx = tx * dstStride0 + ty * dstStride1 +
                            tz * dstStride2 + tk * dstStride3;
    if (dimension == 0) {
      dst[dstIdx] = src[indexVal * srcStride0 + ty * srcStride1 +
                        tz * srcStride2 + tk * srcStride3];
    } else if (dimension == 1) {
      dst[dstIdx] = src[tx * srcStride0 + indexVal * srcStride1 +
                        tz * srcStride2 + tk * srcStride3];
    } else if (dimension == 2) {
      dst[dstIdx] = src[tx * srcStride0 + ty * srcStride1 +
                        indexVal * srcStride2 + tk * srcStride3];
    } else if (dimension == 3) {
      dst[dstIdx] = src[tx * srcStride0 + ty * srcStride1 +
                        tz * srcStride2 + indexVal * srcStride3];
    }
  }
}

template <typename DTYPE, typename ITYPE>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM_1024) __aiv__
    __attribute__((always_inline)) static void SimtGatherUB5D(
        __ubuf__ DTYPE *src, __ubuf__ ITYPE *indices, __ubuf__ DTYPE *dst,
        const uint32_t indexBoundary, const uint32_t numElems,
        const uint32_t shift1, const uint32_t m1, const uint32_t shift2,
        const uint32_t m2, const uint32_t shift3, const uint32_t m3,
        const uint32_t shift4, const uint32_t m4, const uint32_t size1,
        const uint32_t size2, const uint32_t size3, const uint32_t size4,
        const uint32_t idxStride0, const uint32_t idxStride1,
        const uint32_t idxStride2, const uint32_t idxStride3,
        const uint32_t idxStride4, const uint32_t dstStride0,
        const uint32_t dstStride1, const uint32_t dstStride2,
        const uint32_t dstStride3, const uint32_t dstStride4,
        const uint32_t srcStride0, const uint32_t srcStride1,
        const uint32_t srcStride2, const uint32_t srcStride3,
        const uint32_t srcStride4, const uint32_t dimension) {
  for (uint32_t elemIdx = threadIdx.x; elemIdx < numElems;
       elemIdx += blockDim.x) {
    uint32_t tlDiv = UintDivImpl<uint32_t>(elemIdx, m4, shift4);
    uint32_t tl = elemIdx - tlDiv * size4;
    uint32_t tkDiv = UintDivImpl<uint32_t>(tlDiv, m3, shift3);
    uint32_t tk = tlDiv - tkDiv * size3;
    uint32_t tzDiv = UintDivImpl<uint32_t>(tkDiv, m2, shift2);
    uint32_t tz = tkDiv - tzDiv * size2;
    uint32_t tyDiv = UintDivImpl<uint32_t>(tzDiv, m1, shift1);
    uint32_t ty = tzDiv - tyDiv * size1;
    uint32_t tx = tyDiv;
    const uint32_t idxOff = tx * idxStride0 + ty * idxStride1 +
                            tz * idxStride2 + tk * idxStride3 +
                            tl * idxStride4;
    ITYPE indexVal = indices[idxOff];
    if (!normalizeAndCheckIndex(indexVal, indexBoundary)) {
      continue;
    }
    const uint32_t dstIdx = tx * dstStride0 + ty * dstStride1 +
                            tz * dstStride2 + tk * dstStride3 +
                            tl * dstStride4;
    if (dimension == 0) {
      dst[dstIdx] = src[indexVal * srcStride0 + ty * srcStride1 +
                        tz * srcStride2 + tk * srcStride3 + tl * srcStride4];
    } else if (dimension == 1) {
      dst[dstIdx] = src[tx * srcStride0 + indexVal * srcStride1 +
                        tz * srcStride2 + tk * srcStride3 + tl * srcStride4];
    } else if (dimension == 2) {
      dst[dstIdx] = src[tx * srcStride0 + ty * srcStride1 +
                        indexVal * srcStride2 + tk * srcStride3 +
                        tl * srcStride4];
    } else if (dimension == 3) {
      dst[dstIdx] = src[tx * srcStride0 + ty * srcStride1 +
                        tz * srcStride2 + indexVal * srcStride3 +
                        tl * srcStride4];
    } else if (dimension == 4) {
      dst[dstIdx] = src[tx * srcStride0 + ty * srcStride1 +
                        tz * srcStride2 + tk * srcStride3 +
                        indexVal * srcStride4];
    }
  }
}

template <int DIM, typename DTYPE, typename ITYPE>
__aiv__ __attribute__((always_inline)) void
GatherUBToUB(memref_t<__ubuf__ DTYPE, DIM> *src,
             memref_t<__ubuf__ ITYPE, DIM> *indices,
             memref_t<__ubuf__ DTYPE, DIM> *dst, const int32_t dimension) {
  static_assert(DIM >= 1 && DIM <= 5, "DIM must be in [1, 5]");
  static_assert(std::is_same<DTYPE, float>::value ||
                    std::is_same<DTYPE, float16_t>::value ||
                    std::is_same<DTYPE, half>::value ||
                    std::is_same<DTYPE, bfloat16_t>::value ||
                    std::is_same<DTYPE, int8_t>::value ||
                    std::is_same<DTYPE, int16_t>::value ||
                    std::is_same<DTYPE, int32_t>::value ||
                    std::is_same<DTYPE, uint8_t>::value ||
                    std::is_same<DTYPE, uint16_t>::value ||
                    std::is_same<DTYPE, uint32_t>::value,
                "Currently dtype of data supports only \
                  fp32|fp16|bf16|i8|i16|i32|u8|u16|u32");
  static_assert(std::is_same<ITYPE, int32_t>::value ||
                    std::is_same<ITYPE, int64_t>::value,
                "Currently dtype of index supports only int32|int64.");
  // src: multi-dim UB memref, base = aligned + offset; strides taken from
  // src->strides[] for all dimensions.
  // indexBoundary = src->sizes[dimension], no external parameter needed.
  __ubuf__ DTYPE *srcBase =
      reinterpret_cast<__ubuf__ DTYPE *>(src->aligned + src->offset);
  __ubuf__ ITYPE *idxBase =
      reinterpret_cast<__ubuf__ ITYPE *>(indices->aligned + indices->offset);
  __ubuf__ DTYPE *dstBase =
      reinterpret_cast<__ubuf__ DTYPE *>(dst->aligned + dst->offset);
  const uint32_t indexBoundary = src->sizes[dimension];

  if constexpr (DIM == 1) {
    const uint32_t size0 = indices->sizes[0];
    const uint32_t idxStride0 = indices->strides[0];
    const uint32_t dstStride0 = dst->strides[0];
    const uint32_t srcStride0 = src->strides[0];
    cce::async_invoke<SimtGatherUB1D<DTYPE, ITYPE>>(
        cce::dim3{THREAD_DIM_1024}, srcBase, idxBase, dstBase, indexBoundary,
        size0, idxStride0, dstStride0, srcStride0);
  } else if constexpr (DIM == 2) {
    const uint32_t size0 = indices->sizes[0];
    const uint32_t size1 = indices->sizes[1];
    const uint32_t idxStride0 = indices->strides[0];
    const uint32_t idxStride1 = indices->strides[1];
    const uint32_t dstStride0 = dst->strides[0];
    const uint32_t dstStride1 = dst->strides[1];
    const uint32_t srcStride0 = src->strides[0];
    const uint32_t srcStride1 = src->strides[1];
    const uint32_t numElems = size0 * size1;
    uint32_t shift1 = 0;
    uint32_t m1 = 0;
    GetUintDivMagicAndShiftImpl<uint32_t>(m1, shift1, size1);
    cce::async_invoke<SimtGatherUB2D<DTYPE, ITYPE>>(
        cce::dim3{THREAD_DIM_1024}, srcBase, idxBase, dstBase, indexBoundary,
        numElems, shift1, m1, size1, idxStride0, idxStride1, dstStride0,
        dstStride1, srcStride0, srcStride1, dimension);
  } else if constexpr (DIM == 3) {
    const uint32_t size0 = indices->sizes[0];
    const uint32_t size1 = indices->sizes[1];
    const uint32_t size2 = indices->sizes[2];
    const uint32_t idxStride0 = indices->strides[0];
    const uint32_t idxStride1 = indices->strides[1];
    const uint32_t idxStride2 = indices->strides[2];
    const uint32_t dstStride0 = dst->strides[0];
    const uint32_t dstStride1 = dst->strides[1];
    const uint32_t dstStride2 = dst->strides[2];
    const uint32_t srcStride0 = src->strides[0];
    const uint32_t srcStride1 = src->strides[1];
    const uint32_t srcStride2 = src->strides[2];
    const uint32_t numElems = size0 * size1 * size2;
    uint32_t shift1 = 0;
    uint32_t m1 = 0;
    GetUintDivMagicAndShiftImpl<uint32_t>(m1, shift1, size1);
    uint32_t shift2 = 0;
    uint32_t m2 = 0;
    GetUintDivMagicAndShiftImpl<uint32_t>(m2, shift2, size2);
    cce::async_invoke<SimtGatherUB3D<DTYPE, ITYPE>>(
        cce::dim3{THREAD_DIM_1024}, srcBase, idxBase, dstBase, indexBoundary,
        numElems, shift1, m1, shift2, m2, size1, size2, idxStride0, idxStride1,
        idxStride2, dstStride0, dstStride1, dstStride2, srcStride0, srcStride1,
        srcStride2, dimension);
  } else if constexpr (DIM == 4) {
    const uint32_t size0 = indices->sizes[0];
    const uint32_t size1 = indices->sizes[1];
    const uint32_t size2 = indices->sizes[2];
    const uint32_t size3 = indices->sizes[3];
    const uint32_t idxStride0 = indices->strides[0];
    const uint32_t idxStride1 = indices->strides[1];
    const uint32_t idxStride2 = indices->strides[2];
    const uint32_t idxStride3 = indices->strides[3];
    const uint32_t dstStride0 = dst->strides[0];
    const uint32_t dstStride1 = dst->strides[1];
    const uint32_t dstStride2 = dst->strides[2];
    const uint32_t dstStride3 = dst->strides[3];
    const uint32_t srcStride0 = src->strides[0];
    const uint32_t srcStride1 = src->strides[1];
    const uint32_t srcStride2 = src->strides[2];
    const uint32_t srcStride3 = src->strides[3];
    const uint32_t numElems = size0 * size1 * size2 * size3;
    uint32_t shift1 = 0;
    uint32_t m1 = 0;
    GetUintDivMagicAndShiftImpl<uint32_t>(m1, shift1, size1);
    uint32_t shift2 = 0;
    uint32_t m2 = 0;
    GetUintDivMagicAndShiftImpl<uint32_t>(m2, shift2, size2);
    uint32_t shift3 = 0;
    uint32_t m3 = 0;
    GetUintDivMagicAndShiftImpl<uint32_t>(m3, shift3, size3);
    cce::async_invoke<SimtGatherUB4D<DTYPE, ITYPE>>(
        cce::dim3{THREAD_DIM_1024}, srcBase, idxBase, dstBase, indexBoundary,
        numElems, shift1, m1, shift2, m2, shift3, m3, size1, size2, size3,
        idxStride0, idxStride1, idxStride2, idxStride3, dstStride0, dstStride1,
        dstStride2, dstStride3, srcStride0, srcStride1, srcStride2, srcStride3,
        dimension);
  } else if constexpr (DIM == 5) {
    const uint32_t size0 = indices->sizes[0];
    const uint32_t size1 = indices->sizes[1];
    const uint32_t size2 = indices->sizes[2];
    const uint32_t size3 = indices->sizes[3];
    const uint32_t size4 = indices->sizes[4];
    const uint32_t idxStride0 = indices->strides[0];
    const uint32_t idxStride1 = indices->strides[1];
    const uint32_t idxStride2 = indices->strides[2];
    const uint32_t idxStride3 = indices->strides[3];
    const uint32_t idxStride4 = indices->strides[4];
    const uint32_t dstStride0 = dst->strides[0];
    const uint32_t dstStride1 = dst->strides[1];
    const uint32_t dstStride2 = dst->strides[2];
    const uint32_t dstStride3 = dst->strides[3];
    const uint32_t dstStride4 = dst->strides[4];
    const uint32_t srcStride0 = src->strides[0];
    const uint32_t srcStride1 = src->strides[1];
    const uint32_t srcStride2 = src->strides[2];
    const uint32_t srcStride3 = src->strides[3];
    const uint32_t srcStride4 = src->strides[4];
    const uint32_t numElems = size0 * size1 * size2 * size3 * size4;
    uint32_t shift1 = 0;
    uint32_t m1 = 0;
    GetUintDivMagicAndShiftImpl<uint32_t>(m1, shift1, size1);
    uint32_t shift2 = 0;
    uint32_t m2 = 0;
    GetUintDivMagicAndShiftImpl<uint32_t>(m2, shift2, size2);
    uint32_t shift3 = 0;
    uint32_t m3 = 0;
    GetUintDivMagicAndShiftImpl<uint32_t>(m3, shift3, size3);
    uint32_t shift4 = 0;
    uint32_t m4 = 0;
    GetUintDivMagicAndShiftImpl<uint32_t>(m4, shift4, size4);
    cce::async_invoke<SimtGatherUB5D<DTYPE, ITYPE>>(
        cce::dim3{THREAD_DIM_1024}, srcBase, idxBase, dstBase, indexBoundary,
        numElems, shift1, m1, shift2, m2, shift3, m3, shift4, m4, size1, size2,
        size3, size4, idxStride0, idxStride1, idxStride2, idxStride3,
        idxStride4, dstStride0, dstStride1, dstStride2, dstStride3, dstStride4,
        srcStride0, srcStride1, srcStride2, srcStride3, srcStride4, dimension);
  }
}

// Expand all (dtype x itype) combinations at once; callers only specify DIM.
#define REGISTER_GATHER_UB_TO_UB_ALL_TYPES(dim)                                \
  REGISTER_GATHER_UB_TO_UB(dim, float, int32_t);                               \
  REGISTER_GATHER_UB_TO_UB(dim, float16_t, int32_t);                           \
  REGISTER_GATHER_UB_TO_UB(dim, half, int32_t);                                \
  REGISTER_GATHER_UB_TO_UB(dim, bfloat16_t, int32_t);                          \
  REGISTER_GATHER_UB_TO_UB(dim, int8_t, int32_t);                              \
  REGISTER_GATHER_UB_TO_UB(dim, int16_t, int32_t);                             \
  REGISTER_GATHER_UB_TO_UB(dim, int32_t, int32_t);                             \
  REGISTER_GATHER_UB_TO_UB(dim, uint8_t, int32_t);                             \
  REGISTER_GATHER_UB_TO_UB(dim, uint16_t, int32_t);                            \
  REGISTER_GATHER_UB_TO_UB(dim, uint32_t, int32_t);                            \
  REGISTER_GATHER_UB_TO_UB(dim, float, int64_t);                               \
  REGISTER_GATHER_UB_TO_UB(dim, float16_t, int64_t);                           \
  REGISTER_GATHER_UB_TO_UB(dim, half, int64_t);                                \
  REGISTER_GATHER_UB_TO_UB(dim, bfloat16_t, int64_t);                          \
  REGISTER_GATHER_UB_TO_UB(dim, int8_t, int64_t);                              \
  REGISTER_GATHER_UB_TO_UB(dim, int16_t, int64_t);                             \
  REGISTER_GATHER_UB_TO_UB(dim, int32_t, int64_t);                             \
  REGISTER_GATHER_UB_TO_UB(dim, uint8_t, int64_t);                             \
  REGISTER_GATHER_UB_TO_UB(dim, uint16_t, int64_t);                            \
  REGISTER_GATHER_UB_TO_UB(dim, uint32_t, int64_t)

extern "C" {
REGISTER_GATHER_UB_TO_UB_ALL_TYPES(1);
REGISTER_GATHER_UB_TO_UB_ALL_TYPES(2);
REGISTER_GATHER_UB_TO_UB_ALL_TYPES(3);
REGISTER_GATHER_UB_TO_UB_ALL_TYPES(4);
REGISTER_GATHER_UB_TO_UB_ALL_TYPES(5);
}
#undef REGISTER_GATHER_UB_TO_UB_ALL_TYPES
#endif
