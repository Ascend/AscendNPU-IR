/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#include "RegBase/SIMTGatherOutToUB.h"

constexpr uint32_t THREAD_DIM_1024 = 1024;

template <typename DTYPE, typename ITYPE>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM_1024) __aiv__
    __attribute__((always_inline)) static void SimtGather1D(
        __gm__ DTYPE *src, __ubuf__ ITYPE *indices, __ubuf__ DTYPE *dst,
        const uint32_t indexBoundary, const uint32_t endOffset0,
        const uint32_t size0, const uint32_t stride0, const uint32_t dstStride0,
        const uint32_t startOffset0) {
  for (uint32_t i0 = threadIdx.x;
       (i0 < size0) && (i0 + startOffset0 < endOffset0); i0 += blockDim.x) {
    ITYPE indexVal = indices[i0 * stride0];
    if (indexVal < 0) {
      indexVal += indexBoundary;
    }
    if (indexVal < 0 || indexVal >= indexBoundary) {
      break;
    }
    dst[i0 * dstStride0] = src[indexVal];
  }
}

template <typename DTYPE, typename ITYPE>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM_1024) __aiv__
    __attribute__((always_inline)) static void SimtGather2D(
        __gm__ DTYPE *src, __ubuf__ ITYPE *indices, __ubuf__ DTYPE *dst,
        const uint32_t indexBoundary, const uint32_t xStrideIndex0,
        const uint32_t numElems, const uint32_t shift1, const uint32_t m1,
        const uint32_t stride0, const uint32_t size1, const uint32_t stride1,
        const uint32_t dstStride0, const uint32_t dstStride1,
        const uint32_t endOffset0, const uint32_t endOffset1,
        const uint32_t startOffset0, const uint32_t startOffset1,
        const uint32_t dimension) {
  for (uint32_t elemIdx = threadIdx.x; elemIdx < numElems;
       elemIdx += blockDim.x) {
    uint32_t tyDiv = UintDivImpl<uint32_t>(elemIdx, m1, shift1);
    uint32_t ty = elemIdx - tyDiv * size1;
    uint32_t tx = tyDiv;
    uint32_t dim0Index = tx + startOffset0;
    uint32_t dim1Index = ty + startOffset1;
    if (dim0Index >= endOffset0 || dim1Index >= endOffset1) {
      break;
    }
    ITYPE indexVal = indices[tx * stride0 + ty * stride1];
    if (indexVal < 0) {
      indexVal += indexBoundary;
    }
    if (indexVal < 0 || indexVal >= indexBoundary) {
      break;
    }
    if (dimension == 1) {
      dst[tx * dstStride0 + ty * dstStride1] =
          src[dim0Index * xStrideIndex0 + indexVal];
    } else if (dimension == 0) {
      dst[tx * dstStride0 + ty * dstStride1] =
          src[indexVal * xStrideIndex0 + dim1Index];
    }
  }
}

template <typename DTYPE, typename ITYPE>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM_1024) __aiv__
    __attribute__((always_inline)) static void SimtGather3D(
        __gm__ DTYPE *src, __ubuf__ ITYPE *indices, __ubuf__ DTYPE *dst,
        const uint32_t indexBoundary, const uint32_t xStrideIndex0,
        const uint32_t xStrideIndex1, const uint32_t numElems,
        const uint32_t shift1, const uint32_t m1, const uint32_t shift2,
        const uint32_t m2, const uint32_t size1, const uint32_t size2,
        const uint32_t stride0, const uint32_t stride1, const uint32_t stride2,
        const uint32_t dstStride0, const uint32_t dstStride1,
        const uint32_t dstStride2, const uint32_t endOffset0,
        const uint32_t endOffset1, const uint32_t endOffset2,
        const uint32_t startOffset0, const uint32_t startOffset1,
        const uint32_t startOffset2, const uint32_t dimension) {
  for (uint32_t elemIdx = threadIdx.x; elemIdx < numElems;
       elemIdx += blockDim.x) {
    uint32_t tzDiv = UintDivImpl<uint32_t>(elemIdx, m2, shift2);
    uint32_t tz = elemIdx - tzDiv * size2;
    uint32_t tyDiv = UintDivImpl<uint32_t>(tzDiv, m1, shift1);
    uint32_t ty = tzDiv - tyDiv * size1;
    uint32_t tx = tyDiv;
    uint32_t dim0Index = tx + startOffset0;
    uint32_t dim1Index = ty + startOffset1;
    uint32_t dim2Index = tz + startOffset2;
    if (dim0Index >= endOffset0 || dim1Index >= endOffset1 ||
        dim2Index >= endOffset2) {
      break;
    }
    ITYPE indexVal = indices[tx * stride0 + ty * stride1 + tz * stride2];
    if (indexVal < 0) {
      indexVal += indexBoundary;
    }
    if (indexVal < 0 || indexVal >= indexBoundary) {
      break;
    } else if (dimension == 0) {
      dst[tx * dstStride0 + ty * dstStride1 + tz * dstStride2] =
          src[indexVal * xStrideIndex0 + dim1Index * xStrideIndex1 + dim2Index];
    } else if (dimension == 1) {
      dst[tx * dstStride0 + ty * dstStride1 + tz * dstStride2] =
          src[dim0Index * xStrideIndex0 + indexVal * xStrideIndex1 + dim2Index];
    } else if (dimension == 2) {
      dst[tx * dstStride0 + ty * dstStride1 + tz * dstStride2] =
          src[dim0Index * xStrideIndex0 + dim1Index * xStrideIndex1 + indexVal];
    }
  }
}

template <typename DTYPE, typename ITYPE>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM_1024) __aiv__
    __attribute__((always_inline)) static void SimtGather4D(
        __gm__ DTYPE *src, __ubuf__ ITYPE *indices, __ubuf__ DTYPE *dst,
        const uint32_t size0, const uint32_t stride0, const uint32_t size1,
        const uint32_t stride1, const uint32_t size2, const uint32_t stride2,
        const uint32_t size3, const uint32_t stride3, const uint32_t shift1,
        const uint32_t m1, const uint32_t shift2, const uint32_t m2,
        const uint32_t shift3, const uint32_t m3, const uint32_t dstStride0,
        const uint32_t dstStride1, const uint32_t dstStride2,
        const uint32_t dstStride3, const uint32_t xStrideIndex0,
        const uint32_t xStrideIndex1, const uint32_t xStrideIndex2,
        const uint32_t endOffset0, const uint32_t endOffset1,
        const uint32_t endOffset2, const uint32_t endOffset3,
        const uint32_t startOffset0, const uint32_t startOffset1,
        const uint32_t startOffset2, const uint32_t startOffset3,
        const uint32_t dimension, const uint32_t indexBoundary) {
  const uint32_t numElems = size0 * size1 * size2 * size3;
  for (uint32_t elemIdx = threadIdx.x; elemIdx < numElems;
       elemIdx += blockDim.x) {
    uint32_t tkDiv = UintDivImpl<uint32_t>(elemIdx, m3, shift3);
    uint32_t tk = elemIdx - tkDiv * size3;
    uint32_t tzDiv = UintDivImpl<uint32_t>(tkDiv, m2, shift2);
    uint32_t tz = tkDiv - tzDiv * size2;
    uint32_t tyDiv = UintDivImpl<uint32_t>(tzDiv, m1, shift1);
    uint32_t ty = tzDiv - tyDiv * size1;
    uint32_t tx = tyDiv;
    uint32_t dim0Index = tx + startOffset0;
    uint32_t dim1Index = ty + startOffset1;
    uint32_t dim2Index = tz + startOffset2;
    uint32_t dim3Index = tk + startOffset3;
    if (dim0Index >= endOffset0 || dim1Index >= endOffset1 ||
        dim2Index >= endOffset2 || dim3Index >= endOffset3) {
      break;
    }
    uint32_t index = tx * stride0 + ty * stride1 + tz * stride2 + tk * stride3;
    uint32_t dstIdx =
        tx * dstStride0 + ty * dstStride1 + tz * dstStride2 + tk * dstStride3;
    ITYPE indexVal =
        indices[index] < 0 ? indices[index] + indexBoundary : indices[index];
    if (indexVal < 0 || indexVal >= indexBoundary) {
      break;
    } else if (dimension == 0) {
      dst[dstIdx] = src[indexVal * xStrideIndex0 + dim1Index * xStrideIndex1 +
                        dim2Index * xStrideIndex2 + dim3Index];
    } else if (dimension == 1) {
      dst[dstIdx] = src[dim0Index * xStrideIndex0 + indexVal * xStrideIndex1 +
                        dim2Index * xStrideIndex2 + dim3Index];
    } else if (dimension == 2) {
      dst[dstIdx] = src[dim0Index * xStrideIndex0 + dim1Index * xStrideIndex1 +
                        indexVal * xStrideIndex2 + dim3Index];
    } else if (dimension == 3) {
      dst[dstIdx] = src[dim0Index * xStrideIndex0 + dim1Index * xStrideIndex1 +
                        dim2Index * xStrideIndex2 + indexVal];
    }
  }
}

template <typename DTYPE, typename ITYPE>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM_1024) __aiv__
    __attribute__((always_inline)) static void SimtGather5D(
        __gm__ DTYPE *src, __ubuf__ ITYPE *indices, __ubuf__ DTYPE *dst,
        const uint32_t size0, const uint32_t stride0, const uint32_t size1,
        const uint32_t stride1, const uint32_t size2, const uint32_t stride2,
        const uint32_t size3, const uint32_t stride3, const uint32_t size4,
        const uint32_t stride4, const uint32_t shift1, const uint32_t m1,
        const uint32_t shift2, const uint32_t m2, const uint32_t shift3,
        const uint32_t m3, const uint32_t shift4, const uint32_t m4,
        const uint32_t dstStride0, const uint32_t dstStride1,
        const uint32_t dstStride2, const uint32_t dstStride3,
        const uint32_t dstStride4, const uint32_t xStrideIndex0,
        const uint32_t xStrideIndex1, const uint32_t xStrideIndex2,
        const uint32_t xStrideIndex3, const uint32_t endOffset0,
        const uint32_t endOffset1, const uint32_t endOffset2,
        const uint32_t endOffset3, const uint32_t endOffset4,
        const uint32_t startOffset0, const uint32_t startOffset1,
        const uint32_t startOffset2, const uint32_t startOffset3,
        const uint32_t startOffset4, const uint32_t dimension,
        const uint32_t indexBoundary) {
  const uint32_t numElems = size0 * size1 * size2 * size3 * size4;
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
    uint32_t dim0Index = tx + startOffset0;
    uint32_t dim1Index = ty + startOffset1;
    uint32_t dim2Index = tz + startOffset2;
    uint32_t dim3Index = tk + startOffset3;
    uint32_t dim4Index = tl + startOffset4;
    if (dim0Index >= endOffset0 || dim1Index >= endOffset1 ||
        dim2Index >= endOffset2 || dim3Index >= endOffset3 ||
        dim4Index >= endOffset4) {
      break;
    }
    uint32_t index = tx * stride0 + ty * stride1 + tz * stride2 + tk * stride3 +
                     tl * stride4;
    uint32_t dstIdx = tx * dstStride0 + ty * dstStride1 + tz * dstStride2 +
                      tk * dstStride3 + tl * dstStride4;
    ITYPE indexVal =
        indices[index] < 0 ? indices[index] + indexBoundary : indices[index];
    if (indexVal < 0 || indexVal >= indexBoundary) {
      break;
    } else if (dimension == 0) {
      dst[dstIdx] = src[indexVal * xStrideIndex0 + dim1Index * xStrideIndex1 +
                        dim2Index * xStrideIndex2 + dim3Index * xStrideIndex3 +
                        dim4Index];
    } else if (dimension == 1) {
      dst[dstIdx] = src[dim0Index * xStrideIndex0 + indexVal * xStrideIndex1 +
                        dim2Index * xStrideIndex2 + dim3Index * xStrideIndex3 +
                        dim4Index];
    } else if (dimension == 2) {
      dst[dstIdx] =
          src[dim0Index * xStrideIndex0 + dim1Index * xStrideIndex1 +
              indexVal * xStrideIndex2 + dim3Index * xStrideIndex3 + dim4Index];
    } else if (dimension == 3) {
      dst[dstIdx] =
          src[dim0Index * xStrideIndex0 + dim1Index * xStrideIndex1 +
              dim2Index * xStrideIndex2 + indexVal * xStrideIndex3 + dim4Index];
    } else if (dimension == 4) {
      dst[dstIdx] =
          src[dim0Index * xStrideIndex0 + dim1Index * xStrideIndex1 +
              dim2Index * xStrideIndex2 + dim3Index * xStrideIndex3 + indexVal];
    }
  }
}

template <int DIM, typename DTYPE, typename ITYPE>
__aiv__ __attribute__((always_inline)) void
GatherOutToUB(memref_t<__gm__ DTYPE, 1> *src,
              memref_t<__ubuf__ ITYPE, DIM> *indices,
              memref_t<__ubuf__ DTYPE, DIM> *dst, int64_t *srcStride,
              int32_t *endOffset, int32_t *startOffset, const int32_t dimension,
              const int64_t indexBoundary) {
  static_assert(DIM <= 5, "DIM must be <= 5");
  static_assert(std::is_same<DTYPE, float>::value ||
                    std::is_same<DTYPE, float16_t>::value ||
                    std::is_same<DTYPE, bfloat16_t>::value ||
                    std::is_same<DTYPE, half>::value,
                "Currently dtype of data supports only \
                  float|float16_t|bfloat16_t|half");
  static_assert(std::is_same<ITYPE, int32_t>::value ||
                    std::is_same<ITYPE, int64_t>::value,
                "Currently dtype of index supports only int32|int64.");
  unsigned int block_dim_x = 1;
  unsigned int block_dim_y = 1;
  unsigned int block_dim_z = 1;
  if constexpr (DIM == 1) {
    const uint32_t endOffset0 = endOffset[0];
    const uint32_t startOffset0 = startOffset[0];
    const uint32_t indicesSize0 = indices->sizes[0];
    const uint32_t indicesStride0 = indices->strides[0];
    const uint32_t dstStride0 = dst->strides[0];
    cce::async_invoke<SimtGather1D<DTYPE, ITYPE>>(
        cce::dim3{THREAD_DIM_1024},
        reinterpret_cast<__gm__ DTYPE *>(src->aligned),
        reinterpret_cast<__ubuf__ ITYPE *>(indices->aligned + indices->offset),
        reinterpret_cast<__ubuf__ DTYPE *>(dst->aligned + dst->offset),
        indexBoundary, endOffset0, indicesSize0, indicesStride0, dstStride0,
        startOffset0);
  } else if constexpr (DIM == 2) {
    const uint32_t xStrideIndex0 = srcStride[0];
    const uint32_t size0 = indices->sizes[0];
    const uint32_t size1 = indices->sizes[1];
    const uint32_t stride0 = indices->strides[0];
    const uint32_t stride1 = indices->strides[1];
    const uint32_t dstStride0 = dst->strides[0];
    const uint32_t dstStride1 = dst->strides[1];
    const uint32_t endOffset0 = endOffset[0];
    const uint32_t endOffset1 = endOffset[1];
    const uint32_t startOffset0 = startOffset[0];
    const uint32_t startOffset1 = startOffset[1];
    const uint32_t numElems = size0 * size1;
    uint32_t shift1 = 0;
    uint32_t m1 = 0;
    GetUintDivMagicAndShiftImpl<uint32_t>(m1, shift1, size1);
    cce::async_invoke<SimtGather2D<DTYPE, ITYPE>>(
        cce::dim3{THREAD_DIM_1024},
        reinterpret_cast<__gm__ DTYPE *>(src->aligned),
        reinterpret_cast<__ubuf__ ITYPE *>(indices->aligned + indices->offset),
        reinterpret_cast<__ubuf__ DTYPE *>(dst->aligned + dst->offset),
        indexBoundary, xStrideIndex0, numElems, shift1, m1, stride0, size1,
        stride1, dstStride0, dstStride1, endOffset0, endOffset1, startOffset0,
        startOffset1, dimension);
  } else if constexpr (DIM == 3) {
    const uint32_t xStrideIndex0 = srcStride[0];
    const uint32_t xStrideIndex1 = srcStride[1];
    const uint32_t size0 = indices->sizes[0];
    const uint32_t size1 = indices->sizes[1];
    const uint32_t size2 = indices->sizes[2];
    const uint32_t stride0 = indices->strides[0];
    const uint32_t stride1 = indices->strides[1];
    const uint32_t stride2 = indices->strides[2];
    const uint32_t dstStride0 = dst->strides[0];
    const uint32_t dstStride1 = dst->strides[1];
    const uint32_t dstStride2 = dst->strides[2];
    const uint32_t endOffset0 = endOffset[0];
    const uint32_t endOffset1 = endOffset[1];
    const uint32_t endOffset2 = endOffset[2];
    const uint32_t startOffset0 = startOffset[0];
    const uint32_t startOffset1 = startOffset[1];
    const uint32_t startOffset2 = startOffset[2];
    const uint32_t numElems = size0 * size1 * size2;
    uint32_t shift1 = 0;
    uint32_t m1 = 0;
    GetUintDivMagicAndShiftImpl<uint32_t>(m1, shift1, size1);
    uint32_t shift2 = 0;
    uint32_t m2 = 0;
    GetUintDivMagicAndShiftImpl<uint32_t>(m2, shift2, size2);
    cce::async_invoke<SimtGather3D<DTYPE, ITYPE>>(
        cce::dim3{THREAD_DIM_1024},
        reinterpret_cast<__gm__ DTYPE *>(src->aligned),
        reinterpret_cast<__ubuf__ ITYPE *>(indices->aligned + indices->offset),
        reinterpret_cast<__ubuf__ DTYPE *>(dst->aligned + dst->offset),
        indexBoundary, xStrideIndex0, xStrideIndex1, numElems, shift1, m1,
        shift2, m2, size1, size2, stride0, stride1, stride2, dstStride0,
        dstStride1, dstStride2, endOffset0, endOffset1, endOffset2,
        startOffset0, startOffset1, startOffset2, dimension);
  } else if constexpr (DIM == 4) {
    const uint32_t size0 = indices->sizes[0];
    const uint32_t size1 = indices->sizes[1];
    const uint32_t size2 = indices->sizes[2];
    const uint32_t size3 = indices->sizes[3];
    const uint32_t stride0 = indices->strides[0];
    const uint32_t stride1 = indices->strides[1];
    const uint32_t stride2 = indices->strides[2];
    const uint32_t stride3 = indices->strides[3];
    const uint32_t dstStride0 = dst->strides[0];
    const uint32_t dstStride1 = dst->strides[1];
    const uint32_t dstStride2 = dst->strides[2];
    const uint32_t dstStride3 = dst->strides[3];

    const uint32_t xStrideIndex0 = srcStride[0];
    const uint32_t xStrideIndex1 = srcStride[1];
    const uint32_t xStrideIndex2 = srcStride[2];

    const uint32_t endOffset0 = endOffset[0];
    const uint32_t endOffset1 = endOffset[1];
    const uint32_t endOffset2 = endOffset[2];
    const uint32_t endOffset3 = endOffset[3];

    const uint32_t startOffset0 = startOffset[0];
    const uint32_t startOffset1 = startOffset[1];
    const uint32_t startOffset2 = startOffset[2];
    const uint32_t startOffset3 = startOffset[3];
    uint32_t shift1 = 0;
    uint32_t m1 = 0;
    GetUintDivMagicAndShiftImpl<uint32_t>(m1, shift1, size1);
    uint32_t shift2 = 0;
    uint32_t m2 = 0;
    GetUintDivMagicAndShiftImpl<uint32_t>(m2, shift2, size2);
    uint32_t shift3 = 0;
    uint32_t m3 = 0;
    GetUintDivMagicAndShiftImpl<uint32_t>(m3, shift3, size3);
    cce::async_invoke<SimtGather4D<DTYPE, ITYPE>>(
        cce::dim3{THREAD_DIM_1024},
        reinterpret_cast<__gm__ DTYPE *>(src->aligned),
        reinterpret_cast<__ubuf__ ITYPE *>(indices->aligned + indices->offset),
        reinterpret_cast<__ubuf__ DTYPE *>(dst->aligned + dst->offset), size0,
        stride0, size1, stride1, size2, stride2, size3, stride3, shift1, m1,
        shift2, m2, shift3, m3, dstStride0, dstStride1, dstStride2, dstStride3,
        xStrideIndex0, xStrideIndex1, xStrideIndex2, endOffset0, endOffset1,
        endOffset2, endOffset3, startOffset0, startOffset1, startOffset2,
        startOffset3, dimension, indexBoundary);
  } else if constexpr (DIM == 5) {
    const uint32_t stride0 = indices->strides[0];
    const uint32_t stride1 = indices->strides[1];
    const uint32_t stride2 = indices->strides[2];
    const uint32_t stride3 = indices->strides[3];
    const uint32_t stride4 = indices->strides[4];
    const uint32_t size0 = indices->sizes[0];
    const uint32_t size1 = indices->sizes[1];
    const uint32_t size2 = indices->sizes[2];
    const uint32_t size3 = indices->sizes[3];
    const uint32_t size4 = indices->sizes[4];
    const uint32_t dstStride0 = dst->strides[0];
    const uint32_t dstStride1 = dst->strides[1];
    const uint32_t dstStride2 = dst->strides[2];
    const uint32_t dstStride3 = dst->strides[3];
    const uint32_t dstStride4 = dst->strides[4];

    const uint32_t xStrideIndex0 = srcStride[0];
    const uint32_t xStrideIndex1 = srcStride[1];
    const uint32_t xStrideIndex2 = srcStride[2];
    const uint32_t xStrideIndex3 = srcStride[3];

    const uint32_t endOffset0 = endOffset[0];
    const uint32_t endOffset1 = endOffset[1];
    const uint32_t endOffset2 = endOffset[2];
    const uint32_t endOffset3 = endOffset[3];
    const uint32_t endOffset4 = endOffset[4];

    const uint32_t startOffset0 = startOffset[0];
    const uint32_t startOffset1 = startOffset[1];
    const uint32_t startOffset2 = startOffset[2];
    const uint32_t startOffset3 = startOffset[3];
    const uint32_t startOffset4 = startOffset[4];
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
    cce::async_invoke<SimtGather5D<DTYPE, ITYPE>>(
        cce::dim3{THREAD_DIM_1024},
        reinterpret_cast<__gm__ DTYPE *>(src->aligned),
        reinterpret_cast<__ubuf__ ITYPE *>(indices->aligned + indices->offset),
        reinterpret_cast<__ubuf__ DTYPE *>(dst->aligned + dst->offset), size0,
        stride0, size1, stride1, size2, stride2, size3, stride3, size4, stride4,
        shift1, m1, shift2, m2, shift3, m3, shift4, m4, dstStride0, dstStride1,
        dstStride2, dstStride3, dstStride4, xStrideIndex0, xStrideIndex1,
        xStrideIndex2, xStrideIndex3, endOffset0, endOffset1, endOffset2,
        endOffset3, endOffset4, startOffset0, startOffset1, startOffset2,
        startOffset3, startOffset4, dimension, indexBoundary);
  }
}

extern "C" {
REGISTER_GATHER_OUT_TO_UB_1D(1, float, int32_t);
REGISTER_GATHER_OUT_TO_UB_1D(1, float16_t, int32_t);
REGISTER_GATHER_OUT_TO_UB_1D(1, bfloat16_t, int32_t);
REGISTER_GATHER_OUT_TO_UB_1D(1, half, int32_t);

REGISTER_GATHER_OUT_TO_UB_1D(1, float, int64_t);
REGISTER_GATHER_OUT_TO_UB_1D(1, float16_t, int64_t);
REGISTER_GATHER_OUT_TO_UB_1D(1, bfloat16_t, int64_t);
REGISTER_GATHER_OUT_TO_UB_1D(1, half, int64_t);

REGISTER_GATHER_OUT_TO_UB_2D(2, float, int32_t);
REGISTER_GATHER_OUT_TO_UB_2D(2, float16_t, int32_t);
REGISTER_GATHER_OUT_TO_UB_2D(2, bfloat16_t, int32_t);
REGISTER_GATHER_OUT_TO_UB_2D(2, half, int32_t);

REGISTER_GATHER_OUT_TO_UB_2D(2, float, int64_t);
REGISTER_GATHER_OUT_TO_UB_2D(2, float16_t, int64_t);
REGISTER_GATHER_OUT_TO_UB_2D(2, bfloat16_t, int64_t);
REGISTER_GATHER_OUT_TO_UB_2D(2, half, int64_t);

REGISTER_GATHER_OUT_TO_UB_3D(3, float, int32_t);
REGISTER_GATHER_OUT_TO_UB_3D(3, float16_t, int32_t);
REGISTER_GATHER_OUT_TO_UB_3D(3, bfloat16_t, int32_t);
REGISTER_GATHER_OUT_TO_UB_3D(3, half, int32_t);

REGISTER_GATHER_OUT_TO_UB_3D(3, float, int64_t);
REGISTER_GATHER_OUT_TO_UB_3D(3, float16_t, int64_t);
REGISTER_GATHER_OUT_TO_UB_3D(3, bfloat16_t, int64_t);
REGISTER_GATHER_OUT_TO_UB_3D(3, half, int64_t);

REGISTER_GATHER_OUT_TO_UB_4D(4, float, int32_t);
REGISTER_GATHER_OUT_TO_UB_4D(4, float16_t, int32_t);
REGISTER_GATHER_OUT_TO_UB_4D(4, bfloat16_t, int32_t);
REGISTER_GATHER_OUT_TO_UB_4D(4, half, int32_t);

REGISTER_GATHER_OUT_TO_UB_4D(4, float, int64_t);
REGISTER_GATHER_OUT_TO_UB_4D(4, float16_t, int64_t);
REGISTER_GATHER_OUT_TO_UB_4D(4, bfloat16_t, int64_t);
REGISTER_GATHER_OUT_TO_UB_4D(4, half, int64_t);

REGISTER_GATHER_OUT_TO_UB_5D(5, float, int32_t);
REGISTER_GATHER_OUT_TO_UB_5D(5, float16_t, int32_t);
REGISTER_GATHER_OUT_TO_UB_5D(5, bfloat16_t, int32_t);
REGISTER_GATHER_OUT_TO_UB_5D(5, half, int32_t);

REGISTER_GATHER_OUT_TO_UB_5D(5, float, int64_t);
REGISTER_GATHER_OUT_TO_UB_5D(5, float16_t, int64_t);
REGISTER_GATHER_OUT_TO_UB_5D(5, bfloat16_t, int64_t);
REGISTER_GATHER_OUT_TO_UB_5D(5, half, int64_t);
}
#endif
