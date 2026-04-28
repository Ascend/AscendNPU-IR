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

#include "RegBase/VecUtils.h"

constexpr uint32_t MAX_THREAD_NUM = 1024;

template <typename DTYPE, typename ITYPE>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM) __aiv__
    __attribute__((always_inline)) static void SimtScatter1D(
        __gm__ DTYPE *dst, __ubuf__ DTYPE *src, __ubuf__ ITYPE *indices,
        const uint32_t indexBoundary, const uint32_t dstStride,
        const uint32_t srcStride, const uint32_t size0, const uint32_t stride0,
        const uint32_t endOffset0, const uint32_t startOffset0) {
  for (uint32_t tx = threadIdx.x; tx < size0; tx += blockDim.x) {
    uint32_t globalOffset = tx + startOffset0;
    if (globalOffset >= endOffset0) {
      break;
    }
    uint32_t index = tx * stride0;
    ITYPE indexVal = indices[index];
    if (indexVal < 0) {
      indexVal = indexVal + indexBoundary;
    }
    if (indexVal < 0 || indexVal >= indexBoundary) {
      break;
    }
    uint32_t srcIdx = tx * srcStride;
    dst[indexVal * dstStride] = src[srcIdx];
  }
}

template <typename DTYPE, typename ITYPE>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM) __aiv__
    __attribute__((always_inline)) static void SimtScatter2D(
        __gm__ DTYPE *dst, __ubuf__ DTYPE *src, __ubuf__ ITYPE *indices,
        const uint32_t dstStride0, const uint32_t srcStride0,
        const uint32_t srcStride1, const uint32_t size0, const uint32_t size1,
        const uint32_t stride0, const uint32_t stride1, const uint32_t shift1,
        const uint32_t m1, const uint32_t endOffset0, const uint32_t endOffset1,
        const uint32_t startOffset0, const uint32_t startOffset1,
        const uint32_t indexBoundary, const uint32_t dim) {
  // assume the index shape same with src
  const uint32_t numElem = size0 * size1;
  for (uint32_t tx = threadIdx.x; tx < numElem; tx += blockDim.x) {
    uint32_t i1Div = UintDivImpl<uint32_t>(tx, m1, shift1);
    uint32_t i1 = tx - i1Div * size1;
    uint32_t i0 = i1Div;
    uint32_t dim0Index = i0 + startOffset0;
    uint32_t dim1Index = i1 + startOffset1;
    if (dim0Index >= endOffset0 || dim1Index >= endOffset1) {
      break;
    }
    uint32_t index = i0 * stride0 + i1 * stride1;        // 36 1
    uint32_t srcIdx = i0 * srcStride0 + i1 * srcStride1; // 40 1
    ITYPE indexVal = static_cast<ITYPE>(indices[index]);
    if (indexVal < 0) {
      indexVal += indexBoundary;
    }
    if (indexVal < 0 || indexVal >= indexBoundary) {
      break;
    } else if (dim == 1) {
      dst[dim0Index * dstStride0 + indexVal] = src[srcIdx];
    } else if (dim == 0) {
      dst[indexVal * dstStride0 + dim1Index] = src[srcIdx];
    }
  }
}

template <typename DTYPE, typename ITYPE>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM) __aiv__
    __attribute__((always_inline)) static void SimtScatter3D(
        __gm__ DTYPE *dst, __ubuf__ DTYPE *src, __ubuf__ ITYPE *indices,
        const uint32_t dstStride0, const uint32_t dstStride1,
        const uint32_t srcStride0, const uint32_t srcStride1,
        const uint32_t srcStride2, const uint32_t size0, const uint32_t size1,
        const uint32_t size2, const uint32_t stride0, const uint32_t stride1,
        const uint32_t stride2, const uint32_t shift1, const uint32_t m1,
        const uint32_t shift2, const uint32_t m2, const uint32_t endOffset0,
        const uint32_t endOffset1, const uint32_t endOffset2,
        const uint32_t startOffset0, const uint32_t startOffset1,
        const uint32_t startOffset2, const uint32_t indexBoundary,
        const uint32_t dim) {
  const uint32_t numElem = size0 * size1 * size2;
  for (uint32_t tx = threadIdx.x; tx < numElem; tx += blockDim.x) {
    uint32_t i2Div = UintDivImpl<uint32_t>(tx, m2, shift2);
    uint32_t i2 = tx - i2Div * size2;
    uint32_t i1Div = UintDivImpl<uint32_t>(i2Div, m1, shift1);
    uint32_t i1 = i2Div - i1Div * size1;
    uint32_t i0 = i1Div;
    uint32_t dim0Index = i0 + startOffset0;
    uint32_t dim1Index = i1 + startOffset1;
    uint32_t dim2Index = i2 + startOffset2;
    if (dim0Index >= endOffset0 || dim1Index >= endOffset1 ||
        dim2Index >= endOffset2) {
      break;
    }
    uint32_t index = i0 * stride0 + i1 * stride1 + i2 * stride2;
    uint32_t srcIdx = i0 * srcStride0 + i1 * srcStride1 + i2 * srcStride2;
    ITYPE indexVal =
        indices[index] < 0 ? indices[index] + indexBoundary : indices[index];
    if (indexVal < 0 || indexVal >= indexBoundary) {
      break;
    } else if (dim == 0) {
      dst[indexVal * dstStride0 + dim1Index * dstStride1 + dim2Index] =
          src[srcIdx];
    } else if (dim == 1) {
      dst[dim0Index * dstStride0 + indexVal * dstStride1 + dim2Index] =
          src[srcIdx];
    } else if (dim == 2) {
      dst[dim0Index * dstStride0 + dim1Index * dstStride1 + indexVal] =
          src[srcIdx];
    }
  }
}

template <typename DTYPE, typename ITYPE>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM) __aiv__
    __attribute__((always_inline)) static void SimtScatter4D(
        __gm__ DTYPE *dst, __ubuf__ DTYPE *src, __ubuf__ ITYPE *indices,
        const uint32_t dstStride0, const uint32_t dstStride1,
        const uint32_t dstStride2, const uint32_t srcStride0,
        const uint32_t srcStride1, const uint32_t srcStride2,
        const uint32_t srcStride3, const uint32_t size0, const uint32_t size1,
        const uint32_t size2, const uint32_t size3, const uint32_t stride0,
        const uint32_t stride1, const uint32_t stride2, const uint32_t stride3,
        const uint32_t shift1, const uint32_t m1, const uint32_t shift2,
        const uint32_t m2, const uint32_t shift3, const uint32_t m3,
        const uint32_t endOffset0, const uint32_t endOffset1,
        const uint32_t endOffset2, const uint32_t endOffset3,
        const uint32_t startOffset0, const uint32_t startOffset1,
        const uint32_t startOffset2, const uint32_t startOffset3,
        const uint32_t indexBoundary, const uint32_t dim) {
  const uint32_t numElems = size0 * size1 * size2 * size3;
  for (uint32_t tx = threadIdx.x; tx < numElems; tx += blockDim.x) {
    uint32_t i3Div = UintDivImpl<uint32_t>(tx, m3, shift3);
    uint32_t i3 = tx - i3Div * size3;
    uint32_t i2Div = UintDivImpl<uint32_t>(i3Div, m2, shift2);
    uint32_t i2 = i3Div - i2Div * size2;
    uint32_t i1Div = UintDivImpl<uint32_t>(i2Div, m1, shift1);
    uint32_t i1 = i2Div - i1Div * size1;
    uint32_t i0 = i1Div;
    uint32_t dim0Index = i0 + startOffset0;
    uint32_t dim1Index = i1 + startOffset1;
    uint32_t dim2Index = i2 + startOffset2;
    uint32_t dim3Index = i3 + startOffset3;
    if (dim0Index >= endOffset0 || dim1Index >= endOffset1 ||
        dim2Index >= endOffset2 || dim3Index >= endOffset3) {
      break;
    }
    uint32_t index = i0 * stride0 + i1 * stride1 + i2 * stride2 + i3 * stride3;
    uint32_t srcIdx =
        i0 * srcStride0 + i1 * srcStride1 + i2 * srcStride2 + i3 * srcStride3;
    ITYPE indexVal =
        indices[index] < 0 ? indices[index] + indexBoundary : indices[index];

    if (indexVal < 0 || indexVal >= indexBoundary) {
      break;
    } else if (dim == 0) {
      dst[indexVal * dstStride0 + dim1Index * dstStride1 +
          dim2Index * dstStride2 + dim3Index] = src[srcIdx];
    } else if (dim == 1) {
      dst[dim0Index * dstStride0 + indexVal * dstStride1 +
          dim2Index * dstStride2 + dim3Index] = src[srcIdx];
    } else if (dim == 2) {
      dst[dim0Index * dstStride0 + dim1Index * dstStride1 +
          indexVal * dstStride2 + dim3Index] = src[srcIdx];
    } else if (dim == 3) {
      dst[dim0Index * dstStride0 + dim1Index * dstStride1 +
          dim2Index * dstStride2 + indexVal] = src[srcIdx];
    }
  }
}

template <typename DTYPE, typename ITYPE>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM) __aiv__
    __attribute__((always_inline)) static void SimtScatter5D(
        __gm__ DTYPE *dst, __ubuf__ DTYPE *src, __ubuf__ ITYPE *indices,
        const uint32_t dstStride0, const uint32_t dstStride1,
        const uint32_t dstStride2, const uint32_t dstStride3,
        const uint32_t srcStride0, const uint32_t srcStride1,
        const uint32_t srcStride2, const uint32_t srcStride3,
        const uint32_t srcStride4, const uint32_t size0, const uint32_t size1,
        const uint32_t size2, const uint32_t size3, const uint32_t size4,
        const uint32_t stride0, const uint32_t stride1, const uint32_t stride2,
        const uint32_t stride3, const uint32_t stride4, const uint32_t shift1,
        const uint32_t m1, const uint32_t shift2, const uint32_t m2,
        const uint32_t shift3, const uint32_t m3, const uint32_t shift4,
        const uint32_t m4, const uint32_t endOffset0, const uint32_t endOffset1,
        const uint32_t endOffset2, const uint32_t endOffset3,
        const uint32_t endOffset4, const uint32_t startOffset0,
        const uint32_t startOffset1, const uint32_t startOffset2,
        const uint32_t startOffset3, const uint32_t startOffset4,
        const uint32_t indexBoundary, const uint32_t dim) {
  const uint32_t numElems = size0 * size1 * size2 * size3 * size4;
  for (uint32_t tx = threadIdx.x; tx < numElems; tx += blockDim.x) {
    uint32_t i4Div = UintDivImpl<uint32_t>(tx, m4, shift4);
    uint32_t i4 = tx - i4Div * size4;
    uint32_t i3Div = UintDivImpl<uint32_t>(i4Div, m3, shift3);
    uint32_t i3 = i4Div - i3Div * size3;
    uint32_t i2Div = UintDivImpl<uint32_t>(i3Div, m2, shift2);
    uint32_t i2 = i3Div - i2Div * size2;
    uint32_t i1Div = UintDivImpl<uint32_t>(i2Div, m1, shift1);
    uint32_t i1 = i2Div - i1Div * size1;
    uint32_t i0 = i1Div;
    uint32_t dim0Index = i0 + startOffset0;
    uint32_t dim1Index = i1 + startOffset1;
    uint32_t dim2Index = i2 + startOffset2;
    uint32_t dim3Index = i3 + startOffset3;
    uint32_t dim4Index = i4 + startOffset4;
    if (dim0Index >= endOffset0 || dim1Index >= endOffset1 ||
        dim2Index >= endOffset2 || dim3Index >= endOffset3 ||
        dim4Index >= endOffset4) {
      break;
    }
    uint32_t index = i0 * stride0 + i1 * stride1 + i2 * stride2 + i3 * stride3 +
                     i4 * stride4;
    uint32_t srcIdx = i0 * srcStride0 + i1 * srcStride1 + i2 * srcStride2 +
                      i3 * srcStride3 + i4 * srcStride4;
    ITYPE indexVal =
        indices[index] < 0 ? indices[index] + indexBoundary : indices[index];
    if (indexVal < 0 || indexVal >= indexBoundary) {
      break;
    } else if (dim == 0) {
      dst[indexVal * dstStride0 + dim1Index * dstStride1 +
          dim2Index * dstStride2 + dim3Index * dstStride3 + dim4Index] =
          src[srcIdx];
    } else if (dim == 1) {
      dst[dim0Index * dstStride0 + indexVal * dstStride1 +
          dim2Index * dstStride2 + dim3Index * dstStride3 + dim4Index] =
          src[srcIdx];
    } else if (dim == 2) {
      dst[dim0Index * dstStride0 + dim1Index * dstStride1 +
          indexVal * dstStride2 + dim3Index * dstStride3 + dim4Index] =
          src[srcIdx];
    } else if (dim == 3) {
      dst[dim0Index * dstStride0 + dim1Index * dstStride1 +
          dim2Index * dstStride2 + indexVal * dstStride3 + dim4Index] =
          src[srcIdx];
    } else if (dim == 4) {
      dst[dim0Index * dstStride0 + dim1Index * dstStride1 +
          dim2Index * dstStride2 + dim3Index * dstStride3 + indexVal] =
          src[srcIdx];
    }
  }
}

template <int DIM, typename DTYPE, typename ITYPE>
__aiv__ __attribute__((always_inline)) void SimtScatteUBToOut(
    memref_t<__gm__ DTYPE, 1> *dst, memref_t<__ubuf__ DTYPE, DIM> *src,
    memref_t<__ubuf__ ITYPE, DIM> *index, int64_t indexBoundary, int32_t dim,
    int64_t *dstStrides, int32_t *endOffsets, int32_t *startOffsets) {
  static_assert(DIM <= 5, "DIM must be <= 5");
  static_assert(std::is_same<DTYPE, float>::value ||
                    std::is_same<DTYPE, float16_t>::value ||
                    std::is_same<DTYPE, bfloat16_t>::value ||
                    std::is_same<DTYPE, half>::value,
                "Currently DTYPE of data supports only \
                  float|float16_t|bfloat16_t|half.");
  static_assert(std::is_same<ITYPE, int32_t>::value ||
                    std::is_same<ITYPE, int64_t>::value,
                "Currently DTYPE of indices supports only int32|int64.");
  unsigned int block_dim_x = 1;
  unsigned int block_dim_y = 1;
  unsigned int block_dim_z = 1;
  if constexpr (DIM == 1) {
    const uint32_t size0 = index->sizes[0];
    const uint32_t stride0 = index->strides[0];
    const uint32_t srcStride0 = src->strides[0];
    const uint32_t dstStride0 = dstStrides[0];
    const uint32_t endOffset0 = endOffsets[0];
    const uint32_t startOffset0 = startOffsets[0];
    cce::async_invoke<SimtScatter1D<DTYPE, ITYPE>>(
        cce::dim3{MAX_THREAD_NUM},
        reinterpret_cast<__gm__ DTYPE *>(dst->aligned),
        reinterpret_cast<__ubuf__ DTYPE *>(src->aligned),
        reinterpret_cast<__ubuf__ ITYPE *>(index->aligned), indexBoundary,
        dstStride0, srcStride0, size0, stride0, endOffset0, startOffset0);
  } else if constexpr (DIM == 2) {
    const uint32_t size0 = index->sizes[0];
    const uint32_t size1 = index->sizes[1];
    const uint32_t stride0 = index->strides[0];
    const uint32_t stride1 = index->strides[1];
    const uint32_t srcStride0 = src->strides[0];
    const uint32_t srcStride1 = src->strides[1];

    const uint32_t dstStride0 = dstStrides[0];
    const uint32_t dstStride1 = dstStrides[1];
    const uint32_t endOffset0 = endOffsets[0];
    const uint32_t endOffset1 = endOffsets[1];
    const uint32_t startOffset0 = startOffsets[0];
    const uint32_t startOffset1 = startOffsets[1];
    uint32_t shift1 = 0;
    uint32_t m1 = 0;
    GetUintDivMagicAndShiftImpl<uint32_t>(m1, shift1, size1);
    cce::async_invoke<SimtScatter2D<DTYPE, ITYPE>>(
        cce::dim3{MAX_THREAD_NUM},
        reinterpret_cast<__gm__ DTYPE *>(dst->aligned),
        reinterpret_cast<__ubuf__ DTYPE *>(src->aligned),
        reinterpret_cast<__ubuf__ ITYPE *>(index->aligned), dstStride0,
        srcStride0, srcStride1, size0, size1, stride0, stride1, shift1, m1,
        endOffset0, endOffset1, startOffset0, startOffset1, indexBoundary, dim);
  } else if constexpr (DIM == 3) {
    const uint32_t size0 = index->sizes[0];
    const uint32_t size1 = index->sizes[1];
    const uint32_t size2 = index->sizes[2];
    const uint32_t stride0 = index->strides[0];
    const uint32_t stride1 = index->strides[1];
    const uint32_t stride2 = index->strides[2];
    const uint32_t srcStride0 = src->strides[0];
    const uint32_t srcStride1 = src->strides[1];
    const uint32_t srcStride2 = src->strides[2];

    const uint32_t dstStride0 = dstStrides[0];
    const uint32_t dstStride1 = dstStrides[1];
    const uint32_t dstStride2 = dstStrides[2];

    const uint32_t endOffset0 = endOffsets[0];
    const uint32_t endOffset1 = endOffsets[1];
    const uint32_t endOffset2 = endOffsets[2];

    const uint32_t startOffset0 = startOffsets[0];
    const uint32_t startOffset1 = startOffsets[1];
    const uint32_t startOffset2 = startOffsets[2];
    uint32_t shift1 = 0;
    uint32_t m1 = 0;
    GetUintDivMagicAndShiftImpl<uint32_t>(m1, shift1, size1);
    uint32_t shift2 = 0;
    uint32_t m2 = 0;
    GetUintDivMagicAndShiftImpl<uint32_t>(m2, shift2, size2);
    cce::async_invoke<SimtScatter3D<DTYPE, ITYPE>>(
        cce::dim3{MAX_THREAD_NUM},
        reinterpret_cast<__gm__ DTYPE *>(dst->aligned),
        reinterpret_cast<__ubuf__ DTYPE *>(src->aligned),
        reinterpret_cast<__ubuf__ ITYPE *>(index->aligned), dstStride0,
        dstStride1, srcStride0, srcStride1, srcStride2, size0, size1, size2,
        stride0, stride1, stride2, shift1, m1, shift2, m2, endOffset0,
        endOffset1, endOffset2, startOffset0, startOffset1, startOffset2,
        indexBoundary, dim);
  } else if constexpr (DIM == 4) {
    const uint32_t size0 = index->sizes[0];
    const uint32_t size1 = index->sizes[1];
    const uint32_t size2 = index->sizes[2];
    const uint32_t size3 = index->sizes[3];

    const uint32_t stride0 = index->strides[0];
    const uint32_t stride1 = index->strides[1];
    const uint32_t stride2 = index->strides[2];
    const uint32_t stride3 = index->strides[3];

    const uint32_t srcStride0 = src->strides[0];
    const uint32_t srcStride1 = src->strides[1];
    const uint32_t srcStride2 = src->strides[2];
    const uint32_t srcStride3 = src->strides[3];

    const uint32_t dstStride0 = dstStrides[0];
    const uint32_t dstStride1 = dstStrides[1];
    const uint32_t dstStride2 = dstStrides[2];
    const uint32_t dstStride3 = dstStrides[3];

    const uint32_t endOffset0 = endOffsets[0];
    const uint32_t endOffset1 = endOffsets[1];
    const uint32_t endOffset2 = endOffsets[2];
    const uint32_t endOffset3 = endOffsets[3];

    const uint32_t startOffset0 = startOffsets[0];
    const uint32_t startOffset1 = startOffsets[1];
    const uint32_t startOffset2 = startOffsets[2];
    const uint32_t startOffset3 = startOffsets[3];
    uint32_t shift1 = 0;
    uint32_t m1 = 0;
    GetUintDivMagicAndShiftImpl<uint32_t>(m1, shift1, size1);
    uint32_t shift2 = 0;
    uint32_t m2 = 0;
    GetUintDivMagicAndShiftImpl<uint32_t>(m2, shift2, size2);
    uint32_t shift3 = 0;
    uint32_t m3 = 0;
    GetUintDivMagicAndShiftImpl<uint32_t>(m2, shift3, size3);
    cce::async_invoke<SimtScatter4D<DTYPE, ITYPE>>(
        cce::dim3{MAX_THREAD_NUM},
        reinterpret_cast<__gm__ DTYPE *>(dst->aligned),
        reinterpret_cast<__ubuf__ DTYPE *>(src->aligned + src->offset),
        reinterpret_cast<__ubuf__ ITYPE *>(index->aligned + index->offset),
        dstStride0, dstStride1, dstStride2, srcStride0, srcStride1, srcStride2,
        srcStride3, size0, size1, size2, size3, stride0, stride1, stride2,
        stride3, shift1, m1, shift2, m2, shift3, m3, endOffset0, endOffset1,
        endOffset2, endOffset3, startOffset0, startOffset1, startOffset2,
        startOffset3, indexBoundary, dim);
  } else if constexpr (DIM == 5) {
    const uint32_t size0 = index->sizes[0];
    const uint32_t size1 = index->sizes[1];
    const uint32_t size2 = index->sizes[2];
    const uint32_t size3 = index->sizes[3];
    const uint32_t size4 = index->sizes[4];

    const uint32_t stride0 = index->strides[0];
    const uint32_t stride1 = index->strides[1];
    const uint32_t stride2 = index->strides[2];
    const uint32_t stride3 = index->strides[3];
    const uint32_t stride4 = index->strides[4];

    const uint32_t srcStride0 = src->strides[0];
    const uint32_t srcStride1 = src->strides[1];
    const uint32_t srcStride2 = src->strides[2];
    const uint32_t srcStride3 = src->strides[3];
    const uint32_t srcStride4 = src->strides[4];

    const uint32_t dstStride0 = dstStrides[0];
    const uint32_t dstStride1 = dstStrides[1];
    const uint32_t dstStride2 = dstStrides[2];
    const uint32_t dstStride3 = dstStrides[3];
    const uint32_t dstStride4 = dstStrides[4];

    const uint32_t endOffset0 = endOffsets[0];
    const uint32_t endOffset1 = endOffsets[1];
    const uint32_t endOffset2 = endOffsets[2];
    const uint32_t endOffset3 = endOffsets[3];
    const uint32_t endOffset4 = endOffsets[4];

    const uint32_t startOffset0 = startOffsets[0];
    const uint32_t startOffset1 = startOffsets[1];
    const uint32_t startOffset2 = startOffsets[2];
    const uint32_t startOffset3 = startOffsets[3];
    const uint32_t startOffset4 = startOffsets[4];
    uint32_t shift1 = 0;
    uint32_t m1 = 0;
    GetUintDivMagicAndShiftImpl<uint32_t>(m1, shift1, size1);
    uint32_t shift2 = 0;
    uint32_t m2 = 0;
    GetUintDivMagicAndShiftImpl<uint32_t>(m2, shift2, size2);
    uint32_t shift3 = 0;
    uint32_t m3 = 0;
    GetUintDivMagicAndShiftImpl<uint32_t>(m2, shift3, size3);
    uint32_t shift4 = 0;
    uint32_t m4 = 0;
    GetUintDivMagicAndShiftImpl<uint32_t>(m3, shift4, size4);
    cce::async_invoke<SimtScatter5D<DTYPE, ITYPE>>(
        cce::dim3{MAX_THREAD_NUM},
        reinterpret_cast<__gm__ DTYPE *>(dst->aligned),
        reinterpret_cast<__ubuf__ DTYPE *>(src->aligned + src->offset),
        reinterpret_cast<__ubuf__ ITYPE *>(index->aligned + index->offset),
        dstStride0, dstStride1, dstStride2, dstStride3, srcStride0, srcStride1,
        srcStride2, srcStride3, srcStride4, size0, size1, size2, size3, size4,
        stride0, stride1, stride2, stride3, stride4, shift1, m1, shift2, m2,
        shift3, m3, shift4, m4, endOffset0, endOffset1, endOffset2, endOffset3,
        endOffset4, startOffset0, startOffset1, startOffset2, startOffset3,
        startOffset4, indexBoundary, dim);
  }
}

extern "C" {
REGISTER_SCATTER_UB_TO_OUT_1D(float, int32_t);
REGISTER_SCATTER_UB_TO_OUT_1D(float16_t, int32_t);
REGISTER_SCATTER_UB_TO_OUT_1D(bfloat16_t, int32_t);
REGISTER_SCATTER_UB_TO_OUT_1D(half, int32_t);

REGISTER_SCATTER_UB_TO_OUT_1D(float, int64_t);
REGISTER_SCATTER_UB_TO_OUT_1D(float16_t, int64_t);
REGISTER_SCATTER_UB_TO_OUT_1D(bfloat16_t, int64_t);
REGISTER_SCATTER_UB_TO_OUT_1D(half, int64_t);

REGISTER_SCATTER_UB_TO_OUT_2D(float, int32_t);
REGISTER_SCATTER_UB_TO_OUT_2D(float16_t, int32_t);
REGISTER_SCATTER_UB_TO_OUT_2D(bfloat16_t, int32_t);
REGISTER_SCATTER_UB_TO_OUT_2D(half, int32_t);

REGISTER_SCATTER_UB_TO_OUT_2D(float, int64_t);
REGISTER_SCATTER_UB_TO_OUT_2D(float16_t, int64_t);
REGISTER_SCATTER_UB_TO_OUT_2D(bfloat16_t, int64_t);
REGISTER_SCATTER_UB_TO_OUT_2D(half, int64_t);

REGISTER_SCATTER_UB_TO_OUT_3D(float, int32_t);
REGISTER_SCATTER_UB_TO_OUT_3D(float16_t, int32_t);
REGISTER_SCATTER_UB_TO_OUT_3D(bfloat16_t, int32_t);
REGISTER_SCATTER_UB_TO_OUT_3D(half, int32_t);

REGISTER_SCATTER_UB_TO_OUT_3D(float, int64_t);
REGISTER_SCATTER_UB_TO_OUT_3D(float16_t, int64_t);
REGISTER_SCATTER_UB_TO_OUT_3D(bfloat16_t, int64_t);
REGISTER_SCATTER_UB_TO_OUT_3D(half, int64_t);

REGISTER_SCATTER_UB_TO_OUT_4D(float, int32_t);
REGISTER_SCATTER_UB_TO_OUT_4D(float16_t, int32_t);
REGISTER_SCATTER_UB_TO_OUT_4D(bfloat16_t, int32_t);
REGISTER_SCATTER_UB_TO_OUT_4D(half, int32_t);

REGISTER_SCATTER_UB_TO_OUT_4D(float, int64_t);
REGISTER_SCATTER_UB_TO_OUT_4D(float16_t, int64_t);
REGISTER_SCATTER_UB_TO_OUT_4D(bfloat16_t, int64_t);
REGISTER_SCATTER_UB_TO_OUT_4D(half, int64_t);

REGISTER_SCATTER_UB_TO_OUT_5D(float, int32_t);
REGISTER_SCATTER_UB_TO_OUT_5D(float16_t, int32_t);
REGISTER_SCATTER_UB_TO_OUT_5D(bfloat16_t, int32_t);
REGISTER_SCATTER_UB_TO_OUT_5D(half, int32_t);

REGISTER_SCATTER_UB_TO_OUT_5D(float, int64_t);
REGISTER_SCATTER_UB_TO_OUT_5D(float16_t, int64_t);
REGISTER_SCATTER_UB_TO_OUT_5D(bfloat16_t, int64_t);
REGISTER_SCATTER_UB_TO_OUT_5D(half, int64_t);
}
#endif
