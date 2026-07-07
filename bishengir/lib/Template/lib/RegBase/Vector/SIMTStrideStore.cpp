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

#include "RegBase/VecUtils.h"

constexpr uint32_t STRIDE_STORE_THREAD_NUM = 1024;

template <typename DTYPE>
struct IsStrideStoreDTypeSupported {
  static constexpr bool value =
      std::is_same<DTYPE, float>::value ||
      std::is_same<DTYPE, half>::value ||
      std::is_same<DTYPE, bfloat16_t>::value ||
      std::is_same<DTYPE, int8_t>::value ||
      std::is_same<DTYPE, int16_t>::value ||
      std::is_same<DTYPE, int32_t>::value ||
      std::is_same<DTYPE, int64_t>::value ||
      std::is_same<DTYPE, uint8_t>::value ||
      std::is_same<DTYPE, uint16_t>::value ||
      std::is_same<DTYPE, uint32_t>::value ||
      std::is_same<DTYPE, uint64_t>::value;
};

__aiv__ __attribute__((always_inline)) static int32_t
getStrideStoreBoundedSize(int32_t stride, int32_t numel, int32_t srcSize) {
  if (stride <= 0 || numel <= 0 || srcSize <= 0) {
    return 0;
  }
  return numel < srcSize ? numel : srcSize;
}

__aiv__ __attribute__((always_inline)) static int32_t
getStrideStoreSize(int32_t stride, int32_t storeLower, int32_t storeUpper) {
  if (stride <= 0 || storeUpper <= storeLower) {
    return 0;
  }
  return (storeUpper - storeLower) / stride;
}

template <typename DTYPE>
__simt_vf__ LAUNCH_BOUND(STRIDE_STORE_THREAD_NUM) __aiv__
    __attribute__((always_inline)) static void
    simtStrideStore1D(__gm__ DTYPE *dst, __ubuf__ DTYPE *src,
                      const int32_t stride, const int32_t storeLower,
                      const int32_t storeSize, const int32_t srcStride0) {
  for (int32_t i0 = static_cast<int32_t>(threadIdx.x); i0 < storeSize;
       i0 += static_cast<int32_t>(blockDim.x)) {
    dst[storeLower + i0 * stride] = src[i0 * srcStride0];
  }
}

template <typename DTYPE>
__simt_vf__ LAUNCH_BOUND(STRIDE_STORE_THREAD_NUM) __aiv__
    __attribute__((always_inline)) static void
    simtStrideStore2D(__gm__ DTYPE *dst, __ubuf__ DTYPE *src,
                      const int32_t stride0, const int32_t stride1,
                      const int32_t storeLower0, const int32_t storeLower1,
                      const int32_t storeSize0, const int32_t storeSize1,
                      const int32_t srcStride0, const int32_t srcStride1) {
  const int32_t total = storeSize0 * storeSize1;
  for (int32_t linear = static_cast<int32_t>(threadIdx.x); linear < total;
       linear += static_cast<int32_t>(blockDim.x)) {
    const int32_t i0 = linear / storeSize1;
    const int32_t i1 = linear - i0 * storeSize1;
    dst[storeLower0 + i0 * stride0 + storeLower1 + i1 * stride1] =
        src[i0 * srcStride0 + i1 * srcStride1];
  }
}

template <typename DTYPE>
__simt_vf__ LAUNCH_BOUND(STRIDE_STORE_THREAD_NUM) __aiv__
    __attribute__((always_inline)) static void
    simtStrideStore3D(__gm__ DTYPE *dst, __ubuf__ DTYPE *src,
                      const int32_t stride0, const int32_t stride1,
                      const int32_t stride2, const int32_t storeLower0,
                      const int32_t storeLower1, const int32_t storeLower2,
                      const int32_t storeSize0, const int32_t storeSize1,
                      const int32_t storeSize2, const int32_t srcStride0,
                      const int32_t srcStride1, const int32_t srcStride2) {
  const int32_t total = storeSize0 * storeSize1 * storeSize2;
  for (int32_t linear = static_cast<int32_t>(threadIdx.x); linear < total;
       linear += static_cast<int32_t>(blockDim.x)) {
    const int32_t i0 = linear / (storeSize1 * storeSize2);
    const int32_t rest = linear - i0 * storeSize1 * storeSize2;
    const int32_t i1 = rest / storeSize2;
    const int32_t i2 = rest - i1 * storeSize2;
    dst[storeLower0 + i0 * stride0 + storeLower1 + i1 * stride1 +
        storeLower2 + i2 * stride2] =
        src[i0 * srcStride0 + i1 * srcStride1 + i2 * srcStride2];
  }
}

template <typename DTYPE>
__aiv__ __attribute__((always_inline)) static void
stride_store_1d_impl(__gm__ DTYPE *dst, int32_t stride, int32_t storeLower,
                     int32_t storeUpper, __ubuf__ DTYPE *src,
                     int32_t srcStride0) {
  int32_t storeSize = getStrideStoreSize(stride, storeLower, storeUpper);
  if (storeSize > 0) {
    cce::async_invoke<simtStrideStore1D<DTYPE>>(
        cce::dim3{STRIDE_STORE_THREAD_NUM}, dst, src, stride, storeLower,
        storeSize, srcStride0);
  }
}

template <typename DTYPE, typename ITYPE>
__aiv__ __attribute__((always_inline)) void
stride_store_1d(memref_t<__gm__ DTYPE, 1> *dst,
                memref_t<__ubuf__ DTYPE, 1> *src, ITYPE offset, ITYPE stride,
                ITYPE numel) {
  static_assert(IsStrideStoreDTypeSupported<DTYPE>::value,
                "Currently stride_store supports "
                "float|half|bfloat16_t|int8_t|int16_t|int32_t|int64_t|"
                "uint8_t|uint16_t|uint32_t|uint64_t.");

  int32_t innerOffset = static_cast<int32_t>(offset);
  int32_t innerStride = static_cast<int32_t>(stride);
  int32_t innerNumel = static_cast<int32_t>(numel);
  int32_t srcSize = static_cast<int32_t>(src->sizes[0]);
  int32_t storeSize =
      getStrideStoreBoundedSize(innerStride, innerNumel, srcSize);
  int32_t storeLower = innerOffset;
  int32_t storeUpper = innerOffset + storeSize * innerStride;
  stride_store_1d_impl<DTYPE>(
      reinterpret_cast<__gm__ DTYPE *>(dst->aligned + dst->offset),
      innerStride, storeLower, storeUpper,
      reinterpret_cast<__ubuf__ DTYPE *>(src->aligned + src->offset),
      static_cast<int32_t>(src->strides[0]));
}

template <typename DTYPE>
__aiv__ __attribute__((always_inline)) static void stride_store_2d_impl(
    __gm__ DTYPE *dst, int32_t stride0, int32_t stride1, int32_t storeLower0,
    int32_t storeLower1, int32_t storeUpper0, int32_t storeUpper1,
    __ubuf__ DTYPE *src, int32_t srcStride0, int32_t srcStride1) {
  int32_t storeSize0 = getStrideStoreSize(stride0, storeLower0, storeUpper0);
  int32_t storeSize1 = getStrideStoreSize(stride1, storeLower1, storeUpper1);
  if (storeSize0 > 0 && storeSize1 > 0) {
    cce::async_invoke<simtStrideStore2D<DTYPE>>(
        cce::dim3{STRIDE_STORE_THREAD_NUM}, dst, src, stride0, stride1,
        storeLower0, storeLower1, storeSize0, storeSize1, srcStride0,
        srcStride1);
  }
}

template <typename DTYPE, typename ITYPE>
__aiv__ __attribute__((always_inline)) void stride_store_2d(
    memref_t<__gm__ DTYPE, 1> *dst, memref_t<__ubuf__ DTYPE, 2> *src,
    ITYPE offset, ITYPE stride0, ITYPE stride1, ITYPE numel0, ITYPE numel1) {
  static_assert(IsStrideStoreDTypeSupported<DTYPE>::value,
                "Currently stride_store supports "
                "float|half|bfloat16_t|int8_t|int16_t|int32_t|int64_t|"
                "uint8_t|uint16_t|uint32_t|uint64_t.");

  int32_t innerOffset = static_cast<int32_t>(offset);
  int32_t innerStride0 = static_cast<int32_t>(stride0);
  int32_t innerStride1 = static_cast<int32_t>(stride1);
  int32_t innerNumel0 = static_cast<int32_t>(numel0);
  int32_t innerNumel1 = static_cast<int32_t>(numel1);
  int32_t srcSize0 = static_cast<int32_t>(src->sizes[0]);
  int32_t srcSize1 = static_cast<int32_t>(src->sizes[1]);
  int32_t storeSize0 =
      getStrideStoreBoundedSize(innerStride0, innerNumel0, srcSize0);
  int32_t storeSize1 =
      getStrideStoreBoundedSize(innerStride1, innerNumel1, srcSize1);
  int32_t storeLower0 = innerOffset;
  int32_t storeLower1 = 0;
  int32_t storeUpper0 = innerOffset + storeSize0 * innerStride0;
  int32_t storeUpper1 = storeSize1 * innerStride1;
  stride_store_2d_impl<DTYPE>(
      reinterpret_cast<__gm__ DTYPE *>(dst->aligned + dst->offset),
      innerStride0, innerStride1, storeLower0, storeLower1, storeUpper0,
      storeUpper1,
      reinterpret_cast<__ubuf__ DTYPE *>(src->aligned + src->offset),
      static_cast<int32_t>(src->strides[0]),
      static_cast<int32_t>(src->strides[1]));
}

template <typename DTYPE>
__aiv__ __attribute__((always_inline)) static void stride_store_3d_impl(
    __gm__ DTYPE *dst, int32_t stride0, int32_t stride1, int32_t stride2,
    int32_t storeLower0, int32_t storeLower1, int32_t storeLower2,
    int32_t storeUpper0, int32_t storeUpper1, int32_t storeUpper2,
    __ubuf__ DTYPE *src, int32_t srcStride0, int32_t srcStride1,
    int32_t srcStride2) {
  int32_t storeSize0 = getStrideStoreSize(stride0, storeLower0, storeUpper0);
  int32_t storeSize1 = getStrideStoreSize(stride1, storeLower1, storeUpper1);
  int32_t storeSize2 = getStrideStoreSize(stride2, storeLower2, storeUpper2);
  if (storeSize0 > 0 && storeSize1 > 0 && storeSize2 > 0) {
    cce::async_invoke<simtStrideStore3D<DTYPE>>(
        cce::dim3{STRIDE_STORE_THREAD_NUM}, dst, src, stride0, stride1,
        stride2, storeLower0, storeLower1, storeLower2, storeSize0,
        storeSize1, storeSize2, srcStride0, srcStride1, srcStride2);
  }
}

template <typename DTYPE, typename ITYPE>
__aiv__ __attribute__((always_inline)) void stride_store_3d(
    memref_t<__gm__ DTYPE, 1> *dst, memref_t<__ubuf__ DTYPE, 3> *src,
    ITYPE offset, ITYPE stride0, ITYPE stride1, ITYPE stride2, ITYPE numel0,
    ITYPE numel1, ITYPE numel2) {
  static_assert(IsStrideStoreDTypeSupported<DTYPE>::value,
                "Currently stride_store supports "
                "float|half|bfloat16_t|int8_t|int16_t|int32_t|int64_t|"
                "uint8_t|uint16_t|uint32_t|uint64_t.");

  int32_t innerOffset = static_cast<int32_t>(offset);
  int32_t innerStride0 = static_cast<int32_t>(stride0);
  int32_t innerStride1 = static_cast<int32_t>(stride1);
  int32_t innerStride2 = static_cast<int32_t>(stride2);
  int32_t innerNumel0 = static_cast<int32_t>(numel0);
  int32_t innerNumel1 = static_cast<int32_t>(numel1);
  int32_t innerNumel2 = static_cast<int32_t>(numel2);
  int32_t srcSize0 = static_cast<int32_t>(src->sizes[0]);
  int32_t srcSize1 = static_cast<int32_t>(src->sizes[1]);
  int32_t srcSize2 = static_cast<int32_t>(src->sizes[2]);
  int32_t storeSize0 =
      getStrideStoreBoundedSize(innerStride0, innerNumel0, srcSize0);
  int32_t storeSize1 =
      getStrideStoreBoundedSize(innerStride1, innerNumel1, srcSize1);
  int32_t storeSize2 =
      getStrideStoreBoundedSize(innerStride2, innerNumel2, srcSize2);
  int32_t storeLower0 = innerOffset;
  int32_t storeLower1 = 0;
  int32_t storeLower2 = 0;
  int32_t storeUpper0 = innerOffset + storeSize0 * innerStride0;
  int32_t storeUpper1 = storeSize1 * innerStride1;
  int32_t storeUpper2 = storeSize2 * innerStride2;
  stride_store_3d_impl<DTYPE>(
      reinterpret_cast<__gm__ DTYPE *>(dst->aligned + dst->offset),
      innerStride0, innerStride1, innerStride2, storeLower0, storeLower1,
      storeLower2, storeUpper0, storeUpper1, storeUpper2,
      reinterpret_cast<__ubuf__ DTYPE *>(src->aligned + src->offset),
      static_cast<int32_t>(src->strides[0]),
      static_cast<int32_t>(src->strides[1]),
      static_cast<int32_t>(src->strides[2]));
}

extern "C" {
REGISTE_STRIDE_STORE_1D(float, int32_t);
REGISTE_STRIDE_STORE_1D(half, int32_t);
REGISTE_STRIDE_STORE_1D(bfloat16_t, int32_t);
REGISTE_STRIDE_STORE_1D(int8_t, int32_t);
REGISTE_STRIDE_STORE_1D(int16_t, int32_t);
REGISTE_STRIDE_STORE_1D(int32_t, int32_t);
REGISTE_STRIDE_STORE_1D(int64_t, int32_t);
REGISTE_STRIDE_STORE_1D(uint8_t, int32_t);
REGISTE_STRIDE_STORE_1D(uint16_t, int32_t);
REGISTE_STRIDE_STORE_1D(uint32_t, int32_t);
REGISTE_STRIDE_STORE_1D(uint64_t, int32_t);
REGISTE_STRIDE_STORE_1D(float, int64_t);
REGISTE_STRIDE_STORE_1D(half, int64_t);
REGISTE_STRIDE_STORE_1D(bfloat16_t, int64_t);
REGISTE_STRIDE_STORE_1D(int8_t, int64_t);
REGISTE_STRIDE_STORE_1D(int16_t, int64_t);
REGISTE_STRIDE_STORE_1D(int32_t, int64_t);
REGISTE_STRIDE_STORE_1D(int64_t, int64_t);
REGISTE_STRIDE_STORE_1D(uint8_t, int64_t);
REGISTE_STRIDE_STORE_1D(uint16_t, int64_t);
REGISTE_STRIDE_STORE_1D(uint32_t, int64_t);
REGISTE_STRIDE_STORE_1D(uint64_t, int64_t);

REGISTE_STRIDE_STORE_2D(float, int32_t);
REGISTE_STRIDE_STORE_2D(half, int32_t);
REGISTE_STRIDE_STORE_2D(bfloat16_t, int32_t);
REGISTE_STRIDE_STORE_2D(int8_t, int32_t);
REGISTE_STRIDE_STORE_2D(int16_t, int32_t);
REGISTE_STRIDE_STORE_2D(int32_t, int32_t);
REGISTE_STRIDE_STORE_2D(int64_t, int32_t);
REGISTE_STRIDE_STORE_2D(uint8_t, int32_t);
REGISTE_STRIDE_STORE_2D(uint16_t, int32_t);
REGISTE_STRIDE_STORE_2D(uint32_t, int32_t);
REGISTE_STRIDE_STORE_2D(uint64_t, int32_t);
REGISTE_STRIDE_STORE_2D(float, int64_t);
REGISTE_STRIDE_STORE_2D(half, int64_t);
REGISTE_STRIDE_STORE_2D(bfloat16_t, int64_t);
REGISTE_STRIDE_STORE_2D(int8_t, int64_t);
REGISTE_STRIDE_STORE_2D(int16_t, int64_t);
REGISTE_STRIDE_STORE_2D(int32_t, int64_t);
REGISTE_STRIDE_STORE_2D(int64_t, int64_t);
REGISTE_STRIDE_STORE_2D(uint8_t, int64_t);
REGISTE_STRIDE_STORE_2D(uint16_t, int64_t);
REGISTE_STRIDE_STORE_2D(uint32_t, int64_t);
REGISTE_STRIDE_STORE_2D(uint64_t, int64_t);

REGISTE_STRIDE_STORE_3D(float, int32_t);
REGISTE_STRIDE_STORE_3D(half, int32_t);
REGISTE_STRIDE_STORE_3D(bfloat16_t, int32_t);
REGISTE_STRIDE_STORE_3D(int8_t, int32_t);
REGISTE_STRIDE_STORE_3D(int16_t, int32_t);
REGISTE_STRIDE_STORE_3D(int32_t, int32_t);
REGISTE_STRIDE_STORE_3D(int64_t, int32_t);
REGISTE_STRIDE_STORE_3D(uint8_t, int32_t);
REGISTE_STRIDE_STORE_3D(uint16_t, int32_t);
REGISTE_STRIDE_STORE_3D(uint32_t, int32_t);
REGISTE_STRIDE_STORE_3D(uint64_t, int32_t);
REGISTE_STRIDE_STORE_3D(float, int64_t);
REGISTE_STRIDE_STORE_3D(half, int64_t);
REGISTE_STRIDE_STORE_3D(bfloat16_t, int64_t);
REGISTE_STRIDE_STORE_3D(int8_t, int64_t);
REGISTE_STRIDE_STORE_3D(int16_t, int64_t);
REGISTE_STRIDE_STORE_3D(int32_t, int64_t);
REGISTE_STRIDE_STORE_3D(int64_t, int64_t);
REGISTE_STRIDE_STORE_3D(uint8_t, int64_t);
REGISTE_STRIDE_STORE_3D(uint16_t, int64_t);
REGISTE_STRIDE_STORE_3D(uint32_t, int64_t);
REGISTE_STRIDE_STORE_3D(uint64_t, int64_t);
}

#endif
