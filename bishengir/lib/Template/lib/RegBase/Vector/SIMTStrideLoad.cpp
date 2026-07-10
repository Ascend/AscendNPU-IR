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

constexpr uint32_t STRIDE_LOAD_THREAD_NUM = 1024;

template <typename DTYPE>
struct IsStrideLoadDTypeSupported {
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
getStrideLoadBoundedSize(int32_t stride, int32_t numel, int32_t dstSize) {
  if (stride < 0 || numel <= 0 || dstSize <= 0) {
    return 0;
  }
  if (stride == 0) {
    return dstSize == 1 ? 1 : 0;
  }
  return numel < dstSize ? numel : dstSize;
}

__aiv__ __attribute__((always_inline)) static int32_t
getStrideLoadSize(int32_t stride, int32_t loadLower, int32_t loadUpper) {
  if (stride < 0 || loadUpper <= loadLower) {
    return 0;
  }
  if (stride == 0) {
    return 1;
  }
  return (loadUpper - loadLower) / stride;
}

__aiv__ __attribute__((always_inline)) static int32_t
getStrideLoadUpper(int32_t loadLower, int32_t stride, int32_t loadSize) {
  if (loadSize <= 0) {
    return loadLower;
  }
  if (stride == 0) {
    return loadLower + 1;
  }
  return loadLower + loadSize * stride;
}

template <typename DTYPE>
__simt_vf__ LAUNCH_BOUND(STRIDE_LOAD_THREAD_NUM) __aiv__
    __attribute__((always_inline)) static void
    simtStrideLoad1D(volatile __gm__ DTYPE *src, __ubuf__ DTYPE *dst,
                     const int32_t stride, const int32_t loadLower,
                     const int32_t loadSize, const int32_t dstStride0) {
  for (int32_t i0 = static_cast<int32_t>(threadIdx.x); i0 < loadSize;
       i0 += static_cast<int32_t>(blockDim.x)) {
    dst[i0 * dstStride0] = src[loadLower + i0 * stride];
  }
}

template <typename DTYPE>
__simt_vf__ LAUNCH_BOUND(STRIDE_LOAD_THREAD_NUM) __aiv__
    __attribute__((always_inline)) static void
    simtStridePad1D(__ubuf__ DTYPE *dst, const int32_t dstSize0,
                    const int32_t dstStride0, const DTYPE other) {
  for (int32_t i0 = static_cast<int32_t>(threadIdx.x); i0 < dstSize0;
       i0 += static_cast<int32_t>(blockDim.x)) {
    dst[i0 * dstStride0] = other;
  }
}

template <typename DTYPE>
__simt_vf__ LAUNCH_BOUND(STRIDE_LOAD_THREAD_NUM) __aiv__
    __attribute__((always_inline)) static void
    simtStrideLoad2D(volatile __gm__ DTYPE *src, __ubuf__ DTYPE *dst,
                     const int32_t stride0, const int32_t stride1,
                     const int32_t loadLower0, const int32_t loadLower1,
                     const int32_t loadSize0, const int32_t loadSize1,
                     const int32_t dstStride0, const int32_t dstStride1) {
  const int32_t total = loadSize0 * loadSize1;
  for (int32_t linear = static_cast<int32_t>(threadIdx.x); linear < total;
       linear += static_cast<int32_t>(blockDim.x)) {
    const int32_t i0 = linear / loadSize1;
    const int32_t i1 = linear - i0 * loadSize1;
    dst[i0 * dstStride0 + i1 * dstStride1] =
        src[loadLower0 + i0 * stride0 + loadLower1 + i1 * stride1];
  }
}

template <typename DTYPE>
__simt_vf__ LAUNCH_BOUND(STRIDE_LOAD_THREAD_NUM) __aiv__
    __attribute__((always_inline)) static void
    simtStridePad2D(__ubuf__ DTYPE *dst, const int32_t dstSize0,
                    const int32_t dstSize1, const int32_t dstStride0,
                    const int32_t dstStride1, const DTYPE other) {
  const int32_t total = dstSize0 * dstSize1;
  for (int32_t linear = static_cast<int32_t>(threadIdx.x); linear < total;
       linear += static_cast<int32_t>(blockDim.x)) {
    const int32_t i0 = linear / dstSize1;
    const int32_t i1 = linear - i0 * dstSize1;
    dst[i0 * dstStride0 + i1 * dstStride1] = other;
  }
}

template <typename DTYPE>
__simt_vf__ LAUNCH_BOUND(STRIDE_LOAD_THREAD_NUM) __aiv__
    __attribute__((always_inline)) static void
    simtStrideLoad3D(volatile __gm__ DTYPE *src, __ubuf__ DTYPE *dst,
                     const int32_t stride0, const int32_t stride1,
                     const int32_t stride2, const int32_t loadLower0,
                     const int32_t loadLower1, const int32_t loadLower2,
                     const int32_t loadSize0, const int32_t loadSize1,
                     const int32_t loadSize2, const int32_t dstStride0,
                     const int32_t dstStride1, const int32_t dstStride2) {
  const int32_t total = loadSize0 * loadSize1 * loadSize2;
  for (int32_t linear = static_cast<int32_t>(threadIdx.x); linear < total;
       linear += static_cast<int32_t>(blockDim.x)) {
    const int32_t i0 = linear / (loadSize1 * loadSize2);
    const int32_t rest = linear - i0 * loadSize1 * loadSize2;
    const int32_t i1 = rest / loadSize2;
    const int32_t i2 = rest - i1 * loadSize2;
    dst[i0 * dstStride0 + i1 * dstStride1 + i2 * dstStride2] =
        src[loadLower0 + i0 * stride0 + loadLower1 + i1 * stride1 +
            loadLower2 + i2 * stride2];
  }
}

template <typename DTYPE>
__simt_vf__ LAUNCH_BOUND(STRIDE_LOAD_THREAD_NUM) __aiv__
    __attribute__((always_inline)) static void
    simtStridePad3D(__ubuf__ DTYPE *dst, const int32_t dstSize0,
                    const int32_t dstSize1, const int32_t dstSize2,
                    const int32_t dstStride0, const int32_t dstStride1,
                    const int32_t dstStride2, const DTYPE other) {
  const int32_t total = dstSize0 * dstSize1 * dstSize2;
  for (int32_t linear = static_cast<int32_t>(threadIdx.x); linear < total;
       linear += static_cast<int32_t>(blockDim.x)) {
    const int32_t i0 = linear / (dstSize1 * dstSize2);
    const int32_t rest = linear - i0 * dstSize1 * dstSize2;
    const int32_t i1 = rest / dstSize2;
    const int32_t i2 = rest - i1 * dstSize2;
    dst[i0 * dstStride0 + i1 * dstStride1 + i2 * dstStride2] = other;
  }
}

template <typename DTYPE>
__aiv__ __attribute__((always_inline)) static void
stride_load_1d_impl(volatile __gm__ DTYPE *src, int32_t stride,
                    int32_t loadLower, int32_t loadUpper, int32_t padUpper,
                    __ubuf__ DTYPE *dst, int32_t dstStride0,
                    const DTYPE other) {
  int32_t loadSize = getStrideLoadSize(stride, loadLower, loadUpper);
  if (loadSize < padUpper) {
    cce::async_invoke<simtStridePad1D<DTYPE>>(
        cce::dim3{STRIDE_LOAD_THREAD_NUM}, dst, padUpper, dstStride0, other);
  }
  if (loadSize > 0) {
    cce::async_invoke<simtStrideLoad1D<DTYPE>>(
        cce::dim3{STRIDE_LOAD_THREAD_NUM}, src, dst, stride, loadLower,
        loadSize, dstStride0);
  }
}

template <typename DTYPE, typename ITYPE>
__aiv__ __attribute__((always_inline)) void
stride_load_1d(memref_t<__gm__ DTYPE, 1> *src,
               memref_t<__ubuf__ DTYPE, 1> *dst, ITYPE offset, DTYPE other,
               ITYPE stride, ITYPE numel) {
  static_assert(IsStrideLoadDTypeSupported<DTYPE>::value,
                "Currently stride_load supports "
                "float|half|bfloat16_t|int8_t|int16_t|int32_t|int64_t|"
                "uint8_t|uint16_t|uint32_t|uint64_t.");

  int32_t innerOffset = static_cast<int32_t>(offset);
  int32_t innerStride = static_cast<int32_t>(stride);
  int32_t innerNumel = static_cast<int32_t>(numel);
  int32_t dstSize = static_cast<int32_t>(dst->sizes[0]);
  int32_t loadSize =
      getStrideLoadBoundedSize(innerStride, innerNumel, dstSize);
  int32_t loadLower = innerOffset;
  int32_t loadUpper =
      getStrideLoadUpper(loadLower, innerStride, loadSize);
  int32_t padUpper = dstSize;
  stride_load_1d_impl<DTYPE>(
      reinterpret_cast<volatile __gm__ DTYPE *>(src->aligned + src->offset),
      innerStride, loadLower, loadUpper, padUpper,
      reinterpret_cast<__ubuf__ DTYPE *>(dst->aligned + dst->offset),
      static_cast<int32_t>(dst->strides[0]), other);
}

template <typename DTYPE>
__aiv__ __attribute__((always_inline)) static void stride_load_2d_impl(
    volatile __gm__ DTYPE *src, int32_t stride0, int32_t stride1,
    int32_t loadLower0, int32_t loadLower1, int32_t loadUpper0,
    int32_t loadUpper1, int32_t dstSize0, int32_t dstSize1, __ubuf__ DTYPE *dst,
    int32_t dstStride0, int32_t dstStride1, const DTYPE other) {
  int32_t loadSize0 = getStrideLoadSize(stride0, loadLower0, loadUpper0);
  int32_t loadSize1 = getStrideLoadSize(stride1, loadLower1, loadUpper1);
  if (loadSize0 < dstSize0 || loadSize1 < dstSize1) {
    cce::async_invoke<simtStridePad2D<DTYPE>>(
        cce::dim3{STRIDE_LOAD_THREAD_NUM}, dst, dstSize0, dstSize1, dstStride0,
        dstStride1, other);
  }
  if (loadSize0 > 0 && loadSize1 > 0) {
    cce::async_invoke<simtStrideLoad2D<DTYPE>>(
        cce::dim3{STRIDE_LOAD_THREAD_NUM}, src, dst, stride0, stride1,
        loadLower0, loadLower1, loadSize0, loadSize1, dstStride0, dstStride1);
  }
}

template <typename DTYPE, typename ITYPE>
__aiv__ __attribute__((always_inline)) void
stride_load_2d(memref_t<__gm__ DTYPE, 1> *src,
               memref_t<__ubuf__ DTYPE, 2> *dst, ITYPE offset, DTYPE other,
               ITYPE stride0, ITYPE stride1, ITYPE numel0, ITYPE numel1) {
  static_assert(IsStrideLoadDTypeSupported<DTYPE>::value,
                "Currently stride_load supports "
                "float|half|bfloat16_t|int8_t|int16_t|int32_t|int64_t|"
                "uint8_t|uint16_t|uint32_t|uint64_t.");

  int32_t innerOffset = static_cast<int32_t>(offset);
  int32_t innerStride0 = static_cast<int32_t>(stride0);
  int32_t innerStride1 = static_cast<int32_t>(stride1);
  int32_t innerNumel0 = static_cast<int32_t>(numel0);
  int32_t innerNumel1 = static_cast<int32_t>(numel1);
  int32_t dstSize0 = static_cast<int32_t>(dst->sizes[0]);
  int32_t dstSize1 = static_cast<int32_t>(dst->sizes[1]);
  int32_t loadSize0 =
      getStrideLoadBoundedSize(innerStride0, innerNumel0, dstSize0);
  int32_t loadSize1 =
      getStrideLoadBoundedSize(innerStride1, innerNumel1, dstSize1);
  int32_t loadLower0 = innerOffset;
  int32_t loadLower1 = 0;
  int32_t loadUpper0 =
      getStrideLoadUpper(loadLower0, innerStride0, loadSize0);
  int32_t loadUpper1 =
      getStrideLoadUpper(loadLower1, innerStride1, loadSize1);
  stride_load_2d_impl<DTYPE>(
      reinterpret_cast<volatile __gm__ DTYPE *>(src->aligned + src->offset),
      innerStride0, innerStride1, loadLower0, loadLower1, loadUpper0,
      loadUpper1, dstSize0, dstSize1,
      reinterpret_cast<__ubuf__ DTYPE *>(dst->aligned + dst->offset),
      static_cast<int32_t>(dst->strides[0]),
      static_cast<int32_t>(dst->strides[1]), other);
}

template <typename DTYPE>
__aiv__ __attribute__((always_inline)) static void stride_load_3d_impl(
    volatile __gm__ DTYPE *src, int32_t stride0, int32_t stride1,
    int32_t stride2, int32_t loadLower0, int32_t loadLower1,
    int32_t loadLower2, int32_t loadUpper0, int32_t loadUpper1,
    int32_t loadUpper2, int32_t dstSize0, int32_t dstSize1, int32_t dstSize2,
    __ubuf__ DTYPE *dst, int32_t dstStride0, int32_t dstStride1,
    int32_t dstStride2, const DTYPE other) {
  int32_t loadSize0 = getStrideLoadSize(stride0, loadLower0, loadUpper0);
  int32_t loadSize1 = getStrideLoadSize(stride1, loadLower1, loadUpper1);
  int32_t loadSize2 = getStrideLoadSize(stride2, loadLower2, loadUpper2);
  if (loadSize0 < dstSize0 || loadSize1 < dstSize1 || loadSize2 < dstSize2) {
    cce::async_invoke<simtStridePad3D<DTYPE>>(
        cce::dim3{STRIDE_LOAD_THREAD_NUM}, dst, dstSize0, dstSize1, dstSize2,
        dstStride0, dstStride1, dstStride2, other);
  }
  if (loadSize0 > 0 && loadSize1 > 0 && loadSize2 > 0) {
    cce::async_invoke<simtStrideLoad3D<DTYPE>>(
        cce::dim3{STRIDE_LOAD_THREAD_NUM}, src, dst, stride0, stride1, stride2,
        loadLower0, loadLower1, loadLower2, loadSize0, loadSize1, loadSize2,
        dstStride0, dstStride1, dstStride2);
  }
}

template <typename DTYPE, typename ITYPE>
__aiv__ __attribute__((always_inline)) void stride_load_3d(
    memref_t<__gm__ DTYPE, 1> *src, memref_t<__ubuf__ DTYPE, 3> *dst,
    ITYPE offset, DTYPE other, ITYPE stride0, ITYPE stride1, ITYPE stride2,
    ITYPE numel0, ITYPE numel1, ITYPE numel2) {
  static_assert(IsStrideLoadDTypeSupported<DTYPE>::value,
                "Currently stride_load supports "
                "float|half|bfloat16_t|int8_t|int16_t|int32_t|int64_t|"
                "uint8_t|uint16_t|uint32_t|uint64_t.");

  int32_t innerOffset = static_cast<int32_t>(offset);
  int32_t innerStride0 = static_cast<int32_t>(stride0);
  int32_t innerStride1 = static_cast<int32_t>(stride1);
  int32_t innerStride2 = static_cast<int32_t>(stride2);
  int32_t innerNumel0 = static_cast<int32_t>(numel0);
  int32_t innerNumel1 = static_cast<int32_t>(numel1);
  int32_t innerNumel2 = static_cast<int32_t>(numel2);
  int32_t dstSize0 = static_cast<int32_t>(dst->sizes[0]);
  int32_t dstSize1 = static_cast<int32_t>(dst->sizes[1]);
  int32_t dstSize2 = static_cast<int32_t>(dst->sizes[2]);
  int32_t loadSize0 =
      getStrideLoadBoundedSize(innerStride0, innerNumel0, dstSize0);
  int32_t loadSize1 =
      getStrideLoadBoundedSize(innerStride1, innerNumel1, dstSize1);
  int32_t loadSize2 =
      getStrideLoadBoundedSize(innerStride2, innerNumel2, dstSize2);
  int32_t loadLower0 = innerOffset;
  int32_t loadLower1 = 0;
  int32_t loadLower2 = 0;
  int32_t loadUpper0 =
      getStrideLoadUpper(loadLower0, innerStride0, loadSize0);
  int32_t loadUpper1 =
      getStrideLoadUpper(loadLower1, innerStride1, loadSize1);
  int32_t loadUpper2 =
      getStrideLoadUpper(loadLower2, innerStride2, loadSize2);
  stride_load_3d_impl<DTYPE>(
      reinterpret_cast<volatile __gm__ DTYPE *>(src->aligned + src->offset),
      innerStride0, innerStride1, innerStride2, loadLower0, loadLower1,
      loadLower2, loadUpper0, loadUpper1, loadUpper2, dstSize0, dstSize1,
      dstSize2, reinterpret_cast<__ubuf__ DTYPE *>(dst->aligned + dst->offset),
      static_cast<int32_t>(dst->strides[0]),
      static_cast<int32_t>(dst->strides[1]),
      static_cast<int32_t>(dst->strides[2]), other);
}

extern "C" {
REGISTE_STRIDE_LOAD_1D(float, int32_t);
REGISTE_STRIDE_LOAD_1D(half, int32_t);
REGISTE_STRIDE_LOAD_1D(bfloat16_t, int32_t);
REGISTE_STRIDE_LOAD_1D(int8_t, int32_t);
REGISTE_STRIDE_LOAD_1D(int16_t, int32_t);
REGISTE_STRIDE_LOAD_1D(int32_t, int32_t);
REGISTE_STRIDE_LOAD_1D(int64_t, int32_t);
REGISTE_STRIDE_LOAD_1D(uint8_t, int32_t);
REGISTE_STRIDE_LOAD_1D(uint16_t, int32_t);
REGISTE_STRIDE_LOAD_1D(uint32_t, int32_t);
REGISTE_STRIDE_LOAD_1D(uint64_t, int32_t);
REGISTE_STRIDE_LOAD_1D(float, int64_t);
REGISTE_STRIDE_LOAD_1D(half, int64_t);
REGISTE_STRIDE_LOAD_1D(bfloat16_t, int64_t);
REGISTE_STRIDE_LOAD_1D(int8_t, int64_t);
REGISTE_STRIDE_LOAD_1D(int16_t, int64_t);
REGISTE_STRIDE_LOAD_1D(int32_t, int64_t);
REGISTE_STRIDE_LOAD_1D(int64_t, int64_t);
REGISTE_STRIDE_LOAD_1D(uint8_t, int64_t);
REGISTE_STRIDE_LOAD_1D(uint16_t, int64_t);
REGISTE_STRIDE_LOAD_1D(uint32_t, int64_t);
REGISTE_STRIDE_LOAD_1D(uint64_t, int64_t);

REGISTE_STRIDE_LOAD_2D(float, int32_t);
REGISTE_STRIDE_LOAD_2D(half, int32_t);
REGISTE_STRIDE_LOAD_2D(bfloat16_t, int32_t);
REGISTE_STRIDE_LOAD_2D(int8_t, int32_t);
REGISTE_STRIDE_LOAD_2D(int16_t, int32_t);
REGISTE_STRIDE_LOAD_2D(int32_t, int32_t);
REGISTE_STRIDE_LOAD_2D(int64_t, int32_t);
REGISTE_STRIDE_LOAD_2D(uint8_t, int32_t);
REGISTE_STRIDE_LOAD_2D(uint16_t, int32_t);
REGISTE_STRIDE_LOAD_2D(uint32_t, int32_t);
REGISTE_STRIDE_LOAD_2D(uint64_t, int32_t);
REGISTE_STRIDE_LOAD_2D(float, int64_t);
REGISTE_STRIDE_LOAD_2D(half, int64_t);
REGISTE_STRIDE_LOAD_2D(bfloat16_t, int64_t);
REGISTE_STRIDE_LOAD_2D(int8_t, int64_t);
REGISTE_STRIDE_LOAD_2D(int16_t, int64_t);
REGISTE_STRIDE_LOAD_2D(int32_t, int64_t);
REGISTE_STRIDE_LOAD_2D(int64_t, int64_t);
REGISTE_STRIDE_LOAD_2D(uint8_t, int64_t);
REGISTE_STRIDE_LOAD_2D(uint16_t, int64_t);
REGISTE_STRIDE_LOAD_2D(uint32_t, int64_t);
REGISTE_STRIDE_LOAD_2D(uint64_t, int64_t);

REGISTE_STRIDE_LOAD_3D(float, int32_t);
REGISTE_STRIDE_LOAD_3D(half, int32_t);
REGISTE_STRIDE_LOAD_3D(bfloat16_t, int32_t);
REGISTE_STRIDE_LOAD_3D(int8_t, int32_t);
REGISTE_STRIDE_LOAD_3D(int16_t, int32_t);
REGISTE_STRIDE_LOAD_3D(int32_t, int32_t);
REGISTE_STRIDE_LOAD_3D(int64_t, int32_t);
REGISTE_STRIDE_LOAD_3D(uint8_t, int32_t);
REGISTE_STRIDE_LOAD_3D(uint16_t, int32_t);
REGISTE_STRIDE_LOAD_3D(uint32_t, int32_t);
REGISTE_STRIDE_LOAD_3D(uint64_t, int32_t);
REGISTE_STRIDE_LOAD_3D(float, int64_t);
REGISTE_STRIDE_LOAD_3D(half, int64_t);
REGISTE_STRIDE_LOAD_3D(bfloat16_t, int64_t);
REGISTE_STRIDE_LOAD_3D(int8_t, int64_t);
REGISTE_STRIDE_LOAD_3D(int16_t, int64_t);
REGISTE_STRIDE_LOAD_3D(int32_t, int64_t);
REGISTE_STRIDE_LOAD_3D(int64_t, int64_t);
REGISTE_STRIDE_LOAD_3D(uint8_t, int64_t);
REGISTE_STRIDE_LOAD_3D(uint16_t, int64_t);
REGISTE_STRIDE_LOAD_3D(uint32_t, int64_t);
REGISTE_STRIDE_LOAD_3D(uint64_t, int64_t);
}

#endif
