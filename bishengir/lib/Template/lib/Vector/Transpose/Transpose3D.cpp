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

#include "Utils.h"
#include "Vector/Transpose/TransposeUtils.h"
#include "Vector/VecUtils.h"

template <typename T>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_transpose_nlast_3d(memref_t<__ubuf__ T, 3> *src,
                                   memref_t<__ubuf__ T, 3> *dst) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  assert(isAddress32ByteAligned(src_ptr) &&
         "The starting address of src must be 32byte aligned.");
  assert(isAddress32ByteAligned(dst_ptr) &&
         "The starting address of dst must be 32byte aligned.");
  assert((isSizeAlignedToBlock<T>(src->strides[0]) &&
          isSizeAlignedToBlock<T>(src->strides[1]) &&
          isSizeAlignedToBlock<T>(dst->strides[0]) &&
          isSizeAlignedToBlock<T>(dst->strides[1])) &&
         "The src/dst strides[0]/strides[1] must be aligned to block.");
  assert((src->strides[2] == 1 && dst->strides[2] == 1) &&
         "The src/dst last dim must be continuous.");
#endif
}

/// transpose src (a, b, c) with stride [n1, n2, 1] to dst (b, a, c) with
/// stride [m1, m2, 1].
///
/// constraint:
/// 1. dim of src/dst must be 3.
/// 2. the start pointer address, namely aligned + offset, should be aligned
/// to ub_block_unit.
/// 3. n1, n2, m1, m2 should be aligned to ub_block_unit.
template <typename T>
__aiv__ __attribute__((always_inline)) void
transpose_nlast_3d(memref_t<__ubuf__ T, 3> *src, memref_t<__ubuf__ T, 3> *dst) {
  if (is_no_op<3>(src->sizes)) {
    return;
  }
  // Input parameter constraints assert.
  check_inputs_of_transpose_nlast_3d(src, dst);

  const int64_t src_size0 = src->sizes[0];
  const int64_t src_size1 = src->sizes[1];
  const int64_t src_stride0 = src->strides[0];
  const int64_t src_stride1 = src->strides[1];
  const int64_t dst_stride0 = dst->strides[0];
  const int64_t dst_stride1 = dst->strides[1];
  __ubuf__ T *src_ptr = src->aligned + src->offset;
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  const int64_t size2_align = CEIL_FACTOR(src->sizes[2], num_per_block);

  bool is_loop_src_size0 = src_size0 < src_size1;
  int64_t loop_num = is_loop_src_size0 ? src_size0 : src_size1;

  int64_t dst_loop_stride = is_loop_src_size0 ? dst_stride1 : dst_stride0;
  int64_t src_loop_stride = is_loop_src_size0 ? src_stride0 : src_stride1;
  int64_t nburst = is_loop_src_size0 ? src_size1 : src_size0;
  int64_t src_gap = is_loop_src_size0
                        ? (src_stride1 - size2_align) / num_per_block
                        : (src_stride0 - size2_align) / num_per_block;
  int64_t dst_gap = is_loop_src_size0
                        ? (dst_stride0 - size2_align) / num_per_block
                        : (dst_stride1 - size2_align) / num_per_block;
  for (int64_t i = 0; i < loop_num; ++i) {
    INTRINSIC(copy_ubuf_to_ubuf, dst_ptr + i * dst_loop_stride,
              src_ptr + i * src_loop_stride,
              0,                           // sid
              nburst,                      // nBurst
              size2_align / num_per_block, // lenBurst
              src_gap,                     // srcStride
              dst_gap);                    // dstStride
  }
}

/// transpose with last axis, src (a, b, c) to  dst (c, b, c)
/// \param src (type: memref<axbxcxT, stride[m0, n0, 1]>)
/// \param dst (type: memref<cxbxaxT, stride[m1, n1, 1]>)
/// 'm0' and 'm1' are stride of src and dst separately
///
/// constraint:
/// 1. data type should be b8, b16, b32
/// 2. permute = [2, 1, 0].
/// 3. if data type is b8, b16, 'm0' and 'm1' should be aligned to
/// multiple of ub_block_unit
/// 4. if data type is b32, stride should satisfy the one of rules below,
///    1) 'm0' is multiple of 16, and 'm1' is multiple of 8, or
///    2) 'm0' is multiple of 8, and 'm1' is multiple of 16.
///
/// Example:
/// memref<2x5x19xi16, strided<[160, 32, 1]>> => memref<19x5x2xi16, strided<[80,
/// 16, 1]>>
template <typename T>
__aiv__ __attribute__((always_inline)) void
vnchwconv_3d(memref_t<__ubuf__ T, 3> *src, memref_t<__ubuf__ T, 3> *dst) {
  if (is_no_op<3>(src->sizes)) {
    return;
  }
  static_assert((sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4) &&
                "Transpose can not support this type");

  memref_t<__ubuf__ T, 2> new_src_2d{src->allocated,
                                     src->aligned,
                                     src->offset,
                                     {src->sizes[0], src->sizes[2]},
                                     {src->strides[0], src->strides[2]}};
  memref_t<__ubuf__ T, 2> new_dst_2d{dst->allocated,
                                     dst->aligned,
                                     dst->offset,
                                     {dst->sizes[0], dst->sizes[2]},
                                     {dst->strides[0], dst->strides[2]}};

  // TODO select repeat and for loop axis depending on specific size
  for (int64_t i = 0; i < src->sizes[1]; ++i) {
    new_src_2d.offset = src->offset + i * src->strides[1];
    new_dst_2d.offset = dst->offset + i * dst->strides[1];
    vnchwconv_2d<T>(&new_src_2d, &new_dst_2d);
  }
}

/// transpose with last axis, src (a, b, c) to  dst (c, b, c)
/// \param src (type: memref<axbxcxb64, stride[m0, n0, 1]>)
/// \param dst (type: memref<cxbxaxb64, stride[m1, n1, 1]>)
/// 'm0' and 'm1' are stride of src and dst separately
///
/// constraint:
/// 1. data type should be b64
/// 2. permute = [2, 1, 0].
/// 3. 'm0' and 'm1' should be aligned to multiple of ub_block_unit
template <typename T>
__aiv__ __attribute__((always_inline)) void
vnchwconv_3d_with_tmp(memref_t<__ubuf__ T, 3> *src,
                      memref_t<__ubuf__ T, 3> *dst,
                      memref_t<__ubuf__ T, 1> *tmp) {
  if (is_no_op<3>(src->sizes)) {
    return;
  }
  static_assert((sizeof(T) == 8) &&
                "Transpose with tmp can not support this type");

  memref_t<__ubuf__ T, 2> new_src_2d{src->allocated,
                                     src->aligned,
                                     src->offset,
                                     {src->sizes[0], src->sizes[2]},
                                     {src->strides[0], src->strides[2]}};
  memref_t<__ubuf__ T, 2> new_dst_2d{dst->allocated,
                                     dst->aligned,
                                     dst->offset,
                                     {dst->sizes[0], dst->sizes[2]},
                                     {dst->strides[0], dst->strides[2]}};

  // TODO select repeat and for loop axis depending on specific size
  for (int64_t i = 0; i < src->sizes[1]; ++i) {
    new_src_2d.offset = src->offset + i * src->strides[1];
    new_dst_2d.offset = dst->offset + i * dst->strides[1];
    vnchwconv_2d_with_tmp<T>(&new_src_2d, &new_dst_2d, tmp);
  }
}

extern "C" {
//===-------------------------------------------------------------------===//
// Transpose 3d without last axis
//===-------------------------------------------------------------------===//
REGISTE_TRANSPOSE_3D_WITH_NLAST_AXIS(uint8_t)
REGISTE_TRANSPOSE_3D_WITH_NLAST_AXIS(int8_t)
REGISTE_TRANSPOSE_3D_WITH_NLAST_AXIS(uint16_t)
REGISTE_TRANSPOSE_3D_WITH_NLAST_AXIS(int16_t)
REGISTE_TRANSPOSE_3D_WITH_NLAST_AXIS(half)
REGISTE_TRANSPOSE_3D_WITH_NLAST_AXIS(bfloat16_t)
REGISTE_TRANSPOSE_3D_WITH_NLAST_AXIS(uint32_t)
REGISTE_TRANSPOSE_3D_WITH_NLAST_AXIS(int32_t)
REGISTE_TRANSPOSE_3D_WITH_NLAST_AXIS(float)
REGISTE_TRANSPOSE_3D_WITH_NLAST_AXIS(uint64_t)
REGISTE_TRANSPOSE_3D_WITH_NLAST_AXIS(int64_t)

//===-------------------------------------------------------------------===//
// Transpose 3d with last axis
//===-------------------------------------------------------------------===//
REGISTE_TRANSPOSE_WITH_LAST_AXIS(3, uint8_t)
REGISTE_TRANSPOSE_WITH_LAST_AXIS(3, int8_t)
REGISTE_TRANSPOSE_WITH_LAST_AXIS(3, uint16_t)
REGISTE_TRANSPOSE_WITH_LAST_AXIS(3, int16_t)
REGISTE_TRANSPOSE_WITH_LAST_AXIS(3, half)
REGISTE_TRANSPOSE_WITH_LAST_AXIS(3, bfloat16_t)
REGISTE_TRANSPOSE_WITH_LAST_AXIS(3, uint32_t)
REGISTE_TRANSPOSE_WITH_LAST_AXIS(3, int32_t)
REGISTE_TRANSPOSE_WITH_LAST_AXIS(3, float)
REGISTE_TRANSPOSE_WITH_LAST_AXIS_WITH_TMP(3, uint64_t)
REGISTE_TRANSPOSE_WITH_LAST_AXIS_WITH_TMP(3, int64_t)
}