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

#include "DMA/DMAUtils.h"

template <typename T>
__aicore__ __attribute__((always_inline)) void
check_inputs_of_load_gm_to_cbuf_3d_core(memref_t<__gm__ T, 3> *src,
                                        memref_t<__cbuf__ T, 3> *dst,
                                        pad_t pad_mode) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  auto dst_ptr = dst->aligned + dst->offset;
  auto stride0_cbuf = dst->strides[0];
  auto stride1_cbuf = dst->strides[1];
  auto stride2_cbuf = dst->strides[2];
  assert(isAddress32ByteAligned(dst_ptr) &&
         "The starting address of dst must be 32byte aligned.");
  assert(((isSizeAlignedToBlock<T>(stride0_cbuf) || stride0_cbuf == 1) &&
          (isSizeAlignedToBlock<T>(stride1_cbuf) || stride1_cbuf == 1) &&
          (isSizeAlignedToBlock<T>(stride2_cbuf) || stride2_cbuf == 1)) &&
         "The dst strides[0]/strides[1]/strides[2] must be 1 or aligned to"
         "block.");
#endif
}

template <typename T>
__aicore__ __attribute__((always_inline)) void
load_gm_to_cbuf_3d_core(memref_t<__gm__ T, 3> *src,
                        memref_t<__cbuf__ T, 3> *dst,
                        pad_t pad_mode = pad_t::PAD_NONE) {
  if (is_no_op<3>(src->sizes)) {
    return;
  }

  // Input parameter constraints assert.
  check_inputs_of_load_gm_to_cbuf_3d_core(src, dst, pad_mode);

  // deal copy memref<1x1x1> to memref<1x1x1>
  if (dst->sizes[0] == 1 && dst->sizes[1] == 1 && dst->sizes[2] == 1) {
    auto src_ptr = src->aligned + src->offset;
    auto dst_ptr = dst->aligned + dst->offset;
    load_gm_to_cbuf_intrin_core(src_ptr, 0, dst_ptr, 0, 1, 1 * sizeof(T), 0, 0,
                                pad_mode);
    return;
  }

  if (src->strides[2] == 1 && dst->strides[2] == 1) [[likely]] {
    // last dimension is contiguous
    load_gm_to_cbuf_3d_core_with_contiguous_last_dim(src, dst, pad_mode);
    return;
  }

  // last dimension is not contiguous,
  // view the src (size0, size1, size2) with stride [stride0, stride1, stride2]
  // as viewed_src (size0, size1, size2, 1) with stride [stride0, stride1,
  // stride2, 1], where last dimension of viewed_src is contiguous

  int64_t size0 = src->sizes[0];
  int64_t size1 = src->sizes[1];
  int64_t size2 = src->sizes[2];
  int64_t stride0_cbuf = dst->strides[0];
  int64_t stride1_cbuf = dst->strides[1];
  int64_t stride2_cbuf = dst->strides[2];
  int64_t stride0_gm = src->strides[0];
  int64_t stride1_gm = src->strides[1];
  int64_t stride2_gm = src->strides[2];

  // choose minimum dimension as dim0 and set the other two dimensions as dim1
  // and dim2
  int64_t min_axis = 0;
  int64_t min_size = size0;
  if (min_size > size1) {
    min_size = size1;
    min_axis = 1;
  }
  if (min_size > size2) {
    min_size = size2;
    min_axis = 2;
  }

  if (min_axis == 1) {
    size0 = src->sizes[1];
    size1 = src->sizes[0];
    stride0_cbuf = dst->strides[1];
    stride1_cbuf = dst->strides[0];
    stride0_gm = src->strides[1];
    stride1_gm = src->strides[0];
  } else if (min_axis == 2) {
    size0 = src->sizes[2];
    size1 = src->sizes[0];
    size2 = src->sizes[1];
    stride0_cbuf = dst->strides[2];
    stride1_cbuf = dst->strides[0];
    stride2_cbuf = dst->strides[1];
    stride0_gm = src->strides[2];
    stride1_gm = src->strides[0];
    stride2_gm = src->strides[1];
  }

  // throw dim0 as loop and map (dim1, dim2, 1) to the new 3d pattern to
  // guarantee the last axis is contiguous.
  for (int64_t i = 0; i < size0; i++) {
    int64_t offset_loop_gm = stride0_gm * i;
    int64_t offset_loop_cbuf = stride0_cbuf * i;

    memref_t<__gm__ T, 3> gm_3d = {src->allocated,
                                   src->aligned,
                                   src->offset + offset_loop_gm,
                                   {size1, size2, 1},
                                   {stride1_gm, stride2_gm, 1}};
    memref_t<__cbuf__ T, 3> cbuf_3d = {dst->allocated,
                                       dst->aligned,
                                       dst->offset + offset_loop_cbuf,
                                       {size1, size2, 1},
                                       {stride1_cbuf, stride2_cbuf, 1}};
    load_gm_to_cbuf_3d_core_with_contiguous_last_dim(&gm_3d, &cbuf_3d,
                                                     pad_mode);
  }
}

extern "C" {
//===-------------------------------------------------------------------===//
// Load gm to cbuf, 3 dim
//===-------------------------------------------------------------------===//
REGISTE_DMA_L1_LOAD(3, int8_t);
REGISTE_DMA_L1_LOAD(3, uint8_t);
REGISTE_DMA_L1_LOAD(3, int16_t);
REGISTE_DMA_L1_LOAD(3, uint16_t);
REGISTE_DMA_L1_LOAD(3, int32_t);
REGISTE_DMA_L1_LOAD(3, uint32_t);
REGISTE_DMA_L1_LOAD(3, int64_t);
REGISTE_DMA_L1_LOAD(3, uint64_t);
REGISTE_DMA_L1_LOAD(3, half);
REGISTE_DMA_L1_LOAD(3, float);
REGISTE_DMA_L1_LOAD(3, bfloat16_t);
}