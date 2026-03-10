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
check_inputs_of_load_gm_to_cbuf_1d_core(memref_t<__gm__ T, 1> *src,
                                        memref_t<__cbuf__ T, 1> *dst) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  auto dst_ptr = dst->aligned + dst->offset;
  const int64_t stride0_cbuf = dst->strides[0];
  assert(isAddress32ByteAligned(dst_ptr) &&
         "The starting address of dst must be 32byte aligned.");
  assert((isSizeAlignedToBlock<T>(stride0_cbuf) || stride0_cbuf == 1) &&
         "The dst strides must be 1 or aligned to block.");
#endif
}

// core func of load gm <-> l1, 1d
template <typename T>
__aicore__ __attribute__((always_inline)) void
load_gm_to_cbuf_1d_core(memref_t<__gm__ T, 1> *src,
                        memref_t<__cbuf__ T, 1> *dst,
                        pad_t pad_mode = pad_t::PAD_NONE) {
  if (is_no_op<1>(src->sizes)) {
    return;
  }

  // Input parameter constraints assert.
  check_inputs_of_load_gm_to_cbuf_1d_core(src, dst);

  const int64_t stride0_gm = src->strides[0];
  const int64_t stride0_cbuf = dst->strides[0];
  if (stride0_gm == 1 && stride0_cbuf == 1) [[likely]] {
    // last dimension is contiguous
    load_gm_to_cbuf_1d_core_with_contiguous_last_dim<T>(src, dst, pad_mode);
    return;
  }

  // last dimension is not contiguous,
  // view the src (size) with stride [stride] as viewed_src (size, 1) with
  // stride [stride, 1], where last dimension of viewed_src is contiguous
  const int64_t size0 = src->sizes[0];
  memref_t<__gm__ T, 2> gm_2d = {
      src->allocated, src->aligned, src->offset, {size0, 1}, {stride0_gm, 1}};
  memref_t<__cbuf__ T, 2> cbuf_2d = {
      dst->allocated, dst->aligned, dst->offset, {size0, 1}, {stride0_cbuf, 1}};

  load_gm_to_cbuf_2d_core_with_contiguous_last_dim<T>(&gm_2d, &cbuf_2d,
                                                      pad_mode);
}

extern "C" {
//===-------------------------------------------------------------------===//
// Load gm to cbuf, 1 dim
//===-------------------------------------------------------------------===//
REGISTE_DMA_L1_LOAD(1, int8_t);
REGISTE_DMA_L1_LOAD(1, uint8_t);
REGISTE_DMA_L1_LOAD(1, int16_t);
REGISTE_DMA_L1_LOAD(1, uint16_t);
REGISTE_DMA_L1_LOAD(1, int32_t);
REGISTE_DMA_L1_LOAD(1, uint32_t);
REGISTE_DMA_L1_LOAD(1, int64_t);
REGISTE_DMA_L1_LOAD(1, uint64_t);
REGISTE_DMA_L1_LOAD(1, half);
REGISTE_DMA_L1_LOAD(1, float);
REGISTE_DMA_L1_LOAD(1, bfloat16_t);
}