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

#include "DMA/SET2D.h"
#include "Vector/VecUtils.h"

template <typename T>
__aicore__ __attribute__((always_inline)) void
check_inputs_of_set_cbuf_to_2d_core(memref_t<__cbuf__ T, 1> *l1) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  assert((l1->sizes[0] % 2 == 0) && "The tilling size must be even");
  const int64_t stride0 = l1->strides[0];
  auto l1_ptr = l1->aligned + l1->offset;
  assert(isAddress32ByteAligned(l1_ptr) &&
         "The starting address of l1 must be 32byte aligned.");
  assert(stride0 == 1 && "The l1 strides must be 1 or aligned to block.");
#endif
}

template <typename T>
__aicore__ __attribute__((always_inline)) void
set_cbuf_to_2d_core(T scalar, memref_t<__cbuf__ T, 1> *l1) {
  check_inputs_of_set_cbuf_to_2d_core(l1);
  auto l1_ptr = l1->aligned + l1->offset;

  // current for 1d set 2d, repeat time for the outer loop is 1. Hence there is
  // no the set of repeat gap
  uint64_t repeat_time = 1;
  uint64_t block_num = CEIL_DIV(sizeof(T) * l1->sizes[0], L1_ALIGN_BYTES);
  int64_t config = (uint64_t)(repeat_time) | (uint64_t)(block_num << 16);
  set_cbuf_to_2d_intrin_core(set2d_intrin_args<T>{l1_ptr, config, scalar});
}

extern "C" {
REGISTE_SET2D(cbuf, 1, int8_t);
REGISTE_SET2D(cbuf, 1, half);
REGISTE_SET2D(cbuf, 1, float);
REGISTE_SET2D(cbuf, 1, bfloat16_t);
}