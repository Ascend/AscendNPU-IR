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

#include "DMA/ND2NZ.h"
#include "Vector/VecUtils.h"
/// gm (n, d) -> l1 (n1, d1, d0, n0) , where n0*sizeof(T) = 32B, d0 = 16
template <typename T, bool isForBias>
__aicore__ __attribute__((always_inline)) void
copy_gm_to_cbuf_multi_nd2nz_core(memref_t<__gm__ T, 2> *gm,
                                 memref_t<__cbuf__ T, 4> *l1) {

  auto gm_ptr = gm->aligned + gm->offset;
  auto l1_ptr = l1->aligned + l1->offset;

  int64_t n_tile_actual = gm->sizes[0];
  int64_t d_tile_actual = gm->sizes[1];
  int64_t d_val = gm->strides[0];
  // TODO: Remove for bias when fix bias infer layout.
  int64_t n_tile_ceil =
      isForBias || (sizeof(T) == 1 && d_tile_actual < 32)
          ? 1
          : l1->strides[0] / l1->strides[2];
  int64_t c0_size = INTR_BYTES_PER_BLOCK / sizeof(T);

  if (gm->strides[0] < MAX_LEN_UNIT16) {
    copy_gm_to_cbuf_intrin_core(nd2nz_intrin_args<T>{
        l1_ptr, gm_ptr, 0, 1, static_cast<uint16_t>(n_tile_actual),
        static_cast<uint16_t>(d_tile_actual), 0,
        static_cast<uint16_t>(d_val), static_cast<uint16_t>(n_tile_ceil), 1, 1});
  } else {
    for (int64_t i = 0; i < n_tile_actual; i++) {
      copy_gm_to_cbuf_intrin_core(nd2nz_intrin_args<T>{
          l1_ptr + i * c0_size, gm_ptr + i * d_val, 0, 1, 1,
          static_cast<uint16_t>(d_tile_actual), 0, 0, static_cast<uint16_t>(n_tile_ceil), 0, 1});
    }
  }
}

extern "C" {
REGISTE_ND2NZ(gm, cbuf, 2, 4, half);
REGISTE_ND2NZ(gm, cbuf, 2, 4, float);
REGISTE_ND2NZ(gm, cbuf, 2, 4, bfloat16_t);
REGISTE_ND2NZ(gm, cbuf, 2, 4, int32_t);
REGISTE_ND2NZ(gm, cbuf, 2, 4, uint32_t);
REGISTE_ND2NZ(gm, cbuf, 2, 4, int16_t);
REGISTE_ND2NZ(gm, cbuf, 2, 4, uint16_t);
REGISTE_ND2NZ(gm, cbuf, 2, 4, int8_t);
REGISTE_ND2NZ(gm, cbuf, 2, 4, uint8_t);
REGISTE_ND2NZ_FORBIAS(gm, cbuf, 2, 4, half);
REGISTE_ND2NZ_FORBIAS(gm, cbuf, 2, 4, float);
REGISTE_ND2NZ_FORBIAS(gm, cbuf, 2, 4, bfloat16_t);
REGISTE_ND2NZ_FORBIAS(gm, cbuf, 2, 4, int32_t);
REGISTE_ND2NZ_FORBIAS(gm, cbuf, 2, 4, uint32_t);
REGISTE_ND2NZ_FORBIAS(gm, cbuf, 2, 4, int16_t);
REGISTE_ND2NZ_FORBIAS(gm, cbuf, 2, 4, uint16_t);
REGISTE_ND2NZ_FORBIAS(gm, cbuf, 2, 4, int8_t);
REGISTE_ND2NZ_FORBIAS(gm, cbuf, 2, 4, uint8_t);
}