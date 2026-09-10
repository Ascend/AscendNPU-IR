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
                                 memref_t<__cbuf__ T, 4> *l1,
                                 int32_t l2_cache_mode = 0) {

  auto gm_ptr = gm->aligned + gm->offset;
  auto l1_ptr = l1->aligned + l1->offset;

  int64_t n_tile_actual = gm->sizes[0];
  int64_t d_tile_actual = gm->sizes[1];
  // A fully clipped tile (e.g. a masked load whose whole extent lies beyond
  // the tensor) yields a zero-size subview. Issuing the intrinsic with
  // nValue == 0 is not a guaranteed no-op on all archs, and the source base
  // address may already be out of bounds, causing an OOB gm read.
  if (is_no_op<2>(gm->sizes)) {
    return;
  }
  int64_t d_val = gm->strides[0];
  // TODO: Remove for bias when fix bias infer layout.
  int64_t n_tile_ceil =
      isForBias || (sizeof(T) == 1 && d_tile_actual < 32)
          ? 1
          : l1->strides[0] / l1->strides[2];
  int64_t c0_size = INTR_BYTES_PER_BLOCK / sizeof(T);
  auto l2_ctl = static_cast<uint8_t>(l2_cache_mode);

  if (gm->strides[0] < MAX_LEN_UNIT16) {
    copy_gm_to_cbuf_intrin_core(nd2nz_intrin_args<T>{
        l1_ptr, gm_ptr, 0, 1, static_cast<uint16_t>(n_tile_actual),
        static_cast<uint16_t>(d_tile_actual), 0, static_cast<uint16_t>(d_val),
        static_cast<uint16_t>(n_tile_ceil), 1, 1,
        static_cast<uint16_t>(c0_size), l2_ctl});
  } else {
    for (int64_t i = 0; i < n_tile_actual; i++) {
      copy_gm_to_cbuf_intrin_core(
          nd2nz_intrin_args<T>{l1_ptr + i * c0_size, gm_ptr + i * d_val, 0, 1,
                               1, static_cast<uint16_t>(d_tile_actual), 0, 0,
                               static_cast<uint16_t>(n_tile_ceil), 0, 1,
                               static_cast<uint16_t>(c0_size), l2_ctl});
    }
  }
}

/// gm (b, n, d) -> l1 (b, n1, d1, d0, n0). A single MTE2 descriptor covers the
/// whole batch through ndNum, which is what keeps GM->L1 request count from
/// scaling with the batch size.
template <typename T>
__aicore__ __attribute__((always_inline)) void
copy_gm_to_cbuf_batch_nd2nz_core(memref_t<__gm__ T, 3> *gm,
                                 memref_t<__cbuf__ T, 5> *l1,
                                 int32_t l2_cache_mode = 0) {
  auto gm_ptr = gm->aligned + gm->offset;
  auto l1_ptr = l1->aligned + l1->offset;

  int64_t batch = gm->sizes[0];
  int64_t src_nd_stride = gm->strides[0];
  int64_t dst_nz_stride = l1->strides[0];
  int64_t n_tile_actual = gm->sizes[1];
  int64_t d_tile_actual = gm->sizes[2];
  int64_t d_val = gm->strides[1];
  int64_t n_tile_ceil = l1->strides[1] / l1->strides[3];
  int64_t c0_size = INTR_BYTES_PER_BLOCK / sizeof(T);
  auto l2_ctl = static_cast<uint8_t>(l2_cache_mode);

  // Both matrix strides pass through a 16-bit field as an element count and
  // then reach the hardware converted -- the source one to bytes, the
  // destination one to C0 blocks -- so each needs bounding twice.
  if (batch < MAX_LEN_UNIT16 && d_val < MAX_LEN_UNIT16 &&
      src_nd_stride < MAX_LEN_UNIT16 &&
      src_nd_stride * static_cast<int64_t>(sizeof(T)) < MAX_LEN_UNIT16 &&
      dst_nz_stride < MAX_LEN_UNIT16 &&
      dst_nz_stride / c0_size < MAX_LEN_UNIT16) {
    copy_gm_to_cbuf_intrin_core(nd2nz_intrin_args<T>{
        l1_ptr, gm_ptr, 0, static_cast<uint16_t>(batch),
        static_cast<uint16_t>(n_tile_actual),
        static_cast<uint16_t>(d_tile_actual),
        static_cast<uint16_t>(src_nd_stride), static_cast<uint16_t>(d_val),
        static_cast<uint16_t>(n_tile_ceil), 1,
        static_cast<uint16_t>(dst_nz_stride),
        static_cast<uint16_t>(c0_size), l2_ctl});
    return;
  }

  for (int64_t b = 0; b < batch; b++) {
    memref_t<__gm__ T, 2> gm_2d = {gm->allocated,
                                   gm->aligned,
                                   gm->offset + b * src_nd_stride,
                                   {n_tile_actual, d_tile_actual},
                                   {d_val, gm->strides[2]}};
    memref_t<__cbuf__ T, 4> l1_4d = {
        l1->allocated,
        l1->aligned,
        l1->offset + b * dst_nz_stride,
        {l1->sizes[1], l1->sizes[2], l1->sizes[3], l1->sizes[4]},
        {l1->strides[1], l1->strides[2], l1->strides[3], l1->strides[4]}};
    copy_gm_to_cbuf_multi_nd2nz_core<T, false>(&gm_2d, &l1_4d, l2_cache_mode);
  }
}

extern "C" {
REGISTE_ND2NZ_BATCH(gm, cbuf, 3, 5, half);
REGISTE_ND2NZ_BATCH(gm, cbuf, 3, 5, float);
REGISTE_ND2NZ_BATCH(gm, cbuf, 3, 5, bfloat16_t);
REGISTE_ND2NZ_BATCH(gm, cbuf, 3, 5, int32_t);
REGISTE_ND2NZ_BATCH(gm, cbuf, 3, 5, uint32_t);
REGISTE_ND2NZ_BATCH(gm, cbuf, 3, 5, int16_t);
REGISTE_ND2NZ_BATCH(gm, cbuf, 3, 5, uint16_t);
REGISTE_ND2NZ_BATCH(gm, cbuf, 3, 5, int8_t);
REGISTE_ND2NZ_BATCH(gm, cbuf, 3, 5, uint8_t);

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
