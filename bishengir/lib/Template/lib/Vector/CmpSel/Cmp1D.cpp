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

#include "Vector/Cast/CastUtils.h"
#include "Vector/CmpSel/CmpUtils.h"
#include "Vector/VecUtils.h"

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
vector_compare_vv_1d(memref_t<__ubuf__ T, 1> *src0,
                     memref_t<__ubuf__ T, 1> *src1,
                     memref_t<__ubuf__ bool, 1> *dst) {
  // Input parameter constraints assert.
  check_inputs_of_vector_eltwise_vv_1d<T, bool>(src0, src1, dst);
  const int64_t src_total_size = src0->sizes[0];

  // view dst as uint8 because dst type of vcmp instr is uint8
  memref_t<__ubuf__ uint8_t, 1> dst_as_uint8;
  view_as<bool, uint8_t, 1>(dst, &dst_as_uint8);

  __ubuf__ uint8_t *dst_ptr = dst_as_uint8.aligned + dst_as_uint8.offset;
  __ubuf__ T *src0_ptr = src0->aligned + src0->offset;
  __ubuf__ T *src1_ptr = src1->aligned + src1->offset;

  constexpr int src_bytes = sizeof(T);
  constexpr int num_per_repeat = INTR_BYTES_PER_REPEAT / src_bytes;
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / src_bytes;
  constexpr int max_repeat_times = get_vcmp_max_repeat_cnts<T>();

  uint64_t repeat_times = CEIL_DIV(src_total_size, num_per_repeat);
  uint16_t src_repeat_stride = num_per_repeat / num_per_block;

  if (repeat_times >= max_repeat_times) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
    assert(repeat_times / max_repeat_times < 2 && "ub overflow");
#endif

    vector_eltwise_vv_intrin<OP, T>(
        intrin_args<2, T, uint8_t>{dst_ptr,
                                   {src0_ptr, src1_ptr},
                                   0,
                                   max_repeat_times,
                                   1,      // dst_block_stride
                                   {1, 1}, // src_block_stride
                                   0,      // dst_repeat_stride
                                   {src_repeat_stride, src_repeat_stride}});
  }

  if (repeat_times % max_repeat_times != 0) {
    uint64_t main_repeat_times = FLOOR_FACTOR(repeat_times, max_repeat_times);
    uint64_t remain_repeat_times = repeat_times % max_repeat_times;
    uint64_t dst_offset = num_per_repeat / BITS_PER_BYTE * main_repeat_times;
    uint64_t src_offset = num_per_repeat * main_repeat_times;

    vector_eltwise_vv_intrin<OP, T>(intrin_args<2, T, uint8_t>{
        dst_ptr + dst_offset,
        {src0_ptr + src_offset, src1_ptr + src_offset},
        0,
        remain_repeat_times,
        1,      // dst_block_stride
        {1, 1}, // src_block_stride
        0,      // dst_repeat_stride
        {src_repeat_stride, src_repeat_stride}});
  }
}

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
vector_compare_vs_1d(memref_t<__ubuf__ T, 1> *src0, T scalar,
                     memref_t<__ubuf__ bool, 1> *dst) {
  check_inputs_of_vector_eltwise_v_1d<T, bool>(src0, dst);
  const int64_t src_total_size = src0->sizes[0];

  // view dst as uint8 because dst type of vcmp instr is uint8
  memref_t<__ubuf__ uint8_t, 1> dst_as_uint8;
  view_as<bool, uint8_t, 1>(dst, &dst_as_uint8);

  __ubuf__ uint8_t *dst_ptr = dst_as_uint8.aligned + dst_as_uint8.offset;
  __ubuf__ T *src0_ptr = src0->aligned + src0->offset;

  constexpr int src_bytes = sizeof(T);
  constexpr int num_per_repeat = INTR_BYTES_PER_REPEAT / src_bytes;
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / src_bytes;
  constexpr int max_repeat_times = get_vcmp_max_repeat_cnts<T>();

  uint64_t repeat_times = CEIL_DIV(src_total_size, num_per_repeat);
  uint16_t src_repeat_stride = num_per_repeat / num_per_block;

  if (repeat_times >= max_repeat_times) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
    assert(repeat_times / max_repeat_times < 2 && "ub overflow");
#endif

    vector_eltwise_vs_intrin<OP, T>(
        intrin_args<1, T, uint8_t>{dst_ptr,
                                   {src0_ptr},
                                   scalar,
                                   max_repeat_times,
                                   1,   // dst_block_stride
                                   {1}, // src_block_stride
                                   0,   // dst_repeat_stride
                                   {src_repeat_stride}});
  }

  if (repeat_times % max_repeat_times != 0) {
    uint64_t main_repeat_times = FLOOR_FACTOR(repeat_times, max_repeat_times);
    uint64_t remain_repeat_times = repeat_times % max_repeat_times;
    uint64_t dst_offset = num_per_repeat / BITS_PER_BYTE * main_repeat_times;
    uint64_t src_offset = num_per_repeat * main_repeat_times;

    vector_eltwise_vs_intrin<OP, T>(
        intrin_args<1, T, uint8_t>{dst_ptr + dst_offset,
                                   {src0_ptr + src_offset},
                                   scalar,
                                   remain_repeat_times,
                                   1,   // dst_block_stride
                                   {1}, // src_block_stride
                                   0,   // dst_repeat_stride
                                   {src_repeat_stride}});
  }
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_compare_vv_1d<VectorOpTy::VCMP_NE, int32_t>(
    memref_t<__ubuf__ int32_t, 1> *src0, memref_t<__ubuf__ int32_t, 1> *src1,
    memref_t<__ubuf__ bool, 1> *dst) {
  /// because vcmpv instruction not support ne int32,
  /// step 1: dst = vcmpv_eq(src0 , src1)
  /// step 2: dst = vnot(dst)
  vector_compare_vv_1d<VectorOpTy::VCMP_EQ, int32_t>(src0, src1, dst);
  INTRINSIC(pipe_barrier, PIPE_V);
  vector_eltwise_v_1d<VectorOpTy::VNOT, bool>(dst, dst);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_compare_vs_1d<VectorOpTy::VCMPS_NE, int32_t>(
    memref_t<__ubuf__ int32_t, 1> *src0, int32_t scalar,
    memref_t<__ubuf__ bool, 1> *dst) {
  /// because vcmpvs instruction not support ne int32,
  /// step 1: dst = vcmpvs_eq(src0 , src1)
  /// step 2: dst = vnot(dst)
  vector_compare_vs_1d<VectorOpTy::VCMPS_EQ, int32_t>(src0, scalar, dst);
  INTRINSIC(pipe_barrier, PIPE_V);
  vector_eltwise_v_1d<VectorOpTy::VNOT, bool>(dst, dst);
}

extern "C" {
//===-------------------------------------------------------------------===//
// vcmpv, 1 dim
//===-------------------------------------------------------------------===//
REGISTE_CMP_VV(eq, VectorOpTy::VCMP_EQ, 1, half)
REGISTE_CMP_VV(eq, VectorOpTy::VCMP_EQ, 1, float)
REGISTE_CMP_VV(eq, VectorOpTy::VCMP_EQ, 1, int32_t)

REGISTE_CMP_VV(ne, VectorOpTy::VCMP_NE, 1, half)
REGISTE_CMP_VV(ne, VectorOpTy::VCMP_NE, 1, float)
REGISTE_CMP_VV(ne, VectorOpTy::VCMP_NE, 1, int32_t)

REGISTE_CMP_VV(lt, VectorOpTy::VCMP_LT, 1, half)
REGISTE_CMP_VV(lt, VectorOpTy::VCMP_LT, 1, float)

REGISTE_CMP_VV(gt, VectorOpTy::VCMP_GT, 1, half)
REGISTE_CMP_VV(gt, VectorOpTy::VCMP_GT, 1, float)

REGISTE_CMP_VV(ge, VectorOpTy::VCMP_GE, 1, half)
REGISTE_CMP_VV(ge, VectorOpTy::VCMP_GE, 1, float)

REGISTE_CMP_VV(le, VectorOpTy::VCMP_LE, 1, half)
REGISTE_CMP_VV(le, VectorOpTy::VCMP_LE, 1, float)

//===-------------------------------------------------------------------===//
// vcmpvs, 1 dim
//===-------------------------------------------------------------------===//
REGISTE_CMP_VS(eq, VectorOpTy::VCMPS_EQ, 1, half)
REGISTE_CMP_VS(eq, VectorOpTy::VCMPS_EQ, 1, float)
REGISTE_CMP_VS(eq, VectorOpTy::VCMPS_EQ, 1, int32_t)

REGISTE_CMP_VS(ne, VectorOpTy::VCMPS_NE, 1, half)
REGISTE_CMP_VS(ne, VectorOpTy::VCMPS_NE, 1, float)
REGISTE_CMP_VS(ne, VectorOpTy::VCMPS_NE, 1, int32_t)

REGISTE_CMP_VS(lt, VectorOpTy::VCMPS_LT, 1, half)
REGISTE_CMP_VS(lt, VectorOpTy::VCMPS_LT, 1, float)

REGISTE_CMP_VS(gt, VectorOpTy::VCMPS_GT, 1, half)
REGISTE_CMP_VS(gt, VectorOpTy::VCMPS_GT, 1, float)

REGISTE_CMP_VS(ge, VectorOpTy::VCMPS_GE, 1, half)
REGISTE_CMP_VS(ge, VectorOpTy::VCMPS_GE, 1, float)

REGISTE_CMP_VS(le, VectorOpTy::VCMPS_LE, 1, half)
REGISTE_CMP_VS(le, VectorOpTy::VCMPS_LE, 1, float)
}