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

#ifndef BISHENGIR_LIB_TEMPLATE_INCLUDE_CAST_UTILS_H
#define BISHENGIR_LIB_TEMPLATE_INCLUDE_CAST_UTILS_H

#include "DMA/DMAUtils.h"
#include "Utils.h"
#include "Vector/Broadcast/BrcUtils.h"
#include "Vector/Transpose/TransposeUtils.h"
#include "Vector/VecUtils.h"
#include <type_traits>

constexpr uint32_t CTRL_REGISTER_BIT = 59;
constexpr int32_t INTR_TRANSPOSE_PER_REPEAT_BYTS_FOR_I8 = 1024;

enum class CastMode : uint32_t {
  R = 0,
  A = 1,
  F = 2,
  C = 3,
  Z = 4,
  O = 5,
  OVERFLOW = 6,
};

/// Convert buf_2d (x0, x1) to buf_1d (x1)
template <typename T>
__aiv__ __attribute__((always_inline)) void
convert_2d_to_1d_args(memref_t<__ubuf__ T, 2> *buf_2d,
                      memref_t<__ubuf__ T, 1> *buf_1d) {
  buf_1d->allocated = buf_2d->allocated;
  buf_1d->aligned = buf_2d->aligned;
  buf_1d->offset = buf_2d->offset;

  buf_1d->sizes[0] = buf_2d->sizes[1];
  buf_1d->strides[0] = 1;
}

template <typename SRC_T, typename DST_T>
__aiv__ __attribute__((always_inline)) void
vconv(intrin_args<1, SRC_T, DST_T> args, CastMode cast_mode);

template <typename SRC_T, typename DST_T, bool DISABLE_SIZE_ALIGN>
__aiv__ __attribute__((always_inline)) void
vector_cast_with_overflow(memref_t<__ubuf__ SRC_T, 2> *src,
                          memref_t<__ubuf__ DST_T, 2> *dst,
                          memref_t<__ubuf__ SRC_T, 1> *tmp) {
  static_assert(sizeof(SRC_T) > sizeof(DST_T) &&
                "sizeof src dtype must large than dst dtype");
  constexpr int num_per_block_dst = INTR_BYTES_PER_BLOCK / sizeof(DST_T);
  constexpr int num_per_block_src = INTR_BYTES_PER_BLOCK / sizeof(SRC_T);
  auto bits_factor = bitwidthOf<SRC_T>() / bitwidthOf<DST_T>();
  memref_t<__ubuf__ SRC_T, 2> tmp_2d_src_t;
  int64_t tmp_head_offset = 0;
  int64_t tmp_tail_offset = 0;
  auto numerator = tmp->sizes[0] * bits_factor;
  // DISABLE_SIZE_ALIGN == true: unaligned src/dst; tmp = src copy + intermediates (split n/3, 2n/3);
  // DISABLE_SIZE_ALIGN == false: src aligned; tmp only for intermediates (split n/2).
  if constexpr (DISABLE_SIZE_ALIGN) {
    tmp_head_offset = numerator / 3;
    tmp_tail_offset = numerator * 2 / 3;
    // step0: copy src to tmp.
    auto src_size0 = src->sizes[0];
    auto src_size1 = src->sizes[1];
    tmp_2d_src_t = {
      tmp->aligned,
      tmp->allocated,
      tmp->offset,
      {src_size0, src_size1},
      {CEIL_FACTOR(src_size1, num_per_block_src), 1}};
    
    copy_ubuf_to_ubuf_2d_core(src, &tmp_2d_src_t);
    INTRINSIC(pipe_barrier, PIPE_V);
  } else {
    tmp_tail_offset = numerator / 2;
    tmp_2d_src_t = {
      src->aligned,
      src->allocated,
      src->offset,
      {src->sizes[0], src->sizes[1]},
      {src->strides[0], src->strides[1]}};
  }
  
  // step1: Transpose to tmp.
  memref_t<__ubuf__ DST_T, 2> src_as_dst_t;
  view_as<SRC_T, DST_T, 2>(&tmp_2d_src_t, &src_as_dst_t);
  auto src_as_dst_t_size0 = src_as_dst_t.sizes[0];
  auto src_as_dst_t_size1 = src_as_dst_t.sizes[1];

  memref_t<__ubuf__ DST_T, 1> tmp_as_dst_t;
  view_as<SRC_T, DST_T, 1>(tmp, &tmp_as_dst_t);
  memref_t<__ubuf__ DST_T, 2> tmp_2d_as_dst_t{
      tmp_as_dst_t.aligned,
      tmp_as_dst_t.allocated,
      tmp_as_dst_t.offset + tmp_head_offset,
      {src_as_dst_t_size1, src_as_dst_t_size0},
      {CEIL_FACTOR(src_as_dst_t_size0, num_per_block_dst), 1}};
  vnchwconv_2d(&src_as_dst_t, &tmp_2d_as_dst_t);

  // step2: Take out the low-order results through copy_ubuf_to_ubuf.
  __ubuf__ DST_T *tmp_2d_as_dst_t_ptr =
      tmp_2d_as_dst_t.aligned + tmp_2d_as_dst_t.offset;
  auto lenBurst = CEIL_DIV(src_as_dst_t_size0, num_per_block_dst);
  INTRINSIC(pipe_barrier, PIPE_V);
  INTRINSIC(copy_ubuf_to_ubuf,
            tmp_2d_as_dst_t_ptr,                 // src
            tmp_2d_as_dst_t_ptr,                 // dst
            0,                                   // sid
            src_as_dst_t_size1 / bits_factor,    // nburst
            lenBurst,                            // lenBurst
            (bits_factor - 1) * lenBurst,        // srcGap
            0);                                  // dstGap

  tmp_2d_as_dst_t.sizes[0] = src_as_dst_t_size1 / bits_factor;
  if (dst->strides[0] != CEIL_FACTOR(dst->sizes[1], num_per_block_dst)) {
    // step3: Transpose tmp to tmp.
    // The second half of tmp is used to store temporary tranpose results.
    memref_t<__ubuf__ DST_T, 2> tmp_for_transpose{
        tmp_as_dst_t.aligned,
        tmp_as_dst_t.allocated,
        tmp_as_dst_t.offset + tmp_tail_offset,
        {dst->sizes[0], dst->sizes[1]},
        {CEIL_FACTOR(dst->sizes[1], num_per_block_dst), 1}};
    INTRINSIC(pipe_barrier, PIPE_V);
    vnchwconv_2d(&tmp_2d_as_dst_t, &tmp_for_transpose);

    // step4: copy tmp to dst.
    INTRINSIC(pipe_barrier, PIPE_V);
    tmp_for_transpose.sizes[0] = dst->sizes[0];
    tmp_for_transpose.sizes[1] = dst->sizes[1];
    copy_ubuf_to_ubuf_2d_core_with_contiguous_last_dim(&tmp_for_transpose, dst);
    return;
  }
  // step3: Transpose tmp to dst.
  INTRINSIC(pipe_barrier, PIPE_V);
  vnchwconv_2d(&tmp_2d_as_dst_t, dst);
}

// When cast, the display will flip after exceeding the maximum display range.
__aiv__ __attribute__((always_inline)) void set_saturation_ctrl();

// When cast, the display will not be flipped after exceeding the maximum
// display range.
__aiv__ __attribute__((always_inline)) void set_saturation_ctrl_none();

template <typename SRC_T, typename DST_T>
__aiv__ __attribute__((always_inline)) void
vector_cast_1d_with_temp(memref_t<__ubuf__ SRC_T, 1> *src,
                         memref_t<__ubuf__ DST_T, 1> *dst,
                         memref_t<__ubuf__ DST_T, 1> *temp);

template <typename SRC_T, typename DST_T, bool DISABLE_SIZE_ALIGN>
__aiv__ __attribute__((always_inline)) void
vector_cast_1d_with_overflow(memref_t<__ubuf__ SRC_T, 1> *src,
                             memref_t<__ubuf__ DST_T, 1> *dst,
                             memref_t<__ubuf__ SRC_T, 1> *tmp);

#define REGISTE_CAST_1D_WITH_TEMP_CORE(src_type, dst_type)                     \
  template __aiv__ __attribute__((always_inline)) void                         \
  vector_cast_1d_with_temp(memref_t<__ubuf__ src_type, 1> *src,                \
                           memref_t<__ubuf__ dst_type, 1> *dst,                \
                           memref_t<__ubuf__ dst_type, 1> *temp)

template <typename SRC_T, typename DST_T>
__aiv__ __attribute__((always_inline)) void
vector_cast_2d_with_temp(memref_t<__ubuf__ SRC_T, 2> *src,
                         memref_t<__ubuf__ DST_T, 2> *dst,
                         memref_t<__ubuf__ DST_T, 1> *temp);

template <typename SRC_T, typename DST_T, bool DISABLE_SIZE_ALIGN>
__aiv__ __attribute__((always_inline)) void
vector_cast_2d_with_overflow(memref_t<__ubuf__ SRC_T, 2> *src,
                             memref_t<__ubuf__ DST_T, 2> *dst,
                             memref_t<__ubuf__ SRC_T, 1> *tmp);

#define DECLARE_VCONV(src_type, dst_type)                                      \
  template __aiv__ __attribute__((always_inline)) void vconv(                  \
      intrin_args<1, src_type, dst_type> args, CastMode cast_mode)

template <typename SRC_T, typename DST_T>
__aiv__ __attribute__((always_inline)) void
vector_cast_1d_with_mode(memref_t<__ubuf__ SRC_T, 1> *src,
                         memref_t<__ubuf__ DST_T, 1> *dst, CastMode cast_mode);

#define DECLARE_CAST(src_type, dst_type, dim)                                  \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_vcast_##src_type##_to_##dst_type##_##dim##d_with_mode(      \
          memref_t<__ubuf__ src_type, dim> *src,                               \
          memref_t<__ubuf__ dst_type, dim> *dst, CastMode cast_mode)

// Two symbols per overflow variant: size_align (DISABLE_SIZE_ALIGN=false) and
// no_size_align (DISABLE_SIZE_ALIGN=true), no runtime bool param.
#define DECLARE_CAST_OVERFLOW_SIZE_ALIGN(src_type, dst_type, dim)             \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_vcast_##src_type##_to_##dst_type##_##dim##d_with_overflow(  \
          memref_t<__ubuf__ src_type, dim> *src,                               \
          memref_t<__ubuf__ dst_type, dim> *dst,                               \
          memref_t<__ubuf__ src_type, 1> *tmp, CastMode cast_mode)

#define DECLARE_CAST_OVERFLOW_NO_SIZE_ALIGN(src_type, dst_type, dim)                        \
  __aiv__ __attribute__((always_inline)) void                                               \
      _mlir_ciface_vcast_##src_type##_to_##dst_type##_##dim##d_with_overflow_no_size_align( \
          memref_t<__ubuf__ src_type, dim> *src,                                            \
          memref_t<__ubuf__ dst_type, dim> *dst,                                            \
          memref_t<__ubuf__ src_type, 1> *tmp, CastMode cast_mode)

#define DECLARE_CAST_OVERFLOW(src_type, dst_type, dim)                         \
  DECLARE_CAST_OVERFLOW_SIZE_ALIGN(src_type, dst_type, dim);                   \
  DECLARE_CAST_OVERFLOW_NO_SIZE_ALIGN(src_type, dst_type, dim)

#define REGISTE_CAST(src_type, dst_type, dim)                                  \
  DECLARE_CAST(src_type, dst_type, dim) {                                      \
    vector_cast_##dim##d_with_mode<src_type, dst_type>(src, dst, cast_mode);   \
  }

#define REGISTE_CAST_OVERFLOW_SIZE_ALIGN(src_type, dst_type, dim)              \
  DECLARE_CAST_OVERFLOW_SIZE_ALIGN(src_type, dst_type, dim) {                  \
    vector_cast_##dim##d_with_overflow<src_type, dst_type, false>(src, dst,    \
                                                                  tmp);        \
  }

#define REGISTE_CAST_OVERFLOW_NO_SIZE_ALIGN(src_type, dst_type, dim)           \
  DECLARE_CAST_OVERFLOW_NO_SIZE_ALIGN(src_type, dst_type, dim) {               \
    vector_cast_##dim##d_with_overflow<src_type, dst_type, true>(src, dst,     \
                                                                 tmp);         \
  }

#define REGISTE_CAST_OVERFLOW(src_type, dst_type, dim)                         \
  REGISTE_CAST_OVERFLOW_SIZE_ALIGN(src_type, dst_type, dim);                   \
  REGISTE_CAST_OVERFLOW_NO_SIZE_ALIGN(src_type, dst_type, dim)

#define DECLARE_CAST_WITH_TEMP(src_type, dst_type, dim)                        \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_vcast_##src_type##_to_##dst_type##_##dim##d_with_temp(      \
          memref_t<__ubuf__ src_type, dim> *src,                               \
          memref_t<__ubuf__ dst_type, dim> *dst,                               \
          memref_t<__ubuf__ dst_type, 1> *temp, CastMode cast_mode)

#define REGISTE_CAST_WITH_TEMP(src_type, dst_type, dim)                        \
  DECLARE_CAST_WITH_TEMP(src_type, dst_type, dim) {                            \
    vector_cast_##dim##d_with_temp<src_type, dst_type>(src, dst, temp);        \
  }

extern "C" {
//===-------------------------------------------------------------------===//
// cast, 1 dim
//===-------------------------------------------------------------------===//
// cast bool to half
DECLARE_CAST_WITH_TEMP(bool, half, 1);
// cast bool to float
DECLARE_CAST_WITH_TEMP(bool, float, 1);
// cast bool to bfloat16
DECLARE_CAST_WITH_TEMP(bool, bfloat16_t, 1);
// cast bool to int16
DECLARE_CAST_WITH_TEMP(bool, int16_t, 1);
// cast bool to int32
DECLARE_CAST_WITH_TEMP(bool, int32_t, 1);
// cast bool to uint16
DECLARE_CAST_WITH_TEMP(bool, uint16_t, 1);
// cast bool to uint32
DECLARE_CAST_WITH_TEMP(bool, uint32_t, 1);
// cast uint8_t to half
DECLARE_CAST_WITH_TEMP(uint8_t, half, 1);
// cast uint8_t to float
DECLARE_CAST_WITH_TEMP(uint8_t, float, 1);
// cast float to float
DECLARE_CAST(float, float, 1);
// cast float to half
DECLARE_CAST(float, half, 1);
// cast float to bfloat16_t
DECLARE_CAST(float, bfloat16_t, 1);
// cast float to int64
DECLARE_CAST(float, int64_t, 1);
// cast float to int32
DECLARE_CAST(float, int32_t, 1);
// cast float to int16
DECLARE_CAST(float, int16_t, 1);
// cast half to float
DECLARE_CAST(half, float, 1);
// cast half to int32
DECLARE_CAST(half, int32_t, 1);
// cast half to int16
DECLARE_CAST(half, int16_t, 1);
// cast half to int8
DECLARE_CAST(half, int8_t, 1);
// cast half to uint8
DECLARE_CAST(half, uint8_t, 1);
// cast bfloat16_t to float32
DECLARE_CAST(bfloat16_t, float, 1);
// cast bfloat16_t to int32
DECLARE_CAST(bfloat16_t, int32_t, 1);
// cast uint8 to float16
DECLARE_CAST(uint8_t, half, 1);
// cast int8 to float16
DECLARE_CAST(int8_t, half, 1);
// cast int16 to float16
DECLARE_CAST(int16_t, half, 1);
// cast int16 to float
DECLARE_CAST(int16_t, float, 1);
// cast int32 to float
DECLARE_CAST(int32_t, float, 1);
// cast int32 to int64
DECLARE_CAST(int32_t, int64_t, 1);
// cast int32 to int16
DECLARE_CAST(int32_t, int16_t, 1);
// cast int64 to int32
DECLARE_CAST(int64_t, int32_t, 1);
// cast int64 to float32
DECLARE_CAST(int64_t, float, 1);
// cast int32 to int8 overflow
DECLARE_CAST_OVERFLOW(int32_t, int8_t, 1);
// cast int16 to int8 overflow
DECLARE_CAST_OVERFLOW(int16_t, int8_t, 1);
// cast int32 to int16 overflow
DECLARE_CAST_OVERFLOW(int32_t, int16_t, 1);
// cast int64 to int32 overflow
DECLARE_CAST_OVERFLOW(int64_t, int32_t, 1);

//===-------------------------------------------------------------------===//
// cast, 2 dim
//===-------------------------------------------------------------------===//

// cast bool to float
DECLARE_CAST_WITH_TEMP(bool, half, 2);
// cast bool to float
DECLARE_CAST_WITH_TEMP(bool, float, 2);
// cast bool to bfloat16
DECLARE_CAST_WITH_TEMP(bool, bfloat16_t, 2);
// cast bool to int16
DECLARE_CAST_WITH_TEMP(bool, int16_t, 2);
// cast bool to int32
DECLARE_CAST_WITH_TEMP(bool, int32_t, 2);
// cast bool to uint16
DECLARE_CAST_WITH_TEMP(bool, uint16_t, 2);
// cast bool to uint32
DECLARE_CAST_WITH_TEMP(bool, uint32_t, 2);
// cast float to float
DECLARE_CAST(float, float, 2);
// cast float to half
DECLARE_CAST(float, half, 2);
// cast float to bfloat16_t
DECLARE_CAST(float, bfloat16_t, 2);
// cast float to int64
DECLARE_CAST(float, int64_t, 2);
// cast float to int32
DECLARE_CAST(float, int32_t, 2);
// cast float to int16
DECLARE_CAST(float, int16_t, 2);
// cast half to float
DECLARE_CAST(half, float, 2);
// cast half to int32
DECLARE_CAST(half, int32_t, 2);
// cast half to int16
DECLARE_CAST(half, int16_t, 2);
// cast half to int8
DECLARE_CAST(half, int8_t, 2);
// cast half to uint8
DECLARE_CAST(half, uint8_t, 2);
// cast bfloat16_t to float32
DECLARE_CAST(bfloat16_t, float, 2);
// cast bfloat16_t to int32
DECLARE_CAST(bfloat16_t, int32_t, 2);
// cast uint8 to float16
DECLARE_CAST(uint8_t, half, 2);
// cast int8 to float16
DECLARE_CAST(int8_t, half, 2);
// cast int16 to float16
DECLARE_CAST(int16_t, half, 2);
// cast int16 to float
DECLARE_CAST(int16_t, float, 2);
// cast int32 to float
DECLARE_CAST(int32_t, float, 2);
// cast int32 to int64
DECLARE_CAST(int32_t, int64_t, 2);
// cast int32 to int16
DECLARE_CAST(int32_t, int16_t, 2);
// cast int64 to int32
DECLARE_CAST(int64_t, int32_t, 2);
// cast int64 to float32
DECLARE_CAST(int64_t, float, 2);
// cast int32 to int8 overflow
DECLARE_CAST_OVERFLOW(int32_t, int8_t, 2);
// cast int16 to int8 overflow
DECLARE_CAST_OVERFLOW(int16_t, int8_t, 2);
// cast int64 to int32 overflow
DECLARE_CAST_OVERFLOW(int64_t, int32_t, 2);
// cast int32 to int16 overflow
DECLARE_CAST_OVERFLOW(int32_t, int16_t, 2);
}

#endif // BISHENGIR_LIB_TEMPLATE_INCLUDE_CAST_UTILS_H
