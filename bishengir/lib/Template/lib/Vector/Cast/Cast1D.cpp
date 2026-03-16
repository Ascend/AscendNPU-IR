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

#include "Vector/Broadcast/BrcUtils.h"
#include "Vector/Cast/CastUtils.h"

template <typename SRC_T, typename DST_T>
__aiv__ __attribute__((always_inline)) void
vconv(intrin_args<1, SRC_T, DST_T> args, CastMode cast_mode) {
#define CAST_ARGS                                                              \
  args.dst, args.src[0], args.repeat, args.dst_block_stride,                   \
      args.src_block_stride[0], args.dst_repeat_stride,                        \
      args.src_repeat_stride[0]

#define VCONV_POSTFIX_O(INTRINSIC_NAME, MODE, ...)                             \
  else if (MODE == CastMode::O) {                                              \
    INTRINSIC(INTRINSIC_NAME##o, __VA_ARGS__);                                 \
  }

#define VCONV_RAFCZ(INTRINSIC_NAME, MODE, ...)                                 \
  if (MODE == CastMode::R) {                                                   \
    INTRINSIC(INTRINSIC_NAME##r, __VA_ARGS__);                                 \
  } else if (MODE == CastMode::A) {                                            \
    INTRINSIC(INTRINSIC_NAME##a, __VA_ARGS__);                                 \
  } else if (MODE == CastMode::F) {                                            \
    INTRINSIC(INTRINSIC_NAME##f, __VA_ARGS__);                                 \
  } else if (MODE == CastMode::C) {                                            \
    INTRINSIC(INTRINSIC_NAME##c, __VA_ARGS__);                                 \
  } else if (MODE == CastMode::Z) {                                            \
    INTRINSIC(INTRINSIC_NAME##z, __VA_ARGS__);                                 \
  }

#define VCONV_ALL_MODE(INTRINSIC_NAME, MODE, ...)                              \
  VCONV_RAFCZ(INTRINSIC_NAME, MODE, __VA_ARGS__)                               \
  VCONV_POSTFIX_O(INTRINSIC_NAME, MODE, __VA_ARGS__)

  if constexpr (std::is_same<SRC_T, float>::value) {
    // src type is float
    if constexpr (std::is_same<DST_T, float>::value) {
      VCONV_RAFCZ(vconv_f322f32, cast_mode, CAST_ARGS);
    } else if constexpr (std::is_same<DST_T, half>::value) {
      VCONV_ALL_MODE(vconv_f322f16, cast_mode, CAST_ARGS);
    } else if constexpr (std::is_same<DST_T, bfloat16_t>::value) {
      VCONV_RAFCZ(vconv_f322bf16, cast_mode, CAST_ARGS);
    } else if constexpr (std::is_same<DST_T, int64_t>::value) {
      VCONV_RAFCZ(vconv_f322s64, cast_mode, CAST_ARGS);
    } else if constexpr (std::is_same<DST_T, int32_t>::value) {
      VCONV_RAFCZ(vconv_f322s32, cast_mode, CAST_ARGS);
    } else if constexpr (std::is_same<DST_T, int16_t>::value) {
      VCONV_RAFCZ(vconv_f322s16, cast_mode, CAST_ARGS);
    }
  } else if constexpr (std::is_same<SRC_T, half>::value) {
    // src type is half
    if constexpr (std::is_same<DST_T, float>::value) {
      INTRINSIC(vconv_f162f32, CAST_ARGS);
    } else if constexpr (std::is_same<DST_T, int32_t>::value) {
      VCONV_RAFCZ(vconv_f162s32, cast_mode, CAST_ARGS);
    } else if constexpr (std::is_same<DST_T, int16_t>::value) {
      VCONV_RAFCZ(vconv_f162s16, cast_mode, CAST_ARGS);
    } else if constexpr (std::is_same<DST_T, int8_t>::value) {
      VCONV_RAFCZ(vconv_f162s8, cast_mode, CAST_ARGS);
    } else if constexpr (std::is_same<DST_T, uint8_t>::value) {
      VCONV_RAFCZ(vconv_f162u8, cast_mode, CAST_ARGS);
    }
  } else if constexpr (std::is_same<SRC_T, bfloat16_t>::value) {
    // src type is bfloat16
    if constexpr (std::is_same<DST_T, float>::value) {
      INTRINSIC(vconv_bf162f32, CAST_ARGS);
    } else if constexpr (std::is_same<DST_T, int32_t>::value) {
      VCONV_RAFCZ(vconv_bf162s32, cast_mode, CAST_ARGS);
    }
  } else if constexpr (std::is_same<SRC_T, uint8_t>::value &&
                       std::is_same<DST_T, half>::value) {
    // src type is uint8 and dst type is half
    INTRINSIC(vconv_u82f16, CAST_ARGS);
  } else if constexpr (std::is_same<SRC_T, int8_t>::value &&
                       std::is_same<DST_T, half>::value) {
    // src type is int8 and dst type is half
    INTRINSIC(vconv_s82f16, CAST_ARGS);
  } else if constexpr (std::is_same<SRC_T, int16_t>::value) {
    // src type is int16
    if constexpr (std::is_same<DST_T, half>::value) {
      VCONV_RAFCZ(vconv_s162f16, cast_mode, CAST_ARGS);
    } else if constexpr (std::is_same<DST_T, float>::value) {
      INTRINSIC(vconv_s162f32, CAST_ARGS);
    }
  } else if constexpr (std::is_same<SRC_T, int32_t>::value) {
    // src type is int32
    if constexpr (std::is_same<DST_T, float>::value) {
      VCONV_RAFCZ(vconv_s322f32, cast_mode, CAST_ARGS);
    } else if constexpr (std::is_same<DST_T, int64_t>::value) {
      INTRINSIC(vconv_s322s64, CAST_ARGS);
    } else if constexpr (std::is_same<DST_T, int16_t>::value) {
      INTRINSIC(vconv_s322s16, CAST_ARGS);
    }
  } else if constexpr (std::is_same<SRC_T, int64_t>::value) {
    // src type is int64
    if constexpr (std::is_same<DST_T, int32_t>::value) {
      INTRINSIC(vconv_s642s32, CAST_ARGS);
    } else if constexpr (std::is_same<DST_T, float>::value) {
      VCONV_RAFCZ(vconv_s642f32, cast_mode, CAST_ARGS);
    }
  }
}

// When cast, the display will flip after exceeding the maximum display range.
__aiv__ __attribute__((always_inline)) void set_saturation_ctrl() {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
#else
  INTRINSIC(set_ctrl,
            (get_ctrl() | (uint64_t)((uint64_t)1 << CTRL_REGISTER_BIT)));
#endif
}

// When cast, the display will not be flipped after exceeding the maximum
// display range.
__aiv__ __attribute__((always_inline)) void set_saturation_ctrl_none() {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
#else
  INTRINSIC(set_ctrl, (get_ctrl() & (uint64_t)(0xF7FFFFFFFFFFFFFF)));
#endif
}

template <typename SRC_T, typename DST_T>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_vector_cast_1d_with_mode(memref_t<__ubuf__ SRC_T, 1> *src,
                                         memref_t<__ubuf__ DST_T, 1> *dst,
                                         CastMode cast_mode) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  auto dst_ptr = dst->aligned + dst->offset;
  auto src_ptr = src->aligned + src->offset;
  assert(isAddress32ByteAligned(src_ptr) &&
         "The starting address of src must be 32byte aligned.");
  assert(isAddress32ByteAligned(dst_ptr) &&
         "The starting address of dst must be 32byte aligned.");
  assert((src->strides[0] == 1 && dst->strides[0] == 1) &&
         "The src/dst must be continuous vector.");
#endif
}

template <typename SRC_T, typename DST_T>
__aiv__ __attribute__((always_inline)) void
vector_cast_1d_with_mode(memref_t<__ubuf__ SRC_T, 1> *src,
                         memref_t<__ubuf__ DST_T, 1> *dst, CastMode cast_mode) {
  static_assert(!std::is_same<SRC_T, bool>::value &&
                !std::is_same<DST_T, bool>::value &&
                "Src type and dst type can not be bool");

  // Input parameter constraints assert.
  check_inputs_of_vector_cast_1d_with_mode(src, dst, cast_mode);

  INTRINSIC_NO_ARGS(set_mask_count);
  const int64_t n = src->sizes[0];
  INTRINSIC(set_vector_mask, 0, n);

  __ubuf__ DST_T *dst_ptr = dst->aligned + dst->offset;
  __ubuf__ SRC_T *src_ptr = src->aligned + src->offset;

  constexpr int src_bytes = sizeof(SRC_T);
  constexpr int dst_bytes = sizeof(DST_T);
  // TODO: platform information
  constexpr int block_per_repeat = 8;
  uint16_t src_repeat_stride = block_per_repeat;
  uint16_t dst_repeat_stride = block_per_repeat;
  uint16_t factor;
  if constexpr (src_bytes > dst_bytes) {
    factor = src_bytes / dst_bytes;
    dst_repeat_stride = block_per_repeat / factor;
  } else if constexpr (src_bytes < dst_bytes) {
    factor = dst_bytes / src_bytes;
    src_repeat_stride = block_per_repeat / factor;
  }
  vconv<SRC_T, DST_T>(intrin_args<1, SRC_T, DST_T>{dst_ptr,
                                                   {src_ptr},
                                                   0,
                                                   1,   // repeat
                                                   1,   // dst_block_stride
                                                   {1}, // src_block_stride
                                                   dst_repeat_stride,
                                                   {src_repeat_stride}},
                      cast_mode);
  INTRINSIC_NO_ARGS(set_mask_norm);
}

template <typename SRC_T, typename DST_T, bool DISABLE_SIZE_ALIGN>
__aiv__ __attribute__((always_inline)) void
vector_cast_1d_with_overflow(memref_t<__ubuf__ SRC_T, 1> *src,
                             memref_t<__ubuf__ DST_T, 1> *dst,
                             memref_t<__ubuf__ SRC_T, 1> *tmp) {
  if ((std::is_same<SRC_T, int64_t>::value &&
       std::is_same<DST_T, int32_t>::value) ||
      (std::is_same<SRC_T, int32_t>::value &&
       std::is_same<DST_T, int16_t>::value)) {
    // Set the 59-bit ctrl register to truncate mode.
    set_saturation_ctrl();
    vector_cast_1d_with_mode(src, dst, CastMode::R);
    set_saturation_ctrl_none();
    return;
  }

  constexpr int num_per_block_src = INTR_BYTES_PER_BLOCK / sizeof(SRC_T);
  constexpr int num_per_block_dst = INTR_BYTES_PER_BLOCK / sizeof(DST_T);

  memref_t<__ubuf__ SRC_T, 2> src_2d;
  memref_t<__ubuf__ DST_T, 2> dst_2d;
  if (src->sizes[0] <= num_per_block_dst) {
    src_2d = {src->aligned,
              src->allocated,
              src->offset,
              {1, src->sizes[0]},
              {CEIL_FACTOR(src->sizes[0], num_per_block_src), 1}};
    dst_2d = {dst->aligned,
              dst->allocated,
              dst->offset,
              {1, src->sizes[0]},
              {CEIL_FACTOR(src->sizes[0], num_per_block_dst), 1}};
  } else {
    src_2d = {src->aligned,
              src->allocated,
              src->offset,
              {CEIL_DIV(dst->sizes[0], num_per_block_dst), num_per_block_dst},
              {num_per_block_dst, 1}};
    dst_2d = {dst->aligned,
              dst->allocated,
              dst->offset,
              {CEIL_DIV(dst->sizes[0], num_per_block_dst), num_per_block_dst},
              {num_per_block_dst, 1}};
  }
  vector_cast_with_overflow<SRC_T, DST_T, DISABLE_SIZE_ALIGN>(&src_2d, &dst_2d,
                                                             tmp);
}

template <typename SRC_T, typename DST_T>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_vector_cast_1d_with_temp(memref_t<__ubuf__ SRC_T, 1> *src,
                                         memref_t<__ubuf__ DST_T, 1> *dst,
                                         memref_t<__ubuf__ DST_T, 1> *temp) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  auto dst_ptr = dst->aligned + dst->offset;
  auto src_ptr = src->aligned + src->offset;
  auto temp0_ptr = temp->aligned + temp->offset;
  assert(isAddress32ByteAligned(src_ptr) &&
         "The starting address of src must be 32byte aligned.");
  assert(isAddress32ByteAligned(dst_ptr) &&
         "The starting address of dst must be 32byte aligned.");
  assert(isAddress32ByteAligned(temp0_ptr) &&
         "The starting address of tmp must be 32byte aligned.");
  assert((src->strides[0] == 1 && dst->strides[0] == 1) &&
         "The src/dst must be continuous vector.");
#endif
}

// src1_ptr only has one element
template <typename SRC_T, typename DST_T>
__aiv__ __attribute__((always_inline)) void
vector_select_vs_1d_intrin(__ubuf__ DST_T *dst_ptr, __ubuf__ SRC_T *cond_ptr,
                           __ubuf__ DST_T *src0_ptr, __ubuf__ DST_T *src1_ptr) {
  INTRINSIC(set_cmpmask, (__ubuf__ DST_T *)src1_ptr);
  INTRINSIC(pipe_barrier, PIPE_V);

  constexpr uint8_t dstBlockStride = 1;
  constexpr uint8_t src0BlockStride = 0;
  constexpr uint8_t src1BlockStride = 1;
  constexpr uint8_t dstRepeatStride = 8;
  constexpr uint8_t src0RepeatStride = 0;
  constexpr uint8_t src1RepeatStride = 1;
  constexpr uint8_t mode = static_cast<uint8_t>(SelectMode::MODE1);
  constexpr uint8_t repeat = 0;

  INTRINSIC(vsel, (__ubuf__ DST_T *)dst_ptr, (__ubuf__ DST_T *)src0_ptr,
            (__ubuf__ DST_T *)cond_ptr, repeat, dstBlockStride, src0BlockStride,
            src1BlockStride, dstRepeatStride, src0RepeatStride,
            src1RepeatStride, mode);
}

template <typename SRC_T, typename DST_T>
__aiv__ __attribute__((always_inline)) void
vector_select_vv_1d_intrin(__ubuf__ DST_T *dst_ptr, __ubuf__ SRC_T *cond_ptr,
                           __ubuf__ DST_T *src0_ptr, __ubuf__ DST_T *src1_ptr,
                           __ubuf__ DST_T *cond_addr_ptr) {
  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  (*(__ubuf__ uint64_t *)((__ubuf__ DST_T *)cond_addr_ptr)) =
      (uint64_t)((__ubuf__ SRC_T *)cond_ptr);
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);

  INTRINSIC(set_cmpmask, cond_addr_ptr);
  INTRINSIC(pipe_barrier, PIPE_V);

  constexpr uint8_t dstBlockStride = 1;
  constexpr uint8_t src0BlockStride = 0;
  constexpr uint8_t src1BlockStride = 0;
  constexpr uint8_t dstRepeatStride = 8;
  constexpr uint8_t src0RepeatStride = 0;
  constexpr uint8_t src1RepeatStride = 0;
  constexpr uint8_t mode = static_cast<uint8_t>(SelectMode::MODE2);
  constexpr uint8_t repeat = 0;

  INTRINSIC(vsel, (__ubuf__ DST_T *)dst_ptr, (__ubuf__ DST_T *)src0_ptr,
            (__ubuf__ DST_T *)src1_ptr, repeat, dstBlockStride, src0BlockStride,
            src1BlockStride, dstRepeatStride, src0RepeatStride,
            src1RepeatStride, mode);
}

template <typename SRC_T, typename DST_T>
__aiv__ __attribute__((always_inline)) void vector_cast_1d_with_temp_core(
    __ubuf__ DST_T *dst_ptr, __ubuf__ SRC_T *src_ptr,
    __ubuf__ DST_T *ones_buf_ptr, __ubuf__ DST_T *zeros_buf_ptr,
    __ubuf__ DST_T *cond_addr_ptr, const int64_t total_size, DST_T scalar_one,
    DST_T scalar_zero) {
  static_assert(!std::is_same<DST_T, bfloat16_t>::value);
  // src type is bool and dst type is half or float
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(DST_T);
  constexpr int max_b32_vcmp_repeat_cnts_elems =
      get_vcmp_max_repeat_cnts<float>() *
      (INTR_BYTES_PER_REPEAT / sizeof(float));
  if (total_size > max_b32_vcmp_repeat_cnts_elems) {
    // TODO: confirm the reason why mode 1 will cause accurary error when size >
    // max_b32_vcmp_repeat_cnts_elems
    brc_scalar_core_1d<DST_T>(scalar_one, ones_buf_ptr, num_per_block);
    brc_scalar_core_1d<DST_T>(scalar_zero, zeros_buf_ptr, num_per_block);

    INTRINSIC_NO_ARGS(set_mask_count);
    INTRINSIC(set_vector_mask, 0, total_size);
    vector_select_vv_1d_intrin<SRC_T, DST_T>(dst_ptr, src_ptr, ones_buf_ptr,
                                             zeros_buf_ptr, cond_addr_ptr);
    INTRINSIC_NO_ARGS(set_mask_norm);
    return;
  }

  brc_scalar_core_1d<DST_T>(scalar_one, ones_buf_ptr, num_per_block);
  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  zeros_buf_ptr[0] = scalar_zero;
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);

  INTRINSIC_NO_ARGS(set_mask_count);
  INTRINSIC(set_vector_mask, 0, total_size);
  vector_select_vs_1d_intrin<SRC_T, DST_T>(dst_ptr, src_ptr, ones_buf_ptr,
                                           zeros_buf_ptr);
  INTRINSIC_NO_ARGS(set_mask_norm);
}

template <typename SRC_T, typename DST_T>
__aiv__ __attribute__((always_inline)) void vector_cast_1d_with_temp(
    memref_t<__ubuf__ SRC_T, 1> *src, memref_t<__ubuf__ DST_T, 1> *dst,
    memref_t<__ubuf__ DST_T, 1> *temp, DST_T scalar_one, DST_T scalar_zero) {
  // Input parameter constraints assert.
  check_inputs_of_vector_cast_1d_with_temp(src, dst, temp);

  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(DST_T);
  const int64_t total_size = dst->sizes[0];
  __ubuf__ DST_T *dst_ptr = dst->aligned + dst->offset;
  __ubuf__ SRC_T *src_ptr = src->aligned + src->offset;
  __ubuf__ DST_T *ones_buf_ptr = temp->aligned + temp->offset;
  __ubuf__ DST_T *zeros_buf_ptr = temp->aligned + temp->offset + num_per_block;
  __ubuf__ DST_T *cond_addr_ptr =
      temp->aligned + temp->offset + 2 * num_per_block;
  vector_cast_1d_with_temp_core<SRC_T, DST_T>(
      dst_ptr, src_ptr, ones_buf_ptr, zeros_buf_ptr, cond_addr_ptr, total_size,
      scalar_one, scalar_zero);
}

template <typename SRC_T, typename DST_T>
__aiv__ __attribute__((always_inline)) void
vector_cast_1d_with_temp(memref_t<__ubuf__ SRC_T, 1> *src,
                         memref_t<__ubuf__ DST_T, 1> *dst,
                         memref_t<__ubuf__ DST_T, 1> *temp) {
  static_assert((std::is_same<SRC_T, bool>::value ||
                 std::is_same<SRC_T, uint8_t>::value) &&
                (std::is_same<DST_T, half>::value ||
                 std::is_same<DST_T, float>::value ||
                 std::is_same<DST_T, bfloat16_t>::value ||
                 std::is_same<DST_T, int16_t>::value ||
                 std::is_same<DST_T, int32_t>::value ||
                 std::is_same<DST_T, uint16_t>::value ||
                 std::is_same<DST_T, uint32_t>::value) &&
                "Src type must be bool or uint8 and dst type must be one of "
                "(half, float, int16_t, uint16_t, int32_t, uint32_t)");
  DST_T scalar_one;
  DST_T scalar_zero;
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  if constexpr (std::is_same<DST_T, half>::value) {
    scalar_one = {static_cast<unsigned short>(1)};
    scalar_zero = {static_cast<unsigned short>(0)};
  } else if constexpr (!std::is_same<DST_T, bfloat16_t>::value) {
    scalar_one = static_cast<DST_T>(1);
    scalar_zero = static_cast<DST_T>(0);
  }
#else
  if constexpr (!std::is_same<DST_T, bfloat16_t>::value) {
    scalar_one = static_cast<DST_T>(1);
    scalar_zero = static_cast<DST_T>(0);
  }
#endif

  vector_cast_1d_with_temp(src, dst, temp, scalar_one, scalar_zero);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_cast_1d_with_temp<bool, bfloat16_t>(
    memref_t<__ubuf__ bool, 1> *src, memref_t<__ubuf__ bfloat16_t, 1> *dst,
    memref_t<__ubuf__ bfloat16_t, 1> *temp) {
  // convert int16 memref to half memref
  memref_t<__ubuf__ half, 1> dst_as_half;
  memref_t<__ubuf__ half, 1> temp_as_half;

  half scalar_one;
  half scalar_zero;
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  scalar_one = {static_cast<unsigned short>(1.875)};
  scalar_zero = {static_cast<unsigned short>(0)};
#else
  scalar_one = 1.875;
  scalar_zero = 0;
#endif

  view_as<bfloat16_t, half, 1>(dst, &dst_as_half);
  view_as<bfloat16_t, half, 1>(temp, &temp_as_half);

  vector_cast_1d_with_temp<bool, half>(src, &dst_as_half, &temp_as_half,
                                       scalar_one, scalar_zero);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_cast_1d_with_temp<bool, int16_t>(memref_t<__ubuf__ bool, 1> *src,
                                        memref_t<__ubuf__ int16_t, 1> *dst,
                                        memref_t<__ubuf__ int16_t, 1> *temp) {
  // convert int16 memref to half memref
  memref_t<__ubuf__ half, 1> dst_as_half;
  memref_t<__ubuf__ half, 1> temp_as_half;

  int16_t scalar_one = 1;
  int16_t scalar_zero = 0;

  view_as<int16_t, half, 1>(dst, &dst_as_half);
  view_as<int16_t, half, 1>(temp, &temp_as_half);

  vector_cast_1d_with_temp<bool, half>(src, &dst_as_half, &temp_as_half,
                                       *(half *)&scalar_one,
                                       *(half *)&scalar_zero);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_cast_1d_with_temp<bool, uint16_t>(memref_t<__ubuf__ bool, 1> *src,
                                         memref_t<__ubuf__ uint16_t, 1> *dst,
                                         memref_t<__ubuf__ uint16_t, 1> *temp) {
  // convert uint16 memref to half memref
  memref_t<__ubuf__ half, 1> dst_as_half;
  memref_t<__ubuf__ half, 1> temp_as_half;

  uint16_t scalar_one = 1;
  uint16_t scalar_zero = 0;

  view_as<uint16_t, half, 1>(dst, &dst_as_half);
  view_as<uint16_t, half, 1>(temp, &temp_as_half);

  vector_cast_1d_with_temp<bool, half>(src, &dst_as_half, &temp_as_half,
                                       *(half *)&scalar_one,
                                       *(half *)&scalar_zero);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_cast_1d_with_temp<bool, int32_t>(memref_t<__ubuf__ bool, 1> *src,
                                        memref_t<__ubuf__ int32_t, 1> *dst,
                                        memref_t<__ubuf__ int32_t, 1> *temp) {
  // convert int32 memref to float memref
  memref_t<__ubuf__ float, 1> dst_as_float;
  memref_t<__ubuf__ float, 1> temp_as_float;

  int32_t scalar_one = 1;
  int32_t scalar_zero = 0;

  view_as<int32_t, float, 1>(dst, &dst_as_float);
  view_as<int32_t, float, 1>(temp, &temp_as_float);

  vector_cast_1d_with_temp<bool, float>(src, &dst_as_float, &temp_as_float,
                                        *(float *)&scalar_one,
                                        *(float *)&scalar_zero);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_cast_1d_with_temp<bool, uint32_t>(memref_t<__ubuf__ bool, 1> *src,
                                         memref_t<__ubuf__ uint32_t, 1> *dst,
                                         memref_t<__ubuf__ uint32_t, 1> *temp) {
  // convert uint32 memref to float memref
  memref_t<__ubuf__ float, 1> dst_as_float;
  memref_t<__ubuf__ float, 1> temp_as_float;

  uint32_t scalar_one = 1;
  uint32_t scalar_zero = 0;

  view_as<uint32_t, float, 1>(dst, &dst_as_float);
  view_as<uint32_t, float, 1>(temp, &temp_as_float);

  vector_cast_1d_with_temp<bool, float>(src, &dst_as_float, &temp_as_float,
                                        *(float *)&scalar_one,
                                        *(float *)&scalar_zero);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_cast_1d_with_temp<uint8_t, bfloat16_t>(
    memref_t<__ubuf__ uint8_t, 1> *src, memref_t<__ubuf__ bfloat16_t, 1> *dst,
    memref_t<__ubuf__ bfloat16_t, 1> *temp) {
  // convert int16 memref to half memref
  memref_t<__ubuf__ half, 1> dst_as_half;
  memref_t<__ubuf__ half, 1> temp_as_half;

  half scalar_one;
  half scalar_zero;
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  scalar_one = {static_cast<unsigned short>(1.875)};
  scalar_zero = {static_cast<unsigned short>(0)};
#else
  scalar_one = 1.875;
  scalar_zero = 0;
#endif

  view_as<bfloat16_t, half, 1>(dst, &dst_as_half);
  view_as<bfloat16_t, half, 1>(temp, &temp_as_half);

  vector_cast_1d_with_temp<uint8_t, half>(src, &dst_as_half, &temp_as_half,
                                          scalar_one, scalar_zero);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_cast_1d_with_temp<uint8_t, int16_t>(
    memref_t<__ubuf__ uint8_t, 1> *src, memref_t<__ubuf__ int16_t, 1> *dst,
    memref_t<__ubuf__ int16_t, 1> *temp) {
  // convert int16 memref to half memref
  memref_t<__ubuf__ half, 1> dst_as_half;
  memref_t<__ubuf__ half, 1> temp_as_half;

  int16_t scalar_one = 1;
  int16_t scalar_zero = 0;

  view_as<int16_t, half, 1>(dst, &dst_as_half);
  view_as<int16_t, half, 1>(temp, &temp_as_half);

  vector_cast_1d_with_temp<uint8_t, half>(src, &dst_as_half, &temp_as_half,
                                          *(half *)&scalar_one,
                                          *(half *)&scalar_zero);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_cast_1d_with_temp<uint8_t, uint16_t>(
    memref_t<__ubuf__ uint8_t, 1> *src, memref_t<__ubuf__ uint16_t, 1> *dst,
    memref_t<__ubuf__ uint16_t, 1> *temp) {
  // convert uint16 memref to half memref
  memref_t<__ubuf__ half, 1> dst_as_half;
  memref_t<__ubuf__ half, 1> temp_as_half;

  uint16_t scalar_one = 1;
  uint16_t scalar_zero = 0;

  view_as<uint16_t, half, 1>(dst, &dst_as_half);
  view_as<uint16_t, half, 1>(temp, &temp_as_half);

  vector_cast_1d_with_temp<uint8_t, half>(src, &dst_as_half, &temp_as_half,
                                          *(half *)&scalar_one,
                                          *(half *)&scalar_zero);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_cast_1d_with_temp<uint8_t, int32_t>(
    memref_t<__ubuf__ uint8_t, 1> *src, memref_t<__ubuf__ int32_t, 1> *dst,
    memref_t<__ubuf__ int32_t, 1> *temp) {
  // convert int32 memref to float memref
  memref_t<__ubuf__ float, 1> dst_as_float;
  memref_t<__ubuf__ float, 1> temp_as_float;

  int32_t scalar_one = 1;
  int32_t scalar_zero = 0;

  view_as<int32_t, float, 1>(dst, &dst_as_float);
  view_as<int32_t, float, 1>(temp, &temp_as_float);

  vector_cast_1d_with_temp<uint8_t, float>(src, &dst_as_float, &temp_as_float,
                                           *(float *)&scalar_one,
                                           *(float *)&scalar_zero);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_cast_1d_with_temp<uint8_t, uint32_t>(
    memref_t<__ubuf__ uint8_t, 1> *src, memref_t<__ubuf__ uint32_t, 1> *dst,
    memref_t<__ubuf__ uint32_t, 1> *temp) {
  // convert uint32 memref to float memref
  memref_t<__ubuf__ float, 1> dst_as_float;
  memref_t<__ubuf__ float, 1> temp_as_float;

  uint32_t scalar_one = 1;
  uint32_t scalar_zero = 0;

  view_as<uint32_t, float, 1>(dst, &dst_as_float);
  view_as<uint32_t, float, 1>(temp, &temp_as_float);

  vector_cast_1d_with_temp<uint8_t, float>(src, &dst_as_float, &temp_as_float,
                                           *(float *)&scalar_one,
                                           *(float *)&scalar_zero);
}

extern "C" {
//===-------------------------------------------------------------------===//
// Register vector_cast_1d_with_temp func, used for cast 2d with temp
//===-------------------------------------------------------------------===//
REGISTE_CAST_1D_WITH_TEMP_CORE(bool, half);
REGISTE_CAST_1D_WITH_TEMP_CORE(bool, float);

//===-------------------------------------------------------------------===//
// cast, 1 dim
//===-------------------------------------------------------------------===//
// cast bool to half
REGISTE_CAST_WITH_TEMP(bool, half, 1);
// cast bool to float
REGISTE_CAST_WITH_TEMP(bool, float, 1);
// cast bool to bfloat16
REGISTE_CAST_WITH_TEMP(bool, bfloat16_t, 1);
// cast bool to int16_t
REGISTE_CAST_WITH_TEMP(bool, int16_t, 1);
// cast bool to int32_t
REGISTE_CAST_WITH_TEMP(bool, int32_t, 1);
// cast bool to uint16_t
REGISTE_CAST_WITH_TEMP(bool, uint16_t, 1);
// cast bool to uint32_t
REGISTE_CAST_WITH_TEMP(bool, uint32_t, 1);
// cast uint8_t to half
REGISTE_CAST_WITH_TEMP(uint8_t, half, 1);
// cast uint8_t to float
REGISTE_CAST_WITH_TEMP(uint8_t, float, 1);
// cast uint8_t to bfloat16
REGISTE_CAST_WITH_TEMP(uint8_t, bfloat16_t, 1);
// cast uint8_t to int16_t
REGISTE_CAST_WITH_TEMP(uint8_t, int16_t, 1);
// cast uint8_t to int32_t
REGISTE_CAST_WITH_TEMP(uint8_t, int32_t, 1);
// cast uint8_t to uint16_t
REGISTE_CAST_WITH_TEMP(uint8_t, uint16_t, 1);
// cast uint8_t to uint32_t
REGISTE_CAST_WITH_TEMP(uint8_t, uint32_t, 1);
// cast float to float
REGISTE_CAST(float, float, 1);
// cast float to half
REGISTE_CAST(float, half, 1);
// cast float to bfloat16_t
REGISTE_CAST(float, bfloat16_t, 1);
// cast float to int64
REGISTE_CAST(float, int64_t, 1);
// cast float to int32
REGISTE_CAST(float, int32_t, 1);
// cast float to int16
REGISTE_CAST(float, int16_t, 1);
// cast half to float
REGISTE_CAST(half, float, 1);
// cast half to int32
REGISTE_CAST(half, int32_t, 1);
// cast half to int16
REGISTE_CAST(half, int16_t, 1);
// cast half to int8
REGISTE_CAST(half, int8_t, 1);
// cast half to uint8
REGISTE_CAST(half, uint8_t, 1);
// cast bfloat16_t to float32
REGISTE_CAST(bfloat16_t, float, 1);
// cast bfloat16_t to int32
REGISTE_CAST(bfloat16_t, int32_t, 1);
// cast uint8 to float16
REGISTE_CAST(uint8_t, half, 1);
// cast int8 to float16
REGISTE_CAST(int8_t, half, 1);
// cast int16 to float16
REGISTE_CAST(int16_t, half, 1);
// cast int16 to float
REGISTE_CAST(int16_t, float, 1);
// cast int32 to float
REGISTE_CAST(int32_t, float, 1);
// cast int32 to int64
REGISTE_CAST(int32_t, int64_t, 1);
// cast int32 to int16
REGISTE_CAST(int32_t, int16_t, 1);
// cast int64 to int32
REGISTE_CAST(int64_t, int32_t, 1);
// cast int64 to float32
REGISTE_CAST(int64_t, float, 1);
// cast int32 to int8 overflow
REGISTE_CAST_OVERFLOW(int32_t, int8_t, 1);
// cast int16 to int8 overflow
REGISTE_CAST_OVERFLOW(int16_t, int8_t, 1);
// cast int32 to int16 overflow
REGISTE_CAST_OVERFLOW(int32_t, int16_t, 1);
// cast int64 to in32 overflow
REGISTE_CAST_OVERFLOW(int64_t, int32_t, 1);

//===-------------------------------------------------------------------===//
// Register vconv func
//===-------------------------------------------------------------------===//
DECLARE_VCONV(float, float);
DECLARE_VCONV(float, half);
DECLARE_VCONV(float, bfloat16_t);
DECLARE_VCONV(float, int64_t);
DECLARE_VCONV(float, int32_t);
DECLARE_VCONV(float, int16_t);
DECLARE_VCONV(half, float);
DECLARE_VCONV(half, int32_t);
DECLARE_VCONV(half, int16_t);
DECLARE_VCONV(half, int8_t);
DECLARE_VCONV(half, uint8_t);
DECLARE_VCONV(bfloat16_t, float);
DECLARE_VCONV(bfloat16_t, int32_t);
DECLARE_VCONV(uint8_t, half);
DECLARE_VCONV(int8_t, half);
DECLARE_VCONV(int16_t, half);
DECLARE_VCONV(int16_t, float);
DECLARE_VCONV(int32_t, float);
DECLARE_VCONV(int32_t, int64_t);
DECLARE_VCONV(int32_t, int16_t);
DECLARE_VCONV(int64_t, int32_t);
DECLARE_VCONV(int64_t, float);
}
