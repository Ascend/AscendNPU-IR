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

template <typename SRC_T, typename DST_T>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_vector_cast_2d_with_mode(memref_t<__ubuf__ SRC_T, 2> *src,
                                         memref_t<__ubuf__ DST_T, 2> *dst,
                                         CastMode cast_mode) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  auto dst_ptr = dst->aligned + dst->offset;
  auto src_ptr = src->aligned + src->offset;
  assert(isAddress32ByteAligned(src_ptr) &&
         "The starting address of src must be 32byte aligned.");
  assert(isAddress32ByteAligned(dst_ptr) &&
         "The starting address of dst must be 32byte aligned.");
  assert((isSizeAlignedToBlock<SRC_T>(src->strides[0]) &&
          isSizeAlignedToBlock<DST_T>(dst->strides[0])) &&
         "The src/dst strides[0] must be aligned to block.");
  assert((src->strides[1] == 1 && dst->strides[1] == 1) &&
         "The src/dst last dim must be continuous.");
#endif
}

template <typename SRC_T, typename DST_T>
__aiv__ __attribute__((always_inline)) void
vconv_with_arbitrary_repeat(intrin_args<1, SRC_T, DST_T> args,
                            CastMode cast_mode) {
  constexpr int src_bytes = sizeof(SRC_T);
  constexpr int dst_bytes = sizeof(DST_T);
  constexpr int src_num_per_block = INTR_BYTES_PER_BLOCK / src_bytes;
  constexpr int dst_num_per_block = INTR_BYTES_PER_BLOCK / dst_bytes;
  const uint16_t dst_repeat_stride = args.dst_repeat_stride;
  const uint16_t src_repeat_stride = args.src_repeat_stride[0];

  if (args.repeat >= INTR_MAX_REPEAT_CNTS) [[unlikely]] {
    int64_t loop_num = args.repeat / INTR_MAX_REPEAT_CNTS;
    constexpr uint64_t repeat_times = INTR_MAX_REPEAT_CNTS;
    for (int64_t j = 0; j < loop_num; j++) {
      __ubuf__ DST_T *dst_addr = args.dst + j * INTR_MAX_REPEAT_CNTS *
                                                dst_repeat_stride *
                                                dst_num_per_block;
      __ubuf__ SRC_T *src_addr = args.src[0] + j * INTR_MAX_REPEAT_CNTS *
                                                   src_repeat_stride *
                                                   src_num_per_block;
      vconv<SRC_T, DST_T>(intrin_args<1, SRC_T, DST_T>{dst_addr,
                                                       {src_addr},
                                                       0,
                                                       repeat_times,
                                                       1,   // dst_block_stride
                                                       {1}, // src_block_stride
                                                       dst_repeat_stride,
                                                       {src_repeat_stride}},
                          cast_mode);
    }
  }
  if (args.repeat % INTR_MAX_REPEAT_CNTS != 0) [[likely]] {
    __ubuf__ DST_T *dst_addr =
        args.dst + FLOOR_FACTOR(args.repeat, INTR_MAX_REPEAT_CNTS) *
                       dst_repeat_stride * dst_num_per_block;
    __ubuf__ SRC_T *src_addr =
        args.src[0] + FLOOR_FACTOR(args.repeat, INTR_MAX_REPEAT_CNTS) *
                          src_repeat_stride * src_num_per_block;
    const uint64_t repeat_times = args.repeat % INTR_MAX_REPEAT_CNTS;
    vconv<SRC_T, DST_T>(intrin_args<1, SRC_T, DST_T>{dst_addr,
                                                     {src_addr},
                                                     0,
                                                     repeat_times,
                                                     1,   // dst_block_stride
                                                     {1}, // src_block_stride
                                                     dst_repeat_stride,
                                                     {src_repeat_stride}},
                        cast_mode);
  }
}

template <typename SRC_T, typename DST_T>
__aiv__ __attribute__((always_inline)) void
vector_cast_2d_with_mode_main(memref_t<__ubuf__ SRC_T, 2> *src,
                              memref_t<__ubuf__ DST_T, 2> *dst,
                              CastMode cast_mode) {
  const int64_t size0 = src->sizes[0];
  const int64_t size1 = src->sizes[1];
  const int64_t dst_stride0 = dst->strides[0];
  const int64_t src_stride0 = src->strides[0];

  __ubuf__ DST_T *dst_ptr = dst->aligned + dst->offset;
  __ubuf__ SRC_T *src_ptr = src->aligned + src->offset;

  constexpr int src_bytes = sizeof(SRC_T);
  constexpr int dst_bytes = sizeof(DST_T);
  constexpr int bytes = src_bytes > dst_bytes ? src_bytes : dst_bytes;
  constexpr int num_per_repeat = INTR_BYTES_PER_REPEAT / bytes;
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / bytes;

  constexpr int src_num_per_block = INTR_BYTES_PER_BLOCK / src_bytes;
  constexpr int dst_num_per_block = INTR_BYTES_PER_BLOCK / dst_bytes;

  bool repeat_size0 = dst_stride0 <= INTR_MAX_BLOCK_CNTS * num_per_block &&
                      src_stride0 <= INTR_MAX_BLOCK_CNTS * num_per_block &&
                      size0 >= size1 / num_per_repeat;
  uint64_t repeat_num = repeat_size0 ? size0 : size1 / num_per_repeat;
  uint16_t dst_repeat_stride = repeat_size0
                                   ? dst_stride0 / dst_num_per_block
                                   : num_per_repeat / dst_num_per_block;
  uint16_t src_repeat_stride = repeat_size0
                                   ? src_stride0 / src_num_per_block
                                   : num_per_repeat / src_num_per_block;

  int64_t block_dim = repeat_size0 ? size1 / num_per_repeat : size0;
  uint16_t dst_loop_offset = repeat_size0 ? num_per_repeat : dst_stride0;
  uint16_t src_loop_offset = repeat_size0 ? num_per_repeat : src_stride0;

  INTRINSIC_NO_ARGS(set_mask_norm);
  SetMaskValueByCount(num_per_repeat);
  for (int64_t i = 0; i < block_dim; i++) {
    vconv_with_arbitrary_repeat<SRC_T, DST_T>(
        intrin_args<1, SRC_T, DST_T>{dst_ptr + i * dst_loop_offset,
                                     {src_ptr + i * src_loop_offset},
                                     0,
                                     repeat_num,
                                     1,   // dst_block_stride
                                     {1}, // src0_block_stride
                                     dst_repeat_stride,
                                     {src_repeat_stride}},
        cast_mode);
  }
}

template <typename SRC_T, typename DST_T>
__aiv__ __attribute__((always_inline)) void
vector_cast_2d_with_mode_remain(memref_t<__ubuf__ SRC_T, 2> *src,
                                memref_t<__ubuf__ DST_T, 2> *dst,
                                CastMode cast_mode) {
  const int64_t size0 = src->sizes[0];
  const int64_t size1 = src->sizes[1];
  const int64_t dst_stride0 = dst->strides[0];
  const int64_t src_stride0 = src->strides[0];

  __ubuf__ DST_T *dst_ptr = dst->aligned + dst->offset;
  __ubuf__ SRC_T *src_ptr = src->aligned + src->offset;

  constexpr int src_bytes = sizeof(SRC_T);
  constexpr int dst_bytes = sizeof(DST_T);
  constexpr int bytes = src_bytes > dst_bytes ? src_bytes : dst_bytes;
  constexpr int num_per_repeat = INTR_BYTES_PER_REPEAT / bytes;
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / bytes;

  constexpr int src_num_per_block = INTR_BYTES_PER_BLOCK / src_bytes;
  constexpr int dst_num_per_block = INTR_BYTES_PER_BLOCK / dst_bytes;

  bool repeat_size0 = dst_stride0 <= INTR_MAX_BLOCK_CNTS * num_per_block &&
                      src_stride0 <= INTR_MAX_BLOCK_CNTS * num_per_block &&
                      size0 >= 1;
  uint64_t repeat_num = repeat_size0 ? size0 : 1;
  uint16_t dst_repeat_stride =
      repeat_size0 ? dst_stride0 / dst_num_per_block : 0;
  uint16_t src_repeat_stride =
      repeat_size0 ? src_stride0 / src_num_per_block : 0;
  int64_t block_dim = repeat_size0 ? 1 : size0;
  uint16_t dst_loop_offset = repeat_size0 ? 0 : dst_stride0;
  uint16_t src_loop_offset = repeat_size0 ? 0 : src_stride0;

  INTRINSIC_NO_ARGS(set_mask_norm);
  SetMaskValueByCount(size1 % num_per_repeat);
  for (int64_t i = 0; i < block_dim; i++) {
    vconv_with_arbitrary_repeat<SRC_T, DST_T>(
        intrin_args<1, SRC_T, DST_T>{
            dst_ptr + size1 / num_per_repeat * num_per_repeat +
                i * dst_loop_offset,
            {src_ptr + size1 / num_per_repeat * num_per_repeat +
             i * src_loop_offset},
            0,
            repeat_num,
            1,   // dst_block_stride
            {1}, // src0_block_stride
            dst_repeat_stride,
            {src_repeat_stride}},
        cast_mode);
  }
}

template <typename SRC_T, typename DST_T>
__aiv__ __attribute__((always_inline)) void
vector_cast_2d_with_mode(memref_t<__ubuf__ SRC_T, 2> *src,
                         memref_t<__ubuf__ DST_T, 2> *dst, CastMode cast_mode) {
  static_assert(!std::is_same<SRC_T, bool>::value &&
                !std::is_same<DST_T, bool>::value &&
                "Src type and dst type can not be bool");

  // Input parameter constraints assert.
  check_inputs_of_vector_cast_2d_with_mode(src, dst, cast_mode);

  const int64_t size1 = src->sizes[1];
  constexpr int src_bytes = sizeof(SRC_T);
  constexpr int dst_bytes = sizeof(DST_T);
  constexpr int bytes = src_bytes > dst_bytes ? src_bytes : dst_bytes;
  constexpr int num_per_repeat = INTR_BYTES_PER_REPEAT / bytes;

  if (size1 >= num_per_repeat) {
    vector_cast_2d_with_mode_main<SRC_T, DST_T>(src, dst, cast_mode);
  }

  if (size1 % num_per_repeat != 0) {
    vector_cast_2d_with_mode_remain<SRC_T, DST_T>(src, dst, cast_mode);
  }
}

template <typename SRC_T, typename DST_T>
__aiv__ __attribute__((always_inline)) void
vector_cast_2d_with_overflow(memref_t<__ubuf__ SRC_T, 2> *src,
                             memref_t<__ubuf__ DST_T, 2> *dst,
                             memref_t<__ubuf__ SRC_T, 1> *tmp) {
  if ((std::is_same<SRC_T, int64_t>::value &&
       std::is_same<DST_T, int32_t>::value) ||
      (std::is_same<SRC_T, int32_t>::value &&
       std::is_same<DST_T, int16_t>::value)) {
    // Set the 59-bit ctrl register to truncate mode.
    set_saturation_ctrl();
    vector_cast_2d_with_mode(src, dst, CastMode::R);
    set_saturation_ctrl_none();
    return;
  }

  constexpr int src_num_per_block = INTR_BYTES_PER_BLOCK / sizeof(SRC_T);
  // TODO: The performance of this scenario needs to be optimized later.
  // Considering the AlignAllocSize pass for size alignment and axis alignment,
  // there will still be a problem of multiple reads and multiple writes.
  if (src->strides[0] >=
      INTR_TRANSPOSE_PER_REPEAT_BYTS_FOR_I8 / src_num_per_block) {
    for (int i = 0; i < src->sizes[0]; i++) {
      INTRINSIC(pipe_barrier, PIPE_V);
      memref_t<__ubuf__ SRC_T, 1> src_1d{src->aligned,
                                         src->allocated,
                                         src->offset + i * src->strides[0],
                                         {src->sizes[1]},
                                         {1}};
      memref_t<__ubuf__ DST_T, 1> dst_1d{dst->aligned,
                                         dst->allocated,
                                         dst->offset + i * dst->strides[0],
                                         {dst->sizes[1]},
                                         {1}};
      vector_cast_1d_with_overflow(&src_1d, &dst_1d, tmp);
    }
    return;
  }
  vector_cast_with_overflow(src, dst, tmp);
}

template <typename SRC_T, typename DST_T>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_vector_cast_2d_with_temp(memref_t<__ubuf__ SRC_T, 2> *src,
                                         memref_t<__ubuf__ DST_T, 2> *dst,
                                         memref_t<__ubuf__ DST_T, 1> *temp) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  auto temp_ptr = temp->aligned + temp->offset;
  assert(isAddress32ByteAligned(src_ptr) &&
         "The starting address of src must be 32byte aligned.");
  assert(isAddress32ByteAligned(dst_ptr) &&
         "The starting address of dst must be 32byte aligned.");
  assert(isAddress32ByteAligned(temp_ptr) &&
         "The starting address of tmp must be 32byte aligned.");
  assert((isSizeAlignedToBlock<SRC_T>(src->strides[0]) &&
          isSizeAlignedToBlock<DST_T>(dst->strides[0])) &&
         "The src/dst strides[0] must be aligned to block.");
  assert((src->strides[1] == 1 && dst->strides[1] == 1) &&
         "The src/dst last dim must be continuous.");
#endif
}

template <typename SRC_T, typename DST_T>
__aiv__ __attribute__((always_inline)) void
vector_cast_2d_with_temp(memref_t<__ubuf__ SRC_T, 2> *src,
                         memref_t<__ubuf__ DST_T, 2> *dst,
                         memref_t<__ubuf__ DST_T, 1> *temp) {
  static_assert(std::is_same<SRC_T, bool>::value &&
                (std::is_same<DST_T, half>::value ||
                 std::is_same<DST_T, float>::value ||
                 std::is_same<DST_T, bfloat16_t>::value ||
                 std::is_same<DST_T, int16_t>::value ||
                 std::is_same<DST_T, int32_t>::value ||
                 std::is_same<DST_T, uint16_t>::value ||
                 std::is_same<DST_T, uint32_t>::value) &&
                "Src type must be bool and dst type must be one of"
                "(half, float, int16_t, uint16_t, int32_t, uint32_t)");

  // Input parameter constraints assert.
  check_inputs_of_vector_cast_2d_with_temp(src, dst, temp);

  memref_t<__ubuf__ uint8_t, 2> src_u8_2d;
  view_as<SRC_T, uint8_t, 2>(src, &src_u8_2d);

  memref_t<__ubuf__ uint8_t, 1> src_u8_1d;
  memref_t<__ubuf__ DST_T, 1> dst_1d;
  convert_2d_to_1d_args<uint8_t>(&src_u8_2d, &src_u8_1d);
  convert_2d_to_1d_args<DST_T>(dst, &dst_1d);
  for (int64_t i = 0; i < dst->sizes[0]; ++i) {
    src_u8_1d.offset = src_u8_2d.offset + src_u8_2d.strides[0] * i;
    dst_1d.offset = dst->offset + dst->strides[0] * i;
    vector_cast_1d_with_temp<uint8_t, DST_T>(&src_u8_1d, &dst_1d, temp);
  }
}

extern "C" {
//===-------------------------------------------------------------------===//
// cast, 2 dim
//===-------------------------------------------------------------------===//
// cast bool to half
REGISTE_CAST_WITH_TEMP(bool, half, 2);
// cast bool to float
REGISTE_CAST_WITH_TEMP(bool, float, 2);
// cast bool to bfloat16
REGISTE_CAST_WITH_TEMP(bool, bfloat16_t, 2);
// cast bool to int16
REGISTE_CAST_WITH_TEMP(bool, int16_t, 2);
// cast bool to int32
REGISTE_CAST_WITH_TEMP(bool, int32_t, 2);
// cast bool to uint16
REGISTE_CAST_WITH_TEMP(bool, uint16_t, 2);
// cast bool to uint32
REGISTE_CAST_WITH_TEMP(bool, uint32_t, 2);
// cast float to float
REGISTE_CAST(float, float, 2);
// cast float to half
REGISTE_CAST(float, half, 2);
// cast float to bfloat16_t
REGISTE_CAST(float, bfloat16_t, 2);
// cast float to int64
REGISTE_CAST(float, int64_t, 2);
// cast float to int32
REGISTE_CAST(float, int32_t, 2);
// cast float to int16
REGISTE_CAST(float, int16_t, 2);
// cast half to float
REGISTE_CAST(half, float, 2);
// cast half to int32
REGISTE_CAST(half, int32_t, 2);
// cast half to int16
REGISTE_CAST(half, int16_t, 2);
// cast half to int8
REGISTE_CAST(half, int8_t, 2);
// cast half to uint8
REGISTE_CAST(half, uint8_t, 2);
// cast bfloat16_t to float32
REGISTE_CAST(bfloat16_t, float, 2);
// cast bfloat16_t to int32
REGISTE_CAST(bfloat16_t, int32_t, 2);
// cast uint8 to float16
REGISTE_CAST(uint8_t, half, 2);
// cast int8 to float16
REGISTE_CAST(int8_t, half, 2);
// cast int16 to float16
REGISTE_CAST(int16_t, half, 2);
// cast int16 to float
REGISTE_CAST(int16_t, float, 2);
// cast int32 to float
REGISTE_CAST(int32_t, float, 2);
// cast int32 to int64
REGISTE_CAST(int32_t, int64_t, 2);
// cast int32 to int16
REGISTE_CAST(int32_t, int16_t, 2);
// cast int64 to int32
REGISTE_CAST(int64_t, int32_t, 2);
// cast int64 to float32
REGISTE_CAST(int64_t, float, 2);
// cast int32 to int8 overflow
REGISTE_CAST_OVERFLOW(int32_t, int8_t, 2);
// cast int16 to int8 overflow
REGISTE_CAST_OVERFLOW(int16_t, int8_t, 2);
// cast int32 to int16 overflow
REGISTE_CAST_OVERFLOW(int32_t, int16_t, 2);
// cast int64 to int32 overflow
REGISTE_CAST_OVERFLOW(int64_t, int32_t, 2);
}
