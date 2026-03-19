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

#ifndef HIVM_MLIR_TEMPLATE_VECTOR_UTILS_H
#define HIVM_MLIR_TEMPLATE_VECTOR_UTILS_H

#include "Utils.h"
#include <type_traits>
#include <limits>

// limitations of vector intrinsic
constexpr int32_t INTR_MAX_REPEAT_CNTS = 255;
constexpr int32_t INTR_MAX_BLOCK_CNTS = 255;
constexpr int32_t INTR_BYTES_PER_REPEAT = 256;
constexpr int32_t VNCHWCONV_INTR_BYTES_PER_REPEAT = 512;
constexpr int32_t INTR_BYTES_PER_BLOCK = 32;
constexpr int32_t INTR_BLOCKS_PER_REPEAT = 8;
constexpr int32_t kSrcNumPerRepeatOfVBRCB = 8;
constexpr uint16_t MAX_UINT16 = ((uint64_t)1 << 16) - 1;
constexpr uint32_t MAX_UINT32 = ((uint64_t)1 << 32) - 1;
constexpr uint32_t MAX_VCOPY_DST_REPEAT_STRIDE = 4096;
constexpr uint64_t MAX_UINT64 = ((uint64_t)1 << 63) - 1 + ((uint64_t)1 << 63);
constexpr uint64_t HALF_BITS = 16;
constexpr int64_t BITS_PER_BYTE = 8;
constexpr int32_t MAX_VBRCB_REPEAT_TIMES = 254;

#define WARN_SCALAR_IMPL(x) \
cce::printf("Warning: [%s] - This implementation uses scalar instructions, "\
  "which may result in suboptimal performance\n", (x))

enum class VectorOpTy : uint32_t {
  BINARY_VV_BEGIN = 100,
  VADD = 101,
  VSUB = 102,
  VMUL = 103,
  VDIV = 104,
  VMIN = 105,
  VMAX = 106,
  VAND = 107,
  VOR = 108,
  VCMP_EQ = 109,
  VCMP_NE = 110,
  VCMP_LT = 111,
  VCMP_GT = 112,
  VCMP_GE = 113,
  VCMP_LE = 114,
  VXOR = 115,
  BINARY_VV_END = 199,
  BINARY_VS_BEGIN = 200,
  VADDS = 201,
  VMULS = 202,
  VMINS = 203,
  VMAXS = 204,
  VSHL = 205,
  VSHR = 206,
  VCMPS_EQ = 207,
  VCMPS_NE = 208,
  VCMPS_LT = 209,
  VCMPS_GT = 210,
  VCMPS_GE = 211,
  VCMPS_LE = 212,
  BINARY_VS_END = 299,
  UNARY_BEGIN = 300,
  VLN = 301,
  VRELU = 302,
  VRSQRT = 303,
  VSQRT = 304,
  VEXP = 305,
  VREC = 306,
  VNOT = 307,
  VABS = 308,
  UNARY_END = 399,
};

enum class VectorLastAxisMode : uint32_t {
  VV = 0,
  VB = 1,
  BV = 2,
  BB = 3,
  VS = 4,
  BS = 5,
  SV = 6,
  SB = 7,
  V = 8,
  B = 9,
};

// Mode for vsel
// MODE0: src0 and src1 are vectors , cmp mask value is in cmpmask spr
// MODE1: src0 is vectors but src1 is constant number in cmpmask spr
//        cmp mask value is vector in ubuf of which
//        addr is specified by intrinsic pointer.
// MODE2: src0 and src1 are vectors,
//        cmp mask value is vector of which ubuf addr is stored in cmpmask spr.
enum class SelectMode : uint8_t {
  MODE0 = 0,
  MODE1 = 1,
  MODE2 = 2,
  END = 3,
};

// Mode for vreducev2 instruction
enum class PatternMode : uint8_t {
  NORMAL_MODE = 0,             // normal mode
  INDEX_0_FROM_2_ELEMENTS = 1, // 010101...01
  INDEX_1_FROM_2_ELEMENTS = 2, // 101010...10
  INDEX_0_FROM_4_ELEMENTS = 3, // 00010001...0001
  INDEX_1_FROM_4_ELEMENTS = 4, // 00100010...0010
  INDEX_2_FROM_4_ELEMENTS = 5, // 01000100...0100
  INDEX_3_FROM_4_ELEMENTS = 6, // 10001000...1000
  ALL_ELEMENTS = 7,            // 11111111...1111
};

template <size_t OPERANUM, typename SRC_T, typename DST_T = SRC_T>
struct intrin_args {
  __ubuf__ DST_T *dst;
  __ubuf__ SRC_T *src[OPERANUM];
  SRC_T scalar;
  uint64_t repeat;
  uint16_t dst_block_stride;
  uint16_t src_block_stride[OPERANUM];
  uint16_t dst_repeat_stride;
  uint16_t src_repeat_stride[OPERANUM];
};

template <size_t Dim>
__aiv__ __attribute__((always_inline)) bool is_no_op(int64_t sizes[Dim]) {
  for (uint64_t i = 0; i < Dim; ++i) {
    if (!sizes[i]) {
      return true;
    }
  }
  return false;
}

/// mask of num = (1 << num) - 1.
///
/// constraints:
/// 1. num <= 64
__aiv__ __attribute__((always_inline)) uint64_t getMaskOfNumber(int64_t num);

/// get max repeat times of vcmpv/vcmpvs intr
/// for normal vector intr, max repeat time is fixed 255, but for cmp vec op,
/// the max repeat time need guarantee dst operand alignment
template <typename T>
__aiv__ __attribute__((always_inline)) constexpr int32_t
get_vcmp_max_repeat_cnts() {
  constexpr int32_t dst_bytes_per_repeat =
      INTR_BYTES_PER_REPEAT / sizeof(T) / BITS_PER_BYTE;
  return FLOOR_FACTOR(dst_bytes_per_repeat * INTR_MAX_REPEAT_CNTS,
                      INTR_BYTES_PER_BLOCK) /
         dst_bytes_per_repeat;
}

template <
    typename T, PatternMode MODE,
    typename = typename std::enable_if<sizeof(T) == 2 || sizeof(T) == 4>::type>
__aiv__ __attribute__((always_inline)) void
vreducev2_2d_with_pattern_mode(memref_t<__ubuf__ T, 2> *src,
                               memref_t<__ubuf__ T, 2> *dst) {
  INTRINSIC_NO_ARGS(set_mask_count);
  const int64_t n = src->sizes[1];
  INTRINSIC(set_vector_mask, 0, n);

  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  __ubuf__ T *src_ptr = src->aligned + src->offset;
  int64_t src_size0 = src->sizes[0];
  int64_t src_stride0 = src->strides[0];
  int64_t dst_stride0 = dst->strides[0];
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  constexpr int max_repeat_time =
      FLOOR_FACTOR(INTR_MAX_REPEAT_CNTS, num_per_block);
  for (int64_t i = 0; i < src_size0 / max_repeat_time; i++) {
    if constexpr (sizeof(T) == 2) {
      INTRINSIC(
          vreducev2,
          (__ubuf__ uint16_t *)dst_ptr + i * max_repeat_time * dst_stride0,
          (__ubuf__ uint16_t *)src_ptr + i * max_repeat_time * src_stride0,
          (__ubuf__ uint16_t *)src_ptr + i * max_repeat_time * src_stride0,
          max_repeat_time,             // repeat times
          1,                           // sr0_block_stride
          static_cast<uint8_t>(MODE),  // pattern_mode
          src_stride0 / num_per_block, // src0_repeat_stride
          0                            // src1_repeat_stride
      );
    } else if constexpr (sizeof(T) == 4) {
      INTRINSIC(
          vreducev2,
          (__ubuf__ uint32_t *)dst_ptr + i * max_repeat_time * dst_stride0,
          (__ubuf__ uint32_t *)src_ptr + i * max_repeat_time * src_stride0,
          (__ubuf__ uint32_t *)src_ptr + i * max_repeat_time * src_stride0,
          max_repeat_time,             // repeat times
          1,                           // sr0_block_stride
          static_cast<uint8_t>(MODE),  // pattern_mode
          src_stride0 / num_per_block, // src0_repeat_stride
          0                            // src1_repeat_stride
      );
    }
  }

  if (src_size0 % max_repeat_time != 0) {
    uint8_t repeat_num = src_size0 % max_repeat_time;
    if constexpr (sizeof(T) == 2) {
      INTRINSIC(vreducev2,
                (__ubuf__ uint16_t *)dst_ptr +
                    src_size0 / max_repeat_time * max_repeat_time * dst_stride0,
                (__ubuf__ uint16_t *)src_ptr +
                    src_size0 / max_repeat_time * max_repeat_time * src_stride0,
                (__ubuf__ uint16_t *)src_ptr +
                    src_size0 / max_repeat_time * max_repeat_time * src_stride0,
                repeat_num,                  // repeat times
                1,                           // sr0_block_stride
                static_cast<uint8_t>(MODE),  // pattern_mode
                src_stride0 / num_per_block, // src0_repeat_stride
                0                            // src1_repeat_stride
      );
    } else if constexpr (sizeof(T) == 4) {
      INTRINSIC(vreducev2,
                (__ubuf__ uint32_t *)dst_ptr +
                    src_size0 / max_repeat_time * max_repeat_time * dst_stride0,
                (__ubuf__ uint32_t *)src_ptr +
                    src_size0 / max_repeat_time * max_repeat_time * src_stride0,
                (__ubuf__ uint32_t *)src_ptr +
                    src_size0 / max_repeat_time * max_repeat_time * src_stride0,
                repeat_num,                  // repeat times
                1,                           // sr0_block_stride
                static_cast<uint8_t>(MODE),  // pattern_mode
                src_stride0 / num_per_block, // src0_repeat_stride
                0                            // src1_repeat_stride
      );
    }
  }

  INTRINSIC_NO_ARGS(set_mask_norm);
}

template <typename T, PatternMode MODE>
__aiv__ __attribute__((always_inline)) void
vreducev2_1d_with_pattern_mode(memref_t<__ubuf__ T, 1> *src,
                               memref_t<__ubuf__ T, 1> *dst) {
  INTRINSIC_NO_ARGS(set_mask_count);
  const int64_t n = src->sizes[0];
  INTRINSIC(set_vector_mask, 0, n);

  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  __ubuf__ T *src_ptr = src->aligned + src->offset;

  if constexpr (sizeof(T) == 2) {
    INTRINSIC(vreducev2, (__ubuf__ uint16_t *)dst_ptr,
              (__ubuf__ uint16_t *)src_ptr, (__ubuf__ uint16_t *)src_ptr,
              1,                          // repeat times
              1,                          // sr0_block_stride
              static_cast<uint8_t>(MODE), // pattern_mode
              8,                          // src0_repeat_stride
              0                           // src1_repeat_stride
    );
  } else if constexpr (sizeof(T) == 4) {
    INTRINSIC(vreducev2, (__ubuf__ uint32_t *)dst_ptr,
              (__ubuf__ uint32_t *)src_ptr, (__ubuf__ uint32_t *)src_ptr,
              1,                          // repeat times
              1,                          // sr0_block_stride
              static_cast<uint8_t>(MODE), // pattern_mode
              8,                          // src0_repeat_stride
              0                           // src1_repeat_stride
    );
  } else {
    static_assert("vreducev2 instruction's unsupported data type");
  }

  INTRINSIC_NO_ARGS(set_mask_norm);
}

/// select channel0 from N channels support 2 scenarios:
/// 1. src (a, b) with stride [m, n] to dst (a, b) with stride [p, 1]
/// 'a' and 'b' are size of src, 'n' and 'm' are stride of src, 'p' is stride of
/// dst, where m = b*n and p == b
/// 2. src (a, m) with stride [m, 1] to dst (a, b) with stride [b, 1]
/// 'a' and 'm' are size of src, 'a' and 'b' are size of dst, 'n' is
/// channel_num, where m = b*n
template <typename T, int DIM>
__aiv__ __attribute__((always_inline)) void
select_channel0_from_block(memref_t<__ubuf__ T, DIM> *src,
                           memref_t<__ubuf__ T, DIM> *dst, int64_t repeat_times,
                           int64_t src_repeat_stride) {
  __ubuf__ T *src_ptr = src->aligned + src->offset;
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;

  INTRINSIC_NO_ARGS(set_mask_count);
  // set mask num as 1 to select one element per repeat
  INTRINSIC(set_vector_mask, 0, 1);

  if constexpr (sizeof(T) == 2) {
    INTRINSIC(vreducev2, (__ubuf__ uint16_t *)dst_ptr,
              (__ubuf__ uint16_t *)src_ptr, (__ubuf__ uint16_t *)src_ptr,
              repeat_times,
              1, // sr0_block_stride
              static_cast<uint8_t>(PatternMode::ALL_ELEMENTS), // pattern_mode
              src_repeat_stride, // src0_repeat_stride
              0                  // src1_repeat_stride
    );
  } else if constexpr (sizeof(T) == 4) {
    INTRINSIC(vreducev2, (__ubuf__ uint32_t *)dst_ptr,
              (__ubuf__ uint32_t *)src_ptr, (__ubuf__ uint32_t *)src_ptr,
              repeat_times,
              1, // sr0_block_stride
              static_cast<uint8_t>(PatternMode::ALL_ELEMENTS), // pattern_mode
              src_repeat_stride, // src0_repeat_stride
              0                  // src1_repeat_stride
    );
  } else {
    static_assert("vreducev2 instruction's unsupported data type");
  }

  INTRINSIC_NO_ARGS(set_mask_norm);
}

// view as new type. if sizeof(dst_type) is n multiple of size_of(src_type),
// the size will be larger when size of src is not multiple of n.
// note: bool type will be regarded as 1 bit
template <typename SRCTYPE, typename DSTTYPE, int Dim>
__aiv__ __attribute__((always_inline)) void
view_as(memref_t<__ubuf__ SRCTYPE, Dim> *src_buf,
        memref_t<__ubuf__ DSTTYPE, Dim> *dst_buf) {
  dst_buf->allocated = (__ubuf__ DSTTYPE *)src_buf->allocated;
  dst_buf->aligned = (__ubuf__ DSTTYPE *)src_buf->aligned;
  dst_buf->offset = src_buf->offset;
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(DSTTYPE);

  if constexpr (bitwidthOf<SRCTYPE>() == bitwidthOf<DSTTYPE>()) {
    for (int i = 0; i < Dim; ++i) {
      dst_buf->sizes[i] = src_buf->sizes[i];
      dst_buf->strides[i] = src_buf->strides[i];
    }
  } else if constexpr (bitwidthOf<SRCTYPE>() < bitwidthOf<DSTTYPE>()) {
    auto bits_factor = bitwidthOf<DSTTYPE>() / bitwidthOf<SRCTYPE>();
    dst_buf->offset = src_buf->offset / bits_factor;

    for (int i = 0; i < Dim; ++i) {
      if (i != Dim - 1) {
        dst_buf->sizes[i] = src_buf->sizes[i];
        dst_buf->strides[i] = CEIL_DIV(src_buf->strides[i], bits_factor);
      } else {
        dst_buf->sizes[i] = CEIL_DIV(src_buf->sizes[Dim - 1], bits_factor);
        dst_buf->strides[i] = src_buf->strides[i];
      }
    }
  } else if constexpr (bitwidthOf<SRCTYPE>() > bitwidthOf<DSTTYPE>()) {
    auto bits_factor = bitwidthOf<SRCTYPE>() / bitwidthOf<DSTTYPE>();
    dst_buf->offset = src_buf->offset * bits_factor;

    for (int i = 0; i < Dim; ++i) {
      if (i != Dim - 1) {
        dst_buf->sizes[i] = src_buf->sizes[i];
        dst_buf->strides[i] = src_buf->strides[i] * bits_factor;
      } else {
        dst_buf->sizes[i] = src_buf->sizes[Dim - 1] * bits_factor;
        dst_buf->strides[i] = src_buf->strides[i];
      }
    }
  }
}

template <typename SRCTYPE, typename DSTTYPE, int Dim>
__aiv__ __attribute__((always_inline)) void
bitwise_view_as(memref_t<__ubuf__ SRCTYPE, Dim> *src_buf,
                memref_t<__ubuf__ DSTTYPE, Dim> *dst_buf) {
  dst_buf->allocated = (__ubuf__ DSTTYPE *)src_buf->allocated;
  dst_buf->aligned = (__ubuf__ DSTTYPE *)src_buf->aligned;
  dst_buf->offset = src_buf->offset;
  if constexpr (bitwidthOf<SRCTYPE>() < bitwidthOf<DSTTYPE>()) {
    auto bits_factor = bitwidthOf<DSTTYPE>() / bitwidthOf<SRCTYPE>();
    dst_buf->offset = src_buf->offset / bits_factor;
  } else if constexpr (bitwidthOf<SRCTYPE>() > bitwidthOf<DSTTYPE>()) {
    auto bits_factor = bitwidthOf<SRCTYPE>() / bitwidthOf<DSTTYPE>();
    dst_buf->offset = src_buf->offset * bits_factor;
  }
  for (int i = 0; i < Dim; ++i) {
    dst_buf->sizes[i] = src_buf->sizes[i];
    dst_buf->strides[i] = src_buf->strides[i];
  }
}

// Convert buf_Nd (x0, x1, x2, ... , x(N-1)) to buf_Md (x(N-M), x(N-M+1), ... ,
// x(N-1))
template <typename T, size_t N, size_t M,
          typename = typename std::enable_if<N >= M>::type>
__aiv__ __attribute__((always_inline)) void
throw_Nd_to_Md_args(memref_t<__ubuf__ T, N> *buf_Nd,
                    memref_t<__ubuf__ T, M> *buf_Md) {
  buf_Md->allocated = buf_Nd->allocated;
  buf_Md->aligned = buf_Nd->aligned;
  buf_Md->offset = buf_Nd->offset;
  for (size_t i = N - M; i < N; ++i) {
    buf_Md->sizes[i - N + M] = buf_Nd->sizes[i];
    buf_Md->strides[i - N + M] = buf_Nd->strides[i];
  }
}

/// dst = vector_select(condition, src0, src1)
/// here condition_addr_buf is the ub buffer which store the addr of condition
template <typename T>
__aiv__ __attribute__((always_inline)) void
vector_select_1d(memref_t<__ubuf__ T, 1> *condition,
                 memref_t<__ubuf__ T, 1> *src0, memref_t<__ubuf__ T, 1> *src1,
                 memref_t<__ubuf__ T, 1> *dst,
                 memref_t<__ubuf__ T, 1> *condition_addr_buf) {
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  __ubuf__ T *src0_ptr = src0->aligned + src0->offset;
  __ubuf__ T *src1_ptr = src1->aligned + src1->offset;
  __ubuf__ T *condition_addr_ubuf_ptr =
      condition_addr_buf->aligned + condition_addr_buf->offset;
  __ubuf__ T *condition_ptr = condition->aligned + condition->offset;

  const int64_t total_size = dst->sizes[0];
  INTRINSIC_NO_ARGS(set_mask_count);
  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  (*(__ubuf__ uint64_t *)((__ubuf__ T *)condition_addr_ubuf_ptr)) =
      (uint64_t)((__ubuf__ T *)condition_ptr);
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);

  INTRINSIC(set_cmpmask, (__ubuf__ uint64_t *)condition_addr_ubuf_ptr);
  INTRINSIC(set_vector_mask, 0, total_size);
  INTRINSIC(pipe_barrier, PIPE_V);
  constexpr uint8_t dstBlockStride = 1;
  constexpr uint8_t src0BlockStride = 1;
  constexpr uint8_t src1BlockStride = 1;
  constexpr uint8_t dstRepeatStride = 8;
  constexpr uint8_t src0RepeatStride = 8;
  constexpr uint8_t src1RepeatStride = 8;
  constexpr uint8_t mode = static_cast<uint8_t>(SelectMode::MODE2);
  constexpr uint8_t repeat = 0;

  INTRINSIC(vsel, (__ubuf__ T *)dst_ptr, (__ubuf__ T *)src0_ptr,
            (__ubuf__ T *)src1_ptr, repeat, dstBlockStride, src0BlockStride,
            src1BlockStride, dstRepeatStride, src0RepeatStride,
            src1RepeatStride, mode);

  INTRINSIC_NO_ARGS(set_mask_norm);
}

template <typename SRC_TYPE, typename DST_TYPE = SRC_TYPE>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_vector_eltwise_vv_1d(memref_t<__ubuf__ SRC_TYPE, 1> *src0,
                                     memref_t<__ubuf__ SRC_TYPE, 1> *src1,
                                     memref_t<__ubuf__ DST_TYPE, 1> *dst) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  auto dst_ptr = dst->aligned + dst->offset;
  auto src0_ptr = src0->aligned + src0->offset;
  auto src1_ptr = src1->aligned + src1->offset;
  assert(
      (isAddress32ByteAligned(src0_ptr) && isAddress32ByteAligned(src1_ptr)) &&
      "The starting address of src must be 32byte aligned.");
  assert(isAddress32ByteAligned(dst_ptr) &&
         "The starting address of dst must be 32byte aligned.");
  assert((src0->strides[0] == 1 && src1->strides[0] == 1 &&
          dst->strides[0] == 1) &&
         "The src/dst must be continuous vector.");
#endif
}

template <typename SRC_TYPE, typename DST_TYPE = SRC_TYPE>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_vector_eltwise_v_1d(memref_t<__ubuf__ SRC_TYPE, 1> *src0,
                                    memref_t<__ubuf__ DST_TYPE, 1> *dst) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  auto dst_ptr = dst->aligned + dst->offset;
  auto src0_ptr = src0->aligned + src0->offset;
  assert(isAddress32ByteAligned(src0_ptr) &&
         "The starting address of src must be 32byte aligned.");
  assert(isAddress32ByteAligned(dst_ptr) &&
         "The starting address of dst must be 32byte aligned.");
  assert((src0->strides[0] == 1 && dst->strides[0] == 1) &&
         "The src/dst must be continuous vector.");
#endif
}

/// Get the preprocessed mode
/// The following are possible transformations:
/// VV -> VV/VB/BV/BB
/// VS -> VS/VB/BS/BB
/// SV -> BV/BB
/// V  -> V/B
/// B  -> B
/// BS -> BB
/// SB -> BB
/// VB -> VB/BB
/// BV -> BV/BB
/// BB -> BB
template <VectorOpTy OP>
__aiv__ __attribute__((always_inline)) VectorLastAxisMode
get_preprocessed_mode(VectorLastAxisMode mode, bool is_src0_brc_last_dim,
                      bool is_src1_brc_last_dim);

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void eltwise_vv_1d(
    memref_t<__ubuf__ T, 1> *src0, memref_t<__ubuf__ T, 1> *src1,
    memref_t<__ubuf__ T, 1> *dst, memref_t<__ubuf__ T, 1> *tmp_buf = nullptr,
    VectorLastAxisMode mode = VectorLastAxisMode::VV);

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void eltwise_vs_1d(
    memref_t<__ubuf__ T, 1> *src0, T scalar, memref_t<__ubuf__ T, 1> *dst,
    memref_t<__ubuf__ T, 1> *tmp_buf = nullptr, VectorLastAxisMode mode = VectorLastAxisMode::VS);

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void eltwise_sv_1d(
    T scalar, memref_t<__ubuf__ T, 1> *src1, memref_t<__ubuf__ T, 1> *dst,
    memref_t<__ubuf__ T, 1> *tmp_buf = nullptr, VectorLastAxisMode mode = VectorLastAxisMode::SV);

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void eltwise_v_1d(
    memref_t<__ubuf__ T, 1> *src0, memref_t<__ubuf__ T, 1> *dst,
    memref_t<__ubuf__ T, 1> *tmp_buf = nullptr, VectorLastAxisMode mode = VectorLastAxisMode::V);

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void vector_eltwise_vv_1d(
    memref_t<__ubuf__ T, 1> *src0, memref_t<__ubuf__ T, 1> *src1,
    memref_t<__ubuf__ T, 1> *dst, memref_t<__ubuf__ T, 1> *tmp_buf = nullptr,
    VectorLastAxisMode mode = VectorLastAxisMode::VV);

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vs_1d(memref_t<__ubuf__ T, 1> *src0, T scalar,
                     memref_t<__ubuf__ T, 1> *dst,
                     memref_t<__ubuf__ T, 1> *tmp_buf = nullptr,
                     VectorLastAxisMode mode = VectorLastAxisMode::VS);

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_sv_1d(T scalar, memref_t<__ubuf__ T, 1> *src1,
                     memref_t<__ubuf__ T, 1> *dst,
                     memref_t<__ubuf__ T, 1> *tmp_buf = nullptr,
                     VectorLastAxisMode mode = VectorLastAxisMode::SV);

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_v_1d(memref_t<__ubuf__ T, 1> *src0, memref_t<__ubuf__ T, 1> *dst,
                    memref_t<__ubuf__ T, 1> *tmp_buf = nullptr,
                    VectorLastAxisMode mode = VectorLastAxisMode::V);

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_v_1d<VectorOpTy::VNOT, bool>(memref_t<__ubuf__ bool, 1> *src0,
                                            memref_t<__ubuf__ bool, 1> *dst,
                                            memref_t<__ubuf__ bool, 1> *tmp_buf,
                                            VectorLastAxisMode mode);

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_2d_intrin(intrin_args<2, T> args);

#define REGISTE_ELTWISE_2D_INTRIN(op, type)                                    \
  template __aiv__ __attribute__((always_inline)) void                         \
  vector_eltwise_2d_intrin<op, type>(intrin_args<2, type> args)

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void eltwise_vv_2d(
    memref_t<__ubuf__ T, 2> *src0, memref_t<__ubuf__ T, 2> *src1, memref_t<__ubuf__ T, 2> *dst,
    memref_t<__ubuf__ T, 1> *tmp_buf = nullptr, VectorLastAxisMode mode = VectorLastAxisMode::VV);

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void eltwise_vs_2d(
    memref_t<__ubuf__ T, 2> *src0, T scalar, memref_t<__ubuf__ T, 2> *dst,
    memref_t<__ubuf__ T, 1> *tmp_buf = nullptr, VectorLastAxisMode mode = VectorLastAxisMode::VS);

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void eltwise_sv_2d(
    T scalar, memref_t<__ubuf__ T, 2> *src1, memref_t<__ubuf__ T, 2> *dst,
    memref_t<__ubuf__ T, 1> *tmp_buf = nullptr, VectorLastAxisMode mode = VectorLastAxisMode::SV);

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void eltwise_v_2d(
    memref_t<__ubuf__ T, 2> *src0, memref_t<__ubuf__ T, 2> *dst,
    memref_t<__ubuf__ T, 1> *tmp_buf = nullptr, VectorLastAxisMode mode = VectorLastAxisMode::V);

template <VectorOpTy OP, typename T, size_t OPERANUM>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_2d(memref_t<__ubuf__ T, 2> *src0, memref_t<__ubuf__ T, 2> *src1,
                  T scalar, memref_t<__ubuf__ T, 2> *dst,
                  VectorLastAxisMode mode);

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void vector_eltwise_vv_2d(
    memref_t<__ubuf__ T, 2> *src0, memref_t<__ubuf__ T, 2> *src1,
    memref_t<__ubuf__ T, 2> *dst, memref_t<__ubuf__ T, 1> *tmp_buf = nullptr,
    VectorLastAxisMode mode = VectorLastAxisMode::VV);

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vs_2d(memref_t<__ubuf__ T, 2> *src0, T scalar,
                     memref_t<__ubuf__ T, 2> *dst,
                     memref_t<__ubuf__ T, 1> *tmp_buf = nullptr,
                     VectorLastAxisMode mode = VectorLastAxisMode::VS);

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_sv_2d(T scalar, memref_t<__ubuf__ T, 2> *src1,
                     memref_t<__ubuf__ T, 2> *dst,
                     memref_t<__ubuf__ T, 1> *tmp_buf = nullptr,
                     VectorLastAxisMode mode = VectorLastAxisMode::SV);

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_v_2d(memref_t<__ubuf__ T, 2> *src0, memref_t<__ubuf__ T, 2> *dst,
                    memref_t<__ubuf__ T, 1> *tmp_buf = nullptr,
                    VectorLastAxisMode mode = VectorLastAxisMode::V);

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void eltwise_vv_3d(
    memref_t<__ubuf__ T, 3> *src0, memref_t<__ubuf__ T, 3> *src1, memref_t<__ubuf__ T, 3> *dst,
    memref_t<__ubuf__ T, 1> *tmp_buf = nullptr, VectorLastAxisMode mode = VectorLastAxisMode::VV);

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void eltwise_vs_3d(
    memref_t<__ubuf__ T, 3> *src0, T scalar, memref_t<__ubuf__ T, 3> *dst,
    memref_t<__ubuf__ T, 1> *tmp_buf = nullptr, VectorLastAxisMode mode = VectorLastAxisMode::VS);

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void eltwise_sv_3d(
    T scalar, memref_t<__ubuf__ T, 3> *src1, memref_t<__ubuf__ T, 3> *dst,
    memref_t<__ubuf__ T, 1> *tmp_buf = nullptr, VectorLastAxisMode mode = VectorLastAxisMode::SV);

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void eltwise_v_3d(
    memref_t<__ubuf__ T, 3> *src1, memref_t<__ubuf__ T, 3> *dst,
    memref_t<__ubuf__ T, 1> *tmp_buf = nullptr, VectorLastAxisMode mode = VectorLastAxisMode::V);

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void vector_eltwise_vv_3d(
    memref_t<__ubuf__ T, 3> *src0, memref_t<__ubuf__ T, 3> *src1,
    memref_t<__ubuf__ T, 3> *dst, memref_t<__ubuf__ T, 1> *tmp_buf = nullptr,
    VectorLastAxisMode mode = VectorLastAxisMode::VV);

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vs_3d(memref_t<__ubuf__ T, 3> *src0, T scalar,
                     memref_t<__ubuf__ T, 3> *dst,
                     memref_t<__ubuf__ T, 1> *tmp_buf = nullptr,
                     VectorLastAxisMode mode = VectorLastAxisMode::VS);

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_sv_3d(T scalar, memref_t<__ubuf__ T, 3> *src1,
                     memref_t<__ubuf__ T, 3> *dst,
                     memref_t<__ubuf__ T, 1> *tmp_buf = nullptr,
                     VectorLastAxisMode mode = VectorLastAxisMode::SV);

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_v_3d(memref_t<__ubuf__ T, 3> *src0, memref_t<__ubuf__ T, 3> *dst,
                    memref_t<__ubuf__ T, 1> *tmp_buf = nullptr,
                    VectorLastAxisMode mode = VectorLastAxisMode::V);

/// Deal vector elementwise 2d last axis brc inline.
/// src:(a, 1) -> dst:(a, nuumper_block)
template <typename SRC_T, typename DST_T = SRC_T>
__aiv__ __attribute__((always_inline)) void
vector_last_axis_brc_2d(memref_t<__ubuf__ SRC_T, 2> *src,
                        memref_t<__ubuf__ DST_T, 2> *dst,
                        memref_t<__ubuf__ SRC_T, 1> *tmp_buf);

/// Calculate logarithm of base 2 value.
///
/// constraints:
/// 1. max value <= ((int64_t)1 << 31) - 1.
/// 2. The size of value is a power of 2.
template <int VALUE>
__aiv__ __attribute__((always_inline)) constexpr int Log2() {
  int32_t sum = 1;
  constexpr int32_t MAX_VALUE = ((int64_t)1 << 31) - 1;
  static_assert(VALUE <= MAX_VALUE);
  int64_t i = 0;
  for (i = 0; sum < VALUE; i++) {
    sum = sum << 1;
  }
  return i;
}

/// Calculate the largest n that 2^n <= value.
///
/// constraints:
/// 1. max value <= ((int64_t)1 << 31) - 1.
__aiv__ __attribute__((always_inline)) constexpr int Log2(int64_t size) {
  int64_t sum = 1;
  int64_t i = 0;
  for (i = 0; (sum << 1) <= size; ++i) {
    sum = sum << 1;
  }
  return i;
}

// fast floor_log2 using builtin
__aiv__ __attribute__((always_inline)) constexpr int floor_log2(uint64_t v) {
  // TODO: add condition / assert
  return 63 - __builtin_clzll(v);
}

// returns minimal n >= 0 such that (2^k)^n >= ceil(S / T)
// i.e. minimal n with 2^{k*n} >= R, where R = ceil(S/T)
// S: original size (>= 0), T: target size (>= 1), k: power-of-two exponent (>=
// 1)
__aiv__ __attribute__((always_inline)) constexpr int
divisions_needed_by_pow2(int64_t S, int64_t T, int k) {
  // TODO: add condition / assert
  uint64_t R = CEIL_DIV(S, T);
  int L = floor_log2(R);
  // is exact power of two ?
  bool is_pow2 = ((uint64_t)1 << L) == R;
  // ceil(log2(R)) = L if R is power of two else L+1
  int ceil_log2_R = L + (is_pow2 ? 0 : 1);
  int n = (ceil_log2_R + k - 1) / k;
  return n;
}

/// Calculate n^m.
///
/// constraints:
/// 1. n^m <= UINT64_MAX
__aiv__ __attribute__((always_inline)) constexpr uint64_t pow(uint64_t n,
                                                              uint64_t m) {
  uint64_t sum = 1;
  for (uint64_t i = 0; i < m; i++) {
    sum = sum * n;
  }
  return sum;
}

/// Take the r0 numbers sequentially starting from the low bit.
/// This is a general way to set the mask by count.
///
/// constraints:
/// 1. num <= 128
__aiv__ __attribute__((always_inline)) void SetMaskValueByCount(int64_t num);

template <VectorOpTy OP, typename SRC_TYPE, typename DST_TYPE = SRC_TYPE>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_intrin(intrin_args<2, SRC_TYPE, DST_TYPE> args) {
#define ELTWISE_VV_ARGS                                                        \
  args.dst, args.src[0], args.src[1], args.repeat, args.dst_block_stride,      \
      args.src_block_stride[0], args.src_block_stride[1],                      \
      args.dst_repeat_stride, args.src_repeat_stride[0],                       \
      args.src_repeat_stride[1]

  if constexpr (OP == VectorOpTy::VADD) {
    INTRINSIC(vadd, ELTWISE_VV_ARGS);
  } else if constexpr (OP == VectorOpTy::VSUB) {
    INTRINSIC(vsub, ELTWISE_VV_ARGS);
  } else if constexpr (OP == VectorOpTy::VMUL) {
    INTRINSIC(vmul, ELTWISE_VV_ARGS);
  } else if constexpr (OP == VectorOpTy::VDIV) {
    INTRINSIC(vdiv, ELTWISE_VV_ARGS);
  } else if constexpr (OP == VectorOpTy::VMIN) {
    INTRINSIC(vmin, ELTWISE_VV_ARGS);
  } else if constexpr (OP == VectorOpTy::VMAX) {
    INTRINSIC(vmax, ELTWISE_VV_ARGS);
  } else if constexpr (OP == VectorOpTy::VAND) {
    INTRINSIC(vand, ELTWISE_VV_ARGS);
  } else if constexpr (OP == VectorOpTy::VOR) {
    INTRINSIC(vor, ELTWISE_VV_ARGS);
  } else if constexpr (OP == VectorOpTy::VCMP_EQ) {
    INTRINSIC(vcmpv_eq, ELTWISE_VV_ARGS);
  } else if constexpr (OP == VectorOpTy::VCMP_NE) {
    INTRINSIC(vcmpv_ne, ELTWISE_VV_ARGS);
  } else if constexpr (OP == VectorOpTy::VCMP_LT) {
    INTRINSIC(vcmpv_lt, ELTWISE_VV_ARGS);
  } else if constexpr (OP == VectorOpTy::VCMP_GT) {
    INTRINSIC(vcmpv_gt, ELTWISE_VV_ARGS);
  } else if constexpr (OP == VectorOpTy::VCMP_GE) {
    INTRINSIC(vcmpv_ge, ELTWISE_VV_ARGS);
  } else if constexpr (OP == VectorOpTy::VCMP_LE) {
    INTRINSIC(vcmpv_le, ELTWISE_VV_ARGS);
  }
}

template <VectorOpTy OP, typename SRC_TYPE, typename DST_TYPE = SRC_TYPE>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vs_intrin(intrin_args<1, SRC_TYPE, DST_TYPE> args) {
#define ELTWISE_VS_ARGS                                                        \
  args.dst, args.src[0], args.scalar, args.repeat, args.dst_block_stride,      \
      args.src_block_stride[0], args.dst_repeat_stride,                        \
      args.src_repeat_stride[0]

  if constexpr (OP == VectorOpTy::VADDS) {
    INTRINSIC(vadds, ELTWISE_VS_ARGS);
  } else if constexpr (OP == VectorOpTy::VMULS) {
    INTRINSIC(vmuls, ELTWISE_VS_ARGS);
  } else if constexpr (OP == VectorOpTy::VMINS) {
    INTRINSIC(vmins, ELTWISE_VS_ARGS);
  } else if constexpr (OP == VectorOpTy::VMAXS) {
    INTRINSIC(vmaxs, ELTWISE_VS_ARGS);
  } else if constexpr (OP == VectorOpTy::VSHL) {
    INTRINSIC(vshl, ELTWISE_VS_ARGS);
  } else if constexpr (OP == VectorOpTy::VSHR) {
    // TODO support arithmetic shit right (i16/i32)
    // set round mode as 0 by default (logic shift right)
    INTRINSIC(vshr, ELTWISE_VS_ARGS, 0);
  } else if constexpr (OP == VectorOpTy::VCMPS_EQ) {
    INTRINSIC(vcmpvs_eq, ELTWISE_VS_ARGS);
  } else if constexpr (OP == VectorOpTy::VCMPS_NE) {
    INTRINSIC(vcmpvs_ne, ELTWISE_VS_ARGS);
  } else if constexpr (OP == VectorOpTy::VCMPS_LT) {
    INTRINSIC(vcmpvs_lt, ELTWISE_VS_ARGS);
  } else if constexpr (OP == VectorOpTy::VCMPS_GT) {
    INTRINSIC(vcmpvs_gt, ELTWISE_VS_ARGS);
  } else if constexpr (OP == VectorOpTy::VCMPS_GE) {
    INTRINSIC(vcmpvs_ge, ELTWISE_VS_ARGS);
  } else if constexpr (OP == VectorOpTy::VCMPS_LE) {
    INTRINSIC(vcmpvs_le, ELTWISE_VS_ARGS);
  }
}

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_v_intrin(intrin_args<1, T> args) {
#define ELTWISE_V_ARGS                                                         \
  args.dst, args.src[0], args.repeat, args.dst_block_stride,                   \
      args.src_block_stride[0], args.dst_repeat_stride,                        \
      args.src_repeat_stride[0]

  if constexpr (OP == VectorOpTy::VABS) {
    INTRINSIC(vabs, ELTWISE_V_ARGS);
  } else if constexpr (OP == VectorOpTy::VLN) {
    INTRINSIC(vln, ELTWISE_V_ARGS);
  } else if constexpr (OP == VectorOpTy::VRELU) {
    INTRINSIC(vrelu, ELTWISE_V_ARGS);
  } else if constexpr (OP == VectorOpTy::VRSQRT) {
    INTRINSIC(vrsqrt, ELTWISE_V_ARGS);
  } else if constexpr (OP == VectorOpTy::VSQRT) {
    INTRINSIC(vsqrt, ELTWISE_V_ARGS);
  } else if constexpr (OP == VectorOpTy::VEXP) {
    INTRINSIC(vexp, ELTWISE_V_ARGS);
  } else if constexpr (OP == VectorOpTy::VREC) {
    INTRINSIC(vrec, ELTWISE_V_ARGS);
  } else if constexpr (OP == VectorOpTy::VNOT) {
    INTRINSIC(vnot, ELTWISE_V_ARGS);
  }
}

// Adaptation for XOR: Since there are no direct hardware instructions for XOR
// operations, multiple instructions need to be combined.
template <typename T>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_xor_intrin(intrin_args<2, T> args,
                          memref_t<__ubuf__ T, 1> *tmp_buf) {
  __ubuf__ T *tmp_buf_ptr = tmp_buf->aligned + tmp_buf->offset;
  // A xor B = !(A & B) & (A | B)
  vector_eltwise_vv_intrin<VectorOpTy::VAND, T>(intrin_args<2, T>{
      tmp_buf_ptr,
      {args.src[0], args.src[1]},
      args.scalar,
      args.repeat,
      1,
      {args.src_block_stride[0], args.src_block_stride[1]},
      INTR_BLOCKS_PER_REPEAT,
      {args.src_repeat_stride[0], args.src_repeat_stride[1]}});
  INTRINSIC(pipe_barrier, PIPE_V);
  vector_eltwise_v_intrin<VectorOpTy::VNOT, T>(
      intrin_args<1, T>{tmp_buf_ptr,
                        {tmp_buf_ptr},
                        args.scalar,
                        args.repeat,
                        1,
                        {1},
                        INTR_BLOCKS_PER_REPEAT,
                        {INTR_BLOCKS_PER_REPEAT}});

  INTRINSIC(pipe_barrier, PIPE_V);
  vector_eltwise_vv_intrin<VectorOpTy::VOR, T>(intrin_args<2, T>{
      args.dst,
      {args.src[0], args.src[1]},
      args.scalar,
      args.repeat,
      args.dst_block_stride,
      {args.src_block_stride[0], args.src_block_stride[1]},
      args.dst_repeat_stride,
      {args.src_repeat_stride[0], args.src_repeat_stride[1]}});
  INTRINSIC(pipe_barrier, PIPE_V);
  vector_eltwise_vv_intrin<VectorOpTy::VAND, T>(
      intrin_args<2, T>{args.dst,
                        {args.dst, tmp_buf_ptr},
                        args.scalar,
                        args.repeat,
                        args.dst_block_stride,
                        {args.dst_block_stride, 1},
                        args.dst_repeat_stride,
                        {args.dst_repeat_stride, INTR_BLOCKS_PER_REPEAT}});
}

/// compare two 1D memrefs
/// The result is stored in the location to which the ub_mask_ptr points.
template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
vector_cmp(memref_t<__ubuf__ T, 1> *src0, memref_t<__ubuf__ T, 1> *dst_value,
           __ubuf__ uint8_t *ub_mask_ptr) {
  __ubuf__ T *src_ptr = src0->aligned + src0->offset;
  __ubuf__ T *dst_value_ptr = dst_value->aligned + dst_value->offset;

  const int64_t size = src0->sizes[0];
  const int64_t last_dim_align_repeat =
      (size * sizeof(T) + INTR_BYTES_PER_REPEAT - 1) / INTR_BYTES_PER_REPEAT;

  // to get the new index
  // update the result in the dst index buffer
  // get the result of cmp and store it in ub_mask
  INTRINSIC_NO_ARGS(set_mask_count);
  INTRINSIC(set_vector_mask, 0x0, size);
  if constexpr (OP == VectorOpTy::VCMP_LT) {
    INTRINSIC(vcmpv_lt, (__ubuf__ uint8_t *)ub_mask_ptr,
              (__ubuf__ T *)dst_value_ptr, (__ubuf__ T *)src_ptr,
              last_dim_align_repeat,  // repeat
              1,                      // dstBlockStride
              1,                      // src0BlockStride
              1,                      // src1BlockStride
              0,                      // dstRepeatStride unused
              INTR_BLOCKS_PER_REPEAT, // src0RepeatStride
              INTR_BLOCKS_PER_REPEAT  // src1RepeatStride
    );
  } else if constexpr (OP == VectorOpTy::VCMP_GT) {
    INTRINSIC(vcmpv_gt, (__ubuf__ uint8_t *)ub_mask_ptr,
              (__ubuf__ T *)dst_value_ptr, (__ubuf__ T *)src_ptr,
              last_dim_align_repeat, 1, 1, 1, 0, INTR_BLOCKS_PER_REPEAT,
              INTR_BLOCKS_PER_REPEAT);
  } else if constexpr (OP == VectorOpTy::VCMP_GE) {
    INTRINSIC(vcmpv_ge, (__ubuf__ uint8_t *)ub_mask_ptr,
              (__ubuf__ T *)dst_value_ptr, (__ubuf__ T *)src_ptr,
              last_dim_align_repeat, 1, 1, 1, 0, INTR_BLOCKS_PER_REPEAT,
              INTR_BLOCKS_PER_REPEAT);
  } else if constexpr (OP == VectorOpTy::VCMP_LE) {
    INTRINSIC(vcmpv_le, (__ubuf__ uint8_t *)ub_mask_ptr,
              (__ubuf__ T *)dst_value_ptr, (__ubuf__ T *)src_ptr,
              last_dim_align_repeat, 1, 1, 1, 0, INTR_BLOCKS_PER_REPEAT,
              INTR_BLOCKS_PER_REPEAT);
  } else {
    static_assert("Unsupported vector_cmp argument");
  }

  INTRINSIC_NO_ARGS(set_mask_norm);
}

template <VectorOpTy OP>
__aiv__ __attribute__((always_inline)) constexpr bool isHardwareSupportedVS() {
  if constexpr (OP > VectorOpTy::BINARY_VS_BEGIN &&
                OP < VectorOpTy::BINARY_VS_END) {
    return true;
  }
  return false;
}

#define DECLARE_ELTWISE_VV(op_name, op_type, dim, dtype)                       \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_##op_name##_##dim##d_##dtype(                                   \
      memref_t<__ubuf__ dtype, dim> *src0,                                     \
      memref_t<__ubuf__ dtype, dim> *src1, memref_t<__ubuf__ dtype, dim> *dst, \
      memref_t<__ubuf__ dtype, 1> *tmp_buf = nullptr)

#define DECLARE_ELTWISE_VV_WITHOUT_DEFAULT_VALUE(op_name, op_type, dim, dtype) \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_##op_name##_##dim##d_##dtype(                                   \
      memref_t<__ubuf__ dtype, dim> *src0,                                     \
      memref_t<__ubuf__ dtype, dim> *src1, memref_t<__ubuf__ dtype, dim> *dst, \
      memref_t<__ubuf__ dtype, 1> *tmp_buf)

#define REGISTE_ELTWISE_VV(op_name, op_type, dim, dtype)                       \
  DECLARE_ELTWISE_VV_WITHOUT_DEFAULT_VALUE(op_name, op_type, dim, dtype) {     \
    eltwise_vv_##dim##d<op_type, dtype>(src0, src1, dst, tmp_buf);      \
  }

#define DECLARE_ELTWISE_VS(op_name, op_type, dim, dtype)                       \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_##op_name##_vs_##dim##d_##dtype(                                \
      memref_t<__ubuf__ dtype, dim> *src0, dtype scalar,                       \
      memref_t<__ubuf__ dtype, dim> *dst,                                      \
      memref_t<__ubuf__ dtype, 1> *tmp_buf)

#define REGISTE_ELTWISE_VS(op_name, op_type, dim, dtype)                       \
  DECLARE_ELTWISE_VS(op_name, op_type, dim, dtype) {                           \
    eltwise_vs_##dim##d<op_type, dtype>(src0, scalar, dst, tmp_buf);    \
  }

#define DECLARE_ELTWISE_SV(op_name, op_type, dim, dtype)                       \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_##op_name##_sv_##dim##d_##dtype(                                \
      dtype scalar, memref_t<__ubuf__ dtype, dim> *src1,                       \
      memref_t<__ubuf__ dtype, dim> *dst,                                      \
      memref_t<__ubuf__ dtype, 1> *tmp_buf)

#define REGISTE_ELTWISE_SV(op_name, op_type, dim, dtype)                       \
  DECLARE_ELTWISE_SV(op_name, op_type, dim, dtype) {                           \
    eltwise_sv_##dim##d<op_type, dtype>(scalar, src1, dst, tmp_buf);    \
  }

#define DECLARE_ELTWISE_V(op_name, op_type, dim, dtype)                        \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_##op_name##_##dim##d_##dtype(                                   \
      memref_t<__ubuf__ dtype, dim> *src0, memref_t<__ubuf__ dtype, dim> *dst, \
      memref_t<__ubuf__ dtype, 1> *tmp_buf)

#define REGISTE_ELTWISE_V(op_name, op_type, dim, dtype)                        \
  DECLARE_ELTWISE_V(op_name, op_type, dim, dtype) {                            \
    eltwise_v_##dim##d<op_type, dtype>(src0, dst, tmp_buf);             \
  }

template <typename T>
__aiv__ __attribute__((always_inline)) void move_memref_to_aligned_1d(memref_t<__ubuf__ T, 1> *mref, int steps) {
  mref->aligned += steps + mref->offset;
  mref->allocated += steps + mref->offset;
  mref->offset = 0;
  if (mref->sizes[0] != 1) {
    if (mref->sizes[0] < steps) {
      mref->sizes[0] = 0;
    } else {
      mref->sizes[0] -= steps;
    }
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) void move_memref_to_aligned_2d(memref_t<__ubuf__ T, 2> *mref, int steps) {
  mref->aligned += steps + mref->offset;
  mref->allocated += steps + mref->offset;
  mref->offset = 0;
  if (mref->sizes[1] != 1) {
    if (mref->sizes[1] < steps) {
      mref->sizes[1] = 0;
    } else {
      mref->sizes[1] -= steps;
    }
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) void move_memref_to_aligned_3d(memref_t<__ubuf__ T, 3> *mref, int steps) {
  mref->aligned += steps + mref->offset;
  mref->allocated += steps + mref->offset;
  mref->offset = 0;
  if (mref->sizes[2] != 1) {
    if (mref->sizes[2] < steps) {
      mref->sizes[2] = 0;
    } else {
      mref->sizes[2] -= steps;
    }
  }
}

// Get the number of elements that need to be processed using scalar operations.
// When the memory alignment of input or output is not suitable for vectorization, 
// some elements must be handled using scalar operations util the rest element is aligned with 32 bytes.
template <typename T, size_t DIM>
__aiv__ __attribute__((always_inline)) int64_t eltwise_get_element_nums_on_scalar(
    memref_t<__ubuf__ T, DIM> *src0, memref_t<__ubuf__ T, DIM> *src1, memref_t<__ubuf__ T, DIM> *dst) {
  // Check strides for the last dimension
  auto src0_last_stride = src0 == nullptr ? 1 : src0->strides[DIM - 1];
  auto src1_last_stride = src1 == nullptr ? 1 : src1->strides[DIM - 1];
  auto dst_last_stride = dst->strides[DIM - 1];

  if (src0_last_stride != 1 || src1_last_stride != 1 || dst_last_stride != 1) {
    return dst->sizes[DIM - 1];
  }

  // Check if all other dimensions' strides are aligned to block size
  for (size_t i = 0; i < DIM - 1; ++i) {
    auto src0_stride = src0 == nullptr ? 0 : src0->strides[i];
    auto src1_stride = src1 == nullptr ? 0 : src1->strides[i];
    auto dst_stride = dst->strides[i];
    if ((!isSizeAlignedToBlock<T>(src0_stride) && src0_stride != 1) ||
        (!isSizeAlignedToBlock<T>(src1_stride) && src1_stride != 1) ||
        (!isSizeAlignedToBlock<T>(dst_stride) && dst_stride != 1)) {
      return dst->sizes[DIM - 1];
    }
  }

  // Check offset alignment
  bool is_src0_aligned = src0 == nullptr ? true : isAddress32ByteAligned(src0->aligned + src0->offset);
  bool is_src1_aligned = src1 == nullptr ? true : isAddress32ByteAligned(src1->aligned + src1->offset);
  bool is_dst_aligned = isAddress32ByteAligned(dst->aligned + dst->offset);

  if (!is_src0_aligned || !is_src1_aligned || !is_dst_aligned) {
    auto src0_offset = src0 == nullptr ? dst->offset : src0->offset;
    auto src1_offset = src1 == nullptr ? dst->offset : src1->offset;
    auto dst_offset = dst->offset;

    if ((src0_offset * sizeof(T)) % INTR_BYTES_PER_BLOCK != (dst_offset * sizeof(T)) % INTR_BYTES_PER_BLOCK ||
        (src1_offset * sizeof(T)) % INTR_BYTES_PER_BLOCK != (dst_offset * sizeof(T)) % INTR_BYTES_PER_BLOCK) {
      return dst->sizes[DIM - 1];
    }

    auto address = reinterpret_cast<uintptr_t>(dst->aligned + dst->offset);
    auto align_diff = (INTR_BYTES_PER_BLOCK - (address & 0x1F)) & 0x1F;
    auto scalar_num = align_diff / sizeof(T);
    return scalar_num;
  }

  return 0;
}

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) T scalar_operation(T src0_oprand, T src1_oprand) {
  if constexpr (OP == VectorOpTy::VADD || OP == VectorOpTy::VADDS) {
    return src0_oprand + src1_oprand;
  } else if constexpr (OP == VectorOpTy::VSUB) {
    return src0_oprand - src1_oprand;
  } else if constexpr (OP == VectorOpTy::VMUL || OP == VectorOpTy::VMULS) {
    return src0_oprand * src1_oprand;
  } else if constexpr (OP == VectorOpTy::VDIV) {
    return src0_oprand / src1_oprand;
  } else if constexpr (OP == VectorOpTy::VMIN || OP == VectorOpTy::VMINS) {
    return src0_oprand > src1_oprand ? src1_oprand : src0_oprand;
  } else if constexpr (OP == VectorOpTy::VMAX || OP == VectorOpTy::VMAXS) {
    return src0_oprand > src1_oprand ? src0_oprand : src1_oprand;
  } else if constexpr (OP == VectorOpTy::VAND) {
    return src0_oprand & src1_oprand;
  } else if constexpr (OP == VectorOpTy::VOR) {
    return src0_oprand | src1_oprand;
  } else if constexpr (OP == VectorOpTy::VSHL) {
    return src0_oprand << src1_oprand;
  } else if constexpr (OP == VectorOpTy::VSHR) {
    return src0_oprand >> src1_oprand;
  } else if constexpr (OP == VectorOpTy::VXOR) {
    return src0_oprand ^ src1_oprand;
  } else {
    static_assert("unsupported elementwise op");
  }
  return src0_oprand;
}

template <typename T>
__aiv__ __attribute__((always_inline)) T scalar_exp(T x) {
  // Boundary Value Handling
  if (x > (T)88.7f) {
    return std::numeric_limits<T>::infinity();
  }
  if (x < (T)-88.7f) {
    return (T)0.0;
  }
 
  // constant define
  const T LOG2E    = 1.4426950408889634f;
  // High part of ln(2) for precision compensation
  const T LN2_HI   = 0.693145751953125f;
  // low part of ln(2) for precision compensation
  const T LN2_LO   = 1.4286068203094172e-6f;
  const T MAGIC    = 12582912.0f;           // 1.5 * 2^23
 
  // Fast range reduction
  // n = round(x / ln2)
  T z = x * LOG2E + MAGIC;
  uint32_t vi = *(uint32_t*)&z;
  int n = (int)(vi & 0x7FFFFF) - 0x400000; 
  T fn = z - MAGIC;
 
  // r = x - n * ln2
  T r = x - fn * LN2_HI;
  r = r - fn * LN2_LO;
 
  // Remez-optimized polynomial, 5th order term for better robustness
  // Approximate e^r ≈ 1 + c1*r + c2*r^2 + c3*r^3 + c4*r^4 + c5*r^5
  const T c1 = 1.0000000000f;
  const T c2 = 0.4999999206f;
  const T c3 = 0.1666653019f;
  const T c4 = 0.0416573475f;
  const T c5 = 0.0083013598f;
 
  T r2 = r * r;
  T res_r = 1.0f + r * (c1 + r * (c2 + r * (c3 + r * (c4 + r * c5))));
 
  // Exponent reconstruction (2^n)
  uint32_t ei = (uint32_t)(n + 127) << 23;
  T two_n = *(T*)&ei;
 
  // e^r * 2^n
  return (T)(res_r * two_n);
}
 
template <typename T>
__aiv__ __attribute__((always_inline)) T scalar_sqrt(T x) {
  // Boundary Value Handling
  if (x < (T)0.0) {
    return std::numeric_limits<T>::quiet_NaN();
  }
  
  const T kEpsilon = 1e-7;
  // -> 0
  if (x < kEpsilon) {
    return (T)0.0;
  }

  if (x == std::numeric_limits<T>::infinity()) {
    return std::numeric_limits<T>::infinity();
  }

  // fast and accurate initial guess for sqrt(x)
  // 0x1fbd1df5: A magic constant for an IEEE 754 bit-manipulation based fast square root approximation
  union {
      float f;
      uint32_t i;
  } v;
  v.f = (float)x;
  v.i = 0x1fbd1df5 + (v.i >> 1); 
  T guess = (T)v.f;

  
  const T eps = (T)1e-5;
  // Newton-Raphson Iteration with Epsilon Control
  for (int i = 0; i < 6; ++i) {
      T last_guess = guess;
      guess = (T)0.5 * (guess + x / guess);

      T abs_num = guess - last_guess > 0 ? guess - last_guess : last_guess - guess;
      if (abs_num <= eps * guess) {
          break;
      }
  }

  return guess;
}
 
template <typename T>
__aiv__ __attribute__((always_inline)) T scalar_rsqrt(T x) {
  // Boundary Value Handling
  if (x < (T)0.0f) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  const T kEpsilon = 1e-7;
  // -> 0
  if (x < kEpsilon) {
    return std::numeric_limits<T>::infinity();
  }
      
  float xf = (float)x;

  // Initial approximation
  union {
    float f;
    uint32_t i;
  } v;
  v.f = xf;
  // magic number
  v.i = 0x5f375a86 - (v.i >> 1); 
  float y = v.f;

  // Two round of Newton-Raphson iteration to ensure accuracy within 0.01%
  // y = y * (1.5 - 0.5 * x * y^2)
  float x_half = xf * 0.5f;
  y = y * (1.5f - x_half * y * y);
  y = y * (1.5f - x_half * y * y);

  return (T)y;
}
 
 
template <typename T>
__aiv__ __attribute__((always_inline)) T scalar_ln(T x) {
  // Boundary Value Handling
  if (x < (T)0.0) {
      return std::numeric_limits<T>::quiet_NaN();
  }

  const T kEpsilon = 1e-7;
  // -> 0
  if (x < kEpsilon) {
    return -std::numeric_limits<T>::infinity();
  }

  // Fast extraction of exponent and mantissa
  union {
    float f;
    uint32_t i;
  } v;
  v.f = (float)x;
  int e = (int)((v.i >> 23) & 0xFF) - 127;
  
  // Normalize the mantissa to [1, 2))
  v.i = (v.i & 0x7FFFFF) | 0x3F800000;
  float m = v.f;

  // map m to [2/3, 4/3]
  if (m > 1.3333333f) {
      m *= 0.5f;
      e++;
  }
  
  // Pade approximation: ln(1 + t) = 2 * arctanh(t / (2 + t))
  T t = (T)m - (T)1.0;
  T s = t / ((T)2.0 + t);
  T s2 = s * s;

  // High-precision polynomial approximation  for 0.01% precision
  // ln(1+t) = ln((1+s)/(1-s)) = 2 * (s + s^3/3 + s^5/5 + ...)
  T poly = s * ((T)2.0 + s2 * ((T)0.666666667 + s2 * ((T)0.400002 + s2 * (T)0.285262)));

  // Combine the result
  const T LN2 = (T)0.69314718056;
  return (T)e * LN2 + poly;
}
 
template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) T scalar_operation(T src_oprand) {
  if constexpr (OP == VectorOpTy::VABS) {
    return src_oprand > 0 ? src_oprand : -src_oprand;
  } else if constexpr (OP == VectorOpTy::VNOT) {
    int64_t src_oprand_i64 = static_cast<int64_t>(src_oprand);
    int64_t res = ~src_oprand_i64;
    return static_cast<T>(res);
  } else if constexpr (OP == VectorOpTy::VRELU) {
    return src_oprand > 0 ? src_oprand : 0;
  } else if constexpr (OP == VectorOpTy::VREC) {
    return 1.0f / src_oprand;
  } else if constexpr (OP == VectorOpTy::VEXP) {
    return scalar_exp(src_oprand);
  } else if constexpr (OP == VectorOpTy::VSQRT) {
    return scalar_sqrt(src_oprand);
  } else if constexpr (OP == VectorOpTy::VRSQRT) {
    return scalar_rsqrt(src_oprand);
  } else if constexpr (OP == VectorOpTy::VLN) {
    return scalar_ln(src_oprand);
  } else {
    static_assert("unsupported elementwise op");
  }
  return src_oprand;
}

template <VectorOpTy OP>
__aiv__ __attribute__((always_inline))
half handle_half_operation(half src0_oprand, half src1_oprand, VectorLastAxisMode mode) {
  if constexpr (OP == VectorOpTy::VOR || OP == VectorOpTy::VAND) {
    int16_t *src0_oprand_int16 = (int16_t*)&src0_oprand;
    int16_t *src1_oprand_int16 = (int16_t*)&src1_oprand;
    int16_t res_int16 = scalar_operation<OP, int16_t>(*src0_oprand_int16, *src1_oprand_int16);
    half *res_float16 = (half*)&res_int16;
    return *res_float16;
  } else if constexpr (OP == VectorOpTy::VNOT) {
    int16_t *src0_oprand_int16 = (int16_t*)&src0_oprand;
    int16_t res_int16 = scalar_operation<OP, int16_t>(*src0_oprand_int16);
    half *res_float16 = (half*)&res_int16;
    return *res_float16;
  } else {
    float src0_oprand_float = static_cast<float>(src0_oprand);
    float src1_oprand_float = static_cast<float>(src1_oprand);
    float res_float;
    if (mode == VectorLastAxisMode::V) {
      res_float = scalar_operation<OP, float>(src0_oprand_float);
    } else {
      res_float = scalar_operation<OP, float>(src0_oprand_float, src1_oprand_float);
    }
    return static_cast<half>(res_float);
  }
}

template <VectorOpTy OP>
__aiv__ __attribute__((always_inline))
bfloat16_t handle_bfloat16_operation(bfloat16_t src0_oprand, bfloat16_t src1_oprand, VectorLastAxisMode mode) {
  if constexpr (OP == VectorOpTy::VOR || OP == VectorOpTy::VAND) {
    int16_t *src0_oprand_int16 = (int16_t*)&src0_oprand;
    int16_t *src1_oprand_int16 = (int16_t*)&src1_oprand;
    int16_t res_int16 = scalar_operation<OP, int16_t>(*src0_oprand_int16, *src1_oprand_int16);
    bfloat16_t *res_bfloat16 = (bfloat16_t*)&res_int16;
    return *res_bfloat16;
  } else if constexpr (OP == VectorOpTy::VNOT) {
    int16_t *src0_oprand_int16 = (int16_t*)&src0_oprand;
    int16_t res_int16 = scalar_operation<OP, int16_t>(*src0_oprand_int16);
    bfloat16_t *res_bfloat16 = (bfloat16_t*)&res_int16;
    return *res_bfloat16;
  } else {
    float src0_oprand_float = static_cast<float>(src0_oprand);
    float src1_oprand_float = static_cast<float>(src1_oprand);
    float res_float;
    if (mode == VectorLastAxisMode::V) {
      res_float = scalar_operation<OP, float>(src0_oprand_float);
    } else {
      res_float = scalar_operation<OP, float>(src0_oprand_float, src1_oprand_float);
    }
    return static_cast<bfloat16_t>(res_float);
  }
}

template <VectorOpTy OP>
__aiv__ __attribute__((always_inline))
float handle_float_operation(float src0_oprand, float src1_oprand, VectorLastAxisMode mode) {
  if constexpr (OP == VectorOpTy::VOR || OP == VectorOpTy::VAND) {
    int32_t *src0_oprand_int32 = (int32_t*)&src0_oprand;
    int32_t *src1_oprand_int32 = (int32_t*)&src1_oprand;
    int32_t res_int32 = scalar_operation<OP, int32_t>(*src0_oprand_int32, *src1_oprand_int32);
    float *res_float32 = (float*)&res_int32;
    return *res_float32;
  } else if constexpr (OP == VectorOpTy::VNOT) {
    int32_t *src0_oprand_int32 = (int32_t*)&src0_oprand;
    int32_t res_int32 = scalar_operation<OP, int32_t>(*src0_oprand_int32);
    float *res_float32 = (float*)&res_int32;
    return *res_float32;
  } else {
    if (mode == VectorLastAxisMode::V) {
      return scalar_operation<OP, float>(src0_oprand);
    } else {
      return scalar_operation<OP, float>(src0_oprand, src1_oprand);
    }
  }
}

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline))
T handle_vector_operation(T src0_oprand, T src1_oprand, VectorLastAxisMode mode) {
  if constexpr (std::is_same_v<T, half>) {
    return handle_half_operation<OP>(src0_oprand, src1_oprand, mode);
  } else if constexpr (std::is_same_v<T, bfloat16_t>) {
    return handle_bfloat16_operation<OP>(src0_oprand, src1_oprand, mode);
  } else if constexpr (std::is_same_v<T, float>) {
    return handle_float_operation<OP>(src0_oprand, src1_oprand, mode);
  } else {
    if (mode == VectorLastAxisMode::V) {
        return scalar_operation<OP, T>(src0_oprand);
    } else {
        return scalar_operation<OP, T>(src0_oprand, src1_oprand);
    }
  }
}

extern "C" {
//===-------------------------------------------------------------------===//
// eltwise vv, 1 dim
//===-------------------------------------------------------------------===//
DECLARE_ELTWISE_VV(vadd, VectorOpTy::VADD, 1, int16_t);
DECLARE_ELTWISE_VV(vadd, VectorOpTy::VADD, 1, int32_t);
DECLARE_ELTWISE_VV(vadd, VectorOpTy::VADD, 1, half);
DECLARE_ELTWISE_VV(vadd, VectorOpTy::VADD, 1, float);

DECLARE_ELTWISE_VV(vsub, VectorOpTy::VSUB, 1, int16_t);
DECLARE_ELTWISE_VV(vsub, VectorOpTy::VSUB, 1, int32_t);
DECLARE_ELTWISE_VV(vsub, VectorOpTy::VSUB, 1, half);
DECLARE_ELTWISE_VV(vsub, VectorOpTy::VSUB, 1, float);

DECLARE_ELTWISE_VV(vmul, VectorOpTy::VMUL, 1, int16_t);
DECLARE_ELTWISE_VV(vmul, VectorOpTy::VMUL, 1, int32_t);
DECLARE_ELTWISE_VV(vmul, VectorOpTy::VMUL, 1, half);
DECLARE_ELTWISE_VV(vmul, VectorOpTy::VMUL, 1, float);

DECLARE_ELTWISE_VV(vdiv, VectorOpTy::VDIV, 1, half);
DECLARE_ELTWISE_VV(vdiv, VectorOpTy::VDIV, 1, float);

DECLARE_ELTWISE_VV(vmin, VectorOpTy::VMIN, 1, int16_t);
DECLARE_ELTWISE_VV(vmin, VectorOpTy::VMIN, 1, int32_t);
DECLARE_ELTWISE_VV(vmin, VectorOpTy::VMIN, 1, half);
DECLARE_ELTWISE_VV(vmin, VectorOpTy::VMIN, 1, float);

DECLARE_ELTWISE_VV(vmax, VectorOpTy::VMAX, 1, int16_t);
DECLARE_ELTWISE_VV(vmax, VectorOpTy::VMAX, 1, int32_t);
DECLARE_ELTWISE_VV(vmax, VectorOpTy::VMAX, 1, half);
DECLARE_ELTWISE_VV(vmax, VectorOpTy::VMAX, 1, float);

DECLARE_ELTWISE_VV(vand, VectorOpTy::VAND, 1, bool);
DECLARE_ELTWISE_VV(vand, VectorOpTy::VAND, 1, int8_t);
DECLARE_ELTWISE_VV(vand, VectorOpTy::VAND, 1, uint8_t);
DECLARE_ELTWISE_VV(vand, VectorOpTy::VAND, 1, int16_t);
DECLARE_ELTWISE_VV(vand, VectorOpTy::VAND, 1, uint16_t);
DECLARE_ELTWISE_VV(vand, VectorOpTy::VAND, 1, half);
DECLARE_ELTWISE_VV(vand, VectorOpTy::VAND, 1, bfloat16_t);
DECLARE_ELTWISE_VV(vand, VectorOpTy::VAND, 1, int32_t);
DECLARE_ELTWISE_VV(vand, VectorOpTy::VAND, 1, uint32_t);
DECLARE_ELTWISE_VV(vand, VectorOpTy::VAND, 1, float);
DECLARE_ELTWISE_VV(vand, VectorOpTy::VAND, 1, int64_t);
DECLARE_ELTWISE_VV(vand, VectorOpTy::VAND, 1, uint64_t);

DECLARE_ELTWISE_VV(vor, VectorOpTy::VOR, 1, bool);
DECLARE_ELTWISE_VV(vor, VectorOpTy::VOR, 1, int8_t);
DECLARE_ELTWISE_VV(vor, VectorOpTy::VOR, 1, uint8_t);
DECLARE_ELTWISE_VV(vor, VectorOpTy::VOR, 1, int16_t);
DECLARE_ELTWISE_VV(vor, VectorOpTy::VOR, 1, uint16_t);
DECLARE_ELTWISE_VV(vor, VectorOpTy::VOR, 1, half);
DECLARE_ELTWISE_VV(vor, VectorOpTy::VOR, 1, bfloat16_t);
DECLARE_ELTWISE_VV(vor, VectorOpTy::VOR, 1, int32_t);
DECLARE_ELTWISE_VV(vor, VectorOpTy::VOR, 1, uint32_t);
DECLARE_ELTWISE_VV(vor, VectorOpTy::VOR, 1, float);
DECLARE_ELTWISE_VV(vor, VectorOpTy::VOR, 1, int64_t);
DECLARE_ELTWISE_VV(vor, VectorOpTy::VOR, 1, uint64_t);

DECLARE_ELTWISE_VV_WITHOUT_DEFAULT_VALUE(vxor, VectorOpTy::VXOR, 1, bool);
DECLARE_ELTWISE_VV_WITHOUT_DEFAULT_VALUE(vxor, VectorOpTy::VXOR, 1, int8_t);
DECLARE_ELTWISE_VV_WITHOUT_DEFAULT_VALUE(vxor, VectorOpTy::VXOR, 1, uint8_t);
DECLARE_ELTWISE_VV_WITHOUT_DEFAULT_VALUE(vxor, VectorOpTy::VXOR, 1, int16_t);
DECLARE_ELTWISE_VV_WITHOUT_DEFAULT_VALUE(vxor, VectorOpTy::VXOR, 1, uint16_t);
DECLARE_ELTWISE_VV_WITHOUT_DEFAULT_VALUE(vxor, VectorOpTy::VXOR, 1, int32_t);
DECLARE_ELTWISE_VV_WITHOUT_DEFAULT_VALUE(vxor, VectorOpTy::VXOR, 1, uint32_t);
DECLARE_ELTWISE_VV_WITHOUT_DEFAULT_VALUE(vxor, VectorOpTy::VXOR, 1, int64_t);
DECLARE_ELTWISE_VV_WITHOUT_DEFAULT_VALUE(vxor, VectorOpTy::VXOR, 1, uint64_t);

//===-------------------------------------------------------------------===//
// eltwise sv, 1 dim
//===-------------------------------------------------------------------===//
DECLARE_ELTWISE_SV(vsub, VectorOpTy::VSUB, 1, int16_t);
DECLARE_ELTWISE_SV(vsub, VectorOpTy::VSUB, 1, int32_t);
DECLARE_ELTWISE_SV(vsub, VectorOpTy::VSUB, 1, half);
DECLARE_ELTWISE_SV(vsub, VectorOpTy::VSUB, 1, float);

DECLARE_ELTWISE_SV(vdiv, VectorOpTy::VDIV, 1, half);
DECLARE_ELTWISE_SV(vdiv, VectorOpTy::VDIV, 1, float);

//===-------------------------------------------------------------------===//
// eltwise vs, 1 dim
//===-------------------------------------------------------------------===//
DECLARE_ELTWISE_VS(vadds, VectorOpTy::VADDS, 1, int16_t);
DECLARE_ELTWISE_VS(vadds, VectorOpTy::VADDS, 1, int32_t);
DECLARE_ELTWISE_VS(vadds, VectorOpTy::VADDS, 1, half);
DECLARE_ELTWISE_VS(vadds, VectorOpTy::VADDS, 1, float);

DECLARE_ELTWISE_VS(vsub, VectorOpTy::VSUB, 1, int16_t);
DECLARE_ELTWISE_VS(vsub, VectorOpTy::VSUB, 1, int32_t);
DECLARE_ELTWISE_VS(vsub, VectorOpTy::VSUB, 1, half);
DECLARE_ELTWISE_VS(vsub, VectorOpTy::VSUB, 1, float);

DECLARE_ELTWISE_VS(vdiv, VectorOpTy::VDIV, 1, half);
DECLARE_ELTWISE_VS(vdiv, VectorOpTy::VDIV, 1, float);

DECLARE_ELTWISE_VS(vmuls, VectorOpTy::VMULS, 1, int16_t);
DECLARE_ELTWISE_VS(vmuls, VectorOpTy::VMULS, 1, int32_t);
DECLARE_ELTWISE_VS(vmuls, VectorOpTy::VMULS, 1, half);
DECLARE_ELTWISE_VS(vmuls, VectorOpTy::VMULS, 1, float);

DECLARE_ELTWISE_VS(vmins, VectorOpTy::VMINS, 1, int16_t);
DECLARE_ELTWISE_VS(vmins, VectorOpTy::VMINS, 1, int32_t);
DECLARE_ELTWISE_VS(vmins, VectorOpTy::VMINS, 1, half);
DECLARE_ELTWISE_VS(vmins, VectorOpTy::VMINS, 1, float);

DECLARE_ELTWISE_VS(vmaxs, VectorOpTy::VMAXS, 1, int16_t);
DECLARE_ELTWISE_VS(vmaxs, VectorOpTy::VMAXS, 1, int32_t);
DECLARE_ELTWISE_VS(vmaxs, VectorOpTy::VMAXS, 1, half);
DECLARE_ELTWISE_VS(vmaxs, VectorOpTy::VMAXS, 1, float);

DECLARE_ELTWISE_VS(vshls, VectorOpTy::VSHL, 1, int16_t);
DECLARE_ELTWISE_VS(vshls, VectorOpTy::VSHL, 1, uint16_t);
DECLARE_ELTWISE_VS(vshls, VectorOpTy::VSHL, 1, int32_t);
DECLARE_ELTWISE_VS(vshls, VectorOpTy::VSHL, 1, uint32_t);

DECLARE_ELTWISE_VS(vshrs, VectorOpTy::VSHR, 1, int16_t);
DECLARE_ELTWISE_VS(vshrs, VectorOpTy::VSHR, 1, uint16_t);
DECLARE_ELTWISE_VS(vshrs, VectorOpTy::VSHR, 1, int32_t);
DECLARE_ELTWISE_VS(vshrs, VectorOpTy::VSHR, 1, uint32_t);

//===-------------------------------------------------------------------===//
// eltwise unary, 1 dim
//===-------------------------------------------------------------------===//
DECLARE_ELTWISE_V(vabs, VectorOpTy::VABS, 1, half);
DECLARE_ELTWISE_V(vabs, VectorOpTy::VABS, 1, float);

DECLARE_ELTWISE_V(vln, VectorOpTy::VLN, 1, half);
DECLARE_ELTWISE_V(vln, VectorOpTy::VLN, 1, float);

DECLARE_ELTWISE_V(vrelu, VectorOpTy::VRELU, 1, half);
DECLARE_ELTWISE_V(vrelu, VectorOpTy::VRELU, 1, float);
DECLARE_ELTWISE_V(vrelu, VectorOpTy::VRELU, 1, int32_t);

DECLARE_ELTWISE_V(vexp, VectorOpTy::VEXP, 1, half);
DECLARE_ELTWISE_V(vexp, VectorOpTy::VEXP, 1, float);

DECLARE_ELTWISE_V(vrsqrt, VectorOpTy::VRSQRT, 1, half);
DECLARE_ELTWISE_V(vrsqrt, VectorOpTy::VRSQRT, 1, float);

DECLARE_ELTWISE_V(vsqrt, VectorOpTy::VSQRT, 1, half);
DECLARE_ELTWISE_V(vsqrt, VectorOpTy::VSQRT, 1, float);

DECLARE_ELTWISE_V(vrec, VectorOpTy::VREC, 1, half);
DECLARE_ELTWISE_V(vrec, VectorOpTy::VREC, 1, float);

DECLARE_ELTWISE_V(vnot, VectorOpTy::VNOT, 1, bool);
DECLARE_ELTWISE_V(vnot, VectorOpTy::VNOT, 1, int8_t);
DECLARE_ELTWISE_V(vnot, VectorOpTy::VNOT, 1, uint8_t);
DECLARE_ELTWISE_V(vnot, VectorOpTy::VNOT, 1, int16_t);
DECLARE_ELTWISE_V(vnot, VectorOpTy::VNOT, 1, uint16_t);
DECLARE_ELTWISE_V(vnot, VectorOpTy::VNOT, 1, half);
DECLARE_ELTWISE_V(vnot, VectorOpTy::VNOT, 1, bfloat16_t);
DECLARE_ELTWISE_V(vnot, VectorOpTy::VNOT, 1, int32_t);
DECLARE_ELTWISE_V(vnot, VectorOpTy::VNOT, 1, uint32_t);
DECLARE_ELTWISE_V(vnot, VectorOpTy::VNOT, 1, float);
DECLARE_ELTWISE_V(vnot, VectorOpTy::VNOT, 1, int64_t);
DECLARE_ELTWISE_V(vnot, VectorOpTy::VNOT, 1, uint64_t);

//===-------------------------------------------------------------------===//
// eltwise vv, 2 dim
//===-------------------------------------------------------------------===//
DECLARE_ELTWISE_VV(vadd, VectorOpTy::VADD, 2, int16_t);
DECLARE_ELTWISE_VV(vadd, VectorOpTy::VADD, 2, int32_t);
DECLARE_ELTWISE_VV(vadd, VectorOpTy::VADD, 2, half);
DECLARE_ELTWISE_VV(vadd, VectorOpTy::VADD, 2, float);

DECLARE_ELTWISE_VV(vsub, VectorOpTy::VSUB, 2, int16_t);
DECLARE_ELTWISE_VV(vsub, VectorOpTy::VSUB, 2, int32_t);
DECLARE_ELTWISE_VV(vsub, VectorOpTy::VSUB, 2, half);
DECLARE_ELTWISE_VV(vsub, VectorOpTy::VSUB, 2, float);

DECLARE_ELTWISE_VV(vmul, VectorOpTy::VMUL, 2, int16_t);
DECLARE_ELTWISE_VV(vmul, VectorOpTy::VMUL, 2, int32_t);
DECLARE_ELTWISE_VV(vmul, VectorOpTy::VMUL, 2, half);
DECLARE_ELTWISE_VV(vmul, VectorOpTy::VMUL, 2, float);

DECLARE_ELTWISE_VV(vdiv, VectorOpTy::VDIV, 2, half);
DECLARE_ELTWISE_VV(vdiv, VectorOpTy::VDIV, 2, float);

DECLARE_ELTWISE_VV(vmin, VectorOpTy::VMIN, 2, int16_t);
DECLARE_ELTWISE_VV(vmin, VectorOpTy::VMIN, 2, int32_t);
DECLARE_ELTWISE_VV(vmin, VectorOpTy::VMIN, 2, half);
DECLARE_ELTWISE_VV(vmin, VectorOpTy::VMIN, 2, float);

DECLARE_ELTWISE_VV(vmax, VectorOpTy::VMAX, 2, int16_t);
DECLARE_ELTWISE_VV(vmax, VectorOpTy::VMAX, 2, int32_t);
DECLARE_ELTWISE_VV(vmax, VectorOpTy::VMAX, 2, half);
DECLARE_ELTWISE_VV(vmax, VectorOpTy::VMAX, 2, float);

DECLARE_ELTWISE_VV(vand, VectorOpTy::VAND, 2, bool);
DECLARE_ELTWISE_VV(vand, VectorOpTy::VAND, 2, int8_t);
DECLARE_ELTWISE_VV(vand, VectorOpTy::VAND, 2, uint8_t);
DECLARE_ELTWISE_VV(vand, VectorOpTy::VAND, 2, int16_t);
DECLARE_ELTWISE_VV(vand, VectorOpTy::VAND, 2, uint16_t);
DECLARE_ELTWISE_VV(vand, VectorOpTy::VAND, 2, half);
DECLARE_ELTWISE_VV(vand, VectorOpTy::VAND, 2, bfloat16_t);
DECLARE_ELTWISE_VV(vand, VectorOpTy::VAND, 2, int32_t);
DECLARE_ELTWISE_VV(vand, VectorOpTy::VAND, 2, uint32_t);
DECLARE_ELTWISE_VV(vand, VectorOpTy::VAND, 2, float);
DECLARE_ELTWISE_VV(vand, VectorOpTy::VAND, 2, int64_t);
DECLARE_ELTWISE_VV(vand, VectorOpTy::VAND, 2, uint64_t);

DECLARE_ELTWISE_VV(vor, VectorOpTy::VOR, 2, bool);
DECLARE_ELTWISE_VV(vor, VectorOpTy::VOR, 2, int8_t);
DECLARE_ELTWISE_VV(vor, VectorOpTy::VOR, 2, uint8_t);
DECLARE_ELTWISE_VV(vor, VectorOpTy::VOR, 2, int16_t);
DECLARE_ELTWISE_VV(vor, VectorOpTy::VOR, 2, uint16_t);
DECLARE_ELTWISE_VV(vor, VectorOpTy::VOR, 2, half);
DECLARE_ELTWISE_VV(vor, VectorOpTy::VOR, 2, bfloat16_t);
DECLARE_ELTWISE_VV(vor, VectorOpTy::VOR, 2, int32_t);
DECLARE_ELTWISE_VV(vor, VectorOpTy::VOR, 2, uint32_t);
DECLARE_ELTWISE_VV(vor, VectorOpTy::VOR, 2, float);
DECLARE_ELTWISE_VV(vor, VectorOpTy::VOR, 2, int64_t);
DECLARE_ELTWISE_VV(vor, VectorOpTy::VOR, 2, uint64_t);

DECLARE_ELTWISE_VV_WITHOUT_DEFAULT_VALUE(vxor, VectorOpTy::VXOR, 2, bool);
DECLARE_ELTWISE_VV_WITHOUT_DEFAULT_VALUE(vxor, VectorOpTy::VXOR, 2, int8_t);
DECLARE_ELTWISE_VV_WITHOUT_DEFAULT_VALUE(vxor, VectorOpTy::VXOR, 2, uint8_t);
DECLARE_ELTWISE_VV_WITHOUT_DEFAULT_VALUE(vxor, VectorOpTy::VXOR, 2, int16_t);
DECLARE_ELTWISE_VV_WITHOUT_DEFAULT_VALUE(vxor, VectorOpTy::VXOR, 2, uint16_t);
DECLARE_ELTWISE_VV_WITHOUT_DEFAULT_VALUE(vxor, VectorOpTy::VXOR, 2, int32_t);
DECLARE_ELTWISE_VV_WITHOUT_DEFAULT_VALUE(vxor, VectorOpTy::VXOR, 2, uint32_t);
DECLARE_ELTWISE_VV_WITHOUT_DEFAULT_VALUE(vxor, VectorOpTy::VXOR, 2, int64_t);
DECLARE_ELTWISE_VV_WITHOUT_DEFAULT_VALUE(vxor, VectorOpTy::VXOR, 2, uint64_t);

//===-------------------------------------------------------------------===//
// eltwise sv, 2 dim
//===-------------------------------------------------------------------===//
DECLARE_ELTWISE_SV(vsub, VectorOpTy::VSUB, 2, int16_t);
DECLARE_ELTWISE_SV(vsub, VectorOpTy::VSUB, 2, int32_t);
DECLARE_ELTWISE_SV(vsub, VectorOpTy::VSUB, 2, half);
DECLARE_ELTWISE_SV(vsub, VectorOpTy::VSUB, 2, float);

DECLARE_ELTWISE_SV(vdiv, VectorOpTy::VDIV, 2, half);
DECLARE_ELTWISE_SV(vdiv, VectorOpTy::VDIV, 2, float);

//===-------------------------------------------------------------------===//
// eltwise vs, 2 dim
//===-------------------------------------------------------------------===//
DECLARE_ELTWISE_VS(vadds, VectorOpTy::VADDS, 2, int16_t);
DECLARE_ELTWISE_VS(vadds, VectorOpTy::VADDS, 2, int32_t);
DECLARE_ELTWISE_VS(vadds, VectorOpTy::VADDS, 2, half);
DECLARE_ELTWISE_VS(vadds, VectorOpTy::VADDS, 2, float);

DECLARE_ELTWISE_VS(vsub, VectorOpTy::VSUB, 2, int16_t);
DECLARE_ELTWISE_VS(vsub, VectorOpTy::VSUB, 2, int32_t);
DECLARE_ELTWISE_VS(vsub, VectorOpTy::VSUB, 2, half);
DECLARE_ELTWISE_VS(vsub, VectorOpTy::VSUB, 2, float);

DECLARE_ELTWISE_VS(vdiv, VectorOpTy::VDIV, 2, half);
DECLARE_ELTWISE_VS(vdiv, VectorOpTy::VDIV, 2, float);

DECLARE_ELTWISE_VS(vmuls, VectorOpTy::VMULS, 2, int16_t);
DECLARE_ELTWISE_VS(vmuls, VectorOpTy::VMULS, 2, int32_t);
DECLARE_ELTWISE_VS(vmuls, VectorOpTy::VMULS, 2, half);
DECLARE_ELTWISE_VS(vmuls, VectorOpTy::VMULS, 2, float);

DECLARE_ELTWISE_VS(vmins, VectorOpTy::VMINS, 2, int16_t);
DECLARE_ELTWISE_VS(vmins, VectorOpTy::VMINS, 2, int32_t);
DECLARE_ELTWISE_VS(vmins, VectorOpTy::VMINS, 2, half);
DECLARE_ELTWISE_VS(vmins, VectorOpTy::VMINS, 2, float);

DECLARE_ELTWISE_VS(vmaxs, VectorOpTy::VMAXS, 2, int16_t);
DECLARE_ELTWISE_VS(vmaxs, VectorOpTy::VMAXS, 2, int32_t);
DECLARE_ELTWISE_VS(vmaxs, VectorOpTy::VMAXS, 2, half);
DECLARE_ELTWISE_VS(vmaxs, VectorOpTy::VMAXS, 2, float);

DECLARE_ELTWISE_VS(vshls, VectorOpTy::VSHL, 2, int16_t);
DECLARE_ELTWISE_VS(vshls, VectorOpTy::VSHL, 2, uint16_t);
DECLARE_ELTWISE_VS(vshls, VectorOpTy::VSHL, 2, int32_t);
DECLARE_ELTWISE_VS(vshls, VectorOpTy::VSHL, 2, uint32_t);

DECLARE_ELTWISE_VS(vshrs, VectorOpTy::VSHR, 2, int16_t);
DECLARE_ELTWISE_VS(vshrs, VectorOpTy::VSHR, 2, uint16_t);
DECLARE_ELTWISE_VS(vshrs, VectorOpTy::VSHR, 2, int32_t);
DECLARE_ELTWISE_VS(vshrs, VectorOpTy::VSHR, 2, uint32_t);

//===-------------------------------------------------------------------===//
// eltwise unary, 2 dim
//===-------------------------------------------------------------------===//
DECLARE_ELTWISE_V(vabs, VectorOpTy::VABS, 2, half);
DECLARE_ELTWISE_V(vabs, VectorOpTy::VABS, 2, float);

DECLARE_ELTWISE_V(vln, VectorOpTy::VLN, 2, half);
DECLARE_ELTWISE_V(vln, VectorOpTy::VLN, 2, float);

DECLARE_ELTWISE_V(vrelu, VectorOpTy::VRELU, 2, half);
DECLARE_ELTWISE_V(vrelu, VectorOpTy::VRELU, 2, float);
DECLARE_ELTWISE_V(vrelu, VectorOpTy::VRELU, 2, int32_t);

DECLARE_ELTWISE_V(vexp, VectorOpTy::VEXP, 2, half);
DECLARE_ELTWISE_V(vexp, VectorOpTy::VEXP, 2, float);

DECLARE_ELTWISE_V(vrsqrt, VectorOpTy::VRSQRT, 2, half);
DECLARE_ELTWISE_V(vrsqrt, VectorOpTy::VRSQRT, 2, float);

DECLARE_ELTWISE_V(vsqrt, VectorOpTy::VSQRT, 2, half);
DECLARE_ELTWISE_V(vsqrt, VectorOpTy::VSQRT, 2, float);

DECLARE_ELTWISE_V(vrec, VectorOpTy::VREC, 2, half);
DECLARE_ELTWISE_V(vrec, VectorOpTy::VREC, 2, float);

DECLARE_ELTWISE_V(vnot, VectorOpTy::VNOT, 2, bool);
DECLARE_ELTWISE_V(vnot, VectorOpTy::VNOT, 2, int8_t);
DECLARE_ELTWISE_V(vnot, VectorOpTy::VNOT, 2, uint8_t);
DECLARE_ELTWISE_V(vnot, VectorOpTy::VNOT, 2, int16_t);
DECLARE_ELTWISE_V(vnot, VectorOpTy::VNOT, 2, uint16_t);
DECLARE_ELTWISE_V(vnot, VectorOpTy::VNOT, 2, int32_t);
DECLARE_ELTWISE_V(vnot, VectorOpTy::VNOT, 2, uint32_t);
DECLARE_ELTWISE_V(vnot, VectorOpTy::VNOT, 2, int64_t);
DECLARE_ELTWISE_V(vnot, VectorOpTy::VNOT, 2, uint64_t);

//===-------------------------------------------------------------------===//
// eltwise vv, 3 dim
//===-------------------------------------------------------------------===//
DECLARE_ELTWISE_VV(vadd, VectorOpTy::VADD, 3, int16_t);
DECLARE_ELTWISE_VV(vadd, VectorOpTy::VADD, 3, int32_t);
DECLARE_ELTWISE_VV(vadd, VectorOpTy::VADD, 3, half);
DECLARE_ELTWISE_VV(vadd, VectorOpTy::VADD, 3, float);

DECLARE_ELTWISE_VV(vsub, VectorOpTy::VSUB, 3, int16_t);
DECLARE_ELTWISE_VV(vsub, VectorOpTy::VSUB, 3, int32_t);
DECLARE_ELTWISE_VV(vsub, VectorOpTy::VSUB, 3, half);
DECLARE_ELTWISE_VV(vsub, VectorOpTy::VSUB, 3, float);

DECLARE_ELTWISE_VV(vmul, VectorOpTy::VMUL, 3, int16_t);
DECLARE_ELTWISE_VV(vmul, VectorOpTy::VMUL, 3, int32_t);
DECLARE_ELTWISE_VV(vmul, VectorOpTy::VMUL, 3, half);
DECLARE_ELTWISE_VV(vmul, VectorOpTy::VMUL, 3, float);

DECLARE_ELTWISE_VV(vdiv, VectorOpTy::VDIV, 3, half);
DECLARE_ELTWISE_VV(vdiv, VectorOpTy::VDIV, 3, float);

DECLARE_ELTWISE_VV(vmin, VectorOpTy::VMIN, 3, int16_t);
DECLARE_ELTWISE_VV(vmin, VectorOpTy::VMIN, 3, int32_t);
DECLARE_ELTWISE_VV(vmin, VectorOpTy::VMIN, 3, half);
DECLARE_ELTWISE_VV(vmin, VectorOpTy::VMIN, 3, float);

DECLARE_ELTWISE_VV(vmax, VectorOpTy::VMAX, 3, int16_t);
DECLARE_ELTWISE_VV(vmax, VectorOpTy::VMAX, 3, int32_t);
DECLARE_ELTWISE_VV(vmax, VectorOpTy::VMAX, 3, half);
DECLARE_ELTWISE_VV(vmax, VectorOpTy::VMAX, 3, float);

DECLARE_ELTWISE_VV(vand, VectorOpTy::VAND, 3, bool);
DECLARE_ELTWISE_VV(vand, VectorOpTy::VAND, 3, int8_t);
DECLARE_ELTWISE_VV(vand, VectorOpTy::VAND, 3, uint8_t);
DECLARE_ELTWISE_VV(vand, VectorOpTy::VAND, 3, int16_t);
DECLARE_ELTWISE_VV(vand, VectorOpTy::VAND, 3, uint16_t);
DECLARE_ELTWISE_VV(vand, VectorOpTy::VAND, 3, half);
DECLARE_ELTWISE_VV(vand, VectorOpTy::VAND, 3, bfloat16_t);
DECLARE_ELTWISE_VV(vand, VectorOpTy::VAND, 3, float);
DECLARE_ELTWISE_VV(vand, VectorOpTy::VAND, 3, int32_t);
DECLARE_ELTWISE_VV(vand, VectorOpTy::VAND, 3, uint32_t);
DECLARE_ELTWISE_VV(vand, VectorOpTy::VAND, 3, int64_t);
DECLARE_ELTWISE_VV(vand, VectorOpTy::VAND, 3, uint64_t);

DECLARE_ELTWISE_VV(vor, VectorOpTy::VOR, 3, bool);
DECLARE_ELTWISE_VV(vor, VectorOpTy::VOR, 3, int8_t);
DECLARE_ELTWISE_VV(vor, VectorOpTy::VOR, 3, uint8_t);
DECLARE_ELTWISE_VV(vor, VectorOpTy::VOR, 3, int16_t);
DECLARE_ELTWISE_VV(vor, VectorOpTy::VOR, 3, uint16_t);
DECLARE_ELTWISE_VV(vor, VectorOpTy::VOR, 3, half);
DECLARE_ELTWISE_VV(vor, VectorOpTy::VOR, 3, bfloat16_t);
DECLARE_ELTWISE_VV(vor, VectorOpTy::VOR, 3, float);
DECLARE_ELTWISE_VV(vor, VectorOpTy::VOR, 3, int32_t);
DECLARE_ELTWISE_VV(vor, VectorOpTy::VOR, 3, uint32_t);
DECLARE_ELTWISE_VV(vor, VectorOpTy::VOR, 3, int64_t);
DECLARE_ELTWISE_VV(vor, VectorOpTy::VOR, 3, uint64_t);

//===-------------------------------------------------------------------===//
// eltwise sv, 3 dim
//===-------------------------------------------------------------------===//
DECLARE_ELTWISE_SV(vsub, VectorOpTy::VSUB, 3, int16_t);
DECLARE_ELTWISE_SV(vsub, VectorOpTy::VSUB, 3, int32_t);
DECLARE_ELTWISE_SV(vsub, VectorOpTy::VSUB, 3, half);
DECLARE_ELTWISE_SV(vsub, VectorOpTy::VSUB, 3, float);

DECLARE_ELTWISE_SV(vdiv, VectorOpTy::VDIV, 3, half);
DECLARE_ELTWISE_SV(vdiv, VectorOpTy::VDIV, 3, float);

//===-------------------------------------------------------------------===//
// eltwise vs, 3 dim
//===-------------------------------------------------------------------===//
DECLARE_ELTWISE_VS(vadds, VectorOpTy::VADDS, 3, int16_t);
DECLARE_ELTWISE_VS(vadds, VectorOpTy::VADDS, 3, int32_t);
DECLARE_ELTWISE_VS(vadds, VectorOpTy::VADDS, 3, half);
DECLARE_ELTWISE_VS(vadds, VectorOpTy::VADDS, 3, float);

DECLARE_ELTWISE_VS(vsub, VectorOpTy::VSUB, 3, int16_t);
DECLARE_ELTWISE_VS(vsub, VectorOpTy::VSUB, 3, int32_t);
DECLARE_ELTWISE_VS(vsub, VectorOpTy::VSUB, 3, half);
DECLARE_ELTWISE_VS(vsub, VectorOpTy::VSUB, 3, float);

DECLARE_ELTWISE_VS(vdiv, VectorOpTy::VDIV, 3, half);
DECLARE_ELTWISE_VS(vdiv, VectorOpTy::VDIV, 3, float);

DECLARE_ELTWISE_VS(vmuls, VectorOpTy::VMULS, 3, int16_t);
DECLARE_ELTWISE_VS(vmuls, VectorOpTy::VMULS, 3, int32_t);
DECLARE_ELTWISE_VS(vmuls, VectorOpTy::VMULS, 3, half);
DECLARE_ELTWISE_VS(vmuls, VectorOpTy::VMULS, 3, float);

DECLARE_ELTWISE_VS(vmins, VectorOpTy::VMINS, 3, int16_t);
DECLARE_ELTWISE_VS(vmins, VectorOpTy::VMINS, 3, int32_t);
DECLARE_ELTWISE_VS(vmins, VectorOpTy::VMINS, 3, half);
DECLARE_ELTWISE_VS(vmins, VectorOpTy::VMINS, 3, float);

DECLARE_ELTWISE_VS(vmaxs, VectorOpTy::VMAXS, 3, int16_t);
DECLARE_ELTWISE_VS(vmaxs, VectorOpTy::VMAXS, 3, int32_t);
DECLARE_ELTWISE_VS(vmaxs, VectorOpTy::VMAXS, 3, half);
DECLARE_ELTWISE_VS(vmaxs, VectorOpTy::VMAXS, 3, float);

DECLARE_ELTWISE_VS(vshls, VectorOpTy::VSHL, 3, int16_t);
DECLARE_ELTWISE_VS(vshls, VectorOpTy::VSHL, 3, uint16_t);
DECLARE_ELTWISE_VS(vshls, VectorOpTy::VSHL, 3, int32_t);
DECLARE_ELTWISE_VS(vshls, VectorOpTy::VSHL, 3, uint32_t);

DECLARE_ELTWISE_VS(vshrs, VectorOpTy::VSHR, 3, int16_t);
DECLARE_ELTWISE_VS(vshrs, VectorOpTy::VSHR, 3, uint16_t);
DECLARE_ELTWISE_VS(vshrs, VectorOpTy::VSHR, 3, int32_t);
DECLARE_ELTWISE_VS(vshrs, VectorOpTy::VSHR, 3, uint32_t);

//===-------------------------------------------------------------------===//
// eltwise unary, 3 dim
//===-------------------------------------------------------------------===//
DECLARE_ELTWISE_V(vabs, VectorOpTy::VABS, 3, half);
DECLARE_ELTWISE_V(vabs, VectorOpTy::VABS, 3, float);

DECLARE_ELTWISE_V(vln, VectorOpTy::VLN, 3, half);
DECLARE_ELTWISE_V(vln, VectorOpTy::VLN, 3, float);

DECLARE_ELTWISE_V(vrelu, VectorOpTy::VRELU, 3, half);
DECLARE_ELTWISE_V(vrelu, VectorOpTy::VRELU, 3, float);
DECLARE_ELTWISE_V(vrelu, VectorOpTy::VRELU, 3, int32_t);

DECLARE_ELTWISE_V(vexp, VectorOpTy::VEXP, 3, half);
DECLARE_ELTWISE_V(vexp, VectorOpTy::VEXP, 3, float);

DECLARE_ELTWISE_V(vrsqrt, VectorOpTy::VRSQRT, 3, half);
DECLARE_ELTWISE_V(vrsqrt, VectorOpTy::VRSQRT, 3, float);

DECLARE_ELTWISE_V(vsqrt, VectorOpTy::VSQRT, 3, half);
DECLARE_ELTWISE_V(vsqrt, VectorOpTy::VSQRT, 3, float);

DECLARE_ELTWISE_V(vrec, VectorOpTy::VREC, 3, half);
DECLARE_ELTWISE_V(vrec, VectorOpTy::VREC, 3, float);

DECLARE_ELTWISE_V(vnot, VectorOpTy::VNOT, 3, bool);
DECLARE_ELTWISE_V(vnot, VectorOpTy::VNOT, 3, int8_t);
DECLARE_ELTWISE_V(vnot, VectorOpTy::VNOT, 3, uint8_t);
DECLARE_ELTWISE_V(vnot, VectorOpTy::VNOT, 3, int16_t);
DECLARE_ELTWISE_V(vnot, VectorOpTy::VNOT, 3, uint16_t);
DECLARE_ELTWISE_V(vnot, VectorOpTy::VNOT, 3, half);
DECLARE_ELTWISE_V(vnot, VectorOpTy::VNOT, 3, bfloat16_t);
DECLARE_ELTWISE_V(vnot, VectorOpTy::VNOT, 3, float);
DECLARE_ELTWISE_V(vnot, VectorOpTy::VNOT, 3, int32_t);
DECLARE_ELTWISE_V(vnot, VectorOpTy::VNOT, 3, uint32_t);
DECLARE_ELTWISE_V(vnot, VectorOpTy::VNOT, 3, int64_t);
DECLARE_ELTWISE_V(vnot, VectorOpTy::VNOT, 3, uint64_t);
}
#endif // HIVM_MLIR_TEMPLATE_VECTOR_UTILS_H
