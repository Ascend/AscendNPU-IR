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

#ifndef BISHENGIR_LIB_TEMPLATE_INCLUDE_BROADCAST_UTILS_H
#define BISHENGIR_LIB_TEMPLATE_INCLUDE_BROADCAST_UTILS_H

#include "Utils.h"
#include "Vector/VecUtils.h"

template <typename T>
__aiv__ __attribute__((always_inline)) void
brc_scalar_1d_by_scalar(T src_val, memref_t<__ubuf__ T, 1> *dst) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  WARN_SCALAR_IMPL("broadcast scalar 1D");
#endif
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  for (size_t i = 0; i < dst->sizes[0]; ++i) {
    *(dst_ptr + i * dst->strides[0]) = src_val;
  }
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
brc_scalar_2d_by_scalar(T src_val, memref_t<__ubuf__ T, 2> *dst) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  WARN_SCALAR_IMPL("broadcast scalar 2D");
#endif
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  for (size_t i = 0; i < dst->sizes[0]; ++i) {
    for (size_t j = 0; j < dst->sizes[1]; ++j) {
      *(dst_ptr + i * dst->strides[0] + j * dst->strides[1]) = src_val;
    }
  }
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
vector_broadcast_last_axis_by_scalar(memref_t<__ubuf__ T, 2> *src,
                                     memref_t<__ubuf__ T, 2> *dst) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  WARN_SCALAR_IMPL("broadcast last axis");
#endif
  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  for (int i = 0; i < dst->sizes[0]; ++i) {
    for (int j = 0; j < dst->sizes[1]; ++j) {
      *(dst_ptr + i * dst->strides[0] + j * dst->strides[1]) =
          *(src_ptr + i * src->strides[0]);
    }
  }
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);

  return;
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
vector_broadcast_first_axis_by_scalar(memref_t<__ubuf__ T, 2> *src,
                                      memref_t<__ubuf__ T, 2> *dst) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  WARN_SCALAR_IMPL("broadcast first axis");
#endif
  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  for (int i = 0; i < dst->sizes[0]; ++i) {
    for (int j = 0; j < dst->sizes[1]; ++j) {
      *(dst_ptr + i * dst->strides[0] + j * dst->strides[1]) =
          *(src_ptr + j * src->strides[1]);
    }
  }
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);

  return;
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
vector_broadcast_middle_axis_by_scalar(memref_t<__ubuf__ T, 3> *src,
                                       memref_t<__ubuf__ T, 3> *dst) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  WARN_SCALAR_IMPL("broadcast middle axis");
#endif
  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  for (int i = 0; i < dst->sizes[0]; ++i) {
    for (int j = 0; j < dst->sizes[1]; ++j) {
      for (int k = 0; k < dst->sizes[2]; ++k) {
        *(dst_ptr + i * dst->strides[0] + j * dst->strides[1] + k * dst->strides[2]) =
            *(src_ptr + i * src->strides[0] + k * src->strides[2]);
      }
    }
  }
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);

  return;
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
vector_broadcast_first_axis_by_scalar(memref_t<__ubuf__ T, 3> *src,
                                      memref_t<__ubuf__ T, 3> *dst) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  WARN_SCALAR_IMPL("broadcast first axis");
#endif
  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  for (int i = 0; i < dst->sizes[0]; ++i) {
    for (int j = 0; j < dst->sizes[1]; ++j) {
      for (int k = 0; k < dst->sizes[2]; ++k) {
        *(dst_ptr + i * dst->strides[0] + j * dst->strides[1] + k * dst->strides[2]) =
            *(src_ptr + j * src->strides[1] + k * src->strides[2]);
      }
    }
  }
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);

  return;
}

/// Check if tensor has aligned strides
template <typename T>
__aiv__ __attribute__((always_inline)) bool
is_stride_aligned(memref_t<__ubuf__ T, 2> *tensor) {
  int64_t stride0_tensor = tensor->strides[0];
  int64_t stride0_tensor_aligned =
      CEIL_FACTOR(stride0_tensor, UB_ALIGN_BYTES / sizeof(T));
  return stride0_tensor == stride0_tensor_aligned;
}

/// Check if tensor has aligned offset
template <typename T, size_t DIM>
__aiv__ __attribute__((always_inline)) bool
is_offset_aligned(memref_t<__ubuf__ T, DIM> *tensor) {
  return !tensor->offset ||
         isAddress32ByteAligned(tensor->aligned + tensor->offset);
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
brc_last_axis_2d_vbrcb(memref_t<__ubuf__ T, 2> *src,
                       memref_t<__ubuf__ T, 2> *dst);

/// Brc scalar
/// e.g. 1xf16 to memref<10xf16, strided<1>>
/// \tparam T dtype
/// \param src_val src scalar value that to be broadcast
/// \param dst_ptr dst pointer
/// \param count num of value to be broadcast
template <typename T>
__aiv__ __attribute__((always_inline)) void
brc_scalar_core_1d(T src_val, __ubuf__ T *dst_ptr, int64_t count);

__aiv__ __attribute__((always_inline)) uint64_t get_interval_mask(
    uint64_t interval_num, uint64_t interval_index, uint64_t mask_length);

#define REGISTE_BRC_SCALAR_CORE_1D(type)                                       \
  template __aiv__ __attribute__((always_inline)) void brc_scalar_core_1d(     \
      type src_val, __ubuf__ type *dst_ptr, int64_t count)

template <typename T>
__aiv__ __attribute__((always_inline)) void
brc_first_axis_2d_align_core(memref_t<__ubuf__ T, 2> *src,
                             memref_t<__ubuf__ T, 2> *dst);

#define REGISTE_BRC_FIRST_AXIS_2D_ALIGN(type)                                  \
  template __aiv__ __attribute__((always_inline)) void                         \
  brc_first_axis_2d_align_core(memref_t<__ubuf__ type, 2> *src,                \
                               memref_t<__ubuf__ type, 2> *dst)

template <typename T>
__aiv__ __attribute__((always_inline)) void
vector_broadcast_last_axis_unalign_2d(memref_t<__ubuf__ T, 2> *src,
                                      memref_t<__ubuf__ T, 2> *dst,
                                      memref_t<__ubuf__ T, 1> *tmp);

#define REGISTE_BRC_LAST_AXIS_2D_UNALIGN(type)                                 \
  template __aiv__ __attribute__((always_inline)) void                         \
  vector_broadcast_last_axis_unalign_2d(memref_t<__ubuf__ type, 2> *src,       \
                                        memref_t<__ubuf__ type, 2> *dst,       \
                                        memref_t<__ubuf__ type, 1> *tmp)

/// Convert buf_1d (x0*x1) to buf_2d (x0, x1)
/// \param size0 size0 of buf_2d
/// \param size1 size1 of buf_2d
/// Convert buf_1d (x0*x1) to buf_2d (x0, x1)
template <typename T>
__aiv__ __attribute__((always_inline)) void
convert_1d_to_2d_args(memref_t<__ubuf__ T, 1> *buf_1d,
                      memref_t<__ubuf__ T, 2> *buf_2d, int64_t size0,
                      int64_t size1) {
  buf_2d->allocated = buf_1d->allocated;
  buf_2d->aligned = buf_1d->aligned;
  buf_2d->offset = buf_1d->offset;

  buf_2d->sizes[0] = size0;
  buf_2d->sizes[1] = size1;
  buf_2d->strides[0] = size1;
  buf_2d->strides[1] = 1;
}

/// max repeat times of vbrcb intr is 254 instead of 255 when data type is b16,
/// because src_num_per_vbrcb_repeat is 8, if repeat times more than 254, offset
/// of src should be 254 * 8 instead of 255 * 8, where 255 * 8 is not 32 bytes
/// align
template <typename T>
__aiv__ __attribute__((always_inline)) constexpr int32_t
get_vbrcb_max_repeat_cnts() {
  if constexpr (sizeof(T) == BYTES_B16) {
    return MAX_VBRCB_REPEAT_TIMES;
  }
  if constexpr (sizeof(T) == BYTES_B32) {
    return INTR_MAX_REPEAT_CNTS;
  }
  static_assert("vbrcb instruction unsupported data type");
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
vbrcb_intrin(__ubuf__ T *src_ptr, __ubuf__ T *dst_ptr, int64_t dst_block_stride,
             int64_t dst_repeat_stride, int64_t repeat_times) {
  if constexpr (sizeof(T) == BYTES_B16) {
    INTRINSIC(vbrcb, (__ubuf__ uint16_t *)dst_ptr, (__ubuf__ uint16_t *)src_ptr,
              dst_block_stride, dst_repeat_stride, repeat_times);
  } else if constexpr (sizeof(T) == BYTES_B32) {
    INTRINSIC(vbrcb, (__ubuf__ uint32_t *)dst_ptr, (__ubuf__ uint32_t *)src_ptr,
              dst_block_stride, dst_repeat_stride, repeat_times);
  } else {
    static_assert("vbrcb instruction unsupported data type");
  }
}

/// Broadcast scalar to vector, broadcast scalar to dst (b, ), here b is
/// broadcast axis, customized for b64.
///
/// Constraints:
/// 1. dim of dst must be 1, dst must be continuous vector
/// 2. src must be scalar
template <typename T>
__aiv__ __attribute__((always_inline)) void
broadcast_scalar_b64(T scalar, memref_t<__ubuf__ T, 1> *dst);

#define REGISTE_BROADCAST_SCALAR_B64_1D(type)                                  \
  template __aiv__ __attribute__((always_inline)) void                         \
  broadcast_scalar_b64<type>(type scalar, memref_t<__ubuf__ type, 1> * dst)

#define DECLARE_BROADCAST_SCALAR(dim, type)                                    \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_broadcast_scalar_##type##_to_##dim##d(                          \
      type scalar, memref_t<__ubuf__ type, dim> *dst)

#define REGISTE_BROADCAST_SCALAR(dim, type)                                    \
  DECLARE_BROADCAST_SCALAR(dim, type) {                                        \
    if constexpr (!std::is_same<type, bool>::value) {                          \
      broadcast_scalar<type>(scalar, dst);                                     \
    } else {                                                                   \
      broadcast_scalar(scalar, dst);                                           \
    }                                                                          \
  }

template <typename T, size_t Dim>
__aiv__ __attribute__((always_inline)) void
broadcast_scalar(T scalar, memref_t<__ubuf__ T, Dim> *dst);

#define REGISTE_BRC_SCALAR(dim, type)                                          \
  template __aiv__ __attribute__((always_inline)) void                         \
  broadcast_scalar<type, dim>(type scalar, memref_t<__ubuf__ type, dim> * dst)

#define DECLARE_BROADCAST_1D(type)                                             \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_broadcast_1d_##type(memref_t<__ubuf__ type, 1> *src,            \
                                   memref_t<__ubuf__ type, 1> *dst)

#define REGISTE_BROADCAST_1D(type)                                             \
  DECLARE_BROADCAST_1D(type) { broadcast_scalar<type>(src, dst); }

#define DECLARE_BRC_FIRST_AXIS_ALIGN(dim, type)                                \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_broadcast_first_axis_align_##dim##d_##type(                     \
      memref_t<__ubuf__ type, dim> *src, memref_t<__ubuf__ type, dim> *dst)

#define REGISTE_BRC_FIRST_AXIS_ALIGN(dim, type)                                \
  DECLARE_BRC_FIRST_AXIS_ALIGN(dim, type) {                                    \
    if constexpr (!std::is_same<type, bool>::value) {                          \
      brc_first_axis_align_core<type>(src, dst);                               \
    } else {                                                                   \
      brc_first_axis_align_core(src, dst);                                     \
    }                                                                          \
  }

#define DECLARE_BRC_FIRST_AXIS_UNALIGN(dim, type)                              \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_broadcast_first_axis_unalign_##dim##d_##type(                   \
      memref_t<__ubuf__ type, dim> *src, memref_t<__ubuf__ type, dim> *dst,    \
      memref_t<__ubuf__ type, 1> *tmp)

#define REGISTE_BRC_FIRST_AXIS_UNALIGN(dim, type)                              \
  DECLARE_BRC_FIRST_AXIS_UNALIGN(dim, type) {                                  \
    brc_first_axis_unalign_core<type>(src, dst, tmp);                          \
  }

#define DECLARE_BRC_FIRST_AXIS_UNKNOWN_ALIGN(dim, type)                        \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_broadcast_first_axis_unknown_align_##dim##d_##type(             \
      memref_t<__ubuf__ type, dim> *src, memref_t<__ubuf__ type, dim> *dst,    \
      memref_t<__ubuf__ type, 1> *tmp)

#define REGISTE_BRC_FIRST_AXIS_UNKNOWN_ALIGN(dim, type)                        \
  DECLARE_BRC_FIRST_AXIS_UNKNOWN_ALIGN(dim, type) {                            \
    brc_first_axis_unknown_align_core<type>(src, dst, tmp);                    \
  }

#define DECLARE_BROADCAST_MID_AXIS_ALIGN(type)                                 \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_broadcast_middle_axis_align_3d_##type(                          \
      memref_t<__ubuf__ type, 3> *src, memref_t<__ubuf__ type, 3> *dst)

#define REGISTE_BROADCAST_MID_AXIS_ALIGN(type)                                 \
  DECLARE_BROADCAST_MID_AXIS_ALIGN(type) {                                     \
    vector_broadcast_mid_axis_align_3d<type>(src, dst);                        \
  }

#define DECLARE_BROADCAST_LAST_AXIS_ALIGN(type)                                \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_broadcast_last_axis_align_2d_##type(                            \
      memref_t<__ubuf__ type, 2> *src, memref_t<__ubuf__ type, 2> *dst,        \
      memref_t<__ubuf__ type, 1> *tmp)

#define REGISTE_BROADCAST_LAST_AXIS_ALIGN(type)                                \
  DECLARE_BROADCAST_LAST_AXIS_ALIGN(type) {                                    \
    vector_broadcast_last_axis_align_2d<type>(src, dst, tmp);                  \
  }

#define DECLARE_BROADCAST_LAST_AXIS_UNALIGN(type)                              \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_broadcast_last_axis_unalign_2d_##type(                          \
      memref_t<__ubuf__ type, 2> *src, memref_t<__ubuf__ type, 2> *dst,        \
      memref_t<__ubuf__ type, 1> *tmp)

#define REGISTE_BROADCAST_LAST_AXIS_UNALIGN(type)                              \
  DECLARE_BROADCAST_LAST_AXIS_UNALIGN(type) {                                  \
    vector_broadcast_last_axis_unalign_2d<type>(src, dst, tmp);                \
  }

#define DECLARE_BROADCAST_LAST_AXIS_UNKNOWN_ALIGN(type)                        \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_broadcast_last_axis_unknown_align_2d_##type(                    \
      memref_t<__ubuf__ type, 2> *src, memref_t<__ubuf__ type, 2> *dst,        \
      memref_t<__ubuf__ type, 1> *tmp)

#define REGISTE_BROADCAST_LAST_AXIS_UNKNOWN_ALIGN(type)                        \
  DECLARE_BROADCAST_LAST_AXIS_UNKNOWN_ALIGN(type) {                            \
    vector_broadcast_last_axis_unknown_align_2d<type>(src, dst, tmp);          \
  }

/// Declare broadcast first axis with
/// align/unalign/unknown_align
#define DECLARE_BRC_FIRST_AXIS(dim, type)                                      \
  DECLARE_BRC_FIRST_AXIS_ALIGN(dim, type);                                     \
  DECLARE_BRC_FIRST_AXIS_UNALIGN(dim, type);                                   \
  DECLARE_BRC_FIRST_AXIS_UNKNOWN_ALIGN(dim, type)

/// Registe broadcast first axis with
/// align/unalign/unknown_align
#define REGISTE_BRC_FIRST_AXIS(dim, type)                                      \
  REGISTE_BRC_FIRST_AXIS_ALIGN(dim, type)                                      \
  REGISTE_BRC_FIRST_AXIS_UNALIGN(dim, type)                                    \
  REGISTE_BRC_FIRST_AXIS_UNKNOWN_ALIGN(dim, type)

/// Declare broadcast mid axis with align
#define DECLARE_BROADCAST_MID_AXIS(type) DECLARE_BROADCAST_MID_AXIS_ALIGN(type)

/// Registe broadcast mid axis with align
#define REGISTE_BROADCAST_MID_AXIS(type) REGISTE_BROADCAST_MID_AXIS_ALIGN(type)

/// Declare broadcast last axis with
/// align/unalign/unknown_align
#define DECLARE_BROADCAST_LAST_AXIS(type)                                      \
  DECLARE_BROADCAST_LAST_AXIS_ALIGN(type);                                     \
  DECLARE_BROADCAST_LAST_AXIS_UNALIGN(type);                                   \
  DECLARE_BROADCAST_LAST_AXIS_UNKNOWN_ALIGN(type)

/// Registe broadcast last axis with
/// align/unalign/unknown_align
#define REGISTE_BROADCAST_LAST_AXIS(type)                                      \
  REGISTE_BROADCAST_LAST_AXIS_ALIGN(type)                                      \
  REGISTE_BROADCAST_LAST_AXIS_UNALIGN(type)                                    \
  REGISTE_BROADCAST_LAST_AXIS_UNKNOWN_ALIGN(type)

extern "C" {
//===-------------------------------------------------------------------===//
// broadcast scalar to 1d
//===-------------------------------------------------------------------===//
DECLARE_BROADCAST_SCALAR(1, int8_t);
DECLARE_BROADCAST_SCALAR(1, uint8_t);
DECLARE_BROADCAST_SCALAR(1, bfloat16_t);
DECLARE_BROADCAST_SCALAR(1, int16_t);
DECLARE_BROADCAST_SCALAR(1, uint16_t);
DECLARE_BROADCAST_SCALAR(1, half);
DECLARE_BROADCAST_SCALAR(1, int32_t);
DECLARE_BROADCAST_SCALAR(1, uint32_t);
DECLARE_BROADCAST_SCALAR(1, float);
DECLARE_BROADCAST_SCALAR(1, int64_t);
DECLARE_BROADCAST_SCALAR(1, uint64_t);
DECLARE_BROADCAST_SCALAR(1, bool);

//===-------------------------------------------------------------------===//
// broadcast scalar to 2d
//===-------------------------------------------------------------------===//
DECLARE_BROADCAST_SCALAR(2, int8_t);
DECLARE_BROADCAST_SCALAR(2, uint8_t);
DECLARE_BROADCAST_SCALAR(2, bfloat16_t);
DECLARE_BROADCAST_SCALAR(2, int16_t);
DECLARE_BROADCAST_SCALAR(2, uint16_t);
DECLARE_BROADCAST_SCALAR(2, half);
DECLARE_BROADCAST_SCALAR(2, int32_t);
DECLARE_BROADCAST_SCALAR(2, uint32_t);
DECLARE_BROADCAST_SCALAR(2, float);
DECLARE_BROADCAST_SCALAR(2, int64_t);
DECLARE_BROADCAST_SCALAR(2, uint64_t);
DECLARE_BROADCAST_SCALAR(2, bool);

//===-------------------------------------------------------------------===//
// broadcast 1d
//===-------------------------------------------------------------------===//
DECLARE_BROADCAST_1D(int8_t);
DECLARE_BROADCAST_1D(uint8_t);
DECLARE_BROADCAST_1D(bfloat16_t);
DECLARE_BROADCAST_1D(int16_t);
DECLARE_BROADCAST_1D(uint16_t);
DECLARE_BROADCAST_1D(int32_t);
DECLARE_BROADCAST_1D(uint32_t);
DECLARE_BROADCAST_1D(half);
DECLARE_BROADCAST_1D(float);
DECLARE_BROADCAST_1D(int64_t);
DECLARE_BROADCAST_1D(uint64_t);
DECLARE_BROADCAST_1D(bool);

//===-------------------------------------------------------------------===//
// broadcast first axis, 2 dim
//===-------------------------------------------------------------------===//
DECLARE_BRC_FIRST_AXIS(2, bfloat16_t);
DECLARE_BRC_FIRST_AXIS(2, int16_t);
DECLARE_BRC_FIRST_AXIS(2, uint16_t);
DECLARE_BRC_FIRST_AXIS(2, int32_t);
DECLARE_BRC_FIRST_AXIS(2, uint32_t);
DECLARE_BRC_FIRST_AXIS(2, half);
DECLARE_BRC_FIRST_AXIS(2, float);
DECLARE_BRC_FIRST_AXIS(2, int64_t);
DECLARE_BRC_FIRST_AXIS(2, uint64_t);
DECLARE_BRC_FIRST_AXIS_ALIGN(2, bool);
DECLARE_BRC_FIRST_AXIS_ALIGN(2, int8_t);
DECLARE_BRC_FIRST_AXIS_ALIGN(2, uint8_t);

//===-------------------------------------------------------------------===//
// broadcast first axis, 3 dim
//===-------------------------------------------------------------------===//
DECLARE_BRC_FIRST_AXIS(3, bfloat16_t);
DECLARE_BRC_FIRST_AXIS(3, int16_t);
DECLARE_BRC_FIRST_AXIS(3, uint16_t);
DECLARE_BRC_FIRST_AXIS(3, int32_t);
DECLARE_BRC_FIRST_AXIS(3, uint32_t);
DECLARE_BRC_FIRST_AXIS(3, half);
DECLARE_BRC_FIRST_AXIS(3, float);
DECLARE_BRC_FIRST_AXIS(3, int64_t);
DECLARE_BRC_FIRST_AXIS(3, uint64_t);
DECLARE_BRC_FIRST_AXIS_ALIGN(3, bool);
DECLARE_BRC_FIRST_AXIS_ALIGN(3, int8_t);
DECLARE_BRC_FIRST_AXIS_ALIGN(3, uint8_t);

//===-------------------------------------------------------------------===//
// broadcast mid axis, 3 dim
//===-------------------------------------------------------------------===//
DECLARE_BROADCAST_MID_AXIS(bfloat16_t);
DECLARE_BROADCAST_MID_AXIS(int16_t);
DECLARE_BROADCAST_MID_AXIS(uint16_t);
DECLARE_BROADCAST_MID_AXIS(half);
DECLARE_BROADCAST_MID_AXIS(int32_t);
DECLARE_BROADCAST_MID_AXIS(uint32_t);
DECLARE_BROADCAST_MID_AXIS(float);
DECLARE_BROADCAST_MID_AXIS(bool);

//===-------------------------------------------------------------------===//
// broadcast last axis, 2 dim
//===-------------------------------------------------------------------===//
DECLARE_BROADCAST_LAST_AXIS(bfloat16_t);
DECLARE_BROADCAST_LAST_AXIS(int16_t);
DECLARE_BROADCAST_LAST_AXIS(uint16_t);
DECLARE_BROADCAST_LAST_AXIS(half);
DECLARE_BROADCAST_LAST_AXIS(int32_t);
DECLARE_BROADCAST_LAST_AXIS(uint32_t);
DECLARE_BROADCAST_LAST_AXIS(float);
DECLARE_BROADCAST_LAST_AXIS(int64_t);
DECLARE_BROADCAST_LAST_AXIS(uint64_t);
DECLARE_BROADCAST_LAST_AXIS_ALIGN(bool);
}

#endif // BISHENGIR_LIB_TEMPLATE_INCLUDE_BROADCAST_UTILS_H
