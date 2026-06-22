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

#ifndef HIVM_MLIR_TEMPLATE_ND2NZ_UTILS_H
#define HIVM_MLIR_TEMPLATE_ND2NZ_UTILS_H
#include "Utils.h"

constexpr int32_t MAX_LEN_UNIT16 = 65535;

template <typename T> struct nd2nz_intrin_args {
  __cbuf__ T *dst_ptr;
  __gm__ T *src_ptr;
  uint8_t sid;
  uint16_t ndNum;
  uint16_t nValue;
  uint16_t dValue;
  uint16_t srcNdMatrixStride;
  uint16_t srcDValue;
  uint16_t dstNzC0Stride;
  uint16_t dstNzNStride;
  uint16_t dstNzMatrixStride;
};

template <typename T>
__aicore__ __attribute__((always_inline)) void
copy_gm_to_cbuf_intrin_core(nd2nz_intrin_args<T> args) {
  if constexpr (sizeof(T) == 1) {
    INTRINSIC(copy_gm_to_cbuf_multi_nd2nz_b8, args.dst_ptr, args.src_ptr,
              args.sid, args.ndNum, args.nValue, args.dValue,
              args.srcNdMatrixStride, args.srcDValue, args.dstNzC0Stride,
              args.dstNzNStride, args.dstNzMatrixStride);
  } else if constexpr (sizeof(T) == 2) {
    INTRINSIC(copy_gm_to_cbuf_multi_nd2nz_b16, args.dst_ptr, args.src_ptr,
              args.sid, args.ndNum, args.nValue, args.dValue,
              args.srcNdMatrixStride, args.srcDValue, args.dstNzC0Stride,
              args.dstNzNStride, args.dstNzMatrixStride);
  } else if constexpr (sizeof(T) == 4) {
    INTRINSIC(copy_gm_to_cbuf_multi_nd2nz_b32s, args.dst_ptr, args.src_ptr,
              args.sid, args.ndNum, args.nValue, args.dValue,
              args.srcNdMatrixStride, args.srcDValue, args.dstNzC0Stride,
              args.dstNzNStride, args.dstNzMatrixStride);
  } else {
    static_assert("nd2nz unsupport this data type");
  }
}

#define DECLARE_ND2NZ(src_scope, dst_scope, src_dim, dst_dim, type)            \
  __aicore__ __attribute__((always_inline)) void _mlir_ciface_nd2nz_##type(    \
      memref_t<__##src_scope##__ type, src_dim> *src,                          \
      memref_t<__##dst_scope##__ type, dst_dim> *dst)

#define REGISTE_ND2NZ(src_scope, dst_scope, src_dim, dst_dim, type)            \
  DECLARE_ND2NZ(src_scope, dst_scope, src_dim, dst_dim, type) {                \
    copy_##src_scope##_to_##dst_scope##_multi_nd2nz_core<type, false>(src,     \
                                                                      dst);    \
  }

#define DECLARE_ND2NZ_FORBIAS(src_scope, dst_scope, src_dim, dst_dim, type)    \
  __aicore__                                                                   \
      __attribute__((always_inline)) void _mlir_ciface_nd2nz_forbias_##type(   \
          memref_t<__##src_scope##__ type, src_dim> *src,                      \
          memref_t<__##dst_scope##__ type, dst_dim> *dst)

#define REGISTE_ND2NZ_FORBIAS(src_scope, dst_scope, src_dim, dst_dim, type)    \
  DECLARE_ND2NZ_FORBIAS(src_scope, dst_scope, src_dim, dst_dim, type) {        \
    copy_##src_scope##_to_##dst_scope##_multi_nd2nz_core<type, true>(src,      \
                                                                     dst);     \
  }

extern "C" {
DECLARE_ND2NZ(gm, cbuf, 2, 4, half);
DECLARE_ND2NZ(gm, cbuf, 2, 4, float);
DECLARE_ND2NZ(gm, cbuf, 2, 4, bfloat16_t);
DECLARE_ND2NZ(gm, cbuf, 2, 4, int32_t);
DECLARE_ND2NZ(gm, cbuf, 2, 4, uint32_t);
DECLARE_ND2NZ(gm, cbuf, 2, 4, int16_t);
DECLARE_ND2NZ(gm, cbuf, 2, 4, uint16_t);
DECLARE_ND2NZ(gm, cbuf, 2, 4, int8_t);
DECLARE_ND2NZ(gm, cbuf, 2, 4, uint8_t);
DECLARE_ND2NZ_FORBIAS(gm, cbuf, 2, 4, half);
DECLARE_ND2NZ_FORBIAS(gm, cbuf, 2, 4, float);
DECLARE_ND2NZ_FORBIAS(gm, cbuf, 2, 4, bfloat16_t);
DECLARE_ND2NZ_FORBIAS(gm, cbuf, 2, 4, int32_t);
DECLARE_ND2NZ_FORBIAS(gm, cbuf, 2, 4, uint32_t);
DECLARE_ND2NZ_FORBIAS(gm, cbuf, 2, 4, int16_t);
DECLARE_ND2NZ_FORBIAS(gm, cbuf, 2, 4, uint16_t);
DECLARE_ND2NZ_FORBIAS(gm, cbuf, 2, 4, int8_t);
DECLARE_ND2NZ_FORBIAS(gm, cbuf, 2, 4, uint8_t);
}
#endif // HIVM_MLIR_TEMPLATE_ND2NZ_UTILS_H
