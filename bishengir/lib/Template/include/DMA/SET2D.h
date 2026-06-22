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

#ifndef HIVM_MLIR_TEMPLATE_SET2D_UTILS_H
#define HIVM_MLIR_TEMPLATE_SET2D_UTILS_H
#include "Utils.h"

constexpr uint64_t SET_2D_BLOCK_NUM_PER_REPEAT = 16;
constexpr uint64_t SET_2D_L1_REPEAT_GAP = 32;
constexpr uint64_t BLOCK_SIZE_L1 = 32;

template <typename T>
struct set2d_intrin_args {
  __cbuf__ T *dst_ptr;
  int64_t config = 1;
  T value;
};

template <typename T>
__aicore__ __attribute__((always_inline)) uint32_t pack_b8_to_u32(T src_val) {
  static_assert((sizeof(T) == 1) && "set2d do not support this data type");
  uint16_t val_low = ((uint32_t)((uint32_t)src_val << 8ULL)) >> 8ULL;
  uint16_t dst_val = ((uint32_t)(val_low << 8ULL)) +
                     ((uint32_t)(val_low << 16ULL)) +
                     ((uint32_t)(val_low << 24ULL)) + val_low;
  return dst_val;
}

template <typename T>
__aicore__ __attribute__((always_inline)) void
set_cbuf_to_2d_intrin_core(set2d_intrin_args<T> args) {
  if constexpr (sizeof(T) == 1) {
    INTRINSIC(create_cbuf_matrix, (__cbuf__ uint16_t *)args.dst_ptr,
              args.config, pack_b8_to_u32(args.value));
  } else if constexpr (sizeof(T) == 2) {
    INTRINSIC(create_cbuf_matrix, args.dst_ptr, args.config, args.value);
  } else if constexpr (sizeof(T) == 4) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
    INTRINSIC(create_cbuf_matrix, args.dst_ptr, args.config,
              half({static_cast<unsigned short>(args.value)}));
#else
    INTRINSIC(create_cbuf_matrix, args.dst_ptr, args.config,
              static_cast<half>(args.value));
#endif
  } else {
    static_assert("set 2d unsupport this data type");
  }
}

#define DECLARE_SET2D(dst_scope, dst_dim, type)                                \
  __aicore__                                                                   \
      __attribute__((always_inline)) void _mlir_ciface_set_l1_2d_##type(       \
          type scalar, memref_t<__##dst_scope##__ type, dst_dim> *dst)

#define REGISTE_SET2D(dst_scope, dst_dim, type)                                \
  DECLARE_SET2D(dst_scope, dst_dim, type) {                                    \
    set_##dst_scope##_to_2d_core<type>(scalar, dst);                      \
  }

extern "C" {
DECLARE_SET2D(cbuf, 1, int8_t);
DECLARE_SET2D(cbuf, 1, half);
DECLARE_SET2D(cbuf, 1, float);
DECLARE_SET2D(cbuf, 1, bfloat16_t);
}
#endif // HIVM_MLIR_TEMPLATE_SET2D_UTILS_H
