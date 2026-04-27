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

#ifndef HIVM_MLIR_TEMPLATE_NZ2ND_UTILS_H
#define HIVM_MLIR_TEMPLATE_NZ2ND_UTILS_H
#include "Utils.h"

#define DECLARE_NZ2ND(type, src_dim, dst_dim)                                  \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_nz2nd_##src_dim##d_to_##dst_dim##d_##type(                  \
          memref_t<__cbuf__ type, src_dim> *src,                               \
          memref_t<__gm__ type, dst_dim> *dst)

#define REGISTER_NZ2ND(type, src_dim, dst_dim)                                 \
  DECLARE_NZ2ND(type, src_dim, dst_dim) {                                      \
    copy_cbuf_to_gm_##src_dim##d_to_##dst_dim##d_core<type>(src, dst);         \
  }

extern "C" {
DECLARE_NZ2ND(float, 4, 2);
DECLARE_NZ2ND(bfloat16_t, 4, 2);
DECLARE_NZ2ND(half, 4, 2);
DECLARE_NZ2ND(int8_t, 4, 2);
DECLARE_NZ2ND(uint8_t, 4, 2);
DECLARE_NZ2ND(int16_t, 4, 2);
DECLARE_NZ2ND(uint16_t, 4, 2);
DECLARE_NZ2ND(int32_t, 4, 2);
DECLARE_NZ2ND(uint32_t, 4, 2);
DECLARE_NZ2ND(int64_t, 4, 2);
DECLARE_NZ2ND(uint64_t, 4, 2);
}
#endif // HIVM_MLIR_TEMPLATE_NZ2ND_UTILS_H
