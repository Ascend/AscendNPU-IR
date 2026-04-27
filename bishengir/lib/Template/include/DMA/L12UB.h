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

#ifndef HIVM_MLIR_TEMPLATE_L12UB_UTILS_H
#define HIVM_MLIR_TEMPLATE_L12UB_UTILS_H
#include "Utils.h"

#define DECLARE_L12UB(type, src_dim, dst_dim)                                  \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_l12ub_##src_dim##d_to_##dst_dim##d_##type(                  \
          memref_t<__cbuf__ type, src_dim> *src,                               \
          memref_t<__ubuf__ type, dst_dim> *dst)

#define REGISTER_L12UB(type, src_dim, dst_dim)                                 \
  DECLARE_L12UB(type, src_dim, dst_dim) {                                      \
    copy_cbuf_to_ub_##src_dim##d_to_##dst_dim##d_core<type>(src, dst);         \
  }

extern "C" {
DECLARE_L12UB(float, 4, 2);
DECLARE_L12UB(bfloat16_t, 4, 2);
DECLARE_L12UB(half, 4, 2);
DECLARE_L12UB(int8_t, 4, 2);
DECLARE_L12UB(uint8_t, 4, 2);
DECLARE_L12UB(int16_t, 4, 2);
DECLARE_L12UB(uint16_t, 4, 2);
DECLARE_L12UB(int32_t, 4, 2);
DECLARE_L12UB(uint32_t, 4, 2);
DECLARE_L12UB(int64_t, 4, 2);
DECLARE_L12UB(uint64_t, 4, 2);
}
#endif // HIVM_MLIR_TEMPLATE_L12UB_UTILS_H
