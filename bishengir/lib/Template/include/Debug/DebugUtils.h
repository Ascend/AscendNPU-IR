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

#ifndef HIVM_MLIR_TEMPLATE_DEBUG_UTILS_H
#define HIVM_MLIR_TEMPLATE_DEBUG_UTILS_H

#include "Utils.h"

// print

#define DECLARE_PRINT_SCALAR(type, mem)                                        \
  [aicore] __attribute__((always_inline)) void                                 \
      _mlir_ciface_print_scalar_##type##_##mem(                                \
          char *prefix, const int64_t len, type arg, const int8_t hex)

#define DECLARE_PRINT_TENSOR(dim, type, mem)                                   \
  [aicore] __attribute__((always_inline)) void                                 \
      _mlir_ciface_print_##dim##d_##type##_##mem(                              \
          char *prefix, const int64_t len,                                     \
          memref_t<__##mem##__ type, dim> *arg, const int8_t hex)

#define DECLARE_PRINT_1TO8D_TENSOR(type, mem)                                  \
  DECLARE_PRINT_TENSOR(1, type, mem);                                          \
  DECLARE_PRINT_TENSOR(2, type, mem);                                          \
  DECLARE_PRINT_TENSOR(3, type, mem);                                          \
  DECLARE_PRINT_TENSOR(4, type, mem);                                          \
  DECLARE_PRINT_TENSOR(5, type, mem);                                          \
  DECLARE_PRINT_TENSOR(6, type, mem);                                          \
  DECLARE_PRINT_TENSOR(7, type, mem);                                          \
  DECLARE_PRINT_TENSOR(8, type, mem)

#define REGISTER_PRINT_SCALAR(type, mem)                                       \
  DECLARE_PRINT_SCALAR(type, mem) {                                            \
    print_scalar_core<type>(prefix, len, arg, hex);                            \
  }

#define REGISTER_PRINT_TENSOR(dim, type, mem)                                  \
  DECLARE_PRINT_TENSOR(dim, type, mem) {                                       \
    print_##dim##d_core<type, __##mem##__ type>(prefix, len, arg, hex);        \
  }

#define REGISTER_PRINT_1TO8D_TENSOR(type, mem)                                 \
  REGISTER_PRINT_TENSOR(1, type, mem);                                         \
  REGISTER_PRINT_TENSOR(2, type, mem);                                         \
  REGISTER_PRINT_TENSOR(3, type, mem);                                         \
  REGISTER_PRINT_TENSOR(4, type, mem);                                         \
  REGISTER_PRINT_TENSOR(5, type, mem);                                         \
  REGISTER_PRINT_TENSOR(6, type, mem);                                         \
  REGISTER_PRINT_TENSOR(7, type, mem);                                         \
  REGISTER_PRINT_TENSOR(8, type, mem)

// assert

#define DECLARE_ASSERT_SCALAR(mem)                                             \
  [aicore] __attribute__((always_inline)) void                                 \
      _mlir_ciface_assert_scalar_bool_##mem(char *prefix, const int64_t len,   \
                                            bool arg)

#define DECLARE_ASSERT_TENSOR(dim, mem)                                        \
  [aicore] __attribute__((always_inline)) void                                 \
      _mlir_ciface_assert_##dim##d_int8_t_##mem(                               \
          char *prefix, const int64_t len,                                     \
          memref_t<__##mem##__ int8_t, dim> *arg)

#define DECLARE_ASSERT_1TO4D_TENSOR(mem)                                       \
  DECLARE_ASSERT_TENSOR(1, mem);                                               \
  DECLARE_ASSERT_TENSOR(2, mem);                                               \
  DECLARE_ASSERT_TENSOR(3, mem);                                               \
  DECLARE_ASSERT_TENSOR(4, mem)

#define REGISTER_ASSERT_SCALAR(mem)                                            \
  DECLARE_ASSERT_SCALAR(mem) { assert_scalar_core(prefix, len, arg); }

#define REGISTER_ASSERT_TENSOR(dim, mem)                                       \
  DECLARE_ASSERT_TENSOR(dim, mem) {                                            \
    assert_##dim##d_core<__##mem##__ int8_t>(prefix, len, arg);                \
  }

#define REGISTER_ASSERT_1TO4D_TENSOR(mem)                                      \
  REGISTER_ASSERT_TENSOR(1, mem)                                               \
  REGISTER_ASSERT_TENSOR(2, mem)                                               \
  REGISTER_ASSERT_TENSOR(3, mem)                                               \
  REGISTER_ASSERT_TENSOR(4, mem)

extern "C" {
// declare __gm__ versions for both cube core and vector core

DECLARE_PRINT_SCALAR(int8_t, gm);
DECLARE_PRINT_SCALAR(uint8_t, gm);
DECLARE_PRINT_SCALAR(int16_t, gm);
DECLARE_PRINT_SCALAR(uint16_t, gm);
DECLARE_PRINT_SCALAR(int32_t, gm);
DECLARE_PRINT_SCALAR(uint32_t, gm);
DECLARE_PRINT_SCALAR(int64_t, gm);
DECLARE_PRINT_SCALAR(half, gm);
DECLARE_PRINT_SCALAR(bfloat16_t, gm);
DECLARE_PRINT_SCALAR(float, gm);
DECLARE_PRINT_SCALAR(bool, gm);
DECLARE_PRINT_1TO8D_TENSOR(int8_t, gm);
DECLARE_PRINT_1TO8D_TENSOR(uint8_t, gm);
DECLARE_PRINT_1TO8D_TENSOR(int16_t, gm);
DECLARE_PRINT_1TO8D_TENSOR(uint16_t, gm);
DECLARE_PRINT_1TO8D_TENSOR(int32_t, gm);
DECLARE_PRINT_1TO8D_TENSOR(uint32_t, gm);
DECLARE_PRINT_1TO8D_TENSOR(int64_t, gm);
DECLARE_PRINT_1TO8D_TENSOR(half, gm);
DECLARE_PRINT_1TO8D_TENSOR(bfloat16_t, gm);
DECLARE_PRINT_1TO8D_TENSOR(float, gm);
DECLARE_PRINT_1TO8D_TENSOR(bool, gm);
DECLARE_ASSERT_SCALAR(gm);
DECLARE_ASSERT_1TO4D_TENSOR(gm);

// declare __ubuf__ versions for vector core
// Note: bisheng uses the following macro for both print and assert

#ifdef __CCE_ENABLE_PRINT_AICORE_VEC__
DECLARE_PRINT_SCALAR(int8_t, ubuf);
DECLARE_PRINT_SCALAR(uint8_t, ubuf);
DECLARE_PRINT_SCALAR(int16_t, ubuf);
DECLARE_PRINT_SCALAR(uint16_t, ubuf);
DECLARE_PRINT_SCALAR(int32_t, ubuf);
DECLARE_PRINT_SCALAR(uint32_t, ubuf);
DECLARE_PRINT_SCALAR(int64_t, ubuf);
DECLARE_PRINT_SCALAR(half, ubuf);
DECLARE_PRINT_SCALAR(bfloat16_t, ubuf);
DECLARE_PRINT_SCALAR(float, ubuf);
DECLARE_PRINT_SCALAR(bool, ubuf);
DECLARE_PRINT_1TO8D_TENSOR(int8_t, ubuf);
DECLARE_PRINT_1TO8D_TENSOR(uint8_t, ubuf);
DECLARE_PRINT_1TO8D_TENSOR(int16_t, ubuf);
DECLARE_PRINT_1TO8D_TENSOR(uint16_t, ubuf);
DECLARE_PRINT_1TO8D_TENSOR(int32_t, ubuf);
DECLARE_PRINT_1TO8D_TENSOR(uint32_t, ubuf);
DECLARE_PRINT_1TO8D_TENSOR(int64_t, ubuf);
DECLARE_PRINT_1TO8D_TENSOR(half, ubuf);
DECLARE_PRINT_1TO8D_TENSOR(bfloat16_t, ubuf);
DECLARE_PRINT_1TO8D_TENSOR(float, ubuf);
DECLARE_PRINT_1TO8D_TENSOR(bool, ubuf);
DECLARE_ASSERT_SCALAR(ubuf);
DECLARE_ASSERT_1TO4D_TENSOR(ubuf);
#endif

[aicore] __attribute__((always_inline)) void
_mlir_ciface_init_debug(__gm__ cce::internal::DebugTunnelData *DTData);
[aicore] __attribute__((always_inline)) void
_mlir_ciface_finish_debug(__gm__ cce::internal::DebugTunnelData *DTData);
}
#endif // HIVM_MLIR_TEMPLATE_DEBUG_UTILS_H
