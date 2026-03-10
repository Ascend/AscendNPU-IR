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

#ifndef BISHENGIR_LIB_TEMPLATE_INCLUDE_SYNC_UTILS_H
#define BISHENGIR_LIB_TEMPLATE_INCLUDE_SYNC_UTILS_H

#include "Utils.h"
#include <type_traits>

__aiv__ __attribute__((always_inline)) void
sync_block_lock(memref_t<__gm__ int64_t, 1> *lock_var);

__aiv__ __attribute__((always_inline)) void
sync_block_unlock(memref_t<__gm__ int64_t, 1> *lock_var);

#define DECLARE_SYNCBLOCKLOCK()                                                \
  __aiv__ __attribute__((always_inline)) void _mlir_ciface_sync_block_lock(    \
      memref_t<__gm__ int64_t, 1> *lock_var)

#define REGISTE_SYNCBLOCKLOCK()                                                \
  DECLARE_SYNCBLOCKLOCK() { sync_block_lock(lock_var); }

#define DECLARE_SYNCBLOCKUNLOCK()                                              \
  __aiv__ __attribute__((always_inline)) void _mlir_ciface_sync_block_unlock(  \
      memref_t<__gm__ int64_t, 1> *lock_var)

#define REGISTE_SYNCBLOCKUNLOCK()                                              \
  DECLARE_SYNCBLOCKUNLOCK() { sync_block_unlock(lock_var); }

extern "C" {
//===-------------------------------------------------------------------===//
// sync_block_lock
//===-------------------------------------------------------------------===//
DECLARE_SYNCBLOCKLOCK();

//===-------------------------------------------------------------------===//
// sync_block_unlock
//===-------------------------------------------------------------------===//
DECLARE_SYNCBLOCKUNLOCK();
}
#endif // BISHENGIR_LIB_TEMPLATE_INCLUDE_SYNC_UTILS_H