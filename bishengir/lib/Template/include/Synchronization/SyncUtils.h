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

/// sync_block_lock_with_subblock / sync_block_unlock_with_subblock: use
/// block_idx = get_block_idx() * get_subblockdim() + get_subblockid()
__aiv__ __attribute__((always_inline)) void
sync_block_lock_with_subblock(memref_t<__gm__ int64_t, 1> *lock_var);

__aiv__ __attribute__((always_inline)) void
sync_block_unlock_with_subblock(memref_t<__gm__ int64_t, 1> *lock_var);

/// free_lock_var: If lock_var still equals this block's index, completes one
/// lock/unlock pair; if lock_var is already past this block (branch already
/// unlocked), no-op so sync_block_lock is not re-entered (would spin forever).
__aiv__ __attribute__((always_inline)) void
free_lock_var(memref_t<__gm__ int64_t, 1> *lock_var);

/// free_lock_var_with_subblock: Same as free_lock_var with subblock index.
__aiv__ __attribute__((always_inline)) void
free_lock_var_with_subblock(memref_t<__gm__ int64_t, 1> *lock_var);

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

#define DECLARE_SYNCBLOCKLOCK_WITH_SUBBLOCK()                                  \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_sync_block_lock_with_subblock(                                  \
      memref_t<__gm__ int64_t, 1> *lock_var)

#define REGISTE_SYNCBLOCKLOCK_WITH_SUBBLOCK()                                  \
  DECLARE_SYNCBLOCKLOCK_WITH_SUBBLOCK() {                                      \
    sync_block_lock_with_subblock(lock_var);                                   \
  }

#define DECLARE_SYNCBLOCKUNLOCK_WITH_SUBBLOCK()                                \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_sync_block_unlock_with_subblock(                                \
      memref_t<__gm__ int64_t, 1> *lock_var)

#define REGISTE_SYNCBLOCKUNLOCK_WITH_SUBBLOCK()                                \
  DECLARE_SYNCBLOCKUNLOCK_WITH_SUBBLOCK() {                                    \
    sync_block_unlock_with_subblock(lock_var);                                 \
  }

#define DECLARE_FREE_LOCK_VAR()                                                \
  __aiv__ __attribute__((always_inline)) void _mlir_ciface_free_lock_var(      \
      memref_t<__gm__ int64_t, 1> *lock_var)

#define REGISTE_FREE_LOCK_VAR()                                                \
  DECLARE_FREE_LOCK_VAR() { free_lock_var(lock_var); }

#define DECLARE_FREE_LOCK_VAR_WITH_SUBBLOCK()                                  \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_free_lock_var_with_subblock(                                    \
      memref_t<__gm__ int64_t, 1> *lock_var)

#define REGISTE_FREE_LOCK_VAR_WITH_SUBBLOCK()                                  \
  DECLARE_FREE_LOCK_VAR_WITH_SUBBLOCK() {                                      \
    free_lock_var_with_subblock(lock_var);                                     \
  }

extern "C" {
//===-------------------------------------------------------------------===//
// sync_block_lock
//===-------------------------------------------------------------------===//
DECLARE_SYNCBLOCKLOCK();

//===-------------------------------------------------------------------===//
// sync_block_unlock
//===-------------------------------------------------------------------===//
DECLARE_SYNCBLOCKUNLOCK();

//===-------------------------------------------------------------------===//
// sync_block_lock_with_subblock
//===-------------------------------------------------------------------===//
DECLARE_SYNCBLOCKLOCK_WITH_SUBBLOCK();

//===-------------------------------------------------------------------===//
// sync_block_unlock_with_subblock
//===-------------------------------------------------------------------===//
DECLARE_SYNCBLOCKUNLOCK_WITH_SUBBLOCK();

//===-------------------------------------------------------------------===//
// free_lock_var
//===-------------------------------------------------------------------===//
DECLARE_FREE_LOCK_VAR();

//===-------------------------------------------------------------------===//
// free_lock_var_with_subblock
//===-------------------------------------------------------------------===//
DECLARE_FREE_LOCK_VAR_WITH_SUBBLOCK();
}
#endif // BISHENGIR_LIB_TEMPLATE_INCLUDE_SYNC_UTILS_H