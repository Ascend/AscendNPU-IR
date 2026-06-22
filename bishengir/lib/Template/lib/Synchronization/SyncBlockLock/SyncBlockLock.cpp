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

#include "Synchronization/SyncUtils.h"
#include "Utils.h"

/// sync_block_lock op description:
/// loop until the value in lock_var equals the block_idx,
/// it is part of the SyncBlockLock mechanism
///
/// \param lock_var (type: memref<1 x i64>)
///
/// Constraints:
/// 1. currently only works for vector ops
__aiv__ __attribute__((always_inline)) void
sync_block_lock(memref_t<__gm__ int64_t, 1> *lock_var) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
#else
  int64_t block_idx = INTRINSIC_NO_ARGS(get_block_idx);
  volatile __gm__ int64_t *lock_var_ptr = lock_var->aligned + lock_var->offset;
  // using dcci to avoid aicore from loading data from cache while the actual
  // value of lock is changed
  INTRINSIC(dcci, lock_var_ptr, 1);
  volatile int64_t lock_val = *lock_var_ptr;
  while (lock_val != block_idx) {
    INTRINSIC(dcci, lock_var_ptr, 1);
    lock_val = *lock_var_ptr;
    continue;
  }
#endif
}

/// sync_block_unlock op description:
/// increase the value in lock_var by 1 and release the lock_var for the current
/// block
///
/// \param lock_var (type: memref<1 x i64>)
///
/// Constraints:
/// 1. currently only works for vector ops
__aiv__ __attribute__((always_inline)) void
sync_block_unlock(memref_t<__gm__ int64_t, 1> *lock_var) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
#else
  INTRINSIC(pipe_barrier, PIPE_ALL);
  int64_t block_idx = INTRINSIC_NO_ARGS(get_block_idx);
  __gm__ int64_t *lock_var_ptr = lock_var->aligned + lock_var->offset;
  int64_t new_lock_val = block_idx + 1;
  *lock_var_ptr = new_lock_val;
  // insert dcci when writing into lock and invalidate cache
  INTRINSIC(dcci, lock_var_ptr, 1);
#endif
}

/// sync_block_lock_with_subblock: same as sync_block_lock but with block_idx
/// computed as get_block_idx() * get_subblockdim() + get_subblockid().
__aiv__ __attribute__((always_inline)) void
sync_block_lock_with_subblock(memref_t<__gm__ int64_t, 1> *lock_var) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
#else
  int64_t block_idx =
      INTRINSIC_NO_ARGS(get_block_idx) * INTRINSIC_NO_ARGS(get_subblockdim) +
      INTRINSIC_NO_ARGS(get_subblockid);
  volatile __gm__ int64_t *lock_var_ptr = lock_var->aligned + lock_var->offset;
  INTRINSIC(dcci, lock_var_ptr, 1);
  volatile int64_t lock_val = *lock_var_ptr;
  while (lock_val != block_idx) {
    INTRINSIC(dcci, lock_var_ptr, 1);
    lock_val = *lock_var_ptr;
    continue;
  }
#endif
}

/// sync_block_unlock_with_subblock: same as sync_block_unlock but with
/// block_idx computed as get_block_idx() * get_subblockdim() +
/// get_subblockid().
__aiv__ __attribute__((always_inline)) void
sync_block_unlock_with_subblock(memref_t<__gm__ int64_t, 1> *lock_var) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
#else
  INTRINSIC(pipe_barrier, PIPE_ALL);
  int64_t block_idx =
      INTRINSIC_NO_ARGS(get_block_idx) * INTRINSIC_NO_ARGS(get_subblockdim) +
      INTRINSIC_NO_ARGS(get_subblockid);
  __gm__ int64_t *lock_var_ptr = lock_var->aligned + lock_var->offset;
  int64_t new_lock_val = block_idx + 1;
  *lock_var_ptr = new_lock_val;
  INTRINSIC(dcci, lock_var_ptr, 1);
#endif
}

/// free_lock_var: Ensures this block advances lock_var by one step when a path
/// skipped the normal unlock (e.g. if/else vs return). If this block already
/// ran unlock, lock_var is already > block_idx and sync_block_lock would spin
/// forever; read first and no-op in that case.
__aiv__ __attribute__((always_inline)) void
free_lock_var(memref_t<__gm__ int64_t, 1> *lock_var) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
#else
  INTRINSIC(pipe_barrier, PIPE_ALL);
  int64_t block_idx = INTRINSIC_NO_ARGS(get_block_idx);
  volatile __gm__ int64_t *lock_var_ptr = lock_var->aligned + lock_var->offset;
  INTRINSIC(dcci, lock_var_ptr, 1);
  volatile int64_t lock_val = *lock_var_ptr;
  if (lock_val > block_idx)
    return;
  sync_block_lock(lock_var);
  sync_block_unlock(lock_var);
#endif
}

/// free_lock_var_with_subblock: same semantics as free_lock_var using subblock
/// block index.
__aiv__ __attribute__((always_inline)) void
free_lock_var_with_subblock(memref_t<__gm__ int64_t, 1> *lock_var) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
#else
  INTRINSIC(pipe_barrier, PIPE_ALL);
  int64_t block_idx =
      INTRINSIC_NO_ARGS(get_block_idx) * INTRINSIC_NO_ARGS(get_subblockdim) +
      INTRINSIC_NO_ARGS(get_subblockid);
  volatile __gm__ int64_t *lock_var_ptr = lock_var->aligned + lock_var->offset;
  INTRINSIC(dcci, lock_var_ptr, 1);
  volatile int64_t lock_val = *lock_var_ptr;
  if (lock_val > block_idx)
    return;
  sync_block_lock_with_subblock(lock_var);
  sync_block_unlock_with_subblock(lock_var);
#endif
}

extern "C" {
//===-------------------------------------------------------------------===//
// sync_block_lock
//===-------------------------------------------------------------------===//
REGISTE_SYNCBLOCKLOCK();

//===-------------------------------------------------------------------===//
// sync_block_unlock
//===-------------------------------------------------------------------===//
REGISTE_SYNCBLOCKUNLOCK();

//===-------------------------------------------------------------------===//
// sync_block_lock_with_subblock (block_idx = get_block_idx * get_subblockdim
// + get_subblockid)
//===-------------------------------------------------------------------===//
REGISTE_SYNCBLOCKLOCK_WITH_SUBBLOCK();

//===-------------------------------------------------------------------===//
// sync_block_unlock_with_subblock (block_idx = get_block_idx *
// get_subblockdim + get_subblockid)
//===-------------------------------------------------------------------===//
REGISTE_SYNCBLOCKUNLOCK_WITH_SUBBLOCK();

//===-------------------------------------------------------------------===//
// free_lock_var
//===-------------------------------------------------------------------===//
REGISTE_FREE_LOCK_VAR();

//===-------------------------------------------------------------------===//
// free_lock_var_with_subblock
//===-------------------------------------------------------------------===//
REGISTE_FREE_LOCK_VAR_WITH_SUBBLOCK();
}