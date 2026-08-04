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

//===----------------------------------------------------------------------===//
// Unordered lock implemented as a Lamport bakery lock over physical blocks.
// A fixed token order is not valid here: CV kernels such as FZE can skip the
// guarded region on some blocks, so the lock must ignore non-participants.
//
// lock_var layout (sized by the host infer-num callback to
// 1 + 2 * SYNC_LOCK_MAX_PARTICIPANTS cache lines):
//   participant_num at i64 index 0
//   choosing[p] at i64 index (1 + p) * SYNC_LOCK_CACHELINE_I64
//   ticket[p] at i64 index
//       (1 + SYNC_LOCK_MAX_PARTICIPANTS + p) * SYNC_LOCK_CACHELINE_I64
// Each slot occupies a full cache line because dcci works at cache-line
// granularity; packing slots can let one core's writeback clobber another.
// NOTE: SYNC_LOCK_MAX_PARTICIPANTS must match kMaxSyncBlockParticipants in
// InsertInferSyncBlockLockNumAndInitFunc.cpp.
//===----------------------------------------------------------------------===//
static constexpr int64_t SYNC_LOCK_CACHELINE_I64 = 8;
static constexpr int64_t SYNC_LOCK_MAX_PARTICIPANTS = 1024;
static constexpr int64_t SYNC_LOCK_CHOOSING_BASE_I64 =
    SYNC_LOCK_CACHELINE_I64;
static constexpr int64_t SYNC_LOCK_TICKET_BASE_I64 =
    (1 + SYNC_LOCK_MAX_PARTICIPANTS) * SYNC_LOCK_CACHELINE_I64;

__aiv__ __attribute__((always_inline)) int64_t sync_lock_participant_id() {
  return INTRINSIC_NO_ARGS(get_block_idx);
}

// Cross-core GM coherence follows the existing ordered lock's dcci pattern:
// store/read the scalar slot and dcci the corresponding cache line.
static constexpr int64_t SYNC_LOCK_DCCI_MODE = 1;
__aiv__ __attribute__((always_inline)) void
lock_gm_store_i64(volatile __gm__ int64_t *p, int64_t v) {
  *p = v; // scalar GM store
  __asm__ __volatile__("");
  INTRINSIC(dcci, (__gm__ int64_t *)p, SYNC_LOCK_DCCI_MODE);
  __asm__ __volatile__("");
}
__aiv__ __attribute__((always_inline)) int64_t
lock_gm_load_i64(volatile __gm__ int64_t *p) {
  __asm__ __volatile__("");
  INTRINSIC(dcci, (__gm__ int64_t *)p, SYNC_LOCK_DCCI_MODE);
  __asm__ __volatile__("");
  return *p;
}

__aiv__ __attribute__((always_inline)) int64_t
sync_lock_participant_num(__gm__ int64_t *base) {
  // The launcher writes the actual launch-time participant count here after it
  // clamps blockNum to the target's physical block capacity. If the metadata is
  // missing or malformed, fall back to the full buffer capacity for correctness.
  volatile __gm__ int64_t *participant_num_ptr = base;
  int64_t participant_num = lock_gm_load_i64(participant_num_ptr);
  if (participant_num <= 0 ||
      participant_num > SYNC_LOCK_MAX_PARTICIPANTS)
    return SYNC_LOCK_MAX_PARTICIPANTS;
  return participant_num;
}

__aiv__ __attribute__((always_inline)) void
sync_block_lock_unordered(memref_t<__gm__ int64_t, 1> *lock_var) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
#else
  const int64_t i = sync_lock_participant_id();
  __gm__ int64_t *base = lock_var->aligned + lock_var->offset;
  const int64_t N = sync_lock_participant_num(base);
  if (N <= 1)
    return;

  volatile __gm__ int64_t *choosing_i =
      base + SYNC_LOCK_CHOOSING_BASE_I64 + i * SYNC_LOCK_CACHELINE_I64;
  volatile __gm__ int64_t *ticket_i =
      base + SYNC_LOCK_TICKET_BASE_I64 + i * SYNC_LOCK_CACHELINE_I64;

  lock_gm_store_i64(choosing_i, 1);
  INTRINSIC(pipe_barrier, PIPE_ALL);

  int64_t max_ticket = 0;
  for (int64_t j = 0; j < N; ++j) {
    volatile __gm__ int64_t *ticket_j =
        base + SYNC_LOCK_TICKET_BASE_I64 + j * SYNC_LOCK_CACHELINE_I64;
    int64_t ticket = lock_gm_load_i64(ticket_j);
    if (ticket > max_ticket)
      max_ticket = ticket;
  }

  const int64_t my_ticket = max_ticket + 1;
  lock_gm_store_i64(ticket_i, my_ticket);
  INTRINSIC(pipe_barrier, PIPE_ALL);
  lock_gm_store_i64(choosing_i, 0);
  INTRINSIC(pipe_barrier, PIPE_ALL);

  for (int64_t j = 0; j < N; ++j) {
    if (j == i)
      continue;

    volatile __gm__ int64_t *choosing_j =
        base + SYNC_LOCK_CHOOSING_BASE_I64 + j * SYNC_LOCK_CACHELINE_I64;
    volatile __gm__ int64_t *ticket_j =
        base + SYNC_LOCK_TICKET_BASE_I64 + j * SYNC_LOCK_CACHELINE_I64;

    while (lock_gm_load_i64(choosing_j) != 0) {
      continue;
    }

    for (;;) {
      int64_t other_ticket = lock_gm_load_i64(ticket_j);
      bool other_before =
          other_ticket != 0 &&
          (other_ticket < my_ticket ||
           (other_ticket == my_ticket && j < i));
      if (!other_before)
        break;
    }
  }

  INTRINSIC(pipe_barrier, PIPE_ALL);
#endif
}

__aiv__ __attribute__((always_inline)) void
sync_block_unlock_unordered(memref_t<__gm__ int64_t, 1> *lock_var) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
#else
  INTRINSIC(pipe_barrier, PIPE_ALL);
  const int64_t i = sync_lock_participant_id();
  __gm__ int64_t *base = lock_var->aligned + lock_var->offset;
  volatile __gm__ int64_t *ticket_i =
      base + SYNC_LOCK_TICKET_BASE_I64 + i * SYNC_LOCK_CACHELINE_I64;
  lock_gm_store_i64(ticket_i, 0);
#endif
}

/// free_lock_var_unordered: bakery release is idempotent. Clearing an
/// already-zero ticket is a no-op, so this is safe for paths that skipped the
/// normal lock/unlock pair.
__aiv__ __attribute__((always_inline)) void
free_lock_var_unordered(memref_t<__gm__ int64_t, 1> *lock_var) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
#else
  INTRINSIC(pipe_barrier, PIPE_ALL);
  const int64_t i = sync_lock_participant_id();
  __gm__ int64_t *base = lock_var->aligned + lock_var->offset;
  volatile __gm__ int64_t *ticket_i =
      base + SYNC_LOCK_TICKET_BASE_I64 + i * SYNC_LOCK_CACHELINE_I64;
  lock_gm_store_i64(ticket_i, 0);
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

//===-------------------------------------------------------------------===//
// sync_block_lock_unordered (Lamport bakery)
//===-------------------------------------------------------------------===//
REGISTE_SYNCBLOCKLOCK_UNORDERED();

//===-------------------------------------------------------------------===//
// sync_block_unlock_unordered
//===-------------------------------------------------------------------===//
REGISTE_SYNCBLOCKUNLOCK_UNORDERED();

//===-------------------------------------------------------------------===//
// free_lock_var_unordered
//===-------------------------------------------------------------------===//
REGISTE_FREE_LOCK_VAR_UNORDERED();
}
