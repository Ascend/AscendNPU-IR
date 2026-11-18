//===- TcbPool.cpp - Cube TCB pool allocation -----------------------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "RegBase/TcbPoolUtils.h"

namespace {
// First-fit search. slotCount is the sentinel when no contiguous range fits.
__aicore__ inline uint32_t findFreeSlots(const tcb_pool::PoolMetadata &pool,
                                         uint32_t begin, uint32_t end,
                                         uint32_t slots, uint32_t slotCount) {
  uint32_t run = 0;
  for (uint32_t i = begin; i < end; ++i) {
    run = pool.meta[tcb_pool::kSlotOwnersOffset + i] == tcb_pool::kFree
              ? run + 1
              : 0;
    if (run == slots)
      return i + 1 - slots;
  }
  return slotCount;
}

// The target intrinsics operate on 64-bit Scalar values. SSBUF keeps the
// low 16-bit ID bitmap in an i32; only Cube mutates it.
__aicore__ inline uint32_t allocateFlagId(const tcb_pool::PoolMetadata &pool) {
  uint32_t mask = pool.freeFlagMask();
  TCB_POOL_ASSERT(mask != 0);
#if ENABLE_CPU_TRACE_INTRINSIC || defined(__CCE_KT_TEST__)
  uint32_t id = __builtin_ctz(mask);
  pool.setFreeFlagMask(mask & ~(uint32_t(1) << id));
#else
  uint32_t id = INTRINSIC(sff1, uint64_t(mask));
  pool.setFreeFlagMask(
      uint32_t(INTRINSIC(sbitset0, uint64_t(mask), uint64_t(id))));
#endif
  return id;
}

__aicore__ inline void releaseFlagId(const tcb_pool::PoolMetadata &pool,
                                     uint32_t id) {
  uint32_t mask = pool.freeFlagMask();
#if ENABLE_CPU_TRACE_INTRINSIC || defined(__CCE_KT_TEST__)
  pool.setFreeFlagMask(mask | (uint32_t(1) << id));
#else
  pool.setFreeFlagMask(
      uint32_t(INTRINSIC(sbitset1, uint64_t(mask), uint64_t(id))));
#endif
}

// Reclaim one allocation in AIV0's feedback order. The FIX waits protect UB
// reuse until both AIV consumers complete their work.
__aicore__ __attribute__((always_inline)) inline void
popReleaseEntry(const tcb_pool::PoolMetadata &pool, uint32_t slotCount,
                uint32_t flagCount) {
  auto meta = pool.meta;
  uint32_t head = pool.releaseHead();
  while (head == pool.releaseTail()) {
  }
  uint32_t releaseOffset = pool.releaseEntryOffset(head, slotCount, flagCount);
  uint32_t queueId = meta[releaseOffset];
  uint32_t startSlot = meta[releaseOffset + 1];
  uint32_t slots = pool.slotsPerBuffer(queueId);
  uint32_t id = meta[releaseOffset + 2];

  // Match SyncBlockWaitOp's CV1:2 lowering: consume both AIV notifications
  // in the intra-block flag space before reusing this allocation's slots/ID.
  INTRINSIC(wait_intra_block, PIPE_FIX, id + 16);
  INTRINSIC(wait_intra_block, PIPE_FIX, id);
  for (uint32_t i = 0; i < slots; ++i)
    meta[tcb_pool::kSlotOwnersOffset + startSlot + i] = tcb_pool::kFree;
  releaseFlagId(pool, id);
  meta[tcb_pool::kReleaseHeadOffset] = head + 1;
}
} // namespace

extern "C" {
__aicore__ __attribute__((always_inline)) void _mlir_ciface_tcb_pool_init(
    tcb_pool::MetadataMemRef *metadata, uint32_t slotBytes, uint32_t slotCount,
    uint32_t slotsPerBuffer0, uint32_t slotsPerBuffer1, uint32_t searchBegin0,
    uint32_t searchBegin1, uint32_t searchEnd0, uint32_t searchEnd1,
    uint32_t flagMask) {
  TCB_POOL_ASSERT(slotBytes > 0 && slotCount > 0 && flagMask > 0 &&
                  flagMask <= 0xffff);
  auto meta = tcb_pool::getMetadataPointer(metadata);
  meta[tcb_pool::kSlotBytesOffset] = slotBytes;
  meta[tcb_pool::kSlotCountOffset] = slotCount;
  meta[tcb_pool::kSlotsPerBufferOffset] = slotsPerBuffer0;
  meta[tcb_pool::kSlotsPerBufferOffset + 1] = slotsPerBuffer1;
  meta[tcb_pool::kSearchBeginOffset] = searchBegin0;
  meta[tcb_pool::kSearchBeginOffset + 1] = searchBegin1;
  meta[tcb_pool::kSearchEndOffset] = searchEnd0;
  meta[tcb_pool::kSearchEndOffset + 1] = searchEnd1;
  meta[tcb_pool::kFlagMaskOffset] = flagMask;
  tcb_pool::PoolMetadata pool{meta};
  for (uint32_t q = 0; q < tcb_pool::kQueueCount; ++q) {
    TCB_POOL_ASSERT(meta[tcb_pool::kSearchBeginOffset + q] <
                        meta[tcb_pool::kSearchEndOffset + q] &&
                    meta[tcb_pool::kSearchEndOffset + q] <= slotCount &&
                    pool.slotsPerBuffer(q) > 0 &&
                    pool.slotsPerBuffer(q) <=
                        meta[tcb_pool::kSearchEndOffset + q] -
                            meta[tcb_pool::kSearchBeginOffset + q]);
  }
  TCB_POOL_ASSERT(metadata->sizes[0] >= pool.metadataWords());
  meta[tcb_pool::kReleaseHeadOffset] = 0;
  meta[tcb_pool::kReleaseTailOffset] = 0;
  pool.setFreeFlagMask(flagMask);
  for (uint32_t i = tcb_pool::kAllocIndexOffset;
       i < tcb_pool::kSlotOwnersOffset; ++i)
    meta[i] = 0;
  for (uint32_t i = 0; i < slotCount; ++i)
    meta[tcb_pool::kSlotOwnersOffset + i] = tcb_pool::kFree;
}

__aicore__ __attribute__((always_inline)) void
_mlir_ciface_tcb_pool_alloc(tcb_pool::Allocation *result,
                            tcb_pool::MetadataMemRef *metadata,
                            uint64_t poolBase, uint32_t queueId) {
  TCB_POOL_ASSERT(queueId < tcb_pool::kQueueCount);
  auto meta = tcb_pool::getMetadataPointer(metadata);
  tcb_pool::PoolMetadata pool{meta};
  const uint32_t slotCount = pool.slotCount();
  const uint32_t slotBytes = pool.slotBytes();
  const uint32_t slots = pool.slotsPerBuffer(queueId);
  const uint32_t begin = meta[tcb_pool::kSearchBeginOffset + queueId];
  const uint32_t end = meta[tcb_pool::kSearchEndOffset + queueId];
  uint32_t startSlot = findFreeSlots(pool, begin, end, slots, slotCount);
  const uint32_t flagCount = pool.flagCount();
  while (startSlot == slotCount || pool.freeFlagMask() == 0) {
    popReleaseEntry(pool, slotCount, flagCount);
    startSlot = findFreeSlots(pool, begin, end, slots, slotCount);
  }
  uint32_t id = allocateFlagId(pool);
  for (uint32_t i = 0; i < slots; ++i)
    meta[tcb_pool::kSlotOwnersOffset + startSlot + i] = id;
  const uint32_t indexOffset = pool.allocIndexOffset(queueId);
  const uint32_t index = meta[indexOffset];
  const uint32_t addressOffset =
      pool.addressEntryOffset(queueId, index, slotCount);
  meta[addressOffset] = startSlot;
  meta[addressOffset + 1] = id;
  meta[indexOffset] = pool.nextAddressIndex(index);
  result->address = poolBase + uint64_t(startSlot) * slotBytes;
  result->id = id;
}

__aicore__ __attribute__((always_inline)) void
_mlir_ciface_tcb_pool_release(tcb_pool::MetadataMemRef *metadata) {
  auto meta = tcb_pool::getMetadataPointer(metadata);
  tcb_pool::PoolMetadata pool{meta};
  const uint32_t slotCount = pool.slotCount();
  const uint32_t flagMask = meta[tcb_pool::kFlagMaskOffset];
  const uint32_t flagCount = __builtin_popcount(flagMask);
  // An empty feedback ring may still have unpublished allocations. Wait until
  // every allocated ID has been returned, allowing AIV0 to publish meanwhile.
  while (pool.freeFlagMask() != flagMask)
    popReleaseEntry(pool, slotCount, flagCount);
}
}
