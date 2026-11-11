//===- TcbPool.cpp - Vector TCB pool lookup -------------------------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "RegBase/TcbPoolUtils.h"

extern "C" {
__aiv__ __attribute__((always_inline)) void
_mlir_ciface_tcb_pool_get_address(tcb_pool::Allocation *result,
                                  tcb_pool::MetadataMemRef *metadata,
                                  uint64_t poolBase, uint32_t queueId) {
  uint32_t subBlockId = get_subblockid();
  TCB_POOL_ASSERT(queueId < tcb_pool::kQueueCount &&
                  subBlockId < tcb_pool::kVectorCoreCount);
  auto meta = tcb_pool::getMetadataPointer(metadata);
  tcb_pool::PoolMetadata pool{meta};
  const uint32_t slotCount = pool.slotCount();
  const uint32_t slotBytes = pool.slotBytes();
  const uint32_t indexOffset = pool.getIndexOffset(subBlockId, queueId);
  const uint32_t index = meta[indexOffset];
  const uint32_t addressOffset =
      pool.addressEntryOffset(queueId, index, slotCount);
  uint32_t startSlot = meta[addressOffset];
  uint32_t id = meta[addressOffset + 1];
  meta[indexOffset] = pool.nextAddressIndex(index);
  result->address = poolBase + uint64_t(startSlot) * slotBytes;
  result->id = id;
  if (subBlockId == 0) {
    uint32_t tail = pool.releaseTail();
    // Capacity equals the number of reserved IDs. Every unpopped record owns at
    // least one slot, so a get of another live allocation always has room for
    // its feedback.
    TCB_POOL_ASSERT(tail - pool.releaseHead() < pool.flagCount());
    const uint32_t flagCount = pool.flagCount();
    uint32_t releaseOffset =
        pool.releaseEntryOffset(tail, slotCount, flagCount);
    meta[releaseOffset] = queueId;
    meta[releaseOffset + 1] = startSlot;
    meta[releaseOffset + 2] = id;
    // Publish the complete record last. The caller protects the address-ring
    // entry until both AIVs have read it; this feedback does not replace that
    // descriptor lifetime guarantee.
    meta[tcb_pool::kReleaseTailOffset] = tail + 1;
  }
}
}
