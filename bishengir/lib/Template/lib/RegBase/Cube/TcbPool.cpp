//===- TcbPool.cpp - Cube TCB pool allocation
//-------------------------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "RegBase/TcbPoolUtils.h"

extern "C" {
__aicore__ __attribute__((always_inline)) void
_mlir_ciface_tcb_pool_init(tcb_pool::MetadataMemRef *metadata) {
  tcb_pool::metadataPointer(metadata)[0] = 0;
}

__aicore__ __attribute__((always_inline)) uint64_t _mlir_ciface_tcb_pool_alloc(
    tcb_pool::MetadataMemRef *metadata, uint64_t poolBase, uint32_t slotBytes,
    uint32_t entryOffset, uint32_t depth, uint32_t seq, uint32_t slots,
    uint32_t slotCount) {
  TCB_POOL_ASSERT(uint64_t(entryOffset) + depth <=
                  uint64_t(metadata->sizes[0]));
  return tcb_pool::tryAllocate(tcb_pool::metadataPointer(metadata), poolBase,
                               slotBytes, entryOffset, depth, seq, slots,
                               slotCount);
}

__aicore__ __attribute__((always_inline)) void
_mlir_ciface_tcb_pool_release(tcb_pool::MetadataMemRef *metadata,
                              uint32_t entryOffset, uint32_t depth,
                              uint32_t seq, uint32_t slots) {
  TCB_POOL_ASSERT(uint64_t(entryOffset) + depth <=
                  uint64_t(metadata->sizes[0]));
  auto meta = tcb_pool::metadataPointer(metadata);
  uint32_t start = meta[tcb_pool::entryIndex(entryOffset, depth, seq)];
  uint32_t mask = tcb_pool::slotMask(start, slots);
  uint32_t used = meta[0];
  TCB_POOL_ASSERT((used & mask) == mask);
  meta[0] = used & ~mask;
}
}
