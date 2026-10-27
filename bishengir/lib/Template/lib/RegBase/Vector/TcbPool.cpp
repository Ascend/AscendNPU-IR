//===- TcbPool.cpp - Vector TCB pool lookup
//---------------------------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "RegBase/TcbPoolUtils.h"

extern "C" {
__aiv__ __attribute__((always_inline)) uint64_t
_mlir_ciface_tcb_pool_get_address(tcb_pool::MetadataMemRef *metadata,
                                  uint64_t poolBase, uint32_t slotBytes,
                                  uint32_t entryOffset, uint32_t depth,
                                  uint32_t seq) {
  TCB_POOL_ASSERT(uint64_t(entryOffset) + depth <=
                  uint64_t(metadata->sizes[0]));
  uint32_t start = tcb_pool::metadataPointer(
      metadata)[tcb_pool::entryIndex(entryOffset, depth, seq)];
  TCB_POOL_ASSERT(start < tcb_pool::kMaxSlotCount && slotBytes > 0);
  TCB_POOL_ASSERT(poolBase <
                  tcb_pool::kAllocFailed - uint64_t(start) * slotBytes);
  return poolBase + uint64_t(start) * slotBytes;
}
}
