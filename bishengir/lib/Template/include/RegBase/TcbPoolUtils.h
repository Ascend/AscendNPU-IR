//===- TcbPoolUtils.h - Scalar management of a shared TCB pool ---------*- C++
//-*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_LIB_TEMPLATE_INCLUDE_REGBASE_TCBPOOLUTILS_H
#define BISHENGIR_LIB_TEMPLATE_INCLUDE_REGBASE_TCBPOOLUTILS_H

#include "Utils.h"
#include <cstdint>

#if ENABLE_CPU_TRACE_INTRINSIC || defined(__CCE_KT_TEST__)
#include <cassert>
#define TCB_POOL_ASSERT(condition) assert(condition)
#else
#define TCB_POOL_ASSERT(condition) ((void)0)
#endif

namespace tcb_pool {
constexpr uint32_t kMaxSlotCount = 32;
constexpr uint64_t kAllocFailed = UINT64_MAX;

// Word zero is the occupied-slot bitmap. Each queue owns a separate range of
// start-slot words selected by entryOffset and depth. All queues sharing the
// bitmap must use the same pool base and slot size. The layout is supplied
// by the IR planner, not a fixed queue count or depth in the template.
#if ENABLE_CPU_TRACE_INTRINSIC
using MetadataPtr = volatile uint32_t *;
using MetadataMemRef = memref_t<uint32_t, 1>;
#else
using MetadataPtr = volatile __ssbuf__ uint32_t *;
using MetadataMemRef = memref_t<__ssbuf__ uint32_t, 1>;
#endif

__aicore__ inline MetadataPtr metadataPointer(MetadataMemRef *metadata) {
  TCB_POOL_ASSERT(metadata->sizes[0] >= 1 && metadata->strides[0] == 1);
  return metadata->aligned + metadata->offset;
}

__aicore__ inline uint32_t slotMask(uint32_t start, uint32_t slots) {
  TCB_POOL_ASSERT(slots > 0 && slots <= kMaxSlotCount);
  TCB_POOL_ASSERT(start <= kMaxSlotCount - slots);
  return slots == 32 ? UINT32_MAX : ((1u << slots) - 1u) << start;
}

__aicore__ inline uint32_t entryIndex(uint32_t entryOffset, uint32_t depth,
                                      uint32_t seq) {
  TCB_POOL_ASSERT(entryOffset > 0 && depth > 0);
  return entryOffset + seq % depth;
}

// Cube is the sole writer. Allocation tries once in first-fit order. Failure
// changes nothing; queue capacity and hardware completion are caller checks.
__aicore__ inline uint64_t tryAllocate(MetadataPtr meta, uint64_t poolBase,
                                       uint32_t slotBytes, uint32_t entryOffset,
                                       uint32_t depth, uint32_t seq,
                                       uint32_t slots, uint32_t slotCount) {
  TCB_POOL_ASSERT(slotCount > 0 && slotCount <= kMaxSlotCount);
  TCB_POOL_ASSERT(slotBytes > 0 && slots > 0 && slots <= slotCount);
  TCB_POOL_ASSERT(poolBase <= kAllocFailed - uint64_t(slotCount) * slotBytes);
  uint32_t used = meta[0];
  for (uint32_t start = 0; start + slots <= slotCount; ++start) {
    uint32_t mask = slotMask(start, slots);
    if (used & mask)
      continue;
    meta[0] = used | mask;
    meta[entryIndex(entryOffset, depth, seq)] = start;
    return poolBase + uint64_t(start) * slotBytes;
  }
  return kAllocFailed;
}
} // namespace tcb_pool

// The caller owns cumulative per-queue counters and event rings. It checks
// allocated-released < depth before alloc; publishes start before AIV get;
// waits for FIX before the AIV data read; releases only after BOTH AIVs finish.
// Counters must not wrap. Volatile accesses do not provide synchronization.
// All four operations are single attempts without spinning or pipe sync.
// Address zero is valid; UINT64_MAX denotes allocation failure and must never
// be passed to FIX. init clears only the bitmap; release only clears slot bits.
// init runs after all previous users complete. Unpublished entries need no
// initialization: alloc writes each start before any permitted get/release.
extern "C" {
__aicore__ void _mlir_ciface_tcb_pool_init(tcb_pool::MetadataMemRef *metadata);
__aicore__ uint64_t _mlir_ciface_tcb_pool_alloc(
    tcb_pool::MetadataMemRef *metadata, uint64_t poolBase, uint32_t slotBytes,
    uint32_t entryOffset, uint32_t depth, uint32_t seq, uint32_t slots,
    uint32_t slotCount);
__aiv__ uint64_t _mlir_ciface_tcb_pool_get_address(
    tcb_pool::MetadataMemRef *metadata, uint64_t poolBase, uint32_t slotBytes,
    uint32_t entryOffset, uint32_t depth, uint32_t seq);
__aicore__ void
_mlir_ciface_tcb_pool_release(tcb_pool::MetadataMemRef *metadata,
                              uint32_t entryOffset, uint32_t depth,
                              uint32_t seq, uint32_t slots);
}

#endif
