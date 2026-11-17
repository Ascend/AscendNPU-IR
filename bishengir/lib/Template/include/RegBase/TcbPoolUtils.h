//===- TcbPoolUtils.h - Shared TCB pool metadata ------------------*- C++
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
constexpr uint32_t kFree = UINT32_MAX;
constexpr uint32_t kQueueCount = 2;
constexpr uint32_t kVectorCoreCount = 2;
// Must cover each segment's maximum C/V issue gap, enforced by the caller.
constexpr uint32_t kAddressQueueDepth = 3;
// All offsets below count i32 words from the metadata base.
constexpr uint32_t kSlotBytesOffset = 0;
constexpr uint32_t kSlotCountOffset = 1;
constexpr uint32_t kSlotsPerBufferOffset = 2;
constexpr uint32_t kSearchBeginOffset = kSlotsPerBufferOffset + kQueueCount;
constexpr uint32_t kSearchEndOffset = kSearchBeginOffset + kQueueCount;
constexpr uint32_t kFlagMaskOffset = kSearchEndOffset + kQueueCount;
// Nine i32 words: slotBytes, slotCount, three two-element arrays, flagMask.
constexpr uint32_t kConfigWordCount = kFlagMaskOffset + 1;
constexpr uint32_t kReleaseHeadOffset = kConfigWordCount;
constexpr uint32_t kReleaseTailOffset = kReleaseHeadOffset + 1;
constexpr uint32_t kFreeFlagMaskOffset = kReleaseTailOffset + 1;
constexpr uint32_t kAllocIndexOffset = kFreeFlagMaskOffset + 1;
constexpr uint32_t kGetIndexOffset = kAllocIndexOffset + kQueueCount;
constexpr uint32_t kSlotOwnersOffset =
    kGetIndexOffset + kVectorCoreCount * kQueueCount;
constexpr uint32_t kAddressEntryWords = 2;
constexpr uint32_t kReleaseEntryWords = 3;

#if ENABLE_CPU_TRACE_INTRINSIC
using MetadataPtr = volatile uint32_t *;
using MetadataMemRef = memref_t<uint32_t, 1>;
#else
using MetadataPtr = volatile __ssbuf__ uint32_t *;
using MetadataMemRef = memref_t<__ssbuf__ uint32_t, 1>;
#endif

// C-interface result storage for the two MLIR results (i64 address, i32 ID).
struct Allocation {
  uint64_t address;
  uint32_t id;
};

__aicore__ __attribute__((always_inline)) inline MetadataPtr
getMetadataPointer(MetadataMemRef *metadata) {
  TCB_POOL_ASSERT(metadata->strides[0] == 1 &&
                  metadata->sizes[0] >= kSlotOwnersOffset);
  return metadata->aligned + metadata->offset;
}

// SSBUF words: config[9], head, tail, freeFlagMask, allocIndex[2],
// getIndex[2][2], owner[S], addressQ[2][3][2], release[F][3].
// F is popcount(flagMask). Each core is the sole writer of its own indices.
// Address entries are {startSlot, flagId}; release entries are
// {queueId, startSlot, flagId}. owner stores a flag ID or kFree.
// Each unpopped allocation owns one distinct ID, so F release entries suffice.
struct PoolMetadata {
  MetadataPtr meta;

  __aicore__ __attribute__((always_inline)) uint32_t slotBytes() const {
    return meta[kSlotBytesOffset];
  }
  __aicore__ __attribute__((always_inline)) uint32_t slotCount() const {
    return meta[kSlotCountOffset];
  }
  __aicore__ __attribute__((always_inline)) uint32_t
  slotsPerBuffer(uint32_t queueId) const {
    TCB_POOL_ASSERT(queueId < kQueueCount);
    return meta[kSlotsPerBufferOffset + queueId];
  }
  __aicore__ __attribute__((always_inline)) uint32_t flagCount() const {
    return __builtin_popcount(meta[kFlagMaskOffset]);
  }
  __aicore__ __attribute__((always_inline)) uint32_t freeFlagMask() const {
    return meta[kFreeFlagMaskOffset];
  }
  __aicore__ __attribute__((always_inline)) void
  setFreeFlagMask(uint32_t mask) const {
    meta[kFreeFlagMaskOffset] = mask;
  }
  __aicore__ __attribute__((always_inline)) static uint32_t
  allocIndexOffset(uint32_t queueId) {
    TCB_POOL_ASSERT(queueId < kQueueCount);
    return kAllocIndexOffset + queueId;
  }
  __aicore__ __attribute__((always_inline)) static uint32_t
  getIndexOffset(uint32_t subBlockId, uint32_t queueId) {
    TCB_POOL_ASSERT(subBlockId < kVectorCoreCount && queueId < kQueueCount);
    return kGetIndexOffset + subBlockId * kQueueCount + queueId;
  }
  __aicore__ __attribute__((always_inline)) static uint32_t
  nextAddressIndex(uint32_t index) {
    TCB_POOL_ASSERT(index < kAddressQueueDepth);
    return index + 1 == kAddressQueueDepth ? 0 : index + 1;
  }
  // Use cached geometry for offsets; entryIndex is a depth-three index.
  __aicore__ __attribute__((always_inline)) static uint32_t
  addressEntryOffset(uint32_t queueId, uint32_t entryIndex,
                     uint32_t slotCount) {
    TCB_POOL_ASSERT(queueId < kQueueCount && entryIndex < kAddressQueueDepth);
    return kSlotOwnersOffset + slotCount +
           kAddressEntryWords * (queueId * kAddressQueueDepth + entryIndex);
  }
  // The release ring follows both address queues; each record is three words.
  __aicore__ __attribute__((always_inline)) static uint32_t
  releaseEntryOffset(uint32_t seq, uint32_t slotCount, uint32_t flagCount) {
    return kSlotOwnersOffset + slotCount +
           kAddressEntryWords * kQueueCount * kAddressQueueDepth +
           kReleaseEntryWords * (seq % flagCount);
  }
  __aicore__ __attribute__((always_inline)) uint32_t releaseHead() const {
    return meta[kReleaseHeadOffset];
  }
  __aicore__ __attribute__((always_inline)) uint32_t releaseTail() const {
    return meta[kReleaseTailOffset];
  }
  // Total storage in i32 words, including all F release records.
  __aicore__ __attribute__((always_inline)) uint32_t metadataWords() const {
    return kSlotOwnersOffset + slotCount() +
           kAddressEntryWords * kQueueCount * kAddressQueueDepth +
           kReleaseEntryWords * flagCount();
  }
};
} // namespace tcb_pool

// Init resets independent address indices for Cube, AIV0 and AIV1.
// The caller protects the depth-three address rings using both AIV issue
// counts. Init must be visible before get;
// Cube publishes progress after alloc/FIX issue, Vector advances after reading
// the descriptor. AIV0 alone publishes feedback. Caller handoffs must prevent
// descriptor overwrite before BOTH AIVs read it. Volatile is not a barrier.
// Feedback records issue, not completion. Reclamation enqueues both AIV
// completion waits on FIX before logically freeing slots/ID. Reclamation may
// Scalar-spin on empty feedback. Stop allocating before release and allow the
// remaining gets to progress; release waits for unpublished feedback too.
// Normal pipe teardown still applies after release. Release counters must not
// wrap in a pool lifetime.
extern "C" {
// Initialize one pool lifetime; the UB base itself is supplied to alloc/get.
__aicore__ void _mlir_ciface_tcb_pool_init(
    tcb_pool::MetadataMemRef *metadata, uint32_t slotBytes, uint32_t slotCount,
    uint32_t slotsPerBuffer0, uint32_t slotsPerBuffer1, uint32_t searchBegin0,
    uint32_t searchBegin1, uint32_t searchEnd0, uint32_t searchEnd1,
    uint32_t flagMask);
// Allocate one buffer, reclaiming the oldest feedback entries as needed.
__aicore__ void _mlir_ciface_tcb_pool_alloc(tcb_pool::Allocation *result,
                                            tcb_pool::MetadataMemRef *metadata,
                                            uint64_t poolBase,
                                            uint32_t queueId);
// Both AIVs read the descriptor; only AIV0 publishes reclaim feedback.
__aiv__ void
_mlir_ciface_tcb_pool_get_address(tcb_pool::Allocation *result,
                                  tcb_pool::MetadataMemRef *metadata,
                                  uint64_t poolBase, uint32_t queueId);
// Stop allocating before release. Waits for remaining feedback publications
// and enqueues both AIV completion waits on FIX for every live allocation.
// Does not wait on Scalar for Vector work to complete.
__aicore__ void
_mlir_ciface_tcb_pool_release(tcb_pool::MetadataMemRef *metadata);
}

#endif
