//===- DotTilingCostModel.h - Ascend SRAM bank-conflict analysis -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Models bank-conflict cost for the Ascend SRAM geometry (16 banks, 8 bytes
// each, 32 threads accessing concurrently).  Used by transforms that decide
// whether routing a tensor through SRAM is faster than holding it in
// registers — the canonical "Idea E" of the register-pressure plan.
//
// Decoupling: the model takes a per-thread access description as input and
// outputs a conflict factor.  Callers that own a Triton layout (linear,
// blocked, slice, dot_op) lower the layout into a per-thread byte-offset
// sequence first; this analysis stays layout-agnostic so it can serve all
// callers without forcing a particular dialect dependency.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_TRITON_TRANSFORMS_DOTTILINGCOSTMODEL_H
#define BISHENGIR_DIALECT_TRITON_TRANSFORMS_DOTTILINGCOSTMODEL_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>

namespace bishengir {
namespace triton {

/// Ascend SHM geometry; defaults target dav-c310. Override per-instance
/// for other parts (e.g. dav-c100 has a different bank count).
struct AscendMemGeometry {
  unsigned numBanks = 16;
  unsigned bankWidthBytes = 8;
  unsigned numThreadsPerWarp = 32;

  /// Total bytes deliverable per cycle if all banks hit distinct rows.
  unsigned peakBandwidthBytesPerCycleSmem() const {
    return numBanks * bankWidthBytes;
  }
  unsigned peakBandwidthBytesPerCycleGmem() const {
    return 150; // estimated experimantally: can be refined
  }
};

/// One thread's contribution to a single SRAM access cycle.
/// `accessBytes == 0` marks an inactive lane.
struct ThreadAccess {
  uint64_t byteOffset;
  unsigned accessBytes;
};

/// Result of conflict analysis for one access cycle.
/// conflictFactor = max threads sharing any bank (1 = ideal).
/// peakBandwidthUtilization = idealCycles / actualCycles, useful for
/// comparing layouts with different access widths.
struct ConflictReport {
  unsigned conflictFactor = 1;
  double peakBandwidthUtilization = 1.0;
  unsigned distinctBanksHit = 0;
};

/// Compute the conflict report for one cycle of a warp's access pattern.
/// Pre: `accesses.size()` == warp thread count.
ConflictReport
analyzeWarpAccessCycle(llvm::ArrayRef<ThreadAccess> accesses,
                       const AscendMemGeometry &geom = {});

/// Strided-access shortcut: thread i reads `accessBytes` from
/// `baseByteOffset + i * strideBytes`.
ConflictReport analyzeStridedAccess(uint64_t baseByteOffset,
                                    unsigned strideBytes, unsigned accessBytes,
                                    unsigned numThreads = 32,
                                    const AscendMemGeometry &geom = {});

/// Cost comparison for "stage through SHM" vs "stay in registers".
///   direct = spillElements * cyclesPerSpillElement
///   staged = stagingFixedOverhead +
///            conflictFactor * roundTripBytes / peakBandwidth
/// Caller picks the lower-cost option per dot.
struct StagingDecisionInputs {
  unsigned spillElementsIfNoStaging = 0; // 0 if no register spill
  uint64_t roundTripBytesA = 0;
  uint64_t roundTripBytesB = 0;
  bool smemStageA = true; // true if A is staged through SHM
  bool smemStageB = true; // true if B is staged through SHM
  unsigned conflictFactor = 1;
  unsigned stagingFixedOverheadCyclesSmem = 16;
  unsigned stagingFixedOverheadCyclesGmem = 21000; // estimated experimentally: can be refined
  unsigned cyclesPerSpillElement = 200; // Ascend GM round-trip is hundreds/elt
};

struct StagingDecision {
  bool stageThroughMem;
  double directCostCycles;
  double stagedCostCycles;
};

StagingDecision decideStaging(const StagingDecisionInputs &inputs,
                              const AscendMemGeometry &geom = {});

} // namespace triton
} // namespace bishengir

#endif // BISHENGIR_DIALECT_TRITON_TRANSFORMS_DOTTILINGCOSTMODEL_H
