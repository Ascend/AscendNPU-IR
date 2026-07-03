//===- SharedMemConflictModel.cpp - Ascend SRAM bank-conflict analysis ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Models bank-conflict cost for the Ascend SRAM geometry.  The header
// SharedMemConflictModel.h documents the API; this file is the
// implementation.
//
// Bank computation
// ----------------
// Bank id = (byteAddr / bankWidthBytes) mod numBanks (Ascend: A/8 mod 16).
// Same-bank distinct-word accesses serialise; same-word accesses multicast
// (covers sub-bank-width hits like four 1-byte loads of one 8-byte slot).
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Triton/Transforms/SharedMemConflictModel.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <algorithm>

namespace bishengir {
namespace triton {

ConflictReport
analyzeWarpAccessCycle(llvm::ArrayRef<ThreadAccess> accesses,
                       const AscendSmemGeometry &geom) {
  // Per bank: distinct word IDs (multicast collapses same-word hits).
  llvm::DenseMap<unsigned, llvm::DenseSet<uint64_t>> bankToWordIds;
  uint64_t totalBytesRequested = 0;

  for (const auto &acc : accesses) {
    if (acc.accessBytes == 0)
      continue;
    totalBytesRequested += acc.accessBytes;
    // Account for every word touched (an access can straddle bank words).
    uint64_t firstWord = acc.byteOffset / geom.bankWidthBytes;
    uint64_t lastByte = acc.byteOffset + acc.accessBytes - 1;
    uint64_t lastWord = lastByte / geom.bankWidthBytes;
    for (uint64_t w = firstWord; w <= lastWord; ++w) {
      unsigned bank = static_cast<unsigned>(w % geom.numBanks);
      bankToWordIds[bank].insert(w);
    }
  }

  unsigned maxConflict = 0;
  unsigned distinctBanks = 0;
  for (auto &kv : bankToWordIds) {
    distinctBanks++;
    maxConflict = std::max(maxConflict, static_cast<unsigned>(kv.second.size()));
  }
  if (maxConflict == 0)
    maxConflict = 1; // empty cycle

  ConflictReport rep;
  rep.conflictFactor = maxConflict;
  rep.distinctBanksHit = distinctBanks;
  // Peak BW utilisation = idealCycles(totalBytes/peak) / maxConflict.
  if (maxConflict > 0 && geom.peakBandwidthBytesPerCycle() > 0) {
    double idealCycles =
        static_cast<double>(totalBytesRequested) / geom.peakBandwidthBytesPerCycle();
    if (idealCycles < 1.0)
      idealCycles = 1.0;
    rep.peakBandwidthUtilization = idealCycles / static_cast<double>(maxConflict);
    if (rep.peakBandwidthUtilization > 1.0)
      rep.peakBandwidthUtilization = 1.0;
  }
  return rep;
}

ConflictReport analyzeStridedAccess(uint64_t baseByteOffset,
                                     unsigned strideBytes, unsigned accessBytes,
                                     unsigned numThreads,
                                     const AscendSmemGeometry &geom) {
  llvm::SmallVector<ThreadAccess, 32> accesses;
  accesses.reserve(numThreads);
  for (unsigned i = 0; i < numThreads; ++i) {
    ThreadAccess acc;
    acc.byteOffset = baseByteOffset + static_cast<uint64_t>(i) * strideBytes;
    acc.accessBytes = accessBytes;
    accesses.push_back(acc);
  }
  return analyzeWarpAccessCycle(accesses, geom);
}

StagingDecision decideStaging(const StagingDecisionInputs &inputs,
                               const AscendSmemGeometry &geom) {
  StagingDecision dec{};
  dec.directCostCycles =
      static_cast<double>(inputs.spillElementsIfNoStaging) * inputs.cyclesPerSpillElement;

  // Staged path: fixed overhead + conflict-scaled bandwidth charge.
  double bytesPerCycle = static_cast<double>(geom.peakBandwidthBytesPerCycle());
  if (bytesPerCycle <= 0)
    bytesPerCycle = 1.0;
  double bandwidthCycles =
      (static_cast<double>(inputs.roundTripBytes) / bytesPerCycle) * inputs.conflictFactor;
  dec.stagedCostCycles =
      static_cast<double>(inputs.stagingFixedOverheadCycles) + bandwidthCycles;

  dec.stageThroughSmem = dec.stagedCostCycles < dec.directCostCycles;
  return dec;
}

} // namespace triton
} // namespace bishengir
