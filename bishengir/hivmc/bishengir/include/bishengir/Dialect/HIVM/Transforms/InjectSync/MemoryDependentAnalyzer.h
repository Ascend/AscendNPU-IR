//===------ MemoryDependentAnalyzer.cpp ----Sync dependency analysis ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_MEMORYDEPENDENTANALYZER_H
#define BISHENGIR_MEMORYDEPENDENTANALYZER_H

#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncCommon.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"

namespace mlir {
namespace hivm {

class MemoryDependentAnalyzer {
public:
  MemoryDependentAnalyzer() = default;

  /// Analyzing the dependencies of Read and Write, Write and Read, Write and
  /// Write.
  bool DepBetween(const SmallVector<const BaseMemInfo *> &a,
                  const SmallVector<const BaseMemInfo *> &b,
                  DepBaseMemInfoPairVec &depBaseMemInfosVec);

  /// Based on allocate size and base address, determine buffer over lap.
  bool isBufferOverlap(const BaseMemInfo *a, const BaseMemInfo *b, int aIndex,
                       int bIndex);

private:
  /// Analysis of dependency conflicts between BaseMemInfo.
  bool MemAlias(const BaseMemInfo *a, const BaseMemInfo *b);

  /// Determine if GM buffer is overlapping.
  bool isGMBufferOverlap(const BaseMemInfo *a, const BaseMemInfo *b);

  /// Determine if buffer is overlap on base address range.
  bool isBufferAddressRangeOverlap(const BaseMemInfo *a, const BaseMemInfo *b);
};

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_MEMORYDEPENDENTANALYZER_H
