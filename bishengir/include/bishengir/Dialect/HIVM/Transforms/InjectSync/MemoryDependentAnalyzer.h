//===------ MemoryDependentAnalyzer.cpp ----Sync dependency analysis ------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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
  bool isBufferOverlap(const BaseMemInfo *a, const BaseMemInfo *b,
                       uint32_t aIndex, uint32_t bIndex);

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
