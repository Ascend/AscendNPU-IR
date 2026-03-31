//===- NormalizeComparison.cpp ----------------------------------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#include "bishengir/Transforms/Normalize/NormalizeComparison.h"

#include "mlir/IR/PatternMatch.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/NormalizeTraitsBase.h"

namespace mlir::hivm {

//===----------------------------------------------------------------------===//
// HIVMCmpVneTraits - Traits for NormalizeCmpVne pattern
//===----------------------------------------------------------------------===//
struct HIVMCmpVneTraits : public NormalizeTraitsBase {
  static bool shouldNormalize(VCmpOp op) {
    if (!op.getTranspose().empty() || !op.getBroadcast().empty()) {
      op->emitWarning("NormalizeCmpVneOp: skipping op with non-empty broadcast/transpose attributes");
      return false;
    }
    return op.getCompareMode() == CompareMode::NE;
  }
};

using NormalizeCmpVneOp = mlir::NormalizeCmpVneOpTemplate<VCmpOp, HIVMCmpVneTraits>;

void populateNormalizeCmpVnePatterns(RewritePatternSet &patterns) {
  patterns.add<NormalizeCmpVneOp>(patterns.getContext());
}

} // namespace mlir::hivm