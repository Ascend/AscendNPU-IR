//===-------- NormalizeArithmeticTraits.cpp --------------------*- C++ -*-===//
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

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/NormalizePatterns.h"
#include "bishengir/Dialect/HIVM/Transforms/NormalizeTraitsBase.h"
#include "bishengir/Transforms/Normalize/NormalizeArithmeticTemplate.h"

namespace mlir {
/// Normalizes `rsqrt(x)` to `rec(sqrt(x))`
struct HIVMNormalizeRSqrtTraits
    : public hivm::NormalizeTraitsBase {
public:
  static bool shouldNormalizeRSqrt(hivm::VRsqrtOp op) {
    return op.hasPureTensorSemantics() && op.getBroadcast().empty() && op.getTranspose().empty();
  }
};

/// Normalizes `mul(rec_like(x), y)` to `div(y, x)`
/// (1/b) * a -> a/b
/// a * (1/b) -> a/b
struct HIVMNormalizeMulRecTraits : public hivm::NormalizeTraitsBase {
  using RecOpType = hivm::VRecOp;
  using DivOpType = hivm::VDivOp;

  static bool shouldNormalizeMulRec(hivm::VMulOp op) {
    return op.hasPureTensorSemantics();
  }
};

/// Normalizes `div(1, x)` to `rec(x)`.
struct HIVMNormalizeDivVSToRecTraits
    : public hivm::NormalizeTraitsBase {
public:
  static bool shouldNormalizeDiv(hivm::VDivOp op) {
    return op.hasPureTensorSemantics();
  }
};

using NormalizeRSqrtOp =
    NormalizeRSqrtOpTemplate<hivm::VRsqrtOp, HIVMNormalizeRSqrtTraits>;
using NormalizeMulRecOp =
    NormalizeMulRecOpTemplate<hivm::VMulOp, HIVMNormalizeMulRecTraits>;
using NormalizeDivVSToRec =
    NormalizeDivVSToRecTemplate<hivm::VDivOp, HIVMNormalizeDivVSToRecTraits>;

} // namespace mlir

void mlir::hivm::populateNormalizeArithmeticPatterns(
    RewritePatternSet &patterns) {
  patterns.add<NormalizeMulRecOp>(patterns.getContext());
  patterns.add<NormalizeDivVSToRec>(patterns.getContext());
  patterns.add<NormalizeRSqrtOp>(patterns.getContext());
}
