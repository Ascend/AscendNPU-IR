//===- NormalizeMath.cpp ----------------------------------------*- C++ -*-===//
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
#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Transforms/Normalize/NormalizeMathTemplate.h"

namespace mlir::hivm {
namespace {
template <typename OpTy>
bool shouldNormalizeNonBroadcastUnaryOp(OpTy op) {
  return op.hasPureTensorSemantics() && op.getBroadcast().empty() &&
         op.getTranspose().empty();
}

/// Normalizes `hivm.hir.vexp2` to `vexp(ln(2) * x)`.
struct HIVMNormalizeExp2Traits : public NormalizeTraitsBase {
  static bool shouldNormalizeExp2(VExp2Op op) {
    return shouldNormalizeNonBroadcastUnaryOp(op);
  }
};

/// Normalizes `hivm.hir.verf` with the shared erf polynomial template.
struct HIVMNormalizeErfTraits : public NormalizeTraitsBase {
  static bool shouldNormalizeErf(VErfOp op) {
    return shouldNormalizeNonBroadcastUnaryOp(op);
  }
};

using NormalizeExp2Op = mlir::NormalizeExp2OpTemplate<VExp2Op,
                                                      HIVMNormalizeExp2Traits>;
using NormalizeErfOp =
    mlir::NormalizeErfOpTemplate<VErfOp, HIVMNormalizeErfTraits>;

} // namespace

void populateNormalizePrimaryMathPatterns(RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<NormalizeExp2Op>(ctx);
  patterns.add<NormalizeErfOp>(ctx);
}

} // namespace mlir::hivm
