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

/// normalize vlogb(x) to vln(x) / vln(b) when log base b is not e
/// eg.
/// y = hivm.hir.vlog2 x
///  is normalized to
///  y = hivm.hir.vln x / hivm.hir.vln 2
template <typename OpType, int Base>
struct HIVMNormalizeLogLikeTraits : public NormalizeTraitsBase {
  static bool shouldNormalizeLogLike(OpType op) {
    return shouldNormalizeNonBroadcastUnaryOp(op);
  }

  static float getLogBase(OpType) { return static_cast<float>(Base); }

  static Value castBackLogLikeF16Result(PatternRewriter &rewriter, Location loc,
                                        Value result, Value dst) {
    return createCastOp(rewriter, loc, result,
                        getElementTypeOrSelf(dst.getType()),
                        CastRoundKind::Round);
  }
};

/// normalize vlog1p(x) to vln(x + 1)
/// eg.
/// y = hivm.hir.vlog1p x
///  is normalized to
///  y = hivm.hir.vln (x + 1)
struct HIVMNormalizeLog1pTraits : public NormalizeTraitsBase {
  static bool shouldNormalizeLog1p(hivm::VLog1pOp op) {
    return shouldNormalizeNonBroadcastUnaryOp(op);
  }
};

using NormalizeExp2Op =
    mlir::NormalizeExp2OpTemplate<VExp2Op, HIVMNormalizeExp2Traits>;
using NormalizeErfOp =
    mlir::NormalizeErfOpTemplate<VErfOp, HIVMNormalizeErfTraits>;
using NormalizeVLog2Op =
    mlir::NormalizeLogLikeOpTemplate<
        hivm::VLog2Op, HIVMNormalizeLogLikeTraits<hivm::VLog2Op, 2>>;
using NormalizeVLog10Op =
    mlir::NormalizeLogLikeOpTemplate<
        hivm::VLog10Op, HIVMNormalizeLogLikeTraits<hivm::VLog10Op, 10>>;
using NormalizeVLog1pOp =
    mlir::NormalizeLog1pOpTemplate<hivm::VLog1pOp, HIVMNormalizeLog1pTraits>;

} // namespace

void populateNormalizePrimaryMathPatterns(RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<NormalizeVLog2Op>(ctx);
  patterns.add<NormalizeVLog10Op>(ctx);
  patterns.add<NormalizeVLog1pOp>(ctx);
  patterns.add<NormalizeExp2Op>(ctx);
  patterns.add<NormalizeErfOp>(ctx);
}

} // namespace mlir::hivm
