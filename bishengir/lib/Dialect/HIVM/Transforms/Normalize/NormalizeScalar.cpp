//===- NormalizeScalar.cpp ---------------------------------------*- C++ -*-===//
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
#include "bishengir/Transforms/Normalize/NormalizeScalarTemplate.h"

namespace mlir::hivm {

namespace {

/// Normalize scalar-like HIVM structured ops by folding single-element dense
/// tensor constants on scalar-legal operand slots while keeping vector-only
/// operands and all init/output tensors shaped.
/// e.g.
///   hivm.hir.vadd ins(%cst, %src : tensor<1xf16>, tensor<16xf16>)
///                  outs(%dst : tensor<16xf16>)
/// is normalized to
///   %scalar = arith.constant 1.0 : f16
///   hivm.hir.vadd ins(%scalar, %src : f16, tensor<16xf16>)
///                  outs(%dst : tensor<16xf16>)
struct HIVMNormalizeScalarOpTraits {
  // HIVM scalar normalize keeps vector-only operand slots shaped and only
  // scalarizes operand positions that are verifier-legal as scalars.
  static bool shouldKeepInputShaped(Operation *op, unsigned inputIdx) {
    auto hivmOp = dyn_cast_or_null<HIVMStructuredOp>(op);
    return hivmOp && hivmOp.isVectorOnlyOperand(inputIdx);
  }

  // HIVM structured ops still need shaped init/output tensors after normalize.
  static bool shouldKeepInitShaped() {
    return true;
  }

  // Plain scalar-like HIVM structured ops are not broadcast ops, so this hook
  // stays disabled on this trait.
  static bool allowMultiRankUnitDenseBroadcastInput() { return false; }
};

/// Normalize scalar-like `hivm.hir.vbrc` by rebuilding broadcasts whose source
/// is a unit dense tensor into scalar-source `hivm.hir.vbrc`.
/// e.g.
///   hivm.hir.vbrc ins(%cst : tensor<1x1xf32>) outs(%dst : tensor<8x16xf32>)
/// is normalized to
///   %scalar = arith.constant 1.0 : f32
///   hivm.hir.vbrc ins(%scalar : f32) outs(%dst : tensor<8x16xf32>)
struct HIVMNormalizeScalarBroadcastTraits : public NormalizeTraitsBase {
  // Broadcast normalize can scalarize the source operand when it is legal to
  // rebuild the op as scalar-source `hivm.hir.vbrc`.
  static bool shouldKeepInputShaped(Operation * /*op*/, unsigned /*inputIdx*/) {
    return false;
  }

  // The destination/init is still the shaped broadcast output and must stay
  // tensor-typed.
  static bool shouldKeepInitShaped() {
    return false;
  }

  // After `linalg.broadcast` is converted to `hivm.hir.vbrc`, the converter
  // may first `expand_shape` the source to match dst rank. This means a
  // scalar-like broadcast source often reaches HIVM normalize as a
  // multi-rank unit tensor such as tensor<1x1xf32>, not just tensor<f32> or
  // tensor<1xf32>. Allowing unit-dense tensors here keeps
  // `--convert-hfusion-to-hivm --hivm-normalize-ops` closer to
  // `--hfusion-normalize-ops --convert-hfusion-to-hivm`.
  static bool allowMultiRankUnitDenseBroadcastInput() { return true; }
};

template <typename... Ops>
void addScalarLikeTensorPatterns(RewritePatternSet &patterns) {
  addNormalizeScalarLikeTensorOpPatterns<HIVMNormalizeScalarOpTraits, Ops...>(
      patterns);
}

template <typename... Ops>
void addScalarLikeBroadcastPatterns(RewritePatternSet &patterns) {
  addNormalizeScalarLikeTensorBrcOpPatterns<
      HIVMNormalizeScalarBroadcastTraits, Ops...>(patterns);
}

template <typename... Ops>
void addNonDenseScalarLikeBroadcastPatterns(RewritePatternSet &patterns) {
  addNormalizeScalarLikeTensorBrcOpNonDensePatterns<
      HIVMNormalizeScalarBroadcastTraits, Ops...>(patterns);
}

} // namespace

void populateNormalizeScalarLikeHIVMPatterns(RewritePatternSet &patterns) {
  addScalarLikeTensorPatterns<hivm::VAddOp, hivm::VMulOp, hivm::VMaxOp,
                              hivm::VMinOp, hivm::VShROp, hivm::VCmpOp,
                              hivm::VSelOp>(patterns);
  addScalarLikeBroadcastPatterns<hivm::VBrcOp>(patterns);
}

void populateNormalizeNonDenseScalarLikeBroadcastPatterns(
    RewritePatternSet &patterns, bool isRegbased) {
  if (isRegbased)
    addNonDenseScalarLikeBroadcastPatterns<hivm::VBrcOp>(patterns);
}

} // namespace mlir::hivm
