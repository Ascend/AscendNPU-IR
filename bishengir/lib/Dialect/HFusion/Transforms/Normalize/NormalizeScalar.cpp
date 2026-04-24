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

#include "bishengir/Dialect/HFusion/Transforms/NormalizePatterns.h"
#include "bishengir/Dialect/HFusion/Transforms/NormalizeTraitsBase.h"
#include "bishengir/Transforms/Normalize/NormalizeScalarTemplate.h"

namespace mlir::hfusion {

namespace {

/// Normalize scalar-like HFusion ops by scalarizing single-element dense tensor
/// constants on every input/init slot and by rebuilding scalar-like broadcasts
/// through `createFillOp`.
struct HFusionNormalizeScalarTraits : public NormalizeTraitsBase {
  // HFusion scalar normalize does not have vector-only operand slots, so every
  // input can be scalarized when it is a single-element dense tensor constant.
  static bool shouldKeepInputShaped(Operation * /*op*/, unsigned /*inputIdx*/) {
    return false;
  }

  // HFusion scalar normalize also scalarizes single-element init tensors.
  static bool shouldKeepInitShaped() {
    return false;
  }

  // HFusion only accepts rank-0 / rank-1 single-element dense constants on the
  // broadcast-input path; multi-rank all-ones tensors stay untouched here.
  static bool allowMultiRankUnitDenseBroadcastInput() { return false; }
};

template <typename... Ops>
void addScalarLikeTensorPatterns(RewritePatternSet &patterns) {
  addNormalizeScalarLikeTensorOpPatterns<HFusionNormalizeScalarTraits, Ops...>(
      patterns);
}

template <typename... Ops>
void addScalarLikeBroadcastPatterns(RewritePatternSet &patterns) {
  addNormalizeScalarLikeTensorBrcOpPatterns<
      HFusionNormalizeScalarTraits, Ops...>(patterns);
}

template <typename... Ops>
void addNonDenseScalarLikeBroadcastPatterns(RewritePatternSet &patterns) {
  addNormalizeScalarLikeTensorBrcOpNonDensePatterns<
      HFusionNormalizeScalarTraits, Ops...>(patterns);
}

} // namespace

/// normalize i8/i32 CompareOp
///   i8 -> f16
///   i32 -> i64 (except vne and veq)
/// e.g.
///   hfusion.compare ins(%src1, %src2 : tensor<6x6xi32>, tensor<6x6xi32>)
/// is normalized to
///   %cast1 = hfusion.cast %src1 : tensor<6x6xi32> to tensor<6x6xi64>
///   %cast2 = hfusion.cast %src2 : tensor<6x6xi32> to tensor<6x6xi64>
///   hfusion.compare ins(%cast1, %cast2 : tensor<6x6xi64>, tensor<6x6xi64>)

void populateNormalizeScalarLikeHFusionPatterns(RewritePatternSet &patterns) {
  addScalarLikeTensorPatterns<hfusion::ElemwiseUnaryOp,
                              hfusion::ElemwiseBinaryOp, hfusion::CompareOp,
                              hfusion::SelectOp, hfusion::CastOp,
                              linalg::ElemwiseUnaryOp,
                              linalg::ElemwiseBinaryOp>(patterns);
  addScalarLikeBroadcastPatterns<linalg::BroadcastOp>(patterns);
}

void populateNormalizeNonDenseScalarLikeBroadcastPatterns(
    RewritePatternSet &patterns) {
  if (archIsRegbased)
    addNonDenseScalarLikeBroadcastPatterns<linalg::BroadcastOp>(patterns);
}
} // namespace mlir::hfusion
