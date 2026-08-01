//===-------------------- PropagateConvertLayoutVBrc.cpp ------------------===//
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

#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/ConvertLayoutUtils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#define DEBUG_TYPE "hivm-propagate-convert-layout"

using namespace mlir;
using namespace mlir::hivm;

namespace {

/// Pattern: Push convert_layout UP through scalar hivm.hir.vbrc
/// Before:
///   %empty = tensor.empty() : tensor<128x128xbf16>
///   %brc = hivm.hir.vbrc ins(%scalar : bf16) outs(%empty) ->
///   tensor<128x128xbf16> %fractal = hivm.hir.convert_layout %brc {up}  // ND
///   -> Fractal
/// After:
///   %empty_fractal = tensor.empty() : tensor<8x8x16x16xbf16>
///   %brc_fractal = hivm.hir.vbrc ins(%scalar) outs(%empty_fractal)
///       -> tensor<8x8x16x16xbf16>
struct PropagateConvertLayoutUpThroughScalarVBrc
    : public OpRewritePattern<ConvertLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvertLayoutOp convertOp,
                                PatternRewriter &rewriter) const override {
    if (!isPropagatingUp(convertOp))
      return failure();

    auto vbrcOp = convertOp.getSource().getDefiningOp<VBrcOp>();
    if (!vbrcOp)
      return failure();

    if (!isScalarLike(vbrcOp.getSrc().getType()))
      return rewriter.notifyMatchFailure(convertOp,
                                         "vbrc source is not scalar-like");

    if (!vbrcOp.getDst().getDefiningOp<tensor::EmptyOp>())
      return rewriter.notifyMatchFailure(
          convertOp, "vbrc destination is not defined by tensor.empty");

    Location loc = vbrcOp.getLoc();
    auto fractalType = cast<RankedTensorType>(convertOp.getType());

    rewriter.setInsertionPoint(vbrcOp);
    Value newDst = rewriter.create<tensor::EmptyOp>(
        loc, fractalType.getShape(), fractalType.getElementType());

    auto newVbrc = cast<VBrcOp>(rewriter.clone(*vbrcOp));
    rewriter.modifyOpInPlace(newVbrc, [&]() {
      newVbrc.getDstMutable().assign(newDst);
      newVbrc.getResult()[0].setType(fractalType);
    });

    rewriter.replaceOp(convertOp, newVbrc.getResult()[0]);
    return success();
  }
};

} // namespace

void mlir::hivm::populateConvertLayoutVBrc(RewritePatternSet &patterns,
                                           MLIRContext *context) {
  patterns.add<PropagateConvertLayoutUpThroughScalarVBrc>(context);
}
