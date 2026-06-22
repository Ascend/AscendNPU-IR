//===- Reduction.cpp - HFusion to HIVM dialect conversion for reduction ---===//
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

#include "bishengir/Conversion/HFusionToHIVM/HFusionToHIVM.h"
#include "bishengir/Conversion/HFusionToHIVM/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;

static hivm::ReduceOperation
getReduceWithIndexOpAttr(hfusion::ReduceWithIndexOp reduceOp) {
  auto reduceKind = reduceOp.getReduceKindAttr().getReduceWithIndexKind();
  auto tieBreakLeft = reduceOp.getTieBreakLeftAttr().getValue();
  if (reduceKind == hfusion::ReduceWithIndexKind::MAX) {
    return tieBreakLeft ? hivm::ReduceOperation::max_with_index_left
                        : hivm::ReduceOperation::max_with_index_right;
  }
  if (reduceKind == hfusion::ReduceWithIndexKind::MIN) {
    return tieBreakLeft ? hivm::ReduceOperation::min_with_index_left
                        : hivm::ReduceOperation::min_with_index_right;
  }
  reduceOp.emitOpError("unsupported reduce with index operation: ");
  llvm::report_fatal_error("Not implemented");
}

static hivm::ReduceOpAttr getReduceOpAttr(Operation *op) {
  hivm::ReduceOperation kind;
  auto *ctx = op->getContext();

  if (auto reduceOp = dyn_cast<hfusion::ReduceWithIndexOp>(op)) {
    kind = getReduceWithIndexOpAttr(reduceOp);
  } else if (auto reduceOp = dyn_cast<linalg::ReduceOp>(op)) {
    Block &body = reduceOp.getCombiner().front();
    auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
    auto bodyOp = yieldOp.getValues()[0].getDefiningOp();
    Type srcType = reduceOp.getInputs()[0].getType();
    Type elemType = getElementTypeOrSelf(srcType);

    if (isa<arith::AddFOp>(bodyOp) || isa<arith::AddIOp>(bodyOp)) {
      kind = hivm::ReduceOperation::sum;
    } else if (isa<arith::XOrIOp>(bodyOp)) {
      kind = hivm::ReduceOperation::xori;
    } else if (isa<arith::OrIOp>(bodyOp) && elemType.isInteger(1)) {
      kind = hivm::ReduceOperation::any;
    } else if (isa<arith::AndIOp>(bodyOp) && elemType.isInteger(1)) {
      kind = hivm::ReduceOperation::all;
    } else if (isa<arith::OrIOp>(bodyOp)) {
      assert(!elemType.isInteger(1) && "reduce_or unsupport bool");
      kind = hivm::ReduceOperation::ori;
    } else if (isa<arith::AndIOp>(bodyOp)) {
      assert(!elemType.isInteger(1) && "reduce_and unsupport bool");
      kind = hivm::ReduceOperation::andi;
    } else if (isa<arith::MulFOp>(bodyOp) || isa<arith::MulIOp>(bodyOp)) {
      kind = hivm::ReduceOperation::prod;
    } else if (isa<arith::MaximumFOp>(bodyOp) || isa<arith::MaxSIOp>(bodyOp) ||
               isa<arith::MaxNumFOp>(bodyOp)) {
      kind = hivm::ReduceOperation::max;
    } else if (isa<arith::MinimumFOp>(bodyOp) || isa<arith::MinSIOp>(bodyOp) ||
               isa<arith::MinNumFOp>(bodyOp)) {
      kind = hivm::ReduceOperation::min;
    } else {
      reduceOp.emitOpError("unsupported reduce operation: ");
      llvm::report_fatal_error("Not implemented");
    }
  } else {
    op->emitOpError("unsupported reduce operation: ");
    llvm::report_fatal_error("Not implemented");
  }

  return hivm::ReduceOpAttr::get(ctx, kind);
}

namespace {

//===----------------------------------------------------------------------===//
// Linalg/HFusion Reduce-like Op To HIVM Reduce Op
//===----------------------------------------------------------------------===//

template <typename ReduceOpTy>
struct LinalgToHIVMReduceLikeOp : public OpRewritePattern<ReduceOpTy> {
  using OpRewritePattern<ReduceOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOpTy reduceOp,
                                PatternRewriter &rewriter) const override {
    auto reduceOpInputs = reduceOp.getInputs();
    auto reduceOpInits = reduceOp.getInits();

#ifndef NDEBUG
    size_t inputLimit = 1;
    if constexpr (std::is_same<ReduceOpTy, hfusion::ReduceWithIndexOp>::value) {
      inputLimit++;
    }
    if (reduceOpInputs.size() > inputLimit)
      llvm::report_fatal_error("unsupport variadic reduce");
#endif

    const bool hasPureTensor = reduceOp.hasPureTensorSemantics();

    SmallVector<Value> expandShapeOps;
    SmallVector<Type> resTypeVec;
    /// Here we only use the first input as the anchor type because:
    ///   1) we're guaranteed that all inputs have the same shape
    ///   2) for reduce with index op, we support two modes (with and without
    ///      the index input), so the number of inits and inputs might be
    ///      different
    auto targetExpandShapeType = cast<ShapedType>(reduceOpInputs[0].getType());
    for (size_t i = 0; i < reduceOpInits.size(); i++) {
      Value expandShapeOp = hfusion_conversion_utils::createExpandShapeOp(
          reduceOp, rewriter, reduceOpInits[i], targetExpandShapeType);
      expandShapeOps.push_back(expandShapeOp);
      if (hasPureTensor) {
        auto expandShOp =
            expandShapeOps[i].getDefiningOp<tensor::ExpandShapeOp>();
        resTypeVec.emplace_back(expandShOp.getType());
      }
    }

    // If there are no indices input from HFusion/Linalg reduce op, then the hivm
    // reduce op will create it's own indeices input.
    auto indices = (reduceOpInputs.size() > 1) ? reduceOpInputs[1] : nullptr;
    auto hivmOp = rewriter.create<hivm::VReduceOp>(
        reduceOp.getLoc(), TypeRange(resTypeVec), reduceOpInputs[0],
        ValueRange(expandShapeOps), getReduceOpAttr(reduceOp),
        reduceOp.getDimensionsAttr(), indices);

    Value firstCollapseSrc =
        hasPureTensor ? hivmOp.getResult()[0] : hivmOp.getDstValue();
    int64_t outRank = cast<ShapedType>(reduceOpInits[0].getType()).getRank();
    if (outRank != 0) {
      // This is not a rank-0 case. Thus turn to the normal case.
      outRank = cast<ShapedType>(firstCollapseSrc.getType()).getRank();
    }
    auto reassociation =
        reshape_utils::getReAssociation(reduceOp.getDimensions(), outRank);

    SmallVector<Value> collapseShapeOps;
    for (size_t i = 0; i < reduceOpInits.size(); i++) {
      Value collapseSrc =
          hasPureTensor ? hivmOp.getResult()[i] : hivmOp.getDst()[i];
      collapseShapeOps.push_back(
          hfusion_conversion_utils::createCollapseShapeOp(
              rewriter, reduceOp.getLoc(), collapseSrc,
              reduceOpInits[i].getType(), reassociation, hasPureTensor));
    }

    if (hasPureTensor) {
      rewriter.replaceOp(reduceOp, ArrayRef<Value>{collapseShapeOps});
    } else {
      rewriter.eraseOp(reduceOp);
    }
    return success();
  }
};

} // namespace

void mlir::populateReductionPatternsAndLegality(RewritePatternSet &patterns,
                                                ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  target.addIllegalOp<linalg::ReduceOp, hfusion::ReduceWithIndexOp>();
  patterns.add<LinalgToHIVMReduceLikeOp<linalg::ReduceOp>,
               LinalgToHIVMReduceLikeOp<hfusion::ReduceWithIndexOp>>(context);
}
