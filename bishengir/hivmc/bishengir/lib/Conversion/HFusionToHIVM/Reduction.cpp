//===- Reduction.cpp - HFusion to HIVM dialect conversion for reduction ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/HFusionToHIVM/HFusionToHIVM.h"
#include "bishengir/Conversion/HFusionToHIVM/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;

static BoolAttr getReduceTieBreakLeftAttr(Operation *op) {
  if (auto reduceWithIndexOp = dyn_cast<hfusion::ReduceWithIndexOp>(op)) {
    // hfusion.reduce_with_index has to have 'tie_break_left' attr
    return reduceWithIndexOp.getTieBreakLeftAttr();
  }
  // set default value for backward complatibility
  return {};
}

static BoolAttr getReduceUnsignedSourceAttr(Operation *op) {
  if (auto reduceWithIndexOp = dyn_cast<hfusion::ReduceWithIndexOp>(op)) {
    // hfusion.reduce_with_index has to have 'unsigned_src' attr
    return reduceWithIndexOp.getUnsignedSrcAttr();
  }
  // set default value for backward complatibility
  return BoolAttr::get(op->getContext(), false);
}

static hivm::ReduceOperation
getReduceWithIndexKind(hfusion::ReduceWithIndexOp reduceOp) {
  auto reduceKind = reduceOp.getReduceKindAttr().getReduceWithIndexKind();
  if (reduceKind == hfusion::ReduceWithIndexKind::MAX) {
    return hivm::ReduceOperation::max_with_index;
  } else if (reduceKind == hfusion::ReduceWithIndexKind::MIN) {
    return hivm::ReduceOperation::min_with_index;
  } else {
    reduceOp.emitOpError("unsupported reduce with index operation: ");
    return hivm::ReduceOperation::none;
  }
}

static hivm::ReduceOpAttr getReduceOpAttr(Operation *op) {
  hivm::ReduceOperation kind;
  auto ctx = op->getContext();

  if (auto reduceOp = dyn_cast<hfusion::ReduceWithIndexOp>(op)) {
    kind = getReduceWithIndexKind(reduceOp);
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
      llvm_unreachable("Not implemented");
    }
  } else {
    op->emitOpError("unsupported reduce operation: ");
    llvm_unreachable("Not implemented");
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
      llvm_unreachable("unsupport variadic reduce");
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

    // For reduce with index op that has index as input, note that the index is
    // not used in the hivm op because hivm op creates its own index.
    auto hivmOp = rewriter.create<hivm::VReduceOp>(
        reduceOp.getLoc(), TypeRange(resTypeVec), reduceOpInputs[0],
        ValueRange(expandShapeOps), getReduceOpAttr(reduceOp),
        getReduceUnsignedSourceAttr(reduceOp),
        getReduceTieBreakLeftAttr(reduceOp), reduceOp.getDimensionsAttr());

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
