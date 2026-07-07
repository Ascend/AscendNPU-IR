//===------------------------- LowerIndex.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Arith/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include <optional>
#include <utility>

namespace mlir {
#define GEN_PASS_DEF_LIFTARITHINDEXCAST
#include "bishengir/Dialect/Arith/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
namespace {

struct LiftArithIndexCastPass
    : public impl::LiftArithIndexCastBase<LiftArithIndexCastPass> {
  using LiftArithIndexCastBase<LiftArithIndexCastPass>::LiftArithIndexCastBase;

public:
  void runOnOperation() override;
};

// we sometimes see vector<...xindex> (e.g., from triton arange test case)
// in the input MLIR. HIVMAVE lowering pipeline does not consider index
// as a valid vector element type, causing compilation failures for them
//
// this pass moves arith.index_cast towards operands
// for example convert:
//     %0 = vector.broadcast %arg1 : index to vector<64xindex>
//     %1 = arith.addi %0, %cst : vector<64xindex>
//     %2 = arith.index_cast %1 : vector<64xindex> to vector<64xi32>
// to:
//     %0 = arith.index_cast %arg1 : index to i32
//     %1 = vector.broadcast %0 : i32 to vector<64xi32>
//     %2 = arith.addi %1, %cst : vector<64xi32> to vector<64xi32>
//
// We lift arith.index_cast by
// (1) look at what's its input value
// (2) try to convert:
//          arith.index_cast (op (%1, %2, ...))
//     to:
//          op (arith.index_cast (%1), arith.index_cast (%2), ...)
// (3) recursively apply the conversion (bubble up arith.index_cast)
//     until reaching function/block arguments, etc
// If there are arith.constant, convert them from index to int as well
// This reduces vector ops operating on index, simplifying lowering

template <typename CastOp>
struct ArithIndexCast : public OpRewritePattern<CastOp> {
  using OpRewritePattern<CastOp>::OpRewritePattern;

  template <typename OpT>
  OpT rewriteBinaryOpWithType(PatternRewriter &rewriter, CastOp dest, OpT op,
                              Type operandTy, Type resultTy) const {
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type srcTy = lhs.getType();
    if (rhs.getType() != srcTy) {
      llvm_unreachable("Binary operation operand type mismatch");
    }
    auto loc = op.getLoc();
    Value newLhs = rewriter.create<CastOp>(loc, operandTy, lhs).getResult();
    Value newRhs = rewriter.create<CastOp>(loc, operandTy, rhs).getResult();
    auto newOp = rewriter.create<OpT>(op.getLoc(), resultTy, newLhs, newRhs);
    newOp->setAttrs(op->getAttrs());
    rewriter.replaceOp(dest, newOp);
    return newOp;
  }
  template <typename OpT>
  OpT rewriteUnaryOpWithType(PatternRewriter &rewriter, CastOp dest, OpT op,
                             Type operandTy, Type resultTy) const {
    Value lhs = op->getOperand(0);
    auto loc = op.getLoc();
    Value newLhs = rewriter.create<CastOp>(loc, operandTy, lhs).getResult();
    auto newOp = rewriter.create<OpT>(loc, resultTy, newLhs);
    newOp->setAttrs(op->getAttrs());
    rewriter.replaceOp(dest, newOp);
    return newOp;
  }
  Operation *rewriteArithConstant(PatternRewriter &rewriter, CastOp dest,
                                  arith::ConstantOp op, Type ty) const {
    auto value = op.getValue();

    // handle scalar index type arith.constant
    if (auto intValue = dyn_cast<mlir::IntegerAttr>(value)) {
      auto newOp = rewriter.create<arith::ConstantOp>(
          op.getLoc(), ty, IntegerAttr::get(ty, intValue.getInt()));
      rewriter.replaceOp(dest, newOp);
      return newOp;
    }

    // handle vector type
    if (VectorType vecTy = dyn_cast<VectorType>(value.getType())) {
      if (auto intVecValue = dyn_cast<mlir::DenseIntElementsAttr>(value)) {
        Type elementTy = vecTy.getElementType();
        auto newVecValue = intVecValue.mapValues(
            elementTy, [](const llvm::APInt &src) { return src; });
        auto newOp =
            rewriter.create<arith::ConstantOp>(op.getLoc(), ty, newVecValue);
        rewriter.replaceOp(dest, newOp);
        return newOp;
      }
    }

    return nullptr;
  }

  LogicalResult matchAndRewrite(CastOp op,
                                PatternRewriter &rewriter) const override {

    if (!isa<CastOpInterface>(op.getOperation()))
      return failure();
    auto src = op.getIn();
    auto dst = op.getOut();
    Operation *srcOp = src.getDefiningOp();
    if (!srcOp || srcOp->getNumResults() != 1) {
      // cannot transform block arguments, etc
      return failure();
    }
    VectorType srcVecTy = dyn_cast<VectorType>(src.getType());
    VectorType dstVecTy = dyn_cast<VectorType>(dst.getType());
    // double check this cast is doing the following conversion:
    // vector<...xindex> -> vector<...xi32/i16/...>
    if (!srcVecTy || !dstVecTy || !srcVecTy.getElementType().isIndex() ||
        !dstVecTy.getElementType().isInteger())
      return failure();
    // vector of index -> vector of int cast
    // at the time of writing, this code pattern only appears with
    // code using linalg.index (from triton arange), so we mainly target
    // operations in these examples.
    Operation *newOp =
        llvm::TypeSwitch<Operation *, Operation *>(srcOp)
            .Case([&](arith::AddIOp oldOp) -> Operation * {
              return rewriteBinaryOpWithType(rewriter, op, oldOp, dstVecTy,
                                             dstVecTy);
            })
            .Case([&](arith::SubIOp oldOp) -> Operation * {
              return rewriteBinaryOpWithType(rewriter, op, oldOp, dstVecTy,
                                             dstVecTy);
            })
            .Case([&](arith::MulIOp oldOp) -> Operation * {
              return rewriteBinaryOpWithType(rewriter, op, oldOp, dstVecTy,
                                             dstVecTy);
            })
            // vector.broadcast that appeared in input code
            .Case([&](vector::BroadcastOp oldOp) -> Operation * {
              auto operandTy = dstVecTy.getElementType();
              if (auto operandVectorType = dyn_cast<VectorType>(oldOp.getSourceType()))
                operandTy = VectorType::get(operandVectorType.getShape(), operandTy);
              return rewriteUnaryOpWithType(
                  rewriter, op, oldOp, operandTy, dstVecTy);
            })
            // convert arith.constant (direct conversion)
            .Case([&](arith::ConstantOp oldOp) -> Operation * {
              return rewriteArithConstant(rewriter, op, oldOp, dstVecTy);
            })
            .Default([&](Operation *) -> Operation * {
              srcOp->emitWarning("LiftArithIndexCast: Unsupported operation");
              return nullptr;
            });
    return newOp ? success() : failure();
  }
};

} // end anonymous namespace

void LiftArithIndexCastPass::runOnOperation() {
  auto funcOp = getOperation();
  if (!hivm::isVF(funcOp))
    return;

  RewritePatternSet patterns(&getContext());
  patterns.add<ArithIndexCast<arith::IndexCastOp>,
               ArithIndexCast<arith::IndexCastUIOp>>(&getContext());
  (void)applyPatternsGreedily(funcOp, std::move(patterns));
}

std::unique_ptr<Pass> arith::createLiftArithIndexCastPass() {
  return std::make_unique<LiftArithIndexCastPass>();
}
