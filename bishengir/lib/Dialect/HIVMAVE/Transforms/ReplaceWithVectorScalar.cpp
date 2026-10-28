//===--------- ReplaceWithVectorScalar.cpp --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h"
#include "bishengir/Dialect/HIVMRegbaseIntrins/Utils/RegbaseUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <optional>
#include <tuple>
#include <utility>

namespace mlir {
#define GEN_PASS_DEF_REPLACEWITHVECTORSCALAR
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

template <typename BroadcastScalarOp>
Value eliminateBroadcastScalar(BroadcastScalarOp broadcastOp,
                               PatternRewriter &rewriter) {
  auto loc = broadcastOp.getLoc();
  auto src = broadcastOp.getSrc();
  auto srcType = src.getType();
  auto resType = broadcastOp.getRes().getType();
  VectorType resVecType = mlir::dyn_cast<VectorType>(resType);
  auto resElemType = resVecType.getElementType();
  // Special-Case: broadcast scalar i8 -> vector<xxi8> is invalid,
  // the corresponding intrin requires an i16 scalar type input.
  // We create an integer type extention operation to retain the type
  // correctness.
  if (resElemType != srcType) {
    return rewriter.create<arith::TruncIOp>(loc, resElemType, src);
  }
  return src;
}

template <typename... BroadcastScalarOps>
Value foldBroadcastInOperandImpl(Value vector, PatternRewriter &rewriter) {
  Value scalar;
  (([&]() {
     if (!scalar) {
       if (auto broadcast = vector.getDefiningOp<BroadcastScalarOps>()) {
         scalar = eliminateBroadcastScalar(broadcast, rewriter);
       }
     }
   }()),
   ...);
  return scalar;
}

Value foldBroadcastInOperand(Value vector, PatternRewriter &rewriter) {
  return foldBroadcastInOperandImpl<hivmave::VFBroadcastScalarOp,
                                    hivmave::VFBroadcastScalarMaskOp>(vector,
                                                                      rewriter);
}

template <typename HivmVFVVOp>
std::optional<SmallVector<Value>>
foldBroadcastInOperands(HivmVFVVOp op, PatternRewriter &rewriter) {
  auto operands = op.getOperands();
  if (operands.size() < 2) {
    return std::nullopt;
  }
  auto res = foldBroadcastInOperand(operands[1], rewriter);
  if (!res) {
    return std::nullopt;
  }
  auto new_operands = SmallVector<Value>(operands);
  new_operands[1] = res;
  return new_operands;
}

} // namespace

template <typename HivmVFVVOp, typename HivmVFVSOp>
struct FoldBroadcastInBinaryOp : public OpRewritePattern<HivmVFVVOp> {
public:
  using OpRewritePattern<HivmVFVVOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(HivmVFVVOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // The number of result should be 1
    if (op->getNumResults() == 0) {
      return failure();
    }
    auto resType = op.getResult().getType();
    // The result should be a vector
    VectorType resVecType = mlir::dyn_cast<VectorType>(resType);
    if (!resVecType || resVecType.getRank() != 1) {
      return failure();
    }
    auto new_operands = foldBroadcastInOperands(op, rewriter);
    if (new_operands) {
      auto replacer =
          rewriter.create<HivmVFVSOp>(loc, resVecType, *new_operands);
      rewriter.replaceOp(op, replacer);
      return success();
    }
    return failure();
  }
};

namespace {

struct ReplaceWithVectorScalarPass
    : public impl::ReplaceWithVectorScalarBase<ReplaceWithVectorScalarPass> {
  using ReplaceWithVectorScalarBase<
      ReplaceWithVectorScalarPass>::ReplaceWithVectorScalarBase;

public:
  void runOnOperation() override;
};

} // namespace

void ReplaceWithVectorScalarPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  auto funcOp = getOperation();
  RewritePatternSet patterns(ctx);
  patterns.add<FoldBroadcastInBinaryOp<hivmave::VFAddOp, hivmave::VFAddsOp>,
               FoldBroadcastInBinaryOp<hivmave::VFMinOp, hivmave::VFMinsOp>,
               FoldBroadcastInBinaryOp<hivmave::VFMaxOp, hivmave::VFMaxsOp>,
               FoldBroadcastInBinaryOp<hivmave::VMinSIOp, hivmave::VMinsSIOp>,
               FoldBroadcastInBinaryOp<hivmave::VMaxSIOp, hivmave::VMaxsSIOp>,
               FoldBroadcastInBinaryOp<hivmave::VMinUIOp, hivmave::VMinsUIOp>,
               FoldBroadcastInBinaryOp<hivmave::VMaxUIOp, hivmave::VMaxsUIOp>,
               FoldBroadcastInBinaryOp<hivmave::VFMulOp, hivmave::VFMulsOp>,
               FoldBroadcastInBinaryOp<hivmave::VFShrOp, hivmave::VFShrsOp>,
               FoldBroadcastInBinaryOp<hivmave::VFShlOp, hivmave::VFShlsOp>>(
      ctx);

  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> hivmave::createReplaceWithVectorScalarPass() {
  return std::make_unique<ReplaceWithVectorScalarPass>();
}
