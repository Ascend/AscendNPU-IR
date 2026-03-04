//===--------- LegalizeF16ForTriton.cpp - Legalize BF16 type Pass --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// F16 has poor percisions. Need to convert F16 computations to F32 to ensure 
// good percision.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Triton/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton {
#define GEN_PASS_DEF_LEGALIZEF16FORTRITON
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;
using namespace mlir::triton;

static bool shouldLegalizeF16Op(Operation *op) {
  for (Value operand : op->getOperands()) {
    if (isa<BFloat16Type,Float16Type>(getElementTypeOrSelf(operand.getType()))) {
      bool isExcluded = isa<arith::FPToUIOp, arith::FPToSIOp, arith::ExtFOp, arith::TruncFOp,
                          arith::SIToFPOp, arith::UIToFPOp, arith::BitcastOp>(op);
      return !isExcluded; 
    }
  }
  return false; 
}

template <typename Op>
static void createF32ElementTypeOp(Op f16Op, PatternRewriter &rewriter) {
  Operation *op = f16Op.getOperation();
  Location loc = op->getLoc();
  Type f32Type = rewriter.getF32Type();

  SmallVector<Value> castedOperands;
  for (Value operand : op->getOperands()) {
    Type elemTy = getElementTypeOrSelf(operand.getType());
    ShapedType maybeShaped = dyn_cast<ShapedType>(operand.getType());
    Type castResult = maybeShaped ? maybeShaped.clone(f32Type) : f32Type;

    if (isa<BFloat16Type,Float16Type>(elemTy)) {
      Value casted = rewriter.create<arith::ExtFOp>(loc, castResult, operand);
      castedOperands.push_back(casted);
    } else
      castedOperands.push_back(operand);
  }

  // All F16 results should become F32 too
  SmallVector<Type> newResultTypes;
  for (Type resTy : op->getResultTypes()) {
    Type elemTy = getElementTypeOrSelf(resTy);
    ShapedType maybeShaped = dyn_cast<ShapedType>(resTy);
    Type newResTy = maybeShaped ? maybeShaped.clone(f32Type) : f32Type;
    newResultTypes.push_back(isa<BFloat16Type,Float16Type>(elemTy) ? newResTy : resTy);
  }

  // Create the F32 version of the op with the casted operands
  OperationState state(loc, Op::getOperationName());
  state.addOperands(castedOperands);
  state.addTypes(newResultTypes);
  state.addAttributes(op->getAttrs()); 

  Operation *newOp = rewriter.create(state);

  // Cast the results back to F16
  SmallVector<Value> castedResults;
  for (auto [res, oldType] :
       llvm::zip(newOp->getResults(), op->getResultTypes())) {
    if (res.getType() != oldType) {
      Value casted = rewriter.create<arith::TruncFOp>(loc, oldType, res);
      castedResults.push_back(casted);
    } else
      castedResults.push_back(res);
  }

  // For all arith.extf on the original op result,
  // replace them with the result of the new F32 op
  for (auto [newRes, oldRes] :
       llvm::zip(newOp->getResults(), op->getResults())) {
    SmallVector<Operation *> redundantCasts;
    for (Operation *user : oldRes.getUsers()) {
      if (arith::ExtFOp ext = dyn_cast<arith::ExtFOp>(user))
        if (ext.getResult().getType() == newRes.getType())
          redundantCasts.push_back(user);
    }
 
    for (Operation *cast : redundantCasts)
      rewriter.replaceOp(cast, newRes);
  }

  rewriter.replaceOp(op, castedResults);
}

template <typename Op>
static LogicalResult legalizeF16(Op op, PatternRewriter &rewriter) {
  if (!shouldLegalizeF16Op(op))
    return failure();

  createF32ElementTypeOp(op, rewriter);
  return success();
}

namespace bishengir {
namespace triton {

class LegalizeF16ForTritonPass
    : public mlir::triton::impl::LegalizeF16ForTritonBase<
          LegalizeF16ForTritonPass> {
public:
  void runOnOperation() override;
};

template <typename Op>
struct LegalizeF16 : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    return legalizeF16<Op>(op, rewriter);
  }
};

struct legalizeF16ForReduce : public OpRewritePattern<ReduceOp> {
public:
  using OpRewritePattern<ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter &rewriter) const override {
    // Check whether there are FP16/BF16 inputs.
    if (!hasF16ElemType(op.getOperands())) {
      return failure();
    }

    if (!shouldComputeF16ToF32(op)) {
      return failure();
    }

    SmallVector<Value> newInputs = 
        normalizeSrcToTargetType(rewriter, op.getLoc(), op.getOperands());

    ReduceOp newOp = 
        createNewReduceOp(op, rewriter, newInputs);
    rewriter.setInsertionPointAfter(newOp);
    replaceF16ResultsWithF32(op, newOp, rewriter);

    return success();
  }

private:
  bool hasF16ElemType(ValueRange values) const {
    for (Value v : values) {
      if (isa<BFloat16Type,Float16Type>(getElementTypeOrSelf(v.getType())))
        return true;
    }
    return false;
  }

  bool shouldComputeF16ToF32(ReduceOp op) const {
    Region &region = op.getCombineOp();
    if (region.empty()) return false;
    
    Block &block = region.front();

    Operation *terminator = block.getTerminator();
    if (terminator->getNumOperands() == 0) return false;

    Operation *defOp = terminator->getOperand(0).getDefiningOp();
    return defOp && isa<arith::AddFOp>(defOp);
  }

  // upcast the input to F32
  SmallVector<Value> normalizeSrcToTargetType(PatternRewriter &rewriter, 
                                              Location loc,
                                              ValueRange operands) const {
    SmallVector<Value> newOperands;
    auto f32Ty = rewriter.getF32Type();
    
    for (Value operand : operands) {
      Type oldTy = operand.getType();
      Type elemTy = getElementTypeOrSelf(oldTy);
      
      if (elemTy.isF16() || elemTy.isBF16()) {
        Type newTy;
        if (auto tensorTy = dyn_cast<RankedTensorType>(oldTy)) {
          newTy = RankedTensorType::get(tensorTy.getShape(), f32Ty, tensorTy.getEncoding());
        } else {
          newTy = f32Ty;
        }
        newOperands.push_back(rewriter.create<arith::ExtFOp>(loc, newTy, operand));
      } else {
        newOperands.push_back(operand);
      }
    }
    return newOperands;
  }

  // Create a new ReduceOp.
  ReduceOp createNewReduceOp(ReduceOp oldOp, 
                             PatternRewriter &rewriter, 
                             ValueRange newInputs) const {
    Location loc = oldOp.getLoc();
    auto newOp = rewriter.create<ReduceOp>(loc, newInputs, oldOp.getAxis());

    newOp->setAttrs(oldOp->getAttrs());  
    Region &region = newOp.getCombineOp();
    Block *block = rewriter.createBlock(&region);
    
    auto f32Ty = rewriter.getF32Type();
    unsigned numArgs = newInputs.size();
    
    for (unsigned i = 0; i < 2 * numArgs; ++i) {
      block->addArgument(f32Ty, loc);
    }
    
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(block);
      SmallVector<Value> results;
      for (unsigned i = 0; i < numArgs; ++i) {
        Value lhs = block->getArgument(i);
        Value rhs = block->getArgument(i + numArgs);
        results.push_back(rewriter.create<arith::AddFOp>(loc, lhs, rhs));
      }
      rewriter.create<ReduceReturnOp>(loc, results);
    }
    
    return newOp;
  }

  // generate TruncFOp
  void replaceF16ResultsWithF32(ReduceOp oldOp, 
                                ReduceOp newOp, 
                                PatternRewriter &rewriter) const {
    Location loc = oldOp.getLoc();
    SmallVector<Value> finalResults;
    auto newResults = newOp.getResults(); 
    
    for (unsigned i = 0; i < oldOp.getNumResults(); ++i) {
      Value res = newResults[i];
      Type origTy = oldOp->getResult(i).getType();
      Type elemTy = getElementTypeOrSelf(origTy);
      
      if (isa<BFloat16Type,Float16Type>(elemTy)) {
        finalResults.push_back(rewriter.create<arith::TruncFOp>(loc, origTy, res));
      } else {
        finalResults.push_back(res);
      }
    }
    
    rewriter.replaceOp(oldOp, finalResults);
  }
};

/**
 * Pattern: BypassRedundantReduceConversions
 * Intent: Optimize high-precision reduction chains by removing unnecessary 
 * down-casting and up-casting between consecutive ReduceOps.
* * * Example:
 * %in_f32 = arith.extf %src_f16 : f16 to f32
 * %0 = tt.reduce(%in_f32) : f32 -> f32
 * %1 = arith.truncf %0 : f32 to f16
 * %2 = arith.extf %1 : f16 to f32
 * %3 = tt.reduce(%2) : f32 -> f32
 * %res_f16 = arith.truncf %3 : f32 to f16
 * * ---- Optimized to ----
 * %in_f32 = arith.extf %src_f16 : f16 to f32
 * %0 = tt.reduce(%in_f32) : f32 -> f32
 * %3 = tt.reduce(%0) : f32 -> f32
 * %res_f16 = arith.truncf %3 : f32 to f16
 */
struct BypassRedundantReduceConversions : public OpRewritePattern<ReduceOp> {
public:
  using OpRewritePattern<ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOp op, PatternRewriter &rewriter) const override {
    bool foundRedundantChain = false;
    SmallVector<Value> cleanInputs;

    for (Value operand : op.getOperands()) {
      Value cleaned = operand;

      // Match the pattern: extf(truncf(source))
      if (auto extOp = operand.getDefiningOp<arith::ExtFOp>()) {
        if (auto truncOp = extOp.getIn().getDefiningOp<arith::TruncFOp>()) {
          Value source = truncOp.getIn();
          
          // Ensure the original source and the current operand have the same type (typically f32)
          if (source.getType() == operand.getType()) {
            cleaned = source;
            foundRedundantChain = true;
          }
        }
      }
      cleanInputs.push_back(cleaned);
    }

    if (!foundRedundantChain) {
      return failure();
    }

    auto newOp = rewriter.create<ReduceOp>(op.getLoc(), cleanInputs, op.getAxis());
    newOp->setAttrs(op->getAttrs());

    Region &newRegion = newOp.getCombineOp();
    rewriter.cloneRegionBefore(op.getCombineOp(), newRegion, newRegion.end());

    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

template <typename OpType>
void registerOne(RewritePatternSet &patterns) {
  patterns.add<LegalizeF16<OpType>>(patterns.getContext());
}

/// Variadic helper function.
template <typename... OpTypes>
void registerAll(RewritePatternSet &patterns) {
  (registerOne<OpTypes>(patterns), ...);
}

void populateLegalizeF16Pattern(RewritePatternSet &patterns) {
  registerAll<
#define GET_OP_LIST
#include "mlir/Dialect/Arith/IR/ArithOps.cpp.inc"
      >(patterns);
  registerAll<
#define GET_OP_LIST
#include "mlir/Dialect/Math/IR/MathOps.cpp.inc"
      >(patterns);
  patterns.add<legalizeF16ForReduce>(patterns.getContext());
  patterns.add<BypassRedundantReduceConversions>(patterns.getContext());
}

void LegalizeF16ForTritonPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  populateLegalizeF16Pattern(patterns);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<Pass> createLegalizeF16ForTritonPass() {
  return std::make_unique<LegalizeF16ForTritonPass>();
}

} // namespace triton
} // namespace bishengir