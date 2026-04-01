//===------- LegalizeFP8.cpp - Legalize FP8 type Pass -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/IR/HFusionImpl.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
#define GEN_PASS_DEF_LEGALIZEFP8PASS
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hfusion;

class LegalizeFP8Pass : public impl::LegalizeFP8PassBase<LegalizeFP8Pass> {
public:
  void runOnOperation() override;
};

template <typename FP8Type, typename = std::enable_if_t<
                                std::is_same_v<Float8E4M3FNType, FP8Type> ||
                                std::is_same_v<Float8E5M2Type, FP8Type>>>
static bool isFP8(Type type) {
  if constexpr (std::is_same_v<FP8Type, Float8E4M3FNType>) {
    return type.isFloat8E4M3FN();
  } else if constexpr (std::is_same_v<FP8Type, Float8E5M2Type>) {
    return type.isFloat8E5M2();
  } else {
    llvm_unreachable("Unexpected FP8 type");
    return false;
  }
}

template <typename FP8Type, typename = std::enable_if_t<
                                std::is_same_v<Float8E4M3FNType, FP8Type> ||
                                std::is_same_v<Float8E5M2Type, FP8Type>>>
static Type getFP8Ty(PatternRewriter &rewriter) {
  if constexpr (std::is_same_v<FP8Type, Float8E4M3FNType>) {
    return rewriter.getFloat8E4M3FNType();
  } else if constexpr (std::is_same_v<FP8Type, Float8E5M2Type>) {
    return rewriter.getFloat8E5M2Type();
  } else {
    llvm_unreachable("Unexpected FP8 type");
    return Type();
  }
}

template <typename FP8Type, typename = std::enable_if_t<
                                std::is_same_v<Float8E4M3FNType, FP8Type> ||
                                std::is_same_v<Float8E5M2Type, FP8Type>>>
static bool hasFP8Semantic(Operation *op) {
  auto isFP8Semantic = [](auto oper) {
    auto elemTy = getElementTypeOrSelf(oper.getType());
    return isFP8<FP8Type>(elemTy);
  };
  return std::any_of(op->getOperands().begin(), op->getOperands().end(),
                     isFP8Semantic);
}

template <typename FP8Type, typename = std::enable_if_t<
                                std::is_same_v<Float8E4M3FNType, FP8Type> ||
                                std::is_same_v<Float8E5M2Type, FP8Type>>>
static bool isFP8ElemTypeSelect(Operation *op) {
  if (!isa<arith::SelectOp>(op)) {
    return false;
  }
  return hasFP8Semantic<FP8Type>(op);
}

template <typename FP8Type, typename = std::enable_if_t<
                                std::is_same_v<Float8E4M3FNType, FP8Type> ||
                                std::is_same_v<Float8E5M2Type, FP8Type>>>
static bool shouldLegalizeFP8Op(Operation *op) {
  if (!hasFP8Semantic<FP8Type>(op))
    return false;
  // Support the compare operation
  if (isa<hfusion::CompareOp>(op)) {
    return true;
  }
  if (isa<linalg::ElemwiseUnaryOp>(op)) {
    auto unaryOp = cast<linalg::ElemwiseUnaryOp>(op);
    auto funAttr = unaryOp.getFun();
    // Support the ceil,floor,abs,exp,log,sqrt operation
    using unary = linalg::UnaryFn;
    return llvm::is_contained({unary::ceil, unary::floor, unary::abs,
                               unary::exp, unary::log, unary::sqrt},
                              funAttr);
  }
  if (isa<linalg::ElemwiseBinaryOp>(op)) {
    auto binaryOp = cast<linalg::ElemwiseBinaryOp>(op);
    auto funAttr = binaryOp.getFun();
    // Support the div,mul,add,sub operation
    using binary = linalg::BinaryFn;
    return llvm::is_contained(
        {binary::div, binary::mul, binary::add, binary::sub}, funAttr);
  }
  if (isa<linalg::ReduceOp>(op)) {
    return true;
  }
  if (isa<hfusion::ElemwiseUnaryOp>(op)) {
    auto unaryOp = cast<hfusion::ElemwiseUnaryOp>(op);
    auto funAttr = unaryOp.getFun();
    // Support the sin,cos,erf,exp2,log2,rsqrt,sqrt_rn operation
    using unary = hfusion::UnaryFn;
    return llvm::is_contained({unary::sin, unary::cos, unary::erf, unary::exp2,
                               unary::log2, unary::rsqrt, unary::sqrt},
                              funAttr);
  }
  if (isa<hfusion::ElemwiseBinaryOp>(op)) {
    auto binaryOp = cast<hfusion::ElemwiseBinaryOp>(op);
    auto funAttr = binaryOp.getFun();
    // Support the minnumf,maxnumf,maxf,minf operation
    using binary = hfusion::BinaryFn;
    return llvm::is_contained(
        {binary::minnumf, binary::maxnumf, binary::maxf, binary::minf},
        funAttr);
  }
  if (isa<hfusion::CumsumOp>(op)) {
    return true;
  }
  if (isa<hfusion::ReduceWithIndexOp>(op)) {
    return true;
  }
  // TODO: In the future, consider enabling the bisheng compiler to support
  // printing of fp8 types at the underlying level.
  if (isa<hfusion::PrintOp>(op)) {
    return true;
  }
  return false;
}

template <typename Op, typename FP8Type>
static Operation *createNewOp(PatternRewriter &rewriter, Op op,
                              SmallVector<Value> &castedOperands) {
  IRMapping mapper;
  for (const auto &[idx, oper] : llvm::enumerate(op->getOperands()))
    mapper.map(oper, castedOperands[idx]);

  auto *newOp = rewriter.cloneWithoutRegions(*op, mapper);
  auto *ctx = op->getContext();
  for (const auto &[idx, res] : llvm::enumerate(op->getResults())) {
    ShapedType shapedType = cast<ShapedType>(res.getType());
    if (!(shapedType && isFP8<FP8Type>(getElementTypeOrSelf(shapedType)))) {
      continue;
    }
    auto newResTy = shapedType.clone(Float32Type::get(ctx));
    newOp->getResult(idx).setType(newResTy);
  }

  if (op->getNumRegions() <= 0)
    return newOp;

  Region &newRegion = newOp->getRegions().front();
  Block *newBlock = rewriter.createBlock(&newRegion);
  rewriter.setInsertionPointToStart(newBlock);
  Block *block = &op->getRegion(0).front();
  auto targetType = rewriter.getF32Type();
  for (BlockArgument bbArg : block->getArguments()) {
    auto argType = bbArg.getType();
    Type newArgType = isFP8<FP8Type>(argType) ? targetType : argType;
    mapper.map(bbArg, newBlock->addArgument(newArgType, bbArg.getLoc()));
  }
  for (auto &bodyOp : *block) {
    auto *newBodyOp = rewriter.clone(bodyOp, mapper);
    if ((isFP8ElemTypeSelect<FP8Type>(&bodyOp)) ||
        (!isa<linalg::YieldOp>(bodyOp) && !isa<arith::CmpFOp>(bodyOp) &&
         !isa<linalg::IndexOp>(bodyOp) && !isa<arith::IndexCastOp>(bodyOp) &&
         !isa<arith::CmpIOp>(bodyOp) && !isa<arith::SelectOp>(bodyOp))) {
      newBodyOp->getResult(0).setType(Float32Type::get(ctx));
    }
  }
  return newOp;
}

template <typename Op, typename FP8Type>
static void createF32ElementTypeOpRegion(Op fp8Op, PatternRewriter &rewriter) {
  // Handle ops in region too
  for (size_t regionIndex = 0; regionIndex < fp8Op->getNumRegions();
       regionIndex++) {
    fp8Op->getRegion(regionIndex)
        .walk([&](Operation *opInRegion) -> WalkResult {
          if (!shouldLegalizeFP8Op<FP8Type>(opInRegion))
            WalkResult::advance();
          for (auto operand : opInRegion->getOperands()) {
            auto defOp = operand.getDefiningOp();
            if (!defOp)
              continue;
            if (defOp->getParentOp() != opInRegion->getParentOp()) {
              Value castedOperand =
                  isFP8<FP8Type>(getElementTypeOrSelf(operand.getType()))
                      ? castTo(rewriter, operand, rewriter.getF32Type())
                      : operand;

              // only replace operand used in this regionOp, rely on later
              // CSE and DCE to eliminate duplicate value
              rewriter.replaceUsesWithIf(operand, castedOperand,
                                         [&](OpOperand &use) {
                                           Operation *op = use.getOwner();
                                           return op == opInRegion;
                                         });
            }
          }
          return WalkResult::advance();
        });
  }
}

template <typename Op, typename FP8Type>
static void createF32ElementTypeOp(Op fp8Op, PatternRewriter &rewriter) {
  RewriterBase::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(fp8Op);

  Type fp8Type = getFP8Ty<FP8Type>(rewriter);
  Type f32Type = rewriter.getF32Type();

  // Count all the OPs that require casting.
  SmallVector<Value> castedOperands;
  for (auto oper : fp8Op->getOperands()) {
    Value castedOperand = isFP8<FP8Type>(getElementTypeOrSelf(oper.getType()))
                              ? castTo(rewriter, oper, f32Type)
                              : oper;
    castedOperands.push_back(castedOperand);
  }

  auto newOp = createNewOp<Op, FP8Type>(rewriter, fp8Op, castedOperands);
  createF32ElementTypeOpRegion<Operation *, FP8Type>(newOp, rewriter);

  rewriter.setInsertionPointAfter(newOp);
  SmallVector<Value> castedResults;
  for (auto res : newOp->getResults()) {
    auto resType = getElementTypeOrSelf(res.getType());
    Value castedResult = resType.isF32() ? castTo(rewriter, res, fp8Type) : res;
    castedResults.push_back(castedResult);
  }

  rewriter.replaceOp(fp8Op, castedResults);
}

template <typename Op, typename FP8Type>
static LogicalResult legalizeFP8(Op op, PatternRewriter &rewriter) {
  if (!shouldLegalizeFP8Op<FP8Type>(op))
    return failure();

  createF32ElementTypeOp<Op, FP8Type>(op, rewriter);
  return success();
}

template <typename Op>
struct LegalizeFP8E4M3 : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    return legalizeFP8<Op, Float8E4M3FNType>(op, rewriter);
  }
};

template <typename Op>
struct LegalizeFP8E5M2 : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    return legalizeFP8<Op, Float8E5M2Type>(op, rewriter);
  }
};

template <typename OpType>
static void registerOne(RewritePatternSet &patterns) {
  patterns.add<LegalizeFP8E4M3<OpType>>(patterns.getContext());
  patterns.add<LegalizeFP8E5M2<OpType>>(patterns.getContext());
}

template <typename... OpTypes>
static void registerAll(RewritePatternSet &patterns) {
  (registerOne<OpTypes>(patterns), ...);
}

void populateLegalizeFP8Pattern(RewritePatternSet &patterns) {
  registerAll<mlir::linalg::ElemwiseUnaryOp, mlir::linalg::ElemwiseBinaryOp,
              mlir::linalg::ReduceOp, mlir::hfusion::CastOp,
              mlir::hfusion::CompareOp, mlir::hfusion::CumsumOp,
              mlir::hfusion::ElemwiseBinaryOp, mlir::hfusion::ElemwiseUnaryOp,
              mlir::hfusion::ReduceWithIndexOp,
              mlir::hfusion::PrintOp>(patterns);
}

void LegalizeFP8Pass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  populateLegalizeFP8Pattern(patterns);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<Pass> mlir::hfusion::createLegalizeFP8Pass() {
  return std::make_unique<LegalizeFP8Pass>();
}