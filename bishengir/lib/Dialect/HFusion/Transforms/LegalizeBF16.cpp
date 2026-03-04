//===--------- LegalizeBF16.cpp - Legalize BF16 type Pass -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/IR/HFusionImpl.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "hfusion-legalize-bf16"

namespace mlir {
#define GEN_PASS_DEF_LEGALIZEBF16PASS
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hfusion;

static thread_local bool isAscend950Arch{false};

class LegalizeBF16Pass : public impl::LegalizeBF16PassBase<LegalizeBF16Pass> {
public:
  void runOnOperation() override;
};

static bool isBF16ElemTypeSelect(Operation *op) {
  if (!isa<arith::SelectOp>(op)) {
    return false;
  }
  auto oper = op->getOperands()[1];
  auto elemTy = getElementTypeOrSelf(oper.getType());
  return isa<BFloat16Type>(elemTy);
}

static bool shouldLegalizeBF16Op(Operation *op) {
  bool isExcluded = isa<hfusion::CastOp>(op) || isa<linalg::FillOp>(op) ||
                    isa<linalg::CopyOp>(op) || isa<linalg::MatmulOp>(op) ||
                    isa<linalg::BatchMatmulOp>(op) ||
                    isa<linalg::TransposeOp>(op) || isa<hfusion::LoadOp>(op) ||
                    isa<hfusion::StoreOp>(op) || isa<hfusion::BitcastOp>(op);

  if (isAscend950Arch) {
    // Ascend 950 supports hardware instructions for BF16 floor operations.
    if (isa<linalg::ElemwiseUnaryOp>(op)) {
      auto unaryOp = cast<linalg::ElemwiseUnaryOp>(op);
      auto funAttr = unaryOp.getFun();
      isExcluded |= funAttr == linalg::UnaryFn::floor;
    }
    isExcluded |= isa<hfusion::GatherOp>(op);
  }
  return utils::hasBF16Operand(op) && !isExcluded;
}

template <typename Op>
static Operation *createNewOp(PatternRewriter &rewriter, Op bf16Op,
                              SmallVector<Value> &castedOperands) {
  LLVM_DEBUG(llvm::dbgs() << "[createNewOp] Op: " << *bf16Op << "\n";);
  IRMapping mapper;
  Operation *op = bf16Op.getOperation();
  for (const auto &[idx, oper] : llvm::enumerate(op->getOperands()))
    mapper.map(oper, castedOperands[idx]);

  auto *newOp = rewriter.cloneWithoutRegions(*op, mapper);
  auto *ctx = op->getContext();
  for (const auto &[idx, res] : llvm::enumerate(op->getResults())) {
    ShapedType shapedType = cast<ShapedType>(res.getType());
    if (!(shapedType && getElementTypeOrSelf(shapedType).isBF16())) {
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
    Type newArgType = argType.isBF16() ? targetType : argType;
    mapper.map(bbArg, newBlock->addArgument(newArgType, bbArg.getLoc()));
  }

  auto isForbiddenSetResultOp = [](Operation *op) -> bool {
    return (isa<linalg::YieldOp>(op) || isa<tensor::YieldOp>(op) ||
            isa<linalg::IndexOp>(op) || isa<arith::IndexCastOp>(op) ||
            isa<arith::CmpFOp>(op) || isa<arith::CmpIOp>(op) ||
            isa<arith::AndIOp>(op) || isa<arith::OrIOp>(op) ||
            isa<arith::XOrIOp>(op) ||
            // isBF16ElemTypeSelect checks if op is select [i1, bf16, bf16].
            // now isBF16ElemTypeSelect returns false. But op may be select with
            // i32. thus we disable replacing the result because it does not
            // select bf16.
            isa<arith::SelectOp>(op));
  };

  for (auto &bodyOp : *block) {
    LLVM_DEBUG(llvm::dbgs() << "└─ bodyOp: " << bodyOp << "\n";);
    auto *newBodyOp = rewriter.clone(bodyOp, mapper);
    if ((isBF16ElemTypeSelect(&bodyOp)) || !isForbiddenSetResultOp(&bodyOp)) {
      newBodyOp->getResult(0).setType(Float32Type::get(ctx));
    }
  }
  return newOp;
}

template <typename Op>
static void createF32ElementTypeOpRegion(Op bf16Op, PatternRewriter &rewriter) {
  LLVM_DEBUG(llvm::dbgs() << "[createF32ElementTypeOpRegion] Op: " << *bf16Op
                          << "\n";);
  // Handle ops in region too
  for (size_t regionIndex = 0; regionIndex < bf16Op->getNumRegions();
       regionIndex++) {
    bf16Op->getRegion(regionIndex)
        .walk([&](Operation *opInRegion) -> WalkResult {
          if (!shouldLegalizeBF16Op(opInRegion))
            WalkResult::advance();
          LLVM_DEBUG(llvm::dbgs() << "└─ walking Op: " << *opInRegion << "\n";);
          for (auto operand : opInRegion->getOperands()) {
            if (isa<BlockArgument>(operand))
              continue;
            LLVM_DEBUG(llvm::dbgs() << "  └─ operand: " << operand << "\n";);
            if (operand.getDefiningOp()->getParentOp() !=
                opInRegion->getParentOp()) {
              if (getElementTypeOrSelf(operand.getType()).isBF16()) {
                Value castedOperand =
                    castTo(rewriter, operand,
                           /*targetElemType=*/rewriter.getF32Type());
                // only replace operand used in this regionOp, rely on later
                // CSE and DCE to eliminate duplicate value
                rewriter.replaceUsesWithIf(operand, castedOperand,
                                           [&](OpOperand &use) {
                                             Operation *op = use.getOwner();
                                             return op == opInRegion;
                                           });
              }
            }
          }
          LLVM_DEBUG(llvm::dbgs()
                         << "-> rewritten Op: " << *opInRegion << "\n";);
          return WalkResult::advance();
        });
  }
}

template <typename Op>
static void createF32ElementTypeOp(Op bf16Op, PatternRewriter &rewriter) {
  RewriterBase::InsertionGuard g(rewriter);
  Operation *op = bf16Op.getOperation();
  rewriter.setInsertionPoint(op);

  Type bf16Type = rewriter.getBF16Type();
  Type f32Type = rewriter.getF32Type();

  SmallVector<Value> castedOperands;
  for (auto oper : op->getOperands()) {
    Value castedOperand =
        getElementTypeOrSelf(oper.getType()).isBF16()
            ? castTo(rewriter, oper, /*targetElemType=*/f32Type)
            : oper;
    castedOperands.push_back(castedOperand);
  }

  auto newOp = createNewOp<Op>(rewriter, bf16Op, castedOperands);
  createF32ElementTypeOpRegion(newOp, rewriter);

  rewriter.setInsertionPointAfter(newOp);
  SmallVector<Value> castedResults;
  for (auto res : newOp->getResults()) {
    auto resType = getElementTypeOrSelf(res.getType());
    Value castedResult =
        resType.isF32() ? castTo(rewriter, res, /*targetElemType=*/bf16Type)
                        : res;
    castedResults.push_back(castedResult);
  }

  rewriter.replaceOp(op, castedResults);
}

template <typename Op>
static LogicalResult legalizeBF16(Op op, PatternRewriter &rewriter) {
  if (!shouldLegalizeBF16Op(op))
    return failure();

  createF32ElementTypeOp(op, rewriter);
  return success();
}

template <typename Op>
struct LegalizeBF16 : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    return legalizeBF16<Op>(op, rewriter);
  }
};

template <typename OpType>
static void registerOne(RewritePatternSet &patterns) {
  patterns.add<LegalizeBF16<OpType>>(patterns.getContext());
}

/// Variadic helper function.
template <typename... OpTypes>
static void registerAll(RewritePatternSet &patterns) {
  (registerOne<OpTypes>(patterns), ...);
}

void populateLegalizeBF16Pattern(RewritePatternSet &patterns) {
  registerAll<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
      >(patterns);
  registerAll<
#define GET_OP_LIST
#include "bishengir/Dialect/HFusion/IR/HFusionStructuredOps.cpp.inc"
      >(patterns);
  registerOne<hfusion::IsNanOp>(patterns);
  registerOne<hfusion::IsInfOp>(patterns);
  registerOne<hfusion::IsFiniteOp>(patterns);
  registerOne<hfusion::SortOp>(patterns);
  if (!isAscend950Arch) 
    registerOne<tensor::ConcatOp>(patterns);
  registerOne<tensor::PadOp>(patterns);
}

void LegalizeBF16Pass::runOnOperation() {
  ModuleOp moduleOp = getOperation()->getParentOfType<ModuleOp>();
  isAscend950Arch = hacc::utils::isAscend950(moduleOp);
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  populateLegalizeBF16Pattern(patterns);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<Pass> mlir::hfusion::createLegalizeBF16Pass() {
  return std::make_unique<LegalizeBF16Pass>();
}
