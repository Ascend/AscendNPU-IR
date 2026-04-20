//===--------- LegalizeBF16.cpp - Legalize BF16 type Pass -----------------===//
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

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/IR/HFusionImpl.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_LEGALIZEBF16PASS
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hfusion;

class LegalizeBF16Pass : public impl::LegalizeBF16PassBase<LegalizeBF16Pass> {
public:
  void runOnOperation() override;
};

static bool isBF16ElemType(Operation *op) {
  for (auto oper : op->getOperands()) {
    auto elemTy = getElementTypeOrSelf(oper.getType());
    if (isa<BFloat16Type>(elemTy))
      return true;
  }
  return false;
}

static void setFastMathContractAtttr(Operation *castOp) {
  auto fastMathAttr = arith::FastMathFlagsAttr::get(
    castOp->getContext(), arith::FastMathFlags::contract);
  castOp->setAttr(mlir::arith::FastMathFlagsAttr::name, fastMathAttr);
}

static bool shouldLegalizeBF16Op(Operation *op) {
  return !(isa<hfusion::CastOp>(op) || isa<linalg::FillOp>(op) ||
           isa<linalg::CopyOp>(op) || !isBF16ElemType(op) ||
           isa<linalg::MatmulOp>(op) || isa<linalg::BatchMatmulOp>(op) ||
           isa<linalg::TransposeOp>(op) || isa<linalg::BroadcastOp>(op) ||
           isa<hfusion::LoadOp>(op) || isa<hfusion::StoreOp>(op) ||
           isa<hfusion::BitcastOp>(op) || isa<hfusion::SelectOp>(op) ||
           isa<hfusion::Conv1DOp>(op));
}

template <typename Op>
static Operation *createNewOp(PatternRewriter &rewriter, Op bf16Op,
                              SmallVector<Value> &castedOperands) {
  IRMapping mapper;
  Operation *op = bf16Op.getOperation();
  for (const auto &[idx, oper] : llvm::enumerate(op->getOperands()))
    mapper.map(oper, castedOperands[idx]);

  auto *newOp = rewriter.cloneWithoutRegions(*op, mapper);
  auto *ctx = op->getContext();
  for (const auto &[idx, res] : llvm::enumerate(op->getResults())) {
    // Use dyn_cast to safely handle both ShapedType and scalar types
    if (auto shapedType = dyn_cast<ShapedType>(res.getType())) {
      if (getElementTypeOrSelf(shapedType).isBF16()) {
        auto newResTy = shapedType.clone(Float32Type::get(ctx));
        newOp->getResult(idx).setType(newResTy);
      }
    } else if (res.getType().isBF16()) {
      // Handle scalar BF16 types (e.g., from arith.addf with scalar operands)
      newOp->getResult(idx).setType(Float32Type::get(ctx));
    }
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
  for (auto &bodyOp : *block) {
    auto *newBodyOp = rewriter.clone(bodyOp, mapper);
    if (llvm::none_of(bodyOp.getResultTypes(),
                      [](Type type) { return type.isBF16(); })) {
      // skip if op has no bf16 result type to legalize
      continue;
    }
    if (!isa<linalg::YieldOp>(bodyOp) && !isa<arith::CmpFOp>(bodyOp) &&
        !isa<linalg::IndexOp>(bodyOp) && !isa<arith::IndexCastOp>(bodyOp) &&
        !isa<arith::CmpIOp>(bodyOp) && !isa<tensor::YieldOp>(bodyOp)) {
      newBodyOp->getResult(0).setType(Float32Type::get(ctx));
    }
  }
  return newOp;
}

template <typename Op>
static void createF32ElementTypeOpRegion(Op bf16Op, PatternRewriter &rewriter) {
  // Handle ops in region too
  for (size_t regionIndex = 0; regionIndex < bf16Op->getNumRegions();
       regionIndex++) {
    bf16Op->getRegion(regionIndex)
        .walk([&](Operation *opInRegion) -> WalkResult {
          if (!shouldLegalizeBF16Op(opInRegion))
            WalkResult::advance();
          for (auto operand : opInRegion->getOperands()) {
            if (isa<BlockArgument>(operand))
              continue;
            if (operand.getDefiningOp()->getParentOp() !=
                opInRegion->getParentOp()) {
              Value castedOperand =
                  getElementTypeOrSelf(operand.getType()).isBF16()
                      ? castTo(rewriter, operand,
                               /*targetElemType=*/rewriter.getF32Type())
                      : operand;
              if (Operation *castOp =
                      castedOperand.getDefiningOp<hfusion::CastOp>())
                setFastMathContractAtttr(castOp);
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
    if (Operation *castOp = castedOperand.getDefiningOp<hfusion::CastOp>())
      setFastMathContractAtttr(castOp);
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
    if (Operation *castOp = castedResult.getDefiningOp<hfusion::CastOp>())
      setFastMathContractAtttr(castOp);
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
  registerOne<tensor::ConcatOp>(patterns);
  registerOne<tensor::PadOp>(patterns);
  registerOne<arith::AddFOp>(patterns);
}

void LegalizeBF16Pass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  populateLegalizeBF16Pattern(patterns);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<Pass> mlir::hfusion::createLegalizeBF16Pass() {
  return std::make_unique<LegalizeBF16Pass>();
}
