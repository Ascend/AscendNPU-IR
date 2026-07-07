//===--------------------- OutlineVectorFunction.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/TransformOps/HFusionTransformOps.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir {
#define GEN_PASS_DEF_OUTLINEVECTORFUNCTION
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "outline-vector-function"

using namespace mlir;
using namespace mlir::hfusion;

namespace {
struct OutlineVectorFunctionPass
    : public impl::OutlineVectorFunctionBase<OutlineVectorFunctionPass> {
public:
  void runOnOperation() override;
};

static Value getOpTransformHandle(std::string label, OpBuilder &builder,
                                  transform::SequenceOp seqOp) {
  DictionaryAttr opAttr = builder.getDictionaryAttr(
      builder.getNamedAttr(label, builder.getUnitAttr()));
  Value opHandle =
      builder
          .create<transform::MatchOp>(
              seqOp.getLoc(), builder.getType<transform::AnyOpType>(),
              seqOp.getBodyBlock()->getArguments().front(), ArrayAttr(),
              transform::MatchInterfaceEnumAttr{}, opAttr, DictionaryAttr{},
              TypeAttr{}, ArrayAttr{})
          .getResults();
  return opHandle;
}

static bool loopHasOutlinedLoopAttr(Operation *op) {
  for (NamedAttribute attr : op->getAttrs()) {
    if (attr.getName().strref().starts_with("outlined-loop-target-")) {
      return true;
    }
  }
  return false;
}

static void outlineLoopsIntoVF(SmallVector<Value> &loopHandles, Location loc,
                               OpBuilder &builder, std::string vfName) {
  if (loopHandles.size() == 0)
    return;
  auto outlineOp = builder.create<transform::ExtendedLoopOutlineOp>(
      loc,
      SmallVector<Type>{builder.getType<transform::AnyOpType>(),
                        builder.getType<transform::AnyOpType>()},
      loopHandles, vfName);
  builder.create<transform::AnnotateOp>(loc, outlineOp.getFunction(),
                                        mlir::hivm::VectorFunctionAttr::name,
                                        nullptr);
  builder.create<transform::AnnotateOp>(loc, outlineOp.getFunction(),
                                        "no_inline", nullptr);
  builder.create<transform::AnnotateOp>(loc, outlineOp.getCall(),
                                        mlir::hivm::VectorFunctionAttr::name,
                                        nullptr);
  builder.create<transform::AnnotateOp>(loc, outlineOp.getCall(),
                                        "no_inline", nullptr);
  loopHandles.clear();
}

static void removeLabelsForOutlinedLoops(Operation *op) {
  for (NamedAttribute attr : op->getAttrs()) {
    if (attr.getName().strref().contains("outlined")) {
      op->removeAttr(attr.getName().strref());
    }
  }
}

static void duplicateReusedValuesForInnerSCFForOp(scf::ForOp loop,
                                                  OpBuilder &builder) {
  OpBuilder::InsertionGuard g(builder);
  DominanceInfo domInfo;
  auto loc = loop.getLoc();
  DenseSet<Value> set;
  loop.getBody()->walk([&](scf::ForOp innerLoop) {
    for (OpOperand &iterArg : innerLoop.getInitArgsMutable()) {
      Value src = iterArg.get();
      if (!src.getDefiningOp())
        continue;
      auto srcExtractOp = dyn_cast<tensor::ExtractSliceOp>(src.getDefiningOp());
      if (srcExtractOp)
        src = srcExtractOp.getSource();
      if (!src.getDefiningOp() || !domInfo.properlyDominates(src, loop))
        continue;
      if (!set.contains(src)) {
        auto ty = dyn_cast<RankedTensorType>(src.getType());
        if (!ty || !ty.hasStaticShape())
          continue;
        set.insert(src);
      } else {
        builder.setInsertionPoint(loop);
        auto ty = dyn_cast<RankedTensorType>(src.getType());
        Value empty = builder.create<tensor::EmptyOp>(loc, ty.getShape(),
                                                      ty.getElementType());
        auto copyOp = builder.create<linalg::CopyOp>(loc, src, empty);
        if (srcExtractOp) {
          builder.setInsertionPoint(srcExtractOp);
          auto newSrcExtractOp = builder.create<tensor::ExtractSliceOp>(
              loc, srcExtractOp.getType(), copyOp.getResult(0),
              srcExtractOp.getMixedOffsets(), srcExtractOp.getMixedSizes(),
              srcExtractOp.getMixedStrides());
          iterArg.assign(newSrcExtractOp.getResult());
        } else {
          iterArg.assign(copyOp.getResult(0));
        }
      }
    }
  });
}

void OutlineVectorFunctionPass::runOnOperation() {
  ModuleOp module = getOperation();
  OpBuilder builder(module.getContext());
  module.walk([&](func::FuncOp func) {
    func.walk([&](scf::ForOp forOp) {
      if (!loopHasOutlinedLoopAttr(forOp))
        return;
      duplicateReusedValuesForInnerSCFForOp(forOp, builder);
    });
    std::string vfNamePrefix = func.getSymName().str() + "_outlined_vf_";
    int vfIdx = -1;
    builder.setInsertionPointAfter(func);
    transform::SequenceOp seqOp = builder.create<transform::SequenceOp>(
        func.getLoc(), TypeRange(), transform::FailurePropagationMode::Propagate,
        builder.getType<transform::AnyOpType>(),
        [](OpBuilder &b, Location nested, Value rootH) {
          b.create<transform::YieldOp>(nested, ValueRange());
        });
    auto loc = seqOp.getLoc();
    builder.setInsertionPointToStart(seqOp.getBodyBlock());
    func.walk([&](Block *block) {
      int loopIdxInVf = -1;
      Operation *prevLoop = nullptr;
      SmallVector<Value> loopHandles;
      for (Operation &operation : block->getOperations()) {
        Operation *op = &operation;
        if (op->hasAttr("outlinedLoopWithFill")) {
          assert(isa<scf::ForOp>(op));
          outlineLoopsIntoVF(loopHandles, loc, builder,
                             vfNamePrefix + std::to_string(vfIdx));
          vfIdx++;
          std::string loopLabel = vfNamePrefix + std::to_string(vfIdx);
          op->setAttr(loopLabel, builder.getUnitAttr());
          loopHandles.push_back(getOpTransformHandle(loopLabel, builder, seqOp));
          outlineLoopsIntoVF(loopHandles, loc, builder,
                             vfNamePrefix + std::to_string(vfIdx));
          prevLoop = nullptr;
        } else if (loopHasOutlinedLoopAttr(op)) {
          assert(isa<scf::ForOp>(op));
          if (!prevLoop || op->getPrevNode() != prevLoop) {
            vfIdx++;
            loopIdxInVf = -1;
          }
          loopIdxInVf++;
          std::string loopLabel =
              vfNamePrefix + std::to_string(vfIdx) + "_" + std::to_string(loopIdxInVf);
          op->setAttr(loopLabel, builder.getUnitAttr());
          loopHandles.push_back(getOpTransformHandle(loopLabel, builder, seqOp));
          prevLoop = op;
        } else {
          outlineLoopsIntoVF(loopHandles, loc, builder,
                             vfNamePrefix + std::to_string(vfIdx));
          prevLoop = nullptr;
        }
      }
    });
    transform::TransformOptions options;
    options.enableExpensiveChecks(false);
    LogicalResult result = transform::applyTransformNamedSequence(
        func, seqOp, func->getParentOfType<ModuleOp>(), options);
    seqOp->erase();
    if (failed(result))
      func->emitError("Failed to outline vector function from loops of current function.");
  });

  module.walk([&](scf::ForOp forOp) {
    removeLabelsForOutlinedLoops(forOp);
  });
}

} // anonymous namespace

std::unique_ptr<Pass> mlir::hfusion::createOutlineVectorFunctionPass() {
  return std::make_unique<OutlineVectorFunctionPass>();
}
