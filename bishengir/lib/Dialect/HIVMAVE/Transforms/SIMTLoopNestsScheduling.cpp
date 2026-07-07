//===------------------------- SIMTLoopNestsScheduling.cpp ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
#define GEN_PASS_DEF_SIMTLOOPNESTSSCHEDULING
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "simt-loop-nests-scheduling"

using namespace mlir;
static const std::string fusionLabelPrefix = "loop_fusion_target_";

static bool isOpPreventingFusion(Operation *op) {
  return isa<memref::LoadOp, memref::StoreOp>(op) ||
         isa<hivm::HIVMStructuredOp>(op);
}

static void insertScheduleToFuseLoops(OpBuilder &builder,
                                      SmallVector<scf::ForOp> &loops,
                                      int &loopCount,
                                      transform::SequenceOp seqOp) {
  // Fuses loops into the bottom most loop
  std::string firstLabel = fusionLabelPrefix + std::to_string(++loopCount);
  loops.front()->setAttr(firstLabel, builder.getUnitAttr());

  DictionaryAttr firstOpAttr = builder.getDictionaryAttr(
      builder.getNamedAttr(firstLabel, builder.getUnitAttr()));
  Value prevHandle =
      builder
          .create<transform::MatchOp>(
              seqOp.getLoc(), builder.getType<transform::AnyOpType>(),
              seqOp.getBodyBlock()->getArguments().front(), ArrayAttr(),
              transform::MatchInterfaceEnumAttr{}, firstOpAttr,
              DictionaryAttr{}, TypeAttr{}, ArrayAttr{})
          .getResults();
  for (scf::ForOp *loopIter = loops.begin() + 1; loopIter < loops.end();
       ++loopIter) {
    std::string label = fusionLabelPrefix + std::to_string(++loopCount);
    loopIter->getOperation()->setAttr(label, builder.getUnitAttr());

    DictionaryAttr opAttr = builder.getDictionaryAttr(
        builder.getNamedAttr(label, builder.getUnitAttr()));
    Value curHandle =
        builder
            .create<transform::MatchOp>(
                seqOp.getLoc(), builder.getType<transform::AnyOpType>(),
                seqOp.getBodyBlock()->getArguments().front(), ArrayAttr(),
                transform::MatchInterfaceEnumAttr{}, opAttr, DictionaryAttr{},
                TypeAttr{}, ArrayAttr{})
            .getResults();

    prevHandle = builder.create<transform::LoopFuseSiblingOp>(
        seqOp.getLoc(), builder.getType<transform::AnyOpType>(), curHandle,
        prevHandle);
  }
  builder.create<transform::AnnotateOp>(
      seqOp.getLoc(), prevHandle, utils::kMapForToForallAttrName, nullptr);
}

namespace {

struct SIMTLoopNestsSchedulingPass
    : public impl::SIMTLoopNestsSchedulingBase<SIMTLoopNestsSchedulingPass> {
  using impl::SIMTLoopNestsSchedulingBase<
      SIMTLoopNestsSchedulingPass>::SIMTLoopNestsSchedulingBase;

public:
  void runOnOperation() override;

private:
  int loopCount = 0;
  LogicalResult
  tryFuseSIMTLoops(OpBuilder &builder, func::FuncOp func,
                   DenseMap<Operation *, SmallVector<scf::ForOp>> &simtLoops,
                   bool &changed);
};

} // namespace

LogicalResult SIMTLoopNestsSchedulingPass::tryFuseSIMTLoops(
    OpBuilder &builder, func::FuncOp func,
    DenseMap<Operation *, SmallVector<scf::ForOp>> &simtLoops, bool &changed) {
  builder.setInsertionPointAfter(func);
  // generate transform sequence op
  transform::SequenceOp seqOp = builder.create<transform::SequenceOp>(
      func.getLoc(), TypeRange(), transform::FailurePropagationMode::Propagate,
      builder.getType<transform::AnyOpType>(),
      [](OpBuilder &b, Location nested, Value rootH) {
        b.create<transform::YieldOp>(nested, ValueRange());
      });

  builder.setInsertionPointToStart(seqOp.getBodyBlock());

  // Match and fuse each group of
  for (auto &[_, loops] : simtLoops) {
    // Skip if there is only one loop in the scope
    if (loops.size() <= 1)
      continue;

    while (loops.size() >= 1) {
      // Starting from the back, fuse as many loops as possible
      SmallVector<scf::ForOp> loopsToBeFused{loops.back()};
      loops.pop_back();
      while (loops.size() >= 1) {
        scf::ForOp nextLoop = loops.back();
        scf::ForOp curLoop = loopsToBeFused.back();

        // Current condition to control fusion or not:
        // There has to be no hivm op and no memref.load/store between the two
        // loop here assume curLoop is after nextLoop in the IR
        Operation *iter = nextLoop->getNextNode();
        while (iter && !isOpPreventingFusion(iter) && iter != curLoop)
          iter = iter->getNextNode();

        if (iter == curLoop) {
          // can fuse these two
          loopsToBeFused.push_back(nextLoop);
          loops.pop_back();
        } else
          break;
      }
      if (loopsToBeFused.size() > 1) {
        changed = true;
        insertScheduleToFuseLoops(builder, loopsToBeFused, loopCount, seqOp);
      }
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Dumping kernel and schedule:\n");
  LLVM_DEBUG(func.dump());
  LLVM_DEBUG(seqOp.dump());

  LogicalResult result = transform::applyTransformNamedSequence(
      func, seqOp, func->getParentOfType<ModuleOp>(),
      transform::TransformOptions());

  seqOp->erase();
  return result;
}

void SIMTLoopNestsSchedulingPass::runOnOperation() {
  Operation *op = getOperation();
  MLIRContext *context = op->getContext();
  OpBuilder builder(context);

  LLVM_DEBUG(llvm::dbgs() << "Dumping kernel before scheduling:\n");
  LLVM_DEBUG(op->dump(););

  // get all functions
  SmallVector<func::FuncOp> funcList;
  op->walk([&](func::FuncOp func) {
    if (hacc::utils::isDevice(func) &&
        !func->hasAttr(hivm::VectorFunctionAttr::name)) {
      funcList.push_back(func);
    }
  });

  for (func::FuncOp func : funcList) {
    bool changed = true;

    while (changed) {
      changed = false;

      // Collect the SIMT loops, group them based on the scope
      DenseMap<Operation *, SmallVector<scf::ForOp>> simtLoops;
      func.walk([&](scf::ForOp op) {
        if (op->hasAttr(utils::kMapForToForallAttrName)) {
          if (simtLoops.find(op->getParentOp()) == simtLoops.end())
            simtLoops[op->getParentOp()] = SmallVector<scf::ForOp>();
          simtLoops[op->getParentOp()].push_back(op);
        }
      });
      if (failed(tryFuseSIMTLoops(builder, func, simtLoops, changed)))
        func->emitError("Failed to apply fusion schedule.");
    }
  }
}

std::unique_ptr<Pass> hivmave::createSIMTLoopNestsSchedulingPass() {
  return std::make_unique<SIMTLoopNestsSchedulingPass>();
}