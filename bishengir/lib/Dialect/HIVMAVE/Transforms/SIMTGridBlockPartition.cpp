//===--------- SIMTGridBlockPartition.cpp - SIMT Loop Tile pass -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h"
#include "bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrins.h"
#include "bishengir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/Support/Debug.h"
#include <cassert>

#define DEBUG_TYPE "simt-grid-block-partition"

namespace mlir {
#define GEN_PASS_DEF_SIMTGRIDBLOCKPARTITION
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

static bool opCanBeLoopTiled(scf::ParallelOp op) {
  if (isa<scf::ParallelOp>(op->getParentOp()))
    return false; // Inner nested ParallelOp should be ignored
  return true;
}

static void findParallelOps(func::FuncOp func,
                            SmallVector<scf::ParallelOp> &pLoops) {
  func.walk([&](scf::ParallelOp op) {
    if (opCanBeLoopTiled(op)) {
      pLoops.push_back(op);
    }
  });
}

static SmallVector<int64_t> computeTileSizes(scf::ParallelOp op) {
  SmallVector<int64_t> tileSizes;
  for (auto s : op.getUpperBound()) {
    auto cstValue = dyn_cast<arith::ConstantOp>(s.getDefiningOp());
    assert(cstValue && "currently don't support loops with dynamic bound!");
    std::optional<int64_t> cstInt = getConstantIntValue(cstValue.getValue());
    assert(cstInt.has_value());
    tileSizes.push_back(*cstInt);
  }
  return tileSizes;
}

static const std::string parallelLabelPrefix =
    "simt-grid-block-partition-target-";

class SIMTGridBlockPartition
    : public impl::SIMTGridBlockPartitionBase<SIMTGridBlockPartition> {
public:
  explicit SIMTGridBlockPartition() : SIMTGridBlockPartitionBase() {}
  void runOnOperation() override;

private:
  int countParallelOps = 0;

  LogicalResult
  buildAndApplyLoopTileSchedule(OpBuilder &builder, func::FuncOp func,
                                SmallVector<scf::ParallelOp> &pLoops);
};

LogicalResult SIMTGridBlockPartition::buildAndApplyLoopTileSchedule(
    OpBuilder &builder, func::FuncOp func,
    SmallVector<scf::ParallelOp> &pLoops) {
  builder.setInsertionPointAfter(func);
  // generate transform sequence op
  transform::SequenceOp seqOp = builder.create<transform::SequenceOp>(
      func.getLoc(), TypeRange(), transform::FailurePropagationMode::Propagate,
      builder.getType<transform::AnyOpType>(),
      [](OpBuilder &b, Location nested, Value rootH) {
        b.create<transform::YieldOp>(nested, ValueRange());
      });

  builder.setInsertionPointToStart(seqOp.getBodyBlock());

  // Match and vectorize each scf::parallel op
  for (auto pLoop : pLoops) {

    // name the parallel op uniquely so that it can be matched later
    std::string label = parallelLabelPrefix + std::to_string(countParallelOps);
    pLoop->setAttr(label, UnitAttr::get(&getContext()));
    countParallelOps++;

    // compute the tileSizes
    SmallVector<int64_t> tileSizes = computeTileSizes(pLoop);

    DictionaryAttr opAttr = builder.getDictionaryAttr(
        builder.getNamedAttr(label, builder.getUnitAttr()));

    Value parallelOpHandle =
        builder
            .create<transform::MatchOp>(
                seqOp.getLoc(), builder.getType<transform::AnyOpType>(),
                seqOp.getBodyBlock()->getArguments().front(), ArrayAttr(),
                transform::MatchInterfaceEnumAttr{}, opAttr, DictionaryAttr{},
                TypeAttr{}, ArrayAttr{})
            .getResults();

    // Tile the op into compatible vector sizes first
    SmallVector<Type> resultTypes(2, builder.getType<transform::AnyOpType>());
    builder.create<transform::ParallelLoopTileOp>(
        seqOp.getLoc(), resultTypes, parallelOpHandle,
        builder.getI64ArrayAttr(tileSizes));
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

void SIMTGridBlockPartition::runOnOperation() {
  // get all functions
  SmallVector<func::FuncOp> funcList;
  getOperation()->walk([&](func::FuncOp func) {
    if (hacc::utils::isDevice(func))
      funcList.push_back(func);
  });
  for (func::FuncOp func : funcList) {
    SmallVector<scf::ParallelOp> pLoops;
    findParallelOps(func, pLoops);
    auto *ctx = &getContext();
    // Generate transform ops to tile each recorded scf::ParallelOp
    OpBuilder builder(ctx);
    // Apply the transforms in one shot
    if (failed(buildAndApplyLoopTileSchedule(builder, func, pLoops)))
      func->emitError("Failed to tile the parallel loops.");
    else {
      auto kernelAttr = StringAttr::get(
          ctx, mlir::hivm_regbaseintrins::kDavinciKernelAttrName);
      func->setAttr(kernelAttr, builder.getUnitAttr());
      auto targetAttr = StringAttr::get(
          ctx, mlir::hivm_regbaseintrins::kDavinciTargetAttrName);
      func->setAttr(targetAttr, mlir::hivm_regbaseintrins::SIMT_TargetAttr::get(
                                    ctx, "dav-c310"));
    }
  }
}

} // namespace

std::unique_ptr<Pass> mlir::hivmave::createSIMTGridBlockPartitionPass() {
  return std::make_unique<SIMTGridBlockPartition>();
}
