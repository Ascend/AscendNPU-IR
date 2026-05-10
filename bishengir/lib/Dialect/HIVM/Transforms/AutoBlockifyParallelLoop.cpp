//===- AutoBlockifyParallelLoop.cpp - Auto blockify loop ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
#define GEN_PASS_DEF_AUTOBLOCKIFYPARALLELLOOP
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"

} // namespace mlir

#define DEBUG_TYPE "auto-blockify-parallel-loop"

using namespace mlir;
using namespace mlir::hivm;

static constexpr llvm::StringLiteral BlockifyLoopAttrName = "autoblockify.subloop";

namespace {
/// This pass will add a loop over the blocks when the logical block num is
/// larger than physical one.
///
/// for outer from 0,...,ceildiv(logical_block_dim,physical_block_dim)
/// 	 for inner from 0,...,physical_block_dim  <- get as block.idx
///    use(min(outer*physical_block_dim + inner, logical_block_dim))
struct AutoBlockifyParallelLoopPass
    : public impl::AutoBlockifyParallelLoopBase<AutoBlockifyParallelLoopPass> {
  using AutoBlockifyParallelLoopBase<
      AutoBlockifyParallelLoopPass>::AutoBlockifyParallelLoopBase;

public:
  void runOnOperation() override;
};

void traceExceptions(Value input, SmallPtrSet<Operation *, 4> &exceptions) {
  if (isa<BlockArgument>(input)) {
    return;
  }
  Operation *curOp = input.getDefiningOp();
  if (!curOp)
    return;
  exceptions.insert(curOp);
  for (auto opr : curOp->getOperands()) {
    traceExceptions(opr, exceptions);
  }
}

FailureOr<int> getPhysicalBlockNum(func::FuncOp funcOp) {
  auto maybeFuncCoreType = queryFuncCoreType(funcOp);
  if (!maybeFuncCoreType.has_value())
    return failure();
  TFuncCoreType funcCoreType = maybeFuncCoreType.value();
  auto moduleOp = funcOp->getParentOfType<ModuleOp>();
  auto maybeSpecInterface = hacc::utils::getNPUTargetSpec(moduleOp);
  if (funcCoreType == TFuncCoreType::AIC_OR_AIV ||
      !maybeSpecInterface.has_value())
    return failure();
  auto specInterface = maybeSpecInterface.value();
  auto aPhysicalBlockNum = (funcCoreType == TFuncCoreType::AIV)
                               ? specInterface.getSpecForIdentifierEnum(
                                     hacc::DeviceSpec::VECTOR_CORE_COUNT)
                               : specInterface.getSpecForIdentifierEnum(
                                     hacc::DeviceSpec::CUBE_CORE_COUNT);
  IntegerAttr castedAttr = cast<IntegerAttr>(aPhysicalBlockNum.getValue());
  int kPhysicalBlockNum = castedAttr.getValue().getSExtValue();
  return kPhysicalBlockNum;
}

void replaceBlockIdUsers(IRRewriter &rewriter,
                         hivm::GetBlockIdxOp getBlockIdxOp, Value iv,
                         Value logicBlockNum, Operation *castedBlockID) {
  // block idx returns i64 meanwhile all other args are i32 so we cast it
  rewriter.setInsertionPointAfterValue(iv);
  auto loc = getBlockIdxOp->getLoc();
  auto castedIV =
      rewriter.create<arith::ExtSIOp>(loc, rewriter.getI64Type(), iv);
  rewriter.replaceAllUsesExcept(getBlockIdxOp, castedIV, castedBlockID);
}

LogicalResult loopOnLogicBlock(func::FuncOp funcOp, IRRewriter &rewriter) {
  auto &entryBlock = funcOp.getBody().front();
  mlir::Location loc = entryBlock.front().getLoc();
  hivm::GetBlockIdxOp getBlockIdxOp;
  Value logicBlockNum;
  SmallPtrSet<Operation *, 4> exceptions;
  SmallVector<Operation *> opsToMove;
  for (auto &op : entryBlock) {
    if (auto markOp = dyn_cast<annotation::MarkOp>(op)) {
      if (markOp->hasAttr(kLogicalBlockNumAttr)) {
        logicBlockNum = markOp->getOperand(0);
        continue;
      }
    }
    if (!isa<func::ReturnOp>(op)) {
      opsToMove.push_back(&op);
    }
    if (isa<hivm::GetBlockIdxOp>(op)) {
      getBlockIdxOp = dyn_cast<hivm::GetBlockIdxOp>(op);
      rewriter.setInsertionPointAfter(getBlockIdxOp);
    }
  }
  if (!logicBlockNum)
    return funcOp->emitError("Logical Block number not found");
  if (!getBlockIdxOp) {
    return success();
  }

  traceExceptions(logicBlockNum, exceptions);
  exceptions.insert(getBlockIdxOp);
  const int intBits = 32;
  auto kPhysicalBlockNum = getPhysicalBlockNum(funcOp);
  if (failed(kPhysicalBlockNum))
    return funcOp->emitError("Physical block num cannot be inferred");
  Value physicalBlockNum = rewriter.create<arith::ConstantIntOp>(
      loc, kPhysicalBlockNum.value(), intBits);
  auto blockID = rewriter.create<arith::TruncIOp>(loc, rewriter.getI32Type(),
                                                  getBlockIdxOp);
  auto forOp = rewriter.create<scf::ForOp>(loc, blockID, logicBlockNum,
                                           physicalBlockNum);

  Block *loopBody = forOp.getBody();
  Operation *yieldOp = loopBody->getTerminator();
  for (Operation *op : opsToMove) {
    if (exceptions.count(op) == 0) {
      op->moveBefore(yieldOp);
    }
  }
  replaceBlockIdUsers(rewriter, getBlockIdxOp, forOp.getInductionVar(),
                      logicBlockNum, blockID);
  auto unit = UnitAttr::get(forOp->getContext());
  forOp->setAttr(BlockifyLoopAttrName, unit);
  return success();
}
} // namespace

void AutoBlockifyParallelLoopPass::runOnOperation() {
  func::FuncOp funOp = getOperation();
  if (!funOp) {
    return;
  }
  if (!hacc::utils::isDeviceEntry(funOp)) {
    return;
  }
  MLIRContext *ctx = funOp->getContext();
  IRRewriter rewriter(ctx);
  if (failed(loopOnLogicBlock(funOp, rewriter))) {
    return signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::hivm::createAutoBlockifyParallelLoopPass() {
  return std::make_unique<AutoBlockifyParallelLoopPass>();
}
