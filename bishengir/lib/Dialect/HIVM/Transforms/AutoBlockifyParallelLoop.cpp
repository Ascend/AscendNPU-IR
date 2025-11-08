//===- AutoBlockifyParallelLoop.cpp - Auto blockify loop ------------------===//
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
                         Value physicalBlockNum, Value logicBlockNum) {
  // block idx returns i64 meanwhile all other args are i32 so we cast it
  rewriter.setInsertionPointAfterValue(getBlockIdxOp);
  auto loc = getBlockIdxOp->getLoc();
  auto castedBlockID = rewriter.create<arith::TruncIOp>(
      loc, rewriter.getI32Type(), getBlockIdxOp.getResult());
  Value mulOp = rewriter.create<arith::MulIOp>(loc, iv, physicalBlockNum);
  auto sumOp = rewriter.create<arith::AddIOp>(loc, mulOp, castedBlockID);
  auto minVal = rewriter.create<arith::MinSIOp>(loc, sumOp, logicBlockNum);
  auto castedMinVal =
      rewriter.create<arith::ExtSIOp>(loc, rewriter.getI64Type(), minVal);
  rewriter.replaceAllUsesExcept(getBlockIdxOp, castedMinVal, castedBlockID);
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
        rewriter.setInsertionPointAfter(markOp);
        continue;
      }
    }
    if (!isa<func::ReturnOp>(op)) {
      opsToMove.push_back(&op);
    }
    if (isa<hivm::GetBlockIdxOp>(op))
      getBlockIdxOp = dyn_cast<hivm::GetBlockIdxOp>(op);
  }
  if (!logicBlockNum)
    return funcOp->emitError("Logical Block number not found");
  if (!getBlockIdxOp) {
    return success();
  }

  traceExceptions(logicBlockNum, exceptions);
  const int intBits = 32;
  Value lowerBound = rewriter.create<arith::ConstantIntOp>(loc, 0, intBits);
  auto kPhysicalBlockNum = getPhysicalBlockNum(funcOp);
  if (failed(kPhysicalBlockNum))
    return funcOp->emitError("Physical block num cannot be inferred");
  Value physicalBlockNum = rewriter.create<arith::ConstantIntOp>(
      loc, kPhysicalBlockNum.value(), intBits);
  Value upperBound =
      rewriter.create<arith::CeilDivSIOp>(loc, logicBlockNum, physicalBlockNum);
  Value step = rewriter.create<arith::ConstantIntOp>(loc, 1, intBits);
  auto forOp = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);

  Block *loopBody = forOp.getBody();
  Operation *yieldOp = loopBody->getTerminator();
  for (Operation *op : opsToMove) {
    if (exceptions.count(op) == 0) {
      op->moveBefore(yieldOp);
    }
  }
  replaceBlockIdUsers(rewriter, getBlockIdxOp, forOp.getInductionVar(),
                      physicalBlockNum, logicBlockNum);
  return success();
}
} // namespace

void AutoBlockifyParallelLoopPass::runOnOperation() {
  auto funOp = dyn_cast<func::FuncOp>(getOperation());
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
