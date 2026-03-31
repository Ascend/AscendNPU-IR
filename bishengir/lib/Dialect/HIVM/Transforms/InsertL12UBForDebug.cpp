//===--------------------- InsertL12UBForDebug.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass inserts the l12ub op for debug.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

namespace mlir {
#define GEN_PASS_DEF_INSERTL12UBFORDEBUG
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

#define DEBUG_TYPE "hivm-insert-l12ub-for-debug"

namespace {
struct InsertL12UBForDebug
    : public impl::InsertL12UBForDebugBase<InsertL12UBForDebug> {
  using Base::Base;
  void runOnOperation() override;
};

/// Insert l12ub for the inputs of hivm::MmadL1Op.
struct InsertL12UBForDebugPattern : public OpRewritePattern<hivm::MmadL1Op> {
public:
  using OpRewritePattern<hivm::MmadL1Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(hivm::MmadL1Op op,
                                PatternRewriter &rewriter) const override {
    llvm::SmallVector<Value> l1values = {op.getA(), op.getB()};
    bool allInserted = true;
    for (Value val : l1values) {
      if (!isa<TensorType>(val.getType())) {
        // currently only support tensors
        continue;
      }
      if (val.getDefiningOp() == nullptr) {
        // currently only support MmadL1Op inputs with defining op
        continue;
      }
      Operation *definingOp = val.getDefiningOp();
      bool inserted = false;
      for (Operation *user : val.getUsers()) {
        if (isa<hivm::L12UBOp>(user)) {
          inserted = true;
          break;
        }
      }
      if (inserted) {
        continue;
      }
      allInserted = false;
      for (Operation *user : val.getUsers()) {
        if (isa<hivm::DebugOp>(user)) {
          hivm::DebugOp debugOp = cast<hivm::DebugOp>(user);
          rewriter.setInsertionPointAfter(definingOp);

          auto resultTensorType =
              mlir::dyn_cast<RankedTensorType>(val.getType());
          if (!resultTensorType)
            continue;

          auto elemType = resultTensorType.getElementType();
          auto shape = resultTensorType.getShape();
          MLIRContext *ctx = rewriter.getContext();
          auto ubSpaceAttr =
              hivm::AddressSpaceAttr::get(ctx, hivm::AddressSpace::UB);
          auto ubMemrefType = mlir::MemRefType::get(
              shape, elemType, /*layout=*/nullptr, ubSpaceAttr);
          auto noUbMemrefType = mlir::MemRefType::get(shape, elemType);
          Location defLoc = definingOp->getLoc();

          Value alloc = rewriter.create<memref::AllocOp>(defLoc, ubMemrefType);
          Value noUb = rewriter.create<memref::MemorySpaceCastOp>(
              defLoc, noUbMemrefType, alloc);

          auto newFixpipeOp = rewriter.create<hivm::L12UBOp>(defLoc, Type{},
                                                             /*src=*/val,
                                                             /*dst=*/alloc);
          rewriter.setInsertionPointAfter(newFixpipeOp);
          auto toTensor = rewriter.create<bufferization::ToTensorOp>(
              defLoc, resultTensorType, noUb,
              /*restrict=*/true,
              /*writable=*/true);

          rewriter.modifyOpInPlace(debugOp, [&]() {
            OpOperand &arg = debugOp.getArgMutable();
            arg.assign(toTensor.getResult());
          });
        }
      }
    }
    return allInserted ? failure() : success();
  }
};

void InsertL12UBForDebug::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<InsertL12UBForDebugPattern>(patterns.getContext());
  (void)applyPatternsGreedily(getOperation(), std::move(patterns));
}

} // namespace

std::unique_ptr<Pass> mlir::hivm::createInsertL12UBForDebugPass() {
  return std::make_unique<InsertL12UBForDebug>();
}
