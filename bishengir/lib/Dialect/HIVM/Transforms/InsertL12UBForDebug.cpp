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
static constexpr llvm::StringLiteral alreadyInsertL12UB = "alreadyInsertL12UB";

namespace {
struct InsertL12UBForDebug
    : public impl::InsertL12UBForDebugBase<InsertL12UBForDebug> {
  using Base::Base;
  void runOnOperation() override;
};

void markUBAllocAlign(PatternRewriter &rewriter, Location loc, Value alloc,
                      ArrayRef<int64_t> shape) {
  SmallVector<int32_t> alignDims;
  SmallVector<int32_t> alignBytes{32};
  if (shape.size() == 2) {
    alignDims.push_back(1);
  } else if (shape.size() == 3) {
    alignDims.push_back(2);
  }
  auto markOp = rewriter.create<annotation::MarkOp>(loc, alloc);
  markOp->setAttr(StrideAlignDimsAttr::name,
                  rewriter.getDenseI32ArrayAttr(alignDims));
  markOp->setAttr(StrideAlignValueInByteAttr::name,
                  rewriter.getDenseI32ArrayAttr(alignBytes));
}

LogicalResult insertL12UBForOperand(PatternRewriter &rewriter,
                                    hivm::DebugOp debugOp) {
  Value operand = debugOp.getArg();
  Operation *definingOp = operand.getDefiningOp();
  auto resultTensorType = mlir::dyn_cast<RankedTensorType>(operand.getType());

  rewriter.setInsertionPointAfter(definingOp);
  auto elemType = resultTensorType.getElementType();
  auto shape = resultTensorType.getShape();
  MLIRContext *ctx = rewriter.getContext();
  Location loc = definingOp->getLoc();
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic) {
      return emitError(operand.getLoc())
             << "insertL12UBForOperand requires static shape, but got dynamic "
                "dimension in "
             << resultTensorType;
    }
  }

  // step1: Allocate the UB space
  auto ubSpaceAttr = hivm::AddressSpaceAttr::get(ctx, hivm::AddressSpace::UB);
  auto ubMemrefType =
      mlir::MemRefType::get(shape, elemType, /*layout=*/nullptr, ubSpaceAttr);
  auto noUbMemrefType = mlir::MemRefType::get(shape, elemType);
  Value alloc = rewriter.create<memref::AllocOp>(loc, ubMemrefType);
  Value noUbAlloc =
      rewriter.create<memref::MemorySpaceCastOp>(loc, noUbMemrefType, alloc);

  // step2: Due to hardware instruction constraints, the destination of l12ub
  // must be aligned
  markUBAllocAlign(rewriter, loc, alloc, shape);

  // step3: Call l12ub to move the data to the UB so that it can be printed
  // by device_print
  rewriter.create<hivm::L12UBOp>(loc, Type{}, operand, alloc);
  auto toTensor = rewriter.create<bufferization::ToTensorOp>(
      loc, resultTensorType, noUbAlloc, /*restrict*/true, /*writable*/true);

  // step4: Replace printed values of L1 device_print
  rewriter.modifyOpInPlace(debugOp, [&]() {
    debugOp.getArgMutable().assign(toTensor);
    debugOp.setTcoretypeAttr(hivm::TCoreTypeAttr::get(
        debugOp.getContext(), hivm::TCoreType::VECTOR));
    debugOp.setMemscopeAttr(hivm::AddressSpaceAttr::get(
        debugOp.getContext(), hivm::AddressSpace::UB));
  });
  return success();
}

/// Insert l12ub for the inputs of hivm::DebugOp.
struct InsertL12UBForDebugOpPattern : public OpRewritePattern<hivm::DebugOp> {
public:
  using OpRewritePattern<hivm::DebugOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hivm::DebugOp op,
                                PatternRewriter &rewriter) const override {
    auto memscope = op.getMemscope();
    bool isL1Memspace = memscope &&
        memscope->getAddressSpace() == hivm::AddressSpace::L1;
    if (!isL1Memspace || op->getAttr(alreadyInsertL12UB))
      return failure();

    auto printValue = op.getArg();
    auto resultTensorType =
        mlir::dyn_cast<RankedTensorType>(printValue.getType());
    if (!resultTensorType) {
      return failure();
    }
    if (printValue.getDefiningOp() == nullptr) {
      return failure();
    }

    ArrayRef<int64_t> shape = resultTensorType.getShape();
    assert(shape.size() == 2 || shape.size() == 3);
    return insertL12UBForOperand(rewriter, op);
  }
};

void InsertL12UBForDebug::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<InsertL12UBForDebugOpPattern>(patterns.getContext());
  (void)applyPatternsGreedily(getOperation(), std::move(patterns));
}

} // namespace

std::unique_ptr<Pass> mlir::hivm::createInsertL12UBForDebugPass() {
  return std::make_unique<InsertL12UBForDebug>();
}
