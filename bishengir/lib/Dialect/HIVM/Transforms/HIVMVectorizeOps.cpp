//===- HIVMVectorizeOps.cpp - hivm op vectorize ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"
#include "bishengir/Dialect/HIVM/IR/HIVMVectorize.h"
#include "bishengir/Dialect/HIVM/Interfaces/VectorizableOpInterface.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Casting.h"

namespace mlir {
#define GEN_PASS_DEF_HIVMVECTORIZEOPS
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hivm-vectorize-ops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
using namespace mlir;
using namespace mlir::hivm;

namespace {

  unsigned getMaxElemBitWidth(hivm::HIVMStructuredOp op) {

  unsigned maxWidth = 0;
  for (Type type : op->getOperandTypes()) {
    Type elemTy = getElementTypeOrSelf(type);
    if (elemTy.isIndex()) {
      continue;
    }
    unsigned width = elemTy.getIntOrFloatBitWidth();
    maxWidth = std::max(maxWidth, width);
  }

  ModuleOp module = op->getParentOfType<ModuleOp>();
  if (hacc::utils::isAscend310B(module)) {
    // 300/310 does not support 64-bit types, using 32-bit instead
    return (maxWidth == 64) ? 32 : maxWidth;
  }
  return maxWidth;
}

std::optional<int64_t> getFirstNonUnitDim(ArrayRef<int64_t> shape) {
  if (shape.empty()) {
    return std::nullopt;
  }
  int64_t rank = static_cast<int64_t>(shape.size());
  for (int64_t i = 0; i < rank; ++i) {
    if (shape[i] > 1) {
      return i;
    }
  }
  return std::nullopt;
}

// When the shape is dynamic, we only allow one dynamic dim, and the other dims
// should be unit dims. We will assign `capacity` as vector size for this
// dynamic dim, and other dims have vector size of one.
FailureOr<SmallVector<int64_t>>
computeDynamicVectorSizes(HIVMStructuredOp op, ArrayRef<int64_t> shape,
                          int64_t capacity) {
  int64_t rank = static_cast<int64_t>(shape.size());
  SmallVector<int64_t> vectorSizes(rank, 1);
  int64_t nonUnitDims = 0;
  for (int64_t i = rank - 1; i >= 0; --i) {
    if (shape[i] == 1) {
      continue;
    }
    nonUnitDims++;
    if (nonUnitDims >= 2) {
      return op.emitError("Failed to compute dynamic vector sizes");
    }
    vectorSizes[i] = capacity;
  }
  return vectorSizes;
}

/// Computes vector sizes for a HIVM Vector Ops
///
/// Rules:
/// 1. Vector Length is 256 bytes (e.g., 64 elements for f32).
/// 2. If dynamic dims exist, other dims should be unit dim.
/// 3. Assign vector sizes for non-unit dims from right to left, the last
/// non-unit dim will expand to fill the whole vector length.
/// 4. Validates that static tile sizes do not exceed vector length.
FailureOr<SmallVector<int64_t>> computeVectorSizes(hivm::HIVMStructuredOp op) {
  SmallVector<int64_t> shape = op.computeStaticLoopSizes();

  unsigned elemWidth = getMaxElemBitWidth(op);
  if (elemWidth <= 0) {
    return op.emitError("Failed to compute max element bit width");
  }
  int64_t elemWidthInBytes =
      llvm::divideCeil(elemWidth, mlir::utils::INTR_BITS_PER_BYTE);
  int64_t capacity = hivm::util::VL / elemWidthInBytes;
  LDBG("op shape: " << utils::debugger::to_string(shape));
  LDBG("vector capacity: " << capacity);

  if (op.hasDynamicShape()) {
    return computeDynamicVectorSizes(op, shape, capacity);
  }

  int64_t rank = static_cast<int64_t>(shape.size());
  if (rank == 0) {
    return op.emitError("Empty shape: rank is zero");
  }
  auto first = getFirstNonUnitDim(shape);
  int64_t start = first.has_value() ? first.value() : rank - 1;
  int64_t end = rank - 1;
  int64_t remain = capacity;
  SmallVector<int64_t> vectorSizes(rank, 1);
  for (int64_t dim = end; dim >= start; dim--) {
    if (dim < 0 || static_cast<size_t>(dim) >= shape.size()) {
      return op.emitError("Invalid dimension index");
    }
    if (shape[dim] <= 0) {
      return op.emitError("Invalid shape dimension: must be positive");
    }
    if (shape[dim] > remain) {
      return op.emitError("Exceeds vector capacity");
    }
    if (dim == start) {
      vectorSizes[dim] = remain;
      continue;
    }
    vectorSizes[dim] = shape[dim];
    remain /= shape[dim];
  }
  return vectorSizes;
}

struct HIVMVectorizeOpsPattern
    : public OpInterfaceRewritePattern<mlir::hivm::VectorizableOpInterface> {
  using OpInterfaceRewritePattern<
      mlir::hivm::VectorizableOpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(mlir::hivm::VectorizableOpInterface op,
                                PatternRewriter &rewriter) const override {
    auto structuredOp = dyn_cast<HIVMStructuredOp>(op.getOperation());
    if (!structuredOp)
      return failure();
    FailureOr<SmallVector<int64_t>> vectorSizes = computeVectorSizes(structuredOp);
    if (failed(vectorSizes))
      return failure();
    return op.vectorize(rewriter, *vectorSizes);
  }
};

struct HIVMVectorizeOpsPass
    : public impl::HIVMVectorizeOpsBase<HIVMVectorizeOpsPass> {
  void runOnOperation() override;
};
} // namespace

void HIVMVectorizeOpsPass::runOnOperation() {
  auto funcOp = getOperation();
  if (!hivm::isVF(funcOp))
    return;
  RewritePatternSet patterns(&getContext());
  patterns.add<HIVMVectorizeOpsPattern>(&getContext());
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::hivm::createHIVMVectorizeOpsPass() {
  return std::make_unique<HIVMVectorizeOpsPass>();
}