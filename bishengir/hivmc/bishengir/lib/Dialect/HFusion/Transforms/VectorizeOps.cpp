//===- VectorizeOps.cpp - hfusion op vectorize ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Scope/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_VECTORIZEOPS
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hfusion-vectorize-ops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hfusion;

namespace {

unsigned getMaxElemBitWidth(linalg::LinalgOp op) {

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
computeDynamicVectorSizes(linalg::LinalgOp op, ArrayRef<int64_t> shape,
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

/// Computes vector sizes for a LinalgOp
///
/// Rules:
/// 1. Vector Length is 256 bytes (e.g., 64 elements for f32).
/// 2. If dynamic dims exist, other dims should be unit dim.
/// 3. Assign vector sizes for non-unit dims from right to left, the last
/// non-unit dim will expand to fill the whole vector length.
/// 4. Validates that static tile sizes do not exceed vector length.
FailureOr<SmallVector<int64_t>> computeVectorSizes(linalg::LinalgOp op) {
  SmallVector<int64_t> shape = op.getStaticLoopRanges();

  unsigned elemWidth = getMaxElemBitWidth(op);
  if (elemWidth <= 0) {
    return op.emitError("Failed to compute max element bit width");
  }
  int64_t elemWidthInBytes =
      llvm::divideCeil(elemWidth, mlir::utils::INTR_BITS_PER_BYTE);
  int64_t capacity = util::VL / elemWidthInBytes;
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

void markAttr(func::FuncOp funcOp, const std::string &attrName) {

  auto unitAttr = OpBuilder(funcOp->getContext()).getUnitAttr();
  if (!funcOp->hasAttr(attrName)) {
    funcOp->setAttr(attrName, unitAttr);
  }
  ModuleOp moduleOp = funcOp->getParentOfType<ModuleOp>();
  if (!moduleOp) {
    return;
  }

  auto maybeSymbolUses = funcOp.getSymbolUses(moduleOp);
  if (!maybeSymbolUses.has_value()) {
    return;
  }
  SymbolTable::UseRange uses = maybeSymbolUses.value();
  for (SymbolTable::SymbolUse use : uses) {
    func::CallOp callOp = dyn_cast<func::CallOp>(use.getUser());
    if (!callOp) {
      continue;
    }
    if (!callOp->hasAttr(attrName)) {
      callOp->setAttr(attrName, unitAttr);
    }
  }
}

struct HFusionVectorizeOpsPass
    : public impl::VectorizeOpsBase<HFusionVectorizeOpsPass> {
  using VectorizeOpsBase<HFusionVectorizeOpsPass>::VectorizeOpsBase;

  explicit HFusionVectorizeOpsPass(const VectorizeOpsOptions &options)
      : VectorizeOpsBase(options) {}

  void runOnOperation() override;
};

} // namespace

void HFusionVectorizeOpsPass::runOnOperation() {
  auto moduleOp = getOperation();
  SmallVector<func::FuncOp> funcList;
  moduleOp.walk([&](func::FuncOp funcOp) {
    if (forManualScope && !scope::utils::isManualVFScope(funcOp)) {
      return;
    }
    funcList.push_back(funcOp);
  });

  for (func::FuncOp funcOp : funcList) {
    LDBG("vectorizing func: " << funcOp.getSymName());
    auto result = funcOp.walk([&](linalg::LinalgOp linalgOp) {
      auto vectorSizesMaybe = computeVectorSizes(linalgOp);
      if (failed(vectorSizesMaybe)) {
        return WalkResult::interrupt();
      }
      SmallVector<int64_t> vectorSizes = vectorSizesMaybe.value();
      LDBG("vectorSizes: " << utils::debugger::to_string(vectorSizes));
      SmallVector<bool> scalableDims(vectorSizes.size(), false);
      IRRewriter rewriter(funcOp.getContext());
      if (failed(linalg::vectorize(rewriter, linalgOp, vectorSizes,
                                   scalableDims))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
    }

    markAttr(funcOp, "no_inline");
    markAttr(funcOp, "hivm.vector_function");
  }
}

std::unique_ptr<mlir::Pass> mlir::hfusion::createHFusionVectorizeOpsPass(
    const VectorizeOpsOptions &options) {
  return std::make_unique<HFusionVectorizeOpsPass>(options);
}
