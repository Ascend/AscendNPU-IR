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
#include "mlir/Pass/PassManager.h"
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

  LogicalResult preProcess();
};

} // namespace

namespace {

// Force a physical copy on memref to prevent the compiler from optimizing the
// implicit assignment for returning function argument, where the returned
// value/arg is binded with another buffer for memory reuse.
//
// simplified triton script:
//   m_i = m_i_buffer.to_tensor()
//   m_ij = m_ij_buffer.to_tensor()
//   with al.scope(vector_mode="simd")
//     m_i = m_ij
//   bl.to_buffer(m_i, bind_buffer=m_i_buffer)
//
// mlir:
//   func.func @vf(%arg0: tensor<64xf32>) -> tensor<64xf32> {
//       return %arg0 : tensor<64xf32>
//   }
//   %m_i = alloc()
//   %m_ij = alloc()
//   %m_i = call vf(%m_ij)
//   bind_buffer(m_i_buffer, m_ij_buffer)
//
// mlir after bind-buffer:
//   func.func @vf(%arg0: tensor<64xf32>) -> tensor<64xf32> {
//       return %arg0 : tensor<64xf32>
//   }
//   %m_i = %m_ij = alloc()
//   %m_i = call vf(%m_ij)
// m_i and m_ij are optimized to use the same buffer, which is not what we
// expected and will cause precision problem. The root cause is that we miss the
// assignment semantics for `m_i = m_ij`, which requires manually copy out.
//
// After:
//   func.func @vf(%arg0: tensor<64xf32>) -> tensor<64xf32> {
//     %mem_in = bufferization.to_memref %arg0 : memref<64xf32>
//     %mem_out = memref.alloc() : memref<64xf32>
//     linalg.copy ins(%mem_in : memref<64xf32>) outs(%mem_out : memref<64xf32>)
//     %res = bufferization.to_tensor %mem_out restrict writable
//     return %res : tensor<64xf32>
//   }
struct InsertCopyForReturnFuncArg : public OpRewritePattern<func::ReturnOp> {
  using OpRewritePattern<func::ReturnOp>::OpRewritePattern;

  explicit InsertCopyForReturnFuncArg(MLIRContext *context, bool forManualScope)
      : OpRewritePattern<func::ReturnOp>(context),
        forManualScope(forManualScope) {}

  LogicalResult matchAndRewrite(func::ReturnOp op,
                                PatternRewriter &rewriter) const override {
    auto funcOp = op->getParentOfType<func::FuncOp>();
    if (!funcOp) {
      return failure();
    }
    // The missing assignment problem should only exist in manual scope cases.
    // Add restriction to avoid potential performance decline in other cases.
    if (forManualScope && !scope::utils::isManualVFScope(funcOp)) {
      return failure();
    }

    SmallVector<OpOperand *> returnArgs = getReturnFuncArgs(op);
    if (returnArgs.empty()) {
      return failure();
    }
    Location loc = op->getLoc();
    for (OpOperand *operand : returnArgs) {
      Value arg = operand->get();
      ShapedType shapedType = dyn_cast<ShapedType>(arg.getType());
      if (!shapedType) {
        continue;
      }
      auto emptyOp = utils::createEmptyOp(rewriter, loc, arg);
      if (auto *definingOp = emptyOp.getDefiningOp()) {
        definingOp->setAttr(
            "__copy_id__",
            rewriter.getI64IntegerAttr(operand->getOperandNumber()));
      }
      auto memrefType =
          MemRefType::get(shapedType.getShape(), shapedType.getElementType());
      // copy out with bufferization to avoid folding the copy
      auto srcMemref =
          rewriter.create<bufferization::ToMemrefOp>(loc, memrefType, arg);
      auto dstMemref =
          rewriter.create<bufferization::ToMemrefOp>(loc, memrefType, emptyOp);
      rewriter.create<linalg::CopyOp>(loc, ValueRange{srcMemref},
                                      ValueRange{dstMemref});
      auto returnTensor = rewriter.create<bufferization::ToTensorOp>(
          loc, dstMemref,
          /*restrict*/ true, /*writable*/ true);

      rewriter.modifyOpInPlace(op, [&]() {
        op.setOperand(operand->getOperandNumber(), returnTensor);
      });
    }
    return success();
  }

private:
  SmallVector<OpOperand *> getReturnFuncArgs(func::ReturnOp op) const {
    SmallVector<OpOperand *> args;
    for (auto &operand : op->getOpOperands()) {
      auto blockArg = dyn_cast<BlockArgument>(operand.get());
      if (!blockArg || !blockArg.getOwner()->isEntryBlock()) {
        continue;
      }
      args.push_back(&operand);
    }
    return args;
  }

  bool forManualScope{false};
};
} // namespace

LogicalResult HFusionVectorizeOpsPass::preProcess() {
  auto moduleOp = getOperation();
  MLIRContext *ctx = moduleOp.getContext();
  PassManager pm(ctx);
  RewritePatternSet patterns(ctx);
  patterns.add<InsertCopyForReturnFuncArg>(ctx, forManualScope);
  if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
    return moduleOp.emitError("fail to preprocess");
  }
  return success();
}

void HFusionVectorizeOpsPass::runOnOperation() {
  if (failed(preProcess())) {
    signalPassFailure();
    return;
  }
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