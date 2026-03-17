//===- OutlineCopyInVF.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "hivm-outline-copy-in-VF"

namespace mlir {
#define GEN_PASS_DEF_OUTLINECOPYINVF
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace hivm;

namespace {

struct OutlineCopyInVFPass
    : public mlir::impl::OutlineCopyInVFBase<OutlineCopyInVFPass> {
public:
  void runOnOperation() override;
};

struct OutlinedLoadInfo {
  hivm::LoadOp loadOp;
  unsigned srcArgNumber;
  unsigned dstArgNumber;
};

static std::optional<unsigned> getFuncArgNumber(Value value, func::FuncOp funcOp) {
  auto blockArg = dyn_cast<BlockArgument>(value);
  if (!blockArg || blockArg.getOwner() != &funcOp.getFunctionBody().front())
    return std::nullopt;
  return blockArg.getArgNumber();
}

static bool aliasesFuncArg(Value value, Value funcArg) {
  if (!value)
    return false;
  if (!isa<BaseMemRefType>(value.getType()))
    return false;
  return utils::tracebackMemRef(value) == funcArg;
}

static bool mayWriteFuncArgOrAlias(Operation *op, Value funcArg) {
  auto effects = getEffectsRecursively(op);
  if (!effects.has_value()) {
    // Unknown-effect ops are not safe to move across. Even if the top-level op
    // has no aliasing operands, a nested region may capture funcArg from above
    // and write through that capture.
    return true;
  }

  for (const MemoryEffects::EffectInstance &effect : *effects) {
    if (!isa<MemoryEffects::Write>(effect.getEffect()))
      continue;
    if (!effect.getValue())
      return true;
    if (aliasesFuncArg(effect.getValue(), funcArg))
      return true;
  }
  return false;
}

static bool hasModificationBefore(hivm::LoadOp loadOp, Value funcArg) {
  if (loadOp->getBlock() != &loadOp->getParentRegion()->front())
    return true;

  for (Operation &op : *loadOp->getBlock()) {
    if (&op == loadOp.getOperation())
      break;
    if (mayWriteFuncArgOrAlias(&op, funcArg))
      return true;
  }
  return false;
}

static bool isSubviewValue(Value value) {
  return isa_and_nonnull<memref::SubViewOp>(value.getDefiningOp());
}

static bool canLowerToTransferPair(hivm::LoadOp loadOp) {
  // Only lower memref DMA with no result tensor. The transfer pair must be a
  // semantic replacement for the original load, not an approximation.
  // `eviction_policy` is intentionally ignored here because the vector
  // transfer pair has no equivalent cache hint encoding.
  if (loadOp.getNumResults() != 0)
    return false;
  if (loadOp.getPadMode() || loadOp.getPadValue() || loadOp.getRightPaddingNum())
    return false;
  if (loadOp.getLeftPaddingNum() &&
      !isConstantIntValue(OpFoldResult(loadOp.getLeftPaddingNum()), 0))
    return false;
  if (loadOp.getInitOutBuffer() || loadOp.getInitCondition())
    return false;

  auto srcType = dyn_cast<MemRefType>(loadOp.getSource().getType());
  auto dstType = dyn_cast<MemRefType>(loadOp.getDst().getType());
  if (!srcType || !dstType || !srcType.hasStaticShape() || !dstType.hasStaticShape())
    return false;
  if (srcType.getShape() != dstType.getShape() ||
      srcType.getElementType() != dstType.getElementType())
    return false;
  return true;
}

static LogicalResult lowerLoadToTransferPair(hivm::LoadOp loadOp,
                                             IRRewriter &rewriter) {
  if (!canLowerToTransferPair(loadOp))
    return failure();

  rewriter.setInsertionPoint(loadOp);
  auto srcType = cast<MemRefType>(loadOp.getSource().getType());
  Location loc = loadOp.getLoc();
  int64_t rank = srcType.getRank();
  SmallVector<Value> indices(rank, rewriter.create<arith::ConstantIndexOp>(loc, 0));
  SmallVector<bool> inBounds(rank, true);
  auto vecType = VectorType::get(srcType.getShape(), srcType.getElementType());
  auto map = rewriter.getMultiDimIdentityMap(rank);
  Value padding =
      rewriter.create<arith::ConstantOp>(loc, srcType.getElementType(),
                                         rewriter.getZeroAttr(srcType.getElementType()));
  Value read = rewriter.create<vector::TransferReadOp>(
      loc, vecType, loadOp.getSource(), indices, map, padding,
      /*mask=*/Value(), rewriter.getBoolArrayAttr(inBounds));
  auto write = rewriter.create<vector::TransferWriteOp>(
      loc, loadOp->getResultTypes(), read, loadOp.getDst(), indices, map,
      /*mask=*/Value(), rewriter.getBoolArrayAttr(inBounds));
  if (loadOp.getNumResults() == 1 && write->getNumResults() == 1)
    rewriter.replaceAllUsesWith(loadOp.getResult(0), write.getResult());
  rewriter.eraseOp(loadOp);
  return success();
}

} // namespace

/// Temporary workaround for DTS2026030645981.
///
/// This pass runs on the module and scans all `hivm.hir.load` ops in each VF
/// function body.
///
/// It handles three cases:
/// 1. Safe VF-entry argument-to-argument loads are outlined to caller-side
///    `hivm.hir.copy`.
/// 2. Copy-style loads that must stay in the callee, such as subview-based
///    loads or argument loads after earlier writes, are normalized to
///    `vector.transfer_read` + `vector.transfer_write` when that pair is
///    semantically identical to the original load.
/// 3. All other loads are left unchanged.
///
/// The outline rewrite is limited to pure entry DMA with no prior writes to
/// either side.
///
/// Before:
///   func.func @vf(%arg0: memref<16xf32>, %arg1: memref<16xf32>)
///       attributes {hivm.vector_function, no_inline} {
///     hivm.hir.load ins(%arg0 : memref<16xf32>)
///                   outs(%arg1 : memref<16xf32>) eviction_policy = <EvictFirst>
///     ...
///   }
///   func.func @caller(%src: memref<16xf32>, %dst: memref<16xf32>) {
///     call @vf(%src, %dst) {hivm.vector_function, no_inline}
///     ...
///   }
///
/// After:
///   func.func @vf(%arg0: memref<16xf32>, %arg1: memref<16xf32>)
///       attributes {hivm.vector_function, no_inline} {
///     ...
///   }
///   func.func @caller(%src: memref<16xf32>, %dst: memref<16xf32>) {
///     hivm.hir.copy ins(%src : memref<16xf32>) outs(%dst : memref<16xf32>)
///     call @vf(%src, %dst) {hivm.vector_function, no_inline}
///     ...
///   }
///
/// If the load cannot be outlined, but still behaves exactly like a plain copy,
/// it may be rewritten in place as:
///   %vec = vector.transfer_read %src[%c0, %c0], %padding
///   vector.transfer_write %vec, %dst[%c0, %c0]
///
/// The pass is intentionally conservative. If either argument has already been
/// written before the load, the load cannot be moved to the caller because that
/// would change the ordering relative to the earlier write.
void OutlineCopyInVFPass::runOnOperation() {
  auto moduleOp = getOperation();
  IRRewriter rewriter(moduleOp.getContext());

  moduleOp.walk([&](func::FuncOp funcOp) {
    if (!hivm::isVF(funcOp))
      return;

    SmallVector<OutlinedLoadInfo> outlinedLoads;
    SmallVector<hivm::LoadOp> loadsToLower;

    funcOp.walk([&](hivm::LoadOp loadOp) {
      auto srcArgNumber = getFuncArgNumber(loadOp.getSource(), funcOp);
      auto dstArgNumber = getFuncArgNumber(loadOp.getDst(), funcOp);
      bool operandsAreFuncArgs =
          srcArgNumber.has_value() && dstArgNumber.has_value();
      bool operandsFromSubviews =
          isSubviewValue(loadOp.getSource()) || isSubviewValue(loadOp.getDst());

      // Only outline direct function-argument copies that still behave like a
      // pure entry DMA. Other argument/subview loads stay in the VF and are
      // only normalized to vector transfers when the replacement is exactly
      // equivalent to the original hivm.hir.load.
      if (operandsAreFuncArgs &&
          loadOp->getBlock() == &funcOp.getFunctionBody().front() &&
          !hasModificationBefore(loadOp, loadOp.getSource()) &&
          !hasModificationBefore(loadOp, loadOp.getDst())) {
        outlinedLoads.push_back({loadOp, *srcArgNumber, *dstArgNumber});
        return;
      }

      if (operandsAreFuncArgs || operandsFromSubviews)
        loadsToLower.push_back(loadOp);
    });

    for (hivm::LoadOp loadOp : loadsToLower) {
      if (!loadOp || failed(lowerLoadToTransferPair(loadOp, rewriter)))
        continue;
    }

    if (outlinedLoads.empty())
      return;

    auto callSites = funcOp.getSymbolUses(moduleOp);
    if (!callSites.has_value()) {
      funcOp.emitError("expected outlined VF to have valid symbol uses");
      signalPassFailure();
      return;
    }
    for (SymbolTable::SymbolUse callSite : callSites.value()) {
      auto callOp = dyn_cast<func::CallOp>(callSite.getUser());
      if (!callOp)
        continue;
      rewriter.setInsertionPoint(callOp);
      for (const auto &outlinedLoad : outlinedLoads) {
        // Materialize the data movement at each call site so the VF body no
        // longer contains entry DMA between its argument buffers.
        rewriter.create<hivm::CopyOp>(
            callOp.getLoc(), TypeRange(),
            callOp.getOperand(outlinedLoad.srcArgNumber),
            callOp.getOperand(outlinedLoad.dstArgNumber));
      }
    }

    for (const auto &outlinedLoad : outlinedLoads)
      rewriter.eraseOp(outlinedLoad.loadOp);
  });
}

std::unique_ptr<Pass> mlir::hivm::createOutlineCopyInVFPass() {
  return std::make_unique<OutlineCopyInVFPass>();
}
