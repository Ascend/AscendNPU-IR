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
#include "mlir/Dialect/SCF/IR/SCF.h"
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

struct OutlinedCopyInfo {
  hivm::CopyOp copyOp;
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

static bool hasModificationBefore(hivm::CopyOp copyOp, Value funcArg) {
  if (copyOp->getBlock() != &copyOp->getParentRegion()->front())
    return true;

  for (Operation &op : *copyOp->getBlock()) {
    if (&op == copyOp.getOperation())
      break;
    if (mayWriteFuncArgOrAlias(&op, funcArg))
      return true;
  }
  return false;
}

static bool isSubviewValue(Value value) {
  return isa_and_nonnull<memref::SubViewOp>(value.getDefiningOp());
}

static bool canLowerToTransferPair(hivm::CopyOp copyOp) {
  // Only lower plain memref copies with no result tensor. The transfer pair
  // must be a semantic replacement for the original copy, not an
  // approximation.
  if (copyOp.getNumResults() != 0)
    return false;
  if (copyOp.getPadMode() || copyOp.getPadValue())
    return false;

  auto srcType = dyn_cast<MemRefType>(copyOp.getSource().getType());
  auto dstType = dyn_cast<MemRefType>(copyOp.getDst().getType());
  if (!srcType || !dstType || !srcType.hasStaticShape() || !dstType.hasStaticShape())
    return false;
  if (srcType.getRank() != 1 || dstType.getRank() != 1)
    return false;
  if (srcType.getShape() != dstType.getShape() ||
      srcType.getElementType() != dstType.getElementType())
    return false;
  return true;
}

static LogicalResult lowerCopyToTransferPair(hivm::CopyOp copyOp,
                                             IRRewriter &rewriter) {
  if (!canLowerToTransferPair(copyOp))
    return failure();

  rewriter.setInsertionPoint(copyOp);
  auto srcType = cast<MemRefType>(copyOp.getSource().getType());
  Location loc = copyOp.getLoc();
  int64_t numElems = srcType.getShape().front();
  int64_t maxElemsPerTransfer = utils::getNumPerRepeat(srcType.getElementType());
  if (maxElemsPerTransfer <= 0)
    return failure();

  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value padding =
      rewriter.create<arith::ConstantOp>(loc, srcType.getElementType(),
                                         rewriter.getZeroAttr(srcType.getElementType()));

  if (numElems <= maxElemsPerTransfer) {
    SmallVector<Value> indices{c0};
    SmallVector<bool> inBounds(1, true);
    auto vecType = VectorType::get(srcType.getShape(), srcType.getElementType());
    auto map = rewriter.getMultiDimIdentityMap(/*rank=*/1);
    Value read = rewriter.create<vector::TransferReadOp>(
        loc, vecType, copyOp.getSource(), indices, map, padding,
        /*mask=*/Value(), rewriter.getBoolArrayAttr(inBounds));
    auto write = rewriter.create<vector::TransferWriteOp>(
        loc, copyOp->getResultTypes(), read, copyOp.getDst(), indices, map,
        /*mask=*/Value(), rewriter.getBoolArrayAttr(inBounds));
    if (copyOp.getNumResults() == 1 && write->getNumResults() == 1)
      rewriter.replaceAllUsesWith(copyOp.getResult(0), write.getResult());
    rewriter.eraseOp(copyOp);
    return success();
  }

  Value cTotal = rewriter.create<arith::ConstantIndexOp>(loc, numElems);
  Value cStep = rewriter.create<arith::ConstantIndexOp>(loc, maxElemsPerTransfer);
  auto vecType = VectorType::get({maxElemsPerTransfer}, srcType.getElementType());
  auto maskType = VectorType::get({maxElemsPerTransfer}, rewriter.getI1Type());
  auto map = rewriter.getMultiDimIdentityMap(/*rank=*/1);
  auto inBounds = rewriter.getBoolArrayAttr({false});
  auto forOp = rewriter.create<scf::ForOp>(loc, c0, cTotal, cStep);
  rewriter.setInsertionPointToStart(forOp.getBody());
  Value iv = forOp.getInductionVar();
  Value remaining = rewriter.create<arith::SubIOp>(loc, cTotal, iv);
  Value needsTailMask =
      rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, remaining, cStep);
  Value chunkSize = rewriter.create<arith::SelectOp>(loc, needsTailMask, remaining, cStep);
  Value mask = rewriter.create<vector::CreateMaskOp>(loc, maskType, chunkSize);
  SmallVector<Value> indices{iv};
  Value read = rewriter.create<vector::TransferReadOp>(
      loc, vecType, copyOp.getSource(), indices, map, padding, mask, inBounds);
  rewriter.create<vector::TransferWriteOp>(
      loc, copyOp->getResultTypes(), read, copyOp.getDst(), indices, map, mask,
      inBounds);
  rewriter.setInsertionPointAfter(forOp);
  rewriter.eraseOp(copyOp);
  return success();
}

} // namespace

/// Temporary workaround for DTS2026030645981.
///
/// This pass runs on the module and scans all `hivm.hir.copy` ops in each VF
/// function body.
///
/// It handles three cases:
/// 1. Safe VF-entry argument-to-argument copies are outlined to caller-side
///    `hivm.hir.copy`.
/// 2. Copy-style ops that must stay in the callee, such as subview-based
///    copies or argument copies after earlier writes, are normalized to
///    `vector.transfer_read` + `vector.transfer_write` when that pair is
///    semantically identical to the original copy.
/// 3. All other copies are left unchanged.
///
/// The outline rewrite is limited to pure entry DMA with no prior writes to
/// either side.
///
/// Before:
///   func.func @vf(%arg0: memref<16xf32>, %arg1: memref<16xf32>)
///       attributes {hivm.vector_function, no_inline} {
///     hivm.hir.copy ins(%arg0 : memref<16xf32>)
///                   outs(%arg1 : memref<16xf32>)
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
/// If the copy cannot be outlined, but still behaves exactly like a plain copy,
/// it may be rewritten in place as:
///   %vec = vector.transfer_read %src[%c0, %c0], %padding
///   vector.transfer_write %vec, %dst[%c0, %c0]
///
/// The pass is intentionally conservative. If either argument has already been
/// written before the copy, the copy cannot be moved to the caller because that
/// would change the ordering relative to the earlier write.
void OutlineCopyInVFPass::runOnOperation() {
  auto moduleOp = getOperation();
  IRRewriter rewriter(moduleOp.getContext());

  moduleOp.walk([&](func::FuncOp funcOp) {
    if (!hivm::isVF(funcOp))
      return;

    SmallVector<OutlinedCopyInfo> outlinedCopies;
    SmallVector<hivm::CopyOp> copiesToLower;

    funcOp.walk([&](hivm::CopyOp copyOp) {
      auto srcArgNumber = getFuncArgNumber(copyOp.getSource(), funcOp);
      auto dstArgNumber = getFuncArgNumber(copyOp.getDst(), funcOp);
      bool operandsAreFuncArgs =
          srcArgNumber.has_value() && dstArgNumber.has_value();
      bool operandsFromSubviews =
          isSubviewValue(copyOp.getSource()) || isSubviewValue(copyOp.getDst());

      // Only outline direct function-argument copies that still behave like a
      // pure entry DMA. Other argument/subview copies stay in the VF and are
      // only normalized to vector transfers when the replacement is exactly
      // equivalent to the original hivm.hir.copy.
      if (operandsAreFuncArgs &&
          copyOp->getBlock() == &funcOp.getFunctionBody().front() &&
          !hasModificationBefore(copyOp, copyOp.getSource()) &&
          !hasModificationBefore(copyOp, copyOp.getDst())) {
        outlinedCopies.push_back({copyOp, *srcArgNumber, *dstArgNumber});
        return;
      }

      if (operandsAreFuncArgs || operandsFromSubviews)
        copiesToLower.push_back(copyOp);
    });

    for (hivm::CopyOp copyOp : copiesToLower) {
      if (!copyOp || failed(lowerCopyToTransferPair(copyOp, rewriter)))
        continue;
    }

    if (outlinedCopies.empty())
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
      for (const auto &outlinedCopy : outlinedCopies) {
        // Materialize the data movement at each call site so the VF body no
        // longer contains entry DMA between its argument buffers.
        rewriter.create<hivm::CopyOp>(
            callOp.getLoc(), TypeRange(),
            callOp.getOperand(outlinedCopy.srcArgNumber),
            callOp.getOperand(outlinedCopy.dstArgNumber));
      }
    }

    for (const auto &outlinedCopy : outlinedCopies)
      rewriter.eraseOp(outlinedCopy.copyOp);
  });
}

std::unique_ptr<Pass> mlir::hivm::createOutlineCopyInVFPass() {
  return std::make_unique<OutlineCopyInVFPass>();
}
