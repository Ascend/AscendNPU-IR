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
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
  if (!isa<BaseMemRefType>(value.getType()))
    return false;
  return utils::tracebackMemRef(value) == funcArg;
}

static bool mayWriteFuncArgOrAlias(Operation *op, Value funcArg) {
  if (auto memEffects = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance> effects;
    memEffects.getEffects(effects);
    for (const MemoryEffects::EffectInstance &effect : effects) {
      if (!isa<MemoryEffects::Write>(effect.getEffect()))
        continue;
      if (aliasesFuncArg(effect.getValue(), funcArg))
        return true;
    }
    return false;
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

} // namespace

/// Temporary workaround for DTS2026030645981.
///
/// Outline VF entry copies from callee to caller when the load is a pure
/// argument-to-argument DMA with no prior writes to either side.
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
/// The pass is intentionally conservative. If either argument has already been
/// written before the load, the load remains inside the VF because moving it to
/// the caller would change the ordering relative to that earlier write.
void OutlineCopyInVFPass::runOnOperation() {
  auto moduleOp = getOperation();
  IRRewriter rewriter(moduleOp.getContext());
  for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
    if (!hivm::isVF(funcOp))
      continue;

    SmallVector<OutlinedLoadInfo> outlinedLoads;
    funcOp.walk([&](hivm::LoadOp loadOp) {
      if (loadOp->getBlock() != &funcOp.getFunctionBody().front())
        return;

      // Only outline direct function-argument copies. If the load already uses
      // a subview/reinterpret-cast/result value, its caller-side equivalent is
      // not guaranteed to be representable as a plain hivm.hir.copy.
      auto srcArgNumber = getFuncArgNumber(loadOp.getSource(), funcOp);
      auto dstArgNumber = getFuncArgNumber(loadOp.getDst(), funcOp);
      if (!srcArgNumber.has_value() || !dstArgNumber.has_value())
        return;

      // Reject cases such as:
      //   %subview = memref.subview %arg1[0] [1] [1] : ...
      //   vector.transfer_write %v, %subview[%c0] ...
      //   hivm.hir.load ins(%arg0) outs(%arg1)
      // because %arg1 has already been modified before the load.
      if (hasModificationBefore(loadOp, loadOp.getSource()) ||
          hasModificationBefore(loadOp, loadOp.getDst()))
        return;
      outlinedLoads.push_back({loadOp, *srcArgNumber, *dstArgNumber});
    });
    if (outlinedLoads.empty())
      continue;

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

    for (const auto &outlinedLoad : outlinedLoads) {
      rewriter.eraseOp(outlinedLoad.loadOp);
    }
  }
}

std::unique_ptr<Pass> mlir::hivm::createOutlineCopyInVFPass() {
  return std::make_unique<OutlineCopyInVFPass>();
}
