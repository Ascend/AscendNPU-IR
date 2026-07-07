//===----------------------- RemoveCopyOps.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <cassert>

#define DEBUG_TYPE "hivm-remove-copy-ops"

namespace mlir {
#define GEN_PASS_DEF_REMOVECOPYOPS
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

struct RemoveCopyOpsPass : public impl::RemoveCopyOpsBase<RemoveCopyOpsPass> {
  using RemoveCopyOpsBase<RemoveCopyOpsPass>::RemoveCopyOpsBase;

public:
  void runOnOperation() override;
};
} // namespace
using llvm::dbgs;

static bool copyUB2UB(Value &src, Value &dst) {
  if (hivm::getHIVMAddressSpace(src.getType()) == hivm::AddressSpace::UB &&
      hivm::getHIVMAddressSpace(dst.getType()) == hivm::AddressSpace::UB) {
    return true;
  } else
    return false;
}

static Operation *traceUpForAlloc(Value &v) {
  if (isa<BlockArgument>(v)) {
    return nullptr;
  }
  Operation *definingOp = v.getDefiningOp();
  while (true) {
    if (isa<memref::AllocOp>(definingOp)) {
      return definingOp;
    }
    if (auto viewOp = dyn_cast<memref::ViewOp>(definingOp)) {
      definingOp = viewOp.getSource().getDefiningOp();
      continue;
    } else if (auto subviewOp = dyn_cast<memref::SubViewOp>(definingOp)) {
      definingOp = subviewOp.getSource().getDefiningOp();
      continue;
    } else if (auto castOp = dyn_cast<memref::CastOp>(definingOp)) {
      definingOp = castOp.getSource().getDefiningOp();
      continue;
    } else if (auto collapseOp =
                   dyn_cast<memref::CollapseShapeOp>(definingOp)) {
      definingOp = collapseOp.getSrc().getDefiningOp();
      continue;
    } else {
      return nullptr;
    }
  }
}

static bool traceUpForMatchingDimensionSource(Value &opnd, Value &src) {
  auto opndType = mlir::dyn_cast<MemRefType>(opnd.getType());
  assert(opndType && "must be MemRefType");
  while (true) {
    if (auto srcType = mlir::dyn_cast<MemRefType>(src.getType())) {
      Operation *definingOp = src.getDefiningOp();
      if (opndType.getShape().size() != srcType.getShape().size()) {
        if (auto subviewOp = dyn_cast<memref::SubViewOp>(definingOp)) {
          src = subviewOp.getSource();
        } else if (auto viewOp = dyn_cast<memref::ViewOp>(definingOp)) {
          src = viewOp.getSource();
        } else if (auto castOp = dyn_cast<memref::CastOp>(definingOp)) {
          src = castOp.getSource();
        } else if (auto collapseOp =
                       dyn_cast<memref::CollapseShapeOp>(definingOp)) {
          src = collapseOp.getSrc();
        } else {
          return false;
        }
      } else {
        return true; // Found and "src" contains the correct src
      }
    } else {
      return false;
    }
  }
}

static bool isVFOp(Operation *user, SymbolTable &symtab) {
  if (auto vfOp = dyn_cast<func::CallOp>(user)) {
    auto callee = symtab.lookup<func::FuncOp>(vfOp.getCallee());
    if (callee->hasAttr(hivm::VectorFunctionAttr::name)) {
      return true;
    }
  }
  return false;
}

static int copyOpDstReachUses(SymbolTable &symtab, Value &dst, Value &src,
                              Operation *copyOp, Operation *&vfOp,
                              Operation *&copyBackOp) {
  // Returns  0 - possible opportunity
  // Returns -1 - definitely not an opportunity
  vfOp = nullptr;
  copyBackOp = nullptr;
  Operation *nextNode = copyOp->getNextNode();

  for (auto *user : dst.getUsers()) {
    if (user == copyOp) {
      // Ok to reach itself
    } else if (isVFOp(user, symtab) && !vfOp) {
      // Ok to directly reach a vfOp.
      vfOp = user;
    } else if (auto castOp = dyn_cast<memref::CastOp>(user)) {
      // Not Ok to reach a CastOp unless there is
      // a vfOp directly behind the CastOp (one level indirection).
      Value cDst = castOp.getDest();
      Operation *indirectVFOp = nullptr;
      for (auto *iOp : cDst.getUsers()) {
        if (isVFOp(iOp, symtab) && !vfOp) {
          indirectVFOp = iOp;
          break;
        }
      }
      if (indirectVFOp) {
        vfOp = indirectVFOp;
      } else
        return -1; // Bail out, unsafe situation.
    } else if (auto annotateOp = dyn_cast<annotation::MarkOp>(user)) {
      // Ok to reach a MarkOp.
      continue;
    } else if (auto anotherCopyOp = dyn_cast<hivm::CopyOp>(user)) {
      // Not Ok to reach a CopyOp unless this is the copyBackOp.
      Value backSrc = anotherCopyOp.getSrc();
      Value backDst = anotherCopyOp.getDst();
      bool defReachOtherUses = false;
      for (auto *backUser : backDst.getUsers()) {
        if (auto userOp = dyn_cast<hivm::CopyOp>(backUser)) {
          if (userOp == copyOp || userOp == anotherCopyOp) {
            continue;
          }
        }
        defReachOtherUses = true;
      }
      if (!copyBackOp && !defReachOtherUses && backDst == src &&
          copyUB2UB(backSrc, src)) {
        copyBackOp = user;
      } else
        return -1; // Bail out, unsafe situation
    } else
      return -1; // Bail out, not Ok to reach any other Ops
  }
  if (!vfOp && isVFOp(nextNode, symtab)) {
    vfOp = nextNode;
  }
  if (!vfOp)
    return -1;
  return 0; // success
}

static void allocReachVF(OpBuilder &builder, Operation *allocOp,
                         Operation *copyOp, Operation *copyBack, Operation *vf,
                         Value src) {
  builder.setInsertionPointAfter(copyOp);
  SmallVector<Value> newOperands;
  bool foundMatchingSrc = false;
  for (OpOperand &opOperand : vf->getOpOperands()) {
    Value opnd = opOperand.get();
    Operation *matchAllocOp = traceUpForAlloc(opnd);
    if (allocOp == matchAllocOp) {
      foundMatchingSrc = traceUpForMatchingDimensionSource(opnd, src);
      if (foundMatchingSrc) {
        Value casted =
            builder.create<memref::CastOp>(vf->getLoc(), opnd.getType(), src);
        LLVM_DEBUG(dbgs() << "Create Cast Op: \n"; casted.dump();
                   dbgs() << "\n";);
        newOperands.push_back(casted);
      }
    } else {
      newOperands.push_back(opOperand.get());
    }
  }
  if (foundMatchingSrc) {
    vf->setOperands(newOperands);
    LLVM_DEBUG(dbgs() << "Resetting vf operands: \n"; vf->dump();
               dbgs() << "\n";);

    LLVM_DEBUG(dbgs() << "Removing Copy Op: \n"; copyOp->dump();
               dbgs() << "\n");
    copyOp->erase();
    if (copyBack) {
      LLVM_DEBUG(dbgs() << "Removing CopyBack Op: \n"; copyBack->dump();
                 dbgs() << "\n");
      copyBack->erase();
    }
  }
}

void RemoveCopyOpsPass::runOnOperation() {
  auto funcOp = getOperation();
  if (funcOp->hasAttr(hivm::VectorFunctionAttr::name))
    return;

  ModuleOp module = funcOp->getParentOfType<ModuleOp>();
  SymbolTable symtab(module);
  OpBuilder builder(funcOp->getContext());

  getAnalysis<DominanceInfo>();

  funcOp.walk<WalkOrder::PreOrder>([&](hivm::CopyOp copyOp) {
    Value src = copyOp.getSrc();
    Value dst = copyOp.getDst();

    Operation *allocOp = traceUpForAlloc(dst);
    Operation *vf = nullptr;
    Operation *copyBack = nullptr;

    if (!(allocOp && copyUB2UB(src, dst))) {
      WalkResult::advance();
    }
    int checkReachedUses =
        copyOpDstReachUses(symtab, dst, src, copyOp, vf, copyBack);
    if (checkReachedUses != -1) {
      assert(vf && "Must discover a vf Op");
      allocReachVF(builder, allocOp, copyOp, copyBack, vf, src);
    } 
  });
}

std::unique_ptr<Pass> hivm::createRemoveCopyOpsPass() {
  return std::make_unique<RemoveCopyOpsPass>();
}
