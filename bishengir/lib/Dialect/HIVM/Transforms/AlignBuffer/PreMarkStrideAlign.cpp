//===--- PreMarkStrideAlign.cpp --- Pre-analyze skip-stride-align allocs ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Detect DMA-loaded buffers that will be vload'd in vector functions where
// stride alignment would create inter-row gaps (non-contiguous layout) that
// vldsx1 (128-element consecutive flat-pointer load) cannot handle.
// Marks their root allocs with SkipStrideAlignForVLoad so MarkStrideAlign
// can skip them.
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/AlignBuffer/Util.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"

static constexpr const char kDebugType[] = "hivm-pre-mark-stride-align";
#define DEBUG_TYPE kDebugType
inline llvm::raw_ostream &DBGS() {
  return llvm::dbgs() << '[' << DEBUG_TYPE << "] ";
}
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_PREMARKSTRIDEALIGN
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {

struct PreMarkStrideAlignPass
    : public impl::PreMarkStrideAlignBase<PreMarkStrideAlignPass> {
public:
  void runOnOperation() override;
};

} // namespace

//===----------------------------------------------------------------------===//
// Judgment helpers – moved from MarkStrideAlign.cpp
//===----------------------------------------------------------------------===//

// Check if a callee block arg is read by vector::TransferReadOp (vload).
static bool isArgUsedByVLoad(func::FuncOp callee, int argIdx) {
  auto barg = callee.getArgument(argIdx);
  for (Operation *user : barg.getUsers()) {
    Operation *cur = user;
    while (cur) {
      if (isa<memref::SubViewOp, memref::ViewOp>(cur)) {
        if (cur->getUses().empty()) {
          break;
        }
        cur = *cur->getUsers().begin();
        continue;
      }
      if (isa<vector::TransferReadOp>(cur)) {
        return true;
      }
      break;
    }
  }
  return false;
}

// Check if a memref value flows into a vf call whose callee vloads it.
// Stride alignment on vload'd buffers creates non-contiguous gaps that
// vldsx1 cannot handle (128-consecutive-element flat-pointer load).
static bool flowsToVLoadInVF(Value val, ModuleOp moduleOp) {
  SmallVector<Value> worklist = {val};
  DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    if (!visited.insert(cur).second) {
      continue;
    }
    for (Operation *user : cur.getUsers()) {
      if (isa<ViewLikeOpInterface, memref::CastOp>(user)) {
        worklist.push_back(user->getResult(0));
        continue;
      }
      if (auto callOp = dyn_cast<func::CallOp>(user)) {
        for (auto [idx, operand] :
             llvm::enumerate(callOp.getArgOperands())) {
          if (operand != cur) {
            continue;
          }
          auto callee =
              moduleOp.lookupSymbol<func::FuncOp>(callOp.getCallee());
          if (callee && isArgUsedByVLoad(callee, static_cast<int>(idx))) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

// Check if the root value already has a SkipStrideAlignForVLoad mark.
static bool hasSkipMark(Value root) {
  return utils::getAnnotateOpWithAttr(
      root, hivm::SkipStrideAlignForVLoadAttr::name).has_value();
}

// Create annotation.mark on the root alloc value to signal
// MarkStrideAlign that this buffer must skip stride alignment.
static void markSkipStrideAlignForVLoad(OpBuilder &builder, Value root) {
  if (hasSkipMark(root)) {
    return;
  }
  auto *defOp = root.getDefiningOp();
  if (!defOp || !utils::isAllocLikeOp(defOp)) {
    return;
  }
  LDBG("pre-mark: skip stride align for root " << root);
  builder.setInsertionPointAfter(defOp);
  auto markOp = builder.create<annotation::MarkOp>(defOp->getLoc(), root);
  markOp->setAttr(hivm::SkipStrideAlignForVLoadAttr::name,
                  hivm::SkipStrideAlignForVLoadAttr::get(builder.getContext()));
  (void)markOp;
}

//===----------------------------------------------------------------------===//
// Pass entry point
//===----------------------------------------------------------------------===//

void PreMarkStrideAlignPass::runOnOperation() {
  OpBuilder builder(&getContext());
  auto funcOp = getOperation();
  if (hacc::utils::isHost(funcOp)) {
    return;
  }

  auto moduleOp = funcOp->getParentOfType<ModuleOp>();
  bool archIsRegbased = hacc::utils::isRegBasedArch(moduleOp);
  if (!archIsRegbased) {
    return;
  }

  // Walk 1: find hivm::LoadOp results that flow to vload in VFs.
  funcOp->walk([&](hivm::LoadOp loadOp) {
    auto result = loadOp.getDst();
    auto memrefTy = dyn_cast<MemRefType>(result.getType());
    if (!memrefTy || !memrefTy.hasRank()) {
      return WalkResult::advance();
    }

    if (!flowsToVLoadInVF(result, moduleOp)) {
      return WalkResult::advance();
    }

    Value root = hivm::traceToRoot(result);
    markSkipStrideAlignForVLoad(builder, root);
    return WalkResult::advance();
  });

}

std::unique_ptr<Pass> mlir::hivm::createPreMarkStrideAlignPass() {
  return std::make_unique<PreMarkStrideAlignPass>();
}
