//===-------- RemoveRedundantCopy.cpp -- Remove Redundant Copy Pass -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallSet.h"

#define DEBUG_TYPE "memref-remove-redundant-copy"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_REMOVEREDUNDANTCOPY
#include "bishengir/Dialect/MemRef/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::memref;

/// This pass removes redundant copy operations. Additionally, it
/// removes leftover definition and deallocation operations by erasing the
/// copy operation.
struct RemoveRedundantCopyPass
    : public impl::RemoveRedundantCopyBase<RemoveRedundantCopyPass> {
public:
  void runOnOperation() override;

private:
  /// Returns the allocation operation for `value` if it exists.
  /// nullptr otherwise.
  Operation *getAllocationOp(Value value) {
    if (Operation *op = value.getDefiningOp()) {
      if (auto effects = dyn_cast<MemoryEffectOpInterface>(op))
        if (effects.hasEffect<mlir::MemoryEffects::Allocate>())
          return op;
    }
    return nullptr;
  }

  /// Returns the deallocation operation for `value` if it exists.
  /// nullptr otherwise.
  Operation *getDeallocationOp(Value value) const {
    auto valueUsers = value.getUsers();
    auto it = llvm::find_if(valueUsers, [&](Operation *op) {
      auto effects = dyn_cast<MemoryEffectOpInterface>(op);
      return effects && effects.hasEffect<mlir::MemoryEffects::Free>();
    });
    return (it == valueUsers.end() ? nullptr : *it);
  }

  /// Check whether the `val` is used by `op`.
  static bool doesOpUseVal(Value val, Operation *op) {
    return llvm::is_contained(op->getOperands(), val);
  }

  /// Check if an op that lies on one of the paths between `start`
  /// and `end` and satisfies `checkPropertiesOfOperation`.
  bool hasInterveningOp(const Value val, Operation *start, Operation *end,
                        std::function<bool(Value, Operation *)>
                            checkPropertiesOfOperation) const {
    // Check for all paths from operation `fromp` to operation `untilOp` for the
    // given property.
    std::function<bool(Operation *, Operation *)> recur =
        [&](Operation *fromOp, Operation *untilOp) {
          auto fromOpBlock = fromOp->getBlock();
          for (auto iter = ++fromOp->getIterator(), end = fromOpBlock->end();
               iter != end && &*iter != untilOp; ++iter) {
            if (checkPropertiesOfOperation(val, &*iter)) {
              return true;
            }
          }
          return false;
        };
    return recur(start, end);
  }

  // Check if `op` is the last user of `val` in a `block`, and it's the only op
  // of its kind in that `block`.
  bool isLastAndUniqueInBlock(Value val, Operation *op, Block *block) {
    bool found = false;
    for (auto &userOp : block->getOperations()) {
      if (doesOpUseVal(val, &userOp)) {
        if (&userOp == op) {
          found = true;
        } else if (found || userOp.getName() == op->getName()) {
          return false;
        }
      }
    }
    return true;
  }

  void removeCopy(CopyOpInterface copyOp,
                  llvm::SmallPtrSet<Operation *, 4> &opsToErase) {
    Value src = copyOp.getSource();
    Value dest = copyOp.getTarget();
    Operation *srcDefOp = getAllocationOp(src);
    Operation *destDefOp = getAllocationOp(dest);

    /// Input:
    /// func() {
    ///   %a = memref(xxx)
    ///   %source = alloc()
    ///   write_to(%source)
    ///   %dest = subview(%a)
    ///   copy(%source, %dest)
    ///   return
    /// }
    ///
    /// Output:
    /// func(){
    ///   %a = memref(xxx)
    ///   %dest = subview(%a)
    ///   write_to(%dest)
    ///   return
    /// }

    Operation *destSubViewOp = dest.getDefiningOp<memref::SubViewOp>();

    /*
    alloc
        user/write alloc
        ...
        user/write alloc

        subview = sub(a)
        copy alloc -> subview

    - %alloc in an outer block and (%subview, %copy) are in the same inner
    block
    - there is only one copy op and it's the last user of %alloc in the inner
    block
    */
    bool matchAllocWriteCopy =
        srcDefOp && destSubViewOp && srcDefOp->getParentOp() &&
        srcDefOp->getParentOp()->isAncestor(copyOp) &&
        isLastAndUniqueInBlock(src, copyOp, copyOp->getBlock());
    if (matchAllocWriteCopy) {
      opsToErase.insert(copyOp);
      Operation *firstUser = nullptr;
      auto innerBlock = copyOp->getBlock();
      for (Operation &userOp : innerBlock->getOperations()) {
        if (doesOpUseVal(src, &userOp)) {
          userOp.replaceUsesOfWith(src, dest);
          if (!firstUser) {
            firstUser = &userOp;
          }
        }
      }

      if (destSubViewOp->getParentOp() == copyOp->getParentOp()) {
        destSubViewOp->moveBefore(firstUser);
      }

      return;
    }

    /// Constraints:
    /// 1) The `destination` op should be MemoryEffects::Allocate op or
    /// function argument. 2) If the `destination` op is
    /// MemoryEffects::Allocate op, there should not exist any users of
    /// `destination` op before the copy op. We replace the dest by src.
    /// Input:
    /// func() {
    ///   %source = alloc/alloca()
    ///   %destination = alloc/alloca()
    ///   write_to(%source)
    ///   copy(%source, %destination)
    ///   return %destination
    /// }
    ///
    /// Output:
    /// func(){
    ///   %source = alloc/alloca()
    ///   write_to(%source)
    ///   return %source
    /// }

    /// 3) If the `destination` op is function argument, which means there
    /// should not exist any users of `source` op after the copy op. We
    /// replace the src by dest. Input: func(%destination : memref) {
    ///   %source = alloc()
    ///   write_to(%source)
    ///   copy(%source, %destination)
    ///   dealloc(%source)
    ///   return
    /// }
    ///
    /// Output:
    /// func(%destination : memref){
    ///   write_to(%destination)
    ///   return
    /// }

    if (destDefOp == nullptr && !llvm::isa<BlockArgument>(dest)) {
      return;
    }

    Operation *srcDeallocOp = getDeallocationOp(src);
    Operation *destDeallocOp = getDeallocationOp(dest);

    if (dest.getParentRegion() == nullptr || src.getParentRegion() == nullptr) {
      return;
    }

    Operation *firstOpUsingDest = &dest.getParentRegion()->front().front();
    Operation *lastOpUsingSrc = &src.getParentRegion()->back().back();

    if (hasInterveningOp(src, copyOp,
                         srcDeallocOp ? srcDeallocOp : lastOpUsingSrc,
                         &doesOpUseVal) ||
        hasInterveningOp(dest, destDefOp ? destDefOp : firstOpUsingDest, copyOp,
                         &doesOpUseVal)) {
      return;
    }

    if (destDefOp) {
      // replace dst by src
      opsToErase.insert(destDefOp);
      opsToErase.insert(copyOp);
      if (destDeallocOp)
        opsToErase.insert(destDeallocOp);
      dest.replaceAllUsesWith(src);
      return;
    }

    auto destParentOp = dest.getParentBlock()->getParentOp();
    if (!destParentOp)
      return;
    if (srcDefOp && destParentOp->isAncestor(srcDefOp)) {
      // replace src by dst
      opsToErase.insert(srcDefOp);
      opsToErase.insert(copyOp);
      if (srcDeallocOp)
        opsToErase.insert(srcDeallocOp);
      src.replaceAllUsesWith(dest);
      return;
    }
  }
};

void RemoveRedundantCopyPass::runOnOperation() {
  func::FuncOp func = getOperation();
  llvm::SmallPtrSet<Operation *, 4> opsToErase;
  Liveness live(func);
  func.walk([&](CopyOpInterface copyOp) { removeCopy(copyOp, opsToErase); });
  for (Operation *op : opsToErase) {
    assert(op->use_empty() &&
           "uses remaining for copy ops, memref allocation and deallocation "
           "ops that should have ready to be erased");
    op->erase();
  }
  return;
}

std::unique_ptr<Pass> mlir::memref::createRemoveRedundantCopyPass() {
  return std::make_unique<RemoveRedundantCopyPass>();
}
