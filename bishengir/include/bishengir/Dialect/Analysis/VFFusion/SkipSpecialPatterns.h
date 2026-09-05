//===- SkipSpecialPatterns.h --------- CV special skip patterns ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_ANALYSIS_VFFUSION_SKIPSPECIALPATTERNS_H
#define BISHENGIR_DIALECT_ANALYSIS_VFFUSION_SKIPSPECIALPATTERNS_H

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"

namespace mlir {
namespace analysis {

namespace detail {

/// Returns true if any operation in \p moduleOp uses or produces a value of
/// type !llvm.ptr<11> (an LLVM pointer in address space 11). These volatile
/// stores are emitted by debug/trace instrumentation and mark kernels whose
/// structure is incompatible with VFFusion. This is a module-level pre-gate:
/// if no ptr<11> exists, all skip patterns are skipped.
inline bool hasLLVMPtr11(ModuleOp moduleOp) {
  bool found = false;
  moduleOp.walk([&](Operation *op) -> WalkResult {
    auto checkType = [](Type t) {
      if (auto ptrType = dyn_cast<LLVM::LLVMPointerType>(t))
        return ptrType.getAddressSpace() == 11;
      return false;
    };
    for (Type t : op->getOperandTypes())
      if (checkType(t)) {
        found = true;
        return WalkResult::interrupt();
      }
    for (Type t : op->getResultTypes())
      if (checkType(t)) {
        found = true;
        return WalkResult::interrupt();
      }
    return WalkResult::advance();
  });
  return found;
}

inline bool isSumReduce(linalg::ReduceOp reduceOp) {
  if (reduceOp.getRegion().empty())
    return false;
  bool isSum = false;
  reduceOp.walk([&](arith::AddFOp) {
    isSum = true;
    return WalkResult::interrupt();
  });
  return isSum;
}

inline bool resultFeedsHirCopy(Operation *op) {
  for (Value result : op->getResults())
    for (Operation *user : result.getUsers())
      if (isa<hivm::CopyOp>(user))
        return true;
  return false;
}

inline bool valueFeedsCastExpandTranspose(Value v) {
  for (Operation *user : v.getUsers()) {
    auto castOp = dyn_cast<hfusion::CastOp>(user);
    if (!castOp)
      continue;
    for (Value castResult : castOp->getResults()) {
      for (Operation *expandUser : castResult.getUsers()) {
        auto expandOp = dyn_cast<tensor::ExpandShapeOp>(expandUser);
        if (!expandOp)
          continue;
        for (Value expandResult : expandOp->getResults())
          for (Operation *transposeUser : expandResult.getUsers())
            if (isa<linalg::TransposeOp>(transposeUser))
              return true;
      }
    }
  }
  return false;
}

} // namespace detail

// TODO: pattern that needs cast optimization.
inline bool hasCastReduceCopyPattern(ModuleOp moduleOp) {
  if (!detail::hasLLVMPtr11(moduleOp))
    return false;
  bool found = false;
  moduleOp.walk([&](linalg::ReduceOp reduceOp) -> WalkResult {
    if (found)
      return WalkResult::interrupt();
    if (!detail::isSumReduce(reduceOp))
      return WalkResult::advance();
    if (!detail::resultFeedsHirCopy(reduceOp))
      return WalkResult::advance();
    if (reduceOp.getInputs().empty())
      return WalkResult::advance();
    if (!detail::valueFeedsCastExpandTranspose(reduceOp.getInputs().front()))
      return WalkResult::advance();

    func::FuncOp parentFunc = reduceOp->getParentOfType<func::FuncOp>();
    if (!parentFunc)
      return WalkResult::advance();
    bool subCopyOk = false;
    parentFunc.walk([&](linalg::SubOp subOp) -> WalkResult {
      if (detail::resultFeedsHirCopy(subOp)) {
        subCopyOk = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (!subCopyOk)
      return WalkResult::advance();
    found = true;
    return WalkResult::interrupt();
  });
  return found;
}

// TODO: pattern that needs optimization.
inline bool hasPtr11CompareSyncForPattern(ModuleOp moduleOp) {
  if (!detail::hasLLVMPtr11(moduleOp))
    return false;
  bool found = false;
  moduleOp.walk([&](hfusion::CompareOp compareOp) -> WalkResult {
    if (found)
      return WalkResult::interrupt();
    Block *block = compareOp->getBlock();
    enum {
      kLookingForCompare,
      kLookingForSync,
      kLookingForFor,
      kDone
    } state = kLookingForCompare;
    int compareCount = 0;
    int syncCount = 0;
    for (Operation &blockOp : block->getOperations()) {
      switch (state) {
      case kLookingForCompare:
        if (isa<hfusion::CompareOp>(blockOp)) {
          if (++compareCount >= 2)
            state = kLookingForSync;
        }
        break;
      case kLookingForSync:
        if (isa<hivm::SyncBlockSetOp>(blockOp)) {
          if (++syncCount >= 6)
            state = kLookingForFor;
        }
        break;
      case kLookingForFor:
        if (isa<scf::ForOp>(blockOp)) {
          found = true;
          return WalkResult::interrupt();
        }
        break;
      default:
        break;
      }
    }
    return WalkResult::advance();
  });
  return found;
}

} // namespace analysis
} // namespace mlir

#endif // BISHENGIR_DIALECT_ANALYSIS_VFFUSION_SKIPSPECIALPATTERNS_H
