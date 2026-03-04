//===--------------------- NormalizeLoopIterator.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/HIVM/Transforms/NormalizeLoopIterator.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

#include <cassert>
#include <cstddef>
#include <memory>
#include <optional>
#include <utility>

#define DEBUG_TYPE "normalize-loop-iterator"

namespace mlir {
#define GEN_PASS_DEF_NORMALIZELOOPITERATOR
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {
bool isBefore(Operation *before, Operation *after) {
  if (before->getBlock() == after->getBlock()) {
    return before->isBeforeInBlock(after);
  }

  auto afterParentOp = after->getParentOp();
  if (afterParentOp == nullptr) {
    return false;
  }
  return isBefore(before, afterParentOp);
}

FailureOr<Operation *> yieldMemoryInitialization(Value yieldVal,
                                                 LoopLikeOpInterface loopOp) {
  FailureOr<memref::AllocOp> allocOp = getMemRefAlloc(yieldVal);
  if (failed(allocOp) ||
      loopOp.isDefinedOutsideOfLoop((*allocOp).getMemref())) {
    LLVM_DEBUG(llvm::dbgs()
               << yieldVal
               << " unconcerned state of yield value which original root alloc "
                  "is outside from loop or yield value hasn't root alloc ");
    return failure();
  }

  SmallVector<Value> memoryAlias{(*allocOp).getMemref()};
  Operation *firstInitialization = nullptr;
  while (!memoryAlias.empty()) {
    auto curAlias = memoryAlias.pop_back_val();
    for (OpOperand &useOperand : curAlias.getUses()) {
      Operation *useOp = useOperand.getOwner();
      // Here just wanna get the first initialization on the yield value memory
      if (firstInitialization && isBefore(firstInitialization, useOp))
        continue;

      for (auto aliasPair : getOperationAliasInfo(useOp)) {
        assert(curAlias == aliasPair.second);
        memoryAlias.push_back(aliasPair.first);
      }

      if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(useOp)) {
        if (dpsOp.isDpsInit(&useOperand))
          firstInitialization = useOp;
      } else if (isVFCall(useOp)) {
        // VF call op may initialize buffer just as DestinationStyleOpInterface, see issue:
        // https://codehub-y.huawei.com/CompilerKernel/BiShengKernel/BiSheng/issues/3768
        firstInitialization = useOp;
      }
    }
  }
  if (!firstInitialization)
    return failure();
  return firstInitialization;
}

bool existIterArgUseAfterYieldValInit(Value iterArg, Operation *yieldInit) {
  SmallVector<Value> memmoryAlias{iterArg};
  while (!memmoryAlias.empty()) {
    auto curAlias = memmoryAlias.pop_back_val();
    for (OpOperand &useOperand : curAlias.getUses()) {
      Operation *useOp = useOperand.getOwner();

      // Collect all alias info whatever op order
      for (auto aliasPair : getOperationAliasInfo(useOp)) {
        if (curAlias == aliasPair.second) {
          memmoryAlias.push_back(aliasPair.first);
        }
      }

      if (useOp == yieldInit || utils::isBefore(useOp, yieldInit))
        continue;

      if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(useOp)) {
        if (dpsOp.isDpsInput(&useOperand))
          return true;
      } else if (isVFCall(useOp)) {
        // VF call op may write to buffer just as DestinationStyleOpInterface, see issue:
        // https://codehub-y.huawei.com/CompilerKernel/BiShengKernel/BiSheng/issues/3768
        return true;
      }
    }
  }

  return false;
}

class NormalizeIterUseAfterYieldInit
    : public OpInterfaceRewritePattern<LoopLikeOpInterface> {
  using OpInterfaceRewritePattern<
      LoopLikeOpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(LoopLikeOpInterface loopOp,
                                PatternRewriter &rewriter) const override {
    if (loopOp.getLoopRegions().size() != 1 ||
        !loopOp.getLoopRegions()[0]->hasOneBlock()) {
      LLVM_DEBUG(llvm::dbgs()
                 << loopOp
                 << "unsupported loop-like op with multiple regions or blocks");
      return failure();
    }

    auto iterArgs = loopOp.getRegionIterArgs();
    auto yieldVals = loopOp.getYieldedValues();
    assert(iterArgs.size() == yieldVals.size());

    SmallVector<size_t> candidate;
    for (size_t i = 0; i < iterArgs.size(); ++i) {
      if (iterArgs[i] == yieldVals[i] ||
          !llvm::isa<MemRefType>(iterArgs[i].getType()))
        continue;

      if (getHIVMAddressSpace(yieldVals[i].getType()) ==
          hivm::AddressSpace::GM) {
        continue;
      }
      auto yieldFirstInit = yieldMemoryInitialization(yieldVals[i], loopOp);
      if (failed(yieldFirstInit)) {
        LLVM_DEBUG(llvm::dbgs()
                   << yieldVals[i]
                   << "couldn't find memmory first initialization");
        continue;
      }
      if (existIterArgUseAfterYieldValInit(iterArgs[i], *yieldFirstInit))
        candidate.push_back(i);
    }

    if (candidate.empty())
      return failure();

    rewriter.setInsertionPoint(
        loopOp.getLoopRegions()[0]->front().getTerminator());
    for (auto index : candidate) {
      Value iterArg = iterArgs[index];
      Value originYield = yieldVals[index];
      rewriter.create<hivm::CopyOp>(loopOp->getLoc(), TypeRange{},
                                    /*src*/ originYield, /*dst*/ iterArg);

      assert(loopOp.getYieldedValuesMutable().has_value());
      rewriter.modifyOpInPlace(loopOp, [&]() {
        loopOp.getYieldedValuesMutable().value()[index].assign(iterArg);
      });
    }
    return success();
  }
};

} // anonymous namespace

void mlir::hivm::populateNormalizeLoopIneratorPattern(
    RewritePatternSet &patterns) {
  patterns.add<NormalizeIterUseAfterYieldInit>(patterns.getContext());
}

class NormalizeLoopIteratorPass
    : public impl::NormalizeLoopIteratorBase<NormalizeLoopIteratorPass> {
public:
  using NormalizeLoopIteratorBase<
      NormalizeLoopIteratorPass>::NormalizeLoopIteratorBase;
  void runOnOperation() override;
};

void NormalizeLoopIteratorPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  RewritePatternSet rewritePatterns(&getContext());
  populateNormalizeLoopIneratorPattern(rewritePatterns);
  if (failed(applyPatternsGreedily(funcOp, std::move(rewritePatterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::hivm::createNormalizeLoopIteratorPass() {
  return std::make_unique<NormalizeLoopIteratorPass>();
}