//===--------------------- NormalizeLoopIterator.cpp ----------------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/HIVM/Transforms/NormalizeLoopIterator.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
                                                 Value initVal,
                                                 LoopLikeOpInterface loopOp) {
  if (auto mbIfOp = traceDefOp<scf::IfOp>(yieldVal)) {
    auto concreteIfOp = cast<scf::IfOp>(*mbIfOp);
    assert(concreteIfOp.elseBlock());

    for (const auto &[thenVal, elseVal] :
         llvm::zip(concreteIfOp.thenYield().getOperands(),
                   concreteIfOp.elseYield().getOperands())) {
      if (thenVal == initVal) {
        return yieldMemoryInitialization(elseVal, initVal, loopOp);
      }
      
      if (elseVal == initVal) {
        return yieldMemoryInitialization(thenVal, initVal, loopOp);  
      }
    }
  }

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

      auto aliasPairs = getOperationAliasInfo(useOp);
      if (!aliasPairs.empty()) {
        for (auto aliasPair : aliasPairs) {
          assert(curAlias == aliasPair.second);
          memoryAlias.push_back(aliasPair.first);
        }
      }

      if (auto memoryEffectOp = dyn_cast<MemoryEffectOpInterface>(useOp)) {
        if (memoryEffectOp.getEffectOnValue<MemoryEffects::Write>(curAlias))
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
      auto aliasPairs = getOperationAliasInfo(useOp);
      if (!aliasPairs.empty()) {
        for (auto aliasPair : aliasPairs) {
          assert(curAlias == aliasPair.second);
          memmoryAlias.push_back(aliasPair.first);
        }
      }
      if (useOp == yieldInit || isBefore(useOp, yieldInit))
        continue;

      if (auto memoryEffectOp = dyn_cast<MemoryEffectOpInterface>(useOp)) {
        if (memoryEffectOp.getEffectOnValue<MemoryEffects::Read>(curAlias) ||
            isa<scf::YieldOp>(useOp))
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
      auto yieldFirstInit =
          yieldMemoryInitialization(yieldVals[i], iterArgs[i], loopOp);
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
