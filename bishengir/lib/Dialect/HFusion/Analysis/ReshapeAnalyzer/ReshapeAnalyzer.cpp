//===- ReshapeAnalyzer.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Analysis/ReshapeAnalyzer.h"
#include "bishengir/Dialect/HFusion/Transforms/OpFusion/FusibleHelper.h"
#include "bishengir/Dialect/Tensor/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

using namespace mlir;
using namespace mlir::hfusion;
using namespace mlir::hfusion::reshape_utils;
using namespace mlir::utils::debugger;

#define DEBUG_TYPE "reshape-analyzer"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define LLDBG(X)                                                               \
  LLVM_DEBUG(DBGS() << __FILE__ << ":" << __LINE__ << " " << X << "\n")

namespace mlir {
namespace hfusion {
namespace detail {

ReshapeAnalyzer::ReshapeAnalyzer(func::FuncOp funcOp) : func(funcOp) {
  computeReshapeInputs();
  computeUnreshapedOutputs();
  funcOp->walk([this](Operation *op) {
    for (auto &region : op->getRegions()) {
      for (auto &block : region) {
        for (auto val : block.getArguments()) {
          if (valueDependency.contains(val))
            continue;
          valueDependency[val] = valueDependency.size();
        }
      }
    }
    for (auto val : op->getResults()) {
      if (valueDependency.contains(val))
        continue;
      valueDependency[val] = valueDependency.size();
    }
  });
}

void ReshapeAnalyzer::computeReshapeInputs() {
  // Loop all the arguments
  SmallVector<ReshapeValue> simplifiedWorkList;
  // This is all the reshape heads
  for (BlockArgument arg : func.getArguments()) {
    if (!isa<RankedTensorType>(arg.getType()))
      continue;
    getReshapeDescendants(arg, simplifiedWorkList);
  }
  argIdxToReshapedInput.resize(func.getNumArguments());
  for (auto el : simplifiedWorkList) {
    int argNumber =
        static_cast<int>(cast<BlockArgument>(el.source).getArgNumber());
    argIdxToReshapedInput[argNumber].insert(el.endTarget->get());
    reshapedInputToArgIdx[el.endTarget->get()] = argNumber;
    reshapedInputs.insert(el.endTarget->get());
  }
}

void ReshapeAnalyzer::computeUnreshapedOutputs() {
  auto returnOp = dyn_cast_if_present<func::ReturnOp>(
      func.getBlocks().back().getTerminator());
  if (!returnOp)
    return;
  retIdxToUnreshapedOutputs.resize(returnOp.getNumOperands());
  for (auto &val : returnOp->getOpOperands()) {
    if (!isa<TensorType>(val.get().getType()))
      continue;
    auto reshapeHead = getReshapeHead(val.get());
    unreshapedOutputToRetIdx[reshapeHead] =
        static_cast<int64_t>(val.getOperandNumber());
    unreshapedOutputs.insert(reshapeHead);
    retIdxToUnreshapedOutputs[val.getOperandNumber()] = reshapeHead;
  }
}

SmallVector<Value> ReshapeAnalyzer::getReshapeChain(Value val) {
  SmallVector<Value> reshapeChain;
  do {
    reshapeChain.push_back(val);
    if (auto expandShape = val.getDefiningOp<tensor::ExpandShapeOp>()) {
      val = expandShape.getSrc();
    } else if (auto collapseShape =
                   val.getDefiningOp<tensor::CollapseShapeOp>()) {
      val = collapseShape.getSrc();
    } else {
      break;
    }
  } while (true);
  return reshapeChain;
}

SmallVector<Operation *>
ReshapeAnalyzer::getOpsFromReshapeValue(SmallVector<Value> chain) {
  SmallVector<Operation *> res;
  for (auto val : chain) {
    if (!isReshapeOp(val.getDefiningOp())) {
      continue;
    }
    res.push_back(val.getDefiningOp());
  }
  return res;
}

Value ReshapeAnalyzer::getReshapeHead(Value val) {
  auto chain = getReshapeChain(val);
  if (chain.size() < 1) {
    llvm::report_fatal_error("reshape chain is empty");
  }
  return chain.back();
}

Value ReshapeAnalyzer::getFirstReshape(Value val) {
  auto chain = getReshapeChain(val);
  return getFirstReshape(chain);
}

Value ReshapeAnalyzer::getFirstReshape(SmallVector<Value> &chain) {
  if (chain.size() < 2) {
    llvm::report_fatal_error("reshape chain is less than 2");
  }
  return chain[chain.size() - 2]; // get the first reshape is the 2nd back
}

void ReshapeAnalyzer::getReshapeDescendants(
    Value val, SmallVector<ReshapeValue> &descendants) {
  //  Arg (Depth = 0)  --> Collapse (Depth = 1) -------> Expand (Depth = 2)
  //                          Collapse (Depth = 3) <-------|

  std::queue<ReshapeValue> workQueue;
  auto relaxWorkList = [&](Value &curSource, Value &lastReshapeResult,
                           int currentDepth) -> void {
    LDBG("relaxing " << lastReshapeResult << " -- at " << currentDepth);
    for (OpOperand &user : lastReshapeResult.getUses()) {
      LDBG("Found usage " << *user.getOwner());
      ReshapeValue nextWork(curSource, user, currentDepth + 1);
      if (mlir::hfusion::isReshapeOp(user.getOwner())) {
        workQueue.push(nextWork);
      } else if (isExplicitlyAllowedCollapseOp(user.getOwner()) ||
                 isa<DestinationStyleOpInterface>(user.getOwner())) {
        LDBG("Found as descendant: " << nextWork.endTarget->get());
        descendants.push_back(nextWork);
      }
    }
  };
  relaxWorkList(val, val, 0);
  while (!workQueue.empty()) {
    auto currentWork = workQueue.front();
    workQueue.pop();
    Operation *currentOp = currentWork.endTarget->getOwner();
    Value reshapeResult = currentOp->getResult(0);
    relaxWorkList(currentWork.source, reshapeResult, currentWork.depth);
  }
}

void ReshapeAnalyzer::getReshapeDescendants(Value val,
                                            SetVector<Value> &descendants) {
  SmallVector<ReshapeValue> reshapedOpOperands;
  getReshapeDescendants(val, reshapedOpOperands);
  llvm::sort(reshapedOpOperands,
             [this](const ReshapeValue &a, const ReshapeValue &b) {
               return valueDependency.at(a.endTarget->get()) <
                      valueDependency.at(b.endTarget->get());
             });
  for (auto el : reshapedOpOperands) {
    descendants.insert(el.endTarget->get());
  }
}

OpOperand *
ReshapeAnalyzer::traceReshapeAndRewriteInverse(PatternRewriter &rewriter,
                                               OpOperand *outValue) {
  DenseSet<Operation *> createdOperations;
  while (true) {
    bool insideImportantRegion = true;
    for (Operation *user : outValue->get().getUsers()) {
      if (isReshapeOp(user)) {
        insideImportantRegion = false;
        break;
      }
    }
    // Time to stop if
    if (insideImportantRegion) {
      break;
    }
    bool reshapePropagated = false;
    // %arg
    // %expanded = expand_shape %arg
    // return %arg
    //
    // |
    // V
    //
    // %arg
    // %expanded = expand_shape %arg
    // %inverse_expand = collapse_shape %expanded
    // return %inverse_expand

    auto outUsers = outValue->get().getUsers();
    for (Operation *user : outUsers) {
      if (createdOperations.contains(user))
        continue;
      Operation *inverseReshape = nullptr;
      rewriter.setInsertionPoint(outValue->getOwner());
      if (auto expandOp = dyn_cast<tensor::ExpandShapeOp>(user)) {
        // Expand shape here
        inverseReshape =
            tensor::reshape_utils::createExpandInverse(rewriter, expandOp);
      }
      if (auto collapseOp = dyn_cast<tensor::CollapseShapeOp>(user)) {
        inverseReshape =
            tensor::reshape_utils::createCollapseInverse(rewriter, collapseOp);
      }
      if (!inverseReshape)
        continue;
      createdOperations.insert(inverseReshape);
      // Set the opOperands with the result of expand
      Value inverseRes = inverseReshape->getResult(0);
      rewriter.modifyOpInPlace(outValue->getOwner(),
                               [&]() { outValue->set(inverseRes); });
      // Old out value, replace the value with the inverse

      // %arg
      // %expanded = expand_shape %arg
      // %inverse_expand = collapse_shape %expanded
      // return [Replace this from %arg to %inverse_expand]
      outValue = &inverseReshape->getOpOperand(0);
      reshapePropagated = true;
      break;
    }
    LDBG(*outValue->getOwner()->getParentOp());
    if (!reshapePropagated)
      break;
  }
  return outValue;
}
} // namespace detail
} // namespace hfusion
} // namespace mlir
