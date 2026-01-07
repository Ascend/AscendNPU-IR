//===----------------- CacheIO.cpp ----------------------------------------===//
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

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HFusion/Analysis/ReshapeAnalyzer.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/CacheFuncIO.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"

namespace mlir {
#define GEN_PASS_DEF_CACHEIO
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "cache-io"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define LLDBG(X)                                                               \
  LLVM_DEBUG(DBGS() << __FILE__ << ":" << __LINE__ << " " << X << "\n")

#include "bishengir/Dialect/Utils/Util.h"
using namespace mlir::utils::debugger;

using namespace mlir;

namespace {
constexpr static char kFuncArgIdxFormat[] = "__arg{0}__";

/// Set a unit attribute named \c attrName to \c op.
void setNamedUnitAttr(Operation *op, StringRef attrName) {
  assert(op != nullptr);
  op->setAttr(attrName, UnitAttr::get(op->getContext()));
}

// save and restore old operands to avoid return same hfusion.store
// e.g. kernel to be scheduled:
//   %res = linalg.op ins(...)
//   return %res, %res
// after first cache write and restore operands:
//   %res = linalg.op ins(...)
//   %store0 = hfusion.store(%res)
//   return %store0, %res
// after second cache write and restore operands:
//   %res = linalg.op ins(...)
//   %store0 = hfusion.store(%res)
//   %store1 = hfusion.store(%res)
//   return %store0, %store1
// if not save and restore, the final error ir will be:
//   %res = linalg.op ins(...)
//   %store0 = hfusion.store(%res)
//   %store1 = hfusion.store(%res)
//   return %store0, %store0
void restoreOperands(Operation *op, const SmallVector<Value> &oldOperands,
                     size_t excludeStart) {
  for (size_t i = 0; i < op->getNumOperands(); ++i) {
    // We only restore the ops starting from the given index
    if (i <= excludeStart) {
      continue;
    }
    op->getOpOperand(i).assign(oldOperands[i]);
  }
}

void cacheWriteFuncReturn(mlir::OpBuilder &builder, func::FuncOp funcOp,
                          bool annotate, bool writeUnique) {
  hfusion::detail::ReshapeAnalyzer reshapeAnalyzer(funcOp);
  func::ReturnOp returnOp = nullptr;
  funcOp->walk([&returnOp](func::ReturnOp op) { returnOp = op; });
  if (returnOp == nullptr)
    llvm_unreachable("Return Op not found");
  for (size_t i = 0; i < returnOp->getNumOperands(); ++i) {
    if (funcOp.getResultAttr(i, hacc::CachedIOAttr::name)) {
      // ignore already cached func result
      continue;
    }
    Value res = returnOp->getOperand(i);
    // mark cache write on reshape source
    auto traceChain = reshapeAnalyzer.getReshapeChain(res);
    auto tracedRes = traceChain.empty() ? res : traceChain.back();
    // Trace back Store operations too
    while (auto currentStoreOp = dyn_cast_if_present<hfusion::StoreOp>(
              tracedRes.getDefiningOp())) {
      // Add the return operand index attribute if missing
      if (!currentStoreOp->hasAttr(hfusion::ReturnOperandNumAttr::name)) {
        currentStoreOp->setAttr(hfusion::ReturnOperandNumAttr::name,
                                builder.getI64IntegerAttr(i));
      }
      tracedRes = currentStoreOp.getDpsInputs()[0];
    }

    builder.setInsertionPointAfterValue(tracedRes);
    auto tracedOp =
        hfusion::detail::ReshapeAnalyzer::getOpsFromReshapeValue(traceChain);

    // save old operands to avoid change other operands
    SmallVector<Value> oldOperands = returnOp->getOperands();
    // create cache write for current operand
    hfusion::CacheWriteOptions options = {/*outputOnly=*/true,
                                          /*cacheWriteToOutputInit=*/true,
                                          /*reshapeTrace=*/tracedOp};
    auto cachedOp =
        hfusion::createCacheWrite(builder, cast<OpResult>(tracedRes), options);

    // restore operands that have not been handled yet
    restoreOperands(returnOp, oldOperands, /*excludeStart=*/i);

    if (succeeded(cachedOp)) {
      if (annotate)
        setNamedUnitAttr(cachedOp.value(),
                         hfusion::StoreOp::getOperationName());
      if (writeUnique) {
        /// To avoid CSE simplifying duplicate store we set a Unique attribute
        /// for each return Operand
        auto resultOprNumAttr = builder.getI64IntegerAttr(i);
        cachedOp.value()->setAttr(hfusion::ReturnOperandNumAttr::name,
                                  resultOprNumAttr);
      }
    }
  }
}

void cacheReadFuncArg(mlir::OpBuilder &builder, func::FuncOp funcOp,
                      bool annotate) {
  hfusion::detail::ReshapeAnalyzer reshapeAnalyzer(funcOp);
  for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
    LDBG("Iterating arguments " << idx << " " << arg);
    if (!isa<TensorType>(arg.getType())) {
      continue;
    }

    if (funcOp.getArgAttr(idx, hacc::CachedIOAttr::name)) {
      // ignore already cached func argument
      continue;
    }

    bool funcArgIsReshaped = false;
    bool funcResultIsReshaped = false;
    if (auto resultIdx = hfusion::getFuncArgTiedResultReturnIdx(
            arg, funcArgIsReshaped, funcResultIsReshaped)) {
      continue;
    }
    SetVector<Value> descendants;
    reshapeAnalyzer.getReshapeDescendants(arg, descendants);
    auto tracedArg = descendants.empty() ? arg : descendants[0];
    LDBG("Finding traced arg " << tracedArg);
    builder.setInsertionPointAfterValue(tracedArg);
    Operation *cachedOp =
        hfusion::createCacheRead(builder, tracedArg, tracedArg.getLoc());
    if (annotate) {
      setNamedUnitAttr(cachedOp, llvm::formatv(kFuncArgIdxFormat, idx).str());
      setNamedUnitAttr(cachedOp, hfusion::LoadOp::getOperationName());
    }
  }
}

} // namespace

void hfusion::cacheFuncIO(func::FuncOp funcOp, bool annotate,
                          bool writeUnique) {
  OpBuilder builder(funcOp.getContext());
  // cache read for arguments
  cacheReadFuncArg(builder, funcOp, annotate);
  // cache write for return operands
  cacheWriteFuncReturn(builder, funcOp, annotate, writeUnique);
}

namespace mlir {
struct CacheIOPass : public impl::CacheIOBase<CacheIOPass> {
public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    hfusion::cacheFuncIO(funcOp);
  }
};
} // namespace mlir

std::unique_ptr<Pass> mlir::hfusion::createCacheIO() {
  return std::make_unique<CacheIOPass>();
}
