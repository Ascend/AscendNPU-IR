//===- HIVMOptFuncOutput.cpp - HIVM Output Func Buffer Opt ----------------===//
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

#include "bishengir/Dialect/HIVM/Transforms/Passes.h"

#include "llvm/ADT/SmallSet.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir {
#define GEN_PASS_DEF_HIVMOPTFUNCOUTPUT
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {
// Function output optimization after bufferization
struct HIVMOptFuncOutputPass
    : public impl::HIVMOptFuncOutputBase<HIVMOptFuncOutputPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk([&](func::FuncOp func) {
      if (failed(runOnFunc(func, module)))
        signalPassFailure();
    });
  }

private:
  LogicalResult runOnFunc(func::FuncOp func, ModuleOp module) {
    SmallVector<func::CallOp> callOps;
    if (auto funcUses = SymbolTable::getSymbolUses(func, module)) {
      // Loop through all the usage of this function
      for (const auto &use : *funcUses) {
        if (auto callOp = dyn_cast<func::CallOp>(use.getUser())) {
          callOps.push_back(callOp);
        }
      }
    }
    llvm::SmallSet<int, 8> usedReturnIdx;
    auto returnOp =
        llvm::cast<func::ReturnOp>(func.getBody().front().getTerminator());
    collectUsedReturnIdx(func, returnOp, usedReturnIdx);
    SmallVector<Value> usedReturnVal;
    for (const int idx : usedReturnIdx) {
      usedReturnVal.emplace_back(returnOp->getOperand(idx));
    }
    // modify func and its callers
    adjustRedundant(func, callOps, returnOp, usedReturnIdx, usedReturnVal);
    return success();
  }
  void collectUsedReturnIdx(func::FuncOp &func, func::ReturnOp &returnOp,
                            llvm::SmallSet<int, 8> &usedReturnIdx) {
    for (unsigned i = 0; i < returnOp.getNumOperands(); ++i) {
      Value returnValue = returnOp.getOperand(i);
      bool isBuffer = isa<MemRefType>(returnValue.getType());
      auto buffer = dyn_cast<BlockArgument>(returnValue);
      // Check whether this buffer is a function argument
      bool isUnnecessaryReturn =
          (isBuffer && buffer && buffer.getOwner() == &func.getBody().front());
      if (!isUnnecessaryReturn)
        usedReturnIdx.insert(i);
    }
  }
  void adjustRedundant(func::FuncOp &func,
                       MutableArrayRef<func::CallOp> callOps,
                       func::ReturnOp &returnOp,
                       const llvm::SmallSet<int, 8> &usedReturnIdx,
                       const SmallVector<Value> &usedReturnVal) {
    // Fix all the callers with the new one
    for (auto &callOp : callOps) {
      OpBuilder builder(callOp);
      builder.setInsertionPoint(callOp);
      auto newCall = builder.create<func::CallOp>(
          callOp.getLoc(), callOp.getCallee(), TypeRange(usedReturnVal),
          callOp.getOperands());

      // Example:
      // func (arg0, arg1, arg2, arg3){
      //    return x, arg2, arg1
      // }
      // res_0, res_1, res_2 = call func ins(a, b, c, d)
      // res_0 is not a buffer, needs to be passed in arg and a memref
      // Replace all usage of res_1 -> c
      // Replace all usage of res_2 -> b
      // Replace the buffers
      for (unsigned i = 0; i < returnOp.getNumOperands(); ++i) {
        if (usedReturnIdx.contains(i))
          continue;
        auto buffer = dyn_cast<BlockArgument>(returnOp.getOperand(i));
        callOp.getResult(i).replaceAllUsesWith(
            callOp.getOperand(buffer.getArgNumber()));
      }

      // Replace the non buffers
      for (auto [i, usedIdx] : llvm::enumerate(usedReturnIdx)) {
        callOp.getResult(usedIdx).replaceAllUsesWith(newCall.getResult(i));
      }
      callOp.erase();
    }

    // Fix return operation inside calledFunc
    returnOp.getOperation()->setOperands(usedReturnVal);

    // Fix return value of the called func
    auto newFuncType =
        FunctionType::get(func.getContext(), func.getFunctionType().getInputs(),
                          TypeRange(usedReturnVal));
    func.setType(newFuncType);
  }
};
} // namespace

std::unique_ptr<Pass> mlir::hivm::createHIVMOptFuncOutputPass() {
  return std::make_unique<HIVMOptFuncOutputPass>();
}
