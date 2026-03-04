//===- ConstantizeTilingData.cpp -- Optimize cst tiling data -----*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic to optimize tiling data that are compile-time
// constants.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"

#define DEBUG_TYPE "hfusion-constantize-tiling-data"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

namespace mlir {
namespace hfusion {

#define GEN_PASS_DEF_CONSTANTIZETILINGDATA
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"

namespace {

static Value cloneArithConstant(OpBuilder &builder, Operation *constOp,
                                Type targetType) {
  // Clone the constant op
  auto clonedConstOp = cast<arith::ConstantOp>(builder.clone(*constOp));
  auto constReplacement = clonedConstOp->getResult(0);
  return convertScalarToDtype(builder, constReplacement.getLoc(),
                              constReplacement, targetType,
                              /*isUnsignedCast=*/true);
}

class ConstantizeTilingDataPass
    : public impl::ConstantizeTilingDataBase<ConstantizeTilingDataPass> {
  void runOnOperation() final {
    ModuleOp mod = getOperation();
    llvm::StringMap<SmallVector<func::FuncOp>> tilingFuncNameToDeviceFunc;
    mod.walk([&](func::FuncOp func) {
      if (auto attr = func->getAttrOfType<hacc::TilingFunctionAttr>(
              hacc::TilingFunctionAttr::name))
        tilingFuncNameToDeviceFunc[attr.getFuncNameStr()].push_back(func);
    });

    for (auto &[calcFuncName, devFuncs] : tilingFuncNameToDeviceFunc) {
      auto calcFunc = mod.lookupSymbol<func::FuncOp>(calcFuncName);
      if (!calcFunc) {
        mod.emitError("Corresponding calcFunc not found: ") << calcFuncName;
        return signalPassFailure();
      }
      if (failed(tiling::deviceFuncsMatchTilingFunc(devFuncs, calcFunc)))
        return signalPassFailure();

      if (failed(propagateConstantTiling(calcFunc, devFuncs)))
        return signalPassFailure();

      if (failed(collectDeadValues(calcFunc, devFuncs)))
        return signalPassFailure();

      if (failed(removeDeadValues(calcFunc)))
        return signalPassFailure();
    }
  }

private:
  /// Propagate constant tiling data to device function.
  ///
  /// Input:
  /// ```mlir
  ///    func.func @tiling_calc(...) {
  ///      ...
  ///      return %cst0, %dyn, %cst1 : i64, i64, i64
  ///    }
  ///
  ///    func.func @user(%arg0 : i64 {hacc.tiling_data}, ...)
  ///    attributes {hacc.tiling_func = "tiling_calc"} {
  ///      some_use(%arg0)
  ///      ...
  ///    }
  /// ```
  ///
  /// Output:
  /// ```mlir
  ///    func.func @tiling_calc(...) {
  ///      ...
  ///      return %cst0, %dyn, %cst1 : i64, i64, i64
  ///    }
  ///
  ///    func.func @user(%arg0 : i64 {hacc.tiling_data}, ...)
  ///    attributes {hacc.tiling_func = "tiling_calc"} {
  ///      %cloned_cst0 = ...
  ///      some_use(%cloned_cst0)
  ///      ...
  ///    }
  /// ```
  LogicalResult
  propagateConstantTiling(func::FuncOp &calcFunc,
                          SmallVector<func::FuncOp> &deviceFuncs) {
    auto returnOp = utils::getAssumedUniqueReturnOp(calcFunc);
    tilingFuncReturnIdxToConstantOp_.clear();
    tilingFuncDeadResults_ = BitVector(returnOp.getNumOperands(), false);
    // Extract the constant return value
    std::optional<int> constantTilingKeyIdx = std::nullopt;
    for (auto [idx, returnVal] : llvm::enumerate(returnOp.getOperands())) {
      if (!isa<arith::ConstantOp>(returnVal.getDefiningOp()))
        continue;

      auto maybeResultTypeAttr =
          calcFunc.getResultAttrOfType<hacc::KernelArgTypeAttr>(
              idx, hacc::KernelArgTypeAttr::name);
      if (maybeResultTypeAttr &&
          maybeResultTypeAttr.getArgType() == hacc::KernelArgType::kTilingKey)
        constantTilingKeyIdx = idx;

      auto constOp = returnVal.getDefiningOp<arith::ConstantOp>();
      LDBG("Tiling function's return is arith const op");
      tilingFuncReturnIdxToConstantOp_.insert({idx, constOp});
      tilingFuncDeadResults_.set(idx);
    }

    // If we cannot constantize all tiling data at once, and the tiling key
    // happens to be a constant, we need to keep the constant tiling key because
    // there is still going to be a tiling function and we have the constraint
    // that every tiling function returns a tiling key
    if (!tilingFuncDeadResults_.all() && constantTilingKeyIdx.has_value()) {
      tilingFuncDeadResults_.flip(*constantTilingKeyIdx);
      tilingFuncReturnIdxToConstantOp_.erase(*constantTilingKeyIdx);
    }

    for (auto deviceFunc : deviceFuncs) {
      LDBG("Processing device func:\n" << *deviceFunc);
      // Extract the tiling data index
      SmallVector<int> tilingDataDeviceIndex;
      for (auto [deviceIdx, arg] : llvm::enumerate(deviceFunc.getArguments())) {
        if (hacc::utils::isTilingArg(deviceFunc, deviceIdx))
          tilingDataDeviceIndex.push_back(deviceIdx);
      }
      assert(tilingDataDeviceIndex.size() == returnOp.getNumOperands());
      OpBuilder builder(&deviceFunc.getBody().front().front());

      // For each of device function clone all the constants
      for (auto [idx, constOp] : tilingFuncReturnIdxToConstantOp_) {
        auto replacementTarget =
            deviceFunc.getArgument(tilingDataDeviceIndex[idx]);
        auto constReplacement =
            cloneArithConstant(builder, constOp, replacementTarget.getType());
        // Replace uses of the corresponding argument in the device function
        replacementTarget.replaceAllUsesWith(constReplacement);
      }
      LDBG("Successfully modified device func:\n" << *deviceFunc);
    }

    return success();
  }

  LogicalResult collectDeadValues(func::FuncOp &calcFunc,
                                  SmallVector<func::FuncOp> &deviceFuncs) {
    // Dead indices is in terms of the tiling data
    BitVector deadTilingIndices = tilingFuncDeadResults_;
    for (auto deviceFunc : deviceFuncs) {
      // Currently, we assume that for device functions sharing the same
      // tiling function, the order of tiling data argument is exactly the same.
      LDBG("calc func: " << calcFunc.getSymName() << " dead tiling indicies: "
                         << utils::debugger::to_string(deadTilingIndices));
      if (failed(collectDeadValuesImpl(deviceFunc, deadTilingIndices)))
        return failure();
    }
    return success();
  }

  /// Collect dead operands/values related to the target device function.
  ///
  /// This function will find all callee of the device function, and recursively
  /// look up the callers, and record all dead operands/values along the way.
  ///
  /// ```mlir
  ///    func.func @deviceFunc(%arg0 : i64 {hacc.tiling_data}, ...)
  ///      <- dead argument indices = [1, 0, ...]
  ///
  ///    func.caller @caller(..., %arg3: i64 {hacc.tiling_data}) {
  ///      <- dead argument indices = [0, 0, 0, 1, ...]
  ///
  ///      func.call @deviceFunc(%arg3, ...)
  ///        <- dead operand indices = [1, 0, ...]
  ///    }
  /// ```
  LogicalResult collectDeadValuesImpl(func::FuncOp &deviceFunc,
                                      const BitVector &deadTilingIndices) {
    // Map dead tiling indices to the function argument indices in device
    // function
    BitVector deadIndicesInDeviceFunc =
        generateEraseIndicesForDeviceFunc(deviceFunc, deadTilingIndices);
    LDBG(
        "device func: " << deviceFunc.getSymName() << " dead arg indicies: "
                        << utils::debugger::to_string(deadIndicesInDeviceFunc));
    recordEraseIndicies(deviceFunc, deadIndicesInDeviceFunc);

    // Get callers of the device func.
    DenseMap<func::FuncOp, tiling::CallerInfo> workList;
    tiling::getCallerInfo(deviceFunc, getOperation(), workList);

    // Bail out on trivial case where there is no caller
    if (workList.empty())
      return success();

    // Call site's dead operand indicies are the same as the function's dead
    // argument indicies.
    BitVector deadIndicesInCallSite = deadIndicesInDeviceFunc;
    // Repeatedly search for the caller and call site, until there is no caller.
    DenseSet<Operation *> processed;
    while (!workList.empty()) {
      auto &[caller, callerInfo] = *(workList.begin());
      if (processed.contains(caller)) {
        LDBG("Cyclic call detected");
        return failure();
      }

      for (auto callSite : callerInfo.callSites) {
        LDBG(
            "call site: " << *callSite << " dead operand indicies: "
                          << utils::debugger::to_string(deadIndicesInCallSite));
        recordEraseIndicies(callSite, deadIndicesInCallSite);
      }

      deadIndicesInCallSite = generateEraseIndicesForFunctionCaller(
          caller, (*callerInfo.callSites.begin()), deadIndicesInCallSite);
      LDBG("caller: " << caller.getSymName() << " dead arg indicies: "
                      << utils::debugger::to_string(deadIndicesInCallSite));
      recordEraseIndicies(caller, deadIndicesInCallSite);

      tiling::getCallerInfo(caller, getOperation(), workList);
      processed.insert(caller);
      workList.erase(caller);
    }

    return success();
  }

  /// Remove dead operands/values related to the current device function being
  /// processed.
  LogicalResult removeDeadValues(func::FuncOp &calcFunc) {
    for (func::CallOp callSite : callSitesToModify_) {
      IRRewriter builder(callSite->getContext());
      auto deadIndices = op2DeadValues_.at(callSite);
      SmallVector<Value> newOperands;
      for (auto [idx, operand] : llvm::enumerate(callSite.getOperands())) {
        if (deadIndices.test(idx))
          continue;
        newOperands.push_back(operand);
      }
      builder.setInsertionPoint(callSite);
      builder.replaceOpWithNewOp<func::CallOp>(callSite, callSite.getCallee(),
                                               callSite->getResultTypes(),
                                               newOperands);
    }
    if (failed(removeCallerDeadValues()))
      return failure();

    if (failed(removeTilingFuncDeadValues(calcFunc)))
      return failure();

    // Reset all status
    op2DeadValues_.clear();
    callSitesToModify_.clear();
    callersToModify_.clear();
    tilingFuncDeadResults_.clear();
    tilingFuncReturnIdxToConstantOp_.clear();
    return success();
  }

  /// Remove dead function arguments of the callers of device functions.
  LogicalResult removeCallerDeadValues() {
    for (func::FuncOp caller : callersToModify_) {
      OpBuilder builder(caller);
      builder.setInsertionPointToStart(&caller.getFunctionBody().front());
      BitVector argumentsToErase = op2DeadValues_.at(caller);
      FuncArgIdx2TilingIdx callerInfo =
          func2TilingDataInfo_.at(caller.getSymName());
      // block arguments should have no users after modifying the call sites,
      // but if it's still in use, we need to propagate the constants.
      for (auto &ba : caller.getArguments()) {
        if (ba.use_empty())
          continue;
        auto argIdx = ba.getArgNumber();
        // if it's not a argument that we're deleting, continue
        if (!argumentsToErase.test(argIdx))
          continue;
        // if caller info doesn't record this argument as being constant,
        // report error
        if (!callerInfo.contains(argIdx)) {
          return caller.emitError()
                 << "Argument #" << argIdx
                 << " is not a constant tiling data but is set to be erased!";
        }
        auto tilingReturnIdx = callerInfo.at(argIdx);
        if (!tilingFuncReturnIdxToConstantOp_.contains(tilingReturnIdx)) {
          return caller.emitError()
                 << "Argument #" << argIdx << " is said to be Return value #"
                 << tilingReturnIdx
                 << " in the tiling function but the actual result is not "
                    "constant!";
        }
        // insert constant and replace all use
        auto *constOp = tilingFuncReturnIdxToConstantOp_.at(tilingReturnIdx);
        auto constReplacement =
            cloneArithConstant(builder, constOp, ba.getType());
        ba.replaceAllUsesWith(constReplacement);
      }
      caller.eraseArguments(argumentsToErase);
    }
    return success();
  }

  /// Remove dead return values in tiling func and modify all call sites.
  LogicalResult removeTilingFuncDeadValues(func::FuncOp &tilingFunc) {
    auto returnOp = utils::getAssumedUniqueReturnOp(tilingFunc);
    SmallVector<Value, 4> newReturnValues;
    SmallVector<Type, 4> newReturnTypes;
    SmallVector<ArrayRef<NamedAttribute>, 4> newFuncResultAttrs;
    for (auto [i, value] : llvm::enumerate(returnOp.getOperands())) {
      if (!tilingFuncDeadResults_[i]) {
        newReturnValues.push_back(value);
        newReturnTypes.push_back(value.getType());
        newFuncResultAttrs.push_back(tilingFunc.getResultAttrs(i));
      }
    }
    // construct new return op
    IRRewriter builder(&getContext());
    builder.setInsertionPoint(returnOp);
    builder.replaceOpWithNewOp<func::ReturnOp>(returnOp, newReturnValues);
    // update function type
    auto newFuncType = FunctionType::get(
        tilingFunc.getContext(), tilingFunc.getFunctionType().getInputs(),
        newReturnTypes);
    tilingFunc.setType(newFuncType);
    for (auto [i, attrs] : llvm::enumerate(newFuncResultAttrs))
      tilingFunc.setResultAttrs(i, attrs);

    // update the callers of the tiling function
    getOperation().walk([&](func::CallOp callOp) {
      if (callOp.getCallee() == tilingFunc.getName()) {
        SmallVector<Value, 4> newResults;
        for (auto [i, result] : llvm::enumerate(callOp.getResults())) {
          if (!tilingFuncDeadResults_[i])
            newResults.push_back(result);
        }
        OpBuilder builder(callOp);
        auto newCall = builder.create<func::CallOp>(callOp.getLoc(), tilingFunc,
                                                    callOp.getOperands());
        for (auto [oldResult, newResult] :
             llvm::zip(newResults, newCall.getResults()))
          oldResult.replaceAllUsesWith(newResult);
        callOp.erase();
      }
    });

    return success();
  }

  void recordEraseIndicies(Operation *op, const BitVector &erasedIndices) {
    if (auto call = dyn_cast<func::CallOp>(op)) {
      callSitesToModify_.insert(call);
    } else {
      auto func = cast<func::FuncOp>(op);
      callersToModify_.insert(func);
    }
    if (!op2DeadValues_.contains(op)) {
      op2DeadValues_.insert({op, erasedIndices});
      return;
    }
    // union dead value indices
    op2DeadValues_[op] |= erasedIndices;
  }

  /// Get the dead argument indices for device function.
  ///
  /// For example, tiling func is:
  /// ```mlir
  ///    func.func @tiling_calc(...) {
  ///      ...
  ///      return %cst0, %dyn, %cst1 : i64, i64, i64
  ///    }
  /// ```
  /// The dead (a.k.a constantized) tiling data indicies, represented using a
  /// bit vector, is [1, 0, 1]
  ///
  /// We assume that the tiling arguments in device function's input arguments
  /// has the exact same order as the return values of the tiling function.
  ///
  /// For example:
  /// ```mlir
  /// func.func @deviceFunc(%arg0 : tensor<f32>,
  ///                       %arg1: tensor<f32>,
  ///                       %arg2 : i64 {hacc.tiling_data}, ...)
  /// ```
  ///
  /// Then, in terms of the device function, the arguments to erase, represented
  /// using a bit vector, is [0, 0, 1, 0, 1, ...].
  BitVector
  generateEraseIndicesForDeviceFunc(func::FuncOp deviceFunc,
                                    const BitVector &deadTilingIndices) {
    int tilingIndex = 0;
    FuncArgIdx2TilingIdx deviceFuncInfo;
    BitVector erasedIndices(deviceFunc.getNumArguments(), false);
    for (auto [idx, arg] : llvm::enumerate(deviceFunc.getArguments())) {
      if (hacc::utils::isTilingArg(deviceFunc, idx)) {
        if (deadTilingIndices.test(tilingIndex)) {
          erasedIndices.set(idx);
          LDBG("idx - tilingIndex: " << idx << " " << tilingIndex);
          deviceFuncInfo.insert({idx, tilingIndex});
        }
        tilingIndex++;
      }
    }
    auto deviceFuncName = deviceFunc.getSymName();
    LDBG("Recoreded tiling data info for: " << deviceFuncName);
    func2TilingDataInfo_.insert({deviceFuncName, deviceFuncInfo});
    return erasedIndices;
  }

  /// Get the dead operand indices for functions calling the device function.
  ///
  /// For example, say that the device function's arguments to erase,
  /// represented using a bit vector, is [0, 0, 1, 0, 1, ...].
  ///
  /// ```mlir
  ///    func.caller @caller(%arg0, %arg1, %arg2, %arg3) {
  ///      func.call @deviceFunc(%arg0, %arg1, %arg3, ...)
  ///        <- dead operand indices =  [0, 0, 1, 0, 1, ...]
  ///    }
  /// ```
  ///
  /// Then, for the caller, the argument index to erase, represented using a bit
  /// vector, is [0, 0, 0, 1, 0, ...].
  BitVector generateEraseIndicesForFunctionCaller(
      func::FuncOp caller, func::CallOp callSite,
      const BitVector &deadTilingIndicesInCallSite) {
    // get callee's mapping from input argument to tiling
    FuncArgIdx2TilingIdx calleeInfo =
        func2TilingDataInfo_.at(callSite.getCallee());

    FuncArgIdx2TilingIdx callerInfo;
    auto funcArgs = SetVector<BlockArgument>{caller.getArguments().begin(),
                                             caller.getArguments().end()};
    BitVector deadIndicesInCaller(funcArgs.size(), false);
    for (auto [calleeArgIdx, calleeArg] :
         llvm::enumerate(callSite.getArgOperands())) {
      // if the operand is not a block argument...
      auto argInCaller = dyn_cast<BlockArgument>(calleeArg);
      if (!argInCaller)
        continue;
      // or not a function argument...
      if (!funcArgs.contains(argInCaller))
        continue;
      // or is not a dead tiling argument to erase, continue.
      if (!deadTilingIndicesInCallSite.test(calleeArgIdx))
        continue;
      auto callerArgIdx = argInCaller.getArgNumber();
      deadIndicesInCaller.set(callerArgIdx);
      callerInfo.insert({callerArgIdx, calleeInfo.at(calleeArgIdx)});
    }
    auto callerName = caller.getSymName();
    if (func2TilingDataInfo_.contains(callerName)) {
      auto currentInfo = func2TilingDataInfo_.at(callerName);
      LDBG("Updated tiling data info for: " << callerName);
      callerInfo.insert(currentInfo.begin(), currentInfo.end());
    } else {
      LDBG("Recoreded tiling data info for: " << callerName);
    }
    func2TilingDataInfo_[callerName] = callerInfo;
    return deadIndicesInCaller;
  }

private:
  /// Mapping from function argument index to the constantized tiling index
  /// in terms of the tiling function.
  using FuncArgIdx2TilingIdx = DenseMap<int, int>;

  DenseMap<Operation *, BitVector> op2DeadValues_;
  DenseMap<int, Operation *> tilingFuncReturnIdxToConstantOp_;
  DenseMap<StringRef, FuncArgIdx2TilingIdx> func2TilingDataInfo_;
  BitVector tilingFuncDeadResults_;
  SetVector<func::FuncOp> callersToModify_;
  SetVector<func::CallOp> callSitesToModify_;
};

} // namespace

std::unique_ptr<Pass> createConstantizeTilingDataPass() {
  return std::make_unique<ConstantizeTilingDataPass>();
}

} // namespace hfusion
} // namespace mlir
