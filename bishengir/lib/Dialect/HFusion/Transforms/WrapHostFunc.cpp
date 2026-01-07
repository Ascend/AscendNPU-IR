//===--- WrapHostFunc.cpp -- create wrappers for host related functions ---===//
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
//
// This file implements wrappers around host related functions
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"

#include <cassert>
#include <optional>
#include <utility>

#define DEBUG_TYPE "hfusion-wrap-host-func"

namespace mlir {
#define GEN_PASS_DEF_WRAPHOSTFUNC
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace mlir {
using namespace hfusion;

struct WrapHostFuncPass : public impl::WrapHostFuncBase<WrapHostFuncPass> {
public:
  explicit WrapHostFuncPass(const WrapHostFuncOptions &options)
      : WrapHostFuncBase(options) {}
  void runOnOperation() override;

private:
  using CallInfo = std::pair<func::FuncOp, func::CallOp>;

  SmallVector<Value> recursiveClone(IRRewriter &rewriter,
                                    const SmallVector<Value> &values,
                                    Block *clonePoint, IRMapping &curMapping) {
    SmallVector<Value> newValues;
    for (auto value : values) {
      if (isa<BlockArgument>(value)) {
        newValues.push_back(curMapping.lookup(value));
        continue;
      }
      auto *defOperation = value.getDefiningOp();
      assert(defOperation != nullptr);
      auto operands = defOperation->getOperands();
      auto clonedValues =
          recursiveClone(rewriter, operands, clonePoint, curMapping);
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToEnd(clonePoint);
      IRMapping mapping;
      mapping.map(operands, clonedValues);
      auto *clonedOp = rewriter.clone(*defOperation, mapping);
      newValues.push_back(
          clonedOp->getResult(cast<OpResult>(value).getResultNumber()));
    }
    return newValues;
  }

  SmallVector<Value> populateOperandsToTrace(OperandRange operands,
                                             func::FuncOp toBeWrapperFuncOp,
                                             func::FuncOp deviceFuncOp) {
    SmallVector<Value> operandsToTrace;
    // iterate through every func argument in toBeWrapperFuncOp, try to find the
    // matching func argument in deviceFuncOp if found, store it into
    // operandsToTrace.
    for (unsigned hostArgIdx = 0;
         hostArgIdx < toBeWrapperFuncOp.getFunctionType().getNumInputs();
         ++hostArgIdx) {
      auto hostArgDictAttr = toBeWrapperFuncOp.getArgAttrDict(hostArgIdx);
      assert(hostArgDictAttr &&
             "every argument should have at least one arg attribute");

      auto hostArgType = cast<hacc::KernelArgTypeAttr>(
                             hostArgDictAttr.get(hacc::KernelArgTypeAttr::name))
                             .getArgType();
      bool argMatch = false;
      unsigned idxMappingToHost = -1;
      for (unsigned deviceArgIdx = 0;
           deviceArgIdx < deviceFuncOp.getFunctionType().getNumInputs();
           ++deviceArgIdx) {
        // find the hostArgIdx that matches deviceArgType at deviceArgidx
        auto deviceArgDictAttr = deviceFuncOp.getArgAttrDict(deviceArgIdx);
        if (!hacc::utils::isKernelArg(deviceFuncOp, deviceArgIdx,
                                      hostArgType)) {
          continue;
        }
        // early exit if the number of arg attrs does not match
        if (hostArgDictAttr.size() != deviceArgDictAttr.size())
          continue;

        switch (hostArgType) {
        case hacc::KernelArgType::kInput: {
          auto deviceInputIdx =
              hacc::getHACCInputIdx(deviceFuncOp, deviceArgIdx);
          auto hostInputIdx =
              hacc::getHACCInputIdx(toBeWrapperFuncOp, hostArgIdx);
          assert(deviceInputIdx.has_value() && hostInputIdx.has_value());
          argMatch = (*deviceInputIdx) == (*hostInputIdx);
          break;
        }
        case hacc::KernelArgType::kOutput: {
          auto deviceOutputIdx =
              hacc::getHACCOuputIdx(deviceFuncOp, deviceArgIdx);
          auto hostOutputIdx =
              hacc::getHACCOuputIdx(toBeWrapperFuncOp, hostArgIdx);
          assert(deviceOutputIdx.has_value() && hostOutputIdx.has_value());
          argMatch = (*deviceOutputIdx) == (*hostOutputIdx);
          break;
        }
        default:
          argMatch = true;
        }
        if (argMatch) {
          idxMappingToHost = deviceArgIdx;
          break;
        }
      }
      if (!argMatch) {
        continue;
      }
      operandsToTrace.push_back(operands[idxMappingToHost]);
    }
    return operandsToTrace;
  }

  void simplifyWrapperFunction(Block *entryBlock, FunctionType wrapperFuncType,
                               func::FuncOp wrapperFuncOp) {
    SmallVector<int> unusedArgumentInd;
    for (BlockArgument blockArg : entryBlock->getArguments()) {
      if (!blockArg.use_empty()) {
        continue;
      }
      unusedArgumentInd.push_back(blockArg.getArgNumber());
    }
    if (!unusedArgumentInd.empty()) {
      SmallVector<Type> allUsedFunctionTypeInputs(wrapperFuncType.getInputs());
      for (auto &idx : reverse(unusedArgumentInd)) {
        // erase block argument, populate new function type inputs
        allUsedFunctionTypeInputs.erase(allUsedFunctionTypeInputs.begin() +
                                        idx);
        entryBlock->eraseArgument(idx);
      }
      FunctionType allUsedFunctionType = FunctionType::get(
          wrapperFuncOp.getContext(), allUsedFunctionTypeInputs,
          wrapperFuncType.getResults());
      wrapperFuncOp.setFunctionType(allUsedFunctionType);
    }
  }

  // clone all def Op before certain clone point (specified by the operands)
  // into a new funcOp, and attribute setup
  template <typename T>
  void createHostWrapperFunctions(IRRewriter &rewriter,
                                  func::FuncOp toBeWrapperFuncOp,
                                  func::FuncOp deviceFuncOp,
                                  CallInfo hostCallInfo,
                                  bool removeUnusedFuncArguments = false) {
    MLIRContext *ctx = deviceFuncOp->getContext();
    auto [hostEntryFunc, deviceFuncCall] = hostCallInfo;

    auto wrapperFuncName = hacc::constructHostFunctionName(
        hostEntryFunc.getSymName().str(),
        *hacc::symbolizeHostFuncType(T::getMnemonic()));
    auto toBeWrapperFuncName = toBeWrapperFuncOp.getSymName();

    // replace orig host-related func name with wrapper name to be created
    deviceFuncOp->setAttr(T::name, T::get(ctx, wrapperFuncName));

    // when several kernels share host functions(i.e. tiling func),
    // avoid duplicates
    if (SymbolTable::lookupNearestSymbolFrom(
            deviceFuncOp, SymbolRefAttr::get(ctx, wrapperFuncName))) {
      return;
    }
    auto operands = deviceFuncCall->getOperands();

    // determine the operands that need backtracing
    SmallVector<Value> operandsToTrace =
        populateOperandsToTrace(operands, toBeWrapperFuncOp, deviceFuncOp);

    rewriter.setInsertionPointAfter(toBeWrapperFuncOp);

    // new function type's inputs match host entry func's inputs, and results
    // match results of the function to be wrapped
    FunctionType newFuncType =
        FunctionType::get(ctx, hostEntryFunc.getFunctionType().getInputs(),
                          toBeWrapperFuncOp.getFunctionType().getResults());
    func::FuncOp newFuncOp = rewriter.create<func::FuncOp>(
        toBeWrapperFuncOp->getLoc(), wrapperFuncName, newFuncType);
    Block *entryBlock = newFuncOp.addEntryBlock();
    // create mapping between host entry block arguments and wrapper func block
    // arguments
    IRMapping mapping;
    for (auto [oldIn, newIn] :
         llvm::zip(hostEntryFunc.getFunctionBody().getArguments(),
                   entryBlock->getArguments())) {
      mapping.map(oldIn, newIn);
    }
    // backtrace operand used in func.call, clone Ops along the way
    auto vals = recursiveClone(rewriter, operandsToTrace, entryBlock, mapping);
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToEnd(entryBlock);
    auto callee =
        rewriter.create<func::CallOp>(newFuncOp->getLoc(), toBeWrapperFuncName,
                                      newFuncType.getResults(), vals);
    rewriter.create<func::ReturnOp>(newFuncOp->getLoc(), callee->getResults());

    if (removeUnusedFuncArguments) {
      simplifyWrapperFunction(entryBlock, newFuncType, newFuncOp);
    }

    // wrapper func attribute setup
    hacc::utils::setHost(newFuncOp);
    newFuncOp->setAttr(
        hacc::HostFuncTypeAttr::name,
        toBeWrapperFuncOp->getAttr(hacc::HostFuncTypeAttr::name));

    // remove orig host-related func attributes (i.e. function_kind) on old host
    // func side
    toBeWrapperFuncOp->removeAttr(hacc::HostFuncTypeAttr::name);
  }

  template <typename T>
  void addWrapper(func::FuncOp deviceFuncOp, CallInfo hostCallInfo,
                  bool removeUnusedFuncArguments = false) {
    if (!deviceFuncOp->hasAttr(T::name)) {
      return;
    }
    auto funcAttr = deviceFuncOp->template getAttrOfType<T>(T::name);
    StringRef toBeWrapperFuncName = funcAttr.getFuncNameStr();
    auto toBeWrapperFuncOp =
        dyn_cast<func::FuncOp>(SymbolTable::lookupNearestSymbolFrom(
            deviceFuncOp, SymbolRefAttr::get(deviceFuncOp->getContext(),
                                             toBeWrapperFuncName)));
    assert(toBeWrapperFuncOp && "Cannot find to-be-wrapped host function");

    IRRewriter rewriter(deviceFuncOp->getContext());
    createHostWrapperFunctions<T>(rewriter, toBeWrapperFuncOp, deviceFuncOp,
                                  hostCallInfo, removeUnusedFuncArguments);
  }

  std::optional<CallInfo> findHostFuncOp(func::FuncOp funcOp) {
    auto maybeUses = funcOp.getSymbolUses(getOperation());
    if (!maybeUses.has_value()) {
      return std::nullopt;
    }
    for (SymbolTable::SymbolUse use : *maybeUses) {
      func::CallOp call = cast<func::CallOp>(use.getUser());
      func::FuncOp callSite = call->getParentOfType<func::FuncOp>();
      assert(hacc::utils::isHost(callSite));
      auto callSiteFusionKind = tryGetFusionKind(callSite);
      if (callSiteFusionKind.has_value() &&
          *callSiteFusionKind != FusionKind::Unknown) {
        return std::make_pair(callSite, call);
      }
    }
    return std::nullopt;
  }
};

void WrapHostFuncPass::runOnOperation() {
  auto module = getOperation();
  IRRewriter rewriter(module->getContext());

  module.walk([&](func::FuncOp funcOp) {
    if (!hacc::utils::isDeviceEntry(funcOp)) {
      return WalkResult::skip();
    }
    auto hostCallInfo = findHostFuncOp(funcOp);
    if (!hostCallInfo.has_value()) {
      return WalkResult::skip();
    }

    addWrapper<hacc::TilingFunctionAttr>(funcOp, *hostCallInfo);
    addWrapper<hacc::InferOutputShapeFunctionAttr>(funcOp, *hostCallInfo);
    addWrapper<hacc::InferWorkspaceShapeFunctionAttr>(funcOp, *hostCallInfo);
    addWrapper<hacc::GetTilingStructSizeFunctionAttr>(funcOp, *hostCallInfo,
                                                      removeUnusedArguments);
    addWrapper<hacc::InferSyncBlockLockNumFunctionAttr>(funcOp, *hostCallInfo);
    addWrapper<hacc::InferSyncBlockLockInitFunctionAttr>(funcOp, *hostCallInfo);
    addWrapper<hacc::InferTaskTypeFunctionAttr>(funcOp, *hostCallInfo);
    return WalkResult::advance();
  });
}

} // namespace mlir

std::unique_ptr<Pass>
mlir::hfusion::createWrapHostFuncPass(const WrapHostFuncOptions &options) {
  return std::make_unique<WrapHostFuncPass>(options);
}
