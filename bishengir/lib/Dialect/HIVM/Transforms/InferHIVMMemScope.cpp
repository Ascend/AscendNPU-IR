//===- InferHIVMMemScope.cpp - Infer Memory Scope for HIVM Ops ------------===//
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

#include "bishengir/Dialect/HIVM/Transforms/InferHIVMMemScope.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "hivm-infer-mem-scope"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_INFERHIVMMEMSCOPE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace hivm;

namespace {
bool isSingleResultPropagatableMemrefOp(Operation *op) {
  if (!op)
    return false;
  if (isa<ViewLikeOpInterface>(op))
    return true;
  if (isa<memref::TransposeOp, hivm::BitcastOp, arith::SelectOp>(op))
    return true;
  return false;
}

static BlockArgument getTiedWhileBodyIterArg(scf::WhileOp op, OpOperand *opOperand) {
  auto argsMutable = op.getInitsMutable();
  auto *it = llvm::find(argsMutable, *opOperand);
  if (it == argsMutable.end())
    return {};
  return op.getAfterArguments()[std::distance(argsMutable.begin(), it)];
}
} // namespace

LogicalResult
MemScopeInferAndPropagateHelper::propagateMemScopeToUsers(Value val) {
  // Get new memory scope from result.
  auto memrefScope = getHIVMAddressSpaceAttr(val.getType());
  // This function propagates the type change of an SSA result to the operation
  // that uses it. The result type of the updated operation might be affected,
  // so we need to cascade the change.
  auto propagateFn = [&](OpOperand &user) -> LogicalResult {
    Operation *userDefiningOp = user.getOwner();
    return TypeSwitch<Operation *, LogicalResult>(userDefiningOp)
        .Case<scf::YieldOp>([&](scf::YieldOp op) {
          Operation *parentOp = op->getParentOp();
          auto yieldResult = op.getOperand(user.getOperandNumber());
          auto parentResult = parentOp->getResult(user.getOperandNumber());

          Type yieldType = yieldResult.getType();
          if (!isa<BaseMemRefType>(yieldType))
            return success();
          setBaseMemRefTypeScope(parentResult, memrefScope);
          if (failed(propagateMemScopeToUsers(parentResult))) {
            return failure();
          }
          return success();
        })
        .Case<scf::ForOp>([&](scf::ForOp op) {
          auto result = op.getTiedLoopResult(&user);
          setBaseMemRefTypeScope(result, memrefScope);
          auto bbArg = op.getTiedLoopRegionIterArg(&user);
          setBaseMemRefTypeScope(bbArg, memrefScope);
          return success(propagateMemScopeToUsers(bbArg).succeeded() &&
                         propagateMemScopeToUsers(result).succeeded());
        })
        .Case<scf::WhileOp>([&](scf::WhileOp op) {
          auto bbArg = op.getTiedLoopRegionIterArg(&user);
          auto yield = op.getTiedLoopYieldedValue(bbArg);
          auto afterArg = getTiedWhileBodyIterArg(op, &user);
          setBaseMemRefTypeScope(bbArg, memrefScope);
          setBaseMemRefTypeScope(yield->get(), memrefScope);
          setBaseMemRefTypeScope(afterArg, memrefScope);
          return success(propagateMemScopeToUsers(afterArg).succeeded() &&
                         propagateMemScopeToUsers(bbArg).succeeded() &&
                         propagateMemScopeToUsers(yield->get()).succeeded());
        })
        .Case<func::CallOp>([&](auto op) {
          // For function calls, we cannot propagate the memory scope because
          // we don't know the relationship between the inputs and results.
          // But we don't need to report failure because we can run propagation
          // for the results.
          return success();
        })
        .Default([&](Operation *op) {
          // Don't need to update Ops that don't have results.
          if (op->getNumResults() == 0) {
            return success();
          }
          // Or results that are not memrefs.
          if (llvm::none_of(op->getResults(), [&](OpResult result) {
                return isa<MemRefType>(result.getType());
              })) {
            return success();
          }
          if (op->getNumResults() == 1 &&
              isSingleResultPropagatableMemrefOp(op)) {
            auto result = op->getResult(0);
            setBaseMemRefTypeScope(result, memrefScope);
            return propagateMemScopeToUsers(result);
          }
          op->emitOpError("Unsupported user for root alloc op.");
          return failure();
        });
  };
  // Iterate over the users of the val.
  for (OpOperand &user : val.getUses()) {
    // Update the type of the result that corresponds to the operand.
    if (failed(propagateFn(user))) {
      return failure();
    }
  }
  return success();
}

LogicalResult
MemScopeInferAndPropagateHelper::Run(Value operand,
                                     const AddressSpaceAttr &targetMemScope) {
  auto memRefType = dyn_cast<BaseMemRefType>(operand.getType());
  if (!memRefType) {
    return failure();
  }

  auto memSpace = memRefType.getMemorySpace();
  if (memSpace) {
    return success();
  }

  // Update its scope.
  setBaseMemRefTypeScope(operand, targetMemScope);

  // Propagate the new memref type to its users.
  return propagateMemScopeToUsers(operand);
}

namespace {
struct InferHIVMMemScopePass
    : public impl::InferHIVMMemScopeBase<InferHIVMMemScopePass> {
  void runOnOperation() override;

private:
  LogicalResult fixDeviceCallSite(func::FuncOp op);
  LogicalResult fixHostFuncSignature(func::FuncOp op);
};
} // namespace

LogicalResult hivm::inferAndPropagateMemScopeForMmadL1(hivm::MmadL1Op op) {
  if (!op.hasPureBufferSemantics()) {
    return op->emitOpError("Run infer memory scope after bufferization.");
  }

  auto *mA = op.getDpsInputOperand(0);
  auto *mB = op.getDpsInputOperand(1);
  auto *mC = op.getDpsInitOperand(0);

  // mA, mB and mC must originate from an AllocOP
  auto allocA = utils::tracebackMemRefToAlloc(mA->get());
  auto allocB = utils::tracebackMemRefToAlloc(mB->get());
  auto allocC = utils::tracebackMemRefToAlloc(mC->get());

  if (!allocA.has_value()) {
    emitError(op.getLoc())
        << "Cannot find root memref.alloc for mA of this op.";
    return failure();
  }
  if (!allocB.has_value()) {
    emitError(op.getLoc())
        << "Cannot find root memref.alloc for mB of this op.";
    return failure();
  }
  if (!allocC.has_value()) {
    emitError(op.getLoc())
        << "Cannot find root memref.alloc for mC of this op.";
    return failure();
  }

  auto l1SpaceAttr =
      AddressSpaceAttr::get(op->getContext(), hivm::AddressSpace::L1);
  auto l0cSpaceAttr =
      AddressSpaceAttr::get(op->getContext(), hivm::AddressSpace::L0C);

  MemScopeInferAndPropagateHelper helper;

  // For MmadL1Op, operand mA should be in L1.
  if (failed(helper.Run(*allocA, l1SpaceAttr))) {
    return op->emitOpError("Failed to infer/propagate memory scope for mA");
  }
  LDBG("IR after setting mem scope for mA:\n"
       << *(op->getParentOfType<ModuleOp>()));

  // For MmadL1Op, operand mB should be in L1.
  if (failed(helper.Run(*allocB, l1SpaceAttr))) {
    return op->emitOpError("Failed to infer/propagate memory scope for mB");
  }
  LDBG("IR after setting mem scope for mB:\n"
       << *(op->getParentOfType<ModuleOp>()));

  // For MmadL1Op, operand mC should be in L0C.
  if (failed(helper.Run(*allocC, l0cSpaceAttr))) {
    return op->emitOpError("Failed to infer/propagate memory scope for mC");
  }
  LDBG("IR after setting mem scope for mC:\n"
       << *(op->getParentOfType<ModuleOp>()));

  if (auto bias = op.getPerChannelBias()) {
    auto allocBias = utils::tracebackMemRefToAlloc(bias);
    if (!allocBias.has_value()) {
      emitError(op.getLoc())
          << "Cannot find root memref.alloc for bias of this op.";
      return failure();
    }

    // For MmadL1Op, operand bias should be in L1.
    if (failed(helper.Run(allocBias.value(), l1SpaceAttr))) {
      return op->emitOpError("Failed to infer/propagate memory scope for bias");
    }
    LDBG("IR after setting mem scope for bias:\n"
         << *(op->getParentOfType<ModuleOp>()));
  }

  return success();
}

LogicalResult InferHIVMMemScopePass::fixDeviceCallSite(func::FuncOp op) {
  LDBG("Begin fixing call site for " << op.getSymName());

  MemScopeInferAndPropagateHelper helper;
  auto maybeSymbolUses = op.getSymbolUses(getOperation());
  if (!maybeSymbolUses.has_value())
    llvm::report_fatal_error("maybeSymbolUses is null");
  SymbolTable::UseRange uses = maybeSymbolUses.value();
  for (SymbolTable::SymbolUse use : uses) {
    func::CallOp call = cast<func::CallOp>(use.getUser());
    // propagate call operand's memory scope
    for (auto [idx, callOperand] : llvm::enumerate(call.getArgOperands())) {
      if (!isa<BaseMemRefType>(callOperand.getType()))
        continue;

      auto funcOperandType = op.getFunctionType().getInput(idx);
      if (!isa<BaseMemRefType>(funcOperandType))
        continue;

      LDBG("call operand: " << callOperand);
      if (failed(helper.Run(utils::tracebackMemRef(callOperand),
                            getHIVMAddressSpaceAttr(funcOperandType)))) {
        return op->emitOpError()
               << "Failed to propagate memory scope for operand "
               << callOperand;
      }
      LDBG("call operand after: " << callOperand);
    }
    // propagate call return value memory scope
    for (auto [idx, returnValue] : llvm::enumerate(call->getResults())) {
      if (!isa<BaseMemRefType>(returnValue.getType()))
        continue;

      auto funcReturnType = op.getFunctionType().getResult(idx);
      if (!isa<BaseMemRefType>(funcReturnType))
        continue;

      if (failed(helper.Run(returnValue,
                            getHIVMAddressSpaceAttr(funcReturnType)))) {
        return op->emitOpError()
               << "Failed to propagate memory scope for result " << returnValue;
      }
    }
  }
  return success();
}

/// Update the function type for the host function.
///
/// Because we propagate information from the call site to the caller, we only
/// updated the memref type of the BlockArgument of or the return operation
/// within the function (if they are updated at all). So we need to use those
/// information to update the function's type.
LogicalResult InferHIVMMemScopePass::fixHostFuncSignature(func::FuncOp op) {
  // Skip external host functions because we know nothing about it.
  if (op.isExternal())
    return success();

  func::ReturnOp returnOp = utils::getAssumedUniqueReturnOp(op);
  if (!returnOp)
    return failure();

  SmallVector<Type> newArgsType(llvm::map_to_vector(
      op.getArguments(), [](const BlockArgument &ba) { return ba.getType(); }));
  SmallVector<Type> newReturnType(llvm::map_to_vector(
      returnOp.getOperandTypes(), [](const Type &type) { return type; }));
  auto newFt = op.getFunctionType().clone(newArgsType, newReturnType);
  op.setFunctionType(newFt);
  return success();
}

LogicalResult inferAndPropagateMemScopeForExternFunc(func::FuncOp op) {
  if (!op.isExternal())
    return failure();

  auto gmSpaceAttr =
      AddressSpaceAttr::get(op->getContext(), hivm::AddressSpace::GM);
  LDBG("Begin infer and propagate memory scope for extern func"
       << op.getSymName());
  auto newArgTypes = SmallVector<Type>(op.getArgumentTypes());
  for (auto &argType : newArgTypes) {
    // If not base memref and already has memspace then skip
    if (auto memrefType = dyn_cast<BaseMemRefType>(argType)) {
      if (memrefType.getMemorySpace())
        continue;
      argType = getBaseMemRefTypeWithNewScope(memrefType, gmSpaceAttr);
    }
  }
  // For extern functions that have results, we assume that the memory scope
  // is Global Memory.
  auto newReturnTypes = SmallVector<Type>(op.getResultTypes());
  for (auto &resultType : newReturnTypes) {
    // If not base memref and already has memspace then skip
    if (auto memrefType = dyn_cast<BaseMemRefType>(resultType)) {
      if (memrefType.getMemorySpace())
        continue;
      resultType = getBaseMemRefTypeWithNewScope(memrefType, gmSpaceAttr);
    }
  }
  auto newFt = op.getFunctionType().clone(newArgTypes, newReturnTypes);
  op.setFunctionType(newFt);
  return success();
}

LogicalResult hivm::inferAndPropagateMemScopeForFunc(func::FuncOp op) {
  if (op.isExternal())
    return inferAndPropagateMemScopeForExternFunc(op);

  LDBG("Begin infer and propagate memory scope for func" << op.getSymName());
  MemScopeInferAndPropagateHelper helper;
  auto gmSpaceAttr =
      AddressSpaceAttr::get(op->getContext(), hivm::AddressSpace::GM);
  auto args = op.getArguments();
  for (auto arg : args) {
    if (!isa<BaseMemRefType>(arg.getType())) {
      continue;
    }
    if (failed(helper.Run(arg, gmSpaceAttr))) {
      return op->emitOpError()
             << "Failed to propagate memory scope for argument #"
             << arg.getArgNumber();
    }
  }
  if (!args.empty()) {
    auto newFt = op.getFunctionType().clone(
        op.getBody().front().getArgumentTypes(), op.getResultTypes());
    op.setFunctionType(newFt);
  }
  if (op->getNumResults() > 0)
    op.emitWarning()
        << "non-externl function has return value after bufferization!";

  return success();
}

LogicalResult
hivm::inferAndPropagateMemScopeForPointerCast(hivm::PointerCastOp op) {
  LDBG("Begin infer and propagate memory scope for:" << op);

  auto gmSpaceAttr =
      AddressSpaceAttr::get(op->getContext(), hivm::AddressSpace::GM);
  MemScopeInferAndPropagateHelper helper;
  auto res = op.getResult();

  if (util::isGMPointerCastOp(op)) {
    if (failed(helper.Run(res, gmSpaceAttr))) {
      return op->emitOpError(
          "Failed to propagate memory scope for PointerCastOp");
    }
  }
  return success();
}

LogicalResult hivm::inferAndPropagateMemScopeForAlloc(memref::AllocOp op, hivm::AddressSpace space) {
  LDBG("Begin infer and propagate memory scope for: " << *op);
  auto memorySpace = op.getType().getMemorySpace();
  if (memorySpace) {
    return success();
  }

  MemScopeInferAndPropagateHelper helper;
  auto spaceAttr = AddressSpaceAttr::get(op->getContext(), space);
  if (failed(helper.Run(op, spaceAttr))) {
    return op->emitOpError("Failed to propagate memory scope for allocOp");
  }
  return success();
}

void InferHIVMMemScopePass::runOnOperation() {
  SmallVector<func::FuncOp> deviceFuncList;
  SetVector<StringRef> deviceFuncNames;
  SmallVector<func::FuncOp> hostFuncList;
  getOperation()->walk([&](func::FuncOp func) {
    if (!hacc::utils::isHost(func)) {
      deviceFuncList.push_back(func);
      deviceFuncNames.insert(func.getSymName());
      return;
    }
    hostFuncList.push_back(func);
  });

  // Infer and propagate memory scope for device functions.
  for (auto func : deviceFuncList) {
    // Set the memory scope of values related to `hivm::MmadL1Op` to L1 or L0C.
    // Here shouldn't contain `hivm::BatchMmadL1Op` which has been decomposed.
    func->walk([&](mlir::hivm::MmadL1Op op) {
      if (failed(hivm::inferAndPropagateMemScopeForMmadL1(op)))
        signalPassFailure();
    });

    // Set device function arguments' memory scope to GM.
    if (failed(hivm::inferAndPropagateMemScopeForFunc(func)))
      signalPassFailure();

    // Propagate the memory scope by the pointer cast's annotation mark
    func->walk([&](hivm::PointerCastOp op) {
      if (failed(hivm::inferAndPropagateMemScopeForPointerCast(op)))
        signalPassFailure();
    });

    // Finally, set the remaining memory scope in the device kernel.
    auto funcCoreType = queryFuncCoreType(func);
    if (funcCoreType.has_value()) {
      hivm::AddressSpace space = hivm::AddressSpace::UB;
      if (funcCoreType.value() == TFuncCoreType::AIC) {
        space = hivm::AddressSpace::L1;
      }
      func->walk([&](memref::AllocOp op) {
        if (failed(hivm::inferAndPropagateMemScopeForAlloc(op, space))) {
          signalPassFailure();
        }
      });
    }
  }

  for (auto func : deviceFuncList) {
    if (failed(fixDeviceCallSite(func)))
      signalPassFailure();
  }

  for (auto func : hostFuncList) {
    if (failed(fixHostFuncSignature(func)))
      signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::hivm::createInferHIVMMemScopePass() {
  return std::make_unique<InferHIVMMemScopePass>();
}
