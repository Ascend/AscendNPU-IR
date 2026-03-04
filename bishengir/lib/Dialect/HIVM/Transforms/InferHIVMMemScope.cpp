//===- InferHIVMMemScope.cpp - Infer Memory Scope for HIVM Ops ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
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

LogicalResult setMemSpaceForAllocs(Operation *sourceOp,
                                   MemScopeInferAndPropagateHelper &helper,
                                   const SmallVector<Value> &allocs,
                                   hivm::AddressSpaceAttr addressSpace) {
  if (allocs.empty()) {
    sourceOp->emitOpError("Cannot find root memref.alloc for this op.");
    return failure();
  }

  for (auto alloc : allocs) {
    if (failed(helper.Run(alloc, addressSpace))) {
      return sourceOp->emitOpError("Failed to infer/propagate memory scope.");
    }
  }

  return success();
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
          Type valType = val.getType();
          if (!isa<BaseMemRefType>(yieldType))
            return success();
          if (!isa<BaseMemRefType>(valType))
            return success();
          auto mtype = dyn_cast<BaseMemRefType>(yieldType);
          auto vtype = dyn_cast<BaseMemRefType>(valType);
          if (mtype.getElementType() != vtype.getElementType())
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
          if (!bbArg) {
            return failure();
          }
          auto yield = op.getTiedLoopYieldedValue(bbArg);
          if (!yield) {
            return failure();
          }
          auto afterArg = getTiedWhileBodyIterArg(op, &user);
          if (!afterArg) {
            return failure();
          }
          setBaseMemRefTypeScope(bbArg, memrefScope);
          setBaseMemRefTypeScope(yield->get(), memrefScope);
          setBaseMemRefTypeScope(afterArg, memrefScope);
          return success(propagateMemScopeToUsers(afterArg).succeeded() &&
                         propagateMemScopeToUsers(bbArg).succeeded() &&
                         propagateMemScopeToUsers(yield->get()).succeeded());
        })
        .Case<memref::SubViewOp, memref::ViewOp, memref::ReinterpretCastOp,
              memref::CastOp, memref::CollapseShapeOp, memref::ExpandShapeOp,
              memref::ReshapeOp, memref::TransposeOp,
              memref::ExtractStridedMetadataOp, memref::MemorySpaceCastOp>(
            [&](auto op) {
              auto result = op->getResult(0);
              setBaseMemRefTypeScope(result, memrefScope);
              return propagateMemScopeToUsers(result);
            })
        .Case<hivm::BitcastOp>([&](auto op) {
          auto result = op->getResult(0);
          setBaseMemRefTypeScope(result, memrefScope);
          return propagateMemScopeToUsers(result);
        })
        .Case<func::CallOp>([&](auto op) {
          // For function calls, we cannot propagate the memory scope because
          // we don't know the relationship between the inputs and results.
          // But we don't need to report failure because we can run propagation
          // for the results.
          func::FuncOp funcOp = llvm::dyn_cast<func::FuncOp>(
              SymbolTable::lookupNearestSymbolFrom(op, op.getCalleeAttr()));
          if (!funcOp || !funcOp->hasAttr(hivm::SIMTWrapperAttr::getMnemonic()))
            return success();
          
          auto argTypes = funcOp.getArgumentTypes().vec();
          for (size_t idx = 0; idx < op->getOperands().size(); idx++) {
            if (op->getOperand(idx) == val) {
              auto newType = getBaseMemRefTypeWithNewScope(
                  llvm::dyn_cast<BaseMemRefType>(argTypes[idx]), memrefScope);
              argTypes[idx] = newType;
              if (!funcOp->getRegion(0).empty()) {
                funcOp.front().getArgument(idx).setType(newType);
              }
            }
          }
          auto newFt = funcOp.getFunctionType().clone(argTypes,
                                                      funcOp->getResultTypes());
          funcOp.setFunctionType(newFt);

          return success();
        })
        .Case<gpu::LaunchFuncOp>([&](auto op) {
          // Same as above
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
    return propagateMemScopeToUsers(operand);
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
  auto allocsA = utils::tracebackMemRefVec(mA->get());
  auto allocsB = utils::tracebackMemRefVec(mB->get());
  auto allocsC = utils::tracebackMemRefVec(mC->get());

  MemScopeInferAndPropagateHelper helper;
  auto l1SpaceAttr =
      AddressSpaceAttr::get(op->getContext(), hivm::AddressSpace::L1);
  // For MmadL1Op, operand mA should be in L1.
  if (failed(setMemSpaceForAllocs(op, helper, allocsA, l1SpaceAttr)))
    return op->emitOpError("Failed to infer/propagate memory scope for mA");

  // For MmadL1Op, operand mB should be in L1.
  if (failed(setMemSpaceForAllocs(op, helper, allocsB, l1SpaceAttr)))
    return op->emitOpError("Failed to infer/propagate memory scope for mB");

  auto l0cSpaceAttr =
      AddressSpaceAttr::get(op->getContext(), hivm::AddressSpace::L0C);
  // For MmadL1Op, operand mC should be in L0C.
  if (failed(setMemSpaceForAllocs(op, helper, allocsC, l0cSpaceAttr)))
    return op->emitOpError("Failed to infer/propagate memory scope for mC");

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
  auto ubSpaceAttr =
      AddressSpaceAttr::get(op->getContext(), hivm::AddressSpace::UB);
  auto args = op.getArguments();
  for (auto arg : args) {
    if (!isa<BaseMemRefType>(arg.getType())) {
      continue;
    }

    if (op->hasAttr(hivm::VectorFunctionAttr::name)) {
      if (failed(helper.Run(arg, ubSpaceAttr)))
        return op->emitOpError()
               << "Failed to propagate UB memory scope for argument # in VF"
               << arg.getArgNumber();
    } else if (failed(helper.Run(arg, gmSpaceAttr))) {
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

LogicalResult hivm::inferAndPropagateMemScopeForGpuFunc(gpu::GPUFuncOp op) {
  MemScopeInferAndPropagateHelper helper;
  auto gmSpaceAttr =
      AddressSpaceAttr::get(op->getContext(), hivm::AddressSpace::GM);

  auto args = op.getArguments();
  for (auto arg : args) {
    if (!isa<BaseMemRefType>(arg.getType())) {
      continue;
    }

    // TODO: handle case when ub arguments are passed in the GPUFuncOp
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

// 1. Infer the allocation type based on memoryspace (ub/l1).
// 2. If there is no memoryspace and the function is aic/aiv,
// allocate the memory to the corresponding ub or l1;
// otherwise, allocate the memory to ub by default.
LogicalResult
hivm::inferAndPropagateMemScopeForAlloc(memref::AllocOp op,
                                        hivm::AddressSpace space) {
  LDBG("Begin infer and propagate memory scope for: " << *op);
  auto memorySpace = op.getType().getMemorySpace();
  MemScopeInferAndPropagateHelper helper;
  if (memorySpace) {
    auto toAddrSpace =
        cast<hivm::AddressSpaceAttr>(memorySpace).getAddressSpace();
    if ((toAddrSpace != hivm::AddressSpace::UB) &&
        (toAddrSpace != hivm::AddressSpace::L1))
      return success();
    return helper.propagateMemScopeToUsers(op->getResults()[0]);
  }
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

  SmallVector<gpu::GPUFuncOp> gpuFuncList;
  getOperation()->walk([&](gpu::GPUModuleOp gpuModule) {
    gpuModule->walk([&](gpu::GPUFuncOp gpuFunc) -> void {
      gpuFuncList.push_back(gpuFunc);
    });
  });

  for (auto func : gpuFuncList) {
    if (failed(inferAndPropagateMemScopeForGpuFunc(func)))
      signalPassFailure();
  }
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
    hivm::AddressSpace space = hivm::AddressSpace::UB;
    if (funcCoreType.has_value() &&
        funcCoreType.value() == TFuncCoreType::AIC) {
      space = hivm::AddressSpace::L1;
    }
    func->walk([&](memref::AllocOp op) {
      if (failed(hivm::inferAndPropagateMemScopeForAlloc(op, space))) {
        signalPassFailure();
      }
    });
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
