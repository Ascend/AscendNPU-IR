//===------------ IRTranslator.cpp ----Sync information collection---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Transforms/InjectSync/IRTranslator.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncCommon.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <memory>

#define DEBUG_TYPE "hivm-inject-sync"

using namespace mlir;
using namespace mlir::hivm;

void IRTranslator::Build() {
  Region &funcRegion = func_.getBody();
  UpdateKernelArgMemInfo();
  // Recursively obtaining IR information.
  RecursionIR(&funcRegion);
}

static bool skippableWorkspaceArg(BlockArgument workspaceArg) {
  auto argUses = workspaceArg.getUses();
  size_t numUses =
      static_cast<size_t>(std::distance(argUses.begin(), argUses.end()));
  bool multipleUses = numUses > 1;

  for (const auto &use : argUses) {
    if (auto *user = use.getOwner()) {
      auto resUsers = user->getUsers();
      multipleUses |= std::distance(resUsers.begin(), resUsers.end()) > 1;
      if (multipleUses) {
        break;
      }
    }
  }

  return !multipleUses;
}

void IRTranslator::UpdateKernelArgMemInfo() {
  for (auto [i, arg] : llvm::enumerate(func_.getArguments())) {
    if (!dyn_cast_or_null<MemRefType>(arg.getType())) {
      // not memref type, skip
      continue;
    }
    auto newMemInfo = std::make_unique<BaseMemInfo>(
        arg, arg, hivm::AddressSpace::GM, SmallVector<int64_t>(1, 0), 0, false,
        std::nullopt);
    bool isSplittedMixKernel =
        func_->hasAttrOfType<UnitAttr>(hivm::TPartOfMixAttr::name);
    bool isWorkSpaceArg =
        hacc::utils::isKernelArg(func_, i, hacc::KernelArgType::kWorkspace);
    bool includeWorkSpaceArg = true;
    if (isWorkSpaceArg) {
      if (syncAnalysisMode == SyncAnalysisMode::BLOCKSYNC) {
        // skip workspace, it is only used by alloc_workspace and will be handle
        // by alloc_workspace.

        // it is possible that the workspace argument may be used in a context
        // where synchronization is required, eg store_ub_to_gm(a, workarg)
        // followed by copy_gm_to_l1(workarg, b) in the below, we check if the
        // workspace argument is used multiple times, or if a single use has
        // multiple uses, in both cases we should still register the memory info
        if (skippableWorkspaceArg(arg)) {
          includeWorkSpaceArg = false;
        }
      }
      if (syncAnalysisMode == SyncAnalysisMode::NORMALSYNC &&
          isSplittedMixKernel) {
        // check if the kernal was processed by block-sync previously by
        // checking if it was a mix-kernal that was splitted. if not, then
        // proccess this arg normally. this condition was added to handle
        // workspace args for cube-cube kernels.
        includeWorkSpaceArg = false;
      }
    }
    if (!isWorkSpaceArg || includeWorkSpaceArg) {
      buffer2MemInfoMap[arg].emplace_back(newMemInfo->clone());
    }
    buffer2MemInfoMapIncludingWSArgs[arg].emplace_back(newMemInfo->clone());
  }
}

void IRTranslator::RecursionIR(Region *region) {
  auto result = region->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (auto pointerCastOp = dyn_cast<PointerCastOp>(op)) {
      if (failed(UpdateAllocLikeOpMemInfo(op))) {
        return WalkResult::interrupt();
      }
    } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      UpdateForOpInfo(forOp);
      return WalkResult::skip();
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
      UpdateWhileOpInfo(whileOp);
      return WalkResult::skip();
    } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      UpdateIfOpInform(ifOp);
      return WalkResult::skip();
    } else if (scf::YieldOp yieldOp = dyn_cast<scf::YieldOp>(op)) {
      UpdateYieldOpInform(yieldOp);
    } else if (auto dstOp = dyn_cast<DestinationStyleOpInterface>(op)) {
      UpdateDestinationStyleOpInterfaceInform(op, dstOp);
    } else if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      UpdateStoreOrLoadOpInform(loadOp);
    } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      UpdateStoreOrLoadOpInform(storeOp);
    } else if (auto callOp = dyn_cast<func::CallOp>(op)) {
      UpdateCallOp(callOp);
    } else if (auto affineLoadOp = dyn_cast<affine::AffineLoadOp>(op)) {
      UpdateStoreOrLoadOpInform(affineLoadOp);
    } else if (auto affineStoreOp = dyn_cast<affine::AffineStoreOp>(op)) {
      UpdateStoreOrLoadOpInform(affineStoreOp);
    } else if (auto gpuLaunchFuncOp = dyn_cast<gpu::LaunchFuncOp>(op)) {
      UpdateGPULaunchFuncOpInform(gpuLaunchFuncOp);
    } else if (auto aliasPairs = getOperationAliasInfo(op);
               !aliasPairs.empty()) {
      for (auto aliasPair : aliasPairs) {
        UpdateAliasBufferInfo(aliasPair.first, aliasPair.second);
      }
    } else if (failed(CheckIfUnknownOpTouchBuffer(op))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (result == WalkResult::interrupt()) {
    llvm_unreachable("InjectSync Traverse IR Failed! ");
  }
}

bool IRTranslator::isSkippableOp(Operation *op) const {
  return isa<func::ReturnOp, annotation::MarkOp, memref::DimOp, hivm::DebugOp,
             memref::GetGlobalOp, func::CallOp, hivm::SyncBlockLockOp,
             hivm::SyncBlockUnlockOp, memref::ExtractAlignedPointerAsIndexOp,
             scf::ConditionOp>(op);
}

LogicalResult IRTranslator::CheckIfUnknownOpTouchBuffer(Operation *op) const {
  if (isSkippableOp(op)) {
    // This scene can be ignored.
    return success();
  }
  if (isOpTouchLocalBuffer(op) || isOpTouchGlobalBuffer(op)) {
    return op->emitError("InjectSync Fail : Unrecognized type of Operation "
                         "touches local or global buffer! ");
  }
  return success();
}

LogicalResult IRTranslator::UpdateAllocLikeOpMemInfo(Operation *op) {
  hivm::AddressSpace space;
  SmallVector<Value> curAddress;
  Value rootBuffer, baseBuffer;
  std::optional<bishengir::memref_ext::AllocWorkspaceOp> allocWorkspaceOp;
  if (auto pointerCastOp = dyn_cast<PointerCastOp>(op)) {
    auto spaceAttr = GetBufferSpaceAttr(pointerCastOp.getResult());
    if (!spaceAttr.has_value()) {
      return op->emitError(
          "pointer_cast operation expected to have memory space attribute.");
    }
    space = spaceAttr.value().getAddressSpace();
    curAddress = pointerCastOp.getAddrs();
    rootBuffer = pointerCastOp.getResult();
    baseBuffer = pointerCastOp.getResult();
  } else if (auto workspaceOp =
                 dyn_cast<bishengir::memref_ext::AllocWorkspaceOp>(op)) {
    space = hivm::AddressSpace::GM;
    curAddress = workspaceOp.getOffset();
    rootBuffer = workspaceOp.getWorkspaceArg();
    baseBuffer = workspaceOp.getResult();
    allocWorkspaceOp = workspaceOp;
  } else {
    return op->emitError(
        "Only pointer_cast and alloc_workspace operations are supported.");
  }

  int addressNum = static_cast<int>(curAddress.size());
  if (addressNum == 0)
    return op->emitError("MemAllocOp must have at least one address present");

  bool hasVariableAddress = false;
  SmallVector<int64_t> baseAddresses(addressNum,
                                     std::numeric_limits<int64_t>::max());
  if (!util::isGMPointerCastOp(op)) {
    for (size_t i = 0; i < curAddress.size(); i++) {
      if (auto constOp =
              dyn_cast<arith::ConstantOp>(curAddress[i].getDefiningOp())) {
        int64_t offset = cast<IntegerAttr>(constOp.getValue()).getInt();
        int64_t offsetInBits = offset * utils::kBitsToByte;
        baseAddresses[i] = offsetInBits;
      } else {
        hasVariableAddress = true;
      }
    }
  }

  auto bufferSize = GetBufferBitSize(rootBuffer);
  if (!bufferSize.has_value()) {
    return op->emitError("Failed to get buffer size for alloc-like op.");
  }

  auto newMemInfo = std::make_unique<BaseMemInfo>(
      baseBuffer, rootBuffer, space, baseAddresses, bufferSize.value(),
      hasVariableAddress, allocWorkspaceOp);

  buffer2MemInfoMap[baseBuffer].emplace_back(newMemInfo->clone());
  buffer2MemInfoMapIncludingWSArgs[baseBuffer].emplace_back(
      newMemInfo->clone());
  return success();
}

void IRTranslator::UpdateAliasBufferInfo(
    Value result, Value source,
    std::optional<std::reference_wrapper<Buffer2MemInfoMap>>
        buffer2MemInfoMapOpt) {
  if (syncAnalysisMode == SyncAnalysisMode::NORMALSYNC) {
    auto spaceAttr = GetBufferSpaceAttr(result);
    if (!spaceAttr.has_value()) {
      return;
    }
  }

  if (buffer2MemInfoMapOpt.has_value()) {
    auto &buffer2MemInfoMap = buffer2MemInfoMapOpt.value().get();
    if (buffer2MemInfoMap.contains(source)) {
      auto &resultMemInfoVec = buffer2MemInfoMap[result];
      for (auto &memInfo : buffer2MemInfoMap[source]) {
        resultMemInfoVec.emplace_back(memInfo->clone(result));
      }
    }
    return;
  }

  if (buffer2MemInfoMap.contains(source)) {
    auto &resultMemInfoVec = buffer2MemInfoMap[result];
    for (auto &memInfo : buffer2MemInfoMap[source]) {
      resultMemInfoVec.emplace_back(memInfo->clone(result));
    }
  }

  if (buffer2MemInfoMapIncludingWSArgs.contains(source)) {
    auto &resultMemInfoVec = buffer2MemInfoMapIncludingWSArgs[result];
    for (auto &memInfo : buffer2MemInfoMapIncludingWSArgs[source]) {
      resultMemInfoVec.emplace_back(memInfo->clone(result));
    }
  }
}

void IRTranslator::UpdateForOpInfo(scf::ForOp forOp) {
  auto forBeginElement =
      std::make_unique<LoopInstanceElement>(index, index, index);
  forBeginElement->elementOp = forOp.getOperation();
  syncIR.emplace_back(std::move(forBeginElement));
  std::unique_ptr<InstanceElement> &forElement = syncIR[index];
  index++;
  assert(syncIR.size() == index && "Sync IR Construction failed.");
  auto *forBeginPtr = dyn_cast<LoopInstanceElement>(forElement.get());
  assert(forBeginPtr != nullptr);
  UpdateForInitArgsAliasInfo(forOp);
  RecursionIR(&forOp.getRegion());
  forBeginPtr->endId = index;
  auto forEnd = forBeginPtr->CloneFor(KindOfLoop::LOOP_END);
  forEnd->elementOp = forOp.getOperation();
  syncIR.emplace_back(std::move(forEnd));
  index++;
  assert(syncIR.size() == index && "Sync IR Construction failed.");
}

void IRTranslator::UpdateWhileOpInfo(scf::WhileOp whileOp) {
  auto loopBeginElement =
      std::make_unique<LoopInstanceElement>(index, index, index);
  loopBeginElement->elementOp = whileOp.getOperation();
  syncIR.emplace_back(std::move(loopBeginElement));
  auto &loopElement = syncIR.back();
  index++;
  assert(syncIR.size() == index && "Sync IR Construction failed.");
  auto *loopBeginPtr = dyn_cast<LoopInstanceElement>(loopElement.get());
  assert(loopBeginPtr != nullptr);
  UpdateWhileInitArgsAliasInfo(whileOp);
  RecursionIR(&whileOp.getBefore());
  RecursionIR(&whileOp.getAfter());
  UpdateWhileResultAliasInfo(whileOp);
  loopBeginPtr->endId = index;
  auto forEnd = loopBeginPtr->CloneFor(KindOfLoop::LOOP_END);
  forEnd->elementOp = whileOp.getOperation();
  syncIR.emplace_back(std::move(forEnd));
  index++;
  assert(syncIR.size() == index && "Sync IR Construction failed.");
}

void IRTranslator::UpdateForInitArgsAliasInfo(scf::ForOp forOp) {
  if (forOp.getInitArgs().empty()) {
    return;
  }
  assert(forOp.getInitArgs().size() == forOp.getRegionIterArgs().size());
  for (auto [i, arg] : llvm::enumerate(forOp.getInitArgs())) {
    UpdateAliasBufferInfo(forOp.getRegionIterArgs()[i], arg);
  }
}

void IRTranslator::UpdateWhileInitArgsAliasInfo(scf::WhileOp whileOp) {
  if (whileOp.getInits().empty()) {
    return;
  }
  assert(whileOp.getInits().size() == whileOp.getBeforeArguments().size());
  for (auto [initArg, blockArg] :
       llvm::zip(whileOp.getInits(), whileOp.getBeforeArguments())) {
    UpdateAliasBufferInfo(blockArg, initArg);
  }
  auto conditionOp = whileOp.getConditionOp();
  assert(conditionOp.getArgs().size() == whileOp.getAfterArguments().size());
  for (auto [yieldedArg, blockArg] :
       llvm::zip(conditionOp.getArgs(), whileOp.getAfterArguments())) {
    UpdateAliasBufferInfo(blockArg, yieldedArg);
  }
}

void IRTranslator::UpdateWhileResultAliasInfo(scf::WhileOp whileOp) {
    for (auto [resultIdx, result] : llvm::enumerate(whileOp.getResults())) {
        assert(whileOp.getConditionOp().getArgs().size() > resultIdx);
        auto yieldedValueBefore = whileOp.getConditionOp().getArgs()[resultIdx];
        assert(whileOp.getYieldedValues().size() > resultIdx);
        auto yieldedValueAfter = whileOp.getYieldedValues()[resultIdx];
        UpdateAliasBufferInfo(result, yieldedValueBefore);
        UpdateAliasBufferInfo(result, yieldedValueAfter);
    }
}

void IRTranslator::InsertPlaceHolderInst(InstanceElement *parentScope) {
  auto placeHolder = std::make_unique<PlaceHolderInstanceElement>(
      index, parentScope->GetIndex());
  syncIR.emplace_back(std::move(placeHolder));
  index++;
  assert(syncIR.size() == index && "Sync IR Construction failed.");
}

void IRTranslator::UpdateIfOpInform(scf::IfOp ifOp) {
  auto ifBeginElement = std::make_unique<BranchInstanceElement>(
      index, index, KindOfBranch::IF_BEGIN);
  ifBeginElement->elementOp = ifOp.getOperation();
  auto *ifPtr = ifBeginElement.get();
  syncIR.emplace_back(std::move(ifBeginElement));
  index++;
  assert(syncIR.size() == index && "Sync IR Construction failed.");

  RecursionIR(&ifOp.getThenRegion());
  InsertPlaceHolderInst(ifPtr);
  ifPtr->branchId = index;

  if (ifOp.elseBlock()) {
    auto ifElseElement = ifPtr->CloneBranch(KindOfBranch::ELSE_BEGIN);
    ifElseElement->elementOp = ifOp.getOperation();
    auto *elsePtr = ifElseElement.get();

    syncIR.emplace_back(std::move(ifElseElement));
    index++;
    assert(syncIR.size() == index && "Sync IR Construction failed.");

    RecursionIR(&ifOp.getElseRegion());
    InsertPlaceHolderInst(elsePtr);
    elsePtr->endId = index;
  }
  ifPtr->endId = index;
  auto ifEndElement = ifPtr->CloneBranch(KindOfBranch::IF_END);
  ifEndElement->elementOp = ifOp.getOperation();
  syncIR.emplace_back(std::move(ifEndElement));
  index++;
  assert(syncIR.size() == index && "Sync IR Construction failed.");
}

void IRTranslator::UpdateYieldOpInform(scf::YieldOp yieldOp) {
  auto *parentOp = yieldOp->getParentOp();
  if (parentOp == nullptr) {
    return;
  }
  if (isa<scf::WhileOp>(parentOp)) {
    return;
  }
  assert(parentOp->getResults().size() == yieldOp->getOpOperands().size());
  for (auto [yieldVal, resultVal] :
       llvm::zip(yieldOp->getOpOperands(), parentOp->getResults())) {
    auto spaceAttr = GetBufferSpaceAttr(resultVal);
    if (!spaceAttr.has_value()) {
      continue;
    }
    UpdateAliasBufferInfo(resultVal, yieldVal.get());
  }
}

void IRTranslator::UpdateGPULaunchFuncOpInform(gpu::LaunchFuncOp op) {
  hivm::PIPE pipe = hivm::PIPE::PIPE_V;
  SmallVector<const BaseMemInfo *> defVec, useVec;
  UpdateOpDefUseVec(op, defVec, useVec);
  auto copPrt = std::make_unique<CompoundInstanceElement>(index, defVec, useVec,
                                                          pipe, op->getName());
  copPrt->elementOp = op.getOperation();
  syncIR.emplace_back(std::move(copPrt));
  index++;
}

static Value stripCastsAndSubviews(Value v) {
  while (true) {
    if (auto sub = v.getDefiningOp<memref::SubViewOp>()) {
      v = sub.getSource();
      continue;
    }
    if (auto cast = v.getDefiningOp<memref::CastOp>()) {
      v = cast.getSource();
      continue;
    }
    break;
  }
  return v;
}

void IRTranslator::UpdateOpDefUseVec(gpu::LaunchFuncOp gpuLaunchFunc,
                                     SmallVector<const BaseMemInfo *> &defVec,
                                     SmallVector<const BaseMemInfo *> &useVec) {
  auto gpuFuncSymAttr = gpuLaunchFunc.getKernelAttr();
  auto mlirMod = gpuLaunchFunc->getParentOfType<ModuleOp>();
  auto gpuMod =
      SymbolTable::lookupSymbolIn(mlirMod, gpuFuncSymAttr.getRootReference());
  if (!gpuMod || gpuFuncSymAttr.getNestedReferences().empty())
    llvm_unreachable("missing gpu.module or gpu.func");
  auto gpuFunc = dyn_cast<gpu::GPUFuncOp>(SymbolTable::lookupSymbolIn(
      gpuMod, gpuFuncSymAttr.getNestedReferences().front()));
  if (!gpuFunc)
    llvm_unreachable("missing gpu.func inside gpu.module");

  SmallVector<int> useOperands, defOperands;
  gpuFunc.walk([&](Operation *op) {
    if (auto load = dyn_cast<memref::LoadOp>(op)) {
      auto use = stripCastsAndSubviews(load.getMemref());
      if (auto useBBArg = dyn_cast<BlockArgument>(use)) {
        useOperands.push_back(useBBArg.getArgNumber());
      }
    } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
      auto def = stripCastsAndSubviews(store.getMemref());
      if (auto defBBArg = dyn_cast<BlockArgument>(def)) {
        defOperands.push_back(defBBArg.getArgNumber());
      }
    } else if (auto copy = dyn_cast<memref::CopyOp>(op)) {
      auto use = stripCastsAndSubviews(copy.getSource());
      if (auto useBBArg = dyn_cast<BlockArgument>(use)) {
        useOperands.push_back(useBBArg.getArgNumber());
      }
      auto def = stripCastsAndSubviews(copy.getTarget());
      if (auto defBBArg = dyn_cast<BlockArgument>(def)) {
        defOperands.push_back(defBBArg.getArgNumber());
      }
    }
  });
  SmallVector<Value, 8> callOperands(gpuLaunchFunc.getKernelOperands());
  auto populateUseDefVec = [&](SmallVector<const BaseMemInfo *> &vec,
                               SmallVector<int> &operands) {
    for (int idx : operands) {
      if (buffer2MemInfoMap.contains(callOperands[idx])) {
        for (auto &memInfo : buffer2MemInfoMap[callOperands[idx]]) {
          vec.push_back(memInfo.get());
        }
      }
    }
  };
  populateUseDefVec(useVec, useOperands);
  populateUseDefVec(defVec, defOperands);
}

void IRTranslator::UpdateFuncArguments(func::FuncOp funcOp,
                                       Buffer2MemInfoMap &buffer2MemInfoMap) {
  if (!funcOp->hasAttr(hivm::VectorFunctionAttr::name)) {
    return;
  }
  for (auto arg : funcOp.getArguments()) {
    if (llvm::isa_and_present<MemRefType, TensorType>(arg.getType())) {
      auto newMemInfo = std::make_unique<BaseMemInfo>(
          arg, arg, hivm::AddressSpace::UB, SmallVector<int64_t>(1, 0), 0,
          false, std::nullopt);
      buffer2MemInfoMap[arg].emplace_back(std::move(newMemInfo));
    }
  }
}

void IRTranslator::UpdateOpDefUseVec(func::CallOp callOp,
                                     SmallVector<const BaseMemInfo *> &defVec,
                                     SmallVector<const BaseMemInfo *> &useVec) {
  ModuleOp module = func_->getParentOfType<ModuleOp>();
  SymbolTable symtab(module);
  auto funcOp = symtab.lookup<func::FuncOp>(callOp.getCallee());
  if (!funcOp->hasAttr(hivm::VectorFunctionAttr::name) &&
      !util::isSIMTVF(funcOp)) {
    return;
  }

  Buffer2MemInfoMap buffer2MemInfoMap;
  UpdateFuncArguments(funcOp, buffer2MemInfoMap);
  funcOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (auto aliasPairs = getOperationAliasInfo(op); !aliasPairs.empty()) {
      for (auto aliasPair : aliasPairs) {
        UpdateAliasBufferInfo(aliasPair.first, aliasPair.second,
                              buffer2MemInfoMap);
      }
    } else if (auto aliasPairs = getSCFOperationAliasInfo(op);
               !aliasPairs.empty()) {
      for (auto aliasPair : aliasPairs) {
        UpdateAliasBufferInfo(aliasPair.first, aliasPair.second,
                              buffer2MemInfoMap);
      }
    }
    return WalkResult::advance();
  });

  SmallVector<Value> useOperands, defOperands;

  // handle outlined simt function annotated with memory effects for each
  // input arguments
  for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
    // get the attribute by name for i-th argument
    auto memEffectAttr = funcOp.getArgAttr(i, hivm::MemoryEffectAttr::name);
    if (!memEffectAttr) {
      continue;
    }
    auto effect = cast<hivm::MemoryEffectAttr>(memEffectAttr).getEffect();
    auto callArg = callOp->getOperand(i);
    // logic based on the attribute value
    if (effect == hivm::MemoryEffect::READ) {
      useOperands.push_back(callArg);
    } else if (effect == hivm::MemoryEffect::WRITE) {
      defOperands.push_back(callArg);
    } else if (effect == hivm::MemoryEffect::READ_WRITE) {
      useOperands.push_back(callArg);
      defOperands.push_back(callArg);
    }
    funcOp.removeArgAttr(i, hivm::MemoryEffectAttr::name);
  }

  funcOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
    bool isRead = false;
    bool isWrite = false;
    Value curOperand;

    if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
      isRead = true;
      curOperand = loadOp.getMemRef();
    } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
      isWrite = true;
      curOperand = storeOp.getMemRef();
    } else if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      isRead = true;
      curOperand = loadOp.getMemRef();
    } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      isWrite = true;
      curOperand = storeOp.getMemRef();
    } else if (auto tensorExtractOp = dyn_cast<tensor::ExtractOp>(op)) {
      isRead = true;
      curOperand = tensorExtractOp.getTensor();
    } else if (auto tr = dyn_cast<vector::TransferReadOp>(op)) {
      isRead = true;
      curOperand = tr.getSource();
    } else if (auto tw = dyn_cast<vector::TransferWriteOp>(op)) {
      isWrite = true;
      curOperand = tw.getSource();
    } else if (auto g = dyn_cast<vector::GatherOp>(op)) {
      isRead = true;
      curOperand = g.getBase();
    } else {
      return;
    }

    for (auto &memInfo : buffer2MemInfoMap[curOperand]) {
      if (auto blockArg = dyn_cast<BlockArgument>(memInfo->rootBuffer)) {
        if (isRead)
          useOperands.push_back(callOp->getOperand(blockArg.getArgNumber()));
        if (isWrite)
          defOperands.push_back(callOp->getOperand(blockArg.getArgNumber()));
      }
    }
  });

  UpdateDefUseVec(useOperands, useVec);
  UpdateDefUseVec(defOperands, defVec);
}

void IRTranslator::UpdateDefUseVec(
    const SmallVector<Value> &inOutVals,
    SmallVector<const BaseMemInfo *> &memInfoVec) {
  for (auto &buffer : inOutVals) {
    if (buffer2MemInfoMap.contains(buffer)) {
      for (auto &memInfo : buffer2MemInfoMap[buffer]) {
        memInfoVec.push_back(memInfo.get());
      }
    }
  }
}

void IRTranslator::UpdateMacroOpInform(DestinationStyleOpInterface dstOp) {
  auto pipeOp = dyn_cast<hivm::OpPipeInterface>(dstOp.getOperation());
  assert(pipeOp);
  assert(static_cast<unsigned int>(pipeOp.getInPipe()) < getPipeNum());
  assert(static_cast<unsigned int>(pipeOp.getOutPipe()) < getPipeNum());
  SmallVector<const BaseMemInfo *> defVec;
  UpdateDefUseVec(dstOp.getDpsInits(), defVec);
  auto copPtr1 = std::make_unique<CompoundInstanceElement>(
      index, defVec, SmallVector<const BaseMemInfo *>(), pipeOp.getOutPipe(),
      dstOp->getName());
  copPtr1->elementOp = dstOp.getOperation();
  copPtr1->macroOpInstanceId = 0;
  syncIR.emplace_back(std::move(copPtr1));
  index++;

  SmallVector<const BaseMemInfo *> useVec;
  UpdateDefUseVec(dstOp.getDpsInputs(), useVec);
  auto copPtr2 = std::make_unique<CompoundInstanceElement>(
      index, SmallVector<const BaseMemInfo *>(), useVec, pipeOp.getInPipe(),
      dstOp->getName());
  copPtr2->macroOpInstanceId = 1;
  copPtr2->elementOp = dstOp.getOperation();
  syncIR.emplace_back(std::move(copPtr2));
  index++;
}

void IRTranslator::UpdateDestinationStyleOpInterfaceInform(
    Operation *op, DestinationStyleOpInterface dstOp) {
  hivm::PIPE pipe = hivm::PIPE::PIPE_UNASSIGNED;
  if (auto pipeOp = dyn_cast<hivm::OpPipeInterface>(op)) {
    if (pipeOp.isMacroOp()) {
      UpdateMacroOpInform(dstOp);
      return;
    }
    pipe = pipeOp.getPipe();
  }
  SmallVector<const BaseMemInfo *> defVec;
  UpdateDefUseVec(dstOp.getDpsInits(), defVec);
  SmallVector<const BaseMemInfo *> useVec;
  UpdateDefUseVec(dstOp.getDpsInputs(), useVec);
  UpdateTempOpDefVec(op, defVec);
  assert(static_cast<unsigned int>(pipe) < getPipeNum());
  auto copPrt = std::make_unique<CompoundInstanceElement>(
      index, defVec, useVec, pipe, dstOp->getName());
  copPrt->elementOp = op;
  syncIR.emplace_back(std::move(copPrt));
  index++;
}

void IRTranslator::UpdateTempOpDefVec(
    Operation *op, SmallVector<const BaseMemInfo *> &defVec) {
  if (auto extraBufferOp = dyn_cast<ExtraBufferOpInterface>(op)) {
    for (auto buffer : extraBufferOp.getExtraBuffers()) {
      auto memorySpaceAttr = GetBufferSpaceAttr(buffer);
      checkCondition(memorySpaceAttr.has_value(), "temp buffer must has space");
      auto iter = buffer2MemInfoMap.find(buffer);
      assert(iter != buffer2MemInfoMap.end());
      for (auto &memInfo : iter->second) {
        defVec.push_back(memInfo.get());
      }
    }
  }
}

bool IRTranslator::isTensorExtractLoadOp(Operation *op) {
  return llvm::any_of(op->getResults(), [](Value result) {
    auto duplicateTensorExtractForCubeOpt = utils::getAnnotateOpWithAttr(
        result, "DuplicateTensorExtractForCube::replacementLabel");
    return duplicateTensorExtractForCubeOpt.has_value();
  });
}

template <typename OP>
typename std::enable_if<std::is_same_v<OP, memref::LoadOp> ||
                            std::is_same_v<OP, affine::AffineLoadOp> ||
                            std::is_same_v<OP, affine::AffineStoreOp> ||
                            std::is_same_v<OP, memref::StoreOp>,
                        void>::type
IRTranslator::UpdateStoreOrLoadOpInform(OP op) {
  hivm::PIPE pipe = hivm::PIPE::PIPE_S;
  SmallVector<const BaseMemInfo *> defVec;
  SmallVector<const BaseMemInfo *> useVec;
  // do not need to handle sync within simt_vf
  if (op.getOperation()->template getParentOfType<gpu::GPUFuncOp>() !=
      nullptr) {
    return;
  }
  Value memRef = op.getMemRef();
  auto memorySpaceAttr = GetBufferSpaceAttr(memRef);
  if (!memorySpaceAttr.has_value()) {
    return;
  }

  llvm::SmallVector<const BaseMemInfo *> memInfoVec;
  if (buffer2MemInfoMap.contains(memRef)) {
    for (auto &memInfo : buffer2MemInfoMap[memRef]) {
      memInfoVec.push_back(memInfo.get());
    }
  }
  if (isTensorExtractLoadOp(op)) {
    if (buffer2MemInfoMapIncludingWSArgs.contains(memRef)) {
      for (auto &memInfo : buffer2MemInfoMapIncludingWSArgs[memRef]) {
        memInfoVec.push_back(memInfo.get());
      }
    }
  }
  if (memInfoVec.empty()) {
    return;
  }
  if (std::is_same_v<OP, memref::LoadOp> ||
      std::is_same_v<OP, affine::AffineLoadOp>) {
    useVec = memInfoVec;
  } else {
    defVec = memInfoVec;
  }
  assert(static_cast<unsigned int>(pipe) < getPipeNum());
  auto copPrt = std::make_unique<CompoundInstanceElement>(index, defVec, useVec,
                                                          pipe, op->getName());
  copPrt->elementOp = op.getOperation();
  syncIR.emplace_back(std::move(copPrt));
  index++;
}

void IRTranslator::UpdateCallOp(func::CallOp callOp) {
  ModuleOp module = func_->getParentOfType<ModuleOp>();
  SymbolTable symtab(module);
  auto callee = symtab.lookup<func::FuncOp>(callOp.getCallee());
  // All calls to vf funcs with simd or simt mode need to update def & uses.
  if (!callee->hasAttr(hivm::VectorFunctionAttr::name) &&
      !util::isSIMTVF(callee)) {
    return;
  }
  SmallVector<const BaseMemInfo *> defVec, useVec;
  UpdateOpDefUseVec(callOp, defVec, useVec);
  hivm::PIPE pipe = hivm::PIPE::PIPE_V;
  auto copPrt = std::make_unique<CompoundInstanceElement>(
      index, defVec, useVec, pipe, callOp->getName());
  assert(copPrt != nullptr);
  copPrt->elementOp = callOp.getOperation();
  if (syncAnalysisMode == SyncAnalysisMode::BLOCKSYNC) {
    auto coreType = getCoreType(copPrt->elementOp);
    assert(succeeded(coreType));
    assert(coreType.value() != TCoreType::CUBE_OR_VECTOR);
    copPrt->compoundCoreType = coreType.value();
  }
  syncIR.emplace_back(std::move(copPrt));
  index++;
}
