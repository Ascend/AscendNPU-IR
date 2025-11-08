//===------------ IRTranslator.cpp ----Sync information collection---------===//
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

#include "bishengir/Dialect/HIVM/Transforms/InjectSync/IRTranslator.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#define DEBUG_TYPE "hivm-inject-sync"

using namespace mlir;
using namespace mlir::hivm;

void IRTranslator::Build() {
  Region &funcRegion = func_.getBody();
  UpdateKernelArgMemInfo();
  // Recursively obtaining IR information.
  RecursionIR(&funcRegion);
}

void IRTranslator::UpdateKernelArgMemInfo() {
  auto funcParamSize = func_.getNumArguments();
  for (size_t i = 0; i < funcParamSize; i++) {
    if (!dyn_cast_or_null<MemRefType>(func_.getArgument(i).getType())) {
      // not memref type, skip
      continue;
    }
    std::unique_ptr<BaseMemInfo> newMemInfo =
        std::make_unique<BaseMemInfo>(BaseMemInfo{func_.getArgument(i),
                                                  func_.getArgument(i),
                                                  hivm::AddressSpace::GM,
                                                  {0},
                                                  0,
                                                  std::nullopt});
    bool isSplittedMixKernel =
        func_->getAttrOfType<UnitAttr>(hivm::TPartOfMixAttr::name) != nullptr;
    bool isWorkSpaceArg =
        hacc::utils::isKernelArg(func_, i, hacc::KernelArgType::kWorkspace);
    bool includeWorkSpaceArg = true;
    if (isWorkSpaceArg) {
      if (syncAnalysisMode == SyncAnalysisMode::BLOCKSYNC) {
        // skip workspace, it is only used by alloc_workspace and will be
        // handle by alloc_workspace.
        includeWorkSpaceArg = false;
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

    Value funcArg = func_.getArgument(i);
    if (!isWorkSpaceArg || includeWorkSpaceArg) {
      buffer2MemInfoMap[funcArg].emplace_back(newMemInfo->clone());
    }
    buffer2MemInfoMapIncludingWSArgs[funcArg].emplace_back(newMemInfo->clone());
  }
}

void IRTranslator::RecursionIR(Region *region) {
  auto result = region->walk<WalkOrder::PreOrder>([&](Operation *op) {
    auto aliasPairs = getOperationAliasInfo(op);
    if (!aliasPairs.empty()) {
      for (auto aliasPair : aliasPairs) {
        UpdateAliasBufferInfo(aliasPair.first, aliasPair.second);
      }
    } else if (auto pointerCastOp = dyn_cast<PointerCastOp>(op)) {
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
    } else if (auto affineLoadOp = dyn_cast<affine::AffineLoadOp>(op)) {
      UpdateStoreOrLoadOpInform(affineLoadOp);
    } else if (auto affineStoreOp = dyn_cast<affine::AffineStoreOp>(op)) {
      UpdateStoreOrLoadOpInform(affineStoreOp);
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
             func::CallOp, hivm::SyncBlockLockOp, hivm::SyncBlockUnlockOp,
             memref::ExtractAlignedPointerAsIndexOp, scf::ConditionOp>(op);
}

LogicalResult IRTranslator::CheckIfUnknownOpTouchBuffer(Operation *op) const {
  if (isSkippableOp(op)) {
    // This scene can be ignored.
    return success();
  }
  if (isOpTouchLocalBuffer(op) || isOpTouchGlobalBuffer(op)) {
    op->emitError("InjectSync Fail : Unrecognized type of Operation "
                  "touches local or global buffer! ");
    return failure();
  }
  return success();
}

LogicalResult IRTranslator::UpdateAllocLikeOpMemInfo(Operation *op) {
  SmallVector<Value> curAddress;
  hivm::AddressSpace space;
  Value baseBuffer;
  Value rootBuffer;
  std::optional<bishengir::memref_ext::AllocWorkspaceOp> allocWorkspaceOp =
      std::nullopt;
  if (auto pointerCastOp = dyn_cast<PointerCastOp>(op)) {
    auto spaceAttr = GetBufferSpaceAttr(pointerCastOp.getResult());
    if (!spaceAttr) {
      // Only handle buffers within the specified scope.
      return failure();
    }
    space = spaceAttr.value().getAddressSpace();
    curAddress = pointerCastOp.getAddrs();
    baseBuffer = pointerCastOp.getResult();
    rootBuffer = pointerCastOp.getResult();
  } else if (auto workspaceOp =
                 dyn_cast<bishengir::memref_ext::AllocWorkspaceOp>(op)) {
    space = hivm::AddressSpace::GM;
    curAddress = workspaceOp.getOffset();
    baseBuffer = workspaceOp.getResult();
    rootBuffer = workspaceOp.getWorkspaceArg();
    allocWorkspaceOp = workspaceOp;
  } else {
    llvm_unreachable("unsupport op to update buffer2MemInfo");
  }

  int addressNum = static_cast<int>(curAddress.size());
  if (addressNum == 0)
    return op->emitError("MemAllocOp must have at least one address present");

  SmallVector<uint64_t> baseAddresses(addressNum,
                                      std::numeric_limits<uint64_t>::max());
  if (!util::isGMPointerCastOp(op)) {
    for (int i = 0; i < addressNum; i++) {
      auto constOp = dyn_cast<arith::ConstantOp>(curAddress[i].getDefiningOp());
      if (!constOp)
        return op->emitError(
            "Currently, only constant addresses are supported");
      baseAddresses[i] =
          static_cast<uint64_t>(cast<IntegerAttr>(constOp.getValue()).getInt());
    }
  }
  auto bufferSize = GetBufferSize(rootBuffer);
  if (!bufferSize.has_value())
    return op->emitError(
        "There are illegal buffer MemAllocOp can't get buffer size! ");

  auto newMemInfo = std::make_unique<BaseMemInfo>(
      baseBuffer, rootBuffer, space, baseAddresses, bufferSize.value(),
      allocWorkspaceOp);

  buffer2MemInfoMap[baseBuffer].emplace_back(newMemInfo->clone());
  buffer2MemInfoMapIncludingWSArgs[baseBuffer].emplace_back(
      newMemInfo->clone());
  return success();
}

void IRTranslator::UpdateAliasBufferInfo(Value result, Value source) {
  auto spaceAttr = GetBufferSpaceAttr(result);
  if (!spaceAttr) {
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
  auto forBeginElement = std::make_unique<LoopInstanceElement>(
      LoopInstanceElement{index, index, index});
  assert(forBeginElement != nullptr);
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
  auto loopBeginElement = std::make_unique<LoopInstanceElement>(
      LoopInstanceElement{index, index, index});
  assert(loopBeginElement != nullptr);
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

void IRTranslator::UpdateIfOpInform(scf::IfOp ifOp) {
  auto ifBeginElement = std::make_unique<BranchInstanceElement>(
      index, index, KindOfBranch::IF_BEGIN);
  ifBeginElement->elementOp = ifOp.getOperation();
  syncIR.emplace_back(std::move(ifBeginElement));
  std::unique_ptr<InstanceElement> &ifElement = syncIR[index];
  index++;
  assert(syncIR.size() == index && "Sync IR Construction failed.");
  auto *ifPtr = dyn_cast<BranchInstanceElement>(ifElement.get());
  assert(ifPtr != nullptr);

  RecursionIR(&ifOp.getThenRegion());

  ifPtr->branchId = index;
  if (ifOp.elseBlock()) {
    auto ifElseElement = ifPtr->CloneBranch(KindOfBranch::ELSE_BEGIN);
    ifElseElement->elementOp = ifOp.getOperation();
    syncIR.emplace_back(std::move(ifElseElement));
    std::unique_ptr<InstanceElement> &elseElement = syncIR[index];
    index++;
    assert(syncIR.size() == index && "Sync IR Construction failed.");
    BranchInstanceElement *elsePtr =
        dyn_cast<BranchInstanceElement>(elseElement.get());
    assert(elsePtr != nullptr);

    RecursionIR(&ifOp.getElseRegion());

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
    if (!spaceAttr) {
      continue;
    }
    UpdateAliasBufferInfo(resultVal, yieldVal.get());
  }
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
  auto copPrt1 =
      std::make_unique<CompoundInstanceElement>(CompoundInstanceElement{
          index, defVec, {}, pipeOp.getOutPipe(), dstOp->getName()});
  assert(copPrt1 != nullptr);
  copPrt1->elementOp = dstOp.getOperation();
  syncIR.emplace_back(std::move(copPrt1));
  index++;

  SmallVector<const BaseMemInfo *> useVec;
  UpdateDefUseVec(dstOp.getDpsInputs(), useVec);
  auto copPrt2 =
      std::make_unique<CompoundInstanceElement>(CompoundInstanceElement{
          index, {}, useVec, pipeOp.getInPipe(), dstOp->getName()});
  assert(copPrt2 != nullptr);
  copPrt2->elementOp = dstOp.getOperation();
  syncIR.emplace_back(std::move(copPrt2));
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
      CompoundInstanceElement{index, defVec, useVec, pipe, dstOp->getName()});
  assert(copPrt != nullptr);
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

bool isTensorExtractLoadOp(Operation *op) {
  for (auto result : op->getResults()) {
    auto duplicateTensorExtractForCubeOpt = utils::getAnnotateOpWithAttr(
        result, "DuplicateTensorExtractForCube::replacementLabel");
    if (duplicateTensorExtractForCubeOpt.has_value()) {
      return true;
    }
  }
  return false;
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
  assert(copPrt != nullptr);
  copPrt->elementOp = op.getOperation();
  syncIR.emplace_back(std::move(copPrt));
  index++;
}
